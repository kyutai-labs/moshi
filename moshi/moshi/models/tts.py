# Copyright (c) Kyutai, all rights reserved.

import argparse
from collections import deque
from dataclasses import dataclass, field
import json
import re
from pathlib import Path
import sys
import time
import typing as tp

from safetensors.torch import load_file, save_file
from sentencepiece import SentencePieceProcessor
import sphn
import torch

from moshi.conditioners import ConditionAttributes, dropout_all_conditions, TensorCondition
from moshi.models import loaders, MimiModel, LMModel, LMGen


@dataclass
class TokenIds:
    """
    The token ids for special tokens:
        - word: a new word is starting.
        - pad: padding, nothing happens.
        - main: indicate the start of turn of the main speaker.
        - other: indicate the start of turn of the other speaker.
        - zero: special value that will be embedded to exactly 0.
        - ungenerated: indicate that a value is not yet generated but should be
        - card: text cardinality, including the initial token (1 + tokenizer cardinality).

    """
    word = 0
    pad = 3
    main = 1
    other = 2
    zero = -1
    ungenerated = -2
    card: int = 8001


@dataclass
class Entry:
    """One word to generate.

    Args:
        tokens: list of tokens for this word.
        word: word as string.
        padding: if > 0, we will force that many padding after the word.
            To have an effect this should be more than len(tokens), as we will
            prevent new word until we have passed all the text tokens.
        audio_tokens: is used when some audio should be used as a prefix in the model."""
    tokens: list[int]
    word: str
    padding: int = 0
    audio_tokens: torch.Tensor | None = None


@dataclass
class State:
    """State of the TTS Machine.

    Args:
        entries: queue containing the entries to generate.
        remaining_padding: how many times the model can still sample a pad.
        forced_padding: how many times the model is still forced to sample a pad.
        queued: queue containing the main stream text tokens to feed.
        lookahead: queue containing the lookahead text tokens to feed.
        end_step: once we reach the end of the generation, this is set to the current step.
            The end of the generation is once the model samples a `word` but `entries` is empty.
        consumption_times: list of steps at which each entry in `entries` was consumed.
        transcript: list of tuples `(word, step)`, at which each word was consumed.
        audio_tokens_remaining: when using some audio as prefix, this would contain the remaining
            audio tokens to force into the model.
        zero_text_remaining: when using an audio only prefix, for how long we should still force
            the text to be `zero`.
    """
    entries: deque[Entry]
    remaining_padding: int
    forced_padding: int
    queued: deque[int] = field(default_factory=deque)
    lookahead: deque[int] = field(default_factory=deque)
    end_step: int | None = None
    consumption_times: list[int] = field(default_factory=list)
    transcript: list[tuple[str, int]] = field(default_factory=list)
    audio_tokens_remaining: deque[torch.Tensor | None] = field(default_factory=deque)
    zero_text_remaining: int = 0

    def get_tokens_ahead(self, lookahead: int) -> list[int]:
        assert lookahead > 0
        for entry in self.entries:
            if entry.tokens:
                lookahead -= 1
                if lookahead == 0:
                    return entry.tokens
        return []


def _delayed(token_ids: TokenIds, codes: torch.Tensor, delays: list[int]) -> torch.Tensor:
    # Apply the acoustic delay on the provided audio tokens.
    K, T = codes.shape
    out = torch.full((K, T + max(delays)), token_ids.ungenerated, device=codes.device, dtype=torch.long)
    for k in range(K):
        delay = delays[min(k, len(delays) - 1)]
        out[k, delay: delay + T] = codes[k]
    return out


@dataclass
class StateMachine:
    """State machine that manipulates the `State` based on the model prediction.

    Args:
        token_ids: special token values.
        second_stream_ahead: if > 0, the model needs a second stream for lookahead.
        max_paddings: maximum number of padding token that can be sampled in a row.
        initial_padding: number of padding tokens at the beginning, to prevent the first
            word from being cut.
        old_interleaver: if True, the model was trained with the old interleaver.
        audio_delays: semantic vs acoustic delays for the model.
        tts_delay: delay between the audio and the text, in steps.

    """

    token_ids: TokenIds = field(default_factory=TokenIds)
    second_stream_ahead: int = 0
    max_paddings: int = 6
    initial_padding: int = 2
    old_interleaver: bool = False
    audio_delays: list[int] = field(default_factory=lambda: [0, 2])
    tts_delay: int = 25

    def new_state(self, entries: tp.Sequence[Entry]) -> State:
        state = State(
            entries=deque(entries),
            lookahead=deque(),
            remaining_padding=self.initial_padding,
            forced_padding=self.initial_padding,
        )
        return state

    def process(self, step: int, state: State, token: int) -> tuple[int, torch.Tensor | None, bool]:
        """
        Process the output of the model.
        Args:
            step: current step index.
            state: state to act upon.
            token: model prediction

        Returns:
            - output_token: value to use as the text input for the model at the next step.
        """
        audio_tokens = None
        if token not in [self.token_ids.word, self.token_ids.pad]:
            token = self.token_ids.pad

        if state.zero_text_remaining > 0:
            state.zero_text_remaining -= 1
            token = self.token_ids.zero
        elif state.entries and state.entries[0].audio_tokens is not None:
            token = self.token_ids.word
        elif state.queued:
            # Some text tokens are yet to be fed, we must PAD.
            token = self.token_ids.pad
        elif state.forced_padding > 0:
            # We are forced to pad, we must PAD.
            token = self.token_ids.pad
        elif state.remaining_padding <= 0:
            # We are not allowed to pad, we must ask for a new WORD.
            token = self.token_ids.word

        if token == self.token_ids.word:
            if state.entries:
                entry = state.entries.popleft()
                state.consumption_times.append(step)
                if entry.tokens:
                    state.transcript.append((entry.word, step))
                    # Entry contains a new word, we reset the max padding counter.
                    state.queued.extend(entry.tokens)
                    if self.second_stream_ahead:
                        state.lookahead.extend(state.get_tokens_ahead(self.second_stream_ahead))
                    state.remaining_padding = self.max_paddings
                else:
                    # Entry is only here to insert a break, pretend the token was a PAD.
                    token = self.token_ids.pad
                if entry.audio_tokens is not None:
                    state.zero_text_remaining = entry.audio_tokens.shape[1] - 1
                    codes = _delayed(self.token_ids, entry.audio_tokens, self.audio_delays)
                    state.audio_tokens_remaining.extend([None] * self.tts_delay)
                    for t in range(codes.shape[1]):
                        state.audio_tokens_remaining.append(codes[:, t])
                    token = self.token_ids.zero
                state.forced_padding = entry.padding
            else:
                token = self.token_ids.pad
                if self.second_stream_ahead and state.end_step is None:
                    token = self.token_ids.word
                # Trying to consume past the last word, we reached the end.
                if state.end_step is None:
                    state.end_step = step

        output: int | None = None
        if token == self.token_ids.pad:
            # Decrement the counters for remaining and forced pads.
            if state.remaining_padding > 0:
                state.remaining_padding -= 1
            if state.forced_padding > 0:
                state.forced_padding -= 1
            if state.queued:
                # We have some text tokens to feed to the model.
                output = state.queued.popleft()
            else:
                output = self.token_ids.pad
        elif token == self.token_ids.word:
            output = self.token_ids.word
            if self.old_interleaver:
                assert state.queued
                if state.queued[0] in [self.token_ids.main, self.token_ids.other]:
                    output = state.queued.popleft()
        elif token == self.token_ids.zero:
            output = token
        else:
            raise RuntimeError(f"Invalid token {token}")

        if state.audio_tokens_remaining:
            audio_tokens = state.audio_tokens_remaining.popleft()

        if self.second_stream_ahead:
            second = -1
            if output == self.token_ids.word:
                second = self.token_ids.word
                if state.queued:
                    output = state.queued.popleft()
                else:
                    output = self.token_ids.pad
            elif state.lookahead:
                second = state.lookahead.popleft()
            output = (second + 1) * self.token_ids.card + output

        assert output is not None
        return output, audio_tokens


def script_to_entries(tokenizer: SentencePieceProcessor, token_ids: TokenIds, frame_rate: float,
                      script: tp.Sequence[str], old_interleaver: bool = False,
                      use_bos_eos: bool = True, padding_between: int = 0) -> list[Entry]:
    speaker_tokens = [token_ids.main, token_ids.other]
    opened_main = False
    entries = []

    # break is indicated as e.g. <break time="3s"/>
    event_re = re.compile(r"(?:<break\s+time=\"([0-9]+(?:.[0-9]*)?)s\"\s*/?>)|(?:\s+)")

    def _add_entry(word: str):
        nonlocal first_content, opened_main
        assert ' ' not in word
        assert word
        tokens = tokenizer.encode(word)  # type: ignore
        if first_content:
            speaker = idx % len(speaker_tokens)
            if use_bos_eos:
                if old_interleaver:
                    if speaker == 0:
                        tokens.insert(0, speaker_tokens[0])
                        opened_main = True
                    else:
                        if opened_main:
                            entries.append(Entry(tokens=[speaker_tokens[1]], word=""))
                            opened_main = False
                else:
                    tokens.insert(0, speaker_tokens[speaker])
            first_content = False
        padding = 0
        if padding_between > 0:
            padding = max(0, padding_between + len(tokens) - 1)
        entries.append(Entry(tokens=tokens, word=word, padding=padding))

    for idx, line in enumerate(script):
        first_content = True
        line = line.replace('’', "'")
        line = line.replace(' : ', " ")
        line = line.replace('(', "")
        line = line.replace(')', "")
        while line:
            match = event_re.search(line)
            if match is None:
                break
            word = line[:match.start()]
            line = line[match.end():]
            if word:
                _add_entry(word)
            if match.group(1):
                break_duration = float(match.group(1))
                padding = int(round(break_duration * frame_rate))
                entry = Entry(tokens=[], word='', padding=padding)
                entries.append(entry)
        if line:
            _add_entry(line)
    return entries


@dataclass
class TTSResult:
    frames: list[torch.Tensor]
    logged_text_tokens: list[list[tuple[int, int, bool]]]
    end_steps: list[int | None]
    all_consumption_times: list[list[int]]
    all_transcripts: list[list[tuple[str, int]]]


@dataclass
class TTSModel:
    lm: LMModel
    mimi: MimiModel
    tokenizer: SentencePieceProcessor

    machine: StateMachine
    delay_steps: int

    temp: float = 0.6
    cfg_coef: float = 1.0
    final_padding: int = 4
    n_q: int = 32
    max_gen_length: int = 30000
    padding_bonus: float = 0.
    kwargs: dict[str, tp.Any] = field(default_factory=dict)

    def prepare_script(self, script: tp.Sequence[str], use_bos_eos: bool = True,
                       padding_between: int = 0) -> list[Entry]:
        return script_to_entries(
            self.tokenizer, self.machine.token_ids, self.mimi.frame_rate, script,
            self.machine.old_interleaver, use_bos_eos=use_bos_eos, padding_between=padding_between)

    @torch.no_grad()
    def generate(self, all_entries: tp.Sequence[tp.Sequence[Entry]],
                 attributes: tp.Sequence[ConditionAttributes], cfg_is_masked_until: list[int] | None):

        def _main_wrapper(*args, **kwargs):
            transformer_out, text_logits = original(*args, **kwargs)
            if self.padding_bonus:
                text_logits[..., 3] += self.padding_bonus
            return transformer_out, text_logits

        original = self.lm.forward_text
        self.lm.forward_text = _main_wrapper

        try:
            return self._generate(all_entries, attributes, cfg_is_masked_until)
        finally:
            self.lm.forward_text = original

    def _generate(self, all_entries: tp.Sequence[tp.Sequence[Entry]], attributes: tp.Sequence[ConditionAttributes],
                  cfg_is_masked_until: list[int] | None):
        if self.cfg_coef != 1.0:
            nulled = make_null(attributes)
            attributes = list(attributes) + nulled

        assert self.lm.condition_provider is not None
        prepared = self.lm.condition_provider.prepare(attributes)
        condition_tensors = self.lm.condition_provider(prepared)

        states = []
        for entries in all_entries:
            state = self.machine.new_state(entries)
            states.append(state)

        def _on_audio_hook(audio_tokens):
            for q in range(audio_tokens.shape[1]):
                delay = self.lm.delays[q + 1]
                if offset < delay + self.delay_steps:
                    audio_tokens[:, q] = self.machine.token_ids.zero
            for b, forced_audio_tokens in enumerate(all_audio_tokens):
                if forced_audio_tokens is not None:
                    mask = forced_audio_tokens != self.machine.token_ids.ungenerated
                    K = forced_audio_tokens.shape[0]
                    audio_tokens[b, :K] = torch.where(mask, forced_audio_tokens, audio_tokens[b, :K])

        def _on_text_hook(text_tokens):
            all_audio_tokens.clear()
            tokens = text_tokens.tolist()
            out_tokens = []
            for token, state, logged in zip(tokens, states, logged_text_tokens):
                out_token, audio_tokens = self.machine.process(offset, state, token)
                all_audio_tokens.append(audio_tokens)
                out_tokens.append(out_token)
                logged.append((token, out_token, audio_tokens is None))
            text_tokens[:] = torch.tensor(out_tokens, dtype=torch.long, device=text_tokens.device)

        self.lm.dep_q = self.n_q
        kwargs = {}
        if cfg_is_masked_until:
            kwargs['cfg_is_masked_until'] = cfg_is_masked_until
        lm_gen = LMGen(
            self.lm, temp=self.temp, temp_text=self.temp, cfg_coef=self.cfg_coef,
            condition_tensors=condition_tensors, on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook, **self.kwargs)

        logged_text_tokens = [[] for _ in states]
        all_audio_tokens = []
        frames: list[torch.Tensor] = []

        with lm_gen.streaming(len(states)):
            for offset in range(self.max_gen_length):
                if all(state.end_step is not None for state in states):
                    max_end_step = max(state.end_step for state in states)
                    if offset >= max_end_step + self.delay_steps + self.final_padding:
                        break
                missing = self.lm.n_q - self.lm.dep_q
                input_tokens = torch.full((len(states), missing, 1), self.machine.token_ids.zero,
                                          dtype=torch.long, device=self.lm.device)
                frame = lm_gen.step(input_tokens)
                if frame is not None:
                    frames.append(frame.clone())
        return TTSResult(
            frames, logged_text_tokens,
            [state.end_step for state in states],
            [state.consumption_times for state in states],
            [state.transcript for state in states])


def make_condition_attributes(voices: list[Path], max_speakers: int = 5):
    if voices:
        voice_tensor = None
        mask = None
        for idx in range(5):
            if idx < len(voices):
                emb = load_file(voices[idx], device='cpu')['speaker_wavs']
                assert emb.dim() == 3
                if voice_tensor is None:
                    voice_tensor = torch.zeros(1, max_speakers, emb.shape[2], emb.shape[1])
                if mask is None:
                    mask = torch.zeros(1, max_speakers, emb.shape[2], dtype=torch.bool)
                voice_tensor[:, idx, :, :] = emb.transpose(1, 2)
                mask[:, idx, :] = True
        assert voice_tensor is not None
        assert mask is not None
        voice_tensor = voice_tensor.view(1, -1, voice_tensor.shape[-1])
        mask = mask.view(1, -1)
        tensors = {
            'speaker_wavs': TensorCondition(voice_tensor, mask)
        }
    else:
        tensors = {}
    return ConditionAttributes(text={'control': 'ok'}, tensor=tensors)


def get_voice_file(voice_file: str, ext: str) -> Path:
    return Path(voice_file + ext)


def make_null(all_attributes: tp.Sequence[ConditionAttributes]) -> list[ConditionAttributes]:
    return dropout_all_conditions(all_attributes)


def main():
    stdout = sys.stdout
    sys.stdout = sys.stderr
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", type=str, default='ship-it1/prot-1-prev2',
                        help="HF repo to look into for pretrained models.")
    parser.add_argument("--config", "--lm-config", dest="config", type=str, help="The config as a json file.")
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")

    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size to be used for inference.")
    parser.add_argument("--nq", type=int, default=32, help="Number of codebooks to generate.")
    parser.add_argument("--temp", type=float, default=0.6, help="Temperature for text and audio.")
    parser.add_argument("--cfg-coef", type=float, default=2., help="CFG coefficient.")
    parser.add_argument("--cfg-is-no-text", action='store_true')

    parser.add_argument("--max-paddings", type=int, default=6, help="Max padding in a row.")
    parser.add_argument("--initial-padding", type=int, default=2, help="Initial padding.")
    parser.add_argument("--final-padding", type=int, default=4, help="Final padding.")
    parser.add_argument("--padding-bonus", type=float, default=0., help="Bonus to the padding logits.")
    parser.add_argument("--voice-is-prefix", action='store_true')
    parser.add_argument("--no-voice", action='store_true')
    parser.add_argument("--padding-between", type=int, default=0)

    args = parser.parse_args()

    print("retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer, args.config)
    print("loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    mimi.set_num_codebooks(args.nq)
    print("mimi loaded")
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    print("loading moshi")
    lm = checkpoint_info.get_moshi(device=args.device, dtype=torch.bfloat16)
    print("moshi loaded")

    ext = ".safetensors"
    assert checkpoint_info.raw_config is not None
    model_id = checkpoint_info.raw_config['model_id']
    ext = f".{model_id['sig']}@{model_id['epoch']}{ext}"

    old_interleaver = checkpoint_info.tts_config.get('old_interleaver', False)

    token_ids = TokenIds()
    delay_steps = int(checkpoint_info.tts_config['audio_delay'] * mimi.frame_rate)
    machine = StateMachine(
        token_ids=token_ids, max_paddings=args.max_paddings, initial_padding=args.initial_padding,
        second_stream_ahead=checkpoint_info.tts_config.get('second_stream_ahead', 0),
        old_interleaver=old_interleaver, tts_delay=delay_steps, audio_delays=lm.delays[1:])
    kwargs = {
        'cfg_is_no_text': args.cfg_is_no_text,
    }
    tts_model = TTSModel(
        lm=lm, mimi=mimi, tokenizer=text_tokenizer,
        machine=machine, delay_steps=delay_steps, temp=args.temp, cfg_coef=args.cfg_coef,
        final_padding=args.final_padding, n_q=args.nq, padding_bonus=args.padding_bonus, kwargs=kwargs)
    first_batch = True

    @torch.no_grad()
    def _flush():
        nonlocal first_batch
        all_entries = []
        all_attributes = []
        cfg_is_masked_until = None
        all_codes = []
        if args.voice_is_prefix:
            cfg_is_masked_until = []
        begin = time.time()
        for request in batch:
            entries = tts_model.prepare_script(request.script, use_bos_eos=not args.voice_is_prefix,
                                               padding_between=args.padding_between)
            voices = [get_voice_file(voice, ext) for voice in request.voices]
            if args.no_voice:
                voices = []
            elif args.voice_is_prefix:
                assert len(request.voices) == 1
                voices = []
                wav, _ = sphn.read(request.voices[0], sample_rate=mimi.sample_rate)
                codes = mimi.encode(torch.from_numpy(wav).to(device=args.device)[None])[0, :, :-2]
                codes = codes.contiguous()
                all_codes.append(codes)
                entries.insert(0, Entry([], '', audio_tokens=codes))
                assert cfg_is_masked_until is not None
                cfg_is_masked_until.append(codes.shape[-1] + delay_steps)
            all_attributes.append(make_condition_attributes(voices))
            all_entries.append(entries)

        print(f"Starting batch of size {len(batch)}")
        result = tts_model.generate(all_entries, all_attributes, cfg_is_masked_until)
        first_batch = False
        frames = torch.cat(result.frames, dim=-1).cpu()
        total_duration = frames.shape[0] * frames.shape[-1] / mimi.frame_rate
        time_taken = time.time() - begin
        total_speed = total_duration / time_taken
        print(f"[LM] Batch of size {len(batch)} took {time_taken:.2f}s, "
              f"total speed {total_speed:.2f}x")

        wav_frames = []
        with torch.no_grad(), mimi.streaming(len(all_entries)):
            for frame in result.frames[tts_model.delay_steps:]:
                wav_frames.append(mimi.decode(frame[:, 1:]))
        wavs = torch.cat(wav_frames, dim=-1)
        effective_duration = 0.
        for idx, request in enumerate(batch):
            end_step = result.end_steps[idx]
            if end_step is None:
                print(f"Warning: end step is None, generation failed for {request.output_file}")
                wav_length = wavs.shape[-1]
            else:
                wav_length = int((mimi.sample_rate * (end_step + tts_model.final_padding) / mimi.frame_rate))
            effective_duration += wav_length / mimi.sample_rate
            wav = wavs[idx, :, :wav_length]
            start_time = 0.

            if cfg_is_masked_until is not None:
                start_time = (cfg_is_masked_until[idx] - tts_model.delay_steps) / mimi.frame_rate

            start = int(start_time * mimi.sample_rate)
            wav = wav[:, start:]
            duration = wav.shape[-1] / mimi.sample_rate
            filename = Path(request.output_file)
            debug_tensors = {
                'frames': frames[idx].int(),
            }
            if all_codes:
                debug_tensors['prefix_codes'] = all_codes[idx]

            segments = []
            transcript = []
            last_segment_start = 0
            last_speaker = None
            segment_has_content = False
            for entry, step in zip(all_entries[idx], result.all_consumption_times[idx]):
                if not entry.tokens:
                    continue
                timestamp = step / mimi.frame_rate - start_time
                if entry.word:
                    segment_has_content = True
                    transcript.append((entry.word, timestamp))
                if entry.tokens:
                    speakers = [machine.token_ids.main, machine.token_ids.other]
                    try:
                        speaker = speakers.index(entry.tokens[0])
                    except ValueError:
                        pass
                    else:
                        if last_speaker is not None:
                            assert speaker != last_speaker, (speaker, last_speaker, timestamp, entry.word)
                            segments.append((last_speaker, (last_segment_start, timestamp)))
                            last_segment_start = timestamp
                            segment_has_content = False
                        last_speaker = speaker
            if segment_has_content:
                segments.append((last_speaker, (last_segment_start, duration)))

            sphn.write_wav(filename, wav.clamp(-0.99, 0.99).cpu().numpy(), mimi.sample_rate)
            save_file(debug_tensors, filename.with_suffix('.safetensors'))
            entries = all_entries[idx]
            debug_info = {
                'hf_repo': args.hf_repo,
                'model_id': checkpoint_info.model_id,
                'cfg_coef': tts_model.cfg_coef,
                'temp': tts_model.temp,
                'max_padding': tts_model.machine.max_paddings,
                'initial_padding': tts_model.machine.initial_padding,
                'final_padding': tts_model.final_padding,
                'transcript': transcript,
                'segments': segments,
                'consumption_times': result.all_consumption_times[idx],
                'script': request.script,
                'voices': request.voices,
                'logged_text_tokens': result.logged_text_tokens[idx],
                'end_step': end_step,
                'start_time': start_time,
            }
            with open(filename.with_suffix('.json'), 'w') as f:
                json.dump(debug_info, f)
            with open(filename.with_suffix('.segments.json'), 'w') as f:
                json.dump({'segments': segments}, f)
            print("Saved", filename)
        time_taken = time.time() - begin
        total_speed = total_duration / time_taken
        effective_speed = effective_duration / time_taken
        print(f"[TOT] Batch of size {len(batch)} took {time_taken:.2f}s, "
              f"total speed {total_speed:.2f}x, "
              f"effective speed {effective_speed:.2f}x")
        batch.clear()

    while True:
        batch: list[TTSRequest] = []
        line = sys.stdin.readline()
        items = json.loads(line)
        for item in items:
            script = []
            if len(item['speaker_audios']) == 1:
                script = [" ".join(turn.strip() for turn in item['turns'])]
            else:
                script = item['turns']
            batch.append(TTSRequest(voices=item['speaker_audios'],
                                    script=script, output_file=item['output_file']))
        _flush()
        stdout.write("external_tts:" + json.dumps({"status": "ok"}) + "\n")
        stdout.flush()


if __name__ == "__main__":
    main()

