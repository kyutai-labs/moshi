# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Implements the logic for the state machine around the Delayed Streams Modeling (DSM) based TTS model.
Things are more complex than for STT models, where we can simply force feed the audio tokens and
sample the text ones. For TTS we start from pure text, not text properly padded with time alignment.
We co-generate the padded text sequence along with the audio output from the original text by
having the model signal us when it thinks the next step will be the start of a word. We then pop a word
to feed and feed it the token representation of the word over the next few steps.
"""

from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
import mlx.core as mx
import re
from pathlib import Path
import typing as tp

from sentencepiece import SentencePieceProcessor
import sphn

from . import Lm, LmGen
from ..modules.conditioner import (
    ConditionAttributes, LutConditioner, dropout_all_conditions, TensorCondition, ConditionTensor)
from .mimi import Mimi
from ..utils.loaders import hf_get
from ..utils.sampling import Sampler


DEFAULT_DSM_TTS_REPO = 'kyutai/tts-1.6b-en_fr'
DEFAULT_DSM_TTS_VOICE_REPO = 'kyutai/tts-voices'


@dataclass
class TokenIds:
    """
    The token ids for special tokens:
        - card: text cardinality, including the initial token (1 + tokenizer cardinality).
            This is used for multiplexing multiple input tokens into the text stream.
        - new_word: a new word is starting.
        - pad: padding, nothing happens.
        - main: indicates the start of turn of the main speaker.
        - other: indicates the start of turn of the other speaker.
        - zero: special value that is embedded to exactly 0.
        - ungenerated: indicate that a value is not yet generated but should be

    """
    card: int
    new_word: int = 0
    pad: int = 3
    main: int = 1
    other: int = 2
    zero: int = -1
    ungenerated: int = -2


@dataclass
class Entry:
    """One word to generate.

    Args:
        tokens: list of tokens for this word.
        text: word as string.
        padding: if > 0, we will prevent the model from sampling a new word for that
            many steps after the current word. Note that even for `padding=0`, the model
            will be forbidden to sample a new word until all the tokens for the current word are consumed.
        audio_tokens: is used when some audio should be used as a prefix in the model."""
    tokens: list[int]
    text: str
    padding: int = 0
    audio_tokens: mx.array | None = None


@dataclass
class State:
    """State of the TTS Machine.

    Args:
        entries: queue containing the entries to generate.
        remaining_padding: how many times the model can still sample a pad.
        forced_padding: how many times the model is still forced to sample a pad.
        queued: queue containing the main stream text tokens to feed.
        lookahead_queued: queue containing the lookahead text tokens to feed.
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
    lookahead_queued: deque[int] = field(default_factory=deque)
    end_step: int | None = None
    consumption_times: list[int] = field(default_factory=list)
    transcript: list[tuple[str, int]] = field(default_factory=list)

    def get_tokens_ahead(self, lookahead: int) -> list[int]:
        assert lookahead > 0
        for entry in self.entries:
            if entry.tokens:
                lookahead -= 1
                if lookahead == 0:
                    return entry.tokens
        return []


def _delayed(codes: mx.array, delays: list[int], fill_value: int) -> mx.array:
    # Apply the acoustic delay on the provided audio tokens.
    K, T = codes.shape
    out = mx.full((K, T + max(delays)), fill_value, dtype=mx.int64)
    for k, delay in enumerate(delays):
        out[k, delay: delay + T] = codes[k]
    return out


def _make_null(all_attributes: tp.Sequence[ConditionAttributes]) -> list[ConditionAttributes]:
    # When using CFG, returns the null conditions.
    return dropout_all_conditions(all_attributes)


@dataclass
class StateMachine:
    """State machine that manipulates the `State` based on the model prediction.
    In particular, every time the model predicts a `word` (see `TokenIds`) special token,
    the state machine will pop the next word to synthesize and start feeding it.
    The model is optionally equipped with a second input text stream providing a lookahead
    into the future text.

    Args:
        token_ids: special token values.
        second_stream_ahead: if > 0, the model needs a second stream for lookahead.
        max_padding: maximum number of padding tokens that can be sampled in a row.
        initial_padding: number of padding tokens at the beginning, to prevent the first
            word from being cut.

    """

    token_ids: TokenIds
    second_stream_ahead: int = 0
    max_padding: int = 6
    initial_padding: int = 2

    def new_state(self, entries: tp.Sequence[Entry]) -> State:
        state = State(
            entries=deque(entries),
            lookahead_queued=deque(),
            remaining_padding=self.initial_padding,
            forced_padding=self.initial_padding,
        )
        return state

    def process(self, step: int, state: State, token: int) -> tuple[int, bool]:
        """
        Process the output of the model.

        Args:
            step: current step index.
            state: state to act upon.
            token: model prediction

        Returns:
            - output_token: value to use as the text input for the model at the next step.
            - consumed_new_word: True if a new word was consumed.
        """
        consumed_new_word = False
        if token not in [self.token_ids.new_word, self.token_ids.pad]:
            token = self.token_ids.pad

        if state.queued:
            # Some text tokens are yet to be fed, we must PAD.
            token = self.token_ids.pad
        elif state.forced_padding > 0:
            # We are forced to pad, we must PAD.
            token = self.token_ids.pad
        elif state.remaining_padding <= 0:
            # We are not allowed to pad, we must ask for a new WORD.
            token = self.token_ids.new_word

        if token == self.token_ids.new_word:
            if state.entries:
                entry = state.entries.popleft()
                state.consumption_times.append(step)
                if entry.tokens:
                    consumed_new_word = True
                    state.transcript.append((entry.text, step))
                    # We queue the tokens to be fed to the model.
                    state.queued.extend(entry.tokens)
                    if self.second_stream_ahead:
                        # We queue the tokens for the N+lookahead word into the second text stream.
                        state.lookahead_queued.extend(state.get_tokens_ahead(self.second_stream_ahead))
                    # Entry contains a new word, we reset the max padding counter.
                    state.remaining_padding = self.max_padding
                else:
                    # Entry is only here to insert a break, pretend the token was a PAD.
                    token = self.token_ids.pad
                state.forced_padding = entry.padding
            else:
                token = self.token_ids.pad
                if self.second_stream_ahead and state.end_step is None:
                    token = self.token_ids.new_word
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
        elif token == self.token_ids.new_word:
            output = self.token_ids.new_word
        elif token == self.token_ids.zero:
            output = token
        else:
            raise RuntimeError(f"Invalid token {token}")

        if self.second_stream_ahead:
            second = -1
            if output == self.token_ids.new_word:
                # If sampled the `word` special token, we put it on the
                # second text stream instead of the main one.
                second = self.token_ids.new_word
                if state.queued:
                    # This allows us to pass the current word tokens faster.
                    output = state.queued.popleft()
                else:
                    output = self.token_ids.pad
            elif state.lookahead_queued:
                # Otherwise if we have some lookahead tokens we feed them.
                second = state.lookahead_queued.popleft()
            # Then we multiplex the two tokens. We add `+1` to `second` so that
            # we can encode -1, which would translate to an all 0s embedding.
            # This will get de-multiplexed in the embedding in lm.py.
            output = (second + 1) * self.token_ids.card + output

        assert output is not None
        return output, consumed_new_word


def script_to_entries(tokenizer: SentencePieceProcessor, token_ids: TokenIds, frame_rate: float,
                      script: tp.Sequence[str], multi_speaker: bool = True, padding_between: int = 0) -> list[Entry]:
    """Process a given script into a list of `Entry` that will be consumed by the model.

    This function will perform some replacements such as removing some caracters such as ':', etc.
    It also supports a single XML tag from SSML, namely `<break time="0.5s">`. This allow the insertion
    of a pause of roughly the requested duration.

    Args:
        tokenizer: text tokenizer.
        token_ids: See `TokenIds`.
        frame_rate: frame rate of the audio codec.
        script: list of turns, with each element indicating a change of turn. Starts with the main speaker.
            Use an empty first turn to start with the other speaker.
        multi_speaker: whether the model was trained to handle more than one speaker.
        padding_between: amount of padding to force between words. Will make the model articulate
            a bit better with values such as 1.
    """
    speaker_tokens = [token_ids.main, token_ids.other]
    last_speaker = None
    entries = []

    # break is indicated as e.g. <break time="3s"/>
    event_re = re.compile(r"(?:<break\s+time=\"([0-9]+(?:.[0-9]*)?)s\"\s*/?>)|(?:\s+)")

    def _add_entry(idx: int, word: str):
        nonlocal first_content, last_speaker
        assert ' ' not in word
        assert word
        tokens = tokenizer.encode(word)  # type: ignore
        if first_content:
            speaker = idx % len(speaker_tokens)
            if multi_speaker and last_speaker != speaker:
                last_speaker = speaker
                tokens.insert(0, speaker_tokens[speaker])
            first_content = False
        padding = 0
        if padding_between > 0:
            padding = max(0, padding_between + len(tokens) - 1)
        entries.append(Entry(tokens=tokens, text=word, padding=padding))

    for idx, line in enumerate(script):
        first_content = True
        line = line.replace('’', "'")
        line = line.replace(':', " ")
        line = line.replace('(', "")
        line = line.replace(')', "")
        while line:
            match = event_re.search(line)
            if match is None:
                break
            word = line[:match.start()]
            line = line[match.end():]
            if word:
                _add_entry(idx, word)
            if match.group(1):
                break_duration = float(match.group(1))
                padding = int(round(break_duration * frame_rate))
                entry = Entry(tokens=[], text='', padding=padding)
                entries.append(entry)
        if line:
            _add_entry(idx, line)
    return entries


@dataclass
class TTSResult:
    """Represents the result of a run of the TTS model on a batch.

    Args:
        frames: list of long tensors with shape `[B, 1 + Q, 1]` representing
            the audio and text tokens for each step. Note that acoustic delay is already corrected
            at that point.
        logged_text_tokens: for debugging, list of tuples `(predicted_tokens, next_input_token)`.
        end_steps: gives the last valid step in `frames` for each item in the batch, or `None` if the
            full text could not be fully synthesized within the provided budget.
        all_consumption_times: for each item in the batch, a list of steps at which individual entries
            (see `Entry`) were consumed as input to the model.
        all_transcripts: for each item in the batch, a list of pairs `(word, step)` indicating
            at which step the given `word` in the transcript should appear. Divide by the frame rate
            to obtain a time stamp.
    """
    frames: list[mx.array]
    logged_text_tokens: list[list[tuple[int, int]]]
    end_steps: list[int | None]
    all_consumption_times: list[list[int]]
    all_transcripts: list[list[tuple[str, int]]]


@dataclass
class TTSModel:
    """Wrapper around a multi-stream language model, a mimi codec, and a text tokenizer that
    provides the functionality of a TTS model. As an end-user, you should use `from_checkpoint_info`
    rather than trying to build a TTSModel directly.

    Args:
        lm: trained delayed streams model.
        mimi: codec to use.
        tokenizer: text tokenizer to use.
        machine: TTS state machine to use, which depend on how the model was trained.
        delay_steps: delay between the text and audio in steps.
        max_speakers: maximum number of speakers in the cross attention for this model.
        temp: temperature (for both text and audio).
        cfg_coef: classifier free guidance coefficient. Note that some models were trained with
            CFG distillation, e.g. CFG should not be used at inference time.
        final_padding: how many steps to sample past the last word.
        n_q: how many audio codebooks (e.g. RVQ levels) to generate. Trade off between quality and speed.
        max_gen_length: will stop generating after that many steps even if the text has not been fully consumed.
        padding_bonus: additive bonus for the padding logits, positive value will lead to slower speech.
        kwargs: other arguments for `moshi.models.lm.LMGen`.

    """

    lm: Lm
    mimi: Mimi
    tokenizer: SentencePieceProcessor

    voice_suffix: str
    voice_repo: str

    machine: StateMachine
    delay_steps: int
    max_speakers: int = 5

    # The following params can be overriden to customize generation.
    temp: float = 0.6
    cfg_coef: float = 1.0
    final_padding: int = 4
    n_q: int = 32
    max_gen_length: int = 30000
    padding_bonus: float = 0.

    def __init__(
            self,
            lm: Lm,
            mimi: Mimi,
            tokenizer: SentencePieceProcessor,
            temp: float = 0.6,
            cfg_coef: float = 1.0,
            final_padding: int = 4,
            n_q: int = 32,
            max_gen_length: int = 30000,
            padding_bonus: float = 0.,
            initial_padding: int = 2,
            max_padding: int = 8,
            voice_repo: str = DEFAULT_DSM_TTS_VOICE_REPO,
            raw_config: dict = {},
    ):
        self.lm = lm
        self.mimi = mimi
        self.tokenizer = tokenizer
        self.temp = temp
        self.cfg_coef = cfg_coef
        self.final_padding = final_padding
        self.n_q = n_q
        self.max_gen_length = max_gen_length
        self.padding_bonus = padding_bonus
        model_id = raw_config['model_id']
        self.voice_suffix = f".{model_id['sig']}@{model_id['epoch']}.safetensors"
        self.voice_repo = voice_repo

        token_ids = TokenIds(lm.cfg.text_out_vocab_size + 1)
        tts_config = raw_config['tts_config']
        self.delay_steps = int(tts_config['audio_delay'] * mimi.frame_rate)
        second_stream_ahead = tts_config.get('second_stream_ahead', 0)

        self.machine = StateMachine(
            token_ids=token_ids, second_stream_ahead=second_stream_ahead,
            max_padding=max_padding, initial_padding=initial_padding)

    @cached_property
    def valid_cfg_conditionings(self) -> set[float]:
        valid_cfg_conditionings = set()
        if self.lm.condition_provider is not None and 'cfg' in self.lm.condition_provider.conditioners:
            cfg_conditioner = self.lm.condition_provider.conditioners['cfg']
            assert isinstance(cfg_conditioner, LutConditioner)
            assert cfg_conditioner.possible_values is not None
            valid_cfg_conditionings = set(float(x) for x in cfg_conditioner.possible_values)
        return valid_cfg_conditionings

    @cached_property
    def multi_speaker(self) -> bool:
        if self.lm.condition_provider is None:
            return False
        return 'speaker_wavs' in self.lm.condition_provider.conditioners

    def prepare_script(self, script: tp.Sequence[str], padding_between: int = 0) -> list[Entry]:
        """Wrapper around `script_to_entries`."""
        return script_to_entries(
            self.tokenizer, self.machine.token_ids, self.mimi.frame_rate, script,
            multi_speaker=self.multi_speaker, padding_between=padding_between)

    def generate(self, all_entries: tp.Sequence[tp.Sequence[Entry]],
                 attributes: tp.Sequence[tp.Any],
                 prefixes: list[mx.array] | None = None,
                 cfg_is_no_prefix: bool = True,
                 cfg_is_no_text: bool = True,
                 **kwargs
                 ) -> TTSResult:
        """Synthesize text to audio. Returns a `TTSResult`.

        Args:
            all_entries: list with one item per batch item, consisting of a list of `Entry`,
                obtained from `prepare_script`.
            attributes: list of `ConditionAttributes` for speaker conditioning.
            prefixes: this should be the list of the lengths up until when to mask for the CFG.
            cfg_is_no_prefix: if true, the null logits are computed with a masked prefix.
            cfg_is_no_text: if true, the null logits are computed without the text.
            **kwargs: passed to `moshi.models.lm.LMGen`.
        """

        # TODO(laurent):
        # Re-enable the padding bonus.
        # def _main_wrapper(*args, **kwargs):
        #     transformer_out, text_logits = original(*args, **kwargs)
        #     if self.padding_bonus:
        #         text_logits[..., self.machine.token_ids.pad] += self.padding_bonus
        #     return transformer_out, text_logits

        if self.cfg_coef != 1.0:
            if self.valid_cfg_conditionings:
                raise ValueError(
                    "This model does not support direct CFG, but was trained with "
                    "CFG distillation. Pass instead `cfg_coef` to `make_condition_attributes`.")
            nulled = _make_null(attributes)
            attributes = list(attributes) + nulled

        assert self.lm.condition_provider is not None
        ct = None
        cross_attention_src = None
        for _attr in attributes:
            for _key, _value in _attr.text.items():
                _ct = self.lm.condition_provider.condition_tensor(_key, _value)
                if ct is None:
                    ct = _ct
                else:
                    ct = ConditionTensor(ct.tensor + _ct.tensor)
            for _key, _value in _attr.tensor.items():
                _conditioner = self.lm.condition_provider.conditioners[_key]
                _ca_src = _conditioner.condition(_value)
                if cross_attention_src is None:
                    cross_attention_src = _ca_src
                else:
                    raise ValueError("multiple cross-attention conditioners")

        states = []
        for entries in all_entries:
            state = self.machine.new_state(entries)
            states.append(state)

        cfg_is_masked_until = None
        text_prefixes = None
        audio_prefixes = None
        if prefixes is not None:
            assert len(all_entries) == len(prefixes), f"Not enough prefixes, expected {len(all_entries)}."
            if cfg_is_no_prefix:
                cfg_is_masked_until = []
            text_prefixes = []
            audio_prefixes = []
            for prefix in prefixes:
                if cfg_is_masked_until is not None:
                    cfg_is_masked_until.append(prefix.shape[-1] + self.delay_steps)
                K, _ = prefix.shape
                assert K == self.lm.num_codebooks
                text_prefixes.append(deque(prefix[0].tolist()))
                delays = [d + self.delay_steps for d in self.lm.delays[self.lm.audio_offset:]]
                delayed = _delayed(prefix[self.lm.audio_offset:], delays, self.machine.token_ids.ungenerated)
                audio_prefixes.append(deque(delayed.T))

        def _on_audio_hook(audio_tokens):
            delays = self.lm.delays
            ungenerated = self.machine.token_ids.ungenerated
            for q in range(audio_tokens.shape[0]):
                delay = delays[q]
                if offset < delay + self.delay_steps:
                    audio_tokens[q] = self.machine.token_ids.zero
            if audio_prefixes is not None:
                for b, audio_prefix in enumerate(audio_prefixes):
                    if audio_prefix:
                        audio_codes = audio_prefix.popleft()
                        mask = audio_codes != ungenerated
                        audio_tokens[b] = mx.where(mask, audio_codes, audio_tokens[b])

        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for b, (token, state, logged) in enumerate(zip(tokens, states, logged_text_tokens)):
                if text_prefixes is not None and text_prefixes[b]:
                    out_token = text_prefixes[b].popleft()
                else:
                    out_token, _ = self.machine.process(offset, state, token)
                out_tokens.append(out_token)
                logged.append((token, out_token))
            text_tokens[:] = mx.array(out_tokens, dtype=mx.int64)

        lm_gen = LmGen(
            self.lm,
            max_steps=self.max_gen_length,
            text_sampler=Sampler(temp=self.temp),
            audio_sampler=Sampler(temp=self.temp),
            cfg_coef=self.cfg_coef,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
            # TODO(laurent):
            # cfg_is_masked_until=cfg_is_masked_until,
            # cfg_is_no_text=cfg_is_no_text,
        )

        logged_text_tokens = [[] for _ in states]
        frames: list[mx.array] = []

        for offset in range(self.max_gen_length):
            if all(state.end_step is not None for state in states):
                max_end_step = max(state.end_step for state in states)
                if offset >= max_end_step + self.delay_steps + self.final_padding:
                    break
            missing = self.lm.n_q - self.lm.dep_q
            input_tokens = mx.ones((len(states), missing), dtype=mx.int64) * self.machine.token_ids.zero
            lm_gen.step(input_tokens, ct=ct, cross_attention_src=cross_attention_src)
            frame = lm_gen.last_audio_tokens()
            if frame is not None and (frame != self.machine.token_ids.zero).all():
                frames.append(mx.array(frame)[:, :, None])
        return TTSResult(
            frames, logged_text_tokens,
            [state.end_step for state in states],
            [state.consumption_times for state in states],
            [state.transcript for state in states])

    def get_voice_path(self, voice_name: str) -> Path:
        """Returns a local path given a voice name, potentially fetching the voice
        from a HuggingFace repository. To retrieve a voice from another repo, you can also use
        the `hf://REPO/PATH` syntax.
        """
        file = hf_get(voice_name + self.voice_suffix, self.voice_repo,
                      check_local_file_exists=True)
        return Path(file)

    def make_condition_attributes(
            self, voices: list[Path], cfg_coef: float | None = None):
        """Given a list of pre computed voice embeddings, returns a ConditionAttributes.

        Args:
            voices: list of file paths to pre computed voice embeddings, see `get_voice_path`.
            cfg_coef: for model trained with CFG distillation, value of the CFG
                to use as conditioning. Typically, values from 1. to 4. are supported
                with 0.5 increments.
            """
        if voices:
            voice_tensor = None
            mask = None
            for idx in range(5):
                if idx < len(voices):
                    emb: mx.array = mx.load(str(voices[idx]))['speaker_wavs']
                    assert emb.ndim == 3
                    if voice_tensor is None:
                        voice_tensor = mx.zeros((1, self.max_speakers, emb.shape[2], emb.shape[1]))
                    if mask is None:
                        mask = mx.zeros((1, self.max_speakers, emb.shape[2]), dtype=mx.uint8)
                    voice_tensor[:, idx, :, :] = emb.swapaxes(1, 2)
                    mask[:, idx, :] = True
            assert voice_tensor is not None
            assert mask is not None
            voice_tensor = voice_tensor.reshape(1, -1, voice_tensor.shape[-1])
            mask = mask.reshape(1, -1)
            tensors = {
                'speaker_wavs': TensorCondition(voice_tensor, mask)
            }
        else:
            tensors = {}
        text: dict[str, str | None] = {'control': 'ok'}
        if cfg_coef is None:
            text['cfg'] = None
        else:
            if cfg_coef in self.valid_cfg_conditionings:
                text['cfg'] = format(cfg_coef, '.1f')
            else:
                valids = ", ".join(str(x) for x in self.valid_cfg_conditionings)
                raise ValueError(f"Unsupported value for cfg_coef, valid values are {valids}.")
        return ConditionAttributes(text=text, tensor=tensors)

    def get_prefix(self, audio_path: Path | str) -> mx.array:
        wav, _ = sphn.read(audio_path, sample_rate=self.mimi.sample_rate)
        prefix = self.mimi.encode(mx.array(wav)[None])[0, :, :-2]
        null_text = mx.ones_like(prefix[:1]) * self.machine.token_ids.zero
        prefix = mx.concat([null_text, prefix], axis=0)
        return prefix
