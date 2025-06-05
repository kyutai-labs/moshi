# Copyright (c) Kyutai, all rights reserved.

import argparse
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import random
import time
import typing as tp

import numpy as np
from safetensors.torch import load_file
import torch

from moshi.conditioners import ConditionAttributes, dropout_all_conditions, TensorCondition
from moshi.models import loaders, MimiModel, LMModel, LMGen
from moshi.models.lm import _LMGenState
from moshi.modules.transformer import StreamingMultiheadAttention
from pydantic import BaseModel


class MaskFlags(Enum):
    # Output PCM is ready
    HAS_PCM = 1
    # Generation is done, no need to step again.
    IS_EOS = 2
    # One word was consumed in the text stream.
    WORD_FINISHED = 4
    # One AR step was performed.
    AR_STEP = 8
    # AR step was skipped because the client is not sending words fast enough.
    MISSING_WORDS = 16


def flags_out_from_mask_(flags_out: np.ndarray, mask: torch.Tensor, value: int):
    flags_out[mask.numpy()] |= value


class Config(BaseModel):
    log_folder: Path = Path.home() / 'tmp/tts-service'
    hf_repo: str = loaders.DEFAULT_REPO
    mimi_weight: Path | None = None
    moshi_weight: Path = Path.home() / 'models/moshi/moshi_b1d046da_445/checkpoint.safetensors'
    config_path: Path = Path.home() / 'models/moshi/moshi_b1d046da_445/config.json'
    tokenizer: Path = Path.home() / 'models/text-tokenizers/tokenizers/test_en_fr_audio_8000.model'
    device: str = 'cuda'

    n_q: int = 24
    voice_folder: Path = Path.home() / 'models/tts-voices'
    default_voice: str = "barack_demo.wav"

    temp: float = 0.6
    cfg_coef: float = 1.5
    cfg_is_no_text: bool = False

    max_padding: int = 6
    initial_padding: int = 2
    final_padding: int = 4

    interleaved_text_only: int = 2
    debug: bool = False


@dataclass
class TokenIds:
    word = 0
    pad = 3
    main = 1
    other = 2
    zero = -1
    # Text card for the text, embedding, e.g. including the special initial token.
    card: int = 8001


@dataclass
class Entry:
    tokens: list[int]
    word: str
    padding: int = 0


@dataclass
class State:
    entries: deque[Entry]
    lookahead: deque[int]
    remaining_padding: int
    forced_padding: int
    queued: deque[int] = field(default_factory=deque)
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


@dataclass
class StateMachine:
    token_ids: TokenIds
    max_paddings: int = 6
    initial_padding: int = 2
    second_stream_ahead: int = 0

    def new_state(self, entries: tp.Sequence[Entry]) -> State:
        return State(
            entries=deque(entries),
            lookahead=deque(),
            remaining_padding=self.initial_padding,
            forced_padding=self.initial_padding,
        )

    def process(self, step: int, state: State, token: int) -> tuple[int, bool]:
        consumed_new_word = False
        if token not in [self.token_ids.word, self.token_ids.pad]:
            token = self.token_ids.pad

        if state.queued:
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
                    consumed_new_word = True
                    state.transcript.append((entry.word, step))
                    # Entry contains a new word, we reset the max padding counter.
                    state.queued.extend(entry.tokens)
                    if self.second_stream_ahead:
                        state.lookahead.extend(state.get_tokens_ahead(self.second_stream_ahead))
                    state.remaining_padding = self.max_paddings
                else:
                    # Entry is only here to insert a break, pretend the token was a PAD.
                    token = self.token_ids.pad
                state.forced_padding = entry.padding
            else:
                token = self.token_ids.pad
                if self.second_stream_ahead and state.end_step is None:
                    # When using a second input text stream, the model will predict one last WORD token
                    # to mark the end of the stream.
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
        else:
            raise RuntimeError(f"Invalid token {token}")
        assert output is not None

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

        return output, consumed_new_word


def init(batch_size: int, config_override: dict) -> 'TTSService':
    config = Config(**config_override)
    config.log_folder.mkdir(parents=True, exist_ok=True)

    print("retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        config.hf_repo, moshi_weights=config.moshi_weight, mimi_weights=config.mimi_weight,
        config_path=config.config_path, tokenizer=config.tokenizer)
    assert checkpoint_info.model_id is not None
    sig = checkpoint_info.model_id['sig']
    epoch = checkpoint_info.model_id['epoch']
    voice_suffix = f'.{sig}@{epoch}.safetensors'

    print("loading voices")
    all_attributes = {}
    for file in config.voice_folder.glob(f'**/*{voice_suffix}'):
        relative = file.relative_to(config.voice_folder)
        name = str(relative.with_name(relative.name.removesuffix(voice_suffix)))
        try:
            attributes = make_condition_attributes([file, file])
        except Exception:
            print(f"[WARNING] failed to load voice {name}")
        else:
            all_attributes[name] = attributes
    print("loading mimi")
    mimi = checkpoint_info.get_mimi(device=config.device)
    print("mimi loaded")
    print("loading moshi")
    lm = checkpoint_info.get_moshi(device=config.device, dtype=torch.bfloat16)
    print("moshi loaded")

    token_ids = TokenIds(card=lm.text_card + 1)
    machine = StateMachine(
        token_ids=token_ids, max_paddings=config.max_padding, initial_padding=config.initial_padding,
        second_stream_ahead=checkpoint_info.tts_config.get('second_stream_ahead', 0))
    delay_steps = int(checkpoint_info.tts_config['audio_delay'] * mimi.frame_rate)
    service = TTSService(
        batch_size=batch_size, default_attribute_name=config.default_voice,
        all_attributes=all_attributes,
        lm=lm, mimi=mimi, machine=machine, delay_steps=delay_steps,
        temp=config.temp, cfg_coef=config.cfg_coef, cfg_is_no_text=config.cfg_is_no_text,
        final_padding=config.final_padding, n_q=config.n_q, debug=config.debug,
        interleaved_text_only=config.interleaved_text_only)

    return service


@dataclass
class ClientState:
    is_complete: bool = False
    state: State | None = None
    offset: int = 0

    def reset(self, state_machine: StateMachine) -> None:
        self.is_complete = False
        self.offset = 0
        self.state = state_machine.new_state([])


@dataclass
class TTSService:
    batch_size: int
    default_attribute_name: str
    all_attributes: dict[str, ConditionAttributes]

    lm: LMModel
    mimi: MimiModel

    machine: StateMachine
    delay_steps: int

    temp: float = 0.6
    cfg_coef: float = 1.0
    cfg_is_no_text: bool = False
    final_padding: int = 4
    n_q: int = 32
    max_gen_length: int = 64000
    debug: bool = False
    interleaved_text_only: int = 0

    flags_out: np.ndarray | None = None
    clients: list[ClientState] = field(default_factory=list)
    cross_attention_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    cross_attentions: list[StreamingMultiheadAttention] = field(default_factory=list)

    def __post_init__(self):
        self.device = self.lm.device
        self.dtype = self.lm.dtype
        self.lm.dep_q = self.n_q
        self.remaining_text_only = self.interleaved_text_only

        for _ in range(self.batch_size):
            client = ClientState()
            self.clients.append(client)

        print("Filling cross attention cache.")
        for name, attributes in self.all_attributes.items():
            self.cross_attention_cache[name] = self._get_cross_attention_source([attributes])

        assert self.lm.condition_provider is not None
        cas = [self.all_attributes[self.default_attribute_name]] * self.batch_size
        if self.cfg_coef != 1.0:
            nulled = make_null(cas)
            cas = cas + nulled
        prepared = self.lm.condition_provider.prepare(cas)
        condition_tensors = self.lm.condition_provider(prepared)

        for module in self.lm.modules():
            if isinstance(module, StreamingMultiheadAttention) and module.cross_attention:
                self.cross_attentions.append(module)

        self.lm_gen = LMGen(
            self.lm, temp=self.temp, temp_text=self.temp, cfg_coef=self.cfg_coef,
            condition_tensors=condition_tensors, on_text_hook=self._on_text_hook,
            on_audio_hook=self._on_audio_hook, support_out_of_sync=True, cfg_is_no_text=self.cfg_is_no_text)
        self.lm_gen.streaming_forever(self.batch_size)
        self.mimi.streaming_forever(self.batch_size)
        missing = self.lm.n_q - self.lm.dep_q
        self.input_tokens = torch.full(
            (self.batch_size, missing, 1), self.machine.token_ids.zero,
            dtype=torch.long, device=self.device)
        self.no_depformer_tokens = torch.full(
            (self.batch_size, self.lm.dep_q, 1), self.machine.token_ids.zero,
            dtype=torch.long, device=self.device)
        self.last_actives: list[bool] = [False] * self.batch_size
        print("warming up.")
        for _ in range(3):
            self.mimi.set_exec_mask(torch.ones(self.batch_size, dtype=torch.bool))
            self.lm_gen.set_exec_mask(torch.ones(self.batch_size, dtype=torch.bool))
            frame = self.lm_gen.step(self.input_tokens)
            assert frame is not None
            self.mimi.decode(frame[:, 1:].clamp(min=0))
        print("ready to roll.")

    def _get_cross_attention_source(self, all_attributes: list[ConditionAttributes]) -> torch.Tensor:
        assert self.lm.condition_provider is not None
        assert self.lm.fuser is not None
        prepared = self.lm.condition_provider.prepare(all_attributes)
        condition_tensors = self.lm.condition_provider(prepared)
        cross = self.lm.fuser.get_cross(condition_tensors)
        assert cross is not None
        return cross.to(device=self.device, dtype=self.dtype)

    @property
    def _lm_gen_state(self) -> _LMGenState:
        assert self.lm_gen._streaming_state is not None
        return self.lm_gen._streaming_state

    def _on_audio_hook(self, audio_tokens: torch.Tensor) -> None:
        delays = self.lm_gen.delays_cuda[1: 1 + self.lm.dep_q]
        mask = self._lm_gen_state.offsets[:, None] < delays + self.delay_steps
        audio_tokens.masked_fill_(mask, self.machine.token_ids.zero)

    def _on_text_hook(self, text_tokens) -> None:
        tokens = text_tokens.tolist()
        out_tokens = []
        for b, (token, client) in enumerate(zip(tokens, self.clients)):
            if not self.last_actives[b]:
                out_tokens.append(token)
                continue
            assert client.state is not None
            out_token, consumed_new_word = self.machine.process(client.offset, client.state, token)

            if self.flags_out is not None and consumed_new_word:
                self.flags_out[b] |= MaskFlags.WORD_FINISHED.value
            out_tokens.append(out_token)
        text_tokens[:] = torch.tensor(out_tokens, dtype=torch.long, device=text_tokens.device)

    def _print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    @torch.no_grad()
    def step(self, updates: list[tuple[int, list[int], np.ndarray | str | None]], pcm_out: np.ndarray,
             flags_out: np.ndarray, code_out: np.ndarray) -> None:
        self.flags_out = flags_out
        flags_out[:] = 0

        reset_mask = torch.zeros(self.batch_size, dtype=torch.bool)
        # List of pre computed cross attention values.
        new_cross_sources: list[torch.Tensor] = []
        new_cross_indexes: list[int] = []
        # List of new dynamic conditioning that we need to compute.
        new_voice_indexes: list[int] = []
        new_voice_sources: list[torch.Tensor] = []
        for b, new_entry, voice in updates:
            client = self.clients[b]
            if not new_entry:
                self._print(f"[{b}] NO TOKENS REALLY LAURENT.")
            if new_entry[0] == -1:
                client.reset(self.machine)
                reset_mask[b] = True
                new_entry = new_entry[1:]
                if isinstance(voice, np.ndarray):
                    new_voice_indexes.append(b)
                    new_voice_sources.append(torch.from_numpy(voice))
                else:
                    cross_source = self.cross_attention_cache.get(voice or '', None)
                    if cross_source is None:
                        cross_source = self.cross_attention_cache[self.default_attribute_name]
                    new_cross_sources.append(cross_source)
                    new_cross_indexes.append(b)
                self._print(f"[{b}] Reset, voice is {voice}.")
            if client.state is None:
                self._print(f"[{b}] Trying to push {new_entry}, but not assigned.")
            elif new_entry == [-2]:
                self._print(f"[{b}] Done.")
                client.is_complete = True
            else:
                self._print(f"[{b}] Pushing {new_entry}.")
                client.state.entries.append(Entry(new_entry, ''))

        actives = []
        mimi_actives = []
        in_text_onlys = []
        for b, client in enumerate(self.clients):
            if client.state is None:
                # client is not currently assigned.
                active = False
            elif client.is_complete:
                # We got all the words from the client and are wrapping up.
                active = True
            elif client.state.forced_padding > 0:
                # We are sure we won't try to consume a word at this point.
                active = True
            elif len(client.state.entries) > self.machine.second_stream_ahead:
                # We have some words ready to be consumed.
                active = True
            else:
                flags_out[b] |= MaskFlags.MISSING_WORDS.value
                active = False
            actives.append(active)

            real_offset = client.offset - self.lm_gen.max_delay

            mimi_active = active and (real_offset >= self.delay_steps)
            mimi_actives.append(mimi_active)

            in_text_only = active and (client.offset < self.delay_steps)
            in_text_onlys.append(in_text_only)

        in_text_only_mask = torch.tensor(in_text_onlys, dtype=torch.bool)
        run_in_text_only = self.remaining_text_only > 0 and in_text_only_mask.any()

        if run_in_text_only:
            self.remaining_text_only -= 1
            mimi_exec_mask = torch.zeros(self.batch_size, dtype=torch.bool)
            exec_mask = in_text_only_mask
            actives = in_text_onlys
        else:
            self.remaining_text_only = self.interleaved_text_only
            exec_mask = torch.tensor(actives, dtype=torch.bool)
            mimi_exec_mask = torch.tensor(mimi_actives, dtype=torch.bool)
        del mimi_actives
        self.last_actives = actives

        flags_out_from_mask_(flags_out, exec_mask, MaskFlags.AR_STEP.value)
        flags_out_from_mask_(flags_out, mimi_exec_mask, MaskFlags.HAS_PCM.value)

        # We check on exec_mask whether we actually need to run anything, before we move it to CUDA.
        # However, we still need to perform the reset and update of cross attention for models
        # with a text lookahead stream.
        skip_exec = not exec_mask.any()

        exec_mask = exec_mask.to(self.device)
        mimi_exec_mask = mimi_exec_mask.to(self.device)
        need_reset = reset_mask.any()
        reset_mask = reset_mask.to(self.device)

        if new_voice_sources:
            all_attributes = [make_condition_attributes([voice_source])
                              for voice_source in new_voice_sources]
            new_cross_sources += self._get_cross_attention_source(all_attributes).split(1)
            new_cross_indexes += new_voice_indexes
        if new_cross_sources:
            cross_source = torch.cat(new_cross_sources)
            cross_indexes = torch.tensor(new_cross_indexes, dtype=torch.long, device=self.device)
            for attention in self.cross_attentions:
                k, v = attention._compute_cross_attention(cross_source, cross_source)
                state = attention._streaming_state
                assert state is not None
                assert state.k_cross is not None
                assert state.v_cross is not None
                state.k_cross.index_copy_(0, cross_indexes, k)
                state.v_cross.index_copy_(0, cross_indexes, v)

        if need_reset:
            self.lm_gen.reset_streaming(reset_mask=reset_mask)
            self.mimi.reset_streaming(reset_mask=reset_mask)

        if skip_exec:
            return

        self.lm_gen.set_exec_mask(exec_mask)
        self.mimi.set_exec_mask(mimi_exec_mask)

        depformer_replace_tokens = self.no_depformer_tokens if run_in_text_only else None
        frame = self.lm_gen.step(self.input_tokens, depformer_replace_tokens=depformer_replace_tokens)
        assert frame is not None
        audio_frame = frame[:, 1:]
        audio_frame.clamp_(min=0)

        if run_in_text_only:
            pcm = None
        else:
            pcm = self.mimi.decode(audio_frame)
            pcm.clamp_(-0.99, 0.99)

        for b, client in enumerate(self.clients):
            if actives[b]:
                assert client.state is not None
                client.offset += 1
                self._print(f"[{b}] Offset {client.offset: 3d}, pendings={len(client.state.entries): 3d}.")
                if client.is_complete and client.state.end_step is not None:
                    # We were waiting for the end of the generation.
                    real_end = client.state.end_step + self.delay_steps + self.final_padding + self.lm_gen.max_delay
                    if client.offset >= real_end:
                        self._print(f"[{b}] Done.")
                        client.reset(self.machine)
                        flags_out[b] |= MaskFlags.IS_EOS.value
        if pcm is not None:
            pcm_out[:] = pcm[:, 0].cpu().numpy()
        code_out[:, :frame.shape[1]] = frame[:, :, 0].int().cpu().numpy()
        code_out[:, frame.shape[1]:] = 0
        self.flags_out = None


class Profiler:
    """Context manager wrapper for xformers profiler.
    """
    def __init__(self, enabled: bool = False):
        self.profiler: tp.Optional[tp.Any] = None
        if enabled:
            from xformers.profiler import profile
            from xformers.profiler.api import PyTorchProfiler
            output_dir = './profiler_data'
            schedule = (
                (PyTorchProfiler, 6, 12),
            )
            self.profiler = profile(output_dir=output_dir, schedule=schedule)

    def step(self):
        if self.profiler is not None:
            self.profiler.step()  # type: ignore

    def __enter__(self):
        if self.profiler is not None:
            return self.profiler.__enter__()  # type: ignore

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.profiler is not None:
            return self.profiler.__exit__(exc_type, exc_value, exc_tb)  # type: ignore


def make_condition_attributes(voices: list[Path | torch.Tensor],
                              max_speakers: int = 5) -> ConditionAttributes:
    assert voices
    voice_tensor = None
    mask = None
    for idx in range(5):
        if idx < len(voices):
            voice = voices[idx]
            if isinstance(voice, Path):
                emb = load_file(voice, device='cuda')['speaker_wavs']
            else:
                emb = voice
            assert emb.dim() == 3
            if voice_tensor is None:
                voice_tensor = torch.zeros(1, max_speakers, emb.shape[2], emb.shape[1], device='cuda')
            if mask is None:
                mask = torch.zeros(1, max_speakers, emb.shape[2], dtype=torch.bool, device='cuda')
            voice_tensor[:, idx, :, :] = emb.transpose(1, 2)
            mask[:, idx, :] = True
    assert voice_tensor is not None
    assert mask is not None
    voice_tensor = voice_tensor.view(1, -1, voice_tensor.shape[-1])
    mask = mask.view(1, -1)
    tensors = {
        'speaker_wavs': TensorCondition(voice_tensor, mask)
    }
    return ConditionAttributes(text={'control': 'ok'}, tensor=tensors)


def make_null(all_attributes: tp.Sequence[ConditionAttributes]) -> list[ConditionAttributes]:
    return dropout_all_conditions(all_attributes)


if __name__ == '__main__':
    rng = random.Random(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--profile', action='store_true')
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    args = parser.parse_args()
    bs = args.batch_size
    service = init(batch_size=bs, config_override={})
    print("Service initialized")
    pcm_out = np.zeros((bs, 1920))
    flags_out = np.zeros(bs, dtype=np.int32)
    code_out = np.zeros((bs, 33), dtype=np.int32)
    service.step([(0, [-1], '')], pcm_out=pcm_out, flags_out=flags_out, code_out=code_out)
    profiler = Profiler(enabled=args.profile)
    with profiler:
        for _ in range(100):
            inp = []
            if rng.random() < 0.1:
                word = [13, 34]
                inp.append((0, word, None))
            be = time.time()
            service.step(inp, pcm_out=pcm_out, flags_out=flags_out, code_out=code_out)
            el = time.time() - be
            print(f"FR {el * 1000:.1f}ms")
            profiler.step()
