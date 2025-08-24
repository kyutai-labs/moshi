# Copyright (c) Kyutai, all rights reserved.

import argparse
from dataclasses import dataclass, field
from enum import Enum
import huggingface_hub
from pathlib import Path
import random
import time
import typing as tp

import numpy as np
from safetensors.torch import load_file
import torch

from moshi.conditioners import ConditionAttributes, dropout_all_conditions, TensorCondition
from moshi.models import loaders
from moshi.models.lm import _LMGenState, LMGen
from moshi.models.tts import TTSModel, Entry, State, StateMachine, DEFAULT_DSM_TTS_REPO
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


def split_at_specific_separator(text: str, separator: str, index_of_separator: int) -> tuple[str, str]:
    """ kyutai/tts-voices/unmute-prod-website/*.safetensors
    becomes
    ('kyutai/tts-voices', 'unmute-prod-website/*.safetensors)
    with index_of_separator=1.
    """
    if text.count(separator) <= index_of_separator:
        raise ValueError(f"Separator '{separator}' not found {index_of_separator + 1} times in `{text}`.")
    parts = text.split(separator, index_of_separator + 1)
    return separator.join(parts[:-1]), parts[-1]


class Config(BaseModel):
    log_folder: Path = Path.home() / 'tmp/tts-service'
    hf_repo: str = DEFAULT_DSM_TTS_REPO
    mimi_weight: Path | None = None
    moshi_weight: Path | None = None
    config_path: Path | None = None
    tokenizer: Path | None = None
    device: str = 'cuda'

    n_q: int = 24
    # This can have multiple formats:
    # - A path to a folder with voices, e.g. `models/tts`
    # - A huggingface snapshot, e.g. `hf-snapshot://kyutai/tts-voices`
    # - A huggingface snapshot with a pattern,
    #     e.g. `hf-snapshot://kyutai/tts-voices/unmute-prod-website/*.safetensors`
    voice_folder: str = str(Path.home() / 'models/tts-voices')
    default_voice: str = "barack_demo.wav"

    temp: float = 0.6
    cfg_coef: float = 2.

    max_padding: int = 8
    initial_padding: int = 2
    final_padding: int = 4
    padding_between: int = 1
    padding_bonus: float = 0.

    interleaved_text_only: int = 2
    debug: bool = False


def init(batch_size: int, config_override: dict) -> 'TTSService':
    config = Config(**config_override)
    config.log_folder.mkdir(parents=True, exist_ok=True)

    print("retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        config.hf_repo, moshi_weights=config.moshi_weight, mimi_weights=config.mimi_weight,
        config_path=config.config_path, tokenizer=config.tokenizer)

    cfg_condition = None
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info, n_q=config.n_q, temp=config.temp, cfg_coef=config.cfg_coef,
        max_padding=config.max_padding, initial_padding=config.initial_padding, final_padding=config.final_padding,
        device=config.device, padding_bonus=config.padding_bonus)
    if tts_model.valid_cfg_conditionings:
        # Model was trained with CFG distillation.
        cfg_condition = tts_model.cfg_coef
        tts_model.cfg_coef = 1.
        cfg_is_no_text = False
    else:
        cfg_is_no_text = True

    voice_suffix = tts_model.voice_suffix
    print(f"loading voices from {config.voice_folder}, with suffix {voice_suffix}.")
    all_attributes = {}
    voice_folder = config.voice_folder
    if voice_folder.startswith("hf-snapshot://"):
        voice_folder = voice_folder.removeprefix("hf-snapshot://")
        # We detect if there is a pattern in the voice folder.
        if voice_folder.count("/") > 1:
            voice_folder, pattern = split_at_specific_separator(voice_folder, '/', 1)
        else:
            pattern = None
        print(f"retrieving voices from {voice_folder}")
        voice_folder = huggingface_hub.snapshot_download(voice_folder, allow_patterns=pattern)
    voice_folder = Path(voice_folder)

    if tts_model.multi_speaker:
        for file in voice_folder.glob(f'**/*{voice_suffix}'):
            relative = file.relative_to(voice_folder)
            name = str(relative.with_name(relative.name.removesuffix(voice_suffix)))
            try:
                attributes = tts_model.make_condition_attributes([file, file], cfg_coef=cfg_condition)
            except Exception:
                print(f"[WARNING] failed to load voice {name}")
            else:
                all_attributes[name] = attributes

        if not all_attributes:
            raise RuntimeError(
                "No voices found, please check your voice folder. "
                f"Searched for files matching {voice_folder}/**/*{voice_suffix}"
            )

        if config.default_voice not in all_attributes:
            raise RuntimeError(
                f"Default voice {config.default_voice}, please check your voice folder. "
                f"Expected {voice_folder}/{config.default_voice}{voice_suffix} to exist"
            )

    service = TTSService(
        batch_size=batch_size, default_attribute_name=config.default_voice,
        all_attributes=all_attributes,
        tts_model=tts_model,
        cfg_condition=cfg_condition,
        cfg_is_no_text=cfg_is_no_text,
        padding_between=config.padding_between,
        padding_bonus=config.padding_bonus,
        debug=config.debug,
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

    def is_active(self, lookahead: int) -> bool:
        state = self.state
        if state is None:
            return False
        if self.is_complete:
            return True
        if not state.entries:
            # No entries, not safe to run as we might need to pop a new one.
            return False
        if lookahead == 0:
            # If no lookahead, having at least one entry means we can run.
            return True
        if not state.entries[0].tokens:
            # Next entry is just padding, we don't need a lookahead here.
            return True
        # now we need at least lookahead words on top of the current one.
        remaining = lookahead + 1
        # iterating from the start of entries, as it is a deque so harder to skip the first item.
        for entry in state.entries:
            if entry.tokens:
                remaining -= 1
            if remaining <= 0:
                return True
        return False


@dataclass
class TTSService:
    batch_size: int
    default_attribute_name: str
    all_attributes: dict[str, ConditionAttributes]

    tts_model: TTSModel

    cfg_is_no_text: bool = True
    cfg_condition: float | None = None
    padding_between: int = 1
    padding_bonus: float = 0.0
    n_q: int = 32
    debug: bool = False
    interleaved_text_only: int = 0

    flags_out: np.ndarray | None = None
    clients: list[ClientState] = field(default_factory=list)
    cross_attention_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    cross_attentions: list[StreamingMultiheadAttention] = field(default_factory=list)

    def __post_init__(self):
        lm = self.tts_model.lm
        tts_model = self.tts_model
        mimi = self.tts_model.mimi
        machine = self.tts_model.machine

        self.device = lm.device
        self.dtype = lm.dtype
        self.remaining_text_only = self.interleaved_text_only

        for _ in range(self.batch_size):
            client = ClientState()
            self.clients.append(client)

        if tts_model.multi_speaker:
            print("Filling cross attention cache.")
            for name, attributes in self.all_attributes.items():
                self.cross_attention_cache[name] = self._get_cross_attention_source([attributes])
            assert lm.condition_provider is not None

            cas = [self.all_attributes[self.default_attribute_name]] * self.batch_size
            if self.tts_model.cfg_coef != 1.0:
                nulled = make_null(cas)
                cas = cas + nulled
            prepared = lm.condition_provider.prepare(cas)
            condition_tensors = lm.condition_provider(prepared)
        else:
            condition_tensors = {}

        for module in lm.modules():
            if isinstance(module, StreamingMultiheadAttention) and module.cross_attention:
                self.cross_attentions.append(module)

        self.lm_gen = LMGen(
            lm, temp=tts_model.temp, temp_text=tts_model.temp, cfg_coef=tts_model.cfg_coef,
            condition_tensors=condition_tensors, on_text_hook=self._on_text_hook,
            on_audio_hook=self._on_audio_hook, cfg_is_no_text=self.cfg_is_no_text,
            support_out_of_sync=True, on_text_logits_hook=self._on_text_logits_hook)
        self.lm_gen.streaming_forever(self.batch_size)
        mimi.streaming_forever(self.batch_size)

        missing = lm.n_q - lm.dep_q
        self.input_tokens = torch.full(
            (self.batch_size, missing, 1), machine.token_ids.zero,
            dtype=torch.long, device=self.device)
        self.no_depformer_tokens = torch.full(
            (self.batch_size, lm.dep_q, 1), machine.token_ids.zero,
            dtype=torch.long, device=self.device)
        self.last_actives: list[bool] = [False] * self.batch_size
        print("warming up.")
        for _ in range(3):
            mimi.set_exec_mask(torch.ones(self.batch_size, dtype=torch.bool))
            self.lm_gen.set_exec_mask(torch.ones(self.batch_size, dtype=torch.bool))
            frame = self.lm_gen.step(self.input_tokens)
            assert frame is not None
            mimi.decode(frame[:, 1:].clamp(min=0))
        print("ready to roll.")

    def _get_cross_attention_source(self, all_attributes: list[ConditionAttributes]) -> torch.Tensor:
        lm = self.tts_model.lm
        assert lm.condition_provider is not None
        assert lm.fuser is not None
        prepared = lm.condition_provider.prepare(all_attributes)
        condition_tensors = lm.condition_provider(prepared)
        cross = lm.fuser.get_cross(condition_tensors)
        assert cross is not None
        return cross.to(device=self.device, dtype=self.dtype)

    @property
    def _lm_gen_state(self) -> _LMGenState:
        assert self.lm_gen._streaming_state is not None
        return self.lm_gen._streaming_state

    def _on_audio_hook(self, audio_tokens: torch.Tensor) -> None:
        delays = self.lm_gen.delays_cuda[1: 1 + self.tts_model.lm.dep_q]
        mask = self._lm_gen_state.offsets[:, None] < delays + self.tts_model.delay_steps
        audio_tokens.masked_fill_(mask, self.tts_model.machine.token_ids.zero)

    def _on_text_logits_hook(self, text_logits):
        if self.padding_bonus:
            text_logits[..., self.tts_model.machine.token_ids.pad] += self.padding_bonus
        return text_logits

    def _on_text_hook(self, text_tokens) -> None:
        tokens = text_tokens.tolist()
        out_tokens = []
        for b, (token, client) in enumerate(zip(tokens, self.clients)):
            if not self.last_actives[b]:
                out_tokens.append(token)
                continue
            assert client.state is not None
            out_token, consumed_new_word = self.tts_model.machine.process(client.offset, client.state, token)

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
        mimi = self.tts_model.mimi
        machine = self.tts_model.machine
        delay_steps = self.tts_model.delay_steps
        lookahead = self.tts_model.machine.second_stream_ahead
        pad = self.tts_model.machine.token_ids.pad
        multi_speaker = self.tts_model.multi_speaker

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
                client.reset(machine)
                reset_mask[b] = True
                new_entry = new_entry[1:]
                if multi_speaker:
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
            elif new_entry[0] == pad:
                self._print(f"[{b}] Pushing pause {new_entry}.")
                padding = len(new_entry)
                client.state.entries.append(Entry([], '', padding=padding))
            else:
                self._print(f"[{b}] Pushing {new_entry}.")
                padding = 0
                if self.padding_between > 0:
                    padding = max(0, self.padding_between + len(new_entry) - 1)
                client.state.entries.append(Entry(new_entry, '', padding=padding))

        actives = []
        mimi_actives = []
        in_text_onlys = []
        for b, client in enumerate(self.clients):
            active = client.is_active(lookahead)
            if not active and client.state is not None:
                flags_out[b] |= MaskFlags.MISSING_WORDS.value
            actives.append(active)

            real_offset = client.offset - self.lm_gen.max_delay

            mimi_active = active and (real_offset >= delay_steps)
            mimi_actives.append(mimi_active)

            in_text_only = active and (client.offset < delay_steps)
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
            all_attributes = [make_condition_attributes([voice_source], cfg_condition=self.cfg_condition)
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
            mimi.reset_streaming(reset_mask=reset_mask)

        if skip_exec:
            time.sleep(0.001)  # Sleep a bit to avoid busy waiting.
            return

        self.lm_gen.set_exec_mask(exec_mask)
        mimi.set_exec_mask(mimi_exec_mask)

        depformer_replace_tokens = self.no_depformer_tokens if run_in_text_only else None
        frame = self.lm_gen.step(self.input_tokens, depformer_replace_tokens=depformer_replace_tokens)
        assert frame is not None
        audio_frame = frame[:, 1:]
        audio_frame.clamp_(min=0)

        if run_in_text_only:
            pcm = None
        else:
            pcm = mimi.decode(audio_frame)
            pcm.clamp_(-0.99, 0.99)

        for b, client in enumerate(self.clients):
            if actives[b]:
                assert client.state is not None
                client.offset += 1
                self._print(f"[{b}] Offset {client.offset: 3d}, pendings={len(client.state.entries): 3d}.")
                if client.is_complete and client.state.end_step is not None:
                    # We were waiting for the end of the generation.
                    real_end = (
                        client.state.end_step + delay_steps + self.tts_model.final_padding + self.lm_gen.max_delay)
                    if client.offset >= real_end:
                        self._print(f"[{b}] Done.")
                        client.reset(machine)
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
                              max_speakers: int = 5,
                              cfg_condition: float | None = None) -> ConditionAttributes:
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
    text: dict[str, str | None] = {
        'control': 'ok',
    }
    if cfg_condition is None:
        text['cfg'] = None
    else:
        text['cfg'] = format(cfg_condition, '.1f')
    return ConditionAttributes(text=dict(text), tensor=tensors)


def make_null(all_attributes: tp.Sequence[ConditionAttributes]) -> list[ConditionAttributes]:
    return dropout_all_conditions(all_attributes)


if __name__ == '__main__':
    rng = random.Random(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--profile', action='store_true')
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-c', '--cfg_coef', default=2., type=float)
    parser.add_argument('-r', '--hf-repo', default=DEFAULT_DSM_TTS_REPO)
    args = parser.parse_args()
    bs = args.batch_size
    config_override = {
        'hf_repo': args.hf_repo,
        'cfg_coef': args.cfg_coef,
        'voice_folder': 'hf-snapshot://kyutai/tts-voices/unmute-prod-website/*.safetensors',
        'default_voice': 'unmute-prod-website/default_voice.wav',
    }
    service = init(batch_size=bs, config_override=config_override)
    print("Service initialized")
    pcm_out = np.zeros((bs, 1920))
    flags_out = np.zeros(bs, dtype=np.int32)
    code_out = np.zeros((bs, 33), dtype=np.int32)
    service.step([(0, [-1, 32, 21], '')], pcm_out=pcm_out, flags_out=flags_out, code_out=code_out)
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
