# Copyright (c) Kyutai, all rights reserved.

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import moshi.models
import numpy as np
import torch
from moshi.models import LMModel, MimiModel, loaders
from pydantic import BaseModel


class MaskFlags(Enum):
    # This user is active and sending audio
    ACTIVE = 1 << 0
    # This user has sent the end of stream marker, but we are still performing asr
    MARKER_RECEIVED = 1 << 1
    # Marker has been sent back, end of stream
    IS_EOS = 1 << 2


class UpdateFlags(Enum):
    NODATA = 0
    # This user is active and sending audio
    ACTIVE = -1
    # This user needs to be reset
    RESET = -2
    # A strictly positive value indicates a marker and the number of steps left to perform


class Config(BaseModel):
    log_folder: Path = Path.home() / "tmp/tts-service"
    hf_repo: str = loaders.DEFAULT_REPO
    mimi_weight: Path | None = None
    moshi_weight: Path = (
        Path.home() / "models/moshi/moshi_b1d046da_445/checkpoint.safetensors"
    )
    config_path: Path = Path.home() / "models/moshi/moshi_b1d046da_445/config.json"
    tokenizer: Path = (
        Path.home() / "models/mimi/tokenizer-e351c8d8-checkpoint125.safetensors"
    )
    asr_delay_in_tokens: int = 32
    device: str = "cuda"
    temp: float = 0.0
    debug: bool = False


def init(batch_size: int, config_override: dict) -> "ASRService":
    config = Config(**config_override)
    config.log_folder.mkdir(parents=True, exist_ok=True)

    print("retrieving checkpoint...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        config.hf_repo,
        moshi_weights=config.moshi_weight,
        mimi_weights=config.mimi_weight,
        config_path=config.config_path,
        tokenizer=config.tokenizer,
    )
    print("done.")
    assert checkpoint_info.model_id is not None

    print("loading mimi...")
    mimi = checkpoint_info.get_mimi(device=config.device)
    print("mimi loaded.")
    print("loading moshi...")
    lm = checkpoint_info.get_moshi(
        device=config.device,
        dtype=torch.bfloat16,
    )
    print("moshi loaded.")
    service = ASRService(
        batch_size=batch_size,
        temp=config.temp,
        lm=lm,
        mimi=mimi,
        asr_delay_in_tokens=config.asr_delay_in_tokens,
        device=config.device,
        debug=config.debug,
    )
    return service


@dataclass
class ClientState:
    is_complete: bool = False
    active: bool = False
    # Steps since the beginning of the asr.
    offset: int = 0
    # Step when the asr will be complete.
    real_end: int = 0

    def reset(self) -> None:
        self.active = False
        self.is_complete = False
        self.offset = 0
        self.real_end = 0


@dataclass
class ASRService:
    batch_size: int
    lm: LMModel
    mimi: MimiModel
    asr_delay_in_tokens: int

    device: str = "cuda"
    temp: float = 0.0
    debug: bool = False
    clients: list[ClientState] = field(default_factory=list)

    def __post_init__(self):
        self.lm_gen = moshi.models.LMGen(self.lm, temp_text=self.temp)
        self.lm_gen.streaming_forever(self.batch_size)
        self.mimi.streaming_forever(self.batch_size)
        self.clients = [ClientState() for _ in range(self.batch_size)]

        print("warming up...")
        for _ in range(3):
            self.mimi.set_exec_mask(torch.ones(self.batch_size, dtype=torch.bool))
            self.lm_gen.set_exec_mask(torch.ones(self.batch_size, dtype=torch.bool))
            batch_zeros = torch.zeros(
                (self.batch_size, 1, self.mimi.frame_size),
                dtype=torch.float32,
                device=self.device,
            )
            audio_tokens = self.mimi.encode(batch_zeros)
            frame = self.lm_gen.step(audio_tokens)
            assert frame is not None
        print("ready to roll !")

    @torch.no_grad()
    def step(
        self,
        batch_pcm: np.ndarray,
        flags_out: np.ndarray,
        tokens_out: np.ndarray,
        extra_heads: np.ndarray,
        updates: list[int],
    ) -> None:
        reset_mask = torch.zeros(self.batch_size, dtype=torch.bool)
        for batch_idx, update in enumerate(updates):
            if update == UpdateFlags.NODATA.value:
                self.clients[batch_idx].active = False
                flags_out[batch_idx] = 0
            if update == UpdateFlags.ACTIVE.value:
                self.clients[batch_idx].active = True
                flags_out[batch_idx] = MaskFlags.ACTIVE.value
            if update == UpdateFlags.RESET.value:
                self.clients[batch_idx].reset()
                flags_out[batch_idx] = MaskFlags.ACTIVE.value
                reset_mask[batch_idx] = True
                self.clients[batch_idx].active = True
            if update > 0:
                self.clients[batch_idx].is_complete = True
                flags_out[batch_idx] = MaskFlags.MARKER_RECEIVED.value
                self.clients[batch_idx].real_end = (
                    self.clients[batch_idx].offset
                    + update
                    + self.asr_delay_in_tokens
                    + 2  # Add padding to ensure asr is complete
                )

        exec_mask = torch.tensor(
            [client.active for client in self.clients],
            dtype=torch.bool,
            device=self.device,
        )
        need_reset = reset_mask.any()
        reset_mask = reset_mask.to(self.device)
        skip_exec = not exec_mask.any()
        exec_mask = exec_mask.to(self.device)

        if need_reset:
            self.lm_gen.reset_streaming(reset_mask=reset_mask)
            self.mimi.reset_streaming(reset_mask=reset_mask)

        if skip_exec:
            return

        self.lm_gen.set_exec_mask(exec_mask)
        self.mimi.set_exec_mask(exec_mask)

        batch_pcm = torch.from_numpy(batch_pcm).to(self.device)
        frame_size = self.mimi.frame_size
        batch_pcm = batch_pcm.view(self.batch_size, frame_size)

        assert batch_pcm.shape[-1] % frame_size == 0, (
            "batch_pcm length must be a multiple of frame_size"
        )

        batch_pcm = batch_pcm.unsqueeze(1)

        audio_tokens = self.mimi.encode(batch_pcm)
        text_tokens, extra_heads_list = self.lm_gen.step_with_extra_heads(audio_tokens)

        extra_heads_stacked = torch.stack(extra_heads_list, dim=0)
        extra_heads_stacked = extra_heads_stacked[:, :, 0, 0]

        extra_heads[:, :] = (
            extra_heads_stacked.transpose(0, 1).to(torch.float32).cpu().numpy()
        )
        tokens_np = text_tokens.cpu().numpy()
        tokens_out[:] = tokens_np[:, 0, 0]

        for batch_idx, client in enumerate(self.clients):
            if client.active:
                client.offset += 1
                # We were waiting for the end of the generation.
                if client.is_complete and client.offset >= client.real_end:
                    client.active = False
                    flags_out[batch_idx] = MaskFlags.IS_EOS.value
