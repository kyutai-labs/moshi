import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Union

import torch
import torchaudio
from lhotse.utils import Seconds

from moshi.models import loaders

# os.environ['NO_TORCH_COMPILE'] = '1'
os.environ["NO_CUDA_GRAPH"] = "1"


model_path = "/data0/questar/models/hf/moshika-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors"


class MoshiMimi:
    def __init__(
        self,
        model_safetensors_path: Union[Path, str],
        device: Any = None,
        sampling_rate: int = 24000,
        frame_shift: Seconds = 0.08,
        num_quantizers: int = 8,
    ) -> None:
        # Instantiate a pretrained model
        assert sampling_rate == 24000
        assert num_quantizers in [8, 32]
        assert frame_shift == 0.08

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
        if isinstance(device, int):
            device = torch.device("cpu") if device == -1 else torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, torch.device):
            pass
        else:
            raise ValueError(f"Illegal device is used, expected: int, str(cpu, cuda:0) or None, but get {device}")

        self._device = device

        codec = loaders.get_mimi(model_safetensors_path, device)
        logging.warning(
            "MoshiMimi only support 8 codebook for now. \
                please rewrite moshi.models.get_mimi function If 32 codebook is needed."
        )
        self.codec = codec
        self.dtype = torch.float32
        self.codec._start_streaming(1)

        self.frame_shift = frame_shift
        self.num_quantizers = num_quantizers
        self.sample_rate = sampling_rate
        self.chunk_size = None

    @property
    def device(self):
        return self._device

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        # wav.shape == [1, 1, T]

        if isinstance(wav, torch.Tensor):
            raise AssertionError

        if wav.ndim != 3:
            wav = wav.contiguous().view(1, 1, -1)

        # if self.chunk_size == None or self.chunk_size == wav.shape[-1]:
        #     self.chunk_size = wav.shape[-1]
        #     codes = self.codec.encode(wav)

        # elif self.chunk_size != wav.shape[-1]:
        #     assert self.chunk_size > wav.shape[-1]

        #     wav_original_length = wav.shape[-1]
        #     diff_length = self.chunk_size - wav.shape[-1]
        #     wav = torch.nn.functional.pad(wav, (0, diff_length), 'constant', 0)

        #     codes = self.codec.encode(wav)[:, :, :wav_original_length // 1920]

        codes = self.codec.encode(wav)

        if isinstance(codes, torch.Tensor) and torch.is_floating_point(codes):
            return codes

        # codes = codes.type(torch.int16).permute(0, 2, 1)  # BxTxQ
        return codes[:, : self.num_quantizers, :]


if __name__ == "__main__":
    from time import time

    wav_path = ["/home/wumenglin/workspace/devcontainer/temp/data/fe_03_00001.wav"]
    device = "cuda:0"
    audio_chunk_length = 1920
    tokenizer = MoshiMimi(model_path, device=device, sampling_rate=24000, num_quantizers=8)

    with torch.no_grad():
        for path in wav_path:
            audio, _ = torchaudio.load(path)
            audio = torchaudio.transforms.Resample(44100, 24000)(audio)

            audio = audio[0][: 1 * 60 * 24000]
            audio_new = deepcopy(audio)

            start = time()
            codes_list = []
            for i in range(0, audio.shape[-1], audio_chunk_length):
                audio_chunk = audio[..., i : i + audio_chunk_length]
                codes = tokenizer.encode(torch.tensor(audio_chunk).to(device))
                codes_list.append(codes)
            codes_chunck = torch.cat(codes_list, dim=-1)
            wav_list = []
            for c in range(codes_chunck.shape[-1]):
                wav = tokenizer.codec.decode(codes_chunck[..., c : (c + 1)])
                wav_list.append(wav)
            wav = torch.cat(wav_list, dim=-1)

            torchaudio.save(
                f'/home/wumenglin/workspace/devcontainer/temp/{path.split("/")[-1]}', wav.cpu().reshape(1, -1), 24000
            )
            tokenizer.codec.reset_streaming()
            tokenizer.codec.streaming_forever(1)
