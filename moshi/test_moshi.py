import logging
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Union

import librosa
import torch
from lhotse import Recording
from lhotse.utils import Seconds
from torchaudio.transforms import Resample

from moshi.models import loaders

# os.environ['NO_TORCH_COMPILE'] = '1'

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

wav_path = "/home/wumenglin/workspace/devcontainer/temp/data/fe_03_00001.wav"
model_path = "/data0/questar/models/hf/moshika-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors"
device = "cuda:0"

recording = Recording.from_file(wav_path)
audio = recording.load_audio([0])

if recording.sampling_rate != 24000:
    resampler = Resample(orig_freq=recording.sampling_rate, new_freq=24000)
    audio = resampler(torch.tensor(audio))

# audio = audio.reshape(-1, )

audio_new_1 = audio.clone()
audio_new_2 = audio.clone()
audio_new_3 = audio.clone()

mimi = loaders.get_mimi(filename=model_path, device="cuda:0").to(torch.float32)
mimi.streaming_forever(1)


def warmup():
    for chunk in range(4):
        chunk = torch.zeros(1, 1, int(8 * 60 * 24000), dtype=torch.float32, device=device)
        _ = mimi.encode(chunk)
    torch.cuda.synchronize()


overlap_second = 200
wav_chunk_length = 24 * 24000
test_chunk_length = 72 * 24000
frame_wav_offset = int(24000 / 12.5)
overlap_length = overlap_second * 24000
overlap_codes_length = int(12.5 * overlap_second)

rf_tokens = 107
recieve_field = rf_tokens * 1920  # 4280和1920的最小公倍数是107 * 1920

with torch.no_grad():
    codes_list = []
    for i in range(0, test_chunk_length, frame_wav_offset):
        wav_chunk = audio_new_1[..., i : i + frame_wav_offset]
        codes = mimi.encode_to_latent(wav_chunk.to("cuda:0").unsqueeze(0), quantize=True)
        codes_list.append(codes)
    codes_75_list = torch.cat(codes_list, dim=-1)

    mimi.reset_streaming()
    mimi.streaming_forever(1)
    codes_non_overlap_list2 = []
    for i in range(0, test_chunk_length, wav_chunk_length):
        wav_chunk = audio_new_1[..., i : i + wav_chunk_length]
        codes = mimi.encode_to_latent(wav_chunk.to("cuda:0").unsqueeze(0), quantize=True)
        codes_non_overlap_list2.append(codes)
    codes_non_overlap2 = torch.cat(codes_non_overlap_list2, dim=-1)

    import pdb

    pdb.set_trace()

    mimi.reset_streaming()
    mimi.streaming_forever(1)
    codes_non_overlap_list = []
    for i in range(0, test_chunk_length, wav_chunk_length * 2):
        wav_chunk = audio_new_1[..., i : i + wav_chunk_length * 2]
        codes = mimi.encode_to_latent(wav_chunk.to("cuda:0").unsqueeze(0), quantize=False)
        codes_non_overlap_list.append(codes)
    codes_non_overlap = torch.cat(codes_non_overlap_list, dim=-1)

    start = time()
    codes_list = []
    for i in range(0, audio.shape[-1], wav_chunk_length):
        if i == 0:
            wav_chunk = audio[..., i : i + wav_chunk_length]
            codes = mimi.encode(wav_chunk.to("cuda:0").unsqueeze(0))
            codes_list.append(codes)
        else:
            wav_chunk = audio[..., (i - overlap_length) : i + wav_chunk_length]
            codes = mimi.encode(wav_chunk.to("cuda:0").unsqueeze(0))
            codes_list.append(codes[..., overlap_codes_length:])
    codes_overlap = torch.cat(codes_list, dim=-1)
    time_consume = time() - start

    codes_rf_list = []
    for i in range(0, audio_new_3.shape[-1], wav_chunk_length):

        if i == 0:
            wav_chunk = audio_new_3[..., i : i + wav_chunk_length]
            codes = mimi.encode(wav_chunk.to("cuda:0").unsqueeze(0))
            codes_rf_list.append(codes)
        else:
            wav_chunk = audio_new_3[..., (i - recieve_field) : i + wav_chunk_length]
            codes = mimi.encode(wav_chunk.to("cuda:0").unsqueeze(0))
            codes_rf_list.append(codes[..., rf_tokens:])
    codes_overlap_rf = torch.cat(codes_rf_list, dim=-1)

    codes_all = mimi.encode(audio_new_2.to("cuda:0").unsqueeze(0))

    print(codes)


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
        self.codec.streaming_forever(1)

        self.frame_shift = frame_shift
        self.num_quantizers = num_quantizers
        self.sample_rate = sampling_rate

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

        codes = self.codec.encode(wav)

        if isinstance(codes, torch.Tensor) and torch.is_floating_point(codes):
            return codes

        # codes = codes.type(torch.int16).permute(0, 2, 1)  # BxTxQ
        return codes[:, : self.num_quantizers, :]


if __name__ == "__main__":
    wav_path = ["/data0/questar/users/wumenglin/temp/fe_03_00001.wav"]
    device = "cuda:0"
    audio_chunk_length = 4 * 60 * 24000
    tokenizer = MoshiMimi(model_path, device=device, sampling_rate=24000, num_quantizers=8)

    pdb.set_trace()
    with torch.no_grad():
        for path in wav_path:
            audio, _ = librosa.load(path, sr=24000, mono=True)
            audio_new = deepcopy(audio)
            codes_list = []
            j = 0
            for i in range(0, audio.shape[-1], audio_chunk_length):
                print(j, i, audio.shape[-1])
                audio_chunk = audio[..., i : i + audio_chunk_length]
                codes = tokenizer.encode(torch.tensor(audio_chunk).to(device))
                codes_list.append(codes)
                j += 1
            codes_chunck = torch.cat(codes_list, dim=-1)
            tokenizer.codec.reset_streaming()
            tokenizer.codec.streaming_forever(1)
            codes_all = tokenizer.encode(torch.tensor(audio_new).to(device))
