# Copyright (c) Kyutai, all rights reserved.
# Example:
#  uv run --with=./moshi,julius,torchaudio scripts/tts_make_voice.py
#
# Ideally I would use the --script thing of uv, but I can't get it to work with the ./moshi...
#
# It's also possible to pass in a directory containing audio files.
import argparse
import json
import math
from pathlib import Path
import sys
import time

import julius
from safetensors import safe_open
from safetensors.torch import save_file
import sphn
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio.transforms

from moshi.models import loaders


def get_audio_files_in_directory(directory: Path):
    extensions = [".wav", ".mp3", ".ogg"]
    audio_files = []
    for ext in extensions:
        audio_files.extend(directory.glob(f"**/*{ext}"))
    return audio_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi."
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo to look into, defaults Moshiko. "
        "Use this to select a different pre-trained model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run, defaults to 'cuda'.",
    )
    parser.add_argument(
        "--config",
        "--lm-config",
        dest="config",
        type=Path,
        help="The config as a json file.",
    )
    parser.add_argument("--model-root", type=Path,
                        help="Shorthand for giving only once the root of the folder with the config and checkpoints.")

    parser.add_argument("--duration", type=float, default=10.0, help="Duration of the audio conditioning.")
    parser.add_argument("--clean", action="store_true",
                        help="Apply noise suppresion to cleanup the audio, along with volume normalization.")
    parser.add_argument("--save-clean", action="store_true",
                        help="Save the file once cleaned that was used to make the audio conditioning.")
    parser.add_argument("-o", "--out", type=Path, help="Out path if not same as original file.")
    parser.add_argument(
        "files",
        type=Path,
        help="Audio files to process. "
        "If a directory is given, runs on all audio files in this directory, "
        "including subdirectories",
        nargs="*",
    )

    args = parser.parse_args()
    if args.model_root is not None:
        candidates = list(args.model_root.glob('*_mimi_voice.safetensors'))
        assert len(candidates) == 1, candidates
        args.mimi_weight = candidates[0]
        args.config = args.model_root / 'config.json'

    print("retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, mimi_weights=args.mimi_weight,
    )
    # need a bit of manual param override at the moment.
    loaders._quantizer_kwargs["n_q"] = 16
    checkpoint_info.lm_config = None
    print("loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    print("mimi loaded")

    ext = ".safetensors"
    if args.config is None:
        print("A config must be provided to determine the model id.")
        sys.exit(1)
    raw_config = json.loads(args.config.read_text())
    try:
        model_id = raw_config['model_id']
    except KeyError:
        print("The provided config doesn't contain model_id, this is required.")
        sys.exit(1)
    ext = f".{model_id['sig']}@{model_id['epoch']}{ext}"

    files = []
    for file in args.files:
        if file.is_dir():
            files.extend(get_audio_files_in_directory(file))
        else:
            files.append(file)

    with safe_open(checkpoint_info.mimi_weights, framework="pt") as f:
        metadata = f.metadata()

    cleaner = None
    if args.clean:
        cleaner = Cleaner(sample_rate=mimi.sample_rate)
        cleaner.to(device=args.device)

    for file in files:
        out_folder = file.parent if args.out is None else args.out
        out_folder.mkdir(exist_ok=True, parents=True)
        out_file = out_folder / (file.name + ext)
        if out_file.exists():
            print(f"File {out_file} already exists, skipping.")
            continue

        seek = 0.0
        name = file.name
        if "+" in name:
            name, seek_str = name.rsplit("+", 1)
            seek = float(seek_str)
        audio_file = file.parent / name

        wav_np, _ = sphn.read(
            audio_file, seek, args.duration, sample_rate=mimi.sample_rate
        )
        length = int(mimi.sample_rate * args.duration)
        wav = torch.from_numpy(wav_np[:, :length]).float()
        wav = wav.mean(dim=0, keepdim=True)[None]

        if cleaner is not None:
            wav = cleaner(wav.to(device=args.device)).clamp(-0.99, 0.99)
            if args.save_clean:
                clean_file = out_folder / (file.name + ".clean.wav")
                sphn.write_wav(clean_file, wav.cpu().numpy()[0], cleaner.sample_rate)
                print(f"Saved {clean_file}.")

        missing = length - wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, missing))
        assert wav.shape[-1] == length
        emb = mimi.encode_to_latent(wav.to(args.device), quantize=False)
        tensors = {"speaker_wavs": emb.cpu()}
        save_file(tensors, out_file, metadata)

        print(f"Saved {out_file}.")


def normalize_loudness(wav: torch.Tensor, sample_rate: int, loudness_headroom_db: float = 22,
                       energy_floor: float = 2e-3):
    """Normalize an input signal to a user loudness in dB LKFS.
    Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.

    Args:
        wav (torch.Tensor): Input multichannel audio data.
        sample_rate (int): Sample rate.
        loudness_headroom_db (float): Target loudness of the output in dB LUFS.
        energy_floor (float): anything below that RMS level will not be rescaled.
    Returns:
        torch.Tensor: Loudness normalized output data.
    """
    wav = wav - wav.mean(dim=-1, keepdim=True)
    energy = wav.std()
    if energy < energy_floor:
        # Feeding audio lower than that will fail.
        return wav
    transform = torchaudio.transforms.Loudness(sample_rate)
    try:
        input_loudness_db = transform(wav).item()
    except RuntimeError:
        # audio is too short.
        return wav
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = -loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    output = gain * wav
    assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
    return output


def sinc(t: torch.Tensor) -> torch.Tensor:
    """sinc.

    :param t: the input tensor
    """
    return torch.where(t == 0, torch.ones(1, device=t.device, dtype=t.dtype), torch.sin(t) / t)


def kernel_upsample2(zeros=56, device=None):
    """kernel_upsample2.

    """
    win = torch.hann_window(4 * zeros + 1, periodic=False, device=device)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros, device=device)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def upsample2(x, zeros=56):
    """
    Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    *other, time = x.shape
    kernel = kernel_upsample2(zeros, x.device).to(x)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    y = torch.stack([x, out], dim=-1)
    return y.view(*other, -1)


def kernel_downsample2(zeros=56, device=None):
    """kernel_downsample2.

    """
    win = torch.hann_window(4 * zeros + 1, periodic=False, device=device)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros, device=device)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def downsample2(x, zeros=56):
    """
    Downsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = kernel_downsample2(zeros, x.device).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
        *other, time)
    return out.view(*other, -1).mul(0.5)


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class Demucs(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
        - sample_rate (float): sample_rate used for training the model.

    """
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3,
                 sample_rate=16_000):

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.sample_rate = sample_rate

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return std * x


def fast_conv(conv, x):
    """
    Faster convolution evaluation if either kernel size is 1
    or length of sequence is 1.
    """
    batch, chin, length = x.shape
    chout, chin, kernel = conv.weight.shape
    assert batch == 1
    if kernel == 1:
        x = x.view(chin, length)
        out = torch.addmm(
            conv.bias.view(-1, 1),
            conv.weight.view(chout, chin), x)
    elif length == kernel:
        x = x.view(chin * kernel, 1)
        out = torch.addmm(
            conv.bias.view(-1, 1),
            conv.weight.view(chout, chin * kernel), x)
    else:
        out = conv(x)
    return out.view(batch, chout, -1)


class DemucsStreamer:
    """
    Streaming implementation for Demucs. It supports being fed with any amount
    of audio at a time. You will get back as much audio as possible at that
    point.

    Args:
        - demucs (Demucs): Demucs model.
        - dry (float): amount of dry (e.g. input) signal to keep. 0 is maximum
            noise removal, 1 just returns the input signal. Small values > 0
            allows to limit distortions.
        - num_frames (int): number of frames to process at once. Higher values
            will increase overall latency but improve the real time factor.
        - resample_lookahead (int): extra lookahead used for the resampling.
        - resample_buffer (int): size of the buffer of previous inputs/outputs
            kept for resampling.
    """
    def __init__(self, demucs,
                 dry=0,
                 num_frames=1,
                 resample_lookahead=64,
                 resample_buffer=256,
                 mean_decay_duration: float = 10.):
        device = next(iter(demucs.parameters())).device
        self.demucs = demucs
        self.lstm_state = None
        self.conv_state = None
        self.dry = dry
        self.resample_lookahead = resample_lookahead
        resample_buffer = min(demucs.total_stride, resample_buffer)
        self.resample_buffer = resample_buffer
        self.frame_length = demucs.valid_length(1) + demucs.total_stride * (num_frames - 1)
        self.total_length = self.frame_length + self.resample_lookahead
        self.stride = demucs.total_stride * num_frames
        self.resample_in = torch.zeros(demucs.chin, resample_buffer, device=device)
        self.resample_out = torch.zeros(demucs.chin, resample_buffer, device=device)

        self.frames = 0
        self.total_time = 0
        self.mean_variance = 0.
        self.mean_total = 0.
        mean_receptive_field_in_samples = mean_decay_duration * demucs.sample_rate
        mean_receptive_field_in_frames = mean_receptive_field_in_samples / demucs.total_stride
        self.mean_decay = 1 - 1 / mean_receptive_field_in_frames

        self.pending = torch.zeros(demucs.chin, 0, device=device)

        bias = demucs.decoder[0][2].bias
        weight = demucs.decoder[0][2].weight
        chin, chout, kernel = weight.shape
        self._bias = bias.view(-1, 1).repeat(1, kernel).view(-1, 1)
        self._weight = weight.permute(1, 2, 0).contiguous()

    @property
    def variance(self) -> float:
        return self.mean_variance / self.mean_total

    def reset_time_per_frame(self):
        self.total_time = 0
        self.frames = 0

    @property
    def time_per_frame(self):
        return self.total_time / self.frames

    def flush(self):
        """
        Flush remaining audio by padding it with zero and initialize the previous
        status. Call this when you have no more input and want to get back the last
        chunk of audio.
        """
        self.lstm_state = None
        self.conv_state = None
        pending_length = self.pending.shape[1]
        padding = torch.zeros(self.demucs.chin, self.total_length, device=self.pending.device)
        out = self.feed(padding)
        return out[:, :pending_length]

    def feed(self, wav):
        """
        Apply the model to mix using true real time evaluation.
        Normalization is done online as is the resampling.
        """
        begin = time.time()
        demucs = self.demucs
        resample_buffer = self.resample_buffer
        stride = self.stride
        resample = demucs.resample

        if wav.dim() != 2:
            raise ValueError("input wav should be two dimensional.")
        chin, _ = wav.shape
        if chin != demucs.chin:
            raise ValueError(f"Expected {demucs.chin} channels, got {chin}")

        self.pending = torch.cat([self.pending, wav], dim=1)
        outs = []
        while self.pending.shape[1] >= self.total_length:
            self.frames += 1
            frame = self.pending[:, :self.total_length]
            dry_signal = frame[:, :stride]
            if demucs.normalize:
                mono = frame.mean(0)
                variance = (mono**2).mean()
                self.mean_variance = self.mean_variance * self.mean_decay + (1 - self.mean_decay) * variance
                self.mean_total = self.mean_total * self.mean_decay + (1 - self.mean_decay)
                frame = frame / (demucs.floor + torch.sqrt(self.variance))  # type: ignore
            padded_frame = torch.cat([self.resample_in, frame], dim=-1)
            self.resample_in[:] = frame[:, stride - resample_buffer:stride]
            frame = padded_frame

            if resample == 4:
                frame = upsample2(upsample2(frame))
            elif resample == 2:
                frame = upsample2(frame)
            frame = frame[:, resample * resample_buffer:]  # remove pre sampling buffer
            frame = frame[:, :resample * self.frame_length]  # remove extra samples after window

            out, extra = self._separate_frame(frame)
            padded_out = torch.cat([self.resample_out, out, extra], 1)
            self.resample_out[:] = out[:, -resample_buffer:]
            if resample == 4:
                out = downsample2(downsample2(padded_out))
            elif resample == 2:
                out = downsample2(padded_out)
            else:
                out = padded_out

            out = out[:, resample_buffer // resample:]
            out = out[:, :stride]

            if demucs.normalize:
                out *= torch.sqrt(self.variance)  # type: ignore
            out = self.dry * dry_signal + (1 - self.dry) * out
            outs.append(out)
            self.pending = self.pending[:, stride:]

        self.total_time += time.time() - begin
        if outs:
            out = torch.cat(outs, 1)
        else:
            out = torch.zeros(chin, 0, device=wav.device)
        return out

    def _separate_frame(self, frame):
        assert self.conv_state is not None
        demucs = self.demucs
        skips = []
        next_state = []
        first = self.conv_state is None
        stride = self.stride * demucs.resample
        x = frame[None]
        for idx, encode in enumerate(demucs.encoder):
            stride //= demucs.stride
            length = x.shape[2]
            if idx == demucs.depth - 1:
                # This is sligthly faster for the last conv
                x = fast_conv(encode[0], x)
                x = encode[1](x)
                x = fast_conv(encode[2], x)
                x = encode[3](x)
            else:
                prev = None
                if not first:
                    prev = self.conv_state.pop(0)
                    prev = prev[..., stride:]
                    tgt = (length - demucs.kernel_size) // demucs.stride + 1
                    missing = tgt - prev.shape[-1]
                    offset = length - demucs.kernel_size - demucs.stride * (missing - 1)
                    x = x[..., offset:]
                x = encode[1](encode[0](x))
                x = fast_conv(encode[2], x)
                x = encode[3](x)
                if prev is not None:
                    x = torch.cat([prev, x], -1)
                next_state.append(x)
            skips.append(x)

        x = x.permute(2, 0, 1)
        x, self.lstm_state = demucs.lstm(x, self.lstm_state)
        x = x.permute(1, 2, 0)
        # In the following, x contains only correct samples, i.e. the one
        # for which each time position is covered by two window of the upper layer.
        # extra contains extra samples to the right, and is used only as a
        # better padding for the online resampling.
        extra = None
        for idx, decode in enumerate(demucs.decoder):
            skip = skips.pop(-1)
            x += skip[..., :x.shape[-1]]
            x = fast_conv(decode[0], x)
            x = decode[1](x)

            if extra is not None:
                skip = skip[..., x.shape[-1]:]
                extra += skip[..., :extra.shape[-1]]
                extra = decode[2](decode[1](decode[0](extra)))
            x = decode[2](x)
            next_state.append(x[..., -demucs.stride:] - decode[2].bias.view(-1, 1))
            if extra is None:
                extra = x[..., -demucs.stride:]
            else:
                extra[..., :demucs.stride] += next_state[-1]
            x = x[..., :-demucs.stride]

            if not first:
                prev = self.conv_state.pop(0)
                x[..., :demucs.stride] += prev
            if idx != demucs.depth - 1:
                x = decode[3](x)
                extra = decode[3](extra)
        self.conv_state = next_state
        assert extra is not None
        return x[0], extra[0]


def get_demucs():
    model = Demucs(hidden=64)
    url = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns64-a7761ff99a7d5bb6.th"
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


class Cleaner(nn.Module):
    def __init__(self, dry_fraction: float = 0.02, sample_rate: int = 24000):
        super().__init__()
        self.dry_fraction = dry_fraction
        self.sample_rate = sample_rate
        self._demucs = get_demucs()
        demucs_sr = self._demucs.sample_rate
        cutoff = demucs_sr / sample_rate / 2
        self._lowpass = julius.lowpass.LowPassFilter(cutoff)
        self._downsample = julius.resample.ResampleFrac(sample_rate, demucs_sr)
        self._upsample = julius.resample.ResampleFrac(demucs_sr, sample_rate)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor):
        assert wav.dim() == 3, "Must be [B, C, T]"
        low = self._lowpass(wav)
        high = wav - low
        low = self._downsample(low, full=True)

        denoised = self._demucs(low)
        denoised = (1 - self.dry_fraction) * denoised + self.dry_fraction * low
        denoised = self._upsample(denoised, output_length=wav.shape[-1])
        denoised = denoised + high

        denoised = normalize_loudness(denoised, self.sample_rate)
        return denoised


if __name__ == "__main__":
    main()
