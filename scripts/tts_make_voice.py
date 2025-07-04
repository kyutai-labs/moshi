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
import sys
import time
from pathlib import Path

import julius
import sphn
import torch
import torchaudio.transforms
from moshi.models.demucs import get_demucs
from moshi.models import loaders
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn


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
    parser.add_argument("--loudness-headroom",type=float,
                        help="Normalize the loudness of the audio conditioning to this value -dBFS. 22 is a good value. "
                        "Lower values mean louder audio but potentially some clipping.")
    parser.add_argument("--clean", action="store_true",
                        help="Apply noise suppresion to clean up the audio.")
    parser.add_argument("--save-clean", action="store_true",
                        help="Save the file that was used to make the audio conditioning, after cleaning and loudness normalization.")
    parser.add_argument("-o", "--out", type=Path, help="Out path if not same as original file.")
    parser.add_argument(
        "files",
        type=Path,
        help="Audio files to process. "
        "If a directory is given, runs on all audio files in this directory, "
        "including subdirectories",
        nargs="+",
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

    if not files:
        print(f"No audio files found in {args.files}.")
        sys.exit(1)

    with safe_open(checkpoint_info.mimi_weights, framework="pt") as f:
        metadata = f.metadata()

    cleaner = None
    if args.clean:
        cleaner = Cleaner(sample_rate=mimi.sample_rate)
        cleaner.to(device=args.device)

    n_new = 0

    for file in files:
        out_folder = file.parent if args.out is None else args.out
        out_folder.mkdir(exist_ok=True, parents=True)
        out_file = out_folder / (file.name + ext)
        if out_file.exists():
            print(f"File {out_file} already exists, skipping.")
            continue

        print(f"Creating {out_file}")
        n_new += 1

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

        if args.loudness_headroom is not None:
            wav = normalize_loudness(wav, sample_rate=mimi.sample_rate, loudness_headroom_db=args.loudness_headroom)

        if args.save_clean:
            clean_file = out_folder / (file.name + ".clean.wav")
            sphn.write_wav(clean_file, wav.cpu().numpy()[0], cleaner.sample_rate)
            print(f"Saved clean file {clean_file}")

        missing = length - wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, missing))
        assert wav.shape[-1] == length
        emb = mimi.encode_to_latent(wav.to(args.device), quantize=False)
        tensors = {"speaker_wavs": emb.cpu()}
        save_file(tensors, out_file, metadata)
    
    print(f"Created voice embeddings for {n_new} files in {out_folder}, {args.clean=} {args.loudness_headroom=}")


def normalize_loudness(wav: torch.Tensor, sample_rate: int, loudness_headroom_db: float = 18,
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
    if loudness_headroom_db < 0:
        raise ValueError("loudness_headroom_db must be non-negative.")

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

        return denoised


if __name__ == "__main__":
    main()
