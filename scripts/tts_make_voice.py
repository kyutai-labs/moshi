# Copyright (c) Kyutai, all rights reserved.
# Example:
#  uv run --script scripts/tts_make_voice.py \
#    --model-root ~/models/moshi/moshi_e9d43d50@500/ ~/models/tts-voices/myvoice.mp3+5.0
#
# It's also possible to pass in a directory containing audio files.
import argparse
import json
from pathlib import Path
import sys

import sphn
import torch
from moshi.models import loaders
from safetensors import safe_open
from safetensors.torch import save_file


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
        missing = length - wav.shape[-1]
        wav = torch.nn.functional.pad(wav[None], (0, missing))[0]
        wav = wav.mean(dim=0, keepdim=True)
        assert wav.shape[-1] == length
        emb = mimi.encode_to_latent(wav[None].to(args.device), quantize=False)
        tensors = {"speaker_wavs": emb.cpu()}
        save_file(tensors, out_file, metadata)

        print(f"Saved {out_file}.")


if __name__ == "__main__":
    main()
