# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import numpy as np
import time

from huggingface_hub import hf_hub_download
import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece
import sphn

from .client_utils import make_log
from . import models, utils


def log(level: str, msg: str):
    print(make_log(level, msg))


def hf_get(filename: str) -> str:
    if filename.startswith("hf://"):
        parts = filename[5:].split("/")
        repo_name = parts[0] + "/" + parts[1]
        filename = "/".join(parts[2:])
        log("info", f"retrieving {filename} from hf repo {repo_name}")
        return hf_hub_download(repo_name, filename)
    else:
        return filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi-weights", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weights", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default="kyutai/moshiko-mlx-q8")
    parser.add_argument("--lm-config", type=str, help="The LM config as a json file.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cfg-coef", type=float, default=1.)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("infile", type=str, help="Input audio file.")
    parser.add_argument("outfile", type=str, help="Output audio file in wav format.", nargs="?", default="")
    args = parser.parse_args()

    mx.random.seed(299792458)

    lm_config = args.lm_config
    if lm_config is None:
        lm_config = hf_hub_download(args.hf_repo, "config.json")

    log("info", f"loading config from {args.lm_config}")
    with open(hf_get(lm_config), "r") as fobj:
        lm_config = json.load(fobj)
    print(lm_config)
    stt_config = lm_config.get("stt_config", None)

    mimi_weights = args.mimi_weights
    if mimi_weights is None:
        mimi_weights = hf_hub_download(args.hf_repo, lm_config["mimi_name"])
    mimi_weights = hf_get(mimi_weights)

    moshi_weights = args.moshi_weights
    if moshi_weights is None:
        moshi_name = lm_config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_hub_download(args.hf_repo, moshi_name)
    moshi_weights = hf_get(moshi_weights)

    tokenizer = args.tokenizer
    if tokenizer is None:
        tokenizer = hf_hub_download(args.hf_repo, lm_config["tokenizer_name"])
    tokenizer = hf_get(tokenizer)

    lm_config = models.LmConfig.from_config_dict(lm_config)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if moshi_weights.endswith(".q4.safetensors"):
        nn.quantize(model, bits=4, group_size=32)
    elif moshi_weights.endswith(".q8.safetensors"):
        nn.quantize(model, bits=8, group_size=64)

    log("info", f"loading model weights from {moshi_weights}")
    model.load_weights(moshi_weights, strict=True)

    log("info", f"loading the text tokenizer from {tokenizer}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)  # type: ignore

    log("info", f"loading input file {args.infile}")
    in_pcms, _ = sphn.read(args.infile, sample_rate=24000)
    if stt_config is not None:
        pad_right = stt_config.get("audio_delay_seconds", 0.0)
        pad_left = stt_config.get("audio_silence_prefix_seconds", 0.0)
        pad_left = int(pad_left * 24000)
        pad_right = int((pad_right + 1.0) * 24000)
        in_pcms = np.pad(in_pcms, pad_width=[(0, 0), (pad_left, pad_right)], mode="constant")

    log("info", f"loading the audio tokenizer {mimi_weights}")
    generated_codebooks = lm_config.generated_codebooks
    other_codebooks = lm_config.other_codebooks
    mimi_codebooks = max(generated_codebooks, other_codebooks)
    audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)  # type: ignore

    if model.condition_provider is not None:
        ct = model.condition_provider.condition_tensor("description", "very_good")
    else:
        ct = None

    log("info", "warming up the model")
    model.warmup(ct)
    log("info", "done warming up the model")

    steps = np.shape(in_pcms)[-1] // 1920
    gen = models.LmGen(
        model=model,
        max_steps=steps,
        text_sampler=utils.Sampler(top_k=25, temp=args.temp),
        audio_sampler=utils.Sampler(top_k=250, temp=args.temp),
        cfg_coef=args.cfg_coef,
        check=False,
    )

    all_out_pcm = []
    start_time = time.time()
    log("info", f"steps to run: {steps}")
    for idx in range(0, steps):
        pcm_data = in_pcms[:, idx * 1920:(idx + 1) * 1920]
        other_audio_tokens = audio_tokenizer.encode_step(pcm_data[None, 0:1])
        other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[:, :, :other_codebooks]
        text_token = gen.step(other_audio_tokens[0], ct)
        text_token = text_token[0].item()
        audio_tokens = gen.last_audio_tokens()
        _text = None
        if text_token not in (0, 3):
            _text = text_tokenizer.id_to_piece(text_token)  # type: ignore
            _text = _text.replace("▁", " ")
            print(_text, end="", flush=True)
        if audio_tokens is not None and generated_codebooks > 0:
            audio_tokens = np.array(audio_tokens[:, :, None]).astype(np.uint32)
            out_pcm = audio_tokenizer.decode_step(audio_tokens)
            all_out_pcm.append(out_pcm)

    print()
    token_per_second = steps / (time.time() - start_time)
    log("info", f"steps: {steps}, token per sec: {token_per_second}")
    if args.outfile:
        all_out_pcm = np.concatenate(all_out_pcm, axis=-1)
        log("info", f"writing output file {args.outfile}")
        rustymimi.write_wav(args.outfile, all_out_pcm[0, 0], sample_rate=24000)


if __name__ == "__main__":
    main()
