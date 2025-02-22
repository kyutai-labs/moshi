# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from huggingface_hub import hf_hub_download
import numpy as np
import mlx.core as mx
import sphn
import moshi_mlx


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--model-file", type=str)
    parser.add_argument("--hf-repo", type=str, default="kyutai/moshiko-mlx-q4")
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    pcm_in, _ = sphn.read(args.input, sample_rate=24000)
    pcm_in = mx.array(pcm_in[0])[None, None]
    print(pcm_in.shape)

    if args.model_file is None:
        model_file = hf_hub_download(args.hf_repo, "tokenizer-e351c8d8-checkpoint125.safetensors")
    else:
        model_file = args.model_file
    cfg = moshi_mlx.models.mimi.mimi_202407(32)
    print("building model", flush=True)
    model = moshi_mlx.models.mimi.Mimi(cfg)
    print(f"loading weights {model_file}", flush=True)
    model.load_pytorch_weights(model_file, strict=True)
    print("weights loaded")

    if args.streaming:
        chunk_size = 1920
        pcm_out = []
        len_ = pcm_in.shape[-1]
        print("starting streaming conversion")
        for start_idx in range(0, len_, chunk_size):
            end_idx = start_idx + chunk_size
            if end_idx >= len_:
                break
            _pcm_in = pcm_in[..., start_idx:end_idx]
            codes = model.encode_step(_pcm_in)
            _pcm_out = model.decode_step(codes)
            pcm_out.append(_pcm_out)
            pct = int(100 * start_idx / len_)
            print(f"{pct}%", end="\r", flush=True)
        print()
        pcm_out = mx.concat(pcm_out, axis=-1)
    else:
        codes = model.encode(pcm_in)
        print(codes.shape)
        pcm_out = model.decode(codes)
    print("writing output file with audio shape", pcm_out.shape)
    sphn.write_wav("out.wav", np.array(pcm_out[0]), sample_rate=24000)

if __name__ == "__main__":
    run()
