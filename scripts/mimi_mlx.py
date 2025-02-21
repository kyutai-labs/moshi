# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from huggingface_hub import hf_hub_download
import mlx.core as mx
import sphn
import moshi_mlx


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--hf-repo", type=str, default="kyutai/moshiko-mlx-q4")
    args = parser.parse_args()

    pcm_in, _ = sphn.read(args.input, sample_rate=24000)
    pcm_in = mx.array(pcm_in[0])
    print(pcm_in.shape)

    weight_file = hf_hub_download(args.hf_repo, "tokenizer-e351c8d8-checkpoint125.safetensors")
    cfg = moshi_mlx.models.mimi.mimi_202407(16)
    mimi = moshi_mlx.models.mimi.Mimi(cfg)
    mimi.encode(pcm_in)

if __name__ == "__main__":
    run()
