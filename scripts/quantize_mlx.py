# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import mlx.core as mx
import mlx.nn as nn

import moshi_mlx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original_weights", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--config", type=str, default="v0_1")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    model_file = args.original_weights

    if args.config == "v0_1":
        lm_config = moshi_mlx.models.config_v0_1()
    elif args.config == "1b":
        lm_config = moshi_mlx.models.config1b_202412()
    elif args.config == "1b-16rvq":
        lm_config = moshi_mlx.models.config1b_202412_16rvq()
    elif args.config == "helium-2b":
        lm_config = moshi_mlx.models.config_helium_1_preview_2b()
    else:
        raise ValueError(f"unknown config name '{args.config}'")
    print(f"model config:\n{lm_config}")

    model = moshi_mlx.models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    print(f"loading weights {model_file}")
    model.load_weights(model_file, strict=True)
    print("weights loaded")

    nn.quantize(model, bits=args.bits, group_size=args.group_size)
    print(f"saving the quantized q{args.bits} weights in {args.out}")
    model.save_weights(args.out)


if __name__ == "__main__":
    main()
