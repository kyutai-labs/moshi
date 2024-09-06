# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import mlx.core as mx
import mlx.nn as nn

import msh_mlx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original_weights", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--bits", type=int, default=8)
    args = parser.parse_args()

    model_file = args.original_weights

    lm_config = msh_mlx.models.config_v0_1()
    print(f"model config:\n{lm_config}")

    model = msh_mlx.models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    print(f"loading weights {model_file}")
    model.load_weights(model_file, strict=True)
    print("weights loaded")

    nn.quantize(model, bits=args.bits)
    print(f"saving the quantized q{args.bits} weights in {args.out}")
    model.save_weights(args.out)


if __name__ == "__main__":
    main()
