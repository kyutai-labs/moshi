# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import mlx.core as mx
from moshi_mlx import models, utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--hf-repo", type=str)
    args = parser.parse_args()

    lm_config = models.config_helium_1_preview_2b()
    model = models.Lm(lm_config)
    sampler = utils.Sampler()
    token = mx.array([[1]])
    for step_idx in range(args.num_steps):
        logits = model(token)
        token, _ = sampler(logits[:, 0])
        print(step_idx, token)
        token = token[None]


if __name__ == "__main__":
    main()
