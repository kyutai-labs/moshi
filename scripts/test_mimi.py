# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import time

import rustymimi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--steps", default=100, type=int)
    args = parser.parse_args()

    steps = args.steps
    model = rustymimi.Tokenizer(str(args.model))
    print(model)

    start_time = 0
    for i in range(steps + 1):
        if i == 1:
            start_time = time.time()
        pcm_data = np.array([[[0.0] * 1920]]).astype(np.float32)
        out = model.encode_step(pcm_data)
        print(out.shape)
        pcm_data = model.decode_step(out)
        print(pcm_data)
    token_per_second = steps / (time.time() - start_time)
    print(f"steps: {steps}, token per sec: {token_per_second}")


if __name__ == "__main__":
    main()
