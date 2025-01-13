# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sentencepiece
import mlx.core as mx
from moshi_mlx import models, utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--hf-repo", type=str)
    parser.add_argument("--prompt", type=str, default="Hello, this is ")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mx.random.seed(299792458)
    lm_config = models.config_helium_1_preview_2b()
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    model.load_weights(args.weights, strict=True)
    sampler = utils.Sampler()
    tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore
    if args.verbose:
        print("prompt", args.prompt)
    else:
        print(args.prompt, end="", flush=True)
    prompt_tokens = tokenizer.encode(args.prompt)  # type: ignore
    token = mx.array([[1] + prompt_tokens])
    for step_idx in range(args.nsteps):
        logits = model(token)
        token, _ = sampler(logits[:, -1])
        text_token = token.item()
        _text = tokenizer.id_to_piece(text_token)  # type: ignore
        _text = _text.replace("▁", " ")
        _text = _text.replace("<0x0A>", "\n")
        if args.verbose:
            print(step_idx, token, _text)
        else:
            print(_text, end="", flush=True)
        token = token[None]
    print()


if __name__ == "__main__":
    main()
