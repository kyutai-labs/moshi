# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sentencepiece
import huggingface_hub
import mlx.core as mx
import mlx.nn as nn
from moshi_mlx import models, utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--hf-repo", type=str, default="kyutai/helium-1-preview-2b-mlx")
    parser.add_argument("--prompt", type=str, default="Aujourd'hui, il est temps")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quantize-bits", type=int)
    parser.add_argument("--save-quantized", type=str)
    parser.add_argument("--quantize-group-size", type=int, default=64)
    args = parser.parse_args()

    weights = args.weights
    if weights is None:
        weights = huggingface_hub.hf_hub_download(
            args.hf_repo, "helium-1-preview-2b-bf16.safetensors"
        )
    tokenizer = args.tokenizer
    if tokenizer is None:
        tokenizer = huggingface_hub.hf_hub_download(
            args.hf_repo, "tokenizer_spm_48k_multi6_2.model"
        )

    mx.random.seed(299792458)
    lm_config = models.config_helium_1_preview_2b()
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    model.load_weights(weights, strict=True)
    if args.quantize_bits is not None:
        nn.quantize(model, bits=args.quantize_bits, group_size=args.quantize_group_size)
        if args.save_quantized is not None:
            print(f"saving quantized weights in {args.save_quantized}")
            model.save_weights(args.save_quantized)
    sampler = utils.Sampler()
    tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)  # type: ignore
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
