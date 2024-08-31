# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
from pathlib import Path
import sentencepiece

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

import msh_mlx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--steps", default=100, type=int)
    args = parser.parse_args()
    
    model_file = args.model
    tokenizer_file = args.tokenizer
    if model_file is None:
        model_file = str(Path.home() / "tmp/" / "mimi_0abbed5f@100.safetensors")
    if tokenizer_file is None:
        tokenizer_file = str(Path.home() / "tmp" / "tokenizer_spm_32k_3.model")

    
    print(f"loading text tokenizer {tokenizer_file}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)
    mx.random.seed(299792458)
    
    lm_config = msh_mlx.models.config_v0_1()
    if args.verbose:
        print(f"model config:\n{lm_config}")
    
    model = msh_mlx.models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if args.quantized:
        nn.quantize(model, bits=8)

    if args.verbose:
        tree_map_with_path(lambda p, t: print(p, t.shape), model.parameters())

    print(f"loading weights {model_file}")
    model.load_weights(model_file, strict=True)
    print("weights loaded")
    
    cache = None
    start_time = 0
    last_text_token = mx.array([[32000]])
    last_audio_tokens = []
    text_sampler = msh_mlx.utils.Sampler()
    audio_sampler = msh_mlx.utils.Sampler()
    for i in range(args.steps + 1):
        if i == 1:
            start_time = time.time()
        last_text_token, last_audio_tokens, cache = model.sample(
            last_text_token,
            last_audio_tokens,
            text_sampler,
            audio_sampler,
            cache,
        )
        text_token = last_text_token[0].item()
        if text_token not in (0, 3):
            _text = text_tokenizer.id_to_piece(text_token)
            _text = _text.replace("▁", " ")
            print(_text, end='', flush=True)
    
        last_text_token = last_text_token[None]
        if args.text_only:
            last_audio_tokens = []
        else:
            last_audio_tokens = [l for l in last_audio_tokens]
    print()
    token_per_second = args.steps / (time.time() - start_time)
    print(f"steps: {args.steps}, token per sec: {token_per_second}")

if __name__ == "__main__":
    main()
