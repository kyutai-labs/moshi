# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import time
import numpy as np
from pathlib import Path
import sentencepiece

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

import msh_mlx

def run_audio_gen(model: msh_mlx.models.Lm, mimi_path: str, text_tokenizer, steps: int):
    import mimi

    audio_tokenizer = mimi.Tokenizer(mimi_path)

    model.warmup()
    gen = msh_mlx.models.LmGen(
        model=model,
        max_steps=steps + 5,
        text_sampler=msh_mlx.utils.Sampler(),
        audio_sampler=msh_mlx.utils.Sampler(),
        check=False,
    )
    pcm_data = np.array([[[0.] * 1920]]).astype(np.float32)
    all_out_pcm = []
    start_time = time.time()
    for _ in range(steps + 1):
        other_audio_tokens = audio_tokenizer.encode_step(pcm_data)
        other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[:, :, :8]
        text_token = gen.step(other_audio_tokens)
        text_token = text_token[0].item()
        audio_tokens = gen.last_audio_tokens()
        _text = None
        if text_token not in (0, 3):
            _text = text_tokenizer.id_to_piece(text_token)
            _text = _text.replace("▁", " ")
            print(_text, end='', flush=True)
        if audio_tokens is not None:
            audio_tokens = np.array(audio_tokens[:, :, None]).astype(np.uint32)
            out_pcm = audio_tokenizer.decode_step(audio_tokens)
            all_out_pcm.append(out_pcm)

    print()
    token_per_second = steps / (time.time() - start_time)
    print(f"steps: {steps}, token per sec: {token_per_second}")
    all_out_pcm = np.concatenate(all_out_pcm, axis=-1)
    mimi.write_wav("out.wav", all_out_pcm[0, 0], sample_rate=24000)

async def run_audio_gen_stream(model: msh_mlx.models.Lm, mimi_path: str, text_tokenizer, steps: int):
    import mimi

    audio_tokenizer = mimi.StreamTokenizer(mimi_path)

    gen = msh_mlx.models.LmGen(
        model=model,
        max_steps=steps + 5,
        text_sampler=msh_mlx.utils.Sampler(),
        audio_sampler=msh_mlx.utils.Sampler(),
        check=False,
    )
    gen.model
    pcm_data = np.array([[[0.] * 1920]]).astype(np.float32)
    all_out_pcm = []
    start_time = time.time()
    for _ in range(steps + 1):
        other_audio_tokens = audio_tokenizer.encode_step(pcm_data)
        other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[:, :, :8]
        text_token = gen.step(other_audio_tokens)
        text_token = text_token[0].item()
        audio_tokens = gen.last_audio_tokens()
        _text = None
        if text_token not in (0, 3):
            _text = text_tokenizer.id_to_piece(text_token)
            _text = _text.replace("▁", " ")
            print(_text, end='', flush=True)
        if audio_tokens is not None:
            audio_tokens = np.array(audio_tokens[:, :, None]).astype(np.uint32)
            out_pcm = audio_tokenizer.decode_step(audio_tokens)
            all_out_pcm.append(out_pcm)

    print()
    token_per_second = steps / (time.time() - start_time)
    print(f"steps: {steps}, token per sec: {token_per_second}")
    all_out_pcm = np.concatenate(all_out_pcm, axis=-1)
    mimi.write_wav("out.wav", all_out_pcm[0, 0], sample_rate=24000)

def run_text_gen(model: msh_mlx.models.Lm, text_tokenizer, steps: int):
    cache = None
    start_time = 0
    last_text_token = mx.array([[32000]])
    text_sampler = msh_mlx.utils.Sampler()
    audio_sampler = msh_mlx.utils.Sampler()
    for i in range(steps + 1):
        if i == 1:
            start_time = time.time()
        last_text_token, _, cache = model.sample(
            last_text_token,
            [],
            i,
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
    print()
    token_per_second = steps / (time.time() - start_time)
    print(f"steps: {steps}, token per sec: {token_per_second}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--mimi", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("mode", default="text", type=str)
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

    if args.mode == "text":
        run_text_gen(model, text_tokenizer, args.steps)
    elif args.mode == "audio":
        run_audio_gen(model, args.mimi, text_tokenizer, args.steps)
    elif args.mode == "stream":
        asyncio.run(run_audio_gen_stream(model, args.mimi, text_tokenizer, args.steps))
    else:
        raise ValueError(f"unknown mode {args.mode}, try 'text' or 'audio'")

if __name__ == "__main__":
    main()
