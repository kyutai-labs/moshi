# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import queue
import time
import numpy as np
from pathlib import Path
import sentencepiece
import sounddevice as sd

import mlx.core as mx
import mlx.nn as nn

import mimi
import msh_mlx

SAMPLE_RATE = 24000
CHANNELS = 1


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--mimi", type=str)
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--steps", default=2500, type=int)
    args = parser.parse_args()

    model_file = args.model
    tokenizer_file = args.tokenizer
    mimi_file = args.mimi
    if model_file is None:
        model_file = str(Path.home() / "tmp/" / "mimi_0abbed5f@100.safetensors")
    if tokenizer_file is None:
        tokenizer_file = str(Path.home() / "tmp" / "tokenizer_spm_32k_3.model")
    if mimi_file is None:
        mimi_file = str(Path.home() / "tmp" / "tokenizer-de0e421d-checkpoint40.safetensors")
    steps = args.steps


    print(f"loading text tokenizer {tokenizer_file}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)
    mx.random.seed(299792458)

    lm_config = msh_mlx.models.config_v0_1()
    model = msh_mlx.models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if args.quantized:
        nn.quantize(model, bits=8)

    print(f"loading weights {model_file}")
    model.load_weights(model_file, strict=True)
    print("weights loaded")

    audio_tokenizer = mimi.StreamTokenizer(mimi_file)

    model.warmup()
    print("model warmed up")
    gen = msh_mlx.models.LmGen(
        model=model,
        max_steps=steps + 5,
        text_sampler=msh_mlx.utils.Sampler(),
        audio_sampler=msh_mlx.utils.Sampler(),
        check=False,
    )

    async def model_loop():
        while True:
            data = audio_tokenizer.get_encoded()
            if data is None:
                await asyncio.sleep(0.001)
                continue
            data = mx.array(data).transpose(1, 0)[:, :8]
            text_token = gen.step(data)
            text_token = text_token[0].item()
            audio_tokens = gen.last_audio_tokens()
            if text_token not in (0, 3):
                _text = text_tokenizer.id_to_piece(text_token)
                _text = _text.replace("▁", " ")
                print(_text, end='', flush=True)
            if audio_tokens is not None:
                audio_tokens = np.array(audio_tokens).astype(np.uint32)
                audio_tokenizer.decode(audio_tokens)

    def on_input(in_data, frames, time, status):
        in_data = in_data[0].astype(np.float32)
        audio_tokenizer.encode(in_data)

    in_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=1920, callback=on_input
    )

    def on_output(out_data, frames, time, status):
        assert out_data.shape == (1920, 1), out_data.shape
        try:
            pcm_data = audio_tokenizer.get_decoded()
            assert pcm_data.shape == (1920,), pcm_data.shape
            out_data[:, 0] = pcm_data
        except queue.Empty:
            print("SKIP AUDIO")
            out_data.fill(0)

    out_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=1920,
        callback=on_output,
    )

    print("starting the inference loop")
    with in_stream, out_stream:
        await model_loop()

if __name__ == "__main__":
    asyncio.run(main())

