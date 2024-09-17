# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import queue
import json
import time
import numpy as np
from pathlib import Path
import sentencepiece
import typing as tp
import os
import shutil

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

import moshi_mlx


class Stats:
    send_times: tp.List[float] = []
    model_times: tp.List[tp.Tuple[float, float]] = []
    recv_times: tp.List[float] = []

    def __init__(self):
        self.send_times = []
        self.model_times = []
        self.recv_times = []

    def on_send(self, t: float):
        self.send_times.append(t)

    def on_model(self, t1: float, t2: float):
        self.model_times.append((t1, t2))

    def on_recv(self, t: float):
        self.recv_times.append(t)


def run_audio_gen(
    model: moshi_mlx.models.Lm, mimi_path: str, text_tokenizer, steps: int
):
    import rustymimi

    audio_tokenizer = rustymimi.Tokenizer(mimi_path)

    model.warmup()
    gen = moshi_mlx.models.LmGen(
        model=model,
        max_steps=steps + 5,
        text_sampler=moshi_mlx.utils.Sampler(),
        audio_sampler=moshi_mlx.utils.Sampler(),
        check=False,
    )
    pcm_data = np.array([[[0.0] * 1920]]).astype(np.float32)
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
            print(_text, end="", flush=True)
        if audio_tokens is not None:
            audio_tokens = np.array(audio_tokens[:, :, None]).astype(np.uint32)
            out_pcm = audio_tokenizer.decode_step(audio_tokens)
            all_out_pcm.append(out_pcm)

    print()
    token_per_second = steps / (time.time() - start_time)
    print(f"steps: {steps}, token per sec: {token_per_second}")
    all_out_pcm = np.concatenate(all_out_pcm, axis=-1)
    rustymimi.write_wav("out.wav", all_out_pcm[0, 0], sample_rate=24000)


async def run_audio_gen_stream(
    model: moshi_mlx.models.Lm, mimi_path: str, text_tokenizer, steps: int
):
    import rustymimi

    audio_tokenizer = rustymimi.StreamTokenizer(mimi_path)
    stats = Stats()

    model.warmup()
    gen = moshi_mlx.models.LmGen(
        model=model,
        max_steps=steps + 5,
        text_sampler=moshi_mlx.utils.Sampler(),
        audio_sampler=moshi_mlx.utils.Sampler(),
        check=False,
    )
    end_queue = queue.Queue()

    async def send_loop():
        pcm_data = np.array([0.0] * 1920).astype(np.float32)
        for _ in range(steps):
            await asyncio.sleep(1.0 / 13.0)
            stats.on_send(time.time())
            audio_tokenizer.encode(pcm_data)
        await asyncio.sleep(1.0)
        end_queue.put_nowait(True)

    async def model_loop():
        while True:
            data = audio_tokenizer.get_encoded()
            if data is None:
                await asyncio.sleep(0.001)
                try:
                    end_queue.get(block=False)
                    break
                except queue.Empty:
                    pass
                continue
            start_time = time.time()
            data = mx.array(data).transpose(1, 0)[:, :8]
            text_token = gen.step(data)
            text_token = text_token[0].item()
            audio_tokens = gen.last_audio_tokens()
            if text_token not in (0, 3):
                _text = text_tokenizer.id_to_piece(text_token)
                _text = _text.replace("▁", " ")
                print(_text, end="", flush=True)
            if audio_tokens is not None:
                audio_tokens = np.array(audio_tokens).astype(np.uint32)
                audio_tokenizer.decode(audio_tokens)
            stats.on_model(start_time, time.time())

    async def recv_loop():
        all_out_pcm = []
        start_time = time.time()
        while len(all_out_pcm) < steps - 1:
            data = audio_tokenizer.get_decoded()
            if data is None:
                await asyncio.sleep(0.001)
                continue
            stats.on_recv(time.time())
            all_out_pcm.append(data)
        print()
        token_per_second = steps / (time.time() - start_time)
        print(f"steps: {steps}, token per sec: {token_per_second}")
        all_out_pcm = np.concatenate(all_out_pcm, axis=-1)
        rustymimi.write_wav("out.wav", all_out_pcm, sample_rate=24000)

    await asyncio.gather(recv_loop(), send_loop(), model_loop())
    # Discard the first times for sending and processing the first slice as this does not
    # result in an output.
    stats = {
        "send_times": stats.send_times[1:],
        "recv_times": stats.recv_times,
        "model_times": stats.model_times[1:],
    }
    with open("timings.json", "w") as json_file:
        json.dump(stats, json_file)
    model_times = np.array(stats["model_times"])
    model_times = model_times[:, 1] - model_times[:, 0]
    mean, stdev, min_v, max_v = (
        np.mean(model_times),
        np.std(model_times),
        np.min(model_times),
        np.max(model_times),
    )
    print(
        f"model times, mean: {1000 * mean:.2f}ms, std: {1000 * stdev:.2f}ms, min: {1000 * min_v:.2f}ms, max: {1000 * max_v:.2f}ms"
    )


def run_text_gen(model: moshi_mlx.models.Lm, text_tokenizer, steps: int):
    start_time = 0
    last_text_token = mx.array([[32000]])
    text_sampler = moshi_mlx.utils.Sampler()
    audio_sampler = moshi_mlx.utils.Sampler()
    for i in range(steps + 1):
        if i == 1:
            start_time = time.time()
        last_text_token, _ = model.sample(
            last_text_token,
            [],
            i,
            text_sampler,
            audio_sampler,
        )
        text_token = last_text_token[0].item()
        if text_token not in (0, 3):
            _text = text_tokenizer.id_to_piece(text_token)
            _text = _text.replace("▁", " ")
            print(_text, end="", flush=True)

        last_text_token = last_text_token[None]
    print()
    token_per_second = steps / (time.time() - start_time)
    print(f"steps: {steps}, token per sec: {token_per_second}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--mimi", type=str)
    parser.add_argument("--trace-file", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quantized", type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("mode", default="text", type=str)
    args = parser.parse_args()

    model_file = args.model
    tokenizer_file = args.tokenizer
    if model_file is None:
        model_file = str(Path.home() / "tmp/" / "moshiko_mlx_301e30bf@120.safetensors")
    if tokenizer_file is None:
        tokenizer_file = str(Path.home() / "tmp" / "tokenizer_spm_32k_3.model")

    print(f"loading text tokenizer {tokenizer_file}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)
    mx.random.seed(299792458)

    lm_config = moshi_mlx.models.config_v0_1()
    if args.verbose:
        print(f"model config:\n{lm_config}")

    model = moshi_mlx.models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if args.quantized is not None:
        nn.quantize(model, bits=args.quantized)

    if args.verbose:
        tree_map_with_path(lambda p, t: print(p, t.shape), model.parameters())

    print(f"loading weights {model_file}")
    model.load_weights(model_file, strict=True)
    print("weights loaded")

    if args.trace_file is not None:
        if "MTL_CAPTURE_ENABLED" not in os.environ:
            raise ValueError("Set MTL_CAPTURE_ENABLED to record a trace")
        trace_file = Path(args.trace_file)
        if trace_file.exists():
            shutil.rmtree(trace_file)
        mx.metal.start_capture(args.trace_file)

    if args.mode == "text":
        run_text_gen(model, text_tokenizer, args.steps)
    elif args.mode == "audio":
        run_audio_gen(model, args.mimi, text_tokenizer, args.steps)
    elif args.mode == "stream":
        asyncio.run(run_audio_gen_stream(model, args.mimi, text_tokenizer, args.steps))
    else:
        raise ValueError(f"unknown mode {args.mode}, try 'text' or 'audio'")

    if args.trace_file is not None:
        mx.metal.stop_capture()


if __name__ == "__main__":
    main()
