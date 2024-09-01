# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import queue
import time
import numpy as np
import multiprocessing
from pathlib import Path
import sentencepiece
import sounddevice as sd

import mlx.core as mx
import mlx.nn as nn

import mimi
import msh_mlx

SAMPLE_RATE = 24000
CHANNELS = 1

def full_warmup(audio_tokenizer, client_to_server, server_to_client):
    for i in range(4):
        pcm_data = np.array([0.] * 1920).astype(np.float32)
        audio_tokenizer.encode(pcm_data)
        while True:
            time.sleep(0.01)
            data = audio_tokenizer.get_encoded()
            if data is not None:
                break
        client_to_server.put_nowait(data)
        if i == 0:
            continue
        audio_tokens = server_to_client.get()
        audio_tokenizer.decode(audio_tokens)
        while True:
            time.sleep(0.01)
            data = audio_tokenizer.get_decoded()
            if data is not None:
                break

def server(client_to_server, server_to_client, args):
    model_file = args.model
    tokenizer_file = args.tokenizer
    if model_file is None:
        model_file = str(Path.home() / "tmp/" / "mimi_0abbed5f@100.safetensors")
    if tokenizer_file is None:
        tokenizer_file = str(Path.home() / "tmp" / "tokenizer_spm_32k_3.model")
    steps = args.steps
    print(f"[SERVER] loading text tokenizer {tokenizer_file}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)
    mx.random.seed(299792458)
    lm_config = msh_mlx.models.config_v0_1()
    model = msh_mlx.models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if args.quantized:
        nn.quantize(model, bits=8)

    print(f"[SERVER] loading weights {model_file}")
    model.load_weights(model_file, strict=True)
    print("[SERVER] weights loaded")

    model.warmup()
    print("[SERVER] model warmed up")
    gen = msh_mlx.models.LmGen(
        model=model,
        max_steps=steps + 5,
        text_sampler=msh_mlx.utils.Sampler(),
        audio_sampler=msh_mlx.utils.Sampler(),
        check=False,
    )

    server_to_client.put("start")
    while True:
        data = client_to_server.get()
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
            server_to_client.put_nowait(audio_tokens)


def client(client_to_server, server_to_client, args):
    mimi_file = args.mimi
    if mimi_file is None:
        mimi_file = str(Path.home() / "tmp" / "tokenizer-de0e421d-checkpoint40.safetensors")
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    audio_tokenizer = mimi.StreamTokenizer(mimi_file)
    start = server_to_client.get()
    print(f"[CLIENT] received '{start}' from server, starting...")

    full_warmup(audio_tokenizer, client_to_server, server_to_client)


    async def send_loop():
        while True:
            await asyncio.sleep(0.001)
            try:
                pcm_data = input_queue.get(block=False)
                audio_tokenizer.encode(pcm_data)
            except queue.Empty:
                continue

    async def recv_loop():
        while True:
            data = audio_tokenizer.get_decoded()
            if data is None:
                await asyncio.sleep(0.001)
                continue
            output_queue.put_nowait(data)

    async def send_loop2():
        while True:
            data = audio_tokenizer.get_encoded()
            if data is None:
                await asyncio.sleep(0.001)
                continue
            client_to_server.put_nowait(data)

    async def recv_loop2():
        while True:
            try:
                audio_tokens = server_to_client.get(block=False)
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue
            audio_tokenizer.decode(audio_tokens)

    def on_input(in_data, frames, time, status):
        in_data = in_data[:, 0].astype(np.float32)
        input_queue.put_nowait(in_data)

    in_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=1920, callback=on_input
    )

    def on_output(out_data, frames, time, status):
        assert out_data.shape == (1920, 1), out_data.shape
        try:
            pcm_data = output_queue.get(block=False)
            # TODO: handle other shapes by using some form of fifo/ring buffer.
            assert pcm_data.shape == (1920,), pcm_data.shape
            out_data[:, 0] = pcm_data
        except queue.Empty:
            out_data.fill(0)


    out_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=1920,
        callback=on_output,
    )

    print("starting the inference loop")
    async def go():
        with in_stream, out_stream:
            await asyncio.gather(recv_loop(), send_loop(), recv_loop2(), send_loop2())
    asyncio.run(go())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--mimi", type=str)
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--steps", default=2500, type=int)
    args = parser.parse_args()

    client_to_server = multiprocessing.Queue()
    server_to_client = multiprocessing.Queue()

    # Create two processes
    p1 = multiprocessing.Process(target=client, args=(client_to_server, server_to_client, args))
    p2 = multiprocessing.Process(target=server, args=(client_to_server, server_to_client, args))

    # Start the processes
    p1.start()
    p2.start()

    # Wait for both processes to finish
    p1.join()
    p2.join()


if __name__ == "__main__":
    main()

