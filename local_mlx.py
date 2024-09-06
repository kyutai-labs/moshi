# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import json
import queue
import sys
import time
import numpy as np
import multiprocessing
from pathlib import Path
import sentencepiece
import sounddevice as sd
from enum import Enum
import typing as tp

import mlx.core as mx
import mlx.nn as nn

from client_utils import AnyPrinter, Printer, RawPrinter
import mimi
import msh_mlx

SAMPLE_RATE = 24000
CHANNELS = 1

class Stats:
    send_times: tp.List[float] = []
    model_times: tp.List[tp.Tuple[float, float]] = []
    recv_times: tp.List[float] = []

class PrinterType(Enum):
    TOKEN = 1
    PENDING = 2
    INFO = 3
    WARNING = 4
    ERROR = 5
    LAG = 6
    HEADER = 7
    EVENT = 8

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

def server(printer_q, client_to_server, server_to_client, args):
    model_file = args.model
    tokenizer_file = args.tokenizer
    if model_file is None:
        model_file = str(Path.home() / "tmp/" / "mimi_0abbed5f@100.safetensors")
    if tokenizer_file is None:
        tokenizer_file = str(Path.home() / "tmp" / "tokenizer_spm_32k_3.model")
    steps = args.steps
    def log(s):
        printer_q.put_nowait((PrinterType.INFO, s))
    log(f"[SERVER] loading text tokenizer {tokenizer_file}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)
    mx.random.seed(299792458)
    lm_config = msh_mlx.models.config_v0_1()
    model = msh_mlx.models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if args.quantized is not None:
        nn.quantize(model, bits=args.quantized)

    log(f"[SERVER] loading weights {model_file}")
    model.load_weights(model_file, strict=True)
    log("[SERVER] weights loaded")

    model.warmup()
    log("[SERVER] model warmed up")
    gen = msh_mlx.models.LmGen(
        model=model,
        max_steps=steps + 5,
        text_sampler=msh_mlx.utils.Sampler(),
        audio_sampler=msh_mlx.utils.Sampler(),
        check=False,
    )

    server_to_client.put("start")
    log("[SERVER] connected!")
    printed_header = False
    try:
        while True:
            data = client_to_server.get()
            printer_q.put_nowait((PrinterType.EVENT, "s_get"))
            if not printed_header:
                printed_header = True
                printer_q.put_nowait((PrinterType.HEADER, ""))
            data = mx.array(data).transpose(1, 0)[:, :8]
            text_token = gen.step(data)
            text_token = text_token[0].item()
            audio_tokens = gen.last_audio_tokens()
            if text_token not in (0, 3):
                _text = text_tokenizer.id_to_piece(text_token)
                _text = _text.replace("▁", " ")
                printer_q.put_nowait((PrinterType.TOKEN, _text))
            else:
                printer_q.put_nowait((PrinterType.PENDING, ""))
            if audio_tokens is not None:
                audio_tokens = np.array(audio_tokens).astype(np.uint32)
                printer_q.put_nowait((PrinterType.EVENT, "s_put"))
                server_to_client.put_nowait(audio_tokens)
    except KeyboardInterrupt:
        pass


def client(printer_q, client_to_server, server_to_client, args):
    mimi_file = args.mimi
    if mimi_file is None:
        mimi_file = str(Path.home() / "tmp" / "tokenizer-de0e421d-checkpoint40.safetensors")
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    audio_tokenizer = mimi.StreamTokenizer(mimi_file)
    start = server_to_client.get()
    printer_q.put_nowait((PrinterType.INFO, f"[CLIENT] received '{start}' from server, starting..."))

    full_warmup(audio_tokenizer, client_to_server, server_to_client)


    async def send_loop():
        while True:
            await asyncio.sleep(0.001)
            try:
                pcm_data = input_queue.get(block=False)
                printer_q.put_nowait((PrinterType.EVENT, "encode"))
                audio_tokenizer.encode(pcm_data)
            except queue.Empty:
                continue

    async def recv_loop():
        while True:
            data = audio_tokenizer.get_decoded()
            if data is None:
                await asyncio.sleep(0.001)
                continue
            printer_q.put_nowait((PrinterType.EVENT, "decoded"))
            output_queue.put_nowait(data)

    async def send_loop2():
        while True:
            data = audio_tokenizer.get_encoded()
            if data is None:
                await asyncio.sleep(0.001)
                continue
            printer_q.put_nowait((PrinterType.EVENT, "encoded"))
            client_to_server.put_nowait(data)

    async def recv_loop2():
        while True:
            try:
                audio_tokens = server_to_client.get(block=False)
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue
            printer_q.put_nowait((PrinterType.EVENT, "decode"))
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

    async def go():
        with in_stream, out_stream:
            await asyncio.gather(recv_loop(), send_loop(), recv_loop2(), send_loop2())
    try:
        asyncio.run(go())
    except KeyboardInterrupt:
        pass

def main(printer: AnyPrinter):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--mimi", type=str)
    parser.add_argument("--quantized", type=int)
    parser.add_argument("--steps", default=2500, type=int)
    args = parser.parse_args()

    client_to_server = multiprocessing.Queue()
    server_to_client = multiprocessing.Queue()
    printer_q = multiprocessing.Queue()

    # Create two processes
    subprocess_args= printer_q, client_to_server, server_to_client, args
    p1 = multiprocessing.Process(target=client, args=subprocess_args)
    p2 = multiprocessing.Process(target=server, args=subprocess_args)

    # Start the processes
    p1.start()
    p2.start()
    events = []

    try:
        while p1.is_alive() and p2.is_alive():
            time.sleep(0.001)
            try:
                ty, value = printer_q.get_nowait()
                if ty == PrinterType.TOKEN:
                    printer.print_token(value)
                elif ty == PrinterType.PENDING:
                    printer.print_pending()
                elif ty == PrinterType.INFO:
                    printer.log("info", value)
                elif ty == PrinterType.WARNING:
                    printer.log("warning", value)
                elif ty == PrinterType.ERROR:
                    printer.log("error", value)
                elif ty == PrinterType.LAG:
                    printer.print_lag()
                elif ty == PrinterType.HEADER:
                    printer.print_header()
                elif ty == PrinterType.EVENT:
                    events.append({"event": value, "time": time.time() })
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        printer.log("warning", "Interrupting, exiting connection.")
        p1.terminate()
        p2.terminate()

    printer.log("info", "saving trace")
    with open("mlx-trace.json", "w") as fobj:
        json.dump(events, fobj)

    # Wait for both processes to finish
    p1.join()
    p2.join()
    printer.log("info", "All done!")


if __name__ == "__main__":
    printer: AnyPrinter
    if sys.stdout.isatty():
        printer = Printer()
    else:
        printer = RawPrinter()
    main(printer)

