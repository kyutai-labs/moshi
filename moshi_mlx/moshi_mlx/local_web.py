# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import json
import queue
import os
import tarfile
import time
import sys
import numpy as np
import multiprocessing
from pathlib import Path
import sentencepiece
from enum import Enum
import typing as tp
import sphn
import aiohttp
from aiohttp import web
import webbrowser

import mlx.core as mx
import mlx.nn as nn

import rustymimi
from moshi_mlx import models, utils

import huggingface_hub

SAMPLE_RATE = 24000
FRAME_SIZE = 1920
CHANNELS = 1


def colorize(text, color):
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def log(level: str, msg: str):
    if level == "warning":
        prefix = colorize("[Warn]", "1;31")
    elif level == "info":
        prefix = colorize("[Info]", "1;34")
    elif level == "error":
        prefix = colorize("[Err ]", "1;31")
    else:
        raise ValueError(f"Unknown level {level}")
    print(prefix + " " + msg)


def hf_hub_download(repo, path: str) -> str:
    if repo is None or repo == "":
        raise ValueError(f"the --hf-repo flag is required to retrieve {path}")
    return huggingface_hub.hf_hub_download(repo, path)


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
    QSIZE = 9


def full_warmup(audio_tokenizer, client_to_server, server_to_client, max_delay: int):
    for i in range(4):
        pcm_data = np.array([0.0] * 1920).astype(np.float32)
        audio_tokenizer.encode(pcm_data)
        while True:
            time.sleep(0.01)
            data = audio_tokenizer.get_encoded()
            if data is not None:
                break
        client_to_server.put_nowait(data)
        if i < max_delay:
            continue
        while True:
            kind, data = server_to_client.get()
            if kind == 0:
                audio_tokenizer.decode(data)
                break
        while True:
            time.sleep(0.01)
            data = audio_tokenizer.get_decoded()
            if data is not None:
                break


def hf_get(filename: str) -> str:
    if filename.startswith("hf://"):
        parts = filename[5:].split("/")
        repo_name = parts[0] + "/" + parts[1]
        filename = "/".join(parts[2:])
        log("info", f"retrieving {filename} from hf repo {repo_name}")
        return hf_hub_download(repo_name, filename)
    else:
        return filename


def model_server(client_to_server, server_to_client, lm_config, args):
    model_file = args.moshi_weight
    tokenizer_file = args.tokenizer
    if model_file is None:
        if type(lm_config) is dict and "moshi_name" in lm_config:
            model_file = hf_hub_download(args.hf_repo, lm_config["moshi_name"])
        elif args.quantized == 8:
            model_file = hf_hub_download(args.hf_repo, "model.q8.safetensors")
        elif args.quantized == 4:
            model_file = hf_hub_download(args.hf_repo, "model.q4.safetensors")
        elif args.quantized is not None:
            raise ValueError(f"Invalid quantized value: {args.quantized}")
        else:
            model_file = hf_hub_download(args.hf_repo, "model.safetensors")
    model_file = hf_get(model_file)
    if tokenizer_file is None:
        if type(lm_config) is dict and "tokenizer_name" in lm_config:
            tokenizer_file = hf_hub_download(args.hf_repo, lm_config["tokenizer_name"])
        else:
            tokenizer_file = hf_hub_download(args.hf_repo, "tokenizer_spm_32k_3.model")
    tokenizer_file = hf_get(tokenizer_file)
    steps = args.steps

    log("info", f"[SERVER] loading text tokenizer {tokenizer_file}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)  # type: ignore
    mx.random.seed(299792458)
    if type(lm_config) is dict:
        lm_config = models.LmConfig.from_config_dict(lm_config)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if args.quantized is not None:
        group_size = 32 if args.quantized == 4 else 64
        nn.quantize(model, bits=args.quantized, group_size=group_size)

    log("info", f"[SERVER] loading weights {model_file}")
    model.load_weights(model_file, strict=True)
    log("info", "[SERVER] weights loaded")

    if model.condition_provider is not None:
        ct = model.condition_provider.condition_tensor("description", "very_good")
    else:
        ct = None

    log("info", "[SERVER] warming up the model")
    model.warmup(ct)
    log("info", "[SERVER] model warmed up")
    gen = models.LmGen(
        model=model,
        max_steps=steps + 5,
        text_sampler=utils.Sampler(),
        audio_sampler=utils.Sampler(),
        check=False,
    )

    server_to_client.put("start")
    log("info", "[SERVER] connected!")
    try:
        while True:
            data = client_to_server.get()
            data = mx.array(data).transpose(1, 0)[:, : gen.main_codebooks]
            text_token = gen.step(data, ct=ct)
            text_token = text_token[0].item()
            audio_tokens = gen.last_audio_tokens()
            if text_token not in (0, 3):
                _text = text_tokenizer.id_to_piece(text_token)  # type: ignore
                _text = _text.replace("▁", " ")
                server_to_client.put_nowait((1, _text))
            if audio_tokens is not None:
                audio_tokens = np.array(audio_tokens).astype(np.uint32)
                server_to_client.put_nowait((0, audio_tokens))
    except KeyboardInterrupt:
        pass


def web_server(client_to_server, server_to_client, lm_config, args):
    mimi_file = args.mimi_weight
    if mimi_file is None:
        if type(lm_config) is dict and "mimi_name" in lm_config:
            mimi_file = hf_hub_download(args.hf_repo, lm_config["mimi_name"])
        else:
            mimi_file = hf_hub_download(
                args.hf_repo, "tokenizer-e351c8d8-checkpoint125.safetensors"
            )
    mimi_file = hf_get(mimi_file)
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    text_queue = queue.Queue()
    if type(lm_config) is dict:
        nc = lm_config.get("dep_q", 8)
        max_delay = max(lm_config["delays"])
    else:
        nc = lm_config.depformer.num_slices
        max_delay = max(lm_config.audio_delays)
    audio_tokenizer = rustymimi.StreamTokenizer(mimi_file, num_codebooks=nc)  # type: ignore
    start = server_to_client.get()
    log("info", f"[CLIENT] received '{start}' from server, starting...")

    full_warmup(audio_tokenizer, client_to_server, server_to_client, max_delay)

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
                kind, data = server_to_client.get(block=False)
                if kind == 0:
                    audio_tokenizer.decode(data)
                elif kind == 1:
                    text_queue.put_nowait(data)
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue

    lock = asyncio.Lock()

    async def handle_chat(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async def recv_loop():
            nonlocal close
            all_pcm_data = None
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        pcm = opus_reader.append_bytes(payload)
                        if pcm.shape[-1] == 0:
                            continue
                        if all_pcm_data is None:
                            all_pcm_data = pcm
                        else:
                            all_pcm_data = np.concatenate((all_pcm_data, pcm))
                        while all_pcm_data.shape[-1] >= FRAME_SIZE:
                            chunk = all_pcm_data[:FRAME_SIZE]
                            all_pcm_data = all_pcm_data[FRAME_SIZE:]
                            input_queue.put_nowait(chunk)

                    else:
                        log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                log("info", "connection closed")

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                try:
                    pcm_data = output_queue.get(block=False)
                    assert pcm_data.shape == (1920,), pcm_data.shape
                    msg = opus_writer.append_pcm(pcm_data)
                    if len(msg) > 0:
                        await ws.send_bytes(b"\x01" + msg)
                    _text = text_queue.get(block=False)
                    await ws.send_bytes(b"\x02" + bytes(_text, encoding="utf8"))
                except queue.Empty:
                    continue

        log("info", "accepted connection")
        close = False
        async with lock:
            log("info", "lock acquired")
            opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
            opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
            # Send the handshake.
            await ws.send_bytes(b"\x00")
            await asyncio.gather(recv_loop(), send_loop())
        log("info", "done with connection")
        return ws

    async def go():
        app = web.Application()
        app.router.add_get("/api/chat", handle_chat)
        static_path: None | str = None
        if args.static is None:
            log("info", "retrieving the static content")
            dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz")
            dist_tgz = Path(dist_tgz)
            dist = dist_tgz.parent / "dist"
            if not dist.exists():
                with tarfile.open(dist_tgz, "r:gz") as tar:
                    tar.extractall(path=dist_tgz.parent)
            static_path = str(dist)
        elif args.static != "none":
            # When set to the "none" string, we don't serve any static content.
            static_path = args.static
        if static_path is not None:

            async def handle_root(_):
                return web.FileResponse(os.path.join(static_path, "index.html"))

            log("info", f"serving static content from {static_path}")
            app.router.add_get("/", handle_root)
            app.router.add_static("/", path=static_path, name="static")
        runner = web.AppRunner(app)
        await runner.setup()
        ssl_context = None
        protocol = "http"
        if args.ssl is not None:
            import ssl

            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            cert_file = os.path.join(args.ssl, "cert.pem")
            key_file = os.path.join(args.ssl, "key.pem")
            ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
            protocol = "https"
        site = web.TCPSite(runner, args.host, args.port, ssl_context=ssl_context)

        log("info", f"listening to {protocol}://{args.host}:{args.port}")

        if not args.no_browser:
            log("info", f"opening browser at {protocol}://{args.host}:{args.port}")
            webbrowser.open(f"{protocol}://{args.host}:{args.port}")

        await asyncio.gather(
            recv_loop(), send_loop(), recv_loop2(), send_loop2(), site.start()
        )
        await runner.cleanup()

    try:
        asyncio.run(go())
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi-weight", type=str)
    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("-q", "--quantized", type=int, choices=[4, 8])
    parser.add_argument("--steps", default=4000, type=int)
    parser.add_argument("--hf-repo", type=str)
    parser.add_argument("--static", type=str)
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--lm-config", type=str, help="The LM config as a json file.")
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        ),
    )
    parser.add_argument("--no-browser", action="store_true")

    args = parser.parse_args()
    if args.hf_repo is None:
        if args.quantized == 8:
            args.hf_repo = "kyutai/moshiko-mlx-q8"
        elif args.quantized == 4:
            args.hf_repo = "kyutai/moshiko-mlx-q4"
        elif args.quantized is None:
            args.hf_repo = "kyutai/moshiko-mlx-bf16"
        else:
            print(f"Invalid value for quantized {args.quantized}")
            sys.exit(1)

    client_to_server = multiprocessing.Queue()
    server_to_client = multiprocessing.Queue()

    # Get the model config
    lm_config = args.lm_config
    if lm_config is None:
        try:
            lm_config = hf_hub_download(args.hf_repo, "config.json")
        except Exception:
            log("warning", "Cannot download config, using defaults.")
    if lm_config is None:
        lm_config = models.config_v0_1()
    else:
        with open(hf_get(lm_config), "r") as fobj:
            lm_config = json.load(fobj)

    # Create two processes
    subprocess_args = client_to_server, server_to_client, lm_config, args
    p1 = multiprocessing.Process(target=web_server, args=subprocess_args)
    p2 = multiprocessing.Process(target=model_server, args=subprocess_args)

    # Start the processes
    p1.start()
    p2.start()

    try:
        while p1.is_alive() and p2.is_alive():
            time.sleep(0.001)
    except KeyboardInterrupt:
        log("warning", "Interrupting, exiting connection.")
        p1.terminate()
        p2.terminate()

    # Wait for both processes to finish
    p1.join()
    p2.join()
    log("info", "All done!")


if __name__ == "__main__":
    main()
