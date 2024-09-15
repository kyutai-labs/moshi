# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import random
import tarfile
import time

import os
import numpy as np
import sentencepiece
import sphn
import torch
import aiohttp
from aiohttp import web

from huggingface_hub import hf_hub_download

from .models import moshi_, EncodecModel, LMGen

SAMPLE_RATE = moshi_.SAMPLE_RATE
DEVICE = "cuda:0"
ENABLE_PROFILING = False


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


parser = argparse.ArgumentParser()
parser.add_argument("--host", default="localhost", type=str)
parser.add_argument("--port", default=8998, type=int)
parser.add_argument("--static", type=str)
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--moshi-weights", type=str)
parser.add_argument("--mimi-weights", type=str)
parser.add_argument("--hf-repo", type=str, default="kmhf/msh-v0.1")

args = parser.parse_args()

if args.tokenizer is None:
    args.tokenizer = hf_hub_download(args.hf_repo, "tokenizer_spm_32k_3.model")
if args.moshi_weights is None:
    args.moshi_weights = hf_hub_download(
        args.hf_repo, "moshiko_pt_301e30bf@120.safetensors"
    )
if args.mimi_weights is None:
    args.mimi_weights = hf_hub_download(
        args.hf_repo, "tokenizer-e351c8d8-checkpoint125.safetensors"
    )


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


seed_all(42424242)


@dataclass
class ServerState:
    ec: EncodecModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self):
        log("info", "loading mimi")
        self.ec = moshi_.get_encodec(args.mimi_weights, DEVICE)
        log("info", "mimi loaded")
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)
        log("info", "loading moshi")
        lm = moshi_.get_lm(args.moshi_weights, DEVICE)
        self.lm_gen = LMGen(lm)

        self.frame_size = int(self.ec.sample_rate / self.ec.frame_rate)
        self.lock = asyncio.Lock()

        self.ec.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        log("info", "lm loaded")

    def warmup(self):
        for chunk in range(4):
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=torch.float32, device=DEVICE
            )
            codes = self.ec.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.ec.decode(tokens[:, 1:])
        torch.cuda.synchronize()

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async def recv_loop():
            nonlocal close
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
                        opus_reader.append_bytes(payload)
                    else:
                        log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                log("info", "connection closed")

        async def opus_loop():
            all_pcm_data = None

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    be = time.time()
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size :]
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=DEVICE)[None, None]
                    codes = self.ec.encode(chunk)
                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.ec.decode(tokens[:, 1:])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            log("info", f"text token '{_text}'")
                            await ws.send_bytes(msg)
                    log("info", f"frame handled in {1000 * (time.time() - be):.1f}ms")

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        log("info", "accepted connection")
        close = False
        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.ec.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.ec.sample_rate)
            self.ec.reset_streaming()
            self.lm_gen.reset_streaming()
            # Send the handshake.
            await ws.send_bytes(b"\x00")
            await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        log("info", "done with connection")
        return ws


def main():
    state = ServerState()
    log("info", "warming up the model")
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    static_path: None | str = None
    if args.static is None:
        log("info", f"retrieving the static content")
        dist_tgz = hf_hub_download(args.hf_repo, "dist.tgz")
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
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    log("info", f"listening to ws://{args.host}:{args.port}")
    web.run_app(app, port=args.port)


with torch.no_grad():
    main()
