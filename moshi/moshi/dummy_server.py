# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
import os.path
import random
import time

import aiohttp
from aiohttp import web
import numpy as np
import sphn
import torch


from .client_utils import log


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


@dataclass
class ServerState:
    lock: asyncio.Lock

    def __init__(self):
        self.frame_size = 1920
        self.sample_rate = 24000
        self.freq = 440
        self.lock = asyncio.Lock()

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
                        log("warning", "got audio but we are a send only server.")
                    else:
                        log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                log("info", "connection closed")

        async def opus_loop():
            offset = 0
            frame_no = 0

            while True:
                if close:
                    return
                be = time.time()
                t = torch.arange(offset, offset + self.frame_size).float() / self.sample_rate
                offset += self.frame_size
                pcm = 0.1 * torch.cos(2 * 3.14 * self.freq * t)
                opus_writer.append_pcm(pcm.numpy())
                frame_no += 1
                if frame_no % 20 == 0:
                    msg = b"\x02" + bytes(" " + str(frame_no // 20), encoding="utf8")
                    await ws.send_bytes(msg)
                spent = time.time() - be
                sleep = max(0, self.frame_size / self.sample_rate - spent)
                await asyncio.sleep(sleep)

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
            opus_writer = sphn.OpusStreamWriter(self.sample_rate)
            # Send the handshake.
            await ws.send_bytes(b"\x00")
            await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        log("info", "done with connection")
        return ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()

    state = ServerState()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    if args.static is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(args.static, "index.html"))

        log("info", f"serving static content from {args.static}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=args.static, follow_symlinks=True, name="static")

    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = os.path.join(args.ssl, "cert.pem")
        key_file = os.path.join(args.ssl, "key.pem")
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    log("info", f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
