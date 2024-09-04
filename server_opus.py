# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
import random
import time
import traceback

import numpy as np
import sentencepiece
import sphn
import torch
import websockets
from websockets.server import serve

import msh

SAMPLE_RATE = msh.models.moshi.SAMPLE_RATE
DEVICE = "cuda:0"
ENABLE_PROFILING = False

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="localhost", type=str)
parser.add_argument("--port", default=8998, type=int)
parser.add_argument("--max-gen-len", default=2048, type=int)
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--moshi-weights", type=str)
parser.add_argument("--mimi-weights", type=str)
args = parser.parse_args()

if args.tokenizer is None:
    raise ValueError("--tokenizer must be set")
if args.moshi_weights is None:
    raise ValueError("--moshi-weights must be set")
if args.mimi_weights is None:
    raise ValueError("--mimi-weights must be set")


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
    ec: msh.models.EncodecModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: msh.models.LMGen
    lock: asyncio.Lock

    def __init__(self):
        print("loading mimi")
        self.ec = msh.models.moshi.get_encodec(args.mimi_weights, DEVICE)
        print("mimi loaded")
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)
        print("loading moshi")
        lm = msh.models.moshi.get_lm(args.moshi_weights, DEVICE)
        self.lm_gen = msh.models.LMGen(lm)

        self.frame_size = int(self.ec.sample_rate / self.ec.frame_rate)
        self.lock = asyncio.Lock()

        self.ec.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        print("lm loaded")

    def warmup(self):
        for chunk in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=DEVICE)
            codes = self.ec.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.ec.decode(tokens[:, 1:])
        torch.cuda.synchronize()

    async def handle_conn(self, websocket, path):
        async def recv_loop():
            nonlocal close
            try:
                async for message in websocket:
                    if not isinstance(message, bytes):
                        print("unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        print("empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        print("unknown message kind {kind}")
            except websockets.exceptions.WebSocketException:
                print("Lost connection.")
            finally:
                close = True
                print("CLOSED")

        async def opus_loop():
            all_pcm_data = None
            all_pcm_out = []
            all_codes = []
            all_other = []
            all_tl = []
            all_al = []

            while True:
                if close:
                    pcm_out = torch.cat(all_pcm_out, dim=1)
                    sphn.write_wav("./plop.wav", pcm_out[0].numpy(), 24000)
                    codes = torch.cat(all_codes, dim=-1)
                    other = torch.cat(all_other, dim=-1)
                    tl = torch.cat(all_tl, dim=-2)
                    al = torch.cat(all_al, dim=-2)
                    torch.save((other, codes, tl, al), "./codes.th")
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
                    chunk = all_pcm_data[:self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]
                    chunk = torch.from_numpy(chunk)
                    print(chunk.std())
                    chunk = chunk.to(device=DEVICE)[None, None]
                    codes = self.ec.encode(chunk)
                    all_other.append(codes)
                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                        all_tl.append(self.lm_gen.tl.clone())
                        all_al.append(self.lm_gen.al.clone())
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.ec.decode(tokens[:, 1:])
                        main_pcm = main_pcm.cpu()
                        all_pcm_out.append(main_pcm[0])
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        all_codes.append(tokens)
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            print("text token", msg)
                            await websocket.send(msg)
                    print(f"Frame handled in {1000 * (time.time() - be):.1f}")

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await websocket.send(b"\x01" + msg)


        print("OPEN")
        close = False
        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.ec.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.ec.sample_rate)
            self.ec.reset_streaming()
            self.lm_gen.reset_streaming()
            await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        print("DONE")


async def main():
    state = ServerState()
    print("warming up the model")
    state.warmup()
    print(f"listening to ws://{args.host}:{args.port}")
    async with serve(state.handle_conn, args.host, args.port):
        await asyncio.Future()  # run forever


with torch.no_grad():
    asyncio.run(main())


def cb(step, total):
    print(f"{step:06d} / {total:06d}", end="\r")
