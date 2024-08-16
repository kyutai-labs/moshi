import argparse
import asyncio
from dataclasses import dataclass
from websockets.server import serve
import msh
import sentencepiece
import torch
import numpy as np
import random

SAMPLE_RATE = msh.models.moshi.SAMPLE_RATE
DEVICE = "cuda:0"
ENABLE_PROFILING = False

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="localhost", type=str)
parser.add_argument("--port", default=8998, type=int)
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all(42424242)


@dataclass
class ServerState:
    ec: msh.models.EncodecModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm: msh.models.LMModel

    def __init__(self):
        print("loading mimi")
        self.ec = msh.models.moshi.get_encodec(args.mimi_weights, DEVICE)
        print("mimi loaded")
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

        print("loading moshi")
        self.lm = msh.models.moshi.get_lm(args.moshi_weights, DEVICE)
        print("lm loaded")

    async def handle_conn(self, websocket, path):
        print(websocket, path)
        self.lm.reset_streaming()
        max_gen_len = 256
        lm_gen = msh.models.LMGen(self.lm, check=True, max_gen_len=max_gen_len)

        with self.ec.streaming():
            async for message in websocket:
                if not isinstance(message, bytes):
                    print("unsupported message type {type(message)}")
                    continue
                if len(message) == 0:
                    print("empty message")
                    continue
                kind = message[0]
                print("received message", kind)
                if kind == 1:  # audio
                    payload = message[1:]
                    # TODO(laurent): ogg + opus decoding
                    np_array = np.frombuffer(payload, dtype=np.float32)
                    chunk = torch.tensor(np_array, device=DEVICE)[None, None]
                    print("pcm to process", chunk.shape)
                    codes, _scale = self.ec.encode(chunk)
                    print("codes to process", codes.shape)
                    for c in range(codes.shape[-1]):
                        tokens = lm_gen.step(codes[0, :, c].tolist())
                        text_token = tokens[0]
                        print("generated", tokens)
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)
                            _text = _text.replace("▁", " ")
                            msg = b"\0x02" + bytes(_text, encoding="utf8")
                            print("text token", msg)
                            websocket.send(msg)
                        if all([t < 2048 for t in tokens[1:]]):
                            tokens = torch.tensor(tokens[1:], device=DEVICE).reshape(
                                (1, 8, 1)
                            )
                            main_pcm = self.ec.decode(tokens, scale=None)
                            msg = b"\0x01" + main_pcm.cpu().numpy().tobytes()
                            # TODO(laurent): ogg + opus encoding
                            websocket.send(msg)
                else:
                    print("unknown message kind {kind}")


async def main():
    state = ServerState()
    print(f"listening to ws://{args.host}:{args.port}")
    async with serve(state.handle_conn, args.host, args.port):
        await asyncio.Future()  # run forever


with torch.no_grad():
    asyncio.run(main())


def cb(step, total):
    print(f"{step:06d} / {total:06d}", end="\r")
