# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import queue
import sys

import numpy as np
import sphn
import sounddevice as sd
import websockets

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="localhost", type=str)
parser.add_argument("--port", default=8998, type=int)
args = parser.parse_args()

SAMPLE_RATE = 24000
CHANNELS = 1
FRAME_SIZE = 1920


def colorize(text, color):
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class TokenPrinter:
    def __init__(self, max_cols: int = 80, stream=sys.stdout):
        self.max_cols = max_cols
        self.current_line: str = ''
        self.stream = stream
        self._pending_count = 0
        self._pending_printed = False

    def print_header(self):
        print('-' * self.max_cols)

    def print_token(self, token: str):
        if self._pending_printed:
            self._pending_printed = False
            self.stream.write('\r' + self.current_line + ' ' * 2)
        if len(self.current_line) + len(token) <= self.max_cols:
            self.stream.write(token)
            self.current_line += token
        else:
            if token.startswith(' '):
                self.stream.write('\n' + token)
                self.current_line = token
            else:
                if ' ' in self.current_line:
                    line, word = self.current_line.rsplit(' ', 1)
                    sys.stdout.write('\r' + line + ' ' * (len(word) + 1) + '\n' + word + token)
                    self.current_line = word + token
                else:
                    remaining = self.max_cols - len(self.current_line)
                    sys.stdout.write(token[:remaining] + '\n' + token[remaining:])
                    self.current_line = token[remaining:]
        self.stream.flush()

    def print_pending(self):
        chars = ['|', '/', '-', '\\']
        count = int(self._pending_count / 5)
        char = chars[count % len(chars)]
        colors = ['32', '33', '31']
        char = colorize(char, colors[count % len(colors)])
        self.stream.write('\r' + self.current_line + ' ' + char)
        self.stream.flush()
        self._pending_printed = True
        self._pending_count += 1


async def main():
    uri = f"ws://{args.host}:{args.port}"
    log(f"connecting to {uri}")

    opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
    opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
    output_queue = queue.Queue()
    printer = TokenPrinter()

    async with websockets.connect(uri) as websocket:

        async def queue_loop():
            while True:
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await websocket.send(b"\x01" + msg)

        async def decoder_loop():
            all_pcm_data = None
            while True:
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= FRAME_SIZE:
                    output_queue.put(all_pcm_data[:FRAME_SIZE])
                    all_pcm_data = np.array(all_pcm_data[FRAME_SIZE:])

        async def recv_loop():
            log("start recv loop")
            printer.print_header()
            cnt = 0
            while True:
                message = await websocket.recv()
                if not isinstance(message, bytes):
                    log(f"unsupported message type {type(message)}")
                    continue
                if len(message) == 0:
                    log("empty message")
                    continue
                kind = message[0]
                if kind == 1:  # audio
                    payload = message[1:]
                    opus_reader.append_bytes(payload)
                    printer.print_pending()
                    cnt += 1
                elif kind == 2:  # text
                    payload = message[1:]
                    printer.print_token(payload.decode())
                else:
                    log(f"unknown message kind {kind}")

        def on_input(in_data, frames, time_, status):
            assert in_data.shape[0] == FRAME_SIZE
            opus_writer.append_pcm(in_data[:, 0])

        in_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=FRAME_SIZE, callback=on_input
        )

        def on_output(out_data, frames, time_, status):
            assert out_data.shape == (FRAME_SIZE, 1), out_data.shape
            try:
                pcm_data = output_queue.get(block=False)
                # TODO: handle other shapes by using some form of fifo/ring buffer.
                assert pcm_data.shape == (FRAME_SIZE,), pcm_data.shape
                out_data[:, 0] = pcm_data
            except queue.Empty:
                out_data.fill(0)
                log("[Skipped a frame!]")

        out_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=FRAME_SIZE,
            callback=on_output,
        )

        with in_stream, out_stream:
            await asyncio.gather(recv_loop(), queue_loop(), decoder_loop())

asyncio.run(main())
