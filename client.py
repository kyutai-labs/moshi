# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import numpy as np
import queue
import sounddevice as sd
import websockets

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="localhost", type=str)
parser.add_argument("--port", default=8998, type=int)
args = parser.parse_args()

SAMPLE_RATE = 24000
CHANNELS = 1


async def main():
    uri = f"ws://{args.host}:{args.port}"
    print(f"connecting to {uri}")

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    async with websockets.connect(uri) as websocket:

        async def queue_loop():
            print("start queue loop")
            while True:
                await asyncio.sleep(0.001)
                try:
                    msg = input_queue.get(block=False)
                except queue.Empty:
                    continue
                await websocket.send(msg)
                input_queue.task_done()

        async def recv_loop():
            print("start recv loop")
            cnt = 0
            while True:
                message = await websocket.recv()
                if not isinstance(message, bytes):
                    print(f"unsupported message type {type(message)}")
                    continue
                if len(message) == 0:
                    print("empty message")
                    continue
                kind = message[0]
                if kind == 1:  # audio
                    payload = message[1:]
                    # TODO(laurent): ogg + opus decoding + play
                    payload = np.frombuffer(payload, dtype=np.float32)
                    output_queue.put_nowait(payload)
                    cnt += 1
                elif kind == 2:  # text
                    payload = message[1:]
                    print("text", payload)
                else:
                    print(f"unknown message kind {kind}")

        def on_input(in_data, frames, time, status):
            # TODO(laurent): opus encoding
            msg = b"\x01" + in_data.tobytes()
            input_queue.put_nowait(msg)

        in_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=1920, callback=on_input
        )

        def on_output(out_data, frames, time, status):
            # print(frames, type(out_data))
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

        with in_stream, out_stream:
            await asyncio.gather(recv_loop(), queue_loop())


asyncio.run(main())
