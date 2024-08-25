# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import numpy as np
import queue
import sphn
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

    opus_writer = sphn.OpusStreamWriter(24000)
    opus_reader = sphn.OpusStreamReader(24000)
    output_queue = queue.Queue()

    async with websockets.connect(uri) as websocket:

        async def queue_loop():
            print("start queue loop")
            while True:
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await websocket.send(b"\x01" + msg)

        all_pcm_data = None
        async def decoder_loop():
            print("start decoder loop")
            while True:
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                if all_pcm_data.shape[-1] >= 1920:
                    output_queue.put_nowait(all_pcm_data[:1920])
                    all_pcm_data = np.array(all_pcm_data[1920:])

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
                    payload = np.frombuffer(payload, dtype=np.float32)
                    opus_reader.append_bytes(payload)
                    cnt += 1
                elif kind == 2:  # text
                    payload = message[1:]
                    print("text", payload)
                else:
                    print(f"unknown message kind {kind}")

        def on_input(in_data, frames, time, status):
            opus_writer.append_pcm(in_data[0])

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
            await asyncio.gather(recv_loop(), queue_loop(), decoder_loop())


asyncio.run(main())
