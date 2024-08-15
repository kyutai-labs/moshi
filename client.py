import argparse
import asyncio
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

    audio_queue = queue.Queue()

    async with websockets.connect(uri) as websocket:
        async def queue_loop():
            print("start queue loop")
            while True:
                await asyncio.sleep(0.001)
                try:
                    msg = audio_queue.get(block=False)
                except queue.Empty:
                    continue
                await websocket.send(msg)
                audio_queue.task_done()

        async def recv_loop():
            print("start recv loop")
            while True:
                message = await websocket.recv()
                if not isinstance(message, bytes):
                    print("unsupported message type {type(message)}")
                    continue
                if len(message) == 0:
                    print("empty message")
                    continue
                kind = message[0]
                if kind == 1:  # audio
                    payload = message[1:]
                    # TODO(laurent): ogg + opus decoding + play
                    print(payload)
                else:
                    print("unknown message kind {kind}")


        def on_input(in_data, frames, time, status):
            # TODO(laurent): opus encoding
            msg = b'\x01' + in_data.tobytes()
            audio_queue.put_nowait(msg)

        in_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=1920,
            callback=on_input
        )

        def on_output(out_data, frames, time, status):
            print(frames, type(out_data))
            out_data.fill(0)

        out_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=1920,
                callback=on_output
        )

        with in_stream, out_stream:
            await queue_loop()
            #await asyncio.gather(recv_loop(), queue_loop())


asyncio.run(main())
