import argparse
import asyncio
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
    async with websockets.connect(uri) as websocket:

        async def on_input(in_data, frames, time, status):
            print(type(in_data), type(frames), time, status)
            await websocket.send(b"\1hello, server!")

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, callback=on_input
        ):
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


asyncio.run(main())
