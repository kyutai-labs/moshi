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
import aiohttp

from .client_utils import AnyPrinter, Printer, RawPrinter


class Connection:
    def __init__(
        self,
        printer: AnyPrinter,
        websocket: aiohttp.ClientWebSocketResponse,
        sample_rate: float = 24000,
        channels: int = 1,
        frame_size: int = 1920,
    ) -> None:
        self.printer = printer
        self.websocket = websocket
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.channels = channels

        self._done = False
        self._in_stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            blocksize=self.frame_size,
            callback=self._on_audio_input,
        )

        self._out_stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            blocksize=frame_size,
            callback=self._on_audio_output,
        )
        self._opus_writer = sphn.OpusStreamWriter(sample_rate)
        self._opus_reader = sphn.OpusStreamReader(sample_rate)
        self._output_queue = queue.Queue()

    async def _queue_loop(self) -> None:
        while True:
            if self._done:
                return
            await asyncio.sleep(0.001)
            msg = self._opus_writer.read_bytes()
            if len(msg) > 0:
                try:
                    await self.websocket.send_bytes(b"\x01" + msg)
                except Exception as e:
                    print(e)
                    self._lost_connection()
                    return

    async def _decoder_loop(self) -> None:
        all_pcm_data = None
        while True:
            if self._done:
                return
            await asyncio.sleep(0.001)
            pcm = self._opus_reader.read_pcm()
            if all_pcm_data is None:
                all_pcm_data = pcm
            else:
                all_pcm_data = np.concatenate((all_pcm_data, pcm))
            while all_pcm_data.shape[-1] >= self.frame_size:
                self._output_queue.put(all_pcm_data[: self.frame_size])
                all_pcm_data = np.array(all_pcm_data[self.frame_size :])

    async def _recv_loop(self) -> None:
        try:
            async for message in self.websocket:
                if message.type == aiohttp.WSMsgType.CLOSED:
                    self.printer.log("info", "Connection closed")
                    break
                elif message.type == aiohttp.WSMsgType.ERROR:
                    self.printer.log("error", f"{self.websocket.exception()}")
                    break
                elif message.type != aiohttp.WSMsgType.BINARY:
                    self.printer.log("error", f"received from server: {message.type}")
                    continue
                message = message.data
                if not isinstance(message, bytes):
                    self.printer.log(
                        "warning", f"unsupported message type {type(message)}"
                    )
                    continue
                if len(message) == 0:
                    self.printer.log("warning", "empty message")
                    continue
                kind = message[0]
                if kind == 1:  # audio
                    payload = message[1:]
                    self._opus_reader.append_bytes(payload)
                    self.printer.print_pending()
                elif kind == 2:  # text
                    payload = message[1:]
                    self.printer.print_token(payload.decode())
                else:
                    self.printer.log("warning", f"unknown message kind {kind}")
        except Exception as e:
            print(e)
            self._lost_connection()
            return

    def _lost_connection(self) -> None:
        if not self._done:
            self.printer.log("error", "Lost connection with the server!")
            self._done = True

    def _on_audio_input(self, in_data, frames, time_, status) -> None:
        assert in_data.shape == (self.frame_size, self.channels), in_data.shape
        self._opus_writer.append_pcm(in_data[:, 0])

    def _on_audio_output(self, out_data, frames, time_, status) -> None:
        assert out_data.shape == (self.frame_size, self.channels), out_data.shape
        try:
            pcm_data = self._output_queue.get(block=False)
            # TODO: handle other shapes by using some form of fifo/ring buffer.
            assert pcm_data.shape == (self.frame_size,), pcm_data.shape
            out_data[:, 0] = pcm_data
        except queue.Empty:
            out_data.fill(0)
            self.printer.print_lag()

    async def run(self) -> None:
        with self._in_stream, self._out_stream:
            await asyncio.gather(
                self._recv_loop(), self._decoder_loop(), self._queue_loop()
            )


async def run(printer: AnyPrinter, args):
    uri = f"ws://{args.host}:{args.port}/api/chat"
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(uri) as ws:
            printer.log("info", "connected!")
            printer.print_header()
            connection = Connection(printer, ws)
            await connection.run()


def main():
    parser = argparse.ArgumentParser("client_opus")
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    args = parser.parse_args()
    printer: AnyPrinter

    if sys.stdout.isatty():
        printer = Printer()
    else:
        printer = RawPrinter()
    try:
        asyncio.run(run(printer, args))
    except KeyboardInterrupt:
        printer.log("warning", "Interrupting, exiting connection.")
    printer.log("info", "All done!")


if __name__ == "__main__":
    main()
