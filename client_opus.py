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


def colorize(text, color):
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def make_log(level: str, msg: str) -> str:
    if level == "warning":
        prefix = colorize("Warning:", "1;31")
    elif level == "info":
        prefix = colorize("Info:", "1;34")
    elif level == "error":
        prefix = colorize("Error:", "1;31")
    else:
        raise ValueError(f"Unknown level {level}")
    return prefix + ' ' + msg


class RawPrinter:
    def __init__(self, stream=sys.stdout, err_stream=sys.stderr):
        self.stream = stream
        self.err_stream = err_stream

    def print_header(self):
        pass

    def print_token(self, token: str):
        self.stream.write(token)
        self.stream.flush()

    def log(self, level: str, msg: str):
        print(f"{level.capitalize()}: {msg}", file=self.err_stream)

    def print_lag(self):
        self.err_stream.write(colorize(' [LAG]', '31'))
        self.err_stream.flush()

    def print_pending(self):
        pass


class Printer:
    def __init__(self, max_cols: int = 80, stream=sys.stdout, err_stream=sys.stderr):
        self.max_cols = max_cols
        self.current_line: str = ''
        self.stream = stream
        self.err_stream = err_stream
        self._pending_count = 0
        self._pending_printed = False

    def print_header(self):
        print('-' * self.max_cols)

    def _remove_pending(self) -> bool:
        if self._pending_printed:
            self._pending_printed = False
            self.stream.write('\r' + self.current_line + ' ' * 2)
            return True
        return False

    def print_token(self, token: str):
        self._remove_pending()
        if len(self.current_line) + len(token) <= self.max_cols:
            self.stream.write(token)
            self.current_line += token
        else:
            if token.startswith(' '):
                token = token.lstrip()
                self.stream.write('\n' + token)
                self.current_line = token
            else:
                if ' ' in self.current_line:
                    line, word = self.current_line.rsplit(' ', 1)
                    token = word.lstrip() + token
                    sys.stdout.write('\r' + line + ' ' * (len(word) + 1) + '\n' + token)
                    self.current_line = token
                else:
                    remaining = self.max_cols - len(self.current_line)
                    sys.stdout.write(token[:remaining] + '\n' + token[remaining:])
                    self.current_line = token[remaining:]
        self.stream.flush()

    def log(self, level: str, msg: str):
        msg = make_log(level, msg)
        removed = self._remove_pending()
        if self.current_line:
            self.stream.write('\n')
        elif removed:
            self.stream.write('\t')
        self.stream.flush()
        print(msg, file=self.err_stream)
        self.err_stream.flush()

    def print_lag(self):
        self.print_token(colorize(' [LAG]', '31'))

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


AnyPrinter = Printer | RawPrinter


class Connection:
    def __init__(self, printer: AnyPrinter,
                 websocket: websockets.WebSocketClientProtocol,
                 sample_rate: float = 24000, channels: int = 1, frame_size: int = 1920) -> None:
        self.printer = printer
        self.websocket = websocket
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.channels = channels

        self._done = False
        self._in_stream = sd.InputStream(
            samplerate=sample_rate, channels=channels,
            blocksize=self.frame_size, callback=self._on_audio_input)

        self._out_stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            blocksize=frame_size,
            callback=self._on_audio_output)
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
                    await self.websocket.send(b"\x01" + msg)
                except websockets.WebSocketException:
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
                self._output_queue.put(all_pcm_data[:self.frame_size])
                all_pcm_data = np.array(all_pcm_data[self.frame_size:])

    async def _recv_loop(self) -> None:
        while True:
            try:
                message = await self.websocket.recv()
            except websockets.exceptions.WebSocketException:
                self._lost_connection()
                return
            if not isinstance(message, bytes):
                self.printer.log("warning", f"unsupported message type {type(message)}")
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
            await asyncio.gather(self._recv_loop(), self._decoder_loop(), self._queue_loop())


async def do_connection(printer: AnyPrinter, uri: str, action):
    try:
        async with websockets.connect(uri) as websocket:
            printer.log("info", "connected!")
            printer.print_header()
            await action(websocket)
    except websockets.WebSocketException:
        printer.log("error", "Failed to connect!")
        sys.exit(1)


async def run(printer: AnyPrinter, args):
    async def action(websocket: websockets.WebSocketClientProtocol):
        connection = Connection(printer, websocket)
        await connection.run()

    uri = f"ws://{args.host}:{args.port}"
    await do_connection(printer, uri, action)


def main():
    parser = argparse.ArgumentParser('client_opus')
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

if __name__ == '__main__':
    main()
