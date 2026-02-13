# Copyright (c) 2024 OpenClaw
# SPDX-License-Identifier: MIT
"""
JSON-RPC bridge for Swift ↔ moshi_mlx communication.

This module provides a stdin/stdout interface for the MoshiMLXBackend.swift
to communicate with the moshi_mlx inference engine.

Protocol:
- Input (stdin): JSON-RPC messages, one per line
  - {"method": "audio", "params": {"data": "<base64>"}}
  - {"method": "stop"}

- Output (stdout): JSON events, one per line
  - {"type": "loading", "progress": 0.5}
  - {"type": "ready"}
  - {"type": "user_text", "text": "hello", "final": false}
  - {"type": "model_text", "text": "hi there"}
  - {"type": "audio", "data": "<base64>"}
  - {"type": "lag"}
  - {"type": "error", "message": "..."}
"""

import argparse
import asyncio
import base64
import json
import multiprocessing
import queue
import signal
import sys
import time
import typing as tp
from enum import Enum

import numpy as np

# Lazy imports for optional dependencies
mx = None
nn = None
sentencepiece = None
rustymimi = None
models = None
utils = None
huggingface_hub = None

SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # 80ms at 24kHz


class BridgeState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class Bridge:
    """JSON-RPC bridge between Swift and moshi_mlx."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state = BridgeState.IDLE
        self.server_process: tp.Optional[multiprocessing.Process] = None
        self.audio_tokenizer = None
        self.client_to_server: tp.Optional[multiprocessing.Queue] = None
        self.server_to_client: tp.Optional[multiprocessing.Queue] = None
        self.printer_q: tp.Optional[multiprocessing.Queue] = None
        self._shutdown = False
        self._ready_event = multiprocessing.Event()
        self._encode_queue: "asyncio.Queue[bytes]" = asyncio.Queue()

    def emit(self, event: dict) -> None:
        """Emit a JSON event to stdout."""
        line = json.dumps(event, ensure_ascii=False)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def emit_error(self, message: str) -> None:
        """Emit an error event."""
        self.emit({"type": "error", "message": message})

    def emit_loading(self, progress: float) -> None:
        """Emit a loading progress event."""
        self.emit({"type": "loading", "progress": progress})

    def emit_ready(self) -> None:
        """Emit ready event."""
        self.emit({"type": "ready"})

    def emit_user_text(self, text: str, final: bool = False) -> None:
        """Emit recognized user text."""
        self.emit({"type": "user_text", "text": text, "final": final})

    def emit_model_text(self, text: str) -> None:
        """Emit model-generated text."""
        self.emit({"type": "model_text", "text": text})

    def emit_audio(self, audio_data: bytes) -> None:
        """Emit audio output as base64."""
        b64 = base64.b64encode(audio_data).decode("ascii")
        self.emit({"type": "audio", "data": b64})

    def emit_lag(self) -> None:
        """Emit lag warning."""
        self.emit({"type": "lag"})

    async def initialize(self) -> None:
        """Initialize the inference engine."""
        global mx, nn, sentencepiece, rustymimi, models, utils, huggingface_hub

        if self.state != BridgeState.IDLE:
            self.emit_error(f"Cannot initialize in state {self.state.value}")
            return

        self.state = BridgeState.LOADING
        self.emit_loading(0.0)

        try:
            # Import heavy dependencies
            self.emit_loading(0.05)
            import mlx.core as _mx
            import mlx.nn as _nn
            import sentencepiece as _sp
            import rustymimi as _rm
            import huggingface_hub as _hf
            from moshi_mlx import models as _models, utils as _utils

            mx = _mx
            nn = _nn
            sentencepiece = _sp
            rustymimi = _rm
            huggingface_hub = _hf
            models = _models
            utils = _utils

            self.emit_loading(0.1)

            # Set up multiprocessing queues
            self.client_to_server = multiprocessing.Queue()
            self.server_to_client = multiprocessing.Queue()
            self.printer_q = multiprocessing.Queue()

            # Start server process
            self.emit_loading(0.15)
            self.server_process = multiprocessing.Process(
                target=_server_process,
                args=(
                    self.printer_q,
                    self.client_to_server,
                    self.server_to_client,
                    self.args,
                    self._ready_event,
                ),
            )
            self.server_process.start()

            # Load mimi tokenizer for client-side audio encoding
            self.emit_loading(0.2)
            mimi_file = self.args.mimi_weight
            if mimi_file is None:
                mimi_file = huggingface_hub.hf_hub_download(
                    self.args.hf_repo, "tokenizer-e351c8d8-checkpoint125.safetensors"
                )
            self.emit_loading(0.25)
            self.audio_tokenizer = rustymimi.StreamTokenizer(mimi_file)

            # Wait for server to be ready (with progress updates)
            while not self._ready_event.is_set():
                # Check for progress messages from server
                try:
                    ty, value = self.printer_q.get_nowait()
                    if ty == "loading":
                        self.emit_loading(0.3 + 0.6 * value)  # 30-90% for model loading
                    elif ty == "info":
                        pass  # Could log to stderr if needed
                except queue.Empty:
                    pass
                await asyncio.sleep(0.05)

            # Warmup
            self.emit_loading(0.95)
            self._warmup()

            self.state = BridgeState.READY
            self.emit_loading(1.0)
            self.emit_ready()

        except Exception as e:
            self.state = BridgeState.ERROR
            self.emit_error(f"Initialization failed: {e}")
            raise

    def _warmup(self) -> None:
        """Warm up the audio tokenizer and model."""
        if self.audio_tokenizer is None:
            return

        for i in range(4):
            pcm_data = np.zeros(FRAME_SIZE, dtype=np.float32)
            self.audio_tokenizer.encode(pcm_data)
            while True:
                time.sleep(0.01)
                data = self.audio_tokenizer.get_encoded()
                if data is not None:
                    break
            self.client_to_server.put_nowait(data)
            if i == 0:
                continue
            audio_tokens = self.server_to_client.get()
            self.audio_tokenizer.decode(audio_tokens)
            while True:
                time.sleep(0.01)
                data = self.audio_tokenizer.get_decoded()
                if data is not None:
                    break

    async def start(self) -> None:
        """Start the audio processing loop."""
        if self.state != BridgeState.READY:
            self.emit_error(f"Cannot start in state {self.state.value}")
            return

        self.state = BridgeState.RUNNING

    async def stop(self) -> None:
        """Stop the inference engine."""
        self._shutdown = True
        self.state = BridgeState.STOPPED

        if self.server_process and self.server_process.is_alive():
            self.server_process.terminate()
            self.server_process.join(timeout=2)
            if self.server_process.is_alive():
                self.server_process.kill()

        self.emit({"type": "ended"})

    async def send_audio(self, base64_data: str) -> None:
        """Process incoming audio data from Swift."""
        if self.state not in (BridgeState.READY, BridgeState.RUNNING):
            return

        if self.state == BridgeState.READY:
            self.state = BridgeState.RUNNING

        try:
            # Decode base64 to float32 PCM
            raw_bytes = base64.b64decode(base64_data)
            pcm_data = np.frombuffer(raw_bytes, dtype=np.float32)

            # Encode audio to tokens
            self.audio_tokenizer.encode(pcm_data)

        except Exception as e:
            self.emit_error(f"Audio processing error: {e}")

    async def process_audio_loop(self) -> None:
        """Background task for processing encoded audio and responses."""
        while not self._shutdown:
            try:
                # Check for encoded audio to send to server
                if self.audio_tokenizer:
                    encoded = self.audio_tokenizer.get_encoded()
                    if encoded is not None and self.client_to_server:
                        self.client_to_server.put_nowait(encoded)

                    # Check for decoded audio from server
                    decoded = self.audio_tokenizer.get_decoded()
                    if decoded is not None:
                        # Convert to bytes and emit
                        audio_bytes = decoded.astype(np.float32).tobytes()
                        self.emit_audio(audio_bytes)

                # Check for audio tokens from server
                if self.server_to_client:
                    try:
                        audio_tokens = self.server_to_client.get_nowait()
                        if self.audio_tokenizer:
                            self.audio_tokenizer.decode(audio_tokens)
                    except queue.Empty:
                        pass

                # Check for printer queue messages (text, lag, etc.)
                if self.printer_q:
                    try:
                        ty, value = self.printer_q.get_nowait()
                        if ty == "token":
                            self.emit_model_text(value)
                        elif ty == "lag":
                            self.emit_lag()
                        elif ty == "error":
                            self.emit_error(value)
                    except queue.Empty:
                        pass

                await asyncio.sleep(0.001)

            except Exception as e:
                self.emit_error(f"Audio loop error: {e}")
                await asyncio.sleep(0.1)

    def get_status(self) -> dict:
        """Get current bridge status."""
        return {
            "type": "status",
            "state": self.state.value,
            "server_alive": self.server_process.is_alive() if self.server_process else False,
        }

    async def handle_message(self, message: dict) -> None:
        """Handle an incoming JSON-RPC message."""
        method = message.get("method", "")
        params = message.get("params", {})

        if method == "initialize":
            await self.initialize()
        elif method == "start":
            await self.start()
        elif method == "stop":
            await self.stop()
        elif method == "audio":
            data = params.get("data", "")
            await self.send_audio(data)
        elif method == "get_status":
            self.emit(self.get_status())
        else:
            self.emit_error(f"Unknown method: {method}")

    async def run(self) -> None:
        """Main event loop: read stdin, process messages."""
        # Start background audio processing task
        audio_task = asyncio.create_task(self.process_audio_loop())

        try:
            loop = asyncio.get_event_loop()
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            await loop.connect_read_pipe(lambda: protocol, sys.stdin)

            while not self._shutdown:
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=0.1)
                    if not line:
                        break  # EOF
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue

                    try:
                        message = json.loads(line)
                        await self.handle_message(message)
                    except json.JSONDecodeError as e:
                        self.emit_error(f"Invalid JSON: {e}")

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.emit_error(f"Read error: {e}")
                    break

        finally:
            audio_task.cancel()
            try:
                await audio_task
            except asyncio.CancelledError:
                pass


def _server_process(
    printer_q: multiprocessing.Queue,
    client_to_server: multiprocessing.Queue,
    server_to_client: multiprocessing.Queue,
    args: argparse.Namespace,
    ready_event: multiprocessing.Event,
) -> None:
    """Server process that runs the LM inference."""
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import sentencepiece
    import huggingface_hub
    from moshi_mlx import models, utils

    def hf_hub_download(repo, path: str) -> str:
        if repo is None or repo == "":
            raise ValueError(f"the --hf-repo flag is required to retrieve {path}")
        return huggingface_hub.hf_hub_download(repo, path)

    def log(msg: str) -> None:
        printer_q.put_nowait(("info", msg))

    def emit_progress(progress: float) -> None:
        printer_q.put_nowait(("loading", progress))

    try:
        model_file = args.moshi_weight
        tokenizer_file = args.tokenizer
        
        emit_progress(0.1)
        
        if model_file is None:
            if args.quantized == 8:
                model_file = hf_hub_download(args.hf_repo, "model.q8.safetensors")
            elif args.quantized == 4:
                model_file = hf_hub_download(args.hf_repo, "model.q4.safetensors")
            elif args.quantized is not None:
                raise ValueError(f"Invalid quantized value: {args.quantized}")
            else:
                model_file = hf_hub_download(args.hf_repo, "model.safetensors")
        
        emit_progress(0.3)
        
        if tokenizer_file is None:
            tokenizer_file = hf_hub_download(args.hf_repo, "tokenizer_spm_32k_3.model")

        emit_progress(0.4)
        log(f"[SERVER] loading text tokenizer {tokenizer_file}")
        text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)
        
        emit_progress(0.5)
        mx.random.seed(299792458)
        lm_config = models.config_v0_1()
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)
        
        if args.quantized is not None:
            group_size = 32 if args.quantized == 4 else 64
            nn.quantize(model, bits=args.quantized, group_size=group_size)

        emit_progress(0.6)
        log(f"[SERVER] loading weights {model_file}")
        model.load_weights(model_file, strict=True)
        
        emit_progress(0.8)
        log("[SERVER] weights loaded")
        model.warmup()
        
        emit_progress(0.9)
        log("[SERVER] model warmed up")

        gen = models.LmGen(
            model=model,
            max_steps=args.steps + 5,
            text_sampler=utils.Sampler(),
            audio_sampler=utils.Sampler(),
            check=False,
        )

        emit_progress(1.0)
        ready_event.set()
        log("[SERVER] ready!")

        # Main inference loop
        while True:
            try:
                data = client_to_server.get(timeout=1.0)
            except queue.Empty:
                continue

            data = mx.array(data).transpose(1, 0)[:, :8]
            text_token = gen.step(data)
            text_token = text_token[0].item()
            audio_tokens = gen.last_audio_tokens()

            if text_token not in (0, 3):
                text = text_tokenizer.id_to_piece(text_token)
                text = text.replace("▁", " ")
                printer_q.put_nowait(("token", text))

            if audio_tokens is not None:
                audio_tokens = np.array(audio_tokens).astype(np.uint32)
                server_to_client.put_nowait(audio_tokens)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        printer_q.put_nowait(("error", str(e)))


def main():
    parser = argparse.ArgumentParser(
        description="JSON-RPC bridge for moshi_mlx"
    )
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer model")
    parser.add_argument("--moshi-weight", type=str, help="Path to Moshi model weights")
    parser.add_argument("--mimi-weight", type=str, help="Path to Mimi tokenizer weights")
    parser.add_argument(
        "-q", "--quantized", type=int, choices=[4, 8],
        help="Quantization bits (4 or 8)"
    )
    parser.add_argument(
        "--steps", default=4000, type=int,
        help="Maximum inference steps"
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace repository for model weights"
    )

    args = parser.parse_args()

    # Set default HF repo based on quantization
    if args.hf_repo is None:
        if args.quantized == 8:
            args.hf_repo = "kyutai/moshiko-mlx-q8"
        elif args.quantized == 4:
            args.hf_repo = "kyutai/moshiko-mlx-q4"
        else:
            args.hf_repo = "kyutai/moshiko-mlx-bf16"

    # Handle SIGTERM gracefully
    def handle_sigterm(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    # Create and run bridge
    bridge = Bridge(args)

    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        bridge.emit_error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
