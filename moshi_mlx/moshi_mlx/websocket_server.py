#!/usr/bin/env python3
"""
WebSocket server for Moshi MLX - bridges iOS clients to local Moshi
Run: python -m moshi_mlx.websocket_server -q 4 --port 8998
"""

import argparse
import asyncio
import base64
import json
import logging
import signal
import struct
import sys
from typing import Optional

import numpy as np
import websockets
from websockets.server import serve, WebSocketServerProtocol

# Import Moshi components
from .local import SAMPLE_RATE
from moshi_mlx import models, utils
import huggingface_hub
import mlx.core as mx

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class MoshiWebSocketServer:
    """WebSocket server that connects iOS clients to local Moshi inference"""
    
    def __init__(self, quantized: int = 4, hf_repo: str = "kyutai/moshiko-mlx-q4"):
        self.quantized = quantized
        self.hf_repo = hf_repo
        self.model = None
        self.tokenizer = None
        self.mimi = None
        self.clients: set[WebSocketServerProtocol] = set()
        self.running = False
        
    async def load_model(self):
        """Load Moshi MLX model"""
        logger.info(f"Loading Moshi from {self.hf_repo}...")
        
        # Download model files
        model_path = huggingface_hub.snapshot_download(self.hf_repo)
        
        # Load tokenizer
        import sentencepiece
        tokenizer_path = f"{model_path}/tokenizer_spm_32k_3.model"
        self.tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        
        # Load model weights
        if self.quantized:
            weights_path = f"{model_path}/model.q{self.quantized}.safetensors"
        else:
            weights_path = f"{model_path}/model.safetensors"
        
        logger.info(f"Loading weights from {weights_path}")
        self.model = models.load_model(weights_path)
        
        # Load Mimi codec
        logger.info("Loading Mimi audio codec...")
        import rustymimi
        self.mimi = rustymimi.StreamDecoder(SAMPLE_RATE)
        self.mimi_encoder = rustymimi.StreamEncoder(SAMPLE_RATE)
        
        logger.info("Moshi model loaded and ready")
        
    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a single client connection"""
        client_id = id(websocket)
        logger.info(f"Client {client_id} connected")
        self.clients.add(websocket)
        
        # Send ready message
        await websocket.send(json.dumps({"type": "ready"}))
        
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        finally:
            self.clients.discard(websocket)
            
    async def process_message(self, websocket: WebSocketServerProtocol, message: str):
        """Process incoming message from client"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "audio":
                # Decode base64 audio
                audio_b64 = data.get("data", "")
                audio_bytes = base64.b64decode(audio_b64)
                audio_samples = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Process through Moshi (simplified - actual impl needs full pipeline)
                # For now, echo back with greeting
                await self.process_audio(websocket, audio_samples)
                
            elif msg_type == "start":
                logger.info("Client requested start")
                await websocket.send(json.dumps({"type": "started"}))
                
            elif msg_type == "stop":
                logger.info("Client requested stop")
                await websocket.send(json.dumps({"type": "stopped"}))
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON message")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await websocket.send(json.dumps({"type": "error", "message": str(e)}))
            
    async def process_audio(self, websocket: WebSocketServerProtocol, samples: np.ndarray):
        """Process audio through Moshi and send response"""
        # TODO: Full Moshi inference pipeline
        # For now, this is a placeholder that demonstrates the protocol
        
        # Encode audio to tokens via Mimi
        # tokens = self.mimi_encoder.encode(samples)
        
        # Run through Moshi LM
        # response_tokens = self.model.generate(tokens)
        
        # Decode tokens back to audio
        # response_audio = self.mimi.decode(response_tokens)
        
        # For demonstration, just acknowledge receipt
        # Real implementation would stream audio responses
        pass
        
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if self.clients:
            msg = json.dumps(message)
            await asyncio.gather(
                *[client.send(msg) for client in self.clients],
                return_exceptions=True
            )
            
    async def run(self, host: str = "0.0.0.0", port: int = 8998):
        """Run the WebSocket server"""
        self.running = True
        
        # Load model first
        await self.load_model()
        
        logger.info(f"Starting WebSocket server on ws://{host}:{port}")
        
        async with serve(self.handle_client, host, port):
            # Keep running until stopped
            stop_event = asyncio.Event()
            
            def signal_handler():
                logger.info("Shutdown requested")
                stop_event.set()
                
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, signal_handler)
                
            await stop_event.wait()
            
        logger.info("Server stopped")


async def main():
    parser = argparse.ArgumentParser(description="Moshi WebSocket Server")
    parser.add_argument("-q", "--quantized", type=int, choices=[4, 8], default=4,
                        help="Quantization level (4 or 8)")
    parser.add_argument("--hf-repo", type=str, default="kyutai/moshiko-mlx-q4",
                        help="HuggingFace repo for model")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8998,
                        help="Port to listen on")
    args = parser.parse_args()
    
    # Adjust repo based on quantization
    if args.quantized == 8 and "q4" in args.hf_repo:
        args.hf_repo = args.hf_repo.replace("q4", "q8")
    
    server = MoshiWebSocketServer(
        quantized=args.quantized,
        hf_repo=args.hf_repo
    )
    
    await server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    asyncio.run(main())
