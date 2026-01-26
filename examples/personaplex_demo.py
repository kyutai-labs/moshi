#!/usr/bin/env python3
"""
PersonaPlex Demo - NVIDIA Speech-to-Speech Model

This example demonstrates using NVIDIA's PersonaPlex model for 
real-time speech-to-speech conversations.

Requirements:
- NVIDIA GPU with 24GB+ VRAM
- HuggingFace account with PersonaPlex access
- pip install moshi sounddevice numpy

Usage:
    python personaplex_demo.py
"""

import asyncio
import sounddevice as sd
import numpy as np
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, HfApi
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    exit(1)

# Audio settings
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_DURATION = 0.1  # 100ms chunks

MODEL_ID = "nvidia/personaplex-7b-v1"


def check_model_access():
    """Check if user has access to PersonaPlex model."""
    api = HfApi()
    try:
        api.model_info(MODEL_ID)
        return True
    except Exception as e:
        if "401" in str(e) or "403" in str(e):
            print(f"\n⚠️  Access to {MODEL_ID} is restricted.")
            print("Please request access at: https://huggingface.co/nvidia/personaplex-7b-v1")
            print("And run: huggingface-cli login")
            return False
        raise


def download_model():
    """Download PersonaPlex model weights."""
    print(f"Downloading {MODEL_ID}...")
    
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="model.safetensors",
        cache_dir=Path.home() / ".cache" / "personaplex"
    )
    
    print(f"Model downloaded to: {model_path}")
    return model_path


class PersonaPlexDemo:
    """Simple PersonaPlex demonstration."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.running = False
        
        # Placeholder for actual model loading
        # In production, this would load the Moshi model with PersonaPlex weights
        print(f"Loading PersonaPlex from {model_path}...")
        
    async def start_conversation(self):
        """Start a real-time conversation."""
        print("\n🎤 PersonaPlex Demo")
        print("=" * 40)
        print("Speak into your microphone...")
        print("Press Ctrl+C to stop\n")
        
        self.running = True
        
        # Audio input stream
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            # Process audio chunk
            self.process_audio(indata.copy())
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=audio_callback,
                blocksize=int(SAMPLE_RATE * CHUNK_DURATION)
            ):
                while self.running:
                    await asyncio.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nStopping...")
            self.running = False
    
    def process_audio(self, audio_chunk: np.ndarray):
        """Process incoming audio and generate response."""
        # Placeholder for actual inference
        # In production:
        # 1. Encode audio with Mimi codec
        # 2. Run through PersonaPlex model
        # 3. Decode and play response audio
        
        # Simple voice activity detection placeholder
        volume = np.abs(audio_chunk).mean()
        if volume > 0.01:
            print(f"Audio level: {'█' * int(volume * 50)}")


async def main():
    print("=" * 50)
    print("NVIDIA PersonaPlex - Speech-to-Speech Demo")
    print("=" * 50)
    
    # Check access
    if not check_model_access():
        return
    
    # Download model
    try:
        model_path = download_model()
    except Exception as e:
        print(f"Error downloading model: {e}")
        return
    
    # Run demo
    demo = PersonaPlexDemo(model_path)
    await demo.start_conversation()


if __name__ == "__main__":
    asyncio.run(main())
