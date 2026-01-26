# NVIDIA PersonaPlex Integration

[PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1) is NVIDIA's speech-to-speech foundation model built on top of Moshi. It enables real-time, full-duplex spoken dialogue with customizable personas.

## Overview

- **Model**: `nvidia/personaplex-7b-v1`
- **Base**: Moshi (kyutai/moshiko-pytorch-bf16)
- **Pipeline**: audio-to-audio (speech-to-speech)
- **Parameters**: 7B
- **License**: See NVIDIA model card

## Key Features

- **Full-Duplex Dialogue**: Real-time conversational AI that can listen and speak simultaneously
- **Persona Customization**: Configurable voice and personality characteristics
- **Streaming Audio**: Uses Mimi neural audio codec for low-latency streaming
- **Multi-turn Conversations**: Maintains context across dialogue turns

## Installation

```bash
# Clone this repository
git clone https://github.com/potentiallyai/moshi.git
cd moshi

# Install dependencies
pip install -e .

# For NVIDIA PersonaPlex (requires HuggingFace authentication)
huggingface-cli login
```

## Usage with PersonaPlex

### Basic Inference

```python
from moshi import MoshiClient
from huggingface_hub import hf_hub_download

# Download PersonaPlex weights (requires access approval)
model_path = hf_hub_download(
    repo_id="nvidia/personaplex-7b-v1",
    filename="model.safetensors"
)

# Initialize with PersonaPlex weights
client = MoshiClient(
    model_path=model_path,
    device="cuda"
)

# Start a conversation
async with client.stream() as stream:
    # Send audio input
    await stream.send_audio(audio_bytes)
    
    # Receive audio output
    async for chunk in stream.receive_audio():
        play_audio(chunk)
```

### Persona Configuration

```python
# Configure persona characteristics
persona_config = {
    "voice_style": "warm",
    "speaking_rate": 1.0,
    "personality": "helpful assistant"
}

client = MoshiClient(
    model_path=model_path,
    persona=persona_config
)
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with 24GB+ VRAM (RTX 4090, A100, etc.)
- **CUDA**: 12.0+
- **Memory**: 32GB+ RAM recommended

## Performance

| Metric | Value |
|--------|-------|
| Latency (first token) | ~200ms |
| Real-time factor | 0.8x |
| Supported languages | English |

## Related Resources

- [PersonaPlex Paper](https://arxiv.org/abs/2503.04721)
- [Moshi Documentation](https://github.com/kyutai-labs/moshi)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)

## Citation

```bibtex
@article{personaplex2025,
  title={PersonaPlex: Speech-to-Speech Foundation Models with Persona Control},
  author={NVIDIA Research},
  journal={arXiv preprint arXiv:2503.04721},
  year={2025}
}
```

---

*Documentation by ACE ♠️ for Potentially AI*
