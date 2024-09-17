# Moshi - MLX

See the [top-level README.md](../README.md) for more information.
This provides the Rust backend and client implementation.

[Moshi][moshi] is a speech-text foundation model and full-duplex spoken dialogue framework.
It uses [Mimi][moshi], a state-of-the-art streaming neural audio codec. Mimi operates at 12.5 Hz, and compress
audio down to 1.1 kbps, in a fully streaming manner (latency of 80ms, the frame size),
yet performs better than existing, non-streaming, codec like
[SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer) (50 Hz, 4 kbps), or [SemantiCodec](https://github.com/haoheliu/SemantiCodec-inference) (50 Hz, 1kbps).

Moshi models **two streams of audio**: one corresponds to Moshi, and one to the user.
At inference, the one from the user is taken from the audio input,
and the one for Moshi is sampled from. Along that, Moshi predicts text tokens corresponding to its own speech
which greatly improves the quality of its generation. A small depth transformer models inter codebook dependencies for a given step,
while a large, 7B parameters, Transformer models the temporal dependencies. Moshi achieves a theoretical latency
of 160ms (80ms for the frame size of Mimi + 80ms of acoustic delay), with a practical overall latency as low as 200ms.
[Talk to Moshi](https://moshi.chat) now on our live demo.

<p align="center">
<img src="./moshi.png" alt="Schema representing the structure Moshi. Moshi models two streams of audio:
    one corresponds to Moshi, and one to the user. At inference, the one from the user is taken from the audio input,
    and the one for Moshi is sampled from. Along that, Moshi predicts text tokens corresponding to its own speech
    for improved accuracy. A small depth transformer models inter codebook dependencies for a given step."
width="800px"></p>

Mimi is builds on previous neural audio codecs such as [SoundStream](https://arxiv.org/abs/2107.03312)
and [EnCodec](https://github.com/facebookresearch/encodec), adding a Transformer both in the encoder and decoder,
and adapting the strides to match an overall frame rate of 12.5 Hz. This allows to get closer to the
average frame rate of text tokens (~3-4 Hz), and limit the number of auto-regressive step in Moshi.
Similarly to SpeechTokenizer, Mimi uses a distillation loss so that the first codebook tokens match
a self-supervised representation from [WavLM](https://arxiv.org/abs/2110.13900). Interestingly, while
Mimi is fully causal and streaming, it learns to match sufficiently well the non causal representation from WavLM,
without introducing any delays. Finally, and similary to [EBEN](https://arxiv.org/pdf/2210.14090), Mimi
uses only an adversarial training loss, along with feature matching, showing strong improvements in terms of subjective quality
despite its low bitrate.

<p align="center">
<img src="./mimi.png" alt="Schema representing the structure Moshi. Moshi models two streams of audio:
    one corresponds to Moshi, and one to the user. At inference, the one from the user is taken from the audio input,
    and the one for Moshi is sampled from. Along that, Moshi predicts text tokens corresponding to its own speech
    for improved accuracy. A small depth transformer models inter codebook dependencies for a given step."
width="800px"></p>


## Requirements

You will need at least Python 3.10.

```bash
pip install moshi_mlx  # moshi MLX, from PyPI
# Or the bleeding edge versions for Moshi and Moshi-MLX.
pip install -e "git+https://git@github.com/kyutai-labs/moshi#egg=moshi_mlx&subdirectory=moshi_mlx"
```
We have tested the MLX version with MacBook Pro M3.


## Development

If you wish to install from a clone of this repository, maybe to further develop Moshi, you can do the following:
```
# From the current folder (e.g. `moshi_mlx/`)
pip install -e '.[dev]'
pre-commit install
```

## Python (MLX) for local inference on macOS

You can either compile and install the `rustymimi` extension or install it via
pip.
```bash
# Install from pip:
pip install rustymimi==0.1.1
# Alternatively, if you want to compile the package run:
maturin dev -r -m rust/mimi-pyo3/Cargo.toml
```

Then the model can be run with:
```bash
python -m moshi_mlx.local -q 4   # weights quantized to 4 bits
python -m moshi_mlx.local -q 8   # weights quantized to 8 bits
```

This uses a command line interface, which is bare bone. It doesn't do any echo cancellation,
nor does it try to compensate for a growing lag by skipping frames.

Alternatively you can use `python -m moshi_mlx.local_web` to use
the web UI, connection is via http on [localhost:8998](http://localhost:8998).


## License

The present code is provided under the MIT license.

## Citation

If you use either Mimi or Moshi, please cite the following paper,

```
@article{defossez2024moshi,
    title={Moshi: a speech-text foundation model for real-time dialogue},
    author={Alexandre Défossez and Laurent Mazaré and Manu Orsini and Amélie Royer and Patrick Pérez and Hervé Jégou and Edouard Grave and Neil Zeghidour},
    journal={arXiv:TBC},
    year={2024},
}
```
