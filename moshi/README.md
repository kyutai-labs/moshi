# Moshi - PyTorch

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

You will need at least Python 3.10. We kept a minimal set of dependencies for the current project.
It was tested with PyTorch 2.2 or 2.4. If you need a specific CUDA version, please make sure
to have PyTorch properly installed before installing Moshi.

```bash
pip install moshi      # moshi PyTorch, from PyPI
# Or the bleeding edge versions for Moshi
pip install -e "git+https://git@github.com/kyutai-labs/moshi#egg=moshi&subdirectory=moshi"
```

While we hope that the present codebase will work on Windows, we do not provide official support for it.
At the moment, we do not support quantization for the PyTorch version, so you will need a GPU with a significant amount of memory (24GB).


## Usage

This package provides a streaming version of the audio tokenizer (Mimi) and the lm model (Moshi).

In order to run in interactive mode, you need to start a server which will
run the model, you can then use either the web UI or a command line client.

Start the server with:
```bash
python -m moshi.server [--gradio_tunnel]
```

And then access the web UI on [localhost:8998](http://localhost:8998). If your GPU is on a distant machine
with no direct access, `--gradio_tunnel` will create a tunnel with a URL accessible from anywhere.
Keep in mind that this tunnel goes through the US and can add significant latency (up to 500ms from Europe).
Alternatively, you might want to use SSH to redirect your connection.

Accessing a server that is not localhost via http may cause issues around using
the microphone in the web UI (in some browsers this is only allowed using
https).

A local client is also available, as
```bash
python -m moshi.client [--url URL_TO_GRADIO]
```
However note, that unlike the web browser, this client is bare bone. It doesn't do any echo cancellation,
nor does it try to compensate for a growing lag by skipping frames.

## Development

If you wish to install from a clone of this repository, maybe to further develop Moshi, you can do the following:
```bash
# From the current folder (e.g. `moshi/`)
pip install -e '.[dev]'
pre-commit install
```

Once locally installed, Mimi can be tested with the following command, from **the root** of the repository,
```bash
wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
python scripts/mimi_test.py

```

Similary, Moshi can be tested (with a GPU) with
```bash
python scripts/moshi_benchmark.py
```


## License

The present code is provided under the MIT license.
Note that parts of this code is based on [AudioCraft](https://github.com/facebookresearch/audiocraft), released under
the MIT license.

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

[moshi]: https://arxiv.org/
