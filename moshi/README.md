# Moshi - PyTorch

See the [top-level README.md][main_repo] for more information on Moshi.

[Moshi][moshi] is a speech-text foundation model and full-duplex spoken dialogue framework.
It uses [Mimi][moshi], a state-of-the-art streaming neural audio codec. Mimi operates at 12.5 Hz, and compress
audio down to 1.1 kbps, in a fully streaming manner (latency of 80ms, the frame size), yet performs better than existing, non-streaming, codec.

This is the PyTorch implementation for Moshi and Mimi.


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
