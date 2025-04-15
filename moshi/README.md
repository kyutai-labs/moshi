# Moshi - PyTorch

<a target="_blank" href="https://colab.research.google.com/github//kyutai-labs/moshi/blob/main/moshi/demo_moshi.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

See the [top-level README.md][main_repo] for more information on Moshi.

[Moshi][moshi] is a speech-text foundation model and full-duplex spoken dialogue framework.
It uses [Mimi][moshi], a state-of-the-art streaming neural audio codec. Mimi operates at 12.5 Hz, and compresses
24 kHz audio down to 1.1 kbps, in a fully streaming manner (latency of 80ms, the frame size), yet performs better than existing, non-streaming, codec.

This is the PyTorch implementation for Moshi and Mimi.


## Requirements

You will need at least Python 3.10. We kept a minimal set of dependencies for the current project.
It was tested with PyTorch 2.2 or 2.4. If you need a specific CUDA version, please make sure
to have PyTorch properly installed before installing Moshi.

```bash
pip install -U moshi      # moshi PyTorch, from PyPI
# Or the bleeding edge versions for Moshi
pip install -U -e "git+https://git@github.com/kyutai-labs/moshi#egg=moshi&subdirectory=moshi"
```

While we hope that the present codebase will work on Windows, we do not provide official support for it.
At the moment, we do not support quantization for the PyTorch version, so you will need a GPU with a significant amount of memory (24GB).


## Usage

This package provides a streaming version of the audio tokenizer (Mimi) and the lm model (Moshi).

In order to run in interactive mode, you need to start a server which will
run the model, you can then use either the web UI or a command line client.

Start the server with:
```bash
python -m moshi.server [--gradio-tunnel]
```

And then access the web UI on [localhost:8998](http://localhost:8998). If your GPU is on a distant machine
with no direct access, `--gradio-tunnel` will create a tunnel with a URL accessible from anywhere.
Keep in mind that this tunnel goes through the US and can add significant latency (up to 500ms from Europe).
You can use `--gradio-tunnel-token` to set a fixed secret token and reuse the same address over time.
Alternatively, you might want to use SSH to redirect your connection.

You can use `--hf-repo` to select a different pretrained model, by setting the proper Hugging Face repository.
See [the model list](https://github.com/kyutai-labs/moshi?tab=readme-ov-file#models) for a reference of the available models.

Accessing a server that is not localhost via http may cause issues with using
the microphone in the web UI (in some browsers this is only allowed using
https).

A local client is also available, as
```bash
python -m moshi.client [--url URL_TO_GRADIO]
```
However note, that unlike the web browser, this client is barebone. It does not perform any echo cancellation,
nor does it try to compensate for a growing lag by skipping frames.


## API

You can use programmatically the Mimi/Moshi as follows:
```python
from huggingface_hub import hf_hub_download
import torch

from moshi.models import loaders, LMGen

mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device='cpu')
mimi.set_num_codebooks(8)  # up to 32 for mimi, but limited to 8 for moshi.

# wav should be 24kHz, if not, resample using for instance torchaudio.functional.resample
wav = torch.randn(1, 1, 24000 * 10)  # should be [B, C=1, T]

with torch.no_grad():
    codes = mimi.encode(wav)  # [B, K = 8, T]
    decoded = mimi.decode(codes)

    # Supports streaming too.
    frame_size = mimi.frame_size
    all_codes = []
    with mimi.streaming(batch_size=1):
        for offset in range(0, wav.shape[-1], frame_size):
            frame = wav[:, :, offset: offset + frame_size]
            codes = mimi.encode(frame)
            assert codes.shape[-1] == 1, codes.shape
            all_codes.append(codes)

## WARNING: When streaming, make sure to always feed a total amount of audio that is a multiple
#           of the frame size (1920). You should pad or buffer accordingly. Since version 0.2.5a,
            Mimi no longer supports partial frames in streaming mode. Besides, when executing on GPU,
            you should always pass the same amount of audio, as the calls are CUDAGraphed for efficiency.

# Now if you have a GPU around.
mimi.cuda()
moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
moshi = loaders.get_moshi_lm(moshi_weight, device='cuda')
lm_gen = LMGen(moshi, temp=0.8, temp_text=0.7)  # this handles sampling params etc.
out_wav_chunks = []
# Now we will stream over both Moshi I/O, and decode on the fly with Mimi.
with torch.no_grad(), lm_gen.streaming(1), mimi.streaming(1):
    for idx, code in enumerate(all_codes):
        tokens_out = lm_gen.step(code.cuda())
        # tokens_out is [B, 1 + 8, 1], with tokens_out[:, 1] representing the text token.
        if tokens_out is not None:
            wav_chunk = mimi.decode(tokens_out[:, 1:])
            out_wav_chunks.append(wav_chunk)
        print(idx, end='\r')
out_wav = torch.cat(out_wav_chunks, dim=-1)
```


### Streaming execution mask

It is possible to run on desynchronized batches, e.g. batch for which not all items are coming in
at the same rate. You should set the execution mask on both `lm_gen` and `mimi` to indicate which inputs
are valid for processing, and which should be ignored. While you will still get a value back for the ignored
entries, the internal state will be left unchanged until the next call, e.g.

```
with torch.no_grad(), mimi.streaming(4):
    mask = torch.tensor([False, True, False, True])
    mimi.set_exec_mask(mask)
    frame = torch.randn(4, 1, mimi.frame_size)
    codes = mimi.encode(frame)
    # From the point of view of the first and third entries, nothing has happen.
    # The codes for those two should simply be discarded.
```

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
python scripts/mimi_streaming_test.py

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
@techreport{kyutai2024moshi,
    author = {Alexandre D\'efossez and Laurent Mazar\'e and Manu Orsini and Am\'elie Royer and
			  Patrick P\'erez and Herv\'e J\'egou and Edouard Grave and Neil Zeghidour},
    title = {Moshi: a speech-text foundation model for real-time dialogue},
    institution = {Kyutai},
    year={2024},
    month={September},
    url={http://kyutai.org/Moshi.pdf},
}
```

[moshi]: https://kyutai.org/Moshi.pdf
[main_repo]: https://github.com/kyutai-labs/moshi
