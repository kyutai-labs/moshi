# moshi

There are three separate versions of the moshi inference stack in this repo.
- The python version using PyTorch is in the `moshi` directory.
- The python version using MLX is in the `moshi_mlx` directory.
- The rust version used in production is in the `rust` directory.

## Python (PyTorch)

The python api can be found in the `moshi` directory. It provides a streaming
version of the audio tokenizer (mimi) and the lm model (moshi).

In order to run in interactive mode, you need to start a server which will
run the model, you can then use either the web UI or a command line client.

Start the server with:
```bash
PYTHONPATH=moshi python -m moshi.server
```

And then access the web UI on [localhost:8998](http://localhost:8998).

If the server is running on a remote box, you may want to forward the 8998 port
via your ssh connection so as to be able to access the web UI locally.

Accessing a server that is not localhost via http may cause issues around using
the microphone in the web UI (in some browsers this is only allowed using
https).

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
PYTHONPATH=moshi_mlx python -m moshi_mlx.local  \
    --model ~/tmp/moshiko_mlx_301e30bf@120.q8.safetensors \
    --mimi ~/tmp/tokenizer-e351c8d8-checkpoint125.safetensors \
    --quantized 8
```

This uses a command line interface, alternatively you can use `local_web` to use
the web UI, connection is via http on [localhost:8998](http://localhost:8998).

## Rust

In order to run the rust inference server, use the following command from within
the `rust` directory:

```bash
cargo run --features cuda --bin moshi-backend -r -- --config moshi-backend/config.json standalone
```

When using macOS, you can replace `--features cuda` with `--features metal`.

Alternatively you can use `config-q8.json` rather than `config.json` to use the
quantified q8 model.

Once the server has printed 'standalone worker listening', you can use the web
UI. By default the rust version uses https so it will be at
[localhost:8998](https://localhost:8998).

You will get some warnings about the site being unsafe. When using chrome you
can bypass it by selecting "Details" or "Advanced", then "Visit this unsafe
site" or "Proceed to localhost (unsafe)".

## Clients

We recommend using the web UI as it provides some echo cancellation that helps
the overall model quality. Alternatively we provide some command line interfaces
for the rust and python versions, the protocol is the same as with the web UI so
there is nothing to change on the server side.

### Rust Command Line

From within the `rust` directory, run the following:
```bash
cargo run --bin moshi-cli -r -- tui --host localhost
```

### Python with PyTorch

```bash
PYTHONPATH=moshi python -m moshi.client
```

### WebUI

The web UI can be built from this repo via the
following steps (these will require `npm` being installed).
```bash
cd client
npm install
npm run build
```

The web UI can then be found in the `client/dist` directory.
