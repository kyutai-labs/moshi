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

You can eithr compile and install the `rustymimi` extension or install it via
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
## Rust

The rust inference code uses a client-server infrastructure.

In order to run the inference server in standalone mode, run the following
command from within the `rust` directory.

- Start the server.
```bash
cargo run --features cuda --bin moshi-backend -r -- --config moshi-backend/config.json standalone
```
When using macOS, you can replace `--features cuda` with `--features metal`.

Alternatively you can use `config-q8.json` rather than `config.json` to use the
quantified q8 model.

Once the server has printed 'standalone worker listening', you can connect to it
either via the web UI or using the command line interface. Multiple sessions can
be run one after another without having to restart the server.

### Command Line

For the CLI, run.
```bash
cargo run --bin moshi-cli -r -- tui --host localhost
```

### WebUI

The web UI can be used as an alternative to the CLI. In order to do so, run the
following steps (these will require `npm` being installed).
```bash
cd client
npm install
npm run build
```

Then run the server in the same way mentioned above and from your web browser
navigate to `https://127.0.0.1:8080/?worker_addr=127.0.0.1:8080`. You will get
some warnings about the site being unsafe. When using chrome you can bypass it
by selecting "Details" or "Advanced", then "Visit this unsafe site" or "Proceed
to localhost (unsafe)".

