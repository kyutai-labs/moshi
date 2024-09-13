# moshi

There are three separate versions of the moshi inference stack in this repo.
- The rust version used in production is in the `rust` directory.
- The python version using PyTorch is in the `msh` directory.
- The python version using MLX is in the `msh_mlx` directory.

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

## Python (PyTorch)

The python api can be found in the `msh` directory. It provides a streaming
version of the audio tokenizer (mimi) and the lm model (moshi).

In order to run in interactive mode, you need to start a server which will
run the model, and a client that captures the sound from the microphone
and passes it to the server, get some data back from the server and plays it
on the speakers.

The client and server do not have to run on the same machine, the protocol used
to transfer data between the client and the server should be compatible with the
rust version.

Start the server with:
```bash
python server_opus.py \
    --mimi-weights tokenizer-e351c8d8-checkpoint125.safetensors \
    --tokenizer tokenizer_spm_32k_3.model \
    --moshi-weights moshiko_pt_301e30bf@120.safetensors
```

And then starts the client with:
```bash
python client_opus.py
```

When running on different machine, you can add the command line argument
`--host 0.0.0.0` to the server so that it accepts remote connections and
the argument `--host 192.168.0.42` to the client where `192.168.0.42` is
the ip of the server. The default port is `9998` and can be overriden with
`--port`.

## Python (MLX) for local inference on macOS

You can eithr compile and install the `rustymimi` extension or install it via
pip.
```bash
# Install from pip:
pip install rustymimi==0.1.1
# In order to compile the thing, run:
maturin dev -r -m rust/mimi-pyo3/Cargo.toml
```

Then the model can be run with:
```bash
PYTHONPATH=. python local_mlx.py  \
    --model ~/tmp/moshiko_mlx_301e30bf@120.q8.safetensors \
    --mimi ~/tmp/tokenizer-e351c8d8-checkpoint125.safetensors \
    --quantized 8
```
