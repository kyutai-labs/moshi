# moshi-inference

There are three separate versions of the moshi inference stack in this repo.
- The rust version used in production is in the `rust` directory.
- The python version using PyTorch is in the `msh` directory.
- The python version using MLX is in the `msh_mlx` directory.

## Rust

The rust inference code uses a client-server infrastructure.

In order to run the inference server in standalone mode, run the following steps
from the `rust` directory.

- Generate an ssl certificate (`key.pem` and `cert.pem`).
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem
```
- Start the server.
```bash
cargo run --features cuda --bin web-ui-backend -r -- --config web-ui-backend/config.json standalone
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

### Protocol

The connection takes place using a websocket. This handles the message lengths
for us. The binary protocol for messages is as follows. The protocol uses little
endian encoding.

Each message starts by a single byte indicating the message type `MT`.
The format for the rest of the message, aka the payload, depends on `MT`.

```
- Handshake MT=0. The payload is made of two fields.
    1. Protocol version (`u32`) - always 0 for now.
    2. Model version (`u32`).
- Audio MT=1. The payload is made of a single field.
  - Binary data for the ogg frames containing opus encoded audio (24kHz, mono).
- Text MT=2. The payload is made of a single field.
  - UTF8 encoded string.
- Control MT=3. The payload is made of a single field. This is not used in full
  streaming mode.
  - One byte B describing the control itself.
    - Start B=0.
    - EndTurn B=1.
    - Pause B=2.
    - Restart B=3.
- MetaData MT=4. The payload is made of a single field.
  - UTF8 encoded string with json data.
- Error MT=5. The payload is made of a single field.
  - UTF8 encoded string containing the error description.
- Ping MT=6. No payload, this message type is currently unused.
```
Messages with an unknow message types should be discarded.
 
## Python (PyTorch)

The python api can be found in the `msh` directory. It provides streaming
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
    --mimi-weights tokenizer-de0e421d-checkpoint40.safetensors \
    --tokenizer tokenizer_spm_32k_3.model \
    --moshi-weights mimi_0abbed5f@100.safetensors 
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

### Testing
In order to test the audio tokenizer, you can run the following command.

```bash
wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
PYTHONPATH=. python scripts/mimi_test.py --weights .../tokenizer-de0e421d-checkpoint40.safetensors
```

In order to test moshi, run the following.
```bash
PYTHONPATH=. python scripts/moshi_test.py \
    --mimi-weights tokenizer-de0e421d-checkpoint40.safetensors \
    --tokenizer tokenizer_spm_32k_3.model \
    --moshi-weights mimi_0abbed5f@100.safetensors 
```

## Python (MLX) for local inference on macOS

You first have to compile and install the mimi extension.
```bash
maturin dev -r -m rust/mimi-pyo3/Cargo.toml
```

Then the model can be run with:
```bash
PYTHONPATH=. python local_mlx.py  \
    --model ~/tmp/mimi_mlx_0abbed5f@100.q8.safetensors \
    --mimi ~/tmp/tokenizer-de0e421d-checkpoint40.safetensors \
    --quantized 8
```
