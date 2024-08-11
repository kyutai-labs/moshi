# moshi-inference

There are two versions of the moshi inference stack in this repo. The rust
version used for deployment is in the `rust` directory, the python version
is in the `msh` directory.

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

TODO

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
 
## Python

As of 2024-08-11, the python version does not support streaming. A sample can be
generated via `python basic.py`.
