# moshi-inference

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

Once the server is running, you can connect to it either via the web UI or using
the command line interface.

For the CLI, run.
```bash
cargo run --bin cli -r -- tui --host localhost
```
