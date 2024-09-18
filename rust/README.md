# moshi - rust

See the [top-level README.md](../README.md) for more information.

This provides the Rust backend (both Mimi and Moshi) and client implementation.
The Mimi implementation is available through Python bindings, through the  `rustymimi` package.

## Requirements

You will need a recent version of the [Rust toolchain](https://rustup.rs/).

## Rust based Mimi with Python bindings

First, a standalone rust based implementation of Mimi is provided, along with Python bindings.
This is the one used by `moshi_mlx`. It is automatically installed with `moshi_mlx`, but you
can install it separately as
```bash
# Install from pip:
pip install rustymimi==0.1.1
# Alternatively, if you want to compile the package run from the root of the repo.
maturin dev -r -m rust/mimi-pyo3/Cargo.toml
```

## Rust server

In order to run the rust inference server, use the following command from within
the this directory:

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

## Rust client

We recommend using the web UI as it provides some echo cancellation that helps
the overall model quality. Alternatively we provide some command line interfaces
for the rust and python versions, the protocol is the same as with the web UI so
there is nothing to change on the server side.

### Rust Command Line

From within the `rust` directory, run the following:
```bash
cargo run --bin moshi-cli -r -- tui --host localhost
```

## License

The present code is provided under the Apache license.
