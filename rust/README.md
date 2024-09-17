# moshi - rust

See the [top-level README.md](../README.md) for more information.
This provides the Rust backend and client implementation.


## Requirements

You will need a recent version of the [Rust toolchain](https://rustup.rs/).

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

## Citation

If you use either Mimi or Moshi, please cite the following paper.

```
@article{defossez2024moshi,
    title={Moshi: a speech-text foundation model for real-time dialogue},
    author={Alexandre Défossez and Laurent Mazaré and Manu Orsini and Amélie Royer and Patrick Pérez and Hervé Jégou and Edouard Grave and Neil Zeghidour},
    journal={arXiv:TBC},
    year={2024},
}
```
