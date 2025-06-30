// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use clap::Parser;
use std::str::FromStr;

mod audio;
mod benchmark;
mod standalone;
mod stream_both;
mod utils;

#[derive(Parser, Debug)]
#[clap(name = "server", about = "moshi web server")]
struct Args {
    #[clap(short = 'l', long = "log", default_value = "info")]
    log_level: String,

    #[clap(long)]
    config: String,

    #[clap(long)]
    silent: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Parser, Debug)]
struct StandaloneArgs {
    #[clap(long)]
    cpu: bool,
}

#[derive(Clone, Parser, Debug)]
pub struct BenchmarkArgs {
    #[clap(long)]
    cpu: bool,

    #[clap(short = 'n', long, default_value_t = 200)]
    steps: usize,

    #[clap(short = 'n', long, default_value_t = 1)]
    reps: usize,

    #[clap(short = 's', long)]
    stat_file: Option<String>,

    #[clap(long)]
    chrome_tracing: bool,

    #[clap(long)]
    asr: bool,

    #[clap(long)]
    mimi_only: bool,
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    Standalone(StandaloneArgs),
    Benchmark(BenchmarkArgs),
}

/// A TLS acceptor that sets `TCP_NODELAY` on accepted streams.
#[derive(Clone, Debug)]
pub struct NoDelayAcceptor;

impl<S: Send + 'static> axum_server::accept::Accept<tokio::net::TcpStream, S> for NoDelayAcceptor {
    type Stream = tokio::net::TcpStream;
    type Service = S;
    type Future =
        futures_util::future::BoxFuture<'static, std::io::Result<(Self::Stream, Self::Service)>>;

    fn accept(&self, stream: tokio::net::TcpStream, service: S) -> Self::Future {
        Box::pin(async move {
            // Disable Nagle's algorithm.
            stream.set_nodelay(true)?;
            Ok::<_, std::io::Error>((stream, service))
        })
    }
}

fn tracing_init(
    log_dir: &str,
    instance_name: &str,
    log_level: &str,
    silent: bool,
) -> Result<tracing_appender::non_blocking::WorkerGuard> {
    use tracing_subscriber::prelude::*;

    let build_info = utils::BuildInfo::new();
    let file_appender = tracing_appender::rolling::daily(log_dir, format!("log.{instance_name}"));
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    let filter = tracing_subscriber::filter::LevelFilter::from_str(log_level)?;
    let mut layers = vec![tracing_subscriber::fmt::layer()
        .with_writer(non_blocking)
        .with_filter(filter)
        .boxed()];
    if !silent {
        layers.push(Box::new(
            tracing_subscriber::fmt::layer().with_writer(std::io::stdout).with_filter(filter),
        ))
    };
    tracing_subscriber::registry().with(layers).init();
    tracing::info!(?build_info);
    Ok(guard)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Standalone(standalone_args) => {
            let mut config = standalone::Config::load(&args.config)?;
            let _guard = tracing_init(
                &config.stream.log_dir,
                &config.stream.instance_name,
                &args.log_level,
                args.silent,
            )?;
            tracing::info!("starting process with pid {}", std::process::id());

            if config.stream.requires_model_download() {
                standalone::download_from_hub(&mut config.stream).await?;
            }
            if !std::path::PathBuf::from(&config.static_dir).exists() {
                use hf_hub::api::tokio::Api;
                let api = Api::new()?;
                let repo = api.model("kyutai/moshi-artifacts".to_string());
                let dist_tgz = repo.get("dist.tgz").await?;
                if let Some(parent) = dist_tgz.parent() {
                    let dist = parent.join("dist");
                    if !dist.exists() {
                        let output = std::process::Command::new("tar")
                            .arg("-xzf")
                            .arg(&dist_tgz)
                            .arg("-C")
                            .arg(parent)
                            .output()?;
                        if !output.status.success() {
                            anyhow::bail!(
                                "error extract {dist_tgz:?}: {}",
                                String::from_utf8_lossy(&output.stderr)
                            );
                        }
                    }
                    config.static_dir = dist.to_string_lossy().to_string()
                }
            }
            standalone::run(&standalone_args, &config).await?;
        }
        Command::Benchmark(standalone_args) => {
            let config = stream_both::Config::load(&args.config)?;
            let _guard = if standalone_args.chrome_tracing {
                use tracing_chrome::ChromeLayerBuilder;
                use tracing_subscriber::prelude::*;
                let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
                tracing_subscriber::registry().with(chrome_layer).init();
                let b: Box<dyn std::any::Any> = Box::new(guard);
                b
            } else {
                let guard = tracing_init(
                    &config.log_dir,
                    &config.instance_name,
                    &args.log_level,
                    args.silent,
                )?;
                let b: Box<dyn std::any::Any> = Box::new(guard);
                b
            };
            benchmark::run(&standalone_args, &config).await?;
        }
    }
    Ok(())
}
