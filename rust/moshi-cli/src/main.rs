// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use clap::Parser;

mod audio_io;
mod gen;
mod multistream;

use candle::Device;

#[derive(Debug, Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    Client {
        #[arg(long)]
        host: String,

        #[arg(long, default_value_t = 8998)]
        port: usize,
    },
    Tui {
        #[arg(long)]
        host: String,

        #[arg(long, default_value_t = 8998)]
        port: usize,
    },
    Gen {
        #[arg(long)]
        lm_model_file: String,

        #[arg(long)]
        mimi_model_file: String,

        #[arg(long)]
        lm_config_file: String,

        #[arg(long)]
        text_tokenizer: String,

        #[arg(long)]
        audio_input_file: String,

        #[arg(long)]
        audio_output_file: String,

        #[arg(long, default_value_t = 299_792_458)]
        seed: u64,

        #[arg(long)]
        cfg_alpha: Option<f64>,

        /// Run on cpu
        #[arg(long)]
        cpu: bool,
    },
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 10)]
async fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    match args.command {
        Command::Client { host, port } => {
            tracing_subscriber::fmt::init();
            multistream::client::run(host, port).await?
        }
        Command::Tui { host, port } => {
            tracing_subscriber::fmt::init();
            multistream::client_tui::run(host, port).await?
        }
        Command::Gen {
            seed,
            text_tokenizer,
            lm_model_file,
            lm_config_file,
            mimi_model_file,
            audio_input_file,
            audio_output_file,
            cfg_alpha,
            cpu,
        } => {
            let dev = device(cpu)?;
            tracing_subscriber::fmt::init();
            let args = gen::Args {
                lm_model_file,
                mimi_model_file,
                text_tokenizer,
                lm_config_file,
                audio_input_file,
                audio_output_file,
                seed,
                cfg_alpha,
            };
            gen::run(&args, &dev)?
        }
    }
    Ok(())
}
