use anyhow::Result;
use clap::Parser;

mod audio_io;
mod multistream;

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

        #[arg(long, default_value_t = 9999)]
        port: usize,
    },
    Tui {
        #[arg(long)]
        host: String,

        #[arg(long, default_value_t = 9999)]
        port: usize,
    },
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
    }
    Ok(())
}
