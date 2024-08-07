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
    Tui {
        #[arg(long)]
        host: String,

        #[arg(long, default_value_t = 9999)]
        port: usize,

        #[arg(long)]
        tui: bool,
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
        Command::Tui { host, port, tui } => {
            if tui {
                tui_logger::init_logger(log::LevelFilter::Trace).unwrap();
                multistream::client_tui::run(host, port).await?
            } else {
                tracing_subscriber::fmt::init();
                multistream::client::run(host, port).await?
            }
        }
    }
    Ok(())
}
