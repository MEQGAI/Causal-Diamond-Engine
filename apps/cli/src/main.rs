use anyhow::Result;
use clap::{Parser, Subcommand};
use engine::CausalDiamondEngine;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(author, version, about = "Reality's Ledger CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a single reasoning step with a textual prompt
    Step {
        prompt: String,
        #[arg(default_value_t = 1.0)]
        budget: f32,
    },

    /// Display engine banner and diagnostics
    Banner,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Step { prompt, budget } => {
            let mut engine = CausalDiamondEngine::new();
            let output = engine.step(&prompt, budget)?;
            println!("{}", output);
        }
        Command::Banner => {
            println!("{}", engine::banner());
        }
    }
    Ok(())
}
