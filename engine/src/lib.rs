//! Causal-diamond engine core implementing the generalized entropy law
//! S = S_geo + S_ent - S_mod with null-stability guarding.

pub mod config;
pub mod entanglement;
pub mod errors;
pub mod frw;
pub mod geometry;
pub mod io;
pub mod modal;
pub mod runtime;
pub mod stability;
pub mod telemetry;
pub mod utils;

pub use config::{
    Concurrency, EngineConfig, Features, IoConfig, Precision, StabilityConfig, TensorBackend,
};
pub use errors::EngineError;
pub use runtime::{CausalDiamondEngine, DiamondBatch, Engine, EngineReport, JobHandle};

/// Emit a textual banner used by CLI integrations and smoke tests.
pub fn banner() -> String {
    const MSG: &str = "Foundation Engine :: Causal-Diamond Runtime";
    tracing::info!(target = "engine", "{}", MSG);
    MSG.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_mentions_engine() {
        let b = banner();
        assert!(b.contains("Engine"));
    }
}
