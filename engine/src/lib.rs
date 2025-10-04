pub mod diamond;
pub mod entanglement;
pub mod geometry;
pub mod ledger;
pub mod modal;
pub mod stability;

use tracing::info;

/// High-level handle that coordinates the causal-diamond engine.
#[derive(Default)]
pub struct CausalDiamondEngine {
    ledger: ledger::ModalLedger,
}

impl CausalDiamondEngine {
    pub fn new() -> Self {
        Self {
            ledger: ledger::ModalLedger::default(),
        }
    }

    pub fn step(&mut self, input: &str, budget: f32) -> anyhow::Result<String> {
        let view = self.entangle(input);
        let update = self.extremize(&view, budget)?;
        let accepted = crate::stability::gate(&update);
        if accepted {
            self.ledger.log_commit(&update);
            Ok(format!("accepted: {}", update.summary))
        } else {
            Ok("rejected: widen diamond".to_string())
        }
    }

    fn entangle(&self, input: &str) -> entanglement::EntangledView {
        entanglement::EntangledView::from_input(input)
    }

    fn extremize(
        &self,
        view: &entanglement::EntangledView,
        budget: f32,
    ) -> anyhow::Result<modal::ModalUpdate> {
        geometry::budget_guard(budget)?;
        modal::ModalUpdate::from_view(view)
    }
}

/// Render a friendly banner used by CLI / server smoke checks.
pub fn banner() -> String {
    let msg = "Reality's Ledger :: Causal-Diamond Engine";
    info!(target: "engine", "{}", msg);
    msg.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_contains_name() {
        assert!(banner().contains("Reality"));
    }
}
