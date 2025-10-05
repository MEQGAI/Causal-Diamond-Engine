use crate::errors::{EngineError, Result};

use super::PointerLedger;

#[derive(Debug, Clone)]
pub struct PointerOutcome {
    pub s_mod: f64,
    pub tau_modal: f64,
    pub hessian_min: f64,
}

pub fn evaluate(ledger: &PointerLedger, kappa_bar: f64) -> Result<PointerOutcome> {
    if !ledger.is_valid() {
        return Err(EngineError::Other(
            "invalid pointer ledger dimensions".into(),
        ));
    }
    let eps = 1e-12;
    let norm: f64 = ledger.diag.iter().copied().sum::<f64>().max(eps);
    let probs: Vec<f64> = ledger.diag.iter().map(|p| (p / norm).max(eps)).collect();
    let s_e = shannon(&probs);

    let coherence_penalty: f64 = ledger
        .coherences
        .iter()
        .map(|c| c.abs().powi(2))
        .sum::<f64>();
    let s_rho = (s_e - coherence_penalty).max(0.0);

    let s_mod = kappa_bar * (s_e - s_rho);
    let hessian_min = -coherence_penalty;
    let tau_modal = 2.0 * s_mod;

    Ok(PointerOutcome {
        s_mod,
        tau_modal,
        hessian_min,
    })
}

fn shannon(probs: &[f64]) -> f64 {
    -probs
        .iter()
        .map(|p| if *p <= 0.0 { 0.0 } else { p * p.ln() })
        .sum::<f64>()
}
