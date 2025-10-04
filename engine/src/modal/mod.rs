use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::entanglement::EntangledView;

/// Canonical modal update summarising uncertainty collapse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalUpdate {
    pub summary: String,
    pub kl_divergence: f32,
}

impl ModalUpdate {
    pub fn from_view(view: &EntangledView) -> Result<Self> {
        let summary = format!("ledger::{}", view.signal);
        let kl_divergence = (1.0 - view.weight).abs();
        Ok(Self {
            summary,
            kl_divergence,
        })
    }
}
