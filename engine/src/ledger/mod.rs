use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::modal::ModalUpdate;

/// Minimal modal ledger storing accepted updates for auditability.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ModalLedger {
    history: VecDeque<ModalUpdate>,
    window: usize,
}

impl ModalLedger {
    pub fn log_commit(&mut self, update: &ModalUpdate) {
        if self.history.len() >= self.window_size() {
            self.history.pop_front();
        }
        self.history.push_back(update.clone());
    }

    pub fn window_size(&self) -> usize {
        if self.window == 0 {
            64
        } else {
            self.window
        }
    }

    pub fn entries(&self) -> impl Iterator<Item = &ModalUpdate> {
        self.history.iter()
    }
}
