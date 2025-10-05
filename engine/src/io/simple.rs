use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    entanglement::StateSlice,
    geometry::Diamond,
    modal::ModalState,
    runtime::{DiamondBatch, StateBundle},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredBatch {
    pub diamonds: Vec<Diamond>,
    pub entanglement: HashMap<u64, StateSlice>,
    pub modal: HashMap<u64, ModalState>,
}

impl From<StoredBatch> for DiamondBatch {
    fn from(value: StoredBatch) -> Self {
        let mut states = HashMap::new();
        for diamond in &value.diamonds {
            let ent = value
                .entanglement
                .get(&diamond.id)
                .cloned()
                .unwrap_or_default();
            let modal = value.modal.get(&diamond.id).cloned().unwrap_or_default();
            states.insert(
                diamond.id,
                StateBundle {
                    entanglement: ent,
                    modal,
                },
            );
        }
        DiamondBatch {
            diamonds: value.diamonds,
            states,
        }
    }
}

impl From<DiamondBatch> for StoredBatch {
    fn from(batch: DiamondBatch) -> Self {
        let mut ent = HashMap::new();
        let mut modal = HashMap::new();
        for diamond in &batch.diamonds {
            if let Some(state) = batch.states.get(&diamond.id) {
                ent.insert(diamond.id, state.entanglement.clone());
                modal.insert(diamond.id, state.modal.clone());
            }
        }
        StoredBatch {
            diamonds: batch.diamonds,
            entanglement: ent,
            modal,
        }
    }
}
