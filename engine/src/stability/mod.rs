use crate::modal::ModalUpdate;

/// Null-stability gate placeholder that ensures KL is bounded.
pub fn gate(update: &ModalUpdate) -> bool {
    update.kl_divergence <= 1.0
}
