use serde::{Deserialize, Serialize};

use crate::{errors::Result, utils::linalg::RegularizedSvd};

pub mod gaussian;
pub mod pointer;

/// Which backend produced the modal deficit evaluation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackendUsed {
    None,
    Gaussian,
    Pointer,
}

impl Default for BackendUsed {
    fn default() -> Self {
        BackendUsed::None
    }
}

/// Gaussian block representation of the modal ledger.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GaussianBlockLedger {
    pub dim_l: usize,
    pub dim_h: usize,
    pub c_ll: Vec<f64>,
    pub c_lh: Vec<f64>,
    pub c_hh: Vec<f64>,
    #[serde(default = "GaussianBlockLedger::default_tol")]
    pub regularization: f64,
}

impl GaussianBlockLedger {
    const fn default_tol() -> f64 {
        1e-10
    }

    pub fn is_valid(&self) -> bool {
        self.c_ll.len() == self.dim_l * self.dim_l
            && self.c_hh.len() == self.dim_h * self.dim_h
            && self.c_lh.len() == self.dim_l * self.dim_h
    }
}

/// Pointer-basis representation via small density matrices.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PointerLedger {
    pub diag: Vec<f64>,
    pub coherences: Vec<f64>,
    pub dimension: usize,
}

impl PointerLedger {
    pub fn is_valid(&self) -> bool {
        self.diag.len() == self.dimension
            && self.coherences.len() == self.dimension * (self.dimension - 1) / 2
    }
}

/// Aggregated modal state per diamond.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModalState {
    pub gaussian: Option<GaussianBlockLedger>,
    pub pointer: Option<PointerLedger>,
    #[serde(default)]
    pub kappa_bar: f64,
}

/// Output bundle from the modal sector.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModalOutcome {
    pub s_mod: f64,
    pub tau_modal: f64,
    pub backend: BackendUsed,
    pub fallback: Option<String>,
    pub hessian_min: f64,
}

impl ModalOutcome {
    pub fn zero() -> Self {
        Self::default()
    }
}

/// Evaluate the modal deficit selecting the best available backend.
pub fn evaluate(state: &ModalState) -> Result<ModalOutcome> {
    if let Some(gaussian) = &state.gaussian {
        match gaussian::evaluate(gaussian, state.kappa_bar) {
            Ok(outcome) => {
                return Ok(ModalOutcome {
                    s_mod: outcome.s_mod,
                    tau_modal: outcome.tau_modal,
                    backend: BackendUsed::Gaussian,
                    fallback: outcome.fallback,
                    hessian_min: outcome.hessian_min,
                })
            }
            Err(err) => {
                if state.pointer.is_none() {
                    return Err(err);
                }
            }
        }
    }

    if let Some(pointer) = &state.pointer {
        let outcome = pointer::evaluate(pointer, state.kappa_bar)?;
        return Ok(ModalOutcome {
            s_mod: outcome.s_mod,
            tau_modal: outcome.tau_modal,
            backend: BackendUsed::Pointer,
            fallback: None,
            hessian_min: outcome.hessian_min,
        });
    }

    Ok(ModalOutcome::zero())
}

/// Helper to convert a dense row-major slice into an nalgebra matrix.
pub(crate) fn matrix_from_row_major(
    data: &[f64],
    rows: usize,
    cols: usize,
) -> nalgebra::DMatrix<f64> {
    nalgebra::DMatrix::from_iterator(rows, cols, data.iter().cloned())
}

/// Rebuild an inverse matrix from a regularised SVD.
pub(crate) fn rebuild_inverse(svd: &RegularizedSvd, tol: f64) -> nalgebra::DMatrix<f64> {
    let RegularizedSvd { u, s, vt } = svd.clone();
    let sigma_inv =
        nalgebra::DMatrix::from_diagonal(
            &s.map(|value| if value > tol { 1.0 / value } else { 0.0 }),
        );
    vt.transpose() * sigma_inv * u.transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_to_zero_when_empty() {
        let state = ModalState::default();
        let out = evaluate(&state).unwrap();
        assert_eq!(out.backend, BackendUsed::None);
        assert_eq!(out.s_mod, 0.0);
    }
}
