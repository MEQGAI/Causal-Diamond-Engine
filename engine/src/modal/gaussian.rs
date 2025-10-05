use nalgebra::{self, DMatrix};

use crate::{
    errors::{EngineError, Result},
    utils::linalg::svd_regularized,
};

use super::{matrix_from_row_major, rebuild_inverse, GaussianBlockLedger};

#[derive(Debug, Clone)]
pub struct GaussianOutcome {
    pub s_mod: f64,
    pub tau_modal: f64,
    pub hessian_min: f64,
    pub fallback: Option<String>,
}

pub fn evaluate(ledger: &GaussianBlockLedger, kappa_bar: f64) -> Result<GaussianOutcome> {
    if !ledger.is_valid() {
        return Err(EngineError::Other(
            "invalid gaussian ledger dimensions".into(),
        ));
    }
    let tol = ledger.regularization.max(1e-12);
    let c_ll = matrix_from_row_major(&ledger.c_ll, ledger.dim_l, ledger.dim_l);
    let c_lh = matrix_from_row_major(&ledger.c_lh, ledger.dim_l, ledger.dim_h);
    let c_hh = matrix_from_row_major(&ledger.c_hh, ledger.dim_h, ledger.dim_h);

    let svd_ll = svd_regularized(&c_ll, tol)?;
    let svd_hh = svd_regularized(&c_hh, tol)?;
    let c_ll_inv = rebuild_inverse(&svd_ll, tol);
    let c_hh_inv = rebuild_inverse(&svd_hh, tol);

    let temp = &c_ll_inv * &c_lh;
    let m = &temp * &c_hh_inv * c_lh.transpose();
    let s_mod = 0.5 * m.trace();

    let hessian = build_hessian(&c_ll_inv, &c_lh, &c_hh_inv);
    let sym = 0.5 * (&hessian + hessian.transpose());
    let eigen = nalgebra::SymmetricEigen::new(sym.clone());
    let min_eig = eigen
        .eigenvalues
        .iter()
        .fold(f64::INFINITY, |acc, v| acc.min(*v));

    let tau_modal = 2.0 * kappa_bar * s_mod;
    let fallback = if !min_eig.is_finite() {
        Some("hessian ill-conditioned".to_string())
    } else {
        None
    };

    Ok(GaussianOutcome {
        s_mod,
        tau_modal,
        hessian_min: min_eig,
        fallback,
    })
}

fn build_hessian(
    c_ll_inv: &DMatrix<f64>,
    c_lh: &DMatrix<f64>,
    c_hh_inv: &DMatrix<f64>,
) -> DMatrix<f64> {
    let a = c_lh.transpose() * c_ll_inv * c_lh;
    let b = c_hh_inv.clone();
    a + b
}
