use nalgebra::{DMatrix, DVector};
use rand::Rng;

use crate::errors::{EngineError, Result};

/// Container holding the thin SVD of a matrix.
#[derive(Debug, Clone)]
pub struct RegularizedSvd {
    pub u: DMatrix<f64>,
    pub s: DVector<f64>,
    pub vt: DMatrix<f64>,
}

/// Compute a regularized SVD discarding singular values below `tol`.
pub fn svd_regularized(matrix: &DMatrix<f64>, tol: f64) -> Result<RegularizedSvd> {
    let svd = matrix.clone().svd(true, true);
    let s = svd.singular_values.clone();
    let mut mask = vec![true; s.len()];
    for (idx, value) in s.iter().enumerate() {
        if *value < tol {
            mask[idx] = false;
        }
    }
    let retained = mask.iter().filter(|&&m| m).count();
    if retained == 0 {
        return Err(EngineError::LinAlg(
            "matrix is numerically rank deficient".into(),
        ));
    }
    let u = svd
        .u
        .ok_or_else(|| EngineError::LinAlg("SVD missing U".into()))?
        .columns(0, retained)
        .into();
    let vt = svd
        .v_t
        .ok_or_else(|| EngineError::LinAlg("SVD missing V^T".into()))?
        .rows(0, retained)
        .into();
    let singular = DVector::from_iterator(
        retained,
        s.iter()
            .zip(mask.iter())
            .filter(|(_, keep)| **keep)
            .map(|(value, _)| *value),
    );
    Ok(RegularizedSvd { u, s: singular, vt })
}

/// Construct the pseudo-inverse of a matrix using its regularized SVD.
pub fn pseudo_inverse(matrix: &DMatrix<f64>, tol: f64) -> Result<DMatrix<f64>> {
    let RegularizedSvd { u, s, vt } = svd_regularized(matrix, tol)?;
    let sigma_inv = DMatrix::from_diagonal(&s.map(|v| if v > tol { 1.0 / v } else { 0.0 }));
    Ok(vt.transpose() * sigma_inv * u.transpose())
}

/// Estimate the trace of `matrix` via Hutchinson's stochastic estimator.
pub fn hutchinson_trace<R>(matrix: &DMatrix<f64>, samples: usize, rng: &mut R) -> f64
where
    R: Rng + ?Sized,
{
    assert!(matrix.nrows() == matrix.ncols(), "expect square matrix");
    let n = matrix.ncols();
    let mut acc = 0.0;
    for _ in 0..samples {
        let z: DVector<f64> = DVector::from_iterator(
            n,
            (0..n).map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 }),
        );
        let mz = matrix * &z;
        acc += mz.dot(&z);
    }
    acc / samples as f64
}
