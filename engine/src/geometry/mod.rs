use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::{
    config::Features,
    utils::{integrate_weighted, trapezoid_with_weights},
};

pub const NEWTON_G: f64 = 6.674_30e-11;
pub const HBAR: f64 = 1.054_571_817e-34;

/// Metadata describing the shape of the diamond patch.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiamondMeta {
    pub affine_half_width: f64,
    pub radius: f64,
    pub curvature: f64,
}

/// Spatial leaf of the diamond (Σ) parameterised by intrinsic coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Screen {
    pub coords: Vec<[f64; 2]>,
    pub sqrt_h: Vec<f64>,
}

impl Screen {
    pub fn samples(&self) -> usize {
        self.coords.len()
    }
}

/// Null generator k^a sampled along an affine parameter λ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NullGen {
    pub lambda: Vec<f64>,
    pub theta: Vec<f64>,
    pub shear2: Vec<f64>,
    pub d_area: Vec<f64>,
    pub stress_kk: Option<Vec<f64>>,
}

impl NullGen {
    pub fn validate(&self) -> bool {
        self.lambda.len() == self.theta.len()
            && self.theta.len() == self.shear2.len()
            && self.shear2.len() == self.d_area.len()
    }
}

/// Discretised diamond containing null generators and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diamond {
    pub id: u64,
    pub leaf: Screen,
    pub generators: Vec<NullGen>,
    #[serde(default)]
    pub meta: DiamondMeta,
}

impl Diamond {
    pub fn validate(&self) -> bool {
        !self.generators.is_empty() && self.generators.iter().all(NullGen::validate)
    }
}

/// Aggregated result for a single diamond from the geometry channel.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeometryOutcome {
    pub id: u64,
    pub delta_area: f64,
    pub s_geo: f64,
    pub mean_expansion: f64,
    pub integrated_shear: f64,
    pub theta_profile: Vec<f64>,
    pub dtheta_dlambda: Vec<f64>,
}

/// Evaluate geometric variations in the small-diamond limit.
pub fn evaluate_diamond(diamond: &Diamond, features: &Features) -> GeometryOutcome {
    debug_assert!(diamond.validate(), "invalid diamond discretisation");
    let mut delta_area = 0.0;
    let mut theta_acc = 0.0;
    let mut shear_acc = 0.0;
    let mut theta_profile = Vec::new();
    let mut dtheta = Vec::new();

    for gen in &diamond.generators {
        let local_delta = trapezoid_with_weights(&gen.lambda, &gen.theta, &gen.d_area);
        delta_area += local_delta;
        theta_acc += gen.theta.iter().sum::<f64>() / gen.theta.len() as f64;
        shear_acc += trapezoid_with_weights(&gen.lambda, &gen.shear2, &gen.d_area);
        if theta_profile.is_empty() {
            theta_profile = gen.theta.clone();
            dtheta = finite_diff(&gen.lambda, &gen.theta);
        }
    }

    let mut s_geo = delta_area / (4.0 * NEWTON_G * HBAR);
    if features.wald {
        // Add extrinsic curvature counterterms at O(lambda^2) with a conservative estimate.
        let extrinsic = 0.5 * diamond.meta.curvature * diamond.meta.affine_half_width.powi(2);
        s_geo += extrinsic;
    }

    GeometryOutcome {
        id: diamond.id,
        delta_area,
        s_geo,
        mean_expansion: theta_acc / diamond.generators.len() as f64,
        integrated_shear: shear_acc,
        theta_profile,
        dtheta_dlambda: dtheta,
    }
}

/// Build a first-law style estimator for θ'(λ).
fn finite_diff(lambda: &[f64], theta: &[f64]) -> Vec<f64> {
    assert_eq!(lambda.len(), theta.len());
    if lambda.len() < 2 {
        return vec![0.0; lambda.len()];
    }
    let mut out = vec![0.0; lambda.len()];
    for i in 0..lambda.len() - 1 {
        let h = lambda[i + 1] - lambda[i];
        let value = if i == 0 {
            (theta[1] - theta[0]) / h
        } else {
            (theta[i + 1] - theta[i - 1]) / (lambda[i + 1] - lambda[i - 1])
        };
        out[i] = value;
    }
    out[lambda.len() - 1] = out[lambda.len() - 2];
    out
}

/// Compute the entanglement first-law kernel ∫ λ T_kk(λ) dλ dA.
pub fn first_law_kernel(gen: &NullGen) -> Option<f64> {
    let stress = gen.stress_kk.as_ref()?;
    let weighted: Vec<f64> = stress.iter().zip(&gen.d_area).map(|(t, w)| t * w).collect();
    Some(integrate_weighted(&gen.lambda, &weighted))
}

/// Accumulate geometric diagnostics across a batch.
#[derive(Debug, Default)]
pub struct GeometryDiagnostics {
    pub total_delta_area: f64,
    pub total_s_geo: f64,
}

pub fn summarise(outcomes: &[GeometryOutcome]) -> GeometryDiagnostics {
    let mut diag = GeometryDiagnostics::default();
    for outcome in outcomes {
        diag.total_delta_area += outcome.delta_area;
        diag.total_s_geo += outcome.s_geo;
    }
    diag
}

/// Construct a default small-diamond generator with quadratic expansion.
pub fn make_small_diamond(id: u64, radius: f64, stress: f64) -> Diamond {
    let lambda: Vec<f64> = (-5..=5).map(|k| k as f64 * 0.05).collect();
    let theta: Vec<f64> = lambda.iter().map(|λ| -λ * stress).collect();
    let shear2: Vec<f64> = lambda.iter().map(|λ| 0.1 * λ.powi(2)).collect();
    let d_area: Vec<f64> = lambda.iter().map(|_| 4.0 * PI * radius.powi(2)).collect();
    let generator = NullGen {
        lambda: lambda.clone(),
        theta,
        shear2,
        d_area,
        stress_kk: Some(lambda.iter().map(|λ| stress * (1.0 + λ)).collect()),
    };
    Diamond {
        id,
        leaf: Screen {
            coords: vec![[0.0, 0.0]],
            sqrt_h: vec![radius.powi(2)],
        },
        generators: vec![generator],
        meta: DiamondMeta {
            affine_half_width: 0.25,
            radius,
            curvature: 0.0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_outcome_nonzero() {
        let diamond = make_small_diamond(1, 1.0, 0.2);
        let features = Features::default();
        let outcome = evaluate_diamond(&diamond, &features);
        assert!(outcome.s_geo.is_finite());
        assert!(!outcome.theta_profile.is_empty());
    }
}
