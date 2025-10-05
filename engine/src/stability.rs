use serde::{Deserialize, Serialize};

use crate::{
    config::StabilityConfig,
    entanglement::EntanglementOutcome,
    geometry::{Diamond, GeometryOutcome, HBAR, NEWTON_G},
    modal::ModalOutcome,
    utils::stencil::second_derivative,
};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StabilityOutcome {
    pub id: u64,
    pub qfc_pass: bool,
    pub theta_quantum: Vec<f64>,
    pub dtheta_dlambda: Vec<f64>,
    pub failures: Vec<String>,
    pub hessian_min: f64,
}

pub fn evaluate(
    diamond: &Diamond,
    geometry: &GeometryOutcome,
    ent: &EntanglementOutcome,
    modal: &ModalOutcome,
    cfg: &StabilityConfig,
) -> StabilityOutcome {
    let lambda = &diamond.generators[0].lambda;
    let classical: Vec<f64> = geometry
        .theta_profile
        .iter()
        .map(|theta| theta / (4.0 * NEWTON_G * HBAR))
        .collect();

    let affine = diamond.meta.affine_half_width.max(1e-6);
    let quantum_offset = (ent.delta_s_ent - modal.s_mod) / affine;
    let theta_quantum: Vec<f64> = classical.iter().map(|c| c + quantum_offset).collect();
    let dtheta_dlambda = second_derivative(lambda, &theta_quantum);

    let mut failures = Vec::new();
    let qfc_pass = dtheta_dlambda.iter().all(|val| *val <= cfg.qfc_tolerance);
    if !qfc_pass {
        failures.push(format!(
            "qfc violated: max dTheta/dlambda {:.3e}",
            dtheta_dlambda
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        ));
    }

    if modal.hessian_min < cfg.hessian_min {
        failures.push(format!(
            "modal hessian below floor: {:.3e} < {:.3e}",
            modal.hessian_min, cfg.hessian_min
        ));
    }

    StabilityOutcome {
        id: diamond.id,
        qfc_pass,
        theta_quantum,
        dtheta_dlambda,
        failures,
        hessian_min: modal.hessian_min,
    }
}
