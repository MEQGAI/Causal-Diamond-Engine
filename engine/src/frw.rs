use serde::{Deserialize, Serialize};

use crate::config::FrwConfig;

/// Linear response of the background running vacuum sector.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FrwResponse {
    pub rho_l: f64,
    pub stability_satisfied: bool,
}

pub fn evaluate(config: &FrwConfig, h: f64, dh_dt: f64) -> FrwResponse {
    let rho_l =
        crate::geometry::NEWTON_G.recip() * (config.alpha1 * h.powi(2) + config.alpha2 * dh_dt);
    let stability = config.alpha1 + config.kappa * config.alpha2 >= 0.0;
    FrwResponse {
        rho_l,
        stability_satisfied: stability,
    }
}
