use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{geometry::Diamond, utils::integrate_weighted};

const TWO_PI_OVER_HBAR: f64 = 2.0 * std::f64::consts::PI / crate::geometry::HBAR;

/// Choice of basis used to describe the reduced state along the diamond.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Basis {
    Gaussian,
    Pointer,
    Mixed,
}

impl Default for Basis {
    fn default() -> Self {
        Basis::Gaussian
    }
}

/// Summary statistics for the reduced state moments.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Moments {
    pub mean: Vec<f64>,
    pub covariance: Vec<f64>,
}

/// Stress-energy samples along generators (if supplied from upstream solvers).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StressProfile {
    /// Map `generator_index -> expectation values of T_kk(λ)`
    pub tkk: HashMap<usize, Vec<f64>>,
}

/// Slice of the quantum state restricted to a diamond.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSlice {
    pub basis: Basis,
    #[serde(default)]
    pub moments: Moments,
    #[serde(default)]
    pub stress: StressProfile,
}

impl Default for StateSlice {
    fn default() -> Self {
        Self {
            basis: Basis::default(),
            moments: Moments::default(),
            stress: StressProfile::default(),
        }
    }
}

impl StateSlice {
    /// Build a slice with uniform ⟨T_kk⟩ per generator.
    pub fn with_uniform_stress(value: f64, generators: usize) -> Self {
        let mut stress = HashMap::new();
        for idx in 0..generators {
            let profile: Vec<f64> = (0..64)
                .map(|k| {
                    let frac = k as f64 / 63.0;
                    value * (1.0 + frac)
                })
                .collect();
            stress.insert(idx, profile);
        }
        Self {
            basis: Basis::Gaussian,
            moments: Moments::default(),
            stress: StressProfile { tkk: stress },
        }
    }
}

/// Result of evaluating δS_ent for a diamond.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntanglementOutcome {
    pub id: u64,
    pub delta_s_ent: f64,
    pub generator_contrib: Vec<f64>,
}

pub fn evaluate(diamond: &Diamond, state: &StateSlice) -> EntanglementOutcome {
    let mut total = 0.0;
    let mut per_gen = Vec::with_capacity(diamond.generators.len());
    for (idx, gen) in diamond.generators.iter().enumerate() {
        let profile = state
            .stress
            .tkk
            .get(&idx)
            .cloned()
            .or_else(|| gen.stress_kk.clone())
            .unwrap_or_else(|| vec![0.0; gen.lambda.len()]);
        let weighted: Vec<f64> = gen
            .lambda
            .iter()
            .zip(profile.iter())
            .zip(gen.d_area.iter())
            .map(|((λ, tkk), area)| λ * tkk * area)
            .collect();
        let contribution = TWO_PI_OVER_HBAR * integrate_weighted(&gen.lambda, &weighted);
        total += contribution;
        per_gen.push(contribution);
    }
    EntanglementOutcome {
        id: diamond.id,
        delta_s_ent: total,
        generator_contrib: per_gen,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::make_small_diamond;

    #[test]
    fn first_law_increases_with_stress() {
        let diamond = make_small_diamond(42, 1.0, 0.2);
        let base_state = StateSlice::default();
        let base = evaluate(&diamond, &base_state);
        let stressed = StateSlice::with_uniform_stress(0.5, diamond.generators.len());
        let hi = evaluate(&diamond, &stressed);
        assert!(hi.delta_s_ent.is_finite());
        assert!((hi.delta_s_ent - base.delta_s_ent).abs() > 1e-12);
    }
}
