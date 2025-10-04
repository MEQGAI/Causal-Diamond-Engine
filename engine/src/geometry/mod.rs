use anyhow::{anyhow, Result};

/// Verifies that a requested budget respects coarse stability heuristics.
pub fn budget_guard(budget: f32) -> Result<()> {
    if !(0.0..=10.0).contains(&budget) {
        return Err(anyhow!("budget {budget} outside [0, 10]"));
    }
    Ok(())
}
