use serde::{Deserialize, Serialize};

/// Configuration for a causal diamond, representing compute boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiamondConfig {
    pub depth: usize,
    pub width: usize,
    pub horizon: f32,
}

impl Default for DiamondConfig {
    fn default() -> Self {
        Self {
            depth: 8,
            width: 4,
            horizon: 2.0,
        }
    }
}
