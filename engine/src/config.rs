use std::fs;

use serde::{Deserialize, Serialize};

use crate::errors::EngineError;

/// Concurrency strategy for the engine runtime.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Concurrency {
    /// Single-threaded deterministic execution.
    Single,
    /// Rayon work-stealing pool with the specified number of workers.
    Rayon { workers: usize },
}

impl Default for Concurrency {
    fn default() -> Self {
        Self::Rayon {
            workers: num_cpus::get().max(1),
        }
    }
}

/// Numeric precision used for floating-point computations.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    F32,
    F64,
}

impl Default for Precision {
    fn default() -> Self {
        Self::F64
    }
}

/// Tensor backend selection for optional accelerators.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorBackend {
    NdArray,
    Nalgebra,
    #[cfg(feature = "gpu")]
    TchTorch,
    CpuSimd,
}

impl Default for TensorBackend {
    fn default() -> Self {
        Self::NdArray
    }
}

/// Feature toggles controlling optional contributions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Features {
    #[serde(default)]
    pub wald: bool,
    #[serde(default = "Features::default_gaussian")]
    pub gaussian_ledger: bool,
    #[serde(default)]
    pub pointer_ledger: bool,
    #[serde(default = "Features::default_frw")]
    pub frw_map: bool,
    #[serde(default)]
    pub gpu: bool,
}

impl Features {
    fn default_gaussian() -> bool {
        cfg!(feature = "gaussian_ledger")
    }
    fn default_frw() -> bool {
        cfg!(feature = "frw_map")
    }
}

impl Default for Features {
    fn default() -> Self {
        Self {
            wald: false,
            gaussian_ledger: Self::default_gaussian(),
            pointer_ledger: cfg!(feature = "pointer_ledger"),
            frw_map: Self::default_frw(),
            gpu: cfg!(feature = "gpu"),
        }
    }
}

/// Stability thresholds governing QFC/QNEC checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StabilityConfig {
    #[serde(default = "default_qfc_tol")]
    pub qfc_tolerance: f64,
    #[serde(default = "default_hessian_floor")]
    pub hessian_min: f64,
}

const fn default_qfc_tol() -> f64 {
    1e-8
}

const fn default_hessian_floor() -> f64 {
    -1e-6
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            qfc_tolerance: default_qfc_tol(),
            hessian_min: default_hessian_floor(),
        }
    }
}

/// Input/output options for loading diamonds and states.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct IoConfig {
    pub diamonds: Option<String>,
    pub state: Option<String>,
    pub outputs: Option<String>,
}

/// Optional FRW mapping parameters.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FrwConfig {
    pub alpha1: f64,
    pub alpha2: f64,
    #[serde(default = "FrwConfig::default_kappa")]
    pub kappa: f64,
}

impl FrwConfig {
    const fn default_kappa() -> f64 {
        0.2
    }
}

/// Modal ledger calibration constants.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LedgerConfig {
    #[serde(default)]
    pub kappa_bar: f64,
}

/// Engine configuration loaded from TOML/YAML.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineConfig {
    #[serde(default)]
    pub concurrency: Concurrency,
    #[serde(default)]
    pub precision: Precision,
    #[serde(default)]
    pub tensor_backend: TensorBackend,
    #[serde(default)]
    pub features: Features,
    #[serde(default)]
    pub stability: StabilityConfig,
    #[serde(default)]
    pub io: IoConfig,
    pub frw: Option<FrwConfig>,
    pub ledger: Option<LedgerConfig>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            concurrency: Concurrency::default(),
            precision: Precision::default(),
            tensor_backend: TensorBackend::default(),
            features: Features::default(),
            stability: StabilityConfig::default(),
            io: IoConfig::default(),
            frw: None,
            ledger: None,
        }
    }
}

impl EngineConfig {
    /// Load a configuration from a TOML file on disk.
    pub fn from_toml_path<P: AsRef<std::path::Path>>(path: P) -> Result<Self, EngineError> {
        let raw = fs::read_to_string(path)?;
        let cfg: Self = toml::from_str(&raw)?;
        Ok(cfg)
    }

    /// Load a configuration from a YAML file on disk.
    pub fn from_yaml_path<P: AsRef<std::path::Path>>(path: P) -> Result<Self, EngineError> {
        let raw = fs::read_to_string(path)?;
        let cfg: Self = serde_yaml::from_str(&raw)?;
        Ok(cfg)
    }
}
