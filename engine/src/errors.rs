use thiserror::Error;

/// Unified error type for the engine crate.
#[derive(Debug, Error)]
pub enum EngineError {
    /// Wrapper around I/O errors.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Serialization or deserialization failures.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    /// YAML parsing error.
    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    /// TOML parsing error.
    #[error("toml error: {0}")]
    Toml(#[from] toml::de::Error),
    /// Linear algebra problems such as non-PSD matrices.
    #[error("linear algebra error: {0}")]
    LinAlg(String),
    /// Stability rule violation.
    #[error("stability violation: {0}")]
    Stability(String),
    /// Feature not compiled or available at runtime.
    #[error("feature not available: {0}")]
    Feature(String),
    /// Any other context dependent failure.
    #[error("{0}")]
    Other(String),
}

impl EngineError {
    pub fn other<T: Into<String>>(msg: T) -> Self {
        Self::Other(msg.into())
    }
}

pub type Result<T, E = EngineError> = std::result::Result<T, E>;
