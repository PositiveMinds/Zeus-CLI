//! Configuration errors.

use std::path::PathBuf;

/// Result alias for config operations.
pub type Result<T> = std::result::Result<T, ConfigError>;

/// Errors that can occur while loading or writing configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to parse TOML at {path}: {source}")]
    TomlParse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("failed to serialize TOML: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    #[error("could not determine home directory")]
    NoHomeDir,

    #[error("invalid configuration: {0}")]
    Invalid(String),
}
