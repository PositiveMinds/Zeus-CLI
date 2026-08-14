//! Provider errors.

/// Result alias for provider operations.
pub type Result<T> = std::result::Result<T, ProviderError>;

/// Errors from model providers.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("provider not found: {0}")]
    NotFound(String),

    #[error("unsupported provider kind: {0}")]
    UnsupportedKind(String),

    #[error("provider not configured: {0}")]
    NotConfigured(String),

    #[error("missing API key (set env {0})")]
    MissingApiKey(String),

    #[error("request cancelled")]
    Cancelled,

    #[error("embeddings not supported by this provider")]
    EmbeddingsUnsupported,

    #[error("HTTP / transport error: {0}")]
    Transport(String),

    /// A non-success HTTP response from a provider endpoint.
    #[error("API error (HTTP {status}): {message}")]
    Http { status: u16, message: String },

    #[error("API error: {0}")]
    Api(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl ProviderError {
    /// The HTTP status code behind the error, if it came from a provider
    /// endpoint (used by retry / user-facing diagnostics).
    pub fn http_status(&self) -> Option<u16> {
        match self {
            ProviderError::Http { status, .. } => Some(*status),
            _ => None,
        }
    }
}
