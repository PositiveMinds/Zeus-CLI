//! Agent-loop, tool-dispatch, and terminal execution errors.

pub type Result<T> = std::result::Result<T, AgentError>;

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("provider error: {0}")]
    Provider(#[from] zeus_provider::ProviderError),

    #[error("filesystem error: {0}")]
    Fs(#[from] zeus_fs::FsError),

    #[error("terminal error: {0}")]
    Terminal(String),

    #[error("unknown tool: {0}")]
    UnknownTool(String),

    #[error("invalid arguments for tool '{tool}': {reason}")]
    InvalidArguments { tool: String, reason: String },

    #[error("session error: {0}")]
    Session(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
