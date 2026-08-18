//! Filesystem / permission errors.

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, FsError>;

#[derive(Debug, thiserror::Error)]
pub enum FsError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("I/O error: {0}")]
    IoSimple(#[from] std::io::Error),

    #[error("path escapes project root: {0}")]
    PathEscape(PathBuf),

    #[error("permission denied: {0}")]
    Denied(String),

    #[error("permission requires user approval: {0}")]
    NeedsApproval(String),

    #[error("file not found: {0}")]
    NotFound(PathBuf),

    #[error("ambiguous edit match ({count} occurrences); refine the pattern or use replace_all")]
    AmbiguousEdit { count: usize },

    #[error("edit pattern not found in {0}")]
    EditNotFound(PathBuf),

    #[error("stale file (changed on disk since last read): {0}")]
    Stale(PathBuf),

    #[error("must read file before writing/editing existing path: {0}")]
    MustReadFirst(PathBuf),

    #[error("binary file refused: {0} (a plain read can't decode this — if it's an image use read_image; if it's a PDF/DOCX/XLSX/PPTX/HTML document use read_document)")]
    BinaryFile(PathBuf),

    #[error("invalid path: {0}")]
    InvalidPath(String),

    #[error("checkpoint error: {0}")]
    Checkpoint(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

impl FsError {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
