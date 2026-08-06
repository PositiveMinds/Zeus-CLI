//! Filesystem engine, permission gate, file operations, checkpoints, and search.
//!
//! Phase 2 — Safety Core. Every mutating operation routes through the Permission Gate
//! and records a checkpoint entry for undo.

mod checkpoint;
mod diff;
mod error;
mod git;
mod ops;
mod pathutil;
mod permission;
mod search;
mod staleness;
mod workspace;

pub use checkpoint::{CheckpointStore, CheckpointSummary, FileSnapshot};
pub use diff::preview_diff;
pub use error::{FsError, Result};
pub use git::{GitEngine, GitOutput, ResetMode};
pub use ops::{
    BulkEditPlan, BulkEditResult, CopyOptions, EditOptions, FileEngine, ReadOptions, ReadResult,
    WriteOptions,
};
pub use pathutil::{contains_path, resolve_in_project, PathKind};
pub use permission::{
    ApprovalDecision, PermissionContext, PermissionGate, PermissionRequest, ResolvedPermission,
};
pub use search::{GlobMatch, GrepMatch, SearchEngine, SearchOptions};
pub use workspace::Workspace;
