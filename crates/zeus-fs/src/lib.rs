//! Filesystem engine, permission gate, file operations, checkpoints, and search.
//!
//! Phase 2 — Safety Core. Every mutating operation routes through the Permission Gate
//! and records a checkpoint entry for undo.

mod checkpoint;
mod codeint;
mod device;
mod diff;
mod error;
mod git;
mod ops;
mod pathutil;
mod permission;
mod platform;
mod search;
mod staleness;
mod tsint;
mod workspace;

pub use checkpoint::{CheckpointStore, CheckpointSummary, FileSnapshot};
pub use codeint::{
    filter_out_own_index, paths_equal, word_boundary, CallEdge, IndexEngine, Symbol, SymbolIndex,
};
pub use device::{DeviceEngine, DeviceOutput};
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
pub use platform::{PlatformEngine, PlatformOutput};
pub use search::{GlobMatch, GrepMatch, SearchEngine, SearchOptions};
pub use workspace::Workspace;
