//! File operations: read/write/edit/delete/rename/copy/move/bulk-edit.

use crate::checkpoint::CheckpointStore;
use crate::diff::preview_diff;
use crate::error::{FsError, Result};
use crate::pathutil::{display_rel, normalize_lexically, path_kind, resolve_in_project, PathKind};
use crate::permission::{ApprovalDecision, PermissionGate, PermissionRequest};
use crate::staleness::{is_fresh, stamp_file, ReadTracker};
use std::path::{Path, PathBuf};
use tracing::info;

const BULK_PREVIEW_CAP: usize = 20;
const DELETE_DIR_PREVIEW_CAP: usize = 20;

#[derive(Debug, Clone)]
pub struct ReadOptions {
    pub offset: Option<usize>,
    pub limit: Option<usize>,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            offset: None,
            limit: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReadResult {
    pub path: PathBuf,
    pub content: String,
    pub total_lines: usize,
    pub start_line: usize,
}

#[derive(Debug, Clone)]
pub struct WriteOptions {
    /// When true, skip "must read first" for brand-new files only.
    pub allow_create: bool,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self { allow_create: true }
    }
}

#[derive(Debug, Clone)]
pub struct EditOptions {
    pub old_string: String,
    pub new_string: String,
    pub replace_all: bool,
}

#[derive(Debug, Clone)]
pub struct CopyOptions {
    pub overwrite: bool,
}

#[derive(Debug, Clone)]
pub struct BulkEditPlan {
    pub files: Vec<PathBuf>,
    pub old_string: String,
    pub new_string: String,
    pub replace_all: bool,
}

#[derive(Debug, Clone)]
pub struct BulkEditResult {
    pub modified: Vec<PathBuf>,
    pub skipped: Vec<(PathBuf, String)>,
}

/// High-level file engine bound to a project root + permission gate + checkpoints.
pub struct FileEngine {
    pub project_root: PathBuf,
    pub gate: PermissionGate,
    pub checkpoints: CheckpointStore,
    pub reads: ReadTracker,
    /// Active turn id for checkpoint grouping.
    pub turn_id: String,
    /// Paths written this session (write asks less aggressively for re-touches).
    session_touched: std::sync::Mutex<std::collections::HashSet<PathBuf>>,
}

impl FileEngine {
    pub fn new(
        project_root: PathBuf,
        gate: PermissionGate,
        checkpoints: CheckpointStore,
        turn_id: impl Into<String>,
    ) -> Self {
        Self {
            project_root,
            gate,
            checkpoints,
            reads: ReadTracker::new(),
            turn_id: turn_id.into(),
            session_touched: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }

    pub fn set_turn(&mut self, turn_id: impl Into<String>) {
        self.turn_id = turn_id.into();
        let _ = self.checkpoints.begin_turn(&self.turn_id);
    }

    fn resolve(&self, user_path: &Path) -> Result<PathBuf> {
        resolve_in_project(&self.project_root, user_path)
    }

    fn mark_touched(&self, path: &Path) {
        self.session_touched
            .lock()
            .unwrap()
            .insert(path.to_path_buf());
    }

    fn was_touched(&self, path: &Path) -> bool {
        self.session_touched.lock().unwrap().contains(path)
    }

    /// Read a text file with optional line offset/limit. Line numbers are 1-based in output.
    pub fn read(&self, user_path: &Path, opts: ReadOptions) -> Result<ReadResult> {
        let path = self.resolve(user_path)?;
        self.gate.enforce_strict(&PermissionRequest {
            tool: "read".into(),
            path: Some(path.clone()),
            command: None,
            description: format!("read {}", display_rel(&self.project_root, &path)),
            ..Default::default()
        })?;

        if !path.exists() {
            return Err(FsError::NotFound(path));
        }
        if path.is_dir() {
            return Err(FsError::InvalidPath(format!(
                "is a directory: {}",
                path.display()
            )));
        }

        let bytes = std::fs::read(&path).map_err(|e| FsError::io(&path, e))?;
        if looks_binary(&bytes) {
            return Err(FsError::BinaryFile(path));
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let lines: Vec<&str> = text.lines().collect();
        let total_lines = lines.len();
        let start = opts.offset.unwrap_or(0).min(total_lines);
        let end = opts
            .limit
            .map(|l| (start + l).min(total_lines))
            .unwrap_or(total_lines);
        let mut out = String::new();
        for (i, line) in lines[start..end].iter().enumerate() {
            // 1-based line numbers
            out.push_str(&format!("{:>6}→{}\n", start + i + 1, line));
        }

        let stamp = stamp_file(&path).map_err(|e| FsError::io(&path, e))?;
        self.reads.record(&path, stamp);

        Ok(ReadResult {
            path,
            content: out,
            total_lines,
            start_line: start + 1,
        })
    }

    pub fn write<F>(
        &self,
        user_path: &Path,
        content: &str,
        opts: WriteOptions,
        approver: F,
    ) -> Result<()>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let path = self.resolve(user_path)?;
        let exists = path.exists();

        if exists && !self.reads.has_read(&path) && !self.was_touched(&path) {
            return Err(FsError::MustReadFirst(path));
        }
        if exists {
            if let Some(stamp) = self.reads.get(&path) {
                if !is_fresh(&path, &stamp) {
                    return Err(FsError::Stale(path));
                }
            }
        }
        if !exists && !opts.allow_create {
            return Err(FsError::NotFound(path));
        }

        let mut approver = approver;
        // Ask outside files already touched this session.
        let desc = if exists {
            format!(
                "overwrite {}",
                display_rel(&self.project_root, &path)
            )
        } else {
            format!("create {}", display_rel(&self.project_root, &path))
        };
        // Best-effort diff preview: skip silently if the existing file can't
        // be read as text (binary, permissions) rather than blocking the op.
        let preview = if exists {
            std::fs::read_to_string(&path)
                .ok()
                .map(|old| preview_diff(&old, content))
        } else {
            Some(preview_diff("", content))
        };
        self.gate.enforce(
            &PermissionRequest {
                tool: "write".into(),
                path: Some(path.clone()),
                command: None,
                description: desc,
                preview,
                ..Default::default()
            },
            &mut approver,
        )?;

        self.checkpoints
            .snapshot_before(&self.turn_id, &path, &self.project_root)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| FsError::io(parent, e))?;
        }
        std::fs::write(&path, content).map_err(|e| FsError::io(&path, e))?;
        let stamp = stamp_file(&path).map_err(|e| FsError::io(&path, e))?;
        self.reads.record(&path, stamp);
        self.mark_touched(&path);
        info!(path = %path.display(), "write ok");
        Ok(())
    }

    pub fn edit<F>(&self, user_path: &Path, opts: EditOptions, approver: F) -> Result<usize>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let path = self.resolve(user_path)?;
        if !path.exists() {
            return Err(FsError::NotFound(path));
        }
        if !self.reads.has_read(&path) {
            return Err(FsError::MustReadFirst(path));
        }
        if let Some(stamp) = self.reads.get(&path) {
            if !is_fresh(&path, &stamp) {
                return Err(FsError::Stale(path));
            }
        }

        let original = std::fs::read_to_string(&path).map_err(|e| FsError::io(&path, e))?;
        let count = original.matches(&opts.old_string).count();
        if count == 0 {
            return Err(FsError::EditNotFound(path));
        }
        if count > 1 && !opts.replace_all {
            return Err(FsError::AmbiguousEdit { count });
        }

        let new_content = if opts.replace_all {
            original.replace(&opts.old_string, &opts.new_string)
        } else {
            original.replacen(&opts.old_string, &opts.new_string, 1)
        };

        let mut approver = approver;
        self.gate.enforce(
            &PermissionRequest {
                tool: "edit".into(),
                path: Some(path.clone()),
                command: None,
                description: format!(
                    "edit {} ({} occurrence(s))",
                    display_rel(&self.project_root, &path),
                    count
                ),
                preview: Some(preview_diff(&original, &new_content)),
                ..Default::default()
            },
            &mut approver,
        )?;

        self.checkpoints
            .snapshot_before(&self.turn_id, &path, &self.project_root)?;
        std::fs::write(&path, &new_content).map_err(|e| FsError::io(&path, e))?;
        let stamp = stamp_file(&path).map_err(|e| FsError::io(&path, e))?;
        self.reads.record(&path, stamp);
        self.mark_touched(&path);
        Ok(count)
    }

    pub fn delete<F>(&self, user_path: &Path, approver: F) -> Result<()>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let path = self.resolve(user_path)?;
        if !path.exists() {
            return Err(FsError::NotFound(path));
        }

        let (desc, preview) = if path.is_dir() {
            let paths = list_files_recursive(&path, &self.project_root, DELETE_DIR_PREVIEW_CAP);
            let desc = format!(
                "delete directory {} ({} files)",
                display_rel(&self.project_root, &path),
                paths.total
            );
            (desc, Some(paths.render()))
        } else {
            let desc = format!("delete {}", display_rel(&self.project_root, &path));
            let preview = std::fs::read_to_string(&path)
                .ok()
                .map(|content| preview_diff(&content, ""));
            (desc, preview)
        };

        let mut approver = approver;
        self.gate.enforce(
            &PermissionRequest {
                tool: "delete".into(),
                path: Some(path.clone()),
                command: None,
                description: desc,
                preview,
                ..Default::default()
            },
            &mut approver,
        )?;

        if path.is_file() {
            self.checkpoints
                .snapshot_before(&self.turn_id, &path, &self.project_root)?;
            std::fs::remove_file(&path).map_err(|e| FsError::io(&path, e))?;
        } else {
            // Snapshot each file under the directory.
            for entry in walkdir::WalkDir::new(&path)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                self.checkpoints.snapshot_before(
                    &self.turn_id,
                    entry.path(),
                    &self.project_root,
                )?;
            }
            std::fs::remove_dir_all(&path).map_err(|e| FsError::io(&path, e))?;
        }
        Ok(())
    }

    pub fn rename<F>(&self, from: &Path, to: &Path, approver: F) -> Result<()>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let from_p = self.resolve(from)?;
        if !from_p.exists() {
            return Err(FsError::NotFound(from_p));
        }

        // Compute the lexical (non-canonicalized) destination first. On a
        // case-insensitive filesystem, `resolve_in_project` canonicalizing an
        // existing-but-differently-cased destination collapses it back to
        // the real on-disk casing (e.g. requesting "Foo.TXT" resolves to the
        // existing "foo.txt") — which would erase a pure case change before
        // we ever see it. Detect that case here, on the lexical form, before
        // it's lost. Joined against the *canonicalized* project root (not
        // the raw one) so its prefix style matches `from_p` exactly — on
        // Windows `canonicalize()` adds a `\\?\` extended-path prefix that a
        // plain join never produces, and comparing across that mismatch
        // would make even a real case-only path look unequal for reasons
        // that have nothing to do with case.
        let canonical_root =
            std::fs::canonicalize(&self.project_root).unwrap_or_else(|_| self.project_root.clone());
        let to_lexical = normalize_lexically(&if to.is_absolute() {
            to.to_path_buf()
        } else {
            canonical_root.join(to)
        });
        let case_only = to_lexical != from_p && paths_equal_ignoring_case(&from_p, &to_lexical);

        // For the case-only path, the lexical target already denotes the
        // exact same (contained, already-resolved) filesystem entry as
        // `from_p`, just with different casing — so it's safe to use
        // directly without re-running path containment. Otherwise resolve
        // `to` normally (may or may not exist).
        let to_p = if case_only {
            to_lexical
        } else {
            self.resolve(to)?
        };

        let overwrites = !case_only && to_p.exists();
        let mut approver = approver;
        self.gate.enforce(
            &PermissionRequest {
                tool: "rename".into(),
                path: Some(from_p.clone()),
                command: None,
                description: format!(
                    "rename {} → {}{}",
                    display_rel(&self.project_root, &from_p),
                    display_rel(&self.project_root, &to_p),
                    if overwrites {
                        " (overwrites existing file)"
                    } else {
                        ""
                    }
                ),
                overwrites,
                ..Default::default()
            },
            &mut approver,
        )?;
        self.checkpoints
            .snapshot_rename(&self.turn_id, &from_p, &to_p, &self.project_root)?;
        if let Some(parent) = to_p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| FsError::io(parent, e))?;
        }

        // Case-only rename (e.g. "Foo.txt" -> "foo.txt"): a direct rename can
        // silently no-op since the OS sees both paths as the same entry.
        // Force it via a temporary intermediate name, bypassing git mv
        // (which has the same well-known issue on case-insensitive filesystems).
        if case_only {
            let tmp_name = format!(
                ".zeus-case-rename-{}",
                to_p.file_name().and_then(|n| n.to_str()).unwrap_or("tmp")
            );
            let tmp = to_p.with_file_name(tmp_name);
            std::fs::rename(&from_p, &tmp).map_err(|e| FsError::io(&from_p, e))?;
            std::fs::rename(&tmp, &to_p).map_err(|e| FsError::io(&tmp, e))?;
            return Ok(());
        }

        // Prefer git mv when in a git repo.
        if self.project_root.join(".git").exists() {
            let status = std::process::Command::new("git")
                .args(["mv"])
                .arg(&from_p)
                .arg(&to_p)
                .current_dir(&self.project_root)
                .status();
            if let Ok(s) = status {
                if s.success() {
                    return Ok(());
                }
            }
        }

        if let Err(e) = std::fs::rename(&from_p, &to_p) {
            if is_cross_device_error(&e) {
                // Plain rename() cannot move across filesystems/devices;
                // fall back to copy the tree then remove the source.
                copy_recursive(&from_p, &to_p).map_err(|e| FsError::io(&from_p, e))?;
                if from_p.is_dir() {
                    std::fs::remove_dir_all(&from_p).map_err(|e| FsError::io(&from_p, e))?;
                } else {
                    std::fs::remove_file(&from_p).map_err(|e| FsError::io(&from_p, e))?;
                }
                return Ok(());
            }
            return Err(FsError::io(&from_p, e));
        }
        Ok(())
    }

    pub fn copy<F>(&self, from: &Path, to: &Path, opts: CopyOptions, approver: F) -> Result<()>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let from_p = self.resolve(from)?;
        let to_p = self.resolve(to)?;
        if !from_p.exists() {
            return Err(FsError::NotFound(from_p));
        }
        let overwrites = to_p.exists();
        if overwrites && !opts.overwrite {
            return Err(FsError::Other(format!(
                "destination exists (overwrite=false): {}",
                to_p.display()
            )));
        }
        let is_symlink = std::fs::symlink_metadata(&from_p)
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false);
        let desc = format!(
            "copy {} → {}{}",
            display_rel(&self.project_root, &from_p),
            display_rel(&self.project_root, &to_p),
            if is_symlink {
                " (source is a symlink; contents will be copied, not the link itself)"
            } else {
                ""
            }
        );
        // Best-effort diff preview when overwriting a readable text file.
        let preview = if overwrites {
            match (
                std::fs::read_to_string(&to_p),
                std::fs::read_to_string(&from_p),
            ) {
                (Ok(old), Ok(new)) => Some(preview_diff(&old, &new)),
                _ => None,
            }
        } else {
            None
        };
        let mut approver = approver;
        self.gate.enforce(
            &PermissionRequest {
                tool: "copy".into(),
                path: Some(to_p.clone()),
                command: None,
                description: desc,
                preview,
                overwrites,
            },
            &mut approver,
        )?;
        self.checkpoints
            .snapshot_before(&self.turn_id, &to_p, &self.project_root)?;
        if let Some(parent) = to_p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| FsError::io(parent, e))?;
        }
        std::fs::copy(&from_p, &to_p).map_err(|e| FsError::io(&from_p, e))?;
        // Preserve source permissions/attributes on the copy (std::fs::copy
        // already does this on most platforms, but not guaranteed everywhere).
        if let Ok(meta) = std::fs::metadata(&from_p) {
            let _ = std::fs::set_permissions(&to_p, meta.permissions());
        }
        Ok(())
    }

    pub fn move_path<F>(&self, from: &Path, to: &Path, approver: F) -> Result<()>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        // Move = rename with same permission story.
        self.rename(from, to, approver)
    }

    /// Dry-run bulk edit: report which files would change.
    pub fn bulk_edit_plan(
        &self,
        roots: &[PathBuf],
        old: &str,
        new: &str,
        replace_all: bool,
    ) -> Result<BulkEditPlan> {
        let mut files = Vec::new();
        for root in roots {
            let abs = self.resolve(root)?;
            if abs.is_file() {
                if file_contains(&abs, old)? {
                    files.push(abs);
                }
            } else if abs.is_dir() {
                for entry in walkdir::WalkDir::new(&abs)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_type().is_file())
                {
                    let p = entry.path();
                    if looks_binary_path(p) {
                        continue;
                    }
                    if file_contains(p, old)? {
                        files.push(p.to_path_buf());
                    }
                }
            }
        }
        Ok(BulkEditPlan {
            files,
            old_string: old.into(),
            new_string: new.into(),
            replace_all,
        })
    }

    /// Apply bulk edit as one transaction (all-or-nothing): snapshot all first, then apply.
    pub fn bulk_edit_apply<F>(&self, plan: &BulkEditPlan, approver: F) -> Result<BulkEditResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let shown: Vec<String> = plan
            .files
            .iter()
            .take(BULK_PREVIEW_CAP)
            .map(|f| display_rel(&self.project_root, f))
            .collect();
        let mut preview = shown.join("\n");
        if plan.files.len() > shown.len() {
            preview.push_str(&format!(
                "\n… ({} more file(s) not shown)",
                plan.files.len() - shown.len()
            ));
        }

        let mut approver = approver;
        self.gate.enforce(
            &PermissionRequest {
                tool: "bulk_edit".into(),
                path: None,
                command: None,
                description: format!(
                    "bulk edit would modify {} files ({} → {})",
                    plan.files.len(),
                    truncate(&plan.old_string, 40),
                    truncate(&plan.new_string, 40)
                ),
                preview: Some(preview),
                ..Default::default()
            },
            &mut approver,
        )?;

        // Snapshot all first.
        for f in &plan.files {
            self.checkpoints
                .snapshot_before(&self.turn_id, f, &self.project_root)?;
        }

        let mut modified = Vec::new();
        let mut skipped = Vec::new();
        for f in &plan.files {
            match apply_edit_file(f, &plan.old_string, &plan.new_string, plan.replace_all) {
                Ok(_) => {
                    if let Ok(stamp) = stamp_file(f) {
                        self.reads.record(f, stamp);
                    }
                    self.mark_touched(f);
                    modified.push(f.clone());
                }
                Err(e) => skipped.push((f.clone(), e.to_string())),
            }
        }
        Ok(BulkEditResult { modified, skipped })
    }

    pub fn path_kind_of(&self, user_path: &Path) -> Result<PathKind> {
        let p = self.resolve(user_path)?;
        Ok(path_kind(&p))
    }
}

fn apply_edit_file(path: &Path, old: &str, new: &str, replace_all: bool) -> Result<usize> {
    let original = std::fs::read_to_string(path).map_err(|e| FsError::io(path, e))?;
    let count = original.matches(old).count();
    if count == 0 {
        return Err(FsError::EditNotFound(path.to_path_buf()));
    }
    if count > 1 && !replace_all {
        return Err(FsError::AmbiguousEdit { count });
    }
    let content = if replace_all {
        original.replace(old, new)
    } else {
        original.replacen(old, new, 1)
    };
    std::fs::write(path, content).map_err(|e| FsError::io(path, e))?;
    Ok(count)
}

/// True if two paths are identical except for casing (the case-only-rename case).
fn paths_equal_ignoring_case(a: &Path, b: &Path) -> bool {
    a.to_string_lossy()
        .eq_ignore_ascii_case(&b.to_string_lossy())
}

/// True if an I/O error is the OS's "cross-device link" error, meaning a
/// plain `rename()` can't move the path (different filesystem/volume).
fn is_cross_device_error(e: &std::io::Error) -> bool {
    match e.raw_os_error() {
        #[cfg(unix)]
        Some(code) => code == 18, // EXDEV
        #[cfg(windows)]
        Some(code) => code == 17, // ERROR_NOT_SAME_DEVICE
        #[cfg(not(any(unix, windows)))]
        Some(_) => false,
        None => false,
    }
}

/// Recursively copy a file or directory tree (used as the cross-device
/// fallback for rename/move, where `std::fs::rename` cannot cross filesystems).
fn copy_recursive(from: &Path, to: &Path) -> std::io::Result<()> {
    if from.is_dir() {
        std::fs::create_dir_all(to)?;
        for entry in walkdir::WalkDir::new(from).min_depth(1) {
            let entry = entry.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            let rel = entry
                .path()
                .strip_prefix(from)
                .expect("walkdir entry is under `from`");
            let dest = to.join(rel);
            if entry.file_type().is_dir() {
                std::fs::create_dir_all(&dest)?;
            } else {
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::copy(entry.path(), &dest)?;
            }
        }
        Ok(())
    } else {
        std::fs::copy(from, to).map(|_| ())
    }
}

fn file_contains(path: &Path, needle: &str) -> Result<bool> {
    let bytes = std::fs::read(path).map_err(|e| FsError::io(path, e))?;
    if looks_binary(&bytes) {
        return Ok(false);
    }
    Ok(String::from_utf8_lossy(&bytes).contains(needle))
}

fn looks_binary(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let sample = &bytes[..bytes.len().min(8000)];
    sample.contains(&0) || sample.iter().filter(|b| **b < 9 && **b != b'\n' && **b != b'\r' && **b != b'\t').count() > sample.len() / 10
}

fn looks_binary_path(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => matches!(
            ext.to_ascii_lowercase().as_str(),
            "png" | "jpg" | "jpeg" | "gif" | "webp" | "pdf" | "zip" | "exe" | "dll" | "so" | "o"
                | "wasm" | "bin"
        ),
        None => false,
    }
}

struct RecursiveFileList {
    shown: Vec<String>,
    total: usize,
}

impl RecursiveFileList {
    fn render(&self) -> String {
        if self.shown.is_empty() {
            return "(empty directory)".to_string();
        }
        let mut out = self.shown.join("\n");
        if self.total > self.shown.len() {
            out.push_str(&format!(
                "\n… ({} more file(s) not shown)",
                self.total - self.shown.len()
            ));
        }
        out
    }
}

/// List up to `cap` relative file paths under `dir`, plus the true total count.
fn list_files_recursive(dir: &Path, project_root: &Path, cap: usize) -> RecursiveFileList {
    let mut shown = Vec::new();
    let mut total = 0usize;
    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        total += 1;
        if shown.len() < cap {
            shown.push(display_rel(project_root, entry.path()));
        }
    }
    RecursiveFileList { shown, total }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permission::ApprovalDecision;
    use zeus_config::AgentSettings;
    use tempfile::TempDir;

    fn engine(root: &Path) -> FileEngine {
        let settings = AgentSettings::default();
        // Permit write/edit in tests via session after first approve helper.
        let gate = PermissionGate::new(settings, root.to_path_buf());
        let cps = CheckpointStore::new(root.join(".agent/checkpoints"));
        cps.begin_turn("test-turn").unwrap();
        FileEngine::new(root.to_path_buf(), gate, cps, "test-turn")
    }

    fn approve(_: &PermissionRequest) -> ApprovalDecision {
        ApprovalDecision::Approved
    }

    #[test]
    fn read_write_edit_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(root.join("src")).unwrap();
        let eng = engine(&root);

        eng.write(Path::new("src/a.txt"), "hello world", WriteOptions::default(), approve)
            .unwrap();
        let r = eng.read(Path::new("src/a.txt"), ReadOptions::default()).unwrap();
        assert!(r.content.contains("hello world"));

        eng.edit(
            Path::new("src/a.txt"),
            EditOptions {
                old_string: "world".into(),
                new_string: "rust".into(),
                replace_all: false,
            },
            approve,
        )
        .unwrap();
        let body = std::fs::read_to_string(root.join("src/a.txt")).unwrap();
        assert_eq!(body, "hello rust");
    }

    #[test]
    fn must_read_before_overwrite() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("x.txt"), "old").unwrap();
        let eng = engine(&root);
        let err = eng
            .write(Path::new("x.txt"), "new", WriteOptions::default(), approve)
            .unwrap_err();
        assert!(matches!(err, FsError::MustReadFirst(_)));
    }

    #[test]
    fn ambiguous_edit_fails() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let eng = engine(&root);
        eng.write(Path::new("a.txt"), "x x x", WriteOptions::default(), approve)
            .unwrap();
        eng.read(Path::new("a.txt"), ReadOptions::default()).unwrap();
        let err = eng
            .edit(
                Path::new("a.txt"),
                EditOptions {
                    old_string: "x".into(),
                    new_string: "y".into(),
                    replace_all: false,
                },
                approve,
            )
            .unwrap_err();
        assert!(matches!(err, FsError::AmbiguousEdit { count: 3 }));
    }

    #[test]
    fn delete_restorable_via_checkpoint() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        // Pre-existing file (not created in this turn) so restore only undoes delete.
        std::fs::write(root.join("z.txt"), "keepme").unwrap();
        let eng = engine(&root);
        eng.delete(Path::new("z.txt"), approve).unwrap();
        assert!(!root.join("z.txt").exists());
        eng.checkpoints.restore("test-turn", &root).unwrap();
        assert_eq!(std::fs::read_to_string(root.join("z.txt")).unwrap(), "keepme");
    }

    #[test]
    fn path_escape_denied() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let eng = engine(&root);
        let err = eng.read(Path::new("../outside.txt"), ReadOptions::default());
        assert!(err.is_err());
    }

    #[test]
    fn rename_into_new_nested_dir_succeeds() {
        // Regression test for the pathutil bug: moving into a directory that
        // doesn't exist yet must work, not be rejected as escaping the root.
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.txt"), "hi").unwrap();
        let eng = engine(&root);
        eng.rename(Path::new("a.txt"), Path::new("sub/dir/a.txt"), approve)
            .unwrap();
        assert!(root.join("sub/dir/a.txt").exists());
    }

    #[test]
    fn rename_onto_existing_file_requires_overwrite_approval() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.txt"), "new").unwrap();
        std::fs::write(root.join("b.txt"), "old").unwrap();
        let eng = engine(&root);
        let err = eng
            .rename(Path::new("a.txt"), Path::new("b.txt"), |_| {
                ApprovalDecision::Denied
            })
            .unwrap_err();
        assert!(matches!(err, FsError::Denied(_)));
        // Denied — nothing should have moved.
        assert!(root.join("a.txt").exists());
        assert_eq!(std::fs::read_to_string(root.join("b.txt")).unwrap(), "old");
    }

    #[test]
    #[cfg(any(windows, target_os = "macos"))]
    fn case_only_rename_changes_casing_on_disk() {
        // Windows/macOS default filesystems are case-insensitive: a naive
        // rename() from "d.txt" to "D.TXT" can silently no-op since the OS
        // sees them as the same entry. Regression test for that fix.
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("d.txt"), "hi").unwrap();
        let eng = engine(&root);
        eng.rename(Path::new("d.txt"), Path::new("D.TXT"), approve)
            .unwrap();
        let real_name = std::fs::read_dir(&root)
            .unwrap()
            .filter_map(|e| e.ok())
            .find(|e| e.file_name().to_string_lossy().eq_ignore_ascii_case("d.txt"))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .unwrap();
        assert_eq!(real_name, "D.TXT");
    }

    #[test]
    fn copy_recursive_copies_directory_tree() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir_all(src.join("nested")).unwrap();
        std::fs::write(src.join("a.txt"), "a").unwrap();
        std::fs::write(src.join("nested/b.txt"), "b").unwrap();
        let dst = tmp.path().join("dst");
        copy_recursive(&src, &dst).unwrap();
        assert_eq!(std::fs::read_to_string(dst.join("a.txt")).unwrap(), "a");
        assert_eq!(
            std::fs::read_to_string(dst.join("nested/b.txt")).unwrap(),
            "b"
        );
    }
}
