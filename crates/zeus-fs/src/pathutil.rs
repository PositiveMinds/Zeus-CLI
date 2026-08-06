//! Path containment: resolve symlinks and reject escapes from project root.

use crate::error::{FsError, Result};
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathKind {
    File,
    Dir,
    Missing,
    Other,
}

/// Normalize a path (remove `.` / `..`) without requiring it to exist.
pub fn normalize_lexically(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => out.push(comp.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(c) => out.push(c),
        }
    }
    out
}

/// Resolve `user_path` relative to `project_root`, ensuring the result stays inside the root.
///
/// Existing paths are canonicalized (symlink-aware). Missing paths are checked lexically
/// against the canonical project root.
pub fn resolve_in_project(project_root: &Path, user_path: &Path) -> Result<PathBuf> {
    let root = std::fs::canonicalize(project_root).map_err(|e| FsError::io(project_root, e))?;

    let candidate = if user_path.is_absolute() {
        user_path.to_path_buf()
    } else {
        project_root.join(user_path)
    };

    if candidate.exists() {
        let canon = std::fs::canonicalize(&candidate).map_err(|e| FsError::io(&candidate, e))?;
        if !contains_path(&root, &canon) {
            return Err(FsError::PathEscape(canon));
        }
        return Ok(canon);
    }

    // Missing path (possibly several directory levels deep): walk up to the
    // nearest ancestor that actually exists, canonicalize *that* (resolving
    // any symlinks along the existing portion of the path), then re-join the
    // remaining not-yet-created components lexically. Comparing a
    // canonicalized root against a non-canonicalized candidate here was the
    // bug: `canonicalize()` adds a `\\?\`-prefixed extended-path form on
    // Windows that a plain `join()` never produces, so same-location paths
    // compared unequal and legitimate nested writes were rejected.
    let lexical = normalize_lexically(&candidate);
    let mut existing: &Path = &lexical;
    let mut remainder: Vec<&std::ffi::OsStr> = Vec::new();
    while !existing.exists() {
        match (existing.file_name(), existing.parent()) {
            (Some(name), Some(parent)) => {
                remainder.push(name);
                existing = parent;
            }
            // Reached a root/prefix component that doesn't exist either.
            _ => return Err(FsError::PathEscape(lexical)),
        }
    }

    let existing_canon = std::fs::canonicalize(existing).map_err(|e| FsError::io(existing, e))?;
    if !contains_path(&root, &existing_canon) {
        return Err(FsError::PathEscape(lexical));
    }

    let mut resolved = existing_canon;
    for part in remainder.into_iter().rev() {
        resolved.push(part);
    }
    Ok(resolved)
}

/// True if `child` is equal to `root` or a descendant of `root`.
pub fn contains_path(root: &Path, child: &Path) -> bool {
    let root_comps: Vec<_> = root.components().collect();
    let child_comps: Vec<_> = child.components().collect();
    if child_comps.len() < root_comps.len() {
        return false;
    }
    root_comps.iter().zip(child_comps.iter()).all(|(a, b)| a == b)
}

pub fn path_kind(path: &Path) -> PathKind {
    if !path.exists() {
        return PathKind::Missing;
    }
    if path.is_file() {
        PathKind::File
    } else if path.is_dir() {
        PathKind::Dir
    } else {
        PathKind::Other
    }
}

/// Relative path display under project root (for logs / approvals).
pub fn display_rel(project_root: &Path, path: &Path) -> String {
    path.strip_prefix(project_root)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn rejects_escape_with_dotdot() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let err = resolve_in_project(&root, Path::new("../secret")).unwrap_err();
        assert!(matches!(err, FsError::PathEscape(_)));
    }

    #[test]
    fn accepts_nested_path() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        let p = resolve_in_project(&root, Path::new("src/main.rs")).unwrap();
        assert!(p.ends_with("main.rs"));
    }

    #[test]
    fn accepts_missing_nested_subdirectory() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        // "newdir" does not exist yet — this reproduces the reported bug
        // where creating a file inside a not-yet-created subdirectory was
        // wrongly rejected as escaping the project root.
        let p = resolve_in_project(&root, Path::new("newdir/newfile.txt")).unwrap();
        assert!(p.ends_with("newfile.txt"));

        // Multiple missing levels deep should also resolve.
        let p2 = resolve_in_project(&root, Path::new("a/b/c/deep.txt")).unwrap();
        assert!(p2.ends_with("deep.txt"));
    }

    #[test]
    fn contains_path_basic() {
        assert!(contains_path(Path::new("/a/b"), Path::new("/a/b/c")));
        assert!(contains_path(Path::new("/a/b"), Path::new("/a/b")));
        assert!(!contains_path(Path::new("/a/b"), Path::new("/a/c")));
    }
}
