//! Checkpoint-based undo: per-turn file snapshots + conversation pointer.

use crate::error::{FsError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FileSnapshot {
    /// File existed; full content preserved for restore.
    Content {
        path: String,
        content_b64: String,
        /// Original bytes were valid UTF-8 (best-effort).
        utf8: bool,
    },
    /// File did not exist before the op (restore = delete).
    DidNotExist { path: String },
    /// Path mapping for rename/move.
    Renamed { from: String, to: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub turn_id: String,
    pub created_at: String,
    pub ops: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CheckpointSummary {
    pub turn_id: String,
    pub path: PathBuf,
    pub file_count: usize,
}

/// Stores checkpoints under `<project>/.agent/checkpoints/<turn-id>/`.
pub struct CheckpointStore {
    root: PathBuf,
}

impl CheckpointStore {
    pub fn new(checkpoints_dir: PathBuf) -> Self {
        Self {
            root: checkpoints_dir,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn turn_dir(&self, turn_id: &str) -> PathBuf {
        self.root.join(turn_id)
    }

    /// Begin a turn checkpoint directory.
    pub fn begin_turn(&self, turn_id: &str) -> Result<PathBuf> {
        let dir = self.turn_dir(turn_id);
        std::fs::create_dir_all(&dir).map_err(|e| FsError::io(&dir, e))?;
        let meta = CheckpointMeta {
            turn_id: turn_id.to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            ops: Vec::new(),
        };
        self.write_meta(&dir, &meta)?;
        Ok(dir)
    }

    fn write_meta(&self, dir: &Path, meta: &CheckpointMeta) -> Result<()> {
        let path = dir.join("meta.json");
        let text = serde_json::to_string_pretty(meta)?;
        std::fs::write(&path, text).map_err(|e| FsError::io(&path, e))?;
        Ok(())
    }

    fn read_meta(&self, dir: &Path) -> Result<CheckpointMeta> {
        let path = dir.join("meta.json");
        let text = std::fs::read_to_string(&path).map_err(|e| FsError::io(&path, e))?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Snapshot current file state before a mutating op.
    pub fn snapshot_before(&self, turn_id: &str, path: &Path, project_root: &Path) -> Result<()> {
        let dir = self.turn_dir(turn_id);
        std::fs::create_dir_all(&dir).map_err(|e| FsError::io(&dir, e))?;
        let rel = path
            .strip_prefix(project_root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");

        let snap = if path.exists() {
            let bytes = std::fs::read(path).map_err(|e| FsError::io(path, e))?;
            let utf8 = std::str::from_utf8(&bytes).is_ok();
            FileSnapshot::Content {
                path: rel.clone(),
                content_b64: base64_encode(&bytes),
                utf8,
            }
        } else {
            FileSnapshot::DidNotExist { path: rel.clone() }
        };

        self.append_snapshot(&dir, &snap)?;
        let mut meta = self.read_meta(&dir).unwrap_or(CheckpointMeta {
            turn_id: turn_id.to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            ops: Vec::new(),
        });
        meta.ops.push(format!("snapshot {rel}"));
        self.write_meta(&dir, &meta)?;
        Ok(())
    }

    pub fn snapshot_rename(
        &self,
        turn_id: &str,
        from: &Path,
        to: &Path,
        project_root: &Path,
    ) -> Result<()> {
        let dir = self.turn_dir(turn_id);
        std::fs::create_dir_all(&dir).map_err(|e| FsError::io(&dir, e))?;
        let from_rel = from
            .strip_prefix(project_root)
            .unwrap_or(from)
            .to_string_lossy()
            .replace('\\', "/");
        let to_rel = to
            .strip_prefix(project_root)
            .unwrap_or(to)
            .to_string_lossy()
            .replace('\\', "/");
        // Also preserve destination content if overwriting.
        if to.exists() {
            self.snapshot_before(turn_id, to, project_root)?;
        }
        let snap = FileSnapshot::Renamed {
            from: from_rel.clone(),
            to: to_rel.clone(),
        };
        self.append_snapshot(&dir, &snap)?;
        Ok(())
    }

    fn append_snapshot(&self, dir: &Path, snap: &FileSnapshot) -> Result<()> {
        let path = dir.join("files.snapshot");
        let mut list: Vec<FileSnapshot> = if path.exists() {
            let text = std::fs::read_to_string(&path).map_err(|e| FsError::io(&path, e))?;
            serde_json::from_str(&text).unwrap_or_default()
        } else {
            Vec::new()
        };
        list.push(snap.clone());
        let text = serde_json::to_string_pretty(&list)?;
        std::fs::write(&path, text).map_err(|e| FsError::io(&path, e))?;
        Ok(())
    }

    pub fn load_snapshots(&self, turn_id: &str) -> Result<Vec<FileSnapshot>> {
        let path = self.turn_dir(turn_id).join("files.snapshot");
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(&path).map_err(|e| FsError::io(&path, e))?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Restore files from a turn checkpoint (reverse order).
    pub fn restore(&self, turn_id: &str, project_root: &Path) -> Result<usize> {
        let snaps = self.load_snapshots(turn_id)?;
        let mut restored = 0usize;
        for snap in snaps.iter().rev() {
            match snap {
                FileSnapshot::Content {
                    path, content_b64, ..
                } => {
                    let full = project_root.join(path);
                    if let Some(parent) = full.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| FsError::io(parent, e))?;
                    }
                    let bytes = base64_decode(content_b64).map_err(FsError::Checkpoint)?;
                    std::fs::write(&full, bytes).map_err(|e| FsError::io(&full, e))?;
                    restored += 1;
                }
                FileSnapshot::DidNotExist { path } => {
                    let full = project_root.join(path);
                    if full.exists() {
                        std::fs::remove_file(&full).map_err(|e| FsError::io(&full, e))?;
                        restored += 1;
                    }
                }
                FileSnapshot::Renamed { from, to } => {
                    let from_full = project_root.join(from);
                    let to_full = project_root.join(to);
                    if to_full.exists() {
                        if let Some(parent) = from_full.parent() {
                            std::fs::create_dir_all(parent).map_err(|e| FsError::io(parent, e))?;
                        }
                        std::fs::rename(&to_full, &from_full)
                            .map_err(|e| FsError::io(&to_full, e))?;
                        restored += 1;
                    }
                }
            }
        }
        info!(turn_id, restored, "checkpoint restored");
        Ok(restored)
    }

    pub fn list_turns(&self) -> Result<Vec<CheckpointSummary>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.root).map_err(|e| FsError::io(&self.root, e))? {
            let entry = entry.map_err(FsError::IoSimple)?;
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let turn_id = entry.file_name().to_string_lossy().to_string();
            let snaps = self.load_snapshots(&turn_id).unwrap_or_default();
            out.push(CheckpointSummary {
                turn_id,
                path: entry.path(),
                file_count: snaps.len(),
            });
        }
        out.sort_by(|a, b| a.turn_id.cmp(&b.turn_id));
        Ok(out)
    }
}

fn base64_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
}

fn base64_decode(s: &str) -> std::result::Result<Vec<u8>, String> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn snapshot_and_restore_write() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("a.txt");
        std::fs::write(&file, "v1").unwrap();

        let store = CheckpointStore::new(root.join(".agent/checkpoints"));
        store.begin_turn("t1").unwrap();
        store.snapshot_before("t1", &file, &root).unwrap();
        std::fs::write(&file, "v2").unwrap();
        store.restore("t1", &root).unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "v1");
    }

    #[test]
    fn base64_roundtrip() {
        let data = b"hello world!";
        let enc = base64_encode(data);
        let dec = base64_decode(&enc).unwrap();
        assert_eq!(dec, data);
    }
}
