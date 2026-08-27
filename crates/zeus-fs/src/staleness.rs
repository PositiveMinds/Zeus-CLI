//! On-disk staleness detection (hash + mtime) against last Read.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::SystemTime;

#[derive(Debug, Clone)]
pub struct FileStamp {
    pub mtime: Option<SystemTime>,
    pub hash: String,
    pub len: u64,
}

/// Tracks files the agent has Read this session / turn.
/// Uses `RwLock` so concurrent reads don't block each other; only
/// `record` and `clear` take a write lock.
#[derive(Debug, Default)]
pub struct ReadTracker {
    inner: RwLock<HashMap<PathBuf, FileStamp>>,
}

impl ReadTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&self, path: &Path, stamp: FileStamp) {
        self.inner
            .write()
            .unwrap()
            .insert(path.to_path_buf(), stamp);
    }

    pub fn get(&self, path: &Path) -> Option<FileStamp> {
        self.inner.read().unwrap().get(path).cloned()
    }

    pub fn has_read(&self, path: &Path) -> bool {
        self.inner.read().unwrap().contains_key(path)
    }

    pub fn clear(&self) {
        self.inner.write().unwrap().clear();
    }
}

pub fn hash_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

pub fn stamp_file(path: &Path) -> std::io::Result<FileStamp> {
    let meta = std::fs::metadata(path)?;
    let data = std::fs::read(path)?;
    Ok(FileStamp {
        mtime: meta.modified().ok(),
        hash: hash_bytes(&data),
        len: meta.len(),
    })
}

/// Returns true if the file still matches the stamp.
pub fn is_fresh(path: &Path, stamp: &FileStamp) -> bool {
    match stamp_file(path) {
        Ok(now) => now.hash == stamp.hash && now.len == stamp.len,
        Err(_) => false,
    }
}
