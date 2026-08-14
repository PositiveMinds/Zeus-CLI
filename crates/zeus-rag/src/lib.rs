//! RAG for zeus: chunk project files, optionally embed each chunk with a
//! provider's embedding model, and run hybrid (keyword + vector) search over
//! the in-memory index. No SQL, no server, no durable store — the index is
//! rebuilt from the filesystem on demand.
//!
//! ```
//! use zeus_rag::{RagIndex, chunker::Chunk};
//!
//! let mut index = RagIndex::new(std::path::PathBuf::from("."));
//! index.add_chunk(Chunk::new("README.md".into(), 0, "//! coding agent".into()));
//! let hits = index.search("coding agent", 3);
//! assert!(!hits.is_empty());
//! ```

pub mod chunker;
pub mod search;

use chunker::{source_files, Chunk};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use zeus_provider::{EmbeddingRequest, ModelProvider};

/// On-disk filename of the persisted RAG index (matches the `.agent/`
/// convention used by `index.json` for the symbol index).
pub const INDEX_FILE: &str = ".agent/rag_index.json";

/// A per-source-file stamp captured when the index was built, used to decide
/// cheaply whether a persisted index is still current.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileStamp {
    /// Absolute path the chunk came from.
    pub path: PathBuf,
    pub len: u64,
    /// Modification time in seconds since the Unix epoch.
    pub mtime_secs: u64,
}

/// The on-disk snapshot of a [`RagIndex`]: the chunks plus the source-file
/// stamps they were built from. Loading this and checking [`Self::is_fresh`]
/// lets a caller reuse a previously-built index instead of re-walking and
/// re-chunking the whole project on every query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedRagIndex {
    pub root: PathBuf,
    /// Seconds since the Unix epoch when the index was built.
    #[serde(default)]
    pub built_at: u64,
    pub documents: Vec<Chunk>,
    /// Per-document embedding vectors, parallel to `documents`.
    #[serde(default)]
    pub vectors: Option<Vec<Vec<f32>>>,
    /// Source-file stamps this index was built from.
    #[serde(default)]
    pub stamps: Vec<FileStamp>,
}

impl PersistedRagIndex {
    /// The path where a project's persisted RAG index lives.
    pub fn file_path(root: &Path) -> PathBuf {
        root.join(INDEX_FILE)
    }

    /// Snapshot the current in-memory index along with its source stamps.
    pub fn from_index(index: &RagIndex) -> Self {
        let built_at = now_secs();
        let stamps = chunker::source_file_stats(&index.root)
            .into_iter()
            .map(|(path, len, mtime_secs)| FileStamp {
                path,
                len,
                mtime_secs,
            })
            .collect();
        Self {
            root: index.root.clone(),
            built_at,
            documents: index.documents.clone(),
            vectors: index.vectors.clone(),
            stamps,
        }
    }

    /// Convert back into a searchable in-memory index (stamps discarded).
    pub fn into_index(self) -> RagIndex {
        RagIndex {
            root: self.root,
            documents: self.documents,
            vectors: self.vectors,
        }
    }

    /// True when every stamped source file still matches on disk (same size
    /// and mtime) — a cheap metadata-only check, so reuse the index if this
    /// returns true rather than re-walking the whole tree.
    pub fn is_fresh(&self) -> bool {
        let current = chunker::source_file_stats(&self.root);
        if current.len() != self.stamps.len() {
            return false;
        }
        let stamp: std::collections::HashMap<&PathBuf, &FileStamp> =
            self.stamps.iter().map(|s| (&s.path, s)).collect();
        current
            .into_iter()
            .all(|(path, len, mtime_secs)| match stamp.get(&path) {
                Some(s) => s.len == len && s.mtime_secs == mtime_secs,
                None => false,
            })
    }

    /// True when at least one document carries a non-empty embedding.
    pub fn has_vectors(&self) -> bool {
        self.vectors
            .as_ref()
            .map(|vs| vs.iter().any(|v| !v.is_empty()))
            .unwrap_or(false)
    }

    /// Incrementally refresh the chunk index instead of re-walking the whole
    /// project: re-chunks only the files whose size or mtime changed (or that
    /// were added), drops chunks of files that disappeared, and keeps every
    /// untouched chunk as-is. Returns true if anything changed. On any change
    /// the stored vectors are dropped — they no longer align with the
    /// documents — so callers that want vectors must re-embed afterwards.
    pub fn refresh(&mut self, approx_chars: usize, overlap: usize) -> bool {
        let current = chunker::source_file_stats(&self.root);
        let current_map: std::collections::HashMap<PathBuf, (u64, u64)> = current
            .into_iter()
            .map(|(path, len, mtime)| (path, (len, mtime)))
            .collect();
        let stamp: std::collections::HashMap<PathBuf, &FileStamp> =
            self.stamps.iter().map(|s| (s.path.clone(), s)).collect();

        let removed: Vec<PathBuf> = self
            .stamps
            .iter()
            .filter(|s| !current_map.contains_key(&s.path))
            .map(|s| s.path.clone())
            .collect();
        let touched: Vec<PathBuf> = current_map
            .iter()
            .filter(|(path, (len, mtime))| match stamp.get(*path) {
                Some(s) => s.len != *len || s.mtime_secs != *mtime,
                None => true,
            })
            .map(|(path, _)| path.clone())
            .collect();

        if removed.is_empty() && touched.is_empty() {
            return false;
        }

        let drop: std::collections::HashSet<PathBuf> =
            removed.iter().chain(touched.iter()).cloned().collect();
        self.documents.retain(|c| !drop.contains(&c.path));

        let mut offset = self.documents.len();
        for path in &touched {
            if let Ok(text) = std::fs::read_to_string(path) {
                for piece in chunker::chunk_text(&text, approx_chars, overlap) {
                    self.documents.push(Chunk::new(path.clone(), offset, piece));
                    offset += 1;
                }
            }
        }

        self.stamps = current_map
            .into_iter()
            .map(|(path, (len, mtime_secs))| FileStamp {
                path,
                len,
                mtime_secs,
            })
            .collect();
        self.built_at = now_secs();
        self.vectors = None;
        true
    }

    /// Persist to `.agent/rag_index.json`, creating `.agent/` if needed.
    pub fn save(&self, root: &Path) -> std::io::Result<()> {
        let path = Self::file_path(root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&path, json)
    }

    /// Load the persisted index for `root`, if a readable one exists.
    pub fn load(root: &Path) -> Option<Self> {
        let path = Self::file_path(root);
        let text = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&text).ok()
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// In-memory RAG index: parallel vectors of `documents` and (optional)
/// embeddings. When `vectors` is `None` or empty, search degrades gracefully
/// to keyword-only retrieval.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RagIndex {
    pub root: PathBuf,
    #[serde(default)]
    pub documents: Vec<Chunk>,
    /// Per-document embedding vectors, parallel to `documents`.
    #[serde(default)]
    pub vectors: Option<Vec<Vec<f32>>>,
}

impl RagIndex {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            documents: Vec::new(),
            vectors: None,
        }
    }

    pub fn add_chunk(&mut self, chunk: Chunk) {
        // Any new doc invalidates prior embeddings.
        self.vectors = None;
        self.documents.push(chunk);
    }

    /// Manually set the embedding vectors (used by tests / offline loaders).
    pub fn set_vectors(&mut self, vectors: Vec<Vec<f32>>) {
        assert_eq!(vectors.len(), self.documents.len(), "vector count mismatch");
        self.vectors = Some(vectors);
    }

    /// Build an index from every source file under `root`, each chunked to
    /// ~`approx_chars` chars with `overlap` continuity chars.
    pub fn from_project(root: &std::path::Path, approx_chars: usize, overlap: usize) -> Self {
        let mut index = Self::new(root.to_path_buf());
        let mut i = 0usize;
        for (path, text) in source_files(root) {
            for piece in chunker::chunk_text(&text, approx_chars, overlap) {
                index.add_chunk(chunker::Chunk::new(path.clone(), i, piece));
                i += 1;
            }
        }
        index
    }

    /// Embed every chunk in `max_batch`-sized provider calls. Best-effort: a
    /// failed batch keeps its prior vectors (or empty slots) and never aborts
    /// the rest of the index. Returns the number of vectors refreshed.
    pub async fn embed_all(
        &mut self,
        provider: &(dyn ModelProvider + Send + Sync),
        model: &str,
        max_batch: usize,
    ) -> Result<usize, zeus_provider::ProviderError> {
        let n = self.documents.len();
        let mut vectors = self.vectors.take().unwrap_or_else(|| vec![Vec::new(); n]);
        if vectors.len() != n {
            vectors = vec![Vec::new(); n];
        }
        let batch = max_batch.max(1);
        let mut embedded = 0usize;
        for start in (0..n).step_by(batch) {
            let end = (start + batch).min(n);
            let texts: Vec<String> = self.documents[start..end]
                .iter()
                .map(|c| c.text.clone())
                .collect();
            match provider
                .embeddings(EmbeddingRequest {
                    model: model.to_string(),
                    input: texts,
                })
                .await
            {
                Ok(resp) => {
                    for (off, vec_l) in resp.vectors.iter().enumerate() {
                        if let Some(v) = vectors.get_mut(start + off) {
                            *v = vec_l.clone();
                            embedded += 1;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(err = %e, "embedding batch failed; keeping keyword-only for this batch");
                }
            }
        }
        self.vectors = Some(vectors);
        Ok(embedded)
    }

    /// Number of ingested chunks.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_reports_len_and_empty() {
        let mut idx = RagIndex::new("/".into());
        assert!(idx.is_empty());
        idx.add_chunk(chunker::Chunk::new("a".into(), 0, "hello".into()));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn from_project_chunks_source_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").unwrap();
        let idx = RagIndex::from_project(dir.path(), 200, 20);
        assert_eq!(idx.len(), 1);
        assert!(idx.documents[0].text.contains("fn main"));
    }

    #[test]
    fn adding_chunks_invalidates_vectors() {
        let mut idx = RagIndex::new("/".into());
        idx.add_chunk(chunker::Chunk::new("a".into(), 0, "x".into()));
        ctx_fill_vectors(&mut idx);
        assert!(idx.vectors.is_some());
        idx.add_chunk(chunker::Chunk::new("b".into(), 1, "y".into()));
        assert!(idx.vectors.is_none());
    }

    fn ctx_fill_vectors(idx: &mut RagIndex) {
        if idx.documents.len() == 1 {
            idx.set_vectors(vec![vec![1.0, 0.0]]);
        }
    }

    #[test]
    fn persisted_index_roundtrips_and_is_fresh() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").unwrap();
        let idx = RagIndex::from_project(dir.path(), 200, 20);
        let persisted = PersistedRagIndex::from_index(&idx);
        assert_eq!(persisted.documents.len(), 1);
        assert_eq!(persisted.stamps.len(), 1);
        assert!(persisted.is_fresh());

        persisted.save(dir.path()).unwrap();
        let loaded = PersistedRagIndex::load(dir.path()).unwrap();
        assert_eq!(loaded.documents.len(), 1);
        assert_eq!(loaded.documents[0].text, idx.documents[0].text);
        assert!(loaded.is_fresh());
        let rebuilt = loaded.into_index();
        assert_eq!(rebuilt.len(), 1);
    }

    #[test]
    fn persisted_index_detects_stale_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn a() {}\n").unwrap();
        let idx = RagIndex::from_project(dir.path(), 200, 20);
        let persisted = PersistedRagIndex::from_index(&idx);
        assert!(persisted.is_fresh());

        std::fs::write(dir.path().join("a.rs"), "fn a() { changed }\n").unwrap();
        assert!(!persisted.is_fresh());
    }

    #[test]
    fn persisted_index_detects_added_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn a() {}\n").unwrap();
        let idx = RagIndex::from_project(dir.path(), 200, 20);
        let persisted = PersistedRagIndex::from_index(&idx);
        assert!(persisted.is_fresh());

        std::fs::write(dir.path().join("b.rs"), "fn b() {}\n").unwrap();
        assert!(!persisted.is_fresh());
    }

    #[test]
    fn load_missing_index_returns_none() {
        assert!(PersistedRagIndex::load(std::path::Path::new("/does/not/exist")).is_none());
    }

    #[test]
    fn refresh_rechunks_only_changed_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(dir.path().join("b.rs"), "fn b() {}\n").unwrap();
        let idx = RagIndex::from_project(dir.path(), 200, 20);
        let mut persisted = PersistedRagIndex::from_index(&idx);
        assert!(persisted.is_fresh());

        std::fs::write(dir.path().join("a.rs"), "fn a_changed() {}\n").unwrap();
        assert!(!persisted.is_fresh());
        assert!(persisted.refresh(200, 20));
        assert!(persisted.is_fresh());
        assert_eq!(persisted.documents.len(), 2);
        assert!(persisted
            .documents
            .iter()
            .any(|c| c.text.contains("a_changed")));
        assert!(persisted.documents.iter().any(|c| c.text.contains("fn b")));
    }

    #[test]
    fn refresh_drops_removed_file_chunks() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(dir.path().join("b.rs"), "fn b() {}\n").unwrap();
        let idx = RagIndex::from_project(dir.path(), 200, 20);
        let mut persisted = PersistedRagIndex::from_index(&idx);
        assert_eq!(persisted.documents.len(), 2);

        std::fs::remove_file(dir.path().join("b.rs")).unwrap();
        assert!(persisted.refresh(200, 20));
        assert!(persisted.is_fresh());
        assert_eq!(persisted.documents.len(), 1);
        assert!(persisted.documents[0].text.contains("fn a"));
    }

    #[test]
    fn refresh_unchanged_returns_false() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn a() {}\n").unwrap();
        let idx = RagIndex::from_project(dir.path(), 200, 20);
        let mut persisted = PersistedRagIndex::from_index(&idx);
        assert!(!persisted.refresh(200, 20));
    }

    #[test]
    fn persisted_index_roundtrips_vectors() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/a.rs"), "fn keyword() {}\n").unwrap();
        let mut idx = RagIndex::from_project(dir.path(), 200, 20);
        idx.set_vectors(vec![vec![1.0, 0.0]]);
        let persisted = PersistedRagIndex::from_index(&idx);
        assert!(persisted.has_vectors());
        persisted.save(dir.path()).unwrap();

        let loaded = PersistedRagIndex::load(dir.path()).unwrap();
        assert!(loaded.has_vectors());
        assert_eq!(loaded.vectors, idx.vectors);
        let hits = loaded.into_index().search("keyword", 1);
        assert!(!hits.is_empty());
        assert!(hits[0].vector > 0.0);
    }
}
