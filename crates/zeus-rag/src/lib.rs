//! RAG for zeus: chunk project files, optionally embed each chunk with a
//! provider's embedding model, and run hybrid (keyword + vector) search over
//! the in-memory index.
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
pub mod memory;
pub mod search;

use chunker::{source_files, Chunk};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use zeus_provider::{EmbeddingRequest, ModelProvider};

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
    pub async fn embed_all<P>(
        &mut self,
        provider: &P,
        model: &str,
        max_batch: usize,
    ) -> Result<usize, zeus_provider::ProviderError>
    where
        P: ModelProvider + Send + Sync,
    {
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
}
