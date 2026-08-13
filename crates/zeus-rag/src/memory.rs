//! Long-term memory: a compact, keyword-retrievable store of durable notes
//! ("what did the user prefer?", "this module's contract") that survives
//! session restarts. Persisted as JSON; search is the same hybrid path as the
//! RAG index but over memory entries.

use crate::search::tokens;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// One durable memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub text: String,
    /// Source context (e.g. "user", "plan:2", "review").
    pub source: String,
    pub created_at_secs: u64,
}

impl MemoryEntry {
    pub fn new(text: impl Into<String>, source: impl Into<String>) -> Self {
        let created_at_secs =
            match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                Ok(d) => d.as_secs(),
                Err(_) => 0,
            };
        Self {
            text: text.into(),
            source: source.into(),
            created_at_secs,
        }
    }
}

/// A matched memory entry.
#[derive(Debug, Clone)]
pub struct MemoryHit {
    pub entry: MemoryEntry,
    pub score: f32,
}

impl MemoryHit {
    fn new(entry: MemoryEntry, score: f32) -> Self {
        Self { entry, score }
    }
}

/// Persistent memory store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStore {
    pub entries: Vec<MemoryEntry>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, text: impl Into<String>, source: impl Into<String>) {
        self.entries.push(MemoryEntry::new(text, source));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Keyword-only retrieval: the term-weighted sum over entry TF-IDF,
    /// newest-first tie-break. Returns at most `k` entries.
    pub fn recall(&self, query: &str, k: usize) -> Vec<MemoryHit> {
        let q_tokens = tokens(query);
        if q_tokens.is_empty() || self.entries.is_empty() {
            return Vec::new();
        }
        let n = self.entries.len() as f32;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut scored: Vec<(MemoryEntry, f32)> = self
            .entries
            .iter()
            .map(|entry| {
                let dt = tokens(&entry.text);
                let mut total = 0.0f32;
                for qt in &q_tokens {
                    let df = self
                        .entries
                        .iter()
                        .filter(|e| tokens(&e.text).iter().any(|t| t == qt))
                        .count();
                    let idf = (n / (df as f32).max(1.0)).ln() + 1.0;
                    let tf = dt.iter().filter(|t| *t == qt).count() as f32;
                    total += idf * (tf / (tf + 1.5));
                }
                // Recency bias: +0.6% per day fresh, capped.
                let age = now.saturating_sub(entry.created_at_secs);
                let recency = 1.0 + (age / 86_400).min(20) as f32 * 0.006;
                (entry.clone(), total * recency)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .filter(|(_, s)| *s > 0.0)
            .take(k.max(1))
            .map(|(entry, s)| MemoryHit::new(entry, s))
            .collect()
    }

    /// Load from `path` (missing file → empty store).
    pub fn load(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self::new();
        };
        serde_json::from_str(&text).unwrap_or_else(|_| Self::new())
    }

    /// Save to `path`, creating parents if needed.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| e.to_string())
    }

    /// Standard per-session persistence directory: `<root>/.zeus/memory.json`.
    pub fn default_path(root: &Path) -> std::path::PathBuf {
        root.join(".zeus").join("memory.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_finds_by_term() {
        let mut m = MemoryStore::new();
        m.add("user prefers tabs over spaces", "user");
        m.add("deploy command is: zeus project deploy", "plan:0");
        let hits = m.recall("tabs spaces", 5);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].entry.text.contains("tabs"));
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory.json");
        let mut m = MemoryStore::new();
        m.add("remember this", "user");
        m.save(&path).unwrap();
        let loaded = MemoryStore::load(&path);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.entries[0].text, "remember this");
    }

    #[test]
    fn missing_file_loads_empty() {
        let m = MemoryStore::load(Path::new("/does/not/exist.json"));
        assert!(m.is_empty());
    }
}
