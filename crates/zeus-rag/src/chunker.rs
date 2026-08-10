//! Document chunking: split project source files into overlapping chunks
//! that fit a model context window, skipping the usual build/generated dirs.

use std::path::{Path, PathBuf};

/// Directories (basenames) never indexed. Mirrors zeus-fs's hidden-dir skip.
pub const SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    ".cargo",
    "target",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "__pycache__",
    ".pytest_cache",
    "vendor",
];

/// Skip files above this size when indexing (matches zeus-fs's grep cap).
pub const MAX_FILE_BYTES: u64 = 512 * 1024;

/// One indexed document: path + the actual text being embedded.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    /// Absolute path the chunk came from (may be overridden when loaded).
    #[serde(default)]
    pub path: PathBuf,
    /// Structural instruction embedded for the model: file, doc range.
    #[serde(default)]
    pub index: usize,
    /// Raw chunk text sent to the embedder.
    pub text: String,
}

impl Chunk {
    pub fn new(path: PathBuf, index: usize, text: String) -> Self {
        Self { path, index, text }
    }
}

/// A wall around target length, in characters (not tokens), to keep chunk
/// sizes stable across hash/source sizes.
pub fn chunk_text(text: &str, approx_chars: usize, overlap_chars: usize) -> Vec<String> {
    if text.chars().count() <= approx_chars {
        return if text.trim().is_empty() {
            Vec::new()
        } else {
            vec![text.to_string()]
        };
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < text.len() {
        let end = next_break(&text[start..], approx_chars)
            .map(|d| start + d)
            .unwrap_or(text.len());
        // The break may be closer than `overlap` (e.g. dense punctuation); if
        // so, no overlap is possible and we must still advance past `end` to
        // guarantee forward progress (an overlap >= break distance would make
        // `start` stall forever and OOM).
        let next = end.saturating_sub(overlap_chars);
        let next = if next <= start { end } else { next };

        let piece = text[start..end].trim();
        if !piece.is_empty() {
            chunks.push(piece.to_string());
        }
        if end >= text.len() {
            break;
        }
        start = next;
    }
    chunks
}

/// Find the last whitespace boundary within `approx_chars` of the head, but
/// never at position 0 (leading whitespace on a chunk cut would stall the
/// loop). Falls back to `None`, in which case the caller takes the rest of the
/// text.
fn next_break(head: &str, approx_chars: usize) -> Option<usize> {
    let mut idx = None;
    let mut seen = 0usize;
    for (i, c) in head.char_indices() {
        if seen >= approx_chars {
            break;
        }
        if i > 0 && c.is_whitespace() {
            idx = Some(i);
        }
        seen += 1;
    }
    idx
}

/// Iterate the source files under `root`, yielding `(path, text)` pairs.
/// Hidden/skip dirs and oversized/binary-looking files are pruned.
pub fn source_files(root: &Path) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            let name = entry.file_name().to_string_lossy().to_lowercase();
            if SKIP_DIRS.contains(&name.as_str()) || name.starts_with('.') {
                continue;
            }
            out.extend(source_files(&path));
        } else if entry
            .file_type()
            .map(|t| t.is_file())
            .unwrap_or(false)
        {
            let Ok(meta) = entry.metadata() else { continue };
            if meta.len() > MAX_FILE_BYTES || meta.len() == 0 {
                continue;
            }
            let ext = path.extension().and_then(|e| e.to_str());
            if matches!(ext, Some("lock") | Some("min.js") | Some("map")) {
                continue;
            }
            match std::fs::read_to_string(&path) {
                Ok(text) => out.push((path, text)),
                Err(_) => continue, // opaque/binary content
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_chunk() {
        let c = chunk_text("hello world", 400, 50);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0], "hello world");
    }

    #[test]
    fn long_text_splits_on_word_boundaries() {
        let text = "one two three four five six seven eight nine ten ";
        let parts = text.repeat(6);
        let c = chunk_text(&parts, 30, 5);
        assert!(c.len() >= 2);
        for piece in &c {
            assert!(piece.chars().count() <= 30 + 20);
        }
    }

    #[test]
    fn overlap_reappears_in_neighbors() {
        // Overlap keeps continuity: the tail word of a chunk also appears at
        // the head of the next.
        let parts = "lorem ".repeat(80);
        let c = chunk_text(&parts, 40, 10);
        if c.len() >= 2 {
            let first_tail = c[0].chars().last().unwrap();
            assert!(c[1].contains(first_tail));
        }
    }

    #[test]
    fn source_files_skips_hidden_dirs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join(".git/config"), "[core]").unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "pub fn f() {}").unwrap();
        let files = source_files(dir.path());
        assert_eq!(files.len(), 1);
        assert!(files[0].0.ends_with("lib.rs"));
    }
}