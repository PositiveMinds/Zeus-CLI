//! Phase 6 — Code Intelligence (database-free, first cut).
//!
//! A lightweight symbol index and reference tools built on the same
//! ripgrep-backed, database-free foundation as the rest of zeus:
//!
//! - [`IndexEngine::scan`] walks a project's source files with per-language
//!   extraction and writes the result to the already-allocated
//!   `.agent/index.json` (no SQL).
//! - [`SymbolIndex::query`] lets callers look symbols up by name, giving
//!   "go to definition"-style answers.
//! - Cross-project *references* are computed with ripgrep (the shared
//!   [`crate::search::SearchEngine`]) so "find references"/"rename" reuse the
//!   exact search substrate the agent already has.
//!
//! Extraction is layered: languages with a wired tree-sitter grammar
//! ([`crate::tsint`]) are parsed into a real AST (accurate definitions even in
//! nested bodies); the rest fall back to compact per-language regex
//! extractors. Both write the same on-disk contract, so the index format stays
//! forward-compatible.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::error::{FsError, Result};

/// On-disk index filename (matches `ProjectPaths::index_json`).
pub const INDEX_FILE: &str = ".agent/index.json";

/// Skip files larger than this — huge generated/minified blobs add little to a
/// symbol index and cost scan time.
const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;

/// Directories never descended into (besides hidden dirs).
const SKIPPED_DIRS: &[&str] = &[
    ".git",
    ".agent",
    ".claude",
    "node_modules",
    "target",
    "dist",
    "build",
    "out",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "tmp",
    ".idea",
    ".vscode",
];

/// One extracted symbol.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Symbol {
    /// Identifier, e.g. `run`, `Cli`, `bulk_edit_apply`.
    pub name: String,
    /// `function` | `class` | `struct` | `enum` | `interface` | `trait`
    /// | `type` | `const` | `var` | `method` | `module` | `static`.
    pub kind: String,
    /// Path relative to the project root, `/`-separated.
    pub file: String,
    /// 1-based source line.
    pub line: usize,
}

/// The on-disk contract — one JSON file, database-free.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SymbolIndex {
    /// Seconds since the Unix epoch when the index was built.
    pub built_at: u64,
    /// Absolute project root this index was built from.
    pub project_root: String,
    /// How many files were scanned during the last build.
    pub scanned_files: usize,
    pub symbols: Vec<Symbol>,
}

/// Language profile: which extensions map to which `(regex, kind)` pairs.
struct Lang {
    exts: &'static [&'static str],
    /// `(regex, kind)` — each pattern captures the identifier in group 1.
    patterns: &'static [(&'static str, &'static str)],
}

const LANGS: &[Lang] = &[
    Lang {
        exts: &["rs"],
        patterns: &[
            (r"\bfn\s+([a-z_]\w*)\s*\(", "function"),
            (r"\bfn\s+([a-z_]\w*)\s*<", "function"),
            (r"\bstruct\s+([A-Z]\w*)\b", "struct"),
            (r"\benum\s+([A-Z]\w*)\b", "enum"),
            (r"\btrait\s+([A-Z]\w*)\b", "trait"),
            (r"\btype\s+([A-Z]\w*)\s*=", "type"),
            (r"\bconst\s+([a-z_][A-Za-z0-9_]*)\s*[:=]", "const"),
            (r"\bstatic\s+(mut\s+)?([a-z_]\w*)\s*:", "static"),
        ],
    },
    Lang {
        exts: &["js", "jsx", "mjs", "cjs", "ts", "tsx"],
        patterns: &[
            (r"\bclass\s+([A-Za-z_$][\w$]*)\b", "class"),
            (r"\binterface\s+([A-Za-z_$][\w$]*)\b", "interface"),
            (r"\benum\s+([A-Za-z_$][\w$]*)\b", "enum"),
            (r"\btype\s+([A-Z][\w$]*)\s*=", "type"),
            (r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", "function"),
            (
                r"\bconst\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
                "function",
            ),
            (r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", "const"),
        ],
    },
    Lang {
        exts: &["py"],
        patterns: &[
            (r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", "function"),
            (r"^\s*class\s+([A-Za-z_]\w*)\s*(?:\(|:)", "class"),
        ],
    },
    Lang {
        exts: &["go"],
        patterns: &[
            (r"^func\s+\(\w+\s+\*\w+\)\s+([A-Z]\w*)\s*\(", "method"),
            (r"^func\s+\(\w+\s+\w+\)\s+([A-Z]\w*)\s*\(", "method"),
            (r"^func\s+([A-Za-z_]\w*)\s*\(", "function"),
            (r"^type\s+([A-Z]\w*)\s+struct\b", "struct"),
            (r"^type\s+([A-Z]\w*)\s+interface\b", "interface"),
            (r"^const\s+([A-Za-z_]\w*)", "const"),
            (r"^var\s+([A-Za-z_]\w*)", "var"),
        ],
    },
    Lang {
        exts: &["c", "h", "cpp", "cc", "cxx", "hpp", "hh", "hxx"],
        patterns: &[
            (r"\bclass\s+([A-Za-z_]\w*)\b", "class"),
            (r"\bstruct\s+([A-Za-z_]\w*)\b", "struct"),
            (r"\benum\s+([A-Za-z_]\w*)\b", "enum"),
        ],
    },
    Lang {
        exts: &["java", "kt", "kts"],
        patterns: &[
            (r"\bclass\s+([A-Z]\w*)\b", "class"),
            (r"\binterface\s+([A-Z]\w*)\b", "interface"),
            (r"\benum\s+([A-Z]\w*)\b", "enum"),
            (r"\bfun\s+([A-Za-z_]\w*)\s*\(", "function"),
            (
                r"^\s*(?:public|protected|private|static|final|default|\s)+\s+[A-Za-z_][\w<>,?.\[\] ]*\s+([a-z_]\w*)\s*\(",
                "method",
            ),
        ],
    },
    Lang {
        exts: &["cs"],
        patterns: &[
            (r"\bclass\s+([A-Z]\w*)\b", "class"),
            (r"\binterface\s+([A-Z]\w*)\b", "interface"),
            (r"\benum\s+([A-Z]\w*)\b", "enum"),
        ],
    },
    Lang {
        exts: &["rb"],
        patterns: &[
            (r"^\s*def\s+([A-Za-z_]\w*)\s*[\(=]", "method"),
            (r"^\s*class\s+([A-Za-z_]\w*)\b", "class"),
            (r"^\s*module\s+([A-Za-z_]\w*)\b", "module"),
        ],
    },
    Lang {
        exts: &["php"],
        patterns: &[
            (r"\bfunction\s+([A-Za-z_]\w*)\s*\(", "function"),
            (r"\bclass\s+([A-Za-z_]\w*)\b", "class"),
            (r"\binterface\s+([A-Za-z_]\w*)\b", "interface"),
            (r"\btrait\s+([A-Za-z_]\w*)\b", "trait"),
        ],
    },
    Lang {
        exts: &["swift"],
        patterns: &[
            (r"^func\s+([A-Za-z_]\w*)\s*\(", "function"),
            (r"^\s*class\s+([A-Za-z_]\w*)\b", "class"),
            (r"^\s*struct\s+([A-Za-z_]\w*)\b", "struct"),
            (r"^\s*enum\s+([A-Za-z_]\w*)\b", "enum"),
        ],
    },
    Lang {
        exts: &["lua"],
        patterns: &[(
            r"^function\s+(?:[A-Za-z_]\w*[\.:])*([A-Za-z_]\w*)\s*\(",
            "function",
        )],
    },
];

/// Builds symbol indexes for a project root.
#[derive(Debug, Clone)]
pub struct IndexEngine {
    pub project_root: PathBuf,
}

impl IndexEngine {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: root.into(),
        }
    }

    /// Walk the project root and produce a fresh in-memory index.
    pub fn scan(&self) -> Result<SymbolIndex> {
        let root = &self.project_root;
        let mut symbols = Vec::new();
        let mut scanned = 0usize;

        let root_clone = root.clone();
        let walker = walkdir::WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            // Never prune the walk root itself; only descend-skip nested dirs.
            .filter_entry(|e| e.depth() == 0 || !is_skipped_dir(e));

        for entry_result in walker {
            let entry = match entry_result {
                Ok(e) => e,
                Err(_) => continue,
            };
            if entry.file_type().is_dir() {
                continue;
            }
            let path = entry.path();
            let Some(lang) = language_for(path) else {
                continue;
            };

            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_file() || meta.len() > MAX_FILE_BYTES {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(path) else {
                continue;
            };
            scanned += 1;
            let rel = path
                .strip_prefix(&root_clone)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            // Phase 6: real tree-sitter parsing where a grammar is wired;
            // the regex extractors remain the fallback for other languages.
            if let Some(ts) = crate::tsint::ts_language_for(path) {
                crate::tsint::extract_symbols_ts(&rel, ts, &text, &mut symbols);
            } else {
                extract_symbols(&rel, lang, &text, &mut symbols);
            }
        }

        let built_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Ok(SymbolIndex {
            built_at,
            project_root: root.to_string_lossy().into_owned(),
            scanned_files: scanned,
            symbols,
        })
    }
}

/// `filter_entry` predicate: a DirEntry is skipped if it's a skipped/hidden dir.
fn is_skipped_dir(entry: &walkdir::DirEntry) -> bool {
    if !entry.file_type().is_dir() {
        return false;
    }
    let name = entry
        .file_name()
        .to_str()
        .unwrap_or_default()
        .to_lowercase();
    if name.starts_with('.') {
        return true;
    }
    SKIPPED_DIRS.contains(&name.as_str())
}

/// Pick the first language profile whose extensions match `path`.
fn language_for(path: &Path) -> Option<&'static Lang> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_lowercase();
    LANGS.iter().find(|l| l.exts.iter().any(|e| *e == ext))
}

/// Note: identifiers at the *start* of a line that also appear as a substring
/// of a longer token are filtered out by the word-boundary patterns.
fn extract_symbols(rel: &str, lang: &Lang, text: &str, out: &mut Vec<Symbol>) {
    let patterns = compiled(lang);
    for (idx, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        for (re, kind) in patterns {
            if let Some(c) = re.captures(line) {
                if let Some(m) = c.get(1) {
                    out.push(Symbol {
                        name: m.as_str().to_string(),
                        kind: kind.to_string(),
                        file: rel.to_string(),
                        line: idx + 1,
                    });
                    break; // one symbol per line is plenty for an approximate index
                }
            }
        }
    }
}

/// Precompiled `(Regex, kind)` list for a language profile, cached once.
type Pattern = (regex::Regex, &'static str);
fn compiled(lang: &Lang) -> &'static [Pattern] {
    static CACHE: OnceLock<Box<[Vec<Pattern>]>> = OnceLock::new();
    let all: &[Vec<Pattern>] = CACHE.get_or_init(|| {
        LANGS
            .iter()
            .map(|l| {
                l.patterns
                    .iter()
                    .filter_map(|(src, kind)| regex::Regex::new(src).ok().map(|re| (re, *kind)))
                    .collect()
            })
            .collect()
    });
    let index = LANGS
        .iter()
        .position(|l| std::ptr::eq(l, lang))
        .unwrap_or(0);
    &all[index]
}

impl SymbolIndex {
    /// Path of the index JSON file for a project root.
    pub fn file_path(root: &Path) -> PathBuf {
        root.join(INDEX_FILE)
    }

    /// Persist to `.agent/index.json`, creating `.agent/` if needed.
    pub fn save(&self, root: &Path) -> Result<()> {
        let path = Self::file_path(root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| FsError::io(path.clone(), e))?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json).map_err(|e| FsError::io(path, e))
    }

    /// Load the index for a project root, if a readable one exists.
    pub fn load(root: &Path) -> Result<Option<SymbolIndex>> {
        let path = Self::file_path(root);
        if !path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path).map_err(|e| FsError::io(path.clone(), e))?;
        match serde_json::from_str(&text) {
            Ok(idx) => Ok(Some(idx)),
            Err(_) => Ok(None), // a partial/stale index just needs a rebuild
        }
    }

    /// Symbols whose name contains `needle` (case-insensitive), definitions
    /// (exact hits) first, then by file path, then line.
    pub fn query(&self, needle: &str) -> Vec<&Symbol> {
        let lower = needle.to_lowercase();
        let mut hits: Vec<(bool, &Symbol)> = self
            .symbols
            .iter()
            .filter(|s| s.name.to_lowercase().contains(&lower))
            .map(|s| (s.name == needle, s))
            .collect();
        hits.sort_by(|a, b| {
            (b.0, a.1.file.as_str(), a.1.line).cmp(&(a.0, b.1.file.as_str(), b.1.line))
        });
        hits.into_iter().map(|(_, s)| s).collect()
    }
}

/// A regex that matches `name` only at word boundaries — used for reference
/// lookup and rename proposals so `Foo` does not also match inside `FooBar`.
pub fn word_boundary(name: &str) -> String {
    format!(r"\b{}\b", regex::escape(name))
}

/// Case-insensitive path equality (Windows filenames are case-insensitive).
pub fn paths_equal(a: &Path, b: &Path) -> bool {
    let norm = |p: &Path| p.to_string_lossy().replace('\\', "/").to_lowercase();
    norm(a) == norm(b)
}

/// Drop reference hits that point at this project's own symbol-index file —
/// a reference search must never count the generated `.agent/index.json` as a
/// reference. Uses [`crate::search::GrepMatch`] so callers can feed grep
/// results straight in.
/// Drop hits that point at this project's own symbol-index file — a
/// reference search must never report the generated `.agent/index.json` as a
/// reference. Tolerates both absolute and project-relative hit paths.
pub fn filter_out_own_index(
    root: &Path,
    hits: Vec<crate::search::GrepMatch>,
) -> Vec<crate::search::GrepMatch> {
    let own_abs = SymbolIndex::file_path(root);
    let own_rel = PathBuf::from(INDEX_FILE);
    hits.into_iter()
        .filter(|h| !paths_equal(&h.path, &own_abs) && !paths_equal(&h.path, &own_rel))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(text: &str, path: &str) -> Vec<Symbol> {
        let mut out = Vec::new();
        extract_symbols(
            &path.replace('\\', "/"),
            language_for(Path::new(path)).unwrap(),
            text,
            &mut out,
        );
        out
    }

    #[test]
    fn extracts_rust_symbols() {
        let text = "pub fn parse() -> u8 {\nstruct Thing {\nfn helper() {}\n}";
        let out = extract(text, "lib.rs");
        let names: Vec<&str> = out.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"parse"));
        assert!(names.contains(&"Thing"));
        assert!(names.contains(&"helper"));
        assert!(out.iter().all(|s| s.line >= 1 && s.line <= 3));
    }

    #[test]
    fn extracts_python_class_and_def() {
        let out = extract("class Bot:\n    def talk(self):\n        pass\n", "bot.py");
        let names: Vec<&str> = out.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Bot"));
        assert!(names.contains(&"talk"));
    }

    #[test]
    fn query_is_substring_and_orders_by_file() {
        let idx = SymbolIndex {
            symbols: vec![
                Symbol {
                    name: "Foo".into(),
                    kind: "function".into(),
                    file: "a.rs".into(),
                    line: 3,
                },
                Symbol {
                    name: "Foo".into(),
                    kind: "struct".into(),
                    file: "b.rs".into(),
                    line: 1,
                },
                Symbol {
                    name: "NotUsage".into(),
                    kind: "struct".into(),
                    file: "a.rs".into(),
                    line: 9,
                },
            ],
            ..Default::default()
        };
        let r = idx.query("Foo");
        assert_eq!(r.len(), 2);
        assert!(r[0].file == "a.rs");
    }

    #[test]
    fn word_boundary_escapes_special_chars() {
        let p = word_boundary("a.b");
        assert!(p.contains("\\."));
        assert!(p.starts_with(r"\b"));
        assert!(p.ends_with(r"\b"));
    }

    #[test]
    fn filter_own_index_removes_abs_and_rel_index_paths() {
        use crate::search::GrepMatch;
        let root = Path::new("C:/proj");
        let hits = vec![
            GrepMatch {
                path: PathBuf::from("C:/proj/.agent/index.json"),
                line: 7,
                text: "x".into(),
                project: None,
            },
            GrepMatch {
                path: PathBuf::from(".agent/index.json"),
                line: 9,
                text: "y".into(),
                project: None,
            },
            GrepMatch {
                path: PathBuf::from("src/lib.rs"),
                line: 3,
                text: "z".into(),
                project: None,
            },
        ];
        let kept = filter_out_own_index(root, hits);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].line, 3);
    }

    #[test]
    fn scan_and_roundtrip_index() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join(".agent")).unwrap();
        std::fs::write(root.join("main.rs"), "fn main() {}\nstruct App {}\n").unwrap();
        // a skipped dir / large file should be ignored
        std::fs::create_dir_all(root.join("target")).unwrap();

        let idx = IndexEngine::new(root).scan().unwrap();
        assert!(idx.symbols.iter().any(|s| s.name == "main"));
        assert!(idx.symbols.iter().any(|s| s.name == "App"));

        idx.save(root).unwrap();
        let loaded = SymbolIndex::load(root).unwrap().unwrap();
        assert_eq!(loaded.symbols.len(), idx.symbols.len());
    }
}
