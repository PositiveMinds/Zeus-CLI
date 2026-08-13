//! Grep / glob search within a project (and optional cross-project fan-out).

use crate::error::{FsError, Result};
use crate::pathutil::resolve_in_project;
use crate::permission::{PermissionGate, PermissionRequest};
use globset::{Glob, GlobSetBuilder};
use ignore::WalkBuilder;
use regex::Regex;
use std::path::{Path, PathBuf};

/// Skip files larger than this while grepping — a single multi-hundred-MB
/// minified bundle or vendored artifact shouldn't stall a project-wide scan
/// or blow the result set. Large projects routinely contain such files.
const GREP_MAX_FILE_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Regex pattern for content search.
    pub pattern: String,
    /// Optional glob to filter files (e.g. "*.rs").
    pub glob: Option<String>,
    /// Case insensitive.
    pub case_insensitive: bool,
    /// Max matches to return.
    pub max_matches: usize,
    /// Directory relative to project root (default ".").
    pub path: Option<PathBuf>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            pattern: String::new(),
            glob: None,
            case_insensitive: false,
            max_matches: 200,
            path: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrepMatch {
    pub path: PathBuf,
    pub line: usize,
    pub text: String,
    pub project: Option<String>,
}

/// Accumulator for grep results with a result cap shared across roots.
struct GrepSink {
    out: Vec<GrepMatch>,
    max: usize,
}

impl GrepSink {
    fn full(&self) -> bool {
        self.out.len() >= self.max
    }
}

#[derive(Debug, Clone)]
pub struct GlobMatch {
    pub path: PathBuf,
    pub project: Option<String>,
}

pub struct SearchEngine {
    pub project_root: PathBuf,
    pub gate: PermissionGate,
    /// Extra project roots for cross-project search.
    pub extra_roots: Vec<PathBuf>,
}

impl SearchEngine {
    pub fn new(project_root: PathBuf, gate: PermissionGate, extra_roots: Vec<PathBuf>) -> Self {
        Self {
            project_root,
            gate,
            extra_roots,
        }
    }

    pub fn grep(&self, opts: SearchOptions) -> Result<Vec<GrepMatch>> {
        self.gate.enforce_strict(&PermissionRequest {
            tool: "search".into(),
            path: opts.path.clone(),
            command: None,
            description: format!("grep /{}/", opts.pattern),
            ..Default::default()
        })?;

        let mut regex_builder = regex::RegexBuilder::new(&opts.pattern);
        regex_builder.case_insensitive(opts.case_insensitive);
        let re = regex_builder
            .build()
            .map_err(|e| FsError::InvalidPath(format!("invalid regex: {e}")))?;

        let file_glob = opts
            .glob
            .as_ref()
            .map(|g| Glob::new(g).map(|g| g.compile_matcher()))
            .transpose()
            .map_err(|e| FsError::InvalidPath(format!("invalid glob: {e}")))?;

        let mut sink = GrepSink {
            out: Vec::new(),
            max: opts.max_matches,
        };
        let roots = self.collect_roots();
        for (label, root) in roots {
            let start = match &opts.path {
                Some(p) if label.is_none() => resolve_in_project(&root, p)?,
                _ => root.clone(),
            };
            self.grep_root(
                &start,
                &root,
                label.as_deref(),
                &re,
                file_glob.as_ref(),
                &mut sink,
            )?;
            if sink.out.len() >= opts.max_matches {
                break;
            }
        }
        Ok(sink.out)
    }

    fn grep_root(
        &self,
        start: &Path,
        project_root: &Path,
        project_label: Option<&str>,
        re: &Regex,
        file_glob: Option<&globset::GlobMatcher>,
        sink: &mut GrepSink,
    ) -> Result<()> {
        let walker = WalkBuilder::new(start)
            .hidden(false)
            .git_ignore(true)
            .git_global(false)
            .build();

        for entry in walker {
            if sink.full() {
                break;
            }
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let path = entry.path();
            if let Some(g) = file_glob {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !g.is_match(name) && !g.is_match(path) {
                    continue;
                }
            }
            // Skip oversized files (minified bundles, vendored deps, dumps).
            if let Ok(meta) = std::fs::metadata(path) {
                if meta.len() > GREP_MAX_FILE_BYTES {
                    continue;
                }
            }
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            if bytes.contains(&0) {
                continue;
            }
            let text = String::from_utf8_lossy(&bytes);
            for (i, line) in text.lines().enumerate() {
                if sink.full() {
                    break;
                }
                if re.is_match(line) {
                    let rel = path
                        .strip_prefix(project_root)
                        .unwrap_or(path)
                        .to_path_buf();
                    sink.out.push(GrepMatch {
                        path: rel,
                        line: i + 1,
                        text: line.to_string(),
                        project: project_label.map(|s| s.to_string()),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn glob(&self, pattern: &str, max: usize) -> Result<Vec<GlobMatch>> {
        self.gate.enforce_strict(&PermissionRequest {
            tool: "search".into(),
            path: None,
            command: None,
            description: format!("glob {pattern}"),
            ..Default::default()
        })?;

        let mut builder = GlobSetBuilder::new();
        builder.add(Glob::new(pattern).map_err(|e| FsError::InvalidPath(e.to_string()))?);
        let set = builder
            .build()
            .map_err(|e| FsError::InvalidPath(e.to_string()))?;

        let mut out = Vec::new();
        for (label, root) in self.collect_roots() {
            let walker = WalkBuilder::new(&root)
                .hidden(false)
                .git_ignore(true)
                .build();
            for entry in walker {
                if out.len() >= max {
                    break;
                }
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                let path = entry.path();
                let rel = path.strip_prefix(&root).unwrap_or(path);
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                if set.is_match(&rel_str) || set.is_match(path) {
                    out.push(GlobMatch {
                        path: rel.to_path_buf(),
                        project: label.clone(),
                    });
                }
            }
        }
        Ok(out)
    }

    fn collect_roots(&self) -> Vec<(Option<String>, PathBuf)> {
        let mut roots = vec![(None, self.project_root.clone())];
        for r in &self.extra_roots {
            if r != &self.project_root && r.is_dir() {
                let label = r
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| r.display().to_string());
                roots.push((Some(label), r.clone()));
            }
        }
        roots
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use zeus_config::AgentSettings;

    #[test]
    fn grep_finds_line() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "fn foo() {}\nfn bar() {}\n").unwrap();
        let gate = PermissionGate::new(AgentSettings::default(), root.clone());
        let eng = SearchEngine::new(root, gate, vec![]);
        let hits = eng
            .grep(SearchOptions {
                pattern: "fn bar".into(),
                glob: Some("*.rs".into()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line, 2);
    }

    #[test]
    fn glob_finds_files() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/a.rs"), "").unwrap();
        std::fs::write(root.join("src/b.txt"), "").unwrap();
        let gate = PermissionGate::new(AgentSettings::default(), root.clone());
        let eng = SearchEngine::new(root, gate, vec![]);
        let hits = eng.glob("**/*.rs", 50).unwrap();
        assert_eq!(hits.len(), 1);
    }
}
