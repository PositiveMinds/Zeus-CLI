//! Local model file discovery: scans zeus's own downloads directory plus a
//! few other places local model files commonly end up, so a downloaded
//! model shows up as "detected" without needing any server running.
//!
//! Scan locations, in order:
//! 1. `~/.zeus/models/` — where `zeus pull hf ...` (this crate's own
//!    downloader) saves files; always scanned, since we control it.
//! 2. Best-effort common defaults for other tools (LM Studio, a Hugging
//!    Face cache) — scanned only if they actually exist. These are guesses
//!    about where those tools *usually* put things, not verified against a
//!    real install of each; a wrong guess just means nothing is found
//!    there, not a false result.
//! 3. User-configured `extra_model_dirs` (`AgentSettings`) — the reliable
//!    fallback for anything in a nonstandard location, since guessing can't
//!    cover every setup.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Extensions treated as "a model file" for scanning purposes.
const MODEL_EXTENSIONS: &[&str] = &["gguf", "safetensors"];

/// How deep to recurse into each scan root — bounded so a huge or oddly
/// structured directory (e.g. accidentally pointing at `/`) can't make a
/// scan run away.
const MAX_DEPTH: usize = 6;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalModelFile {
    pub path: PathBuf,
    pub size_bytes: u64,
    /// Which scan location this came from, e.g. "zeus" or
    /// "lm-studio (best-effort default)" — shown so a wrong best-effort
    /// guess is distinguishable from a deliberately configured location.
    pub source: String,
}

/// Best-effort default directories for other local tools, rooted at
/// `home` — existence is checked before scanning, so an absent/wrong guess
/// is silently skipped rather than reported as an error. Takes `home`
/// explicitly (rather than calling `dirs::home_dir()` internally) so tests
/// can point it at an isolated fake home instead of the real one — on a
/// machine that actually has LM Studio or a Hugging Face cache installed,
/// scanning the *real* home directory in a test would pick up real files
/// and make the test's expectations environment-dependent.
fn best_effort_default_dirs(home: &Path) -> Vec<(PathBuf, &'static str)> {
    vec![
        (
            home.join(".cache").join("lm-studio").join("models"),
            "lm-studio (best-effort default)",
        ),
        (
            home.join(".lmstudio").join("models"),
            "lm-studio (best-effort default)",
        ),
        (
            home.join(".cache").join("huggingface").join("hub"),
            "huggingface-cache (best-effort default)",
        ),
    ]
}

/// Scan `zeus_models_dir` (always) plus best-effort default locations and
/// `extra_dirs` (both only if they exist) for model files. Never errors —
/// an unreadable/missing directory is just skipped, same "one bad location
/// shouldn't block the rest" policy used for MCP servers and plugins.
pub fn scan_local_models(zeus_models_dir: &Path, extra_dirs: &[PathBuf]) -> Vec<LocalModelFile> {
    scan_local_models_from(zeus_models_dir, extra_dirs, dirs::home_dir().as_deref())
}

fn scan_local_models_from(
    zeus_models_dir: &Path,
    extra_dirs: &[PathBuf],
    home: Option<&Path>,
) -> Vec<LocalModelFile> {
    let mut roots: Vec<(PathBuf, String)> =
        vec![(zeus_models_dir.to_path_buf(), "zeus".to_string())];
    if let Some(home) = home {
        for (dir, label) in best_effort_default_dirs(home) {
            roots.push((dir, label.to_string()));
        }
    }
    for dir in extra_dirs {
        roots.push((dir.clone(), "configured extra_model_dirs".to_string()));
    }

    let mut found = Vec::new();
    for (root, source) in roots {
        if !root.is_dir() {
            continue;
        }
        scan_dir(&root, &source, MAX_DEPTH, &mut found);
    }
    found
}

fn scan_dir(dir: &Path, source: &str, depth_remaining: usize, out: &mut Vec<LocalModelFile>) {
    if depth_remaining == 0 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            scan_dir(&path, source, depth_remaining - 1, out);
        } else if file_type.is_file() {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if MODEL_EXTENSIONS.contains(&ext.as_str()) {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                out.push(LocalModelFile {
                    path,
                    size_bytes: size,
                    source: source.to_string(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn finds_gguf_and_safetensors_recursively() {
        let tmp = TempDir::new().unwrap();
        let models = tmp.path().join("models");
        std::fs::create_dir_all(models.join("nested")).unwrap();
        std::fs::write(models.join("a.gguf"), b"fake").unwrap();
        std::fs::write(models.join("nested").join("b.safetensors"), b"fake2").unwrap();
        std::fs::write(models.join("readme.txt"), b"not a model").unwrap();

        let fake_home = tmp.path().join("fake-home");
        std::fs::create_dir_all(&fake_home).unwrap();
        let found = scan_local_models_from(&models, &[], Some(&fake_home));
        let names: Vec<String> = found
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"a.gguf".to_string()));
        assert!(names.contains(&"b.safetensors".to_string()));
        assert!(!names.contains(&"readme.txt".to_string()));
        assert!(found.iter().all(|f| f.source == "zeus"));
    }

    #[test]
    fn missing_zeus_dir_does_not_error_just_finds_nothing_there() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("does-not-exist");
        // `home: None` skips best-effort scanning entirely — also exercises
        // the "no home dir found" edge case, not just an isolated one.
        let found = scan_local_models_from(&missing, &[], None);
        assert!(found.is_empty());
    }

    #[test]
    fn scans_configured_extra_dirs() {
        let tmp = TempDir::new().unwrap();
        let zeus_dir = tmp.path().join("zeus-models");
        std::fs::create_dir_all(&zeus_dir).unwrap();
        let extra = tmp.path().join("my-other-models");
        std::fs::create_dir_all(&extra).unwrap();
        std::fs::write(extra.join("custom.gguf"), b"fake").unwrap();

        let found = scan_local_models_from(&zeus_dir, &[extra.clone()], None);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].source, "configured extra_model_dirs");
    }

    #[test]
    fn depth_limit_prevents_runaway_recursion() {
        let tmp = TempDir::new().unwrap();
        let mut deep = tmp.path().join("models");
        std::fs::create_dir_all(&deep).unwrap();
        for _ in 0..(MAX_DEPTH + 3) {
            deep = deep.join("d");
            std::fs::create_dir_all(&deep).unwrap();
        }
        std::fs::write(deep.join("too-deep.gguf"), b"fake").unwrap();

        let found = scan_local_models_from(&tmp.path().join("models"), &[], None);
        assert!(found.is_empty(), "file beyond MAX_DEPTH should not be found");
    }
}
