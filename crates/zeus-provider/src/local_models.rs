//! Local model file discovery: scans zeus's own downloads directory plus
//! wherever else model files commonly end up on a machine, so a downloaded
//! model shows up as "detected" without needing any server running.
//!
//! `scan_local_models` covers the controlled locations, in order:
//! 1. `~/.zeus/models/` — where `zeus pull hf ...` (this crate's own
//!    downloader) saves files; always scanned, since we control it.
//! 2. Best-effort common defaults for other tools (LM Studio, Hugging Face
//!    cache, text-generation-webui) — scanned only if they actually exist.
//!    These are guesses about where those tools *usually* put things, not
//!    verified against a real install of each; a wrong guess just means
//!    nothing is found there, not a false result.
//! 3. User-configured `extra_model_dirs` (`AgentSettings`) — the reliable
//!    fallback for anything in a nonstandard location.
//!
//! `scan_system_models` additionally casts a wider net across the user
//! profile (Downloads / Desktop / Documents / a `models` folder) so files
//! saved anywhere obvious on the machine are surfaced too. These broad
//! roots are bounded to a shallow depth and a blocklist of busy subtrees
//! (AppData, node_modules, ...) so a scan can't walk the whole disk.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Extensions treated as "a model file" for scanning purposes.
const MODEL_EXTENSIONS: &[&str] = &["gguf", "safetensors"];

/// How deep to recurse into a targeted root (zeus dir, tool defaults,
/// configured extra dirs) — bounded so a huge or oddly structured directory
/// can't make a scan run away.
const MAX_DEPTH: usize = 6;

/// Shallow depth for the broad user-profile roots (Downloads/Desktop/...).
const HOME_LIKELY_DEPTH: usize = 4;

/// Subdirectory names never descended into on the broad profile scan —
/// caching/tooling trees that are both huge and pathologically unlikely to
/// hold bare model files.
const SKIP_DIR_NAMES: &[&str] = &[
    "AppData",
    "Application Support",
    ".cache",
    ".cargo",
    ".conda",
    ".config",
    ".electron",
    ".git",
    ".gradle",
    ".local",
    ".m2",
    ".npm",
    ".rustup",
    ".venv",
    ".vscode",
    ".yarn",
    "Microsoft",
    "OneDrive",
    "Program Files",
    "Program Files (x86)",
    "System Volume Information",
    "Windows",
    "$Recycle.Bin",
    "node_modules",
    "vendor",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalModelFile {
    pub path: PathBuf,
    pub size_bytes: u64,
    /// Which scan location this came from, e.g. "zeus" or
    /// "lm-studio (best-effort default)" — shown so a wrong best-effort
    /// guess is distinguishable from a deliberately configured location.
    pub source: String,
}

/// Targeted default directories for other local tools, rooted at `home` —
/// existence is checked before scanning, so an absent/wrong guess is
/// silently skipped. `home` is a parameter (rather than calling
/// `dirs::home_dir()` internally) so tests can point it at an isolated fake
/// home instead of the real one.
fn best_effort_default_dirs(home: &Path) -> Vec<(PathBuf, &'static str)> {
    let mut dirs: Vec<(PathBuf, &'static str)> = Vec::new();

    if cfg!(target_os = "windows") {
        dirs.push((
            home.join(".lmstudio").join("models"),
            "lm-studio (best-effort default)",
        ));
        if let Ok(local_app) = std::env::var("LOCALAPPDATA") {
            dirs.push((
                PathBuf::from(local_app)
                    .join("LM Studio")
                    .join("models")
                    .to_path_buf(),
                "lm-studio (best-effort default)",
            ));
        }
    } else if cfg!(target_os = "macos") {
        dirs.push((
            home.join("Library")
                .join("Application Support")
                .join("LM Studio")
                .join("models"),
            "lm-studio (best-effort default)",
        ));
    } else {
        dirs.push((
            home.join(".lmstudio").join("models"),
            "lm-studio (best-effort default)",
        ));
    }

    dirs.push((
        home.join(".cache").join("lm-studio").join("models"),
        "lm-studio (best-effort default)",
    ));

    dirs.push((
        home.join(".cache").join("huggingface").join("hub"),
        "huggingface-cache (best-effort default)",
    ));
    if cfg!(target_os = "macos") {
        dirs.push((
            home.join("Library")
                .join("Caches")
                .join("huggingface")
                .join("hub"),
            "huggingface-cache (best-effort default)",
        ));
    }

    dirs.push((
        home.join("text-generation-webui").join("models"),
        "text-generation-webui (best-effort default)",
    ));

    dirs
}

/// Broad roots inside the user profile where a downloaded model may have
/// simply been dumped. Scanned only by `scan_system_models`, existence
/// checked, shallow depth, skip-list filtered.
fn home_likely_dirs(home: &Path) -> Vec<(PathBuf, &'static str)> {
    vec![
        (home.join("Downloads"), "home/Downloads"),
        (home.join("Desktop"), "home/Desktop"),
        (home.join("Documents"), "home/Documents"),
        (home.join("Models"), "home/Models"),
        (home.join("models"), "home/models"),
    ]
}

/// A scan root: directory, display label, recursion depth, and whether the
/// blocklist filtering applies (used for the broad user-profile roots).
struct Root {
    dir: PathBuf,
    source: String,
    depth: usize,
    filtered: bool,
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
    let mut roots = vec![Root {
        dir: zeus_models_dir.to_path_buf(),
        source: "zeus".to_string(),
        depth: MAX_DEPTH,
        filtered: false,
    }];
    if let Some(home) = home {
        for (dir, label) in best_effort_default_dirs(home) {
            roots.push(Root {
                dir,
                source: label.to_string(),
                depth: MAX_DEPTH,
                filtered: false,
            });
        }
    }
    for dir in extra_dirs {
        roots.push(Root {
            dir: dir.clone(),
            source: "configured extra_model_dirs".to_string(),
            depth: MAX_DEPTH,
            filtered: false,
        });
    }

    let mut found = Vec::new();
    let mut seen = HashSet::new();
    for root in roots {
        if root.dir.is_dir() {
            scan_dir(&root.dir, &root.source, root.depth, root.filtered, &mut found, &mut seen);
        }
    }
    found
}

/// System-wide variant of `scan_local_models`: same targeted roots, plus the
/// user-profile `Downloads` / `Desktop` / `Documents` / `models` folders so a
/// model file that was saved anywhere on the machine surfaces here. Results
/// are deduplicated across overlapping roots.
pub fn scan_system_models(zeus_models_dir: &Path, extra_dirs: &[PathBuf]) -> Vec<LocalModelFile> {
    scan_system_models_from(zeus_models_dir, extra_dirs, dirs::home_dir().as_deref())
}

fn scan_system_models_from(
    zeus_models_dir: &Path,
    extra_dirs: &[PathBuf],
    home: Option<&Path>,
) -> Vec<LocalModelFile> {
    let mut roots = vec![Root {
        dir: zeus_models_dir.to_path_buf(),
        source: "zeus".to_string(),
        depth: MAX_DEPTH,
        filtered: false,
    }];
    if let Some(home) = home {
        for (dir, label) in best_effort_default_dirs(home) {
            roots.push(Root {
                dir,
                source: label.to_string(),
                depth: MAX_DEPTH,
                filtered: false,
            });
        }
        for (dir, label) in home_likely_dirs(home) {
            roots.push(Root {
                dir,
                source: label.to_string(),
                depth: HOME_LIKELY_DEPTH,
                filtered: true,
            });
        }
    }
    for dir in extra_dirs {
        roots.push(Root {
            dir: dir.clone(),
            source: "configured extra_model_dirs".to_string(),
            depth: MAX_DEPTH,
            filtered: false,
        });
    }

    let mut found = Vec::new();
    let mut seen = HashSet::new();
    for root in roots {
        if root.dir.is_dir() {
            scan_dir(&root.dir, &root.source, root.depth, root.filtered, &mut found, &mut seen);
        }
    }
    found
}

fn scan_dir(
    dir: &Path,
    source: &str,
    depth_remaining: usize,
    filtered: bool,
    out: &mut Vec<LocalModelFile>,
    seen: &mut HashSet<String>,
) {
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
        if file_type.is_symlink() {
            continue; // avoids directory loops into the same tree
        }
        if file_type.is_dir() {
            if filtered {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                if SKIP_DIR_NAMES.contains(&name) {
                    continue;
                }
            }
            scan_dir(&path, source, depth_remaining - 1, filtered, out, seen);
        } else if file_type.is_file() {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if MODEL_EXTENSIONS.contains(&ext.as_str()) {
                let key = dedupe_key(&path);
                if seen.insert(key) {
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
}

/// Case-insensitive on Windows so the same path found via two roots isn't
/// reported twice.
fn dedupe_key(path: &Path) -> String {
    let key = path.to_string_lossy().to_string();
    if cfg!(windows) {
        key.to_lowercase()
    } else {
        key
    }
}

/// Bring a discovered model file into the zeus models library: copy (or,
/// with `do_move`, relocate) it into `dest_dir` preserving its file name.
/// Fails if a file of that name already exists in the library, and on a
/// cross-device move it transparently falls back to copy-then-remove.
pub fn import_model_file(
    file: &LocalModelFile,
    dest_dir: &Path,
    do_move: bool,
) -> crate::Result<PathBuf> {
    let name = file
        .path
        .file_name()
        .ok_or_else(|| crate::ProviderError::InvalidRequest(format!(
            "no file name for {}",
            file.path.display()
        )))?;
    std::fs::create_dir_all(dest_dir).map_err(|e| crate::ProviderError::Other(Box::new(e)))?;
    let dest = dest_dir.join(name);
    if dest.exists() {
        return Err(crate::ProviderError::InvalidRequest(format!(
            "a model named '{}' already exists in {}",
            name.to_string_lossy(),
            dest_dir.display()
        )));
    }
    if do_move {
        if std::fs::rename(&file.path, &dest).is_err() {
            // Cross-device move: fall back to copy + delete.
            std::fs::copy(&file.path, &dest).map_err(|e| {
                crate::ProviderError::Other(Box::new(e))
            })?;
            std::fs::remove_file(&file.path).map_err(|e| crate::ProviderError::Other(Box::new(e)))?;
        }
    } else {
        std::fs::copy(&file.path, &dest).map_err(|e| crate::ProviderError::Other(Box::new(e)))?;
    }
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_fake(dir: &Path, rel: &str, contents: &[u8]) {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, contents).unwrap();
    }

    #[test]
    fn finds_gguf_and_safetensors_recursively() {
        let tmp = TempDir::new().unwrap();
        let models = tmp.path().join("models");
        std::fs::create_dir_all(&models).unwrap();
        write_fake(&models, "a.gguf", b"fake");
        write_fake(&models, "nested/b.safetensors", b"fake2");
        write_fake(&models, "readme.txt", b"not a model");

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
        // `home: None` skips home-root scanning entirely — also exercises
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

        let found = scan_local_models_from(&zeus_dir, std::slice::from_ref(&extra), None);
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

    #[test]
    fn system_scan_finds_models_in_profile_locations() {
        let tmp = TempDir::new().unwrap();
        let zeus_dir = tmp.path().join("zeus-models");
        std::fs::create_dir_all(&zeus_dir).unwrap();
        let fake_home = tmp.path().join("home");
        std::fs::create_dir_all(fake_home.join("Downloads")).unwrap();
        std::fs::create_dir_all(fake_home.join(".cache").join("lm-studio").join("models")).unwrap();
        write_fake(&fake_home, "Downloads/phi3.gguf", b"m1");
        write_fake(
            &fake_home,
            ".cache/lm-studio/models/qwen.safetensors",
            b"m2",
        );

        let found = scan_system_models_from(&zeus_dir, &[], Some(&fake_home));
        let sources: Vec<&str> = found.iter().map(|f| f.source.as_str()).collect();
        assert!(sources.contains(&"home/Downloads"));
        assert!(sources.contains(&"lm-studio (best-effort default)"));
    }

    #[test]
    fn system_scan_dedupes_overlapping_roots() {
        let tmp = TempDir::new().unwrap();
        let zeus_dir = tmp.path().join("zeus-models");
        std::fs::create_dir_all(&zeus_dir).unwrap();
        let fake_home = tmp.path().join("home");
        std::fs::create_dir_all(fake_home.join("Downloads")).unwrap();
        write_fake(&fake_home, "Downloads/mistral.gguf", b"m1");

        // Downloads is both a profile root and a configured extra dir.
        let extra = fake_home.join("Downloads");
        let found = scan_system_models_from(&zeus_dir, std::slice::from_ref(&extra), Some(&fake_home));
        assert_eq!(found.len(), 1, "same file via two roots reported once");
    }

    #[test]
    fn import_copies_and_moves_file_into_library() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        write_fake(&src, "n/model.gguf", b"model-bytes");
        let lib = tmp.path().join("lib");

        let file = LocalModelFile {
            path: src.join("n").join("model.gguf"),
            size_bytes: 11,
            source: "test".into(),
        };

        let copied = import_model_file(&file, &lib, false).unwrap();
        assert_eq!(copied.file_name().unwrap().to_string_lossy(), "model.gguf");
        assert!(lib.join("model.gguf").exists());

        // Second copy refuses to clobber.
        assert!(import_model_file(&file, &lib, false).is_err());

        // Move relocates and leaves nothing behind.
        let lib2 = tmp.path().join("lib2");
        let other = LocalModelFile {
            path: src.join("n").join("model.gguf"),
            size_bytes: 11,
            source: "test".into(),
        };
        import_model_file(&other, &lib2, true).unwrap();
        assert!(lib2.join("model.gguf").exists());
        assert!(!other.path.exists());
    }
}