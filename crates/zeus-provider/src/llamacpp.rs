//! llama.cpp serving: resolve (or download) the `llama-server` binary, make
//! sure a GGUF model file is on disk (optionally pulling it into
//! `~/.zeus/models/`), then spawn a detached llama-server on a local port
//! that the OpenAI-compatible `llamacpp` provider can talk to.
//!
//! This is what turns "pick a local model" into something that works
//! end-to-end: zeus finds `llama-server` on PATH (or downloads a llama.cpp
//! release build into `~/.zeus/bin/` on first use), fetches the model file
//! if missing, launches the server, and waits until it answers `/v1/models`
//! before returning the server's origin to the caller.

use crate::detect::is_reachable;
use crate::download::download_hf_file;
use crate::error::{ProviderError, Result};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use zeus_config::{GlobalPaths, LlamaCppSettings, LocalModelEntry};

/// How deep to search an extracted binary dir for `llama-server`.
const MAX_SEARCH_DEPTH: usize = 3;

/// Names `llama-server` is installed as on each platform.
fn server_binary_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["llama-server.exe", "llama-server"]
    } else {
        &["llama-server"]
    }
}

/// Search PATH for any of `names` and return the first hit, if any.
pub fn find_on_path(names: &[&str]) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for name in names {
            let full = dir.join(name);
            if full.is_file() {
                return Some(full);
            }
        }
    }
    None
}

/// Recursively find a file named like `llama-server` under `dir`, bounded in
/// depth so a stray huge tree can't make the scan run away.
fn find_server_binary(dir: &Path, depth: usize) -> Option<PathBuf> {
    if depth == 0 {
        return None;
    }
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_server_binary(&path, depth - 1) {
                return Some(found);
            }
        } else {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if server_binary_names().contains(&name) && path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

/// Return a previously-downloaded llama-server under `bin_dir`, if any.
fn find_extracted_server(bin_dir: &Path) -> Option<PathBuf> {
    find_server_binary(bin_dir, MAX_SEARCH_DEPTH)
}

/// Which llama.cpp release archive this platform should use, e.g.
/// `win-cpu-x64` / `ubuntu-x64` / `macos-arm64`. Matched by substring against
/// the release's asset names.
fn platform_asset_key() -> &'static str {
    if cfg!(target_os = "windows") {
        "win-cpu-x64"
    } else if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            "macos-arm64"
        } else {
            "macos-x64"
        }
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        "ubuntu-arm64"
    } else {
        "ubuntu-x64"
    }
}

/// Info about a running llama-server.
#[derive(Debug, Clone)]
pub struct ServerInfo {
    /// Base origin, e.g. `http://127.0.0.1:8080`.
    pub origin: String,
    /// Process id (0 when we re-used an already-running server).
    pub pid: u32,
}

/// The best llama-server we can find *without* hitting the network: an
/// explicitly-configured path, then PATH, then a previously-downloaded copy.
/// Returns `None` if none exist so the caller can fall back to downloading.
pub fn locate_server_binary(binary_override: Option<&str>, bin_dir: &Path) -> Option<PathBuf> {
    if let Some(p) = binary_override {
        let path = PathBuf::from(p);
        return path.is_file().then_some(path);
    }
    if let Some(p) = find_on_path(server_binary_names()) {
        return Some(p);
    }
    find_extracted_server(bin_dir)
}

/// Resolve the `llama-server` binary, downloading and extracting a llama.cpp
/// release into `bin_dir` if it isn't already available locally.
pub async fn ensure_server_binary(
    binary_override: Option<&str>,
    bin_dir: &Path,
) -> Result<PathBuf> {
    if let Some(found) = locate_server_binary(binary_override, bin_dir) {
        return Ok(found);
    }
    download_server_binary(bin_dir).await
}

/// Stream a URL body to `dest` (async, so it works inside the tokio runtime
/// this crate's callers run on).
async fn stream_download(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| ProviderError::Transport(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(ProviderError::Api(format!(
            "binary download failed ({}) for {url}",
            resp.status()
        )));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| ProviderError::Transport(e.to_string()))?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| ProviderError::Api(format!("create bin dir: {e}")))?;
    }
    std::fs::write(dest, &bytes).map_err(|e| ProviderError::Api(format!("write binary: {e}")))?;
    Ok(())
}

/// Query the llama.cpp GitHub "latest" release, pick the archive matching
/// this platform, download + unzip it into `bin_dir`, and return the path to
/// the extracted `llama-server`.
async fn download_server_binary(bin_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(bin_dir)
        .map_err(|e| ProviderError::Api(format!("create bin dir: {e}")))?;

    let api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";
    let client = reqwest::Client::new();
    let resp = client
        .get(api_url)
        .header("User-Agent", "zeus-cli")
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| ProviderError::Transport(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(ProviderError::Api(format!(
            "failed to query llama.cpp latest release ({})",
            resp.status()
        )));
    }
    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| ProviderError::Api(format!("parse release json: {e}")))?;
    let tag = json["tag_name"]
        .as_str()
        .ok_or_else(|| ProviderError::Api("release missing tag_name".into()))?
        .to_string();
    let assets = json["assets"]
        .as_array()
        .ok_or_else(|| ProviderError::Api("release missing assets".into()))?;

    let key = platform_asset_key();
    let asset = assets
        .iter()
        .find(|a| {
            let name = a["name"].as_str().unwrap_or("");
            name.contains(key) && name.ends_with(".zip")
        })
        .ok_or_else(|| {
            ProviderError::Api(format!(
                "no llama.cpp '{key}' .zip asset found for release {tag}"
            ))
        })?;
    let file_name = asset["name"].as_str().unwrap_or("llama-server.zip");
    let url = asset["browser_download_url"]
        .as_str()
        .ok_or_else(|| ProviderError::Api("asset missing download url".into()))?;

    let zip_path = bin_dir.join(file_name);
    stream_download(url, &zip_path).await?;

    let extract_dir = bin_dir.join(file_name.trim_end_matches(".zip"));
    if !extract_dir.is_dir() {
        unzip(&zip_path, &extract_dir)?;
    }
    // Best-effort record of the version we installed (informational).
    let _ = std::fs::write(bin_dir.join("llama-server.version"), tag);

    find_server_binary(&extract_dir, MAX_SEARCH_DEPTH).ok_or_else(|| {
        ProviderError::Api("extracted llama.cpp archive contained no llama-server".into())
    })
}

/// Extract a `.zip` into `dest`, preserving paths and marking executables on
/// unix.
fn unzip(src: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest)
        .map_err(|e| ProviderError::Api(format!("create extract dir: {e}")))?;
    let file =
        std::fs::File::open(src).map_err(|e| ProviderError::Api(format!("open zip: {e}")))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| ProviderError::Api(format!("read zip: {e}")))?;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| ProviderError::Api(format!("zip entry {i}: {e}")))?;
        let rel = entry.name().replace('\\', "/");
        let out_path = dest.join(&rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&out_path).ok();
            continue;
        }
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let mut out = std::fs::File::create(&out_path)
            .map_err(|e| ProviderError::Api(format!("write {rel}: {e}")))?;
        std::io::copy(&mut entry, &mut out)
            .map_err(|e| ProviderError::Api(format!("extract {rel}: {e}")))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if rel.ends_with("llama-server") || rel.ends_with(".so") || rel.ends_with(".dylib") {
                let _ = std::fs::set_permissions(&out_path, std::fs::Permissions::from_mode(0o755));
            }
        }
    }
    Ok(())
}

/// Return the path to a GGUF model file for `repo / file`, downloading it into
/// `models_dir` first if it isn't already there.
pub async fn ensure_model_file(repo: &str, file: &str, models_dir: &Path) -> Result<PathBuf> {
    let name = Path::new(file)
        .file_name()
        .ok_or_else(|| ProviderError::InvalidRequest("empty model filename".into()))?;
    let path = models_dir.join(name);
    if path.is_file() {
        return Ok(path);
    }
    download_hf_file(repo, file, models_dir, |_, _| {}).await?;
    Ok(path)
}

/// Given a resolved binary and a model file path, spawn a detached
/// llama-server on `port` and wait until `/v1/models` answers. Logs go to
/// `logs_dir`. Returns once the server is answering requests.
pub async fn spawn_server_and_wait(
    binary: &Path,
    model_path: &Path,
    port: u16,
    logs_dir: &Path,
    threads: Option<u32>,
    ctx: Option<u32>,
    timeout: Duration,
) -> Result<ServerInfo> {
    std::fs::create_dir_all(logs_dir)
        .map_err(|e| ProviderError::Api(format!("create logs dir: {e}")))?;
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(logs_dir.join("llamacpp.stdout.log"))
        .map_err(|e| ProviderError::Api(format!("open stdout log: {e}")))?;
    let stderr = OpenOptions::new()
        .create(true)
        .append(true)
        .open(logs_dir.join("llamacpp.stderr.log"))
        .map_err(|e| ProviderError::Api(format!("open stderr log: {e}")))?;

    let mut cmd = Command::new(binary);
    cmd.arg("-m").arg(model_path);
    cmd.arg("--host").arg("127.0.0.1");
    cmd.arg("--port").arg(port.to_string());
    if let Some(t) = threads {
        cmd.args(["--threads", &t.to_string()]);
    }
    if let Some(c) = ctx {
        cmd.args(["--ctx-size", &c.to_string()]);
    }
    cmd.current_dir(logs_dir);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::from(stdout));
    cmd.stderr(Stdio::from(stderr));

    let child = cmd
        .spawn()
        .map_err(|e| ProviderError::Api(format!("spawn llama-server: {e}")))?;
    let pid = child.id();
    // Deliberately drop the handle: this detaches the server so it outlives
    // the `zeus` process, exactly like the background-task registry.
    drop(child);

    let origin = format!("http://127.0.0.1:{port}");
    let probe = format!("{origin}/v1/models");
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if is_reachable(&probe, Duration::from_millis(200)).await {
            return Ok(ServerInfo { origin, pid });
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    Err(ProviderError::Api(format!(
        "llama-server (pid {pid}) did not become ready at {origin}; check {}",
        logs_dir.join("llamacpp.stderr.log").display()
    )))
}

/// A small built-in catalog of common GGUF models so a bare name like
/// `llama3.2` can be auto-downloaded + served with no configuration. Users
/// can override/extend this via `[settings.llamacpp.models]`. Each entry is
/// `(name, hf_repo, hf_file)`.
pub const DEFAULT_MODEL_CATALOG: &[(&str, &str, &str)] = &[
    (
        "llama3.2",
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    ),
    (
        "qwen2.5-7b",
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    ),
    (
        "gemma3-4b",
        "bartowski/gemma-3-4b-it-GGUF",
        "gemma-3-4b-it-Q4_K_M.gguf",
    ),
];

/// Look up a model by name: a configured `settings.models` entry first, then
/// the built-in catalog. Returns an owned entry for either source.
pub fn resolve_local_model(settings: &LlamaCppSettings, name: &str) -> Option<LocalModelEntry> {
    if let Some(m) = settings.models.iter().find(|m| m.name == name) {
        return Some(m.clone());
    }
    DEFAULT_MODEL_CATALOG
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, repo, file)| LocalModelEntry {
            name: name.to_string(),
            repo: (*repo).to_string(),
            file: (*file).to_string(),
        })
}

/// One-call helper for the CLI: ensure the server binary and `model` are
/// present, then start (or reuse) a ready llama-server for it. Returns the
/// origin to point a `llamacpp` provider at.
pub async fn serve(
    cfg: &LlamaCppSettings,
    model: &LocalModelEntry,
    global: &GlobalPaths,
) -> Result<ServerInfo> {
    let port = cfg.port;
    let origin = format!("http://127.0.0.1:{port}");
    // If something is already answering on this port, reuse it.
    if is_reachable(&format!("{origin}/v1/models"), Duration::from_millis(300)).await {
        return Ok(ServerInfo { origin, pid: 0 });
    }
    let binary = ensure_server_binary(cfg.binary.as_deref(), &global.bin).await?;
    let model_path = ensure_model_file(&model.repo, &model.file, &global.models).await?;
    spawn_server_and_wait(
        &binary,
        &model_path,
        port,
        &global.logs,
        cfg.threads,
        cfg.ctx,
        Duration::from_secs(60),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn platform_asset_key_is_known() {
        assert!(!platform_asset_key().is_empty());
    }

    #[test]
    fn server_names_include_platform_expected() {
        assert!(!server_binary_names().is_empty());
        if cfg!(windows) {
            assert!(server_binary_names().contains(&"llama-server.exe"));
        } else {
            assert!(server_binary_names().contains(&"llama-server"));
        }
    }

    #[test]
    fn empty_bin_dir_yields_no_server() {
        let tmp = TempDir::new().unwrap();
        assert!(find_extracted_server(tmp.path()).is_none());
    }

    #[test]
    fn finds_server_nested_under_bin_dir() {
        let tmp = TempDir::new().unwrap();
        let exe = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        let dir = tmp.path().join("llama-b5144-bin-win-cpu-x64");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(exe), b"bin").unwrap();
        let found = find_server_binary(tmp.path(), MAX_SEARCH_DEPTH);
        assert!(found.is_some());
        assert_eq!(found.unwrap().file_name().unwrap(), exe);
    }

    #[test]
    fn locate_prefers_override_when_it_exists() {
        let tmp = TempDir::new().unwrap();
        let bin = tmp.path().join(if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        });
        std::fs::write(&bin, b"bin").unwrap();
        let found = locate_server_binary(Some(bin.to_str().unwrap()), tmp.path());
        assert_eq!(found, Some(bin));
    }

    #[test]
    fn missing_override_returns_none() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("nope");
        assert!(locate_server_binary(Some(missing.to_str().unwrap()), tmp.path()).is_none());
    }
}
