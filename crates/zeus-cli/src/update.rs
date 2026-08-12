//! Self-update support, modeled on how opencode's CLI does it: detect which
//! channel the running binary actually came from, ask that channel's own
//! source what the latest version is, then delegate the upgrade to that
//! channel's own tooling rather than trying to patch the running executable
//! in place. Zeus only ships one real distribution channel today (the
//! install.ps1/install.bat/install.sh script, downloading a prebuilt binary
//! from the public releases mirror) — a `cargo install`/source checkout is
//! treated as "update it yourself" rather than guessed at, since there's no
//! guarantee the exe came from a repo this process can re-fetch.

use anyhow::{bail, Context, Result};
use serde::Deserialize;

/// The prebuilt binaries are mirrored to a separate public repo (see
/// `.github/workflows/release.yml`) because the main repo's default
/// `GITHUB_TOKEN` can't publish releases cross-repo — this is also the
/// repo the install scripts themselves already point at.
pub const RELEASES_OWNER: &str = "PositiveMinds";
pub const RELEASES_REPO: &str = "Zeus-CLI-releases";

pub fn current_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallMethod {
    /// Installed by install.ps1/install.bat/install.sh into the fixed
    /// per-OS directory those scripts use — safe to re-run to upgrade.
    Script,
    /// A `cargo build`/`cargo install` output (dev checkout or `~/.cargo/bin`)
    /// — no repo access assumed, so this is surfaced but not auto-upgraded.
    Cargo,
    Unknown,
}

impl InstallMethod {
    pub fn label(self) -> &'static str {
        match self {
            InstallMethod::Script => "install script",
            InstallMethod::Cargo => "cargo",
            InstallMethod::Unknown => "unknown",
        }
    }
}

/// Where install.ps1/install.bat put the binary on Windows, and where
/// install.sh (added alongside this) puts it on Unix.
fn script_install_dir() -> Option<std::path::PathBuf> {
    if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(|d| std::path::PathBuf::from(d).join("zeus"))
    } else {
        dirs_home().map(|h| h.join(".local").join("share").join("zeus").join("bin"))
    }
}

fn dirs_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

pub fn detect_install_method() -> InstallMethod {
    let Ok(exe) = std::env::current_exe() else {
        return InstallMethod::Unknown;
    };
    let exe = match exe.canonicalize() {
        Ok(p) => p,
        Err(_) => exe,
    };
    if let Some(dir) = script_install_dir() {
        if let Ok(dir) = dir.canonicalize() {
            if exe.starts_with(&dir) {
                return InstallMethod::Script;
            }
        } else if exe.starts_with(&dir) {
            return InstallMethod::Script;
        }
    }
    let s = exe.to_string_lossy().to_ascii_lowercase();
    if s.contains(".cargo") || s.contains("target/debug") || s.contains("target/release")
        || s.contains("target\\debug") || s.contains("target\\release")
    {
        return InstallMethod::Cargo;
    }
    InstallMethod::Unknown
}

#[derive(Deserialize)]
struct GhRelease {
    tag_name: String,
}

/// Latest published version, tag-prefix (`v`) stripped.
pub async fn latest_version() -> Result<String> {
    let url = format!(
        "https://api.github.com/repos/{RELEASES_OWNER}/{RELEASES_REPO}/releases/latest"
    );
    let client = reqwest::Client::builder()
        .user_agent(format!("zeus-cli/{}", current_version()))
        .build()
        .context("build http client")?;
    let resp = client
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .context("request latest release")?
        .error_for_status()
        .context("GitHub API error — no release published yet?")?;
    let rel: GhRelease = resp.json().await.context("parse release response")?;
    Ok(rel.tag_name.trim_start_matches('v').to_string())
}

/// Parses a `major.minor.patch`-shaped version into a comparable tuple;
/// unparsable segments fall back to 0 rather than failing, since a garbled
/// version string should compare as "not obviously newer" rather than crash.
fn parse_version(v: &str) -> (u64, u64, u64) {
    let mut parts = v.split('.').map(|p| p.parse::<u64>().unwrap_or(0));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

pub fn is_newer(latest: &str, current: &str) -> bool {
    parse_version(latest) > parse_version(current)
}

/// Re-runs the published install script, pinned to `target_version` via the
/// same `ZEUS_VERSION` env var the scripts already read — so a release that
/// lands mid-upgrade can't leave the machine on a half-applied version.
pub fn run_script_update(target_version: &str) -> Result<()> {
    let status = if cfg!(windows) {
        let script_url = format!(
            "https://raw.githubusercontent.com/{RELEASES_OWNER}/{RELEASES_REPO}/main/install.ps1"
        );
        std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                &format!("$env:ZEUS_VERSION = '{target_version}'; irm {script_url} | iex"),
            ])
            .status()
            .context("spawn powershell to run install.ps1")?
    } else {
        let script_url = format!(
            "https://raw.githubusercontent.com/{RELEASES_OWNER}/{RELEASES_REPO}/main/install.sh"
        );
        std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("curl -fsSL {script_url} | ZEUS_VERSION={target_version} sh"))
            .status()
            .context("spawn shell to run install.sh")?
    };
    if !status.success() {
        bail!("install script exited with {status}");
    }
    Ok(())
}

/// `zeus update [--check]`: report on / apply the latest release, based on
/// how this binary was actually installed.
pub async fn cmd_update(check_only: bool) -> Result<()> {
    let current = current_version();
    println!("current version: {current}");

    let method = detect_install_method();
    let latest = latest_version().await?;

    if !is_newer(&latest, current) {
        println!("already up to date (latest: {latest}).");
        return Ok(());
    }

    println!("update available: {current} -> {latest} (installed via {})", method.label());

    if check_only {
        println!("run `zeus update` (without --check) to install it.");
        return Ok(());
    }

    match method {
        InstallMethod::Script => {
            println!("re-running the install script, pinned to {latest}...");
            run_script_update(&latest)?;
            println!("updated to {latest}. Restart zeus to use it.");
        }
        InstallMethod::Cargo => {
            println!(
                "this binary looks like a cargo build/install, not the published installer — \
                 update it the same way you built it (e.g. `cargo build --release -p zeus-cli` \
                 after pulling, or `cargo install` from wherever you originally installed it)."
            );
        }
        InstallMethod::Unknown => {
            println!(
                "couldn't tell how zeus was installed here — reinstall manually from \
                 https://github.com/{RELEASES_OWNER}/{RELEASES_REPO}/releases/tag/v{latest}"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_compare() {
        assert!(is_newer("1.2.0", "1.1.9"));
        assert!(is_newer("2.0.0", "1.9.9"));
        assert!(!is_newer("1.1.0", "1.1.0"));
        assert!(!is_newer("1.0.9", "1.1.0"));
    }

    #[test]
    fn version_compare_tolerates_garbage() {
        assert!(!is_newer("not-a-version", "1.0.0"));
    }
}
