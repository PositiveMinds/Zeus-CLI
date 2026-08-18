//! Self-update support. `zeus update` downloads the correct prebuilt release
//! asset for the running platform and replaces the current executable in
//! place — it works no matter *where* that executable happens to sit (the
//! install script's directory, a manually-copied test folder, wherever),
//! since it doesn't need to recognize the location, only write to it. The
//! one case this deliberately leaves alone is a `cargo build`/`cargo install`
//! dev checkout: self-replacing someone's own build with a downloaded
//! binary would be surprising, not helpful, so that's still just pointed at
//! rebuilding from source instead.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

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
    /// A `cargo build`/`cargo install` output (dev checkout or `~/.cargo/bin`)
    /// — left alone; there's no way to tell this apart from "someone's
    /// actively working on the source" from the exe path alone.
    Cargo,
    /// Anything else — a prebuilt binary sitting somewhere on disk,
    /// regardless of how it got there. Safe to self-replace in place.
    Direct,
}

pub fn detect_install_method() -> InstallMethod {
    let Ok(exe) = std::env::current_exe() else {
        return InstallMethod::Direct;
    };
    let s = exe.to_string_lossy().to_ascii_lowercase();
    if s.contains(".cargo")
        || s.contains("target/debug")
        || s.contains("target/release")
        || s.contains("target\\debug")
        || s.contains("target\\release")
    {
        return InstallMethod::Cargo;
    }
    InstallMethod::Direct
}

#[derive(Deserialize)]
struct GhRelease {
    tag_name: String,
}

fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(format!("zeus-cli/{}", current_version()))
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .context("build http client")
}

/// Latest published version, tag-prefix (`v`) stripped.
pub async fn latest_version() -> Result<String> {
    let url =
        format!("https://api.github.com/repos/{RELEASES_OWNER}/{RELEASES_REPO}/releases/latest");
    let resp = http_client()?
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

/// The release asset name for the platform this binary is actually running
/// on — matches exactly what `.github/workflows/release.yml`'s build matrix
/// produces (`zeus-<target>.zip` on Windows, `.tar.gz` everywhere else).
fn platform_asset_name() -> Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => Ok("zeus-x86_64-pc-windows-msvc.zip"),
        ("linux", "x86_64") => Ok("zeus-x86_64-unknown-linux-gnu.tar.gz"),
        ("macos", "x86_64") => Ok("zeus-x86_64-apple-darwin.tar.gz"),
        ("macos", "aarch64") => Ok("zeus-aarch64-apple-darwin.tar.gz"),
        (os, arch) => bail!("no prebuilt release published for {os}/{arch}"),
    }
}

/// Extracts a downloaded release archive with whatever the platform already
/// has on hand — the `zip` crate (path-safety guarded, same rules as the
/// provider's own model-archive extractor) for the `.zip` on Windows, `tar`
/// (universal on Linux/macOS) for the `.tar.gz` — rather than pulling in a
/// zip/tar crate just for this one-shot use. (Expand-Archive was the previous
/// Windows path but .NET Framework's implementation is the classic zip-slip
/// vector; the `zip` crate with an explicit containment check is stricter.)
fn extract_archive(archive: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest).context("create extract directory")?;
    if cfg!(windows) {
        extract_zip_safely(archive, dest)
    } else {
        // GNU/BSD tar both strip leading `/` and `..` path components from
        // member names during extraction, so a traversal entry can't escape
        // `-C dest` here the way it can with a raw zip writer.
        let status = std::process::Command::new("tar")
            .args([
                "-xzf",
                &archive.to_string_lossy(),
                "-C",
                &dest.to_string_lossy(),
            ])
            .status()
            .context("spawn tar to extract the downloaded archive")?;
        if !status.success() {
            bail!("extracting the downloaded archive failed with {status}");
        }
        Ok(())
    }
}

/// Extract a `.zip` into `dest`, refusing absolute paths and `..` traversal
/// components — the same zip-slip guard the provider's model-archive
/// extractor enforces. The update archive comes from zeus's own GitHub
/// releases today, but a poisoned/mirrored release must not be able to write
/// outside `dest` even then.
fn extract_zip_safely(archive: &Path, dest: &Path) -> Result<()> {
    let file = std::fs::File::open(archive).context("open downloaded zip")?;
    let mut archive = zip::ZipArchive::new(file).context("read downloaded zip")?;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .with_context(|| format!("read zip entry {i}"))?;
        let rel = entry.name().replace('\\', "/");
        let rel_path = Path::new(&rel);
        if rel_path.is_absolute() || rel.split('/').any(|c| c == "..") {
            bail!("zip entry '{rel}' would escape the extract directory; refusing to extract");
        }
        let out_path = dest.join(&rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&out_path).with_context(|| format!("create dir '{rel}'"))?;
            continue;
        }
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create dir '{}'", parent.display()))?;
        }
        let mut out =
            std::fs::File::create(&out_path).with_context(|| format!("create file '{rel}'"))?;
        std::io::copy(&mut entry, &mut out).with_context(|| format!("extract '{rel}'"))?;
    }
    Ok(())
}

/// The release archive's internal layout differs by platform (the Windows
/// zip is flat, the Unix tar.gz has one nested staging folder — see
/// `release.yml`), so this walks the extracted tree looking for the binary
/// by name instead of assuming a fixed relative path either way.
fn find_extracted_binary(dir: &Path, exe_name: &str) -> Option<PathBuf> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().and_then(|n| n.to_str()) == Some(exe_name) {
                return Some(path);
            }
        }
    }
    None
}

/// Swaps the currently-running executable for `new_binary`, wherever the
/// former happens to live. Windows won't let you delete or overwrite a
/// `.exe` that's actively executing, but it *will* let you rename its
/// directory entry away — so the running binary gets moved to a `.old`
/// sibling first, then the new one takes its place. Unix allows replacing a
/// running binary's path directly, but the same two-step sequence is used
/// there too rather than special-casing it, since it's simpler to reason
/// about one code path than two.
fn self_replace(new_binary: &Path) -> Result<()> {
    let current_exe = std::env::current_exe().context("locate the running executable")?;
    let old_name = match current_exe.extension().and_then(|e| e.to_str()) {
        Some(ext) => format!(
            "{}.old.{ext}",
            current_exe
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
        ),
        None => format!(
            "{}.old",
            current_exe
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        ),
    };
    let old_path = current_exe.with_file_name(old_name);

    // Best-effort: a `.old` left behind by a previous update that couldn't
    // clean itself up yet (still running at the time) shouldn't block this
    // one — Windows will simply fail this silently if that file is itself
    // still in use for some other reason, which is fine, we just overwrite
    // the rename target instead momentarily below.
    let _ = std::fs::remove_file(&old_path);
    std::fs::rename(&current_exe, &old_path)
        .context("rename the running executable out of the way")?;
    // `copy`, not `rename`, into place — the new binary lives in a temp
    // directory that may be on a different drive/filesystem than the
    // install location, and `rename` can't cross that boundary.
    std::fs::copy(new_binary, &current_exe).context("install the downloaded binary")?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&current_exe)
            .context("read new binary's permissions")?
            .permissions();
        perms.set_mode(perms.mode() | 0o111);
        std::fs::set_permissions(&current_exe, perms).context("mark new binary executable")?;
    }

    // Best-effort cleanup — this reliably fails on Windows (the file is
    // this very process, still executing) and that's fine; it just leaves
    // the `.old` file to be replaced by the next update's rename step above.
    let _ = std::fs::remove_file(&old_path);
    Ok(())
}

/// Downloads the release asset for this platform/version, extracts it, and
/// self-replaces the running binary with what's inside. The download uses the
/// provider crate's parallel, resumable downloader (with retries and resume
/// via HTTP `Range`), so a dropped connection re-fetches only the missing
/// chunk rather than forcing the whole archive to restart — keeping updates
/// quick even on flaky links.
async fn self_update(version: &str) -> Result<()> {
    let asset = platform_asset_name()?;
    let tmp_dir = std::env::temp_dir().join(format!("zeus-update-{version}"));
    let _ = std::fs::remove_dir_all(&tmp_dir); // stale leftovers from a prior attempt
    std::fs::create_dir_all(&tmp_dir).context("create a temp directory for the update")?;

    let url = format!(
        "https://github.com/{RELEASES_OWNER}/{RELEASES_REPO}/releases/download/v{version}/{asset}"
    );
    let archive_path = tmp_dir.join(asset);
    let version_owned = version.to_owned();
    zeus_provider::download_asset(&url, &archive_path, move |done, total| {
        match total {
            Some(t) if t > 0 => {
                eprint!(
                    "\r\x1b[Kdownloading v{version_owned}... {done}/{t} bytes ({:.0}%)",
                    done as f64 * 100.0 / t as f64
                );
            }
            _ => {
                eprint!("\r\x1b[Kdownloading v{version_owned}... {done} bytes");
            }
        }
        let _ = io::stderr().flush();
    })
    .await
    .context("download the release asset")?;
    eprintln!();
    let _ = io::stderr().flush();

    extract_archive(&archive_path, &tmp_dir)?;

    let exe_name = if cfg!(windows) { "zeus.exe" } else { "zeus" };
    let new_binary = find_extracted_binary(&tmp_dir, exe_name)
        .with_context(|| format!("couldn't find '{exe_name}' inside the downloaded archive"))?;

    self_replace(&new_binary)?;

    let _ = std::fs::remove_dir_all(&tmp_dir);
    Ok(())
}

/// `zeus update [--check]`: report on / apply the latest release.
///
/// `notify_on_completion` is the user's persisted display setting; when set,
/// a successful install rings the terminal bell (BEL) so the user is alerted
/// that the update finished, not just that it started.
pub async fn cmd_update(check_only: bool, notify_on_completion: bool) -> Result<()> {
    crate::tui::theme::set_notify_on_completion(notify_on_completion);
    let current = current_version();
    println!("current version: {current}");

    let method = detect_install_method();
    let latest = latest_version().await?;

    if !is_newer(&latest, current) {
        println!("already up to date (latest: {latest}).");
        return Ok(());
    }

    println!("update available: {current} -> {latest}");

    if check_only {
        println!("run `zeus update` (without --check) to install it.");
        return Ok(());
    }

    match method {
        InstallMethod::Cargo => {
            println!(
                "this binary looks like a cargo build/install, not a prebuilt release — \
                 update it the same way you built it (e.g. `cargo build --release -p zeus-cli` \
                 after pulling, or `cargo install` from wherever you originally installed it)."
            );
        }
        InstallMethod::Direct => {
            println!("downloading v{latest} for this platform...");
            self_update(&latest).await?;
            println!("updated to {latest}. Restart zeus to use it.");
            if crate::tui::theme::notify_on_completion() {
                use std::io::Write as _;
                let _ = write!(io::stdout(), "\x07");
                let _ = io::stdout().flush();
            }
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

    #[test]
    fn platform_asset_name_resolves_on_every_ci_target() {
        // CI (and any dev machine) only ever runs this on one of the four
        // platforms release.yml's build matrix actually produces — a fifth,
        // genuinely unsupported platform is meant to fail loudly at runtime
        // via `cmd_update`'s `?`, not be silently tolerated here.
        assert!(platform_asset_name().is_ok());
    }

    #[test]
    fn find_extracted_binary_walks_nested_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("zeus-x86_64-unknown-linux-gnu");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("zeus"), b"fake binary").unwrap();

        let found = find_extracted_binary(tmp.path(), "zeus").unwrap();
        assert_eq!(found, nested.join("zeus"));
    }

    #[test]
    fn find_extracted_binary_finds_flat_layout_too() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("zeus.exe"), b"fake binary").unwrap();

        let found = find_extracted_binary(tmp.path(), "zeus.exe").unwrap();
        assert_eq!(found, tmp.path().join("zeus.exe"));
    }

    #[test]
    fn find_extracted_binary_returns_none_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(find_extracted_binary(tmp.path(), "zeus").is_none());
    }
}
