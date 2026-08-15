//! Direct Hugging Face file download — no browser needed. HF serves model
//! files over plain HTTPS (`huggingface.co/{repo}/resolve/main/{file}`),
//! the same mechanism `huggingface-cli download` uses; this doesn't need
//! any HF-specific SDK, just a GET request.
//!
//! Downloads land in the configured destination directory (normally
//! `~/.zeus/models/`, see `zeus-config`'s `GlobalPaths::models`), which
//! `scan_local_models` already scans — so a downloaded file is "detected"
//! without any extra step.
//!
//! Transfers are resumable: progress accumulates in a `<file>.part` sidecar
//! that a later run continues from (via `Range`) instead of restarting, so a
//! dropped connection on a multi-GB model never loses completed work. On
//! success the `.part` file is renamed to the final name atomically.

use crate::error::{ProviderError, Result};
use futures::StreamExt;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;

/// Cap on establishing the connection (DNS / TCP / TLS). A directory with no
/// reachable server must surface as an error instead of blocking the turn on
/// an OS-level connect retry loop. Deliberately scoped to connect, not the
/// whole transfer: model files can be multi-GB and a total request timeout
/// would cut off legitimately slow downloads mid-stream.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// How many times to re-attempt a transfer that died mid-stream. Each retry
/// resumes from the current `.part` size, so repeated drops still converge.
const MAX_ATTEMPTS: u32 = 5;

fn map_reqwest_err(e: reqwest::Error) -> ProviderError {
    ProviderError::Transport(e.to_string())
}

/// Parse a `Content-Range` header like `bytes 0-1023/4096` into
/// `(start, total)`, returning `None` for the unknown-total `bytes */N` form.
fn parse_content_range(value: &str) -> Option<(u64, Option<u64>)> {
    let rest = value.strip_prefix("bytes ")?;
    let (range, total) = rest.split_once('/')?;
    let total = total.parse::<u64>().ok();
    let (start, _end) = range.split_once('-')?;
    start.parse::<u64>().ok().map(|s| (s, total))
}

/// One transfer attempt from `resume_from` bytes onward. Returns the total
/// bytes expected (best-effort), or an error if the connection died part-way.
async fn download_attempt(
    client: &reqwest::Client,
    url: &str,
    tmp_path: &Path,
    resume_from: u64,
    on_progress: &mut impl FnMut(u64, Option<u64>),
) -> Result<Option<u64>> {
    let mut req = client.get(url);
    if resume_from > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={resume_from}-"));
    }
    let resp = req.send().await.map_err(map_reqwest_err)?;

    // `206 Partial Content` honours our resume range; `200` means the server
    // ignored it (sent the full body) — treat that as starting over.
    let (offset, total) = if resp.status() == reqwest::StatusCode::PARTIAL_CONTENT {
        let from_header = resp
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_content_range);
        let total = from_header.and_then(|(_, t)| t).or_else(|| resp.content_length());
        let start = from_header.map(|(s, _)| s).unwrap_or(resume_from);
        (start, total)
    } else {
        (0, resp.content_length())
    };

    // Recreate or append depending on whether we resume. `append(true)` is
    // used for the resume case; a 200-with-ignored-range restarts the file.
    // `write(true)` is always set — Windows rejects `truncate(true)` without
    // write/append access, and both paths need to write bytes anyway.
    let mut out = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(offset > 0)
        .truncate(offset == 0)
        .open(tmp_path)
        .await
        .map_err(|e| ProviderError::Api(format!("create temp file: {e}")))?;

    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = offset;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(map_reqwest_err)?;
        out.write_all(&chunk)
            .await
            .map_err(|e| ProviderError::Api(format!("write temp file: {e}")))?;
        downloaded += chunk.len() as u64;
        on_progress(downloaded, total);
    }
    out.flush()
        .await
        .map_err(|e| ProviderError::Api(format!("flush temp file: {e}")))?;
    drop(out);

    // A truncated body (server advertised more than it sent) must not be
    // treated as success — leave the `.part` for the next attempt to resume.
    if let Some(expect) = total {
        if downloaded < expect {
            return Err(ProviderError::Transport(format!(
                "truncated download: {downloaded} of {expect} bytes"
            )));
        }
    }
    Ok(total)
}

/// Download `{repo}/resolve/main/{file}` from Hugging Face into `dest_dir`,
/// calling `on_progress(bytes_downloaded, total_bytes_if_known)` as data
/// arrives. Writes to a `.part` file (resuming it if present) and renames to
/// the final name only on success, so an interrupted download never looks
/// like a complete one and completed work is never re-downloaded.
pub async fn download_hf_file(
    repo: &str,
    file: &str,
    dest_dir: &Path,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf> {
    let filename = Path::new(file)
        .file_name()
        .ok_or_else(|| ProviderError::InvalidRequest("empty filename".into()))?
        .to_os_string();

    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let client = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .map_err(|e| ProviderError::Transport(format!("build http client: {e}")))?;

    tokio::fs::create_dir_all(dest_dir)
        .await
        .map_err(|e| ProviderError::Api(format!("create destination dir: {e}")))?;
    let dest_path = dest_dir.join(&filename);
    let tmp_path = dest_dir.join(format!("{}.part", filename.to_string_lossy()));

    let mut attempt = 0u32;
    loop {
        // How far along is the partial file already?
        let resume_from = tokio::fs::metadata(&tmp_path).await.map(|m| m.len()).unwrap_or(0);

        match download_attempt(&client, &url, &tmp_path, resume_from, &mut on_progress).await {
            Ok(_) => {
                tokio::fs::rename(&tmp_path, &dest_path)
                    .await
                    .map_err(|e| ProviderError::Api(format!("finalize download: {e}")))?;
                return Ok(dest_path);
            }
            Err(e) => {
                attempt += 1;
                if attempt >= MAX_ATTEMPTS {
                    return Err(e);
                }
                eprintln!(
                    "download interrupted ({e}); resuming from {} bytes (attempt {attempt}/{MAX_ATTEMPTS})",
                    resume_from
                );
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Live network test against a real, small, stable public file — proves
    /// the actual HTTP + redirect + streaming-write path works, not just
    /// that the code compiles. Uses a tiny test-fixture repo's config.json
    /// (a few hundred bytes), not a real multi-GB model file.
    ///
    /// Skips unless `ZEUS_LIVE_TESTS=1` is set: a default `cargo test` must
    /// never require (or hang on) internet access, in CI or offline dev.
    #[tokio::test]
    async fn downloads_a_real_small_file_from_hugging_face() {
        if std::env::var("ZEUS_LIVE_TESTS").is_err() {
            eprintln!("skipped: set ZEUS_LIVE_TESTS=1 to run live network tests");
            return;
        }
        let tmp = TempDir::new().unwrap();
        let mut last_progress = (0u64, None);
        let path = download_hf_file(
            "hf-internal-testing/tiny-random-gpt2",
            "config.json",
            tmp.path(),
            |downloaded, total| last_progress = (downloaded, total),
        )
        .await
        .unwrap();

        assert!(path.exists());
        assert_eq!(path.file_name().unwrap(), "config.json");
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains('{'), "should be JSON: {content}");
        assert!(last_progress.0 > 0, "progress callback should have fired");
        // No leftover .part file after a successful download.
        assert!(!tmp.path().join("config.json.part").exists());
    }

    /// Same live-network gating as `downloads_a_real_small_file_from_hugging_face`.
    #[tokio::test]
    async fn missing_file_returns_a_clear_error_not_a_panic() {
        if std::env::var("ZEUS_LIVE_TESTS").is_err() {
            eprintln!("skipped: set ZEUS_LIVE_TESTS=1 to run live network tests");
            return;
        }
        let tmp = TempDir::new().unwrap();
        let err = download_hf_file(
            "hf-internal-testing/tiny-random-gpt2",
            "this-file-does-not-exist-12345.bin",
            tmp.path(),
            |_, _| {},
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ProviderError::Api(_)));
    }

    #[test]
    fn parses_content_range() {
        assert_eq!(parse_content_range("bytes 0-1023/4096"), Some((0, Some(4096))));
        assert_eq!(
            parse_content_range("bytes 754300000-2019377695/2019377696"),
            Some((754300000, Some(2019377696)))
        );
        assert_eq!(parse_content_range("bytes */2019377696"), None);
        assert_eq!(parse_content_range("garbage"), None);
    }
}
