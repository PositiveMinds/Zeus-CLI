//! Direct Hugging Face file download — no browser needed. HF serves model
//! files over plain HTTPS (`huggingface.co/{repo}/resolve/main/{file}`),
//! the same mechanism `huggingface-cli download` uses; this doesn't need
//! any HF-specific SDK, just a GET request.
//!
//! Downloads land in the configured destination directory (normally
//! `~/.zeus/models/`, see `zeus-config`'s `GlobalPaths::models`), which
//! `scan_local_models` already scans — so a downloaded file is "detected"
//! without any extra step.

use crate::error::{ProviderError, Result};
use futures::StreamExt;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

fn map_reqwest_err(e: reqwest::Error) -> ProviderError {
    ProviderError::Transport(e.to_string())
}

/// Download `{repo}/resolve/main/{file}` from Hugging Face into `dest_dir`,
/// calling `on_progress(bytes_downloaded, total_bytes_if_known)` as data
/// arrives. Writes to a `.part` file and renames to the final name only on
/// success, so an interrupted download never looks like a complete one.
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
    let client = reqwest::Client::new();
    let resp = client.get(&url).send().await.map_err(map_reqwest_err)?;
    if !resp.status().is_success() {
        return Err(ProviderError::Api(format!(
            "download failed ({}) for {url}",
            resp.status()
        )));
    }
    let total = resp.content_length();

    tokio::fs::create_dir_all(dest_dir)
        .await
        .map_err(|e| ProviderError::Api(format!("create destination dir: {e}")))?;
    let dest_path = dest_dir.join(&filename);
    let tmp_path = dest_dir.join(format!("{}.part", filename.to_string_lossy()));

    let mut out = tokio::fs::File::create(&tmp_path)
        .await
        .map_err(|e| ProviderError::Api(format!("create temp file: {e}")))?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
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

    tokio::fs::rename(&tmp_path, &dest_path)
        .await
        .map_err(|e| ProviderError::Api(format!("finalize download: {e}")))?;
    Ok(dest_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Live network test against a real, small, stable public file — proves
    /// the actual HTTP + redirect + streaming-write path works, not just
    /// that the code compiles. Uses a tiny test-fixture repo's config.json
    /// (a few hundred bytes), not a real multi-GB model file.
    #[tokio::test]
    async fn downloads_a_real_small_file_from_hugging_face() {
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

    #[tokio::test]
    async fn missing_file_returns_a_clear_error_not_a_panic() {
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
}
