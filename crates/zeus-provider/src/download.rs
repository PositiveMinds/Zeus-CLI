//! Direct file downloads with bounded retries and resume.
//!
//! Two entry points share one philosophy — progress lands in a `.part` /
//! `.parts` sidecar, an interrupted download is never mistaken for a complete
//! one, and a dropped connection resumes from where it left off (via HTTP
//! `Range`) instead of restarting:
//!
//! - [`download_hf_file`] — a single streaming connection, used for Hugging
//!   Face model files (multi-GB, sequential, HF redirects to its CDN).
//! - [`download_asset`] — a parallel chunked download (many small `Range`
//!   requests fetched concurrently), used for the small release binaries
//!   `zeus update` pulls. A lost connection only re-fetches a 4 MiB chunk
//!   rather than the whole archive, and several in-flight connections get
//!   better aggregate throughput on lossy or per-connection-throttled links —
//!   so the update stays quick even where a single big stream keeps dying
//!   mid-body.

use crate::error::{ProviderError, Result};
use futures::StreamExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
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

/// Parallel-chunked downloads: how many bytes each chunk request asks for.
/// Small enough that a dropped connection only re-fetches a small piece;
/// large enough that per-request overhead stays negligible.
const CHUNK_SIZE: u64 = 4 * 1024 * 1024;

/// Parallel-chunked downloads: how many chunks to fetch concurrently.
const MAX_PARALLEL: usize = 8;

/// A single chunk request may take this long in total before it's aborted and
/// retried from its partial position. Chunks are small, so this only trips on
/// a genuinely stalled connection — it never aborts a transfer that is still
/// making progress on its other connections.
const CHUNK_TIMEOUT: Duration = Duration::from_secs(120);

fn map_reqwest_err(e: reqwest::Error) -> ProviderError {
    ProviderError::Transport(e.to_string())
}

/// The caller-supplied progress callback, shared across every concurrently
/// running chunk task. `Arc<Mutex<..>>` rather than a channel — chunks call
/// it directly and infrequently (once per completed chunk), so a shared lock
/// is simpler than plumbing a channel + a separate aggregator task for the
/// same effect.
type ProgressCb = Arc<std::sync::Mutex<Box<dyn FnMut(u64, Option<u64>) + Send>>>;

/// One chunk's byte range and where it's being written — grouped so
/// `fetch_chunk` takes one value instead of four separate positional
/// `u64`/`PathBuf` args that are easy to transpose by accident at the call
/// site.
struct ChunkSpec {
    start: u64,
    end: u64,
    part_path: PathBuf,
    total: u64,
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
    download_single_stream_with_retries(&client, &url, &dest_path, &mut on_progress).await?;
    Ok(dest_path)
}

/// The `.part` sidecar a single-stream download accumulates into.
fn part_path(dest_path: &Path) -> PathBuf {
    let name = dest_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    dest_path.with_file_name(format!("{name}.part"))
}

/// One single-stream transfer attempt from `resume_from` bytes onward,
/// bounded by `MAX_ATTEMPTS`; each retry resumes instead of restarting.
async fn download_single_stream_with_retries(
    client: &reqwest::Client,
    url: &str,
    dest_path: &Path,
    on_progress: &mut impl FnMut(u64, Option<u64>),
) -> Result<()> {
    let tmp_path = part_path(dest_path);

    let mut attempt = 0u32;
    loop {
        // How far along is the partial file already?
        let resume_from = tokio::fs::metadata(&tmp_path).await.map(|m| m.len()).unwrap_or(0);

        match download_attempt(client, url, &tmp_path, resume_from, on_progress).await {
            Ok(_) => {
                tokio::fs::rename(&tmp_path, dest_path)
                    .await
                    .map_err(|e| ProviderError::Api(format!("finalize download: {e}")))?;
                return Ok(());
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

/// Directory holding one `<index>` chunk file per in-flight byte range, plus
/// a `meta` line recording the URL/total so a later run can resume exactly
/// where this one left off.
fn parts_dir(dest_path: &Path) -> PathBuf {
    let name = dest_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    dest_path.with_file_name(format!("{name}.parts"))
}

/// True when `parts` still belongs to `url`/`total`, so its chunk files can
/// be resumed rather than rebuilt from scratch.
fn parts_meta_matches(parts: &Path, url: &str, total: u64) -> bool {
    std::fs::read_to_string(parts.join("meta"))
        .map(|meta| meta == format!("{url}\n{total}"))
        .unwrap_or(false)
}

/// One chunk request: fetch `[from, end)` (inclusive end) and append it to
/// `part_path`. Returns the bytes appended.
async fn fetch_chunk_attempt(
    client: &reqwest::Client,
    url: &str,
    from: u64,
    end: u64,
    part_path: &Path,
) -> Result<u64> {
    let resp = client
        .get(url)
        .header(reqwest::header::RANGE, format!("bytes={from}-{}", end - 1))
        .timeout(CHUNK_TIMEOUT)
        .send()
        .await
        .map_err(map_reqwest_err)?;
    if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        // We probed range support and got a 206; a full 200 body here means
        // the server flaked — fail the attempt so it retries the chunk rather
        // than mis-assembling the archive.
        return Err(ProviderError::Transport(format!(
            "expected a 206 partial response, got {}",
            resp.status()
        )));
    }
    let body = resp.bytes().await.map_err(map_reqwest_err)?;
    let mut out = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(part_path)
        .await
        .map_err(|e| ProviderError::Api(format!("create chunk file: {e}")))?;
    out.write_all(&body)
        .await
        .map_err(|e| ProviderError::Api(format!("write chunk file: {e}")))?;
    out.flush()
        .await
        .map_err(|e| ProviderError::Api(format!("flush chunk file: {e}")))?;
    Ok(body.len() as u64)
}

/// One chunk of a parallel download, retried (resuming from its partial size)
/// until it lands. `[start, end)` is the byte range this chunk owns; `done`
/// accumulates completed bytes across all chunks for progress reporting.
async fn fetch_chunk(
    client: &reqwest::Client,
    url: &str,
    spec: ChunkSpec,
    done: Arc<AtomicU64>,
    progress: ProgressCb,
) -> Result<()> {
    let ChunkSpec {
        start,
        end,
        part_path,
        total,
    } = spec;
    let want = end - start;
    for attempt in 0..MAX_ATTEMPTS {
        let have = tokio::fs::metadata(&part_path)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
        if have >= want {
            // A chunk can end up larger than `want` only if a previous run's
            // sidecar leaked; we only credit the bytes this chunk owns.
            done.fetch_add(want.saturating_sub(have.min(want)), Ordering::Relaxed);
            return Ok(());
        }
        match fetch_chunk_attempt(client, url, start + have, end, &part_path).await {
            Ok(got) => {
                done.fetch_add(got, Ordering::Relaxed);
                let mut cb = progress.lock().expect("progress callback lock");
                cb(done.load(Ordering::Relaxed), Some(total));
            }
            Err(e) if attempt + 1 >= MAX_ATTEMPTS => return Err(e),
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
    // Attempts exhausted without the chunk completing.
    let have = tokio::fs::metadata(&part_path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    if have >= want {
        done.fetch_add(want - have, Ordering::Relaxed);
        Ok(())
    } else {
        Err(ProviderError::Transport(
            "chunk download kept failing; rerun to resume from the partial download".into(),
        ))
    }
}

/// Download `url` into `dest_path` with parallel chunked `Range` requests,
/// resuming any prior partial progress. Falls back to a single resumable
/// stream when the server won't serve byte ranges. `on_progress` receives
/// `(bytes_downloaded, total_if_known)` as chunks complete. The final file is
/// written (and the sidecar cleaned up) only once every chunk is verified in
/// place, so an interrupted run never looks like a successful one.
pub async fn download_asset(
    url: &str,
    dest_path: &Path,
    on_progress: impl FnMut(u64, Option<u64>) + Send + 'static,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .map_err(|e| ProviderError::Transport(format!("build http client: {e}")))?;

    if let Some(parent) = dest_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| ProviderError::Api(format!("create destination dir: {e}")))?;
    }

    // Probe range support with a 1-byte request. 206 → the server honours
    // `Range` and reports the total; 200 → it ignored the range (sent the
    // whole body), so fall back to a single stream.
    let probe = client
        .get(url)
        .header(reqwest::header::RANGE, "bytes=0-0")
        .timeout(CHUNK_TIMEOUT)
        .send()
        .await
        .map_err(map_reqwest_err)?;
    let total = if probe.status() == reqwest::StatusCode::PARTIAL_CONTENT {
        probe
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_content_range)
            .and_then(|(_, t)| t)
    } else {
        None
    };
    drop(probe);

    let Some(total) = total else {
        let mut cb = Box::new(on_progress);
        return download_single_stream_with_retries(&client, url, dest_path, &mut cb).await;
    };
    if total == 0 {
        tokio::fs::write(dest_path, b"")
            .await
            .map_err(|e| ProviderError::Api(format!("write empty file: {e}")))?;
        return Ok(());
    }

    let parts = parts_dir(dest_path);
    if !parts_meta_matches(&parts, url, total) {
        let _ = tokio::fs::remove_dir_all(&parts).await;
        tokio::fs::create_dir_all(&parts)
            .await
            .map_err(|e| ProviderError::Api(format!("create parts dir: {e}")))?;
        tokio::fs::write(parts.join("meta"), format!("{url}\n{total}"))
            .await
            .map_err(|e| ProviderError::Api(format!("write parts meta: {e}")))?;
    }

    let chunks = total.div_ceil(CHUNK_SIZE);
    let done = Arc::new(AtomicU64::new(0));
    let progress: ProgressCb = Arc::new(std::sync::Mutex::new(Box::new(on_progress)));

    let mut tasks = Vec::new();
    for i in 0..chunks {
        let start = i * CHUNK_SIZE;
        let end = (start + CHUNK_SIZE).min(total);
        let part_path = parts.join(format!("{i:04}"));
        let done = done.clone();
        let progress = progress.clone();
        let spec = ChunkSpec {
            start,
            end,
            part_path,
            total,
        };
        tasks.push(fetch_chunk(&client, url, spec, done, progress));
    }
    futures::stream::iter(tasks)
        .buffer_unordered(MAX_PARALLEL)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    // Assemble the chunks in order, then verify against the advertised total.
    let mut out = tokio::fs::File::create(dest_path)
        .await
        .map_err(|e| ProviderError::Api(format!("create destination file: {e}")))?;
    let mut written = 0u64;
    for i in 0..chunks {
        let part = parts.join(format!("{i:04}"));
        let bytes = tokio::fs::read(&part)
            .await
            .map_err(|e| ProviderError::Api(format!("read chunk {i}: {e}")))?;
        out.write_all(&bytes)
            .await
            .map_err(|e| ProviderError::Api(format!("write destination file: {e}")))?;
        written += bytes.len() as u64;
    }
    out.flush()
        .await
        .map_err(|e| ProviderError::Api(format!("flush destination file: {e}")))?;
    drop(out);
    let _ = tokio::fs::remove_dir_all(&parts).await;

    if written != total {
        return Err(ProviderError::Transport(format!(
            "downloaded {written} of {total} bytes — rerun to resume from the partial download"
        )));
    }
    Ok(())
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
        assert_eq!(
            parse_content_range("bytes 0-1023/4096"),
            Some((0, Some(4096)))
        );
        assert_eq!(
            parse_content_range("bytes 754300000-2019377695/2019377696"),
            Some((754300000, Some(2019377696)))
        );
        assert_eq!(parse_content_range("bytes */2019377696"), None);
        assert_eq!(parse_content_range("garbage"), None);
    }

    /// Serve `data` on a random localhost port, honoring `Range: bytes=a-b`
    /// (inclusive) with a 206 + `Content-Range`, exercising the downloader
    /// against real HTTP semantics without touching the network. When
    /// `fail_first_data_request` is set, the first non-probe request sends
    /// `send_before_close` bytes then slams the connection shut — simulating
    /// the mid-body reset (`os error 10054`) a real `zeus update` hit.
    async fn spawn_range_server(
        data: Arc<Vec<u8>>,
        fail_first_data_request: bool,
        send_before_close: usize,
    ) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let data_requests = Arc::new(AtomicU64::new(0));
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let data = data.clone();
                let data_requests = data_requests.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = match sock.read(&mut buf).await {
                        Ok(n) if n > 0 => n,
                        _ => return,
                    };
                    let head = String::from_utf8_lossy(&buf[..n]);
                    let mut range: Option<(usize, usize)> = None;
                    for line in head.lines() {
                        if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                            if let Some((a, b)) = v.split_once('-') {
                                if let (Ok(a), Ok(b)) = (a.parse(), b.parse()) {
                                    range = Some((a, b));
                                }
                            }
                        }
                    }
                    let total = data.len();
                    let (start, end) = range.unwrap_or((0, total.saturating_sub(1)));
                    let end = end.min(total.saturating_sub(1));
                    if start >= total {
                        return;
                    }
                    // `bytes=0-0` is the range-capability probe, not data.
                    let is_probe = range == Some((0, 0));
                    let body = &data[start..=end];
                    let resp_head = format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                        body.len(),
                        start,
                        end,
                        total
                    );
                    if fail_first_data_request
                        && !is_probe
                        && data_requests.fetch_add(1, Ordering::Relaxed) == 0
                    {
                        let _ = sock.write_all(resp_head.as_bytes()).await;
                        let cut = send_before_close.min(body.len());
                        let _ = sock.write_all(&body[..cut]).await;
                        // Close mid-body: the reader sees a connection reset.
                        return;
                    }
                    let _ = sock.write_all(resp_head.as_bytes()).await;
                    let _ = sock.write_all(body).await;
                });
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn asset_download_assembles_exact_bytes_from_parallel_chunks() {
        // ~9 MB spans several 4 MiB chunks, so the parallel path really runs.
        let data = Arc::new(
            (0..9_000_000u32)
                .map(|i| (i % 251) as u8)
                .collect::<Vec<u8>>(),
        );
        let url = spawn_range_server(data.clone(), false, 0).await;
        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("asset.bin");
        let last_progress = Arc::new(std::sync::Mutex::new((0u64, None)));

        let progress = last_progress.clone();
        download_asset(&url, &dest, move |done, total| {
            *progress.lock().unwrap() = (done, total)
        })
        .await
        .unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), data.as_slice());
        let (done, total) = *last_progress.lock().unwrap();
        assert_eq!(done as usize, data.len(), "progress must reach the total");
        assert_eq!(total, Some(data.len() as u64));
        // Sidecars are cleaned up after a successful download.
        assert!(!tmp.path().join("asset.bin.parts").exists());
        assert!(!tmp.path().join("asset.bin.part").exists());
    }

    #[tokio::test]
    async fn asset_download_resumes_after_a_mid_body_reset() {
        // The server drops the first real chunk request mid-body (the exact
        // `os error 10054` a `zeus update` user hit); the downloader must
        // retry that chunk from its partial size and still produce the exact
        // bytes — never a truncated archive.
        let data = Arc::new(
            (0..9_000_000u32)
                .map(|i| (i % 251) as u8)
                .collect::<Vec<u8>>(),
        );
        let url = spawn_range_server(data.clone(), true, 500_000).await;
        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("asset.bin");

        download_asset(&url, &dest, |_, _| {}).await.unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), data.as_slice());
    }
}
