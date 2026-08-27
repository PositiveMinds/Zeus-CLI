//! Free utility functions used by tool implementations.

use std::net::IpAddr;
use std::path::Path;

use super::ToolResult;
use crate::error::Result;
use zeus_fs::{DeviceOutput, GitOutput};

/// Render a `GitOutput` (or the permission/spawn error that prevented one)
/// as a `ToolResult` — a non-zero exit is a soft error visible to the model
/// (so it can read `git`'s own message and react), not a hard `Err` that
/// would abort the tool-call cycle. Matches the same convention already
/// used for `bash` and every other tool here.
pub(crate) fn git_result(result: zeus_fs::Result<GitOutput>) -> Result<ToolResult> {
    match result {
        Ok(out) => {
            let text = format!(
                "exit={:?}\n--- stdout ---\n{}--- stderr ---\n{}",
                out.exit_code, out.stdout, out.stderr
            );
            if out.success {
                Ok(ToolResult::ok(text))
            } else {
                Ok(ToolResult::err(text))
            }
        }
        Err(e) => Ok(ToolResult::err(e.to_string())),
    }
}

/// Same convention as `git_result`/`platform_result` for the adb-backed
/// device engine. `DeviceOutput.success` is false when the command exits
/// nonzero OR the capture itself failed (no device, timeout) — in both cases
/// zeus must present it as an error so the model can react, not shrug.
pub(super) fn device_result(out: DeviceOutput) -> ToolResult {
    let artifact = out
        .artifact
        .as_ref()
        .map(|p| format!("\nartifact: {}", p.display()))
        .unwrap_or_default();
    let text = format!(
        "exit={:?}\n--- stdout ---\n{}--- stderr ---\n{}{}",
        out.exit_code, out.stdout, out.stderr, artifact
    );
    if out.success {
        ToolResult::ok(text)
    } else {
        ToolResult::err(text)
    }
}

/// Detect the most likely test command for a project by looking at its
/// manifests. Best-effort; the tool falls back to an explicit override when
/// nothing matches.
pub(crate) fn detect_test_command(root: &Path) -> Option<String> {
    let dir = |name: &str| root.join(name);
    // Ordered by likelihood/portability. `cargo test` and `go test ./...`
    // are the two that never need an extra runner installed.
    if dir("Cargo.toml").is_file() {
        return Some("cargo test".into());
    }
    if dir("go.mod").is_file() {
        return Some("go test ./...".into());
    }
    if dir("pyproject.toml").is_file() {
        return Some("python -m pytest -q".into());
    }
    if dir("package.json").is_file() {
        if dir("pnpm-lock.yaml").is_file() || dir("pnpm-workspace.yaml").is_file() {
            return Some("pnpm test".into());
        }
        if dir("yarn.lock").is_file() {
            return Some("yarn test".into());
        }
        return Some("npm test".into());
    }
    if dir("Makefile").is_file() {
        return Some("make test".into());
    }
    if dir("Gemfile").is_file() {
        return Some("bundle exec rspec".into());
    }
    None
}

/// Pull the handful of verdict lines (e.g. `test result: ok. 12 passed; 0
/// failed`, `12 passed in 1.2s`, `Done in 1.1s`) out of raw runner output so
/// the model gets a compact summary instead of a wall of dots.
pub(super) fn summarize_test_output(stdout: &str) -> String {
    let mut seen: Vec<&str> = Vec::new();
    for raw in stdout.lines() {
        let line = raw.trim();
        let interesting = line.starts_with("test result:")
            || line.starts_with("ok ")
            || line.starts_with("FAIL")
            || (line.contains("passed") && line.contains("failed"))
            || line.contains(" passed in ")
            || line.contains("Tests:")
            || line.contains("Test Suites:")
            || line.contains("Error:")
            || line.starts_with("no test data")
            || line.starts_with("All tests passed");
        if interesting && !seen.contains(&line) {
            seen.push(line);
        }
        if seen.len() >= 8 {
            break;
        }
    }
    if seen.is_empty() {
        "(no summary lines captured)".to_string()
    } else {
        seen.join("\n")
    }
}

/// Launch the platform's default browser opener for `url`, launch-and-forget.
/// Rejects non-`{http,https}://` (and scheme-less `host:port`) targets so a
/// stray string can't be misinterpreted as a shell flag or command. `file://`
/// is deliberately refused — the browser tool is for *web* URLs; a local file
/// path belongs to the file tools, which carry their own permission gates.
pub(super) fn open_browser_url(url: &str) -> std::io::Result<()> {
    let url = url.trim();
    if !(url.starts_with("http://")
        || url.starts_with("https://")
        || url.starts_with("localhost:")
        || url.starts_with("127.0.0.1:")
        || {
            // Bare hostname or `host:port` (no scheme) — must look like a URL,
            // not a filesystem path. `C:/foo.txt`, `..\\secret`, or `/etc/hosts`
            // would otherwise pass the dot check and get handed to the platform
            // opener as a path.
            url.contains('.')
                && !url.contains(' ')
                && !url.starts_with('-')
                && !url.contains(['/', '\\'])
        })
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("'{url}' isn't a usable web URL — expect something like http://localhost:5173"),
        ));
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .arg("/C")
            .arg("start")
            .arg("")
            .arg(url)
            .spawn()
            .map(|_| ())?;
    }
    #[cfg(not(target_os = "windows"))]
    {
        #[cfg(target_os = "macos")]
        let mut cmd = std::process::Command::new("open");
        #[cfg(all(not(target_os = "macos"), target_os = "linux"))]
        let mut cmd = std::process::Command::new("xdg-open");
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let mut cmd = std::process::Command::new("xdg-open");
        cmd.arg(url).spawn().map(|_| ())?;
    }
    Ok(())
}

/// Map an image file extension to its MIME type; `None` for non-raster
/// formats that a vision model cannot ingest.
pub(super) fn image_mime(path: &Path) -> Option<&'static str> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())?
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" | "jpe" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "bmp" => Some("image/bmp"),
        _ => None,
    }
}

/// Returns `Some(reason)` if `url` points at a loopback/private target that
/// `web_fetch` should refuse to scrape (the fetch tool retrieves content for
/// the model, so pointing it at the user's local services would leak them).
/// Returns the refusal reason when `url` must be blocked. Refusals cover
/// loopback, RFC1918 private ranges, link-local, and cloud-metadata hosts —
/// both by literal name and by resolved IP — so `127.0.0.2`, `10.x`,
/// `192.168.x`, `[::ffff:127.0.0.1]`, trailing-dot hostnames, and a hostname
/// that merely *resolves to* an internal address are all refused, not just
/// the exact strings a name-only check would catch.
pub(super) fn reject_web_target(url: &str) -> Option<String> {
    use std::net::ToSocketAddrs;

    let host = url
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("")
        .trim();
    let host = host.rsplit('@').next().unwrap_or("").trim();
    let host = host.trim_matches(|c| c == '[' || c == ']'); // IPv6 brackets
    if host.is_empty() {
        return Some("no host in url".into());
    }
    // A single trailing dot is valid DNS but wouldn't match the literal
    // blocklist below — normalize it away before comparing.
    let norm = host.trim_end_matches('.');
    if norm.is_empty() {
        return Some("'host' has no name".into());
    }
    for bad in [
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
        "169.254.169.254",
        "metadata.google.internal",
    ] {
        if norm.eq_ignore_ascii_case(bad) {
            return Some(format!(
                "'{host}' resolves to the loopback/metadata services"
            ));
        }
    }

    // Literal-IP fast path first (no DNS), then resolve the hostname and
    // check *every* resolved address — any internal address wins the block.
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_blocked_ip(ip) {
            return Some(format!("'{host}' is an internal address"));
        }
        return None;
    }
    // Unresolvable hosts (DNS failure) fall through — the fetch itself will
    // fail naturally; don't fabricate a refusal for a name we can't look up.
    if let Ok(addrs) = norm.to_socket_addrs() {
        for addr in addrs {
            let ip = addr.ip();
            if is_blocked_ip(ip) {
                return Some(format!("'{host}' resolves to internal address '{ip}'"));
            }
        }
    }
    None
}

/// True for addresses the agent must never be pointed at: loopback, RFC1918
/// private space, link-local (incl. cloud metadata), unspecified/broadcast,
/// and IPv4-mapped IPv6 forms of any of the above.
pub(super) fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unicast_link_local()
                || v6.is_unique_local()
                || v6
                    .to_ipv4_mapped()
                    .is_some_and(|v4| is_blocked_ip(IpAddr::V4(v4)))
        }
    }
}

/// Gregorian leap-year rule (a year divisible by 400 is a leap year; other
/// centuries are not). Kept as a small local helper so the clock tool doesn't
/// depend on chrono's date-trait gymnastics.
pub(super) fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Public alias for the doc-extraction module to reuse.
/// HTML → text with table structure preserved: cells become `a | b | c`
/// rows (one per `<tr>`), block elements break lines, and script/style blocks
/// are dropped. Used for HTML documents and epub chapters, where a flat tag
/// strip would lose tabular data.
pub(crate) fn strip_html_with_tables(html: &str) -> String {
    // Drop <script>/<style> blocks verbatim first — the XML reader would
    // choke on the `<`/`>` inside their bodies.
    let mut clipped = String::with_capacity(html.len());
    let mut rest = html;
    loop {
        let mut skip = None;
        let mut best = rest.len();
        for tag in ["<script", "<style"] {
            if let Some(idx) = rest.find(tag) {
                if idx < best {
                    best = idx;
                    skip = Some(tag);
                }
            }
        }
        let Some(open) = skip else { break };
        clipped.push_str(&rest[..best]);
        let tail = &rest[best..];
        let close = if open == "<script" {
            "</script"
        } else {
            "</style"
        };
        match tail.find(close) {
            Some(end) => rest = &tail[end..],
            None => break,
        }
    }
    clipped.push_str(rest);

    let mut out = String::new();
    let mut reader = quick_xml::Reader::from_str(&clipped);
    reader.config_mut().trim_text(true);
    let mut in_table = 0usize;
    let mut cell_started = false;
    let mut cell_in_row = false;
    loop {
        use quick_xml::events::Event;
        match reader.read_event() {
            Ok(Event::Start(e)) => match e.name().as_ref() {
                b"table" => {
                    in_table += 1;
                    out.push('\n');
                }
                b"tr" => {
                    out.push('\n');
                    cell_in_row = false;
                }
                b"td" | b"th" => {
                    if in_table > 0 {
                        if cell_in_row {
                            out.push_str("| ");
                        }
                        cell_started = true;
                    }
                }
                b"br" | b"p" | b"div" | b"li" | b"blockquote" | b"h1" | b"h2" | b"h3" | b"h4"
                | b"h5" | b"h6"
                    if in_table == 0 =>
                {
                    out.push('\n');
                }
                _ => {}
            },
            Ok(Event::Empty(e)) => {
                if e.name().as_ref() == b"br" && in_table == 0 {
                    out.push('\n');
                }
            }
            Ok(Event::Text(e)) => {
                if let Ok(decoded) = e.unescape() {
                    let t = decoded.trim();
                    if !t.is_empty() {
                        out.push_str(t);
                        if !cell_started {
                            out.push(' ');
                        }
                    }
                }
            }
            Ok(Event::End(e)) => match e.name().as_ref() {
                b"table" => {
                    in_table = in_table.saturating_sub(1);
                    out.push('\n');
                }
                b"tr" => {
                    out.push('\n');
                    cell_in_row = false;
                }
                b"td" | b"th" => {
                    if in_table > 0 {
                        out.push(' ');
                        cell_in_row = true;
                        cell_started = false;
                    }
                }
                b"p" | b"div" | b"li" | b"blockquote" | b"h1" | b"h2" | b"h3" | b"h4" | b"h5"
                | b"h6"
                    if in_table == 0 =>
                {
                    out.push('\n');
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            _ => {}
        }
    }
    out.split('\n')
        .map(|l| {
            l.trim()
                .trim_end_matches(" |")
                .trim_end_matches('|')
                .trim()
                .to_string()
        })
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Crude-but-effective HTML → text: drops scripts/styles/head, then tags,
/// then decodes common entities and collapses whitespace. Good enough for
/// scraping docs/pages into something the model can read.
pub(super) fn strip_html(html: &str) -> String {
    let mut clipped: String = html.to_string();
    for (open_tag, close_tag) in [("<script", "</script"), ("<style", "</style")] {
        let mut buffer = String::with_capacity(clipped.len());
        let mut rest = clipped.as_str();
        while let Some(start) = rest.find(open_tag) {
            buffer.push_str(&rest[..start]);
            rest = &rest[start..];
            match rest.find(close_tag) {
                Some(end) => rest = &rest[end..],
                None => break,
            }
        }
        buffer.push_str(rest);
        clipped = buffer;
    }
    let without_tags = clipped;
    let mut text = String::with_capacity(without_tags.len());
    for seg in without_tags.split('<') {
        match seg.find('>') {
            Some(idx) if !seg[..idx].trim().is_empty() => text.push('\n'),
            _ => {}
        }
        if let Some(idx) = seg.find('>') {
            text.push_str(&seg[idx + 1..]);
        } else {
            text.push_str(seg);
        }
    }
    for (entity, ch) in [
        ("&amp;", '&'),
        ("&lt;", '<'),
        ("&gt;", '>'),
        ("&quot;", '"'),
        ("&#39;", '\''),
        ("&nbsp;", ' '),
    ] {
        text = text.replace(entity, &ch.to_string());
    }
    let text = text.replace('\r', "");
    text.split('\n')
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Percent-encode a string for use inside a URL query value.
pub(super) fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}
