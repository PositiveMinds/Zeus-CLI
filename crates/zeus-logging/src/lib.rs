//! Logging for zeus.
//!
//! - Human-readable console output (respects `RUST_LOG` / configured level)
//! - Optional daily JSON logs under the global logs directory

use std::fs::{File, OpenOptions};
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::writer::MakeWriter;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

/// Errors from logging setup.
#[derive(Debug, thiserror::Error)]
pub enum LoggingError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("logging already initialized")]
    AlreadyInitialized,
}

/// Options for initializing the global subscriber.
#[derive(Debug, Clone)]
pub struct LoggingOptions {
    /// e.g. "info", "debug", "trace", "warn"
    pub level: String,
    /// When true, also write to `logs_dir/zeus-YYYY-MM-DD.log`
    pub file: bool,
    pub logs_dir: Option<PathBuf>,
    /// When false, skip the stderr layer entirely. Set this to false
    /// whenever the caller is about to hand the terminal over to a raw-mode
    /// alternate-screen UI (the ratatui TUI) — a log line written straight
    /// to stderr mid-session corrupts that screen (the terminal ends up
    /// with stray text interleaved into the TUI's own cell-addressed
    /// output), since the UI assumes exclusive ownership of the display.
    /// File logging (if enabled) is unaffected either way.
    pub console: bool,
}

impl Default for LoggingOptions {
    fn default() -> Self {
        Self {
            level: "info".into(),
            file: true,
            logs_dir: None,
            console: true,
        }
    }
}

static INIT: OnceLock<()> = OnceLock::new();

/// Shared file sink that implements `MakeWriter`.
#[derive(Clone)]
struct SharedFile(Arc<Mutex<File>>);

struct FileLineWriter {
    file: Arc<Mutex<File>>,
}

impl Write for FileLineWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut guard = self.file.lock().unwrap_or_else(|e| e.into_inner());
        guard.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        let mut guard = self.file.lock().unwrap_or_else(|e| e.into_inner());
        guard.flush()
    }
}

impl<'a> MakeWriter<'a> for SharedFile {
    type Writer = FileLineWriter;

    fn make_writer(&'a self) -> Self::Writer {
        FileLineWriter {
            file: Arc::clone(&self.0),
        }
    }
}

/// Initialize tracing. Safe to call once; subsequent calls are no-ops.
pub fn init(opts: LoggingOptions) -> Result<(), LoggingError> {
    if INIT.get().is_some() {
        return Ok(());
    }

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(normalize_level(&opts.level)));

    // `Option<Layer>` is itself a no-op `Layer` when `None` (a standard
    // tracing-subscriber pattern) — this drops the stderr layer entirely
    // rather than just filtering it, so nothing can write to the terminal.
    let console_layer = opts.console.then(|| {
        fmt::layer()
            .with_target(true)
            .with_thread_ids(false)
            .with_ansi(console_ansi_supported())
            .with_writer(io::stderr)
    });

    let result = if opts.file {
        if let Some(dir) = &opts.logs_dir {
            std::fs::create_dir_all(dir)?;
            let path = daily_log_path(dir);
            let file = OpenOptions::new().create(true).append(true).open(&path)?;
            let shared = SharedFile(Arc::new(Mutex::new(file)));
            let file_layer = fmt::layer()
                .with_ansi(false)
                .with_target(true)
                .json()
                .with_writer(shared);
            tracing_subscriber::registry()
                .with(filter)
                .with(console_layer)
                .with(file_layer)
                .try_init()
        } else {
            tracing_subscriber::registry()
                .with(filter)
                .with(console_layer)
                .try_init()
        }
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(console_layer)
            .try_init()
    };

    // Another subscriber may already be installed (tests). Treat as OK.
    let _ = result;
    let _ = INIT.set(());
    Ok(())
}

/// Append a single human-readable session history line (command execution log, etc.).
pub fn append_session_log(logs_dir: &Path, line: &str) -> Result<(), LoggingError> {
    std::fs::create_dir_all(logs_dir)?;
    let path = logs_dir.join("session-commands.log");
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    let ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S");
    writeln!(f, "[{ts}] {line}")?;
    Ok(())
}

fn daily_log_path(logs_dir: &Path) -> PathBuf {
    let day = chrono::Local::now().format("%Y-%m-%d");
    logs_dir.join(format!("zeus-{day}.log"))
}

/// Whether the console log layer should emit ANSI color codes. `fmt::layer()`
/// defaults to always-on regardless of whether stderr is actually a
/// terminal, which leaked raw escape codes into every piped/redirected
/// invocation of `zeus` for the whole session until this was caught —
/// explicitly gate it the same way the chat UI layer already does.
fn console_ansi_supported() -> bool {
    io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

fn normalize_level(level: &str) -> String {
    let l = level.trim().to_ascii_lowercase();
    match l.as_str() {
        "trace" | "debug" | "info" | "warn" | "error" | "off" => l,
        _ => "info".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn daily_log_path_contains_date() {
        let dir = Path::new("/tmp/logs");
        let p = daily_log_path(dir);
        let name = p.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("zeus-"));
        assert!(name.ends_with(".log"));
    }

    #[test]
    fn append_session_log_writes() {
        let tmp = TempDir::new().unwrap();
        append_session_log(tmp.path(), "git status exit=0").unwrap();
        let content = std::fs::read_to_string(tmp.path().join("session-commands.log")).unwrap();
        assert!(content.contains("git status exit=0"));
    }

    #[test]
    fn normalize_level_falls_back() {
        assert_eq!(normalize_level("DEBUG"), "debug");
        assert_eq!(normalize_level("nope"), "info");
    }

    #[test]
    fn no_color_env_var_disables_ansi_regardless_of_terminal() {
        // Can't force stderr to look like a non-terminal in a unit test, but
        // NO_COLOR must short-circuit to false unconditionally — that part
        // is deterministic and worth locking in.
        std::env::set_var("NO_COLOR", "1");
        assert!(!console_ansi_supported());
        std::env::remove_var("NO_COLOR");
    }
}
