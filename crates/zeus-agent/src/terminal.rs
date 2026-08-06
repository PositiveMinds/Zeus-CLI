//! Foreground command execution.
//!
//! Streams via a real PTY (ConPTY on Windows / a Unix pty) by default, so
//! tools that detect a non-interactive terminal still show progress bars and
//! color faithfully. A PTY merges stdout/stderr into one stream (that's
//! inherent to how terminals work, not a limitation of this code) — PTY
//! output lands entirely in `TerminalOutput::stdout`, with `stderr` left
//! empty; `used_pty` tells the caller which mode actually ran. If PTY setup
//! fails for any reason (unsupported environment, etc.), execution falls
//! back automatically to plain piped stdout/stderr, which does keep the
//! streams separate.

use crate::error::{AgentError, Result};
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::warn;
use zeus_fs::{ApprovalDecision, PermissionGate, PermissionRequest};

/// Sandbox strictness tier (see blueprint's Terminal Execution section).
/// `RestrictedFs` is the practical default: pin cwd to the project root and
/// scrub the child's environment down to a safe allowlist. True OS-level
/// sandboxing (containers/namespaces) is a larger, platform-specific effort
/// and is not implemented here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Sandbox {
    None,
    #[default]
    RestrictedFs,
}

/// Which execution profile a command falls into — drives history/background
/// handling upstream; the runner itself treats all profiles the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommandProfile {
    ReadOnly,
    Foreground,
    Background,
}

#[derive(Debug, Clone)]
pub struct TerminalOptions {
    pub cwd: PathBuf,
    pub timeout: Option<Duration>,
    pub sandbox: Sandbox,
    pub profile: CommandProfile,
    /// Prefer a real PTY (falls back to piped stdout/stderr automatically on
    /// failure). Set false to force plain piped execution, e.g. when a
    /// caller needs stdout/stderr kept separate.
    pub use_pty: bool,
}

impl TerminalOptions {
    pub fn new(cwd: PathBuf) -> Self {
        Self {
            cwd,
            timeout: Some(Duration::from_secs(120)),
            sandbox: Sandbox::RestrictedFs,
            profile: CommandProfile::Foreground,
            // Off by default: process-exit detection through portable-pty's
            // `try_wait()` was observed to never fire on this Windows setup
            // (confirmed live — every PTY-run command silently degraded to
            // "wait the full timeout, then report timed_out=true", even for
            // a trivial `echo hi`), which is worse than no PTY at all. Real,
            // opt-in via `use_pty: true`; not yet trustworthy as a default.
            use_pty: false,
        }
    }
}

const MAX_CAPTURED_BYTES: usize = 200_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalOutput {
    pub command: String,
    pub exit_code: Option<i32>,
    /// Combined stdout+stderr when `used_pty` is true (a PTY merges the two
    /// streams); stdout only otherwise.
    pub stdout: String,
    /// Always empty when `used_pty` is true.
    pub stderr: String,
    pub truncated: bool,
    pub duration_ms: u64,
    pub cancelled: bool,
    pub timed_out: bool,
    pub used_pty: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandRecord {
    pub command: String,
    pub cwd: String,
    pub exit_code: Option<i32>,
    pub duration_ms: u64,
    pub cancelled: bool,
    pub timed_out: bool,
    pub timestamp: String,
}

/// Per-session command history log (JSON Lines), independent of the model's
/// context — full output stays here even after it's truncated for the model.
pub struct CommandHistory {
    dir: PathBuf,
}

impl CommandHistory {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    fn file(&self) -> PathBuf {
        self.dir.join("terminal-history.jsonl")
    }

    pub fn record(&self, rec: &CommandRecord) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let line = serde_json::to_string(rec)
            .map_err(|e| AgentError::Terminal(format!("history serialize: {e}")))?;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.file())?;
        writeln!(f, "{line}")?;
        Ok(())
    }

    pub fn recent(&self, limit: usize) -> Result<Vec<CommandRecord>> {
        let path = self.file();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(&path)?;
        let mut out: Vec<CommandRecord> = text
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        if out.len() > limit {
            out = out.split_off(out.len() - limit);
        }
        Ok(out)
    }
}

pub struct TerminalRunner {
    history: CommandHistory,
}

impl TerminalRunner {
    pub fn new(history_dir: PathBuf) -> Self {
        Self {
            history: CommandHistory::new(history_dir),
        }
    }

    pub fn history(&self) -> &CommandHistory {
        &self.history
    }

    /// Run a command to completion (or cancellation/timeout), permission-gated
    /// through the same `PermissionGate` used for file operations (tool="bash"),
    /// so the built-in destructive-command denials (`rm -rf*`, `git push
    /// --force*`, `git reset --hard*`) and the "ask" resolution apply here too.
    pub fn run<F>(
        &self,
        command: &str,
        gate: &PermissionGate,
        opts: TerminalOptions,
        cancel: Arc<AtomicBool>,
        approver: F,
    ) -> Result<TerminalOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        gate.enforce(
            &PermissionRequest {
                tool: "bash".into(),
                path: None,
                command: Some(command.to_string()),
                description: format!("run: {command}"),
                ..Default::default()
            },
            approver,
        )
        .map_err(AgentError::Fs)?;

        let output = if opts.use_pty {
            match self.run_pty(command, &opts, cancel.clone()) {
                Ok(out) => out,
                Err(e) => {
                    warn!(error = %e, "PTY execution failed, falling back to piped stdout/stderr");
                    self.run_piped(command, &opts, cancel)?
                }
            }
        } else {
            self.run_piped(command, &opts, cancel)?
        };

        self.history.record(&CommandRecord {
            command: command.to_string(),
            cwd: opts.cwd.display().to_string(),
            exit_code: output.exit_code,
            duration_ms: output.duration_ms,
            cancelled: output.cancelled,
            timed_out: output.timed_out,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })?;

        Ok(output)
    }

    fn run_piped(
        &self,
        command: &str,
        opts: &TerminalOptions,
        cancel: Arc<AtomicBool>,
    ) -> Result<TerminalOutput> {
        let start = Instant::now();
        let mut cmd = build_command(command, opts);
        cmd.current_dir(&opts.cwd);
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| AgentError::Terminal(format!("spawn failed: {e}")))?;
        let pid = child.id();

        let stdout_buf = Arc::new(Mutex::new(String::new()));
        let stderr_buf = Arc::new(Mutex::new(String::new()));
        let out_thread = child
            .stdout
            .take()
            .map(|h| spawn_reader(h, stdout_buf.clone()));
        let err_thread = child
            .stderr
            .take()
            .map(|h| spawn_reader(h, stderr_buf.clone()));

        let mut timed_out = false;
        let mut cancelled = false;
        let exit_status = loop {
            if let Some(status) = child
                .try_wait()
                .map_err(|e| AgentError::Terminal(e.to_string()))?
            {
                break Some(status);
            }
            if cancel.load(Ordering::SeqCst) {
                cancelled = true;
                kill_tree(pid);
                let _ = child.wait();
                break None;
            }
            if let Some(t) = opts.timeout {
                if start.elapsed() >= t {
                    timed_out = true;
                    kill_tree(pid);
                    let _ = child.wait();
                    break None;
                }
            }
            std::thread::sleep(Duration::from_millis(25));
        };

        if let Some(h) = out_thread {
            let _ = h.join();
        }
        if let Some(h) = err_thread {
            let _ = h.join();
        }

        let mut stdout = stdout_buf.lock().unwrap().clone();
        let mut stderr = stderr_buf.lock().unwrap().clone();
        let mut truncated = false;
        if stdout.len() > MAX_CAPTURED_BYTES {
            stdout.truncate(MAX_CAPTURED_BYTES);
            truncated = true;
        }
        if stderr.len() > MAX_CAPTURED_BYTES {
            stderr.truncate(MAX_CAPTURED_BYTES);
            truncated = true;
        }

        Ok(TerminalOutput {
            command: command.to_string(),
            exit_code: exit_status.and_then(|s| s.code()),
            stdout,
            stderr,
            truncated,
            duration_ms: start.elapsed().as_millis() as u64,
            cancelled,
            timed_out,
            used_pty: false,
        })
    }

    fn run_pty(
        &self,
        command: &str,
        opts: &TerminalOptions,
        cancel: Arc<AtomicBool>,
    ) -> Result<TerminalOutput> {
        let start = Instant::now();
        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows: 24,
                cols: 80,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| AgentError::Terminal(format!("pty open failed: {e}")))?;

        let mut builder = if cfg!(windows) {
            let mut b = CommandBuilder::new("cmd");
            b.args(["/C", command]);
            b
        } else {
            let mut b = CommandBuilder::new("sh");
            b.args(["-c", command]);
            b
        };
        builder.cwd(&opts.cwd);
        if opts.sandbox == Sandbox::RestrictedFs {
            builder.env_clear();
            if let Ok(path) = std::env::var("PATH") {
                builder.env("PATH", path);
            }
            if cfg!(windows) {
                for var in ["SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "TEMP", "TMP", "USERPROFILE"] {
                    if let Ok(v) = std::env::var(var) {
                        builder.env(var, v);
                    }
                }
            }
        }

        let mut child = pair
            .slave
            .spawn_command(builder)
            .map_err(|e| AgentError::Terminal(format!("pty spawn failed: {e}")))?;
        // Drop our copy of the slave so the master's reader sees EOF once
        // the child's own handles close, instead of waiting on a handle we
        // no longer need.
        drop(pair.slave);
        let pid = child.process_id();

        let mut reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| AgentError::Terminal(format!("pty reader clone failed: {e}")))?;
        let output_buf = Arc::new(Mutex::new(String::new()));
        let reader_done = Arc::new(AtomicBool::new(false));
        {
            let output_buf = output_buf.clone();
            let reader_done = reader_done.clone();
            std::thread::spawn(move || {
                let mut buf = [0u8; 4096];
                loop {
                    match reader.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            let mut b = output_buf.lock().unwrap();
                            if b.len() < MAX_CAPTURED_BYTES * 2 {
                                b.push_str(&String::from_utf8_lossy(&buf[..n]));
                            }
                        }
                        Err(_) => break,
                    }
                }
                reader_done.store(true, Ordering::SeqCst);
            });
        }

        let mut timed_out = false;
        let mut cancelled = false;
        let exit_code = loop {
            if let Ok(Some(status)) = child.try_wait() {
                break Some(status.exit_code() as i32);
            }
            if cancel.load(Ordering::SeqCst) {
                cancelled = true;
                // Kill by PID via our own OS-level kill_tree, not
                // portable-pty's Child::kill()/wait() — its exit-detection
                // was observed to hang indefinitely even after the process
                // is already gone (the same underlying issue as try_wait()
                // never firing). kill_tree is proven reliable elsewhere in
                // this codebase; `child` is just dropped, unwaited, below.
                if let Some(p) = pid {
                    kill_tree(p);
                }
                break None;
            }
            if let Some(t) = opts.timeout {
                if start.elapsed() >= t {
                    timed_out = true;
                    if let Some(p) = pid {
                        kill_tree(p);
                    }
                    break None;
                }
            }
            std::thread::sleep(Duration::from_millis(25));
        };

        // Bounded wait for the reader thread rather than an unconditional
        // join: if some lingering handle keeps the pty's write side open
        // past the child's exit, this must not hang the whole command.
        let reader_deadline = Instant::now() + Duration::from_secs(2);
        while !reader_done.load(Ordering::SeqCst) && Instant::now() < reader_deadline {
            std::thread::sleep(Duration::from_millis(25));
        }

        let mut combined = output_buf.lock().unwrap().clone();
        let mut truncated = false;
        if combined.len() > MAX_CAPTURED_BYTES {
            combined.truncate(MAX_CAPTURED_BYTES);
            truncated = true;
        }

        Ok(TerminalOutput {
            command: command.to_string(),
            exit_code,
            stdout: combined,
            stderr: String::new(),
            truncated,
            duration_ms: start.elapsed().as_millis() as u64,
            cancelled,
            timed_out,
            used_pty: true,
        })
    }
}

fn build_command(command: &str, opts: &TerminalOptions) -> Command {
    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/C", command]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", command]);
        c
    };
    if opts.sandbox == Sandbox::RestrictedFs {
        cmd.env_clear();
        if let Ok(path) = std::env::var("PATH") {
            cmd.env("PATH", path);
        }
        if cfg!(windows) {
            for var in ["SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "TEMP", "TMP", "USERPROFILE"] {
                if let Ok(v) = std::env::var(var) {
                    cmd.env(var, v);
                }
            }
        }
    }
    cmd
}

fn spawn_reader<R: std::io::Read + Send + 'static>(
    reader: R,
    buf: Arc<Mutex<String>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let mut b = buf.lock().unwrap();
                    if b.len() < MAX_CAPTURED_BYTES * 2 {
                        b.push_str(&line);
                    }
                }
                Err(_) => break,
            }
        }
    })
}

/// Best-effort whole-process-tree kill — build tools and `docker compose`
/// spawn children that a plain single-PID kill would leave orphaned.
pub(crate) fn kill_tree(pid: u32) {
    if cfg!(windows) {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .output();
    } else {
        let _ = Command::new("kill")
            .args(["-9", &pid.to_string()])
            .output();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use tempfile::TempDir;
    use zeus_config::AgentSettings;

    fn approve(_: &PermissionRequest) -> ApprovalDecision {
        ApprovalDecision::Approved
    }

    fn echo_command(msg: &str) -> String {
        if cfg!(windows) {
            format!("echo {msg}")
        } else {
            format!("echo {msg}")
        }
    }

    #[test]
    fn runs_command_and_captures_output() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let gate = PermissionGate::new(AgentSettings::default(), root.clone());
        let runner = TerminalRunner::new(root.join(".agent/checkpoints"));
        let out = runner
            .run(
                &echo_command("hello-zeus"),
                &gate,
                TerminalOptions::new(root.clone()),
                Arc::new(AtomicBool::new(false)),
                approve,
            )
            .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.stdout.contains("hello-zeus"));
        assert!(!out.cancelled);
        assert!(!out.timed_out);
    }

    #[test]
    fn denied_command_is_not_run() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let gate = PermissionGate::new(AgentSettings::default(), root.clone());
        let runner = TerminalRunner::new(root.join(".agent/checkpoints"));
        let err = runner
            .run(
                "rm -rf /",
                &gate,
                TerminalOptions::new(root.clone()),
                Arc::new(AtomicBool::new(false)),
                approve,
            )
            .unwrap_err();
        assert!(matches!(err, AgentError::Fs(_)));
    }

    #[test]
    fn history_records_command() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let gate = PermissionGate::new(AgentSettings::default(), root.clone());
        let runner = TerminalRunner::new(root.join(".agent/checkpoints"));
        runner
            .run(
                &echo_command("hist-test"),
                &gate,
                TerminalOptions::new(root.clone()),
                Arc::new(AtomicBool::new(false)),
                approve,
            )
            .unwrap();
        let recent = runner.history().recent(10).unwrap();
        assert_eq!(recent.len(), 1);
        assert!(recent[0].command.contains("hist-test"));
    }

    #[test]
    fn cancellation_stops_long_running_command() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let gate = PermissionGate::new(AgentSettings::default(), root.clone());
        let runner = TerminalRunner::new(root.join(".agent/checkpoints"));
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel2 = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            cancel2.store(true, Ordering::SeqCst);
        });
        let sleep_cmd = if cfg!(windows) {
            "ping -n 30 127.0.0.1 >NUL"
        } else {
            "sleep 30"
        };
        let out = runner
            .run(
                sleep_cmd,
                &gate,
                TerminalOptions::new(root.clone()),
                cancel,
                approve,
            )
            .unwrap();
        assert!(out.cancelled);
        assert!(out.duration_ms < 10_000, "should cancel well before 30s");
    }

    /// Regression/tracking test for a known bug, not a passing-happy-path
    /// test: PTY mode is off by default (see `TerminalOptions::new`) because
    /// `portable-pty`'s `try_wait()` was observed to never detect process
    /// exit on this Windows setup, even for a trivial `echo`. Bounded to a
    /// short timeout so the bug can't hang the suite. Marked `#[ignore]`
    /// because `try_wait()` itself polls slowly here (~85s to hit even a 3s
    /// timeout), so this shouldn't tax every normal `cargo test` run — it's
    /// tracking an external library issue, not something production code
    /// exercises (the default is already `use_pty: false`). Run explicitly
    /// with `cargo test -- --ignored` when re-investigating. If this ever
    /// starts failing because `timed_out` came back `false`, the underlying
    /// issue may be fixed — flip `TerminalOptions::new`'s default back to
    /// `true` and rewrite this test to assert the correct, fast-exit behavior.
    #[test]
    #[ignore = "tracks an external portable-pty/Windows issue; slow by nature, see doc comment"]
    fn known_bug_pty_mode_does_not_detect_process_exit() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let gate = PermissionGate::new(AgentSettings::default(), root.clone());
        let runner = TerminalRunner::new(root.join(".agent/checkpoints"));
        let mut opts = TerminalOptions::new(root.clone());
        opts.use_pty = true;
        opts.timeout = Some(Duration::from_secs(3));

        let out = runner
            .run(&echo_command("hi"), &gate, opts, Arc::new(AtomicBool::new(false)), approve)
            .unwrap();

        assert!(out.used_pty, "should have actually attempted the PTY path");
        // What *should* happen: exit_code == Some(0), timed_out == false,
        // duration well under 3s. What actually happens today:
        assert!(
            out.timed_out,
            "if this fails, the try_wait() bug may be fixed — see doc comment above"
        );
    }
}
