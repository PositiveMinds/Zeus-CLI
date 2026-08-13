//! Hooks: user-configurable shell scripts for `pre-tool-use` (block/modify a
//! call before it runs), `post-tool-use` (feed output back to the model —
//! this is what makes a diagnostics/test hook useful for self-correction,
//! per the blueprint's Hooks section), and `on-stop` (notify/log, no
//! feedback into the conversation). Project-scoped only (`.agent/hooks/`),
//! matching the directory layout — hooks are project automation, not a
//! global user preference.

use serde_json::Value;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreToolUseOutcome {
    /// Allowed to run. `modified_arguments` replaces the original JSON
    /// arguments when the hook printed valid JSON to stdout.
    Allow {
        modified_arguments: Option<String>,
    },
    Block {
        reason: String,
    },
}

#[derive(Debug, Clone, Copy)]
enum HookKind {
    PowerShell,
    Shell,
    Cmd,
}

pub struct HookRunner {
    hooks_dir: PathBuf,
    project_root: PathBuf,
}

impl HookRunner {
    pub fn new(hooks_dir: PathBuf, project_root: PathBuf) -> Self {
        Self {
            hooks_dir,
            project_root,
        }
    }

    fn find_hook(&self, event: &str) -> Option<(PathBuf, HookKind)> {
        for (suffix, kind) in [
            (".ps1", HookKind::PowerShell),
            (".sh", HookKind::Shell),
            (".cmd", HookKind::Cmd),
            (".bat", HookKind::Cmd),
        ] {
            let path = self.hooks_dir.join(format!("{event}{suffix}"));
            if path.is_file() {
                return Some((path, kind));
            }
        }
        let bare = self.hooks_dir.join(event);
        if bare.is_file() {
            return Some((bare, HookKind::Shell));
        }
        None
    }

    fn build_command(&self, path: &Path, kind: HookKind) -> Command {
        let mut cmd = match kind {
            HookKind::PowerShell => {
                let mut c = Command::new("powershell");
                c.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]);
                c.arg(path);
                c
            }
            HookKind::Shell => {
                let mut c = Command::new("sh");
                c.arg(path);
                c
            }
            HookKind::Cmd => {
                let mut c = Command::new("cmd");
                c.args(["/C"]);
                c.arg(path);
                c
            }
        };
        cmd.current_dir(&self.project_root);
        cmd
    }

    /// Run the `pre-tool-use` hook if one is configured (no-op → `Allow`
    /// with no modification if not). Exit code 0 = allow; non-zero = block,
    /// with stderr as the denial reason surfaced to the model.
    pub fn run_pre_tool_use(&self, tool: &str, arguments: &str) -> PreToolUseOutcome {
        let Some((path, kind)) = self.find_hook("pre-tool-use") else {
            return PreToolUseOutcome::Allow {
                modified_arguments: None,
            };
        };
        let mut cmd = self.build_command(&path, kind);
        cmd.env("ZEUS_HOOK_EVENT", "pre-tool-use");
        cmd.env("ZEUS_HOOK_TOOL", tool);
        cmd.env("ZEUS_HOOK_ARGUMENTS", arguments);
        cmd.env("ZEUS_PROJECT_ROOT", self.project_root.display().to_string());

        match run_capturing(cmd, Duration::from_secs(30)) {
            Ok(output) => {
                if output.success {
                    let trimmed = output.stdout.trim();
                    let modified =
                        if !trimmed.is_empty() && serde_json::from_str::<Value>(trimmed).is_ok() {
                            Some(trimmed.to_string())
                        } else {
                            None
                        };
                    PreToolUseOutcome::Allow {
                        modified_arguments: modified,
                    }
                } else {
                    let reason = if output.stderr.trim().is_empty() {
                        format!(
                            "pre-tool-use hook blocked '{tool}' (exit {:?})",
                            output.exit_code
                        )
                    } else {
                        output.stderr.trim().to_string()
                    };
                    PreToolUseOutcome::Block { reason }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "pre-tool-use hook failed to run; allowing by default");
                PreToolUseOutcome::Allow {
                    modified_arguments: None,
                }
            }
        }
    }

    /// Run the `post-tool-use` hook if configured. Its combined stdout+stderr
    /// (if non-empty) is returned so the caller can append it to the tool
    /// result — this is what lets a diagnostics/test hook drive
    /// self-correction: the model has to actually see the failure.
    pub fn run_post_tool_use(
        &self,
        tool: &str,
        arguments: &str,
        result_content: &str,
        is_error: bool,
    ) -> Option<String> {
        let (path, kind) = self.find_hook("post-tool-use")?;
        let mut cmd = self.build_command(&path, kind);
        cmd.env("ZEUS_HOOK_EVENT", "post-tool-use");
        cmd.env("ZEUS_HOOK_TOOL", tool);
        cmd.env("ZEUS_HOOK_ARGUMENTS", arguments);
        cmd.env("ZEUS_HOOK_RESULT", result_content);
        cmd.env("ZEUS_HOOK_IS_ERROR", is_error.to_string());
        cmd.env("ZEUS_PROJECT_ROOT", self.project_root.display().to_string());

        match run_capturing(cmd, Duration::from_secs(60)) {
            Ok(output) => {
                let combined = format!("{}{}", output.stdout, output.stderr);
                if combined.trim().is_empty() {
                    None
                } else {
                    Some(combined)
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "post-tool-use hook failed to run");
                None
            }
        }
    }

    /// Run the `on-stop` hook if configured. Fire-and-forget: failures are
    /// logged, never surfaced to the conversation — there's no tool result
    /// to attach them to at this point.
    pub fn run_on_stop(&self, session_id: &str, summary: &str) {
        let Some((path, kind)) = self.find_hook("on-stop") else {
            return;
        };
        let mut cmd = self.build_command(&path, kind);
        cmd.env("ZEUS_HOOK_EVENT", "on-stop");
        cmd.env("ZEUS_SESSION_ID", session_id);
        cmd.env("ZEUS_HOOK_SUMMARY", summary);
        cmd.env("ZEUS_PROJECT_ROOT", self.project_root.display().to_string());
        if let Err(e) = run_capturing(cmd, Duration::from_secs(30)) {
            tracing::warn!(error = %e, "on-stop hook failed to run");
        }
    }
}

struct CapturedOutput {
    success: bool,
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
}

/// Spawn + capture stdout/stderr via background reader threads (not a
/// blocking read-after-wait) and enforce `timeout` via polling — the same
/// pattern proven in `terminal.rs`'s piped runner, needed here for the same
/// reason: a hook that writes enough output to fill a pipe buffer before
/// exiting would otherwise deadlock against a read-after-exit.
fn run_capturing(mut cmd: Command, timeout: Duration) -> std::io::Result<CapturedOutput> {
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn()?;

    let stdout_buf = Arc::new(Mutex::new(String::new()));
    let stderr_buf = Arc::new(Mutex::new(String::new()));
    let stdout_done = Arc::new(AtomicBool::new(false));
    let stderr_done = Arc::new(AtomicBool::new(false));
    let _ = child
        .stdout
        .take()
        .map(|h| spawn_reader(h, stdout_buf.clone(), stdout_done.clone()));
    let _ = child
        .stderr
        .take()
        .map(|h| spawn_reader(h, stderr_buf.clone(), stderr_done.clone()));

    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break s;
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "hook timed out",
            ));
        }
        std::thread::sleep(Duration::from_millis(25));
    };

    // Bounded wait for the readers instead of an unconditional join: a
    // lingering pipe from a grandchild hook process can keep the reader side
    // open past the child's exit, and an unbounded join would hang the turn
    // (same class of hang guarded in the terminal runner).
    let reader_deadline = Instant::now() + Duration::from_millis(500);
    while Instant::now() < reader_deadline
        && !(stdout_done.load(Ordering::SeqCst) && stderr_done.load(Ordering::SeqCst))
    {
        std::thread::sleep(Duration::from_millis(10));
    }

    let stdout = stdout_buf.lock().unwrap().clone();
    let stderr = stderr_buf.lock().unwrap().clone();
    Ok(CapturedOutput {
        success: status.success(),
        exit_code: status.code(),
        stdout,
        stderr,
    })
}

fn spawn_reader<R: Read + Send + 'static>(
    mut reader: R,
    buf: Arc<Mutex<String>>,
    done: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        if reader.read_to_end(&mut bytes).is_ok() {
            buf.lock()
                .unwrap()
                .push_str(&String::from_utf8_lossy(&bytes));
        }
        done.store(true, Ordering::SeqCst);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_hook(dir: &Path, name: &str, body: &str) {
        std::fs::create_dir_all(dir).unwrap();
        let ext = if cfg!(windows) { ".cmd" } else { ".sh" };
        std::fs::write(dir.join(format!("{name}{ext}")), body).unwrap();
    }

    fn win_or_unix(win: &str, unix: &str) -> String {
        if cfg!(windows) {
            win.to_string()
        } else {
            unix.to_string()
        }
    }

    #[test]
    fn no_hook_configured_allows_unmodified() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let runner = HookRunner::new(root.join(".agent/hooks"), root.clone());
        let outcome = runner.run_pre_tool_use("write", r#"{"path":"a.txt"}"#);
        assert_eq!(
            outcome,
            PreToolUseOutcome::Allow {
                modified_arguments: None
            }
        );
    }

    #[test]
    fn pre_tool_use_hook_can_block() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let hooks_dir = root.join(".agent/hooks");
        let body = win_or_unix(
            "@echo off\r\necho blocked by policy 1>&2\r\nexit /b 1\r\n",
            "#!/bin/sh\necho 'blocked by policy' 1>&2\nexit 1\n",
        );
        write_hook(&hooks_dir, "pre-tool-use", &body);
        let runner = HookRunner::new(hooks_dir, root);
        let outcome = runner.run_pre_tool_use("delete", r#"{"path":"a.txt"}"#);
        match outcome {
            PreToolUseOutcome::Block { reason } => assert!(reason.contains("blocked by policy")),
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[test]
    fn pre_tool_use_hook_can_modify_arguments() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let hooks_dir = root.join(".agent/hooks");
        let body = win_or_unix(
            "@echo off\r\necho {\"path\":\"b.txt\"}\r\n",
            "#!/bin/sh\necho '{\"path\":\"b.txt\"}'\n",
        );
        write_hook(&hooks_dir, "pre-tool-use", &body);
        let runner = HookRunner::new(hooks_dir, root);
        let outcome = runner.run_pre_tool_use("write", r#"{"path":"a.txt"}"#);
        match outcome {
            PreToolUseOutcome::Allow {
                modified_arguments: Some(args),
            } => {
                assert!(args.contains("b.txt"));
            }
            other => panic!("expected modified Allow, got {other:?}"),
        }
    }

    #[test]
    fn post_tool_use_hook_output_is_surfaced() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let hooks_dir = root.join(".agent/hooks");
        let body = win_or_unix(
            "@echo off\r\necho 2 errors found\r\n",
            "#!/bin/sh\necho '2 errors found'\n",
        );
        write_hook(&hooks_dir, "post-tool-use", &body);
        let runner = HookRunner::new(hooks_dir, root);
        let output = runner.run_post_tool_use("edit", "{}", "edited a.txt", false);
        assert!(output.unwrap().contains("2 errors found"));
    }

    #[test]
    fn on_stop_hook_runs_without_panicking() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let hooks_dir = root.join(".agent/hooks");
        let body = win_or_unix("@echo off\r\necho done\r\n", "#!/bin/sh\necho done\n");
        write_hook(&hooks_dir, "on-stop", &body);
        let runner = HookRunner::new(hooks_dir, root);
        runner.run_on_stop("session-1", "summary text");
    }
}
