//! Background task registry for long-running/persistent commands (dev
//! servers, `docker compose up`) that outlive a single foreground
//! run-and-wait invocation.
//!
//! Because `zeus` is a one-shot CLI — each subcommand is a fresh process —
//! an in-memory registry would be useless: a task spawned by one invocation
//! has to be listable/stoppable from a *separate, later* invocation. So
//! this tracks tasks by persisting metadata to `<dir>/<id>.json` and
//! redirecting each child's stdout/stderr straight to log files on disk,
//! rather than holding pipes/`Child` handles in memory. Liveness is checked
//! by PID (`tasklist`/`kill -0`), not by owning the process.

use crate::error::{AgentError, Result};
use crate::terminal::{kill_tree, resume_process, suspend_process};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Running,
    /// Process is temporarily suspended (SIGSTOP / NtSuspendProcess); its
    /// threads are frozen but the process hasn't exited. `resume` continues
    /// it exactly where it stopped.
    Paused,
    /// Process is gone. Exit code isn't recoverable this way once the
    /// spawning process has exited, so this doesn't distinguish clean exit
    /// from crash — check the log files for that.
    Exited,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundTask {
    pub id: u64,
    pub command: String,
    pub pid: u32,
    pub cwd: String,
    pub started_at: String,
    /// Whether the process is currently suspended. Persisted so a later
    /// `zeus bg list` invocation (a fresh process) can report Paused.
    #[serde(default)]
    pub paused: bool,
}

pub struct BackgroundTaskRegistry {
    dir: PathBuf,
}

impl BackgroundTaskRegistry {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    fn meta_path(&self, id: u64) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }
    fn stdout_path(&self, id: u64) -> PathBuf {
        self.dir.join(format!("{id}.stdout.log"))
    }
    fn stderr_path(&self, id: u64) -> PathBuf {
        self.dir.join(format!("{id}.stderr.log"))
    }
    fn counter_path(&self) -> PathBuf {
        self.dir.join("next_id")
    }

    fn next_id(&self) -> Result<u64> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.counter_path();
        let current: u64 = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let next = current + 1;
        std::fs::write(&path, next.to_string())?;
        Ok(next)
    }

    /// Spawn a command that keeps running after this process exits. Output
    /// is redirected to `<id>.stdout.log` / `<id>.stderr.log` under the
    /// registry directory rather than captured in memory.
    pub fn spawn(&self, command: &str, cwd: &Path) -> Result<u64> {
        std::fs::create_dir_all(&self.dir)?;
        let id = self.next_id()?;

        let stdout_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.stdout_path(id))?;
        let stderr_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.stderr_path(id))?;

        let mut cmd = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", command]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", command]);
            c
        };
        cmd.current_dir(cwd);
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::from(stdout_file));
        cmd.stderr(Stdio::from(stderr_file));
        detach_into_own_session(&mut cmd);

        let child = cmd
            .spawn()
            .map_err(|e| AgentError::Terminal(format!("spawn failed: {e}")))?;
        let pid = child.id();
        // Deliberately drop without waiting — this *is* the detach for a
        // one-shot CLI's process model. The OS keeps the child running once
        // `zeus` exits; we track it by PID + metadata file, not a live handle.
        drop(child);

        let task = BackgroundTask {
            id,
            command: format!("shell: {command}"),
            pid,
            cwd: cwd.display().to_string(),
            started_at: chrono::Utc::now().to_rfc3339(),
            paused: false,
        };
        let text = serde_json::to_string_pretty(&task)
            .map_err(|e| AgentError::Terminal(e.to_string()))?;
        std::fs::write(self.meta_path(id), text)?;
        Ok(id)
    }

    /// Spawn an arbitrary program directly (no shell wrapping), keeping it
    /// running after this process exits. Output goes to `<id>.stdout.log` /
    /// `<id>.stderr.log` like `spawn`. Used for background **orchestrated
    /// agent runs**, where shell-wrapping the goal would let its content be
    /// mangled by quoting/expansion — argv is passed through untouched.
    pub fn spawn_argv<I: IntoIterator<Item = String>>(
        &self,
        program: &str,
        args: I,
        cwd: &Path,
    ) -> Result<u64> {
        std::fs::create_dir_all(&self.dir)?;
        let id = self.next_id()?;

        let stdout_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.stdout_path(id))?;
        let stderr_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.stderr_path(id))?;

        let args: Vec<String> = args.into_iter().collect();

        let mut cmd = Command::new(program);
        cmd.args(&args);
        cmd.current_dir(cwd);
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::from(stdout_file));
        cmd.stderr(Stdio::from(stderr_file));
        detach_into_own_session(&mut cmd);

        let child = cmd
            .spawn()
            .map_err(|e| AgentError::Terminal(format!("spawn failed: {e}")))?;
        let pid = child.id();
        drop(child);

        let task = BackgroundTask {
            id,
            command: format!("{program} {}", id_desc(&args)),
            pid,
            cwd: cwd.display().to_string(),
            started_at: chrono::Utc::now().to_rfc3339(),
            paused: false,
        };
        let text = serde_json::to_string_pretty(&task)
            .map_err(|e| AgentError::Terminal(e.to_string()))?;
        std::fs::write(self.meta_path(id), text)?;
        Ok(id)
    }

    /// Persist updated task metadata (used when toggling the paused flag).
    fn write_meta(&self, task: &BackgroundTask) -> Result<()> {
        let text = serde_json::to_string_pretty(task)
            .map_err(|e| AgentError::Terminal(e.to_string()))?;
        std::fs::write(self.meta_path(task.id), text).map_err(|e| AgentError::Terminal(e.to_string()))
    }

    /// Derive a task's status from its persisted `paused` flag plus liveness.
    fn status_of(task: &BackgroundTask) -> TaskStatus {
        if !is_alive(task.pid) {
            TaskStatus::Exited
        } else if task.paused {
            TaskStatus::Paused
        } else {
            TaskStatus::Running
        }
    }

    pub fn list(&self) -> Result<Vec<(BackgroundTask, TaskStatus)>> {
        std::fs::create_dir_all(&self.dir)?;
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let text = std::fs::read_to_string(&path)?;
            if let Ok(task) = serde_json::from_str::<BackgroundTask>(&text) {
                out.push((task.clone(), Self::status_of(&task)));
            }
        }
        out.sort_by_key(|(t, _)| t.id);
        Ok(out)
    }

    pub fn get(&self, id: u64) -> Result<Option<(BackgroundTask, TaskStatus)>> {
        let path = self.meta_path(id);
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)?;
        let task: BackgroundTask =
            serde_json::from_str(&text).map_err(|e| AgentError::Terminal(e.to_string()))?;
        Ok(Some((task.clone(), Self::status_of(&task))))
    }

    /// Suspend a running task in place. The process keeps its PID and its
    /// log files; nothing is re-spawned on resume.
    pub fn pause(&self, id: u64) -> Result<()> {
        let Some((mut task, status)) = self.get(id)? else {
            return Err(AgentError::Terminal(format!(
                "no such background task: {id}"
            )));
        };
        match status {
            TaskStatus::Exited => Err(AgentError::Terminal(format!(
                "task {id} has already exited — nothing to pause"
            ))),
            TaskStatus::Paused => Err(AgentError::Terminal(format!(
                "task {id} is already paused"
            ))),
            TaskStatus::Running => {
                suspend_process(task.pid)
                    .map_err(|e| AgentError::Terminal(format!("pause task {id}: {e}")))?;
                task.paused = true;
                self.write_meta(&task)?;
                Ok(())
            }
        }
    }

    /// Continue a paused task from exactly where it was suspended.
    pub fn resume(&self, id: u64) -> Result<()> {
        let Some((mut task, status)) = self.get(id)? else {
            return Err(AgentError::Terminal(format!(
                "no such background task: {id}"
            )));
        };
        match status {
            TaskStatus::Exited => Err(AgentError::Terminal(format!(
                "task {id} has already exited — nothing to resume"
            ))),
            TaskStatus::Running => Err(AgentError::Terminal(format!(
                "task {id} is not paused"
            ))),
            TaskStatus::Paused => {
                resume_process(task.pid)
                    .map_err(|e| AgentError::Terminal(format!("resume task {id}: {e}")))?;
                task.paused = false;
                self.write_meta(&task)?;
                Ok(())
            }
        }
    }

    /// Captured output so far: (stdout, stderr), read straight from the log
    /// files — works from any process, not just the one that spawned it.
    pub fn output(&self, id: u64) -> (String, String) {
        (
            std::fs::read_to_string(self.stdout_path(id)).unwrap_or_default(),
            std::fs::read_to_string(self.stderr_path(id)).unwrap_or_default(),
        )
    }

    pub fn stop(&self, id: u64) -> Result<()> {
        let Some((task, status)) = self.get(id)? else {
            return Err(AgentError::Terminal(format!(
                "no such background task: {id}"
            )));
        };
        if status == TaskStatus::Exited {
            let _ = std::fs::remove_file(self.meta_path(id));
            return Ok(());
        }
        // Suspended processes can't always be killed by killing their
        // children first (taskkill /T walks them fine, but a plain SIGKILL
        // leaves stopped children parked) — resume briefly so nothing is
        // left frozen behind the task.
        let _ = resume_process(task.pid);
        kill_tree(task.pid);
        let _ = std::fs::remove_file(self.meta_path(id));
        Ok(())
    }

    /// Stop every tracked task — call on session end so nothing is left
    /// orphaned (a background dev server outliving the agent session is
    /// exactly the failure mode this guards against).
    pub fn shutdown_all(&self) -> Result<()> {
        for (task, status) in self.list()? {
            if status == TaskStatus::Exited {
                let _ = std::fs::remove_file(self.meta_path(task.id));
                continue;
            }
            let _ = resume_process(task.pid);
            kill_tree(task.pid);
            let _ = std::fs::remove_file(self.meta_path(task.id));
        }
        Ok(())
    }
}

/// Compact argv summary for the persisted command field.
fn id_desc(args: &[String]) -> String {
    if args.is_empty() {
        return "(no args)".into();
    }
    let mut s = args.join(" ");
    if s.len() > 80 {
        s = s.chars().take(77).collect::<String>();
        s.push_str("...");
    }
    s
}

fn is_alive(pid: u32) -> bool {    if cfg!(windows) {
        match Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
        {
            Ok(o) => String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()),
            Err(_) => false,
        }
    } else {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}

/// Give a background child its own process group/session on Unix. Without
/// this, the child (e.g. `sh -c "sleep 30"`) shares the caller's process
/// group — and a later `kill -STOP -<pgid>` / `kill -9 -<pgid>` would hit
/// the caller (and, in tests, the test harness) instead of just the task.
#[cfg(unix)]
fn detach_into_own_session(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;

    // Raw FFI avoids a libc dependency just for setsid(2).
    unsafe extern "C" {
        fn setsid() -> i32;
    }

    unsafe {
        cmd.pre_exec(|| {
            // The child inherited the caller's process group; setsid() makes
            // it a session leader with its own group so later -<pgid> signals
            // (suspend/resume/kill-tree) can't touch the caller.
            if unsafe { setsid() } == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(windows)]
fn detach_into_own_session(_cmd: &mut Command) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    fn wait_for<F: Fn() -> bool>(f: F, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        while !f() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    #[test]
    fn spawn_list_and_stop() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let registry = BackgroundTaskRegistry::new(root.join(".agent/background"));

        let sleep_cmd = if cfg!(windows) {
            "ping -n 30 127.0.0.1 >NUL"
        } else {
            "sleep 30"
        };
        let id = registry.spawn(sleep_cmd, &root).unwrap();

        let (task, status) = registry.get(id).unwrap().unwrap();
        assert_eq!(task.id, id);
        assert_eq!(status, TaskStatus::Running);

        let listed = registry.list().unwrap();
        assert!(listed.iter().any(|(t, s)| t.id == id && *s == TaskStatus::Running));

        registry.stop(id).unwrap();
        assert!(registry.get(id).unwrap().is_none());
    }

    #[test]
    fn task_exits_naturally_and_output_is_on_disk() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let registry = BackgroundTaskRegistry::new(root.join(".agent/background"));
        let id = registry.spawn("echo hi", &root).unwrap();

        wait_for(
            || matches!(registry.get(id).unwrap(), Some((_, TaskStatus::Exited))),
            Duration::from_secs(5),
        );
        let (_, status) = registry.get(id).unwrap().unwrap();
        assert_eq!(status, TaskStatus::Exited);
        let (stdout, _) = registry.output(id);
        assert!(stdout.contains("hi"));
    }

    #[test]
    fn spawn_argv_passes_args_untouched_and_lists_running() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let registry = BackgroundTaskRegistry::new(root.join(".agent/background"));

        // Spawn a long-lived-ish child with meaningful argv; we only assert
        // it starts Running and is tracked like shell spawns — the point is
        // the argv path exists for detached headless orchestration (goals
        // with quotes/spaces survive without shell polish).
        let (program, args) = if cfg!(windows) {
            (
                "powershell".to_string(),
                vec![
                    "-NoProfile".to_string(),
                    "-Command".to_string(),
                    "Start-Sleep -Seconds 30".to_string(),
                ],
            )
        } else {
            ("sleep".to_string(), vec!["30".to_string()])
        };
        let id = registry.spawn_argv(&program, args, &root).unwrap();

        let (task, status) = registry.get(id).unwrap().unwrap();
        assert_eq!(task.id, id);
        assert_eq!(status, TaskStatus::Running);
        assert!(task.command.contains(&program));

        let listed = registry.list().unwrap();
        assert!(listed.iter().any(|(t, s)| t.id == id && *s == TaskStatus::Running));

        registry.stop(id).unwrap();
        assert!(registry.get(id).unwrap().is_none());
    }

    #[test]
    fn spawn_argv_requires_existing_program() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let registry = BackgroundTaskRegistry::new(root.join(".agent/background"));
        assert!(registry
            .spawn_argv("zeus_definitely_not_a_program_xyz", vec!["hi".into()], &root)
            .is_err());
    }

    #[test]
    fn survives_across_separate_registry_instances() {
        // Simulates "spawn in one CLI invocation, list/stop in a later one" —
        // the whole reason this is file-backed rather than in-memory.
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let dir = root.join(".agent/background");

        let sleep_cmd = if cfg!(windows) {
            "ping -n 30 127.0.0.1 >NUL"
        } else {
            "sleep 30"
        };
        let id = {
            let first = BackgroundTaskRegistry::new(dir.clone());
            first.spawn(sleep_cmd, &root).unwrap()
        }; // `first` dropped — nothing in memory ties it to the running process

        let second = BackgroundTaskRegistry::new(dir);
        let (task, status) = second.get(id).unwrap().unwrap();
        assert_eq!(task.id, id);
        assert_eq!(status, TaskStatus::Running);
        second.stop(id).unwrap();
    }

    #[test]
    fn pauses_and_resumes_in_place() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let registry = BackgroundTaskRegistry::new(root.join(".agent/background"));
        let sleep_cmd = if cfg!(windows) {
            "ping -n 30 127.0.0.1 >NUL"
        } else {
            "sleep 30"
        };
        let id = registry.spawn(sleep_cmd, &root).unwrap();

        // Cancelled paused state rejected, running accepted.
        assert!(registry.pause(id).is_ok());
        let (_, status) = registry.get(id).unwrap().unwrap();
        assert_eq!(status, TaskStatus::Paused);
        // Pausing twice is an error, not a no-op.
        assert!(registry.pause(id).is_err());
        // The process is still alive while paused, just frozen.
        let pid = registry.get(id).unwrap().unwrap().0.pid;
        assert!(is_alive(pid), "paused process must stay alive");

        registry.resume(id).unwrap();
        let (_, status) = registry.get(id).unwrap().unwrap();
        assert_eq!(status, TaskStatus::Running);
        assert!(registry.resume(id).is_err(), "resuming a running task fails");

        // Persisted: a fresh registry instance still sees the paused state.
        registry.pause(id).unwrap();
        let fresh = BackgroundTaskRegistry::new(root.join(".agent/background"));
        let (_, status) = fresh.get(id).unwrap().unwrap();
        assert_eq!(status, TaskStatus::Paused);
        registry.resume(id).unwrap();

        registry.stop(id).unwrap();
        assert!(registry.get(id).unwrap().is_none());
    }

    #[test]
    fn shutdown_all_stops_everything() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let registry = BackgroundTaskRegistry::new(root.join(".agent/background"));
        let sleep_cmd = if cfg!(windows) {
            "ping -n 30 127.0.0.1 >NUL"
        } else {
            "sleep 30"
        };
        registry.spawn(sleep_cmd, &root).unwrap();
        registry.spawn(sleep_cmd, &root).unwrap();
        assert_eq!(registry.list().unwrap().len(), 2);
        registry.shutdown_all().unwrap();
        assert_eq!(registry.list().unwrap().len(), 0);
    }
}
