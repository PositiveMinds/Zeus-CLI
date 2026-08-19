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
//! by PID, not by owning the process; on Windows the recorded creation time
//! also guards against PID reuse.

use crate::error::{AgentError, Result};
use crate::terminal::{kill_tree, resume_process, suspend_process};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

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
    /// Windows process creation time (`FILETIME` u64) captured at spawn. The
    /// liveness check compares it against the live process's creation time so
    /// a recycled PID (a *different* process that reused the number) can't
    /// keep an exited task looking `Running` forever. `None` on unix and for
    /// metas written before this field existed.
    #[cfg(windows)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid_started_ft: Option<u64>,
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
        // The counter is a read-modify-write, and `zeus bg spawn` runs in its
        // own fresh process each time — so without locking, two concurrent
        // invocations can both read the same value and mint duplicate ids.
        // Hold an exclusive advisory lock on the counter file itself for the
        // duration of the update (released on drop/process exit, so a crash
        // can't wedge it).
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            // Don't truncate on open: we hold the lock and rewrite via
            // set_len(0) + seek(0) below, and truncating before locking
            // could discard a counter a concurrent process just wrote.
            .truncate(false)
            .open(&path)?;
        file.lock().map_err(AgentError::Io)?;
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut file, &mut buf).map_err(AgentError::Io)?;
        let current: u64 = buf.trim().parse().ok().unwrap_or(0);
        let next = current + 1;
        file.set_len(0).map_err(AgentError::Io)?;
        std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(0)).map_err(AgentError::Io)?;
        std::io::Write::write_all(&mut file, next.to_string().as_bytes())
            .map_err(AgentError::Io)?;
        let _ = file.unlock();
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
            #[cfg(windows)]
            pid_started_ft: capture_creation_time(pid),
        };
        let text =
            serde_json::to_string_pretty(&task).map_err(|e| AgentError::Terminal(e.to_string()))?;
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
            #[cfg(windows)]
            pid_started_ft: capture_creation_time(pid),
        };
        let text =
            serde_json::to_string_pretty(&task).map_err(|e| AgentError::Terminal(e.to_string()))?;
        std::fs::write(self.meta_path(id), text)?;
        Ok(id)
    }

    /// Persist updated task metadata (used when toggling the paused flag).
    fn write_meta(&self, task: &BackgroundTask) -> Result<()> {
        let text =
            serde_json::to_string_pretty(task).map_err(|e| AgentError::Terminal(e.to_string()))?;
        std::fs::write(self.meta_path(task.id), text)
            .map_err(|e| AgentError::Terminal(e.to_string()))
    }

    /// Derive a task's status from its persisted `paused` flag plus liveness.
    fn status_of(task: &BackgroundTask) -> TaskStatus {
        if !task_pid_is_alive(task) {
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
            TaskStatus::Paused => Err(AgentError::Terminal(format!("task {id} is already paused"))),
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
            TaskStatus::Running => Err(AgentError::Terminal(format!("task {id} is not paused"))),
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

    /// Stream a task's captured output live (`tail -f` style): print
    /// everything captured so far, then print new lines as they're appended
    /// to the log files, until the task exits and its logs are fully drained.
    /// Returns promptly — the caller's shell isn't held hostage by a still-
    /// running background task.
    pub fn follow(&self, id: u64) -> Result<()> {
        let Some((_, status)) = self.get(id)? else {
            return Err(AgentError::Terminal(format!(
                "no such background task: {id}"
            )));
        };
        let mut stdout_pos = self.drain_from(self.stdout_path(id), 0);
        let mut stderr_pos = self.drain_from(self.stderr_path(id), 0);
        if status == TaskStatus::Exited {
            return Ok(());
        }
        loop {
            std::thread::sleep(Duration::from_millis(200));
            stdout_pos = self.drain_from(self.stdout_path(id), stdout_pos);
            stderr_pos = self.drain_from(self.stderr_path(id), stderr_pos);
            let status = match self.get(id)? {
                Some((_, status)) => status,
                None => {
                    return Err(AgentError::Terminal(format!(
                        "no such background task: {id}"
                    )))
                }
            };
            if status == TaskStatus::Exited {
                // One final drain so nothing the process flushed just before
                // exiting is left unprinted.
                let _ = self.drain_from(self.stdout_path(id), stdout_pos);
                let _ = self.drain_from(self.stderr_path(id), stderr_pos);
                return Ok(());
            }
        }
    }

    /// Print bytes appended to `path` since `pos`, returning the new end
    /// position. Missing/unreadable files are treated as "nothing new".
    fn drain_from(&self, path: PathBuf, pos: u64) -> u64 {
        let Ok(meta) = std::fs::metadata(&path) else {
            return pos;
        };
        let len = meta.len();
        if len <= pos {
            return pos;
        }
        use std::io::{Read, Seek, SeekFrom};
        let Ok(mut file) = std::fs::File::open(&path) else {
            return pos;
        };
        if file.seek(SeekFrom::Start(pos)).is_err() {
            return pos;
        }
        let mut buf = Vec::new();
        if file.read_to_end(&mut buf).is_err() {
            return pos;
        }
        print!("{}", String::from_utf8_lossy(&buf));
        let _ = std::io::stdout().flush();
        len
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

#[cfg(windows)]
fn is_alive(pid: u32) -> bool {
    crate::terminal::process_is_alive(pid)
}

#[cfg(windows)]
fn capture_creation_time(pid: u32) -> Option<u64> {
    crate::terminal::process_creation_time(pid)
}

#[cfg(windows)]
fn task_pid_is_alive(task: &BackgroundTask) -> bool {
    let alive = is_alive(task.pid);
    if !alive {
        return false;
    }
    // PID-reuse guard: on a busy machine the OS can recycle a PID within
    // seconds, so a *fresh* process may answer for an exited task's number.
    // When we recorded the creation time at spawn, require it to still match
    // — otherwise the number belongs to someone else and the task is gone.
    match task.pid_started_ft {
        Some(expected) => crate::terminal::process_creation_time(task.pid) == Some(expected),
        // Old metas (or an unqueryable spawn) skip the guard — trust liveness.
        None => true,
    }
}

#[cfg(unix)]
fn task_pid_is_alive(task: &BackgroundTask) -> bool {
    is_alive(task.pid)
}

#[cfg(unix)]
fn is_alive(pid: u32) -> bool {
    // kill -0 alone is not enough: an exited-but-unreaped child (a zombie,
    // since the registry drops the Child handle without waiting) still
    // answers kill -0 successfully. Only waitpid(WNOHANG) distinguishes a
    // running child from a zombie. When our child, reap it and report dead;
    // when the pid belongs to another process (the one-shot CLI spawns from
    // a *different* invocation than the session that later lists/stops),
    // waitpid returns ECHILD and we fall back to the kill -0 answer.
    unsafe extern "C" {
        fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
        fn kill(pid: i32, sig: i32) -> i32;
    }
    const WNOHANG: i32 = 1;

    // Raw `kill(pid, 0)` probes liveness without spawning a `kill` subprocess.
    let knock = unsafe { kill(pid as i32, 0) == 0 };
    if !knock {
        return false;
    }
    let mut status: i32 = 0;
    match unsafe { waitpid(pid as i32, &mut status, WNOHANG) } {
        // Reaped a zombie -> process had exited.
        n if n == pid as i32 => false,
        // Still a live, running child.
        0 => true,
        // ECHILD (not our child) or other error: trust kill -0.
        _ => true,
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
            if setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

/// Give a background child its own process group/session on Windows.
/// `DETACHED_PROCESS` creates the child with no console and no inherited
/// console handles, so it keeps running after the spawning `zeus` process
/// exits — and more importantly it does *not* hold the parent's stdout/stderr
/// pipe handles open, which would otherwise make an interactive shell wait
/// for the detached task (e.g. `cmd /C ping -n 600`) before the prompt
/// returns. This is the Windows equivalent of the Unix `setsid` above.
#[cfg(windows)]
fn detach_into_own_session(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;

    const DETACHED_PROCESS: u32 = 0x0000_0008;
    cmd.creation_flags(DETACHED_PROCESS);
}

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

    /// A command that sleeps far longer than any test body can run, so a
    /// pause/resume dance (each step spawns a `tasklist`/`ps`/`kill`
    /// subprocess) can never outlive the process it is operating on. The old
    /// `ping -n 30` / `sleep 30` gave ~29s of headroom, which a heavily
    /// loaded parallel CI run could burn through — the process exited before
    /// the final resume and the test failed spuriously.
    fn long_sleep_cmd() -> &'static str {
        if cfg!(windows) {
            "ping -n 600 127.0.0.1 >NUL"
        } else {
            "sleep 600"
        }
    }

    #[test]
    fn next_id_is_monotonic_across_registry_instances() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join(".agent/background");
        // Two separate registry instances (as separate `zeus bg` processes
        // would be) share the same on-disk counter. Sequential calls from
        // each must mint distinct, increasing ids — no reuse after the other
        // instance bumped the counter.
        let a = BackgroundTaskRegistry::new(dir.clone());
        let b = BackgroundTaskRegistry::new(dir.clone());

        assert_eq!(a.next_id().unwrap(), 1);
        assert_eq!(b.next_id().unwrap(), 2);
        assert_eq!(a.next_id().unwrap(), 3);
        assert_eq!(b.next_id().unwrap(), 4);

        // The counter file was rewritten by the second call, not the first.
        let counter = std::fs::read_to_string(a.counter_path()).unwrap();
        assert_eq!(counter.trim(), "4");
    }

    #[test]
    fn spawn_list_and_stop() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let registry = BackgroundTaskRegistry::new(root.join(".agent/background"));

        let sleep_cmd = long_sleep_cmd();
        let id = registry.spawn(sleep_cmd, &root).unwrap();

        let (task, status) = registry.get(id).unwrap().unwrap();
        assert_eq!(task.id, id);
        assert_eq!(status, TaskStatus::Running);

        let listed = registry.list().unwrap();
        assert!(listed
            .iter()
            .any(|(t, s)| t.id == id && *s == TaskStatus::Running));

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
        assert!(listed
            .iter()
            .any(|(t, s)| t.id == id && *s == TaskStatus::Running));

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
            .spawn_argv(
                "zeus_definitely_not_a_program_xyz",
                vec!["hi".into()],
                &root
            )
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

        let sleep_cmd = long_sleep_cmd();
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
        let sleep_cmd = long_sleep_cmd();
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
        assert!(
            registry.resume(id).is_err(),
            "resuming a running task fails"
        );

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
        let sleep_cmd = long_sleep_cmd();
        registry.spawn(sleep_cmd, &root).unwrap();
        registry.spawn(sleep_cmd, &root).unwrap();
        assert_eq!(registry.list().unwrap().len(), 2);
        registry.shutdown_all().unwrap();
        assert_eq!(registry.list().unwrap().len(), 0);
    }

    #[test]
    fn follow_streams_output_and_returns_when_task_exits() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let registry = BackgroundTaskRegistry::new(root.join(".agent/background"));

        // Emit three lines with a gap between them so `follow` must wait for
        // the later ones, then exit — proving it streams, not just dumps.
        // (`spawn` already wraps in a shell, so no `cmd /C`/`sh -c` here.)
        let cmd = if cfg!(windows) {
            "(echo one & ping -n 2 127.0.0.1 >NUL & echo two & ping -n 2 127.0.0.1 >NUL & echo three)"
        } else {
            "(echo one; sleep 1; echo two; sleep 1; echo three)"
        };
        let id = registry.spawn(cmd, &root).unwrap();

        // Follow should return by the time the task has exited on its own.
        let deadline = Instant::now() + Duration::from_secs(15);
        registry.follow(id).unwrap();
        assert!(
            Instant::now() < deadline,
            "follow must return once the task exits"
        );

        let (stdout, _) = registry.output(id);
        assert!(stdout.contains("one"), "stdout should capture 'one'");
        assert!(stdout.contains("three"), "stdout should capture 'three'");
    }

    #[test]
    fn follow_returns_immediately_for_unknown_task() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let registry = BackgroundTaskRegistry::new(root.join(".agent/background"));
        assert!(registry.follow(999).is_err());
    }

    /// The Windows liveness guard must reject a live PID whose recorded
    /// creation time doesn't match (the PID-reuse failure mode this field
    /// exists for) and accept one that does.
    #[cfg(windows)]
    #[test]
    fn task_pid_is_alive_guards_against_pid_reuse() {
        let tmp = TempDir::new().unwrap();
        let mut child = Command::new("cmd")
            .args(["/C", "ping -n 600 127.0.0.1 >NUL"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let pid = child.id();

        let task = BackgroundTask {
            id: 1,
            command: "test".into(),
            pid,
            cwd: tmp.path().display().to_string(),
            started_at: chrono::Utc::now().to_rfc3339(),
            paused: false,
            pid_started_ft: Some(0),
        };
        assert!(
            !task_pid_is_alive(&task),
            "a live pid whose creation time doesn't match the record must count as dead"
        );

        let task = BackgroundTask {
            pid_started_ft: capture_creation_time(pid),
            ..task
        };
        assert!(
            task_pid_is_alive(&task),
            "a live pid whose creation time matches must count as alive"
        );

        kill_tree(pid);
        let _ = child.wait();
    }
}
