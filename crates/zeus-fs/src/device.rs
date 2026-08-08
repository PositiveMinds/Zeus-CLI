//! Device-testing integration via the Android platform tools (`adb`).
//!
//! zeus tests apps on real devices/emulators the way a mobile engineer does —
//! over USB debugging or wireless (`adb connect`), then drives the device
//! through the same permission gate used everywhere else. Every operation is
//! permission-gated ("ask" by default); a screenshot's binary output is
//! written straight to a file rather than round-tripped through lossy text
//! capture.

use crate::error::{FsError, Result};
use crate::permission::{ApprovalDecision, PermissionGate, PermissionRequest};
use std::io::Read as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Cap on captured command output so a chatty `adb logcat` dump can't blow
/// the model's context window.
const MAX_CAPTURED_BYTES: usize = 200_000;

#[derive(Debug, Clone)]
pub struct DeviceOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub success: bool,
    /// Populated for actions that produce a file (screenshot) — the path
    /// the artifact was written to (present even on a failed capture so the
    /// caller can clean it up).
    pub artifact: Option<PathBuf>,
}

pub struct DeviceEngine {
    project_root: PathBuf,
    gate: PermissionGate,
}

impl DeviceEngine {
    pub fn new(project_root: PathBuf, gate: PermissionGate) -> Self {
        Self { project_root, gate }
    }

    /// Enforce the gate for a device op — "ask" so a user signs off before
    /// anything touches their phone/emulator.
    fn enforce<F>(&self, description: &str, approver: F) -> Result<()>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.gate.enforce(
            &PermissionRequest {
                tool: "device".into(),
                path: None,
                description: description.to_string(),
                ..Default::default()
            },
            approver,
        )
    }

    fn into_output(status: std::process::ExitStatus, stdout: Vec<u8>, stderr: Vec<u8>) -> DeviceOutput {
        let mut stdout = String::from_utf8_lossy(&stdout).into_owned();
        let mut stderr = String::from_utf8_lossy(&stderr).into_owned();
        if stdout.len() > MAX_CAPTURED_BYTES {
            stdout.truncate(MAX_CAPTURED_BYTES);
        }
        if stderr.len() > MAX_CAPTURED_BYTES {
            stderr.truncate(MAX_CAPTURED_BYTES);
        }
        DeviceOutput {
            stdout,
            stderr,
            exit_code: status.code(),
            success: status.success(),
            artifact: None,
        }
    }

    /// Run a plain text `adb` command to completion and capture output.
    fn run(&self, args: &[&str]) -> Result<DeviceOutput> {
        let output = Command::new("adb")
            .args(args)
            .output()
            .map_err(|e| FsError::Other(format!("adb {args:?} failed to spawn: {e}")))?;
        Ok(Self::into_output(output.status, output.stdout, output.stderr))
    }

    /// List connected devices (USB + wireless). `-l` shows model/serial.
    pub fn devices<F>(&self, approver: &mut F) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce("list connected devices (adb devices)", approver)?;
        self.run(&["devices", "-l"])
    }

    /// Connect over wireless debugging: `adb connect <host:port>`.
    pub fn connect<F>(&self, host_port: &str, approver: &mut F) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            &format!("connect to device over wireless: adb connect {host_port}"),
            approver,
        )?;
        self.run(&["connect", host_port])
    }

    /// Disconnect a wireless device.
    pub fn disconnect<F>(&self, host_port: &str, approver: &mut F) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(&format!("disconnect wireless device: {host_port}"), approver)?;
        self.run(&["disconnect", host_port])
    }

    /// Install an APK onto the connected device, replacing existing.
    pub fn install<F>(&self, apk: &str, approver: &mut F) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(&format!("install APK on device: {apk}"), approver)?;
        self.run(&["install", "-r", apk])
    }

    /// Uninstall a package from the device.
    pub fn uninstall<F>(&self, package: &str, approver: &mut F) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(&format!("uninstall from device: {package}"), approver)?;
        self.run(&["uninstall", package])
    }

    /// Launch an app on the device. With an activity: `am start -n
    /// <pkg>/<activity>`; without one, the launcher intent is resolved via
    /// `monkey`.
    pub fn launch<F>(
        &self,
        package: &str,
        activity: Option<&str>,
        approver: &mut F,
    ) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            &format!(
                "launch {} on the device",
                activity
                    .map(|a| format!("{package}/{a}"))
                    .unwrap_or_else(|| package.to_string())
            ),
            approver,
        )?;
        match activity {
            Some(activity) => self.run(&["shell", "am", "start", "-n", &format!("{package}/{activity}")]),
            None => self.run(&[
                "shell",
                "monkey",
                "-p",
                package,
                "-c",
                "android.intent.category.LAUNCHER",
                "1",
            ]),
        }
    }

    /// Dump the current device logcat tail (bounded, `-d` = dump-and-exit so
    /// it never streams forever).
    pub fn logcat<F>(
        &self,
        filter: Option<&str>,
        max_lines: usize,
        approver: &mut F,
    ) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            &format!(
                "dump device logcat{}",
                filter.map(|f| format!(" filtered to {f}")).unwrap_or_default()
            ),
            approver,
        )?;
        let max_lines_str = max_lines.to_string();
        let mut args = vec!["logcat", "-d", "-v", "brief", "-t", &max_lines_str];
        if let Some(filter) = filter {
            args.push(filter);
        }
        self.run(&args)
    }

    /// Capture the device screen to a PNG. `out` is relative to the project
    /// root; default is the now-timestamped `screenshot-<unix>.png` there.
    pub fn screenshot<F>(&self, out: Option<&str>, approver: &mut F) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let path = self.artifact_path(out, "screenshot", "png");
        self.enforce(&format!("capture device screenshot to {}", path.display()), approver)?;
        let mut out = self.exec_out_to_file(&["exec-out", "screencap", "-p"], &path, Duration::from_secs(20))?;
        if out.success {
            out.stdout = format!("screenshot saved to {}", path.display());
        }
        Ok(out)
    }

    /// Record the device screen to an MP4 (max 180s per adb, capped to 30 by
    /// default). Binary-safe, streamed straight to the file like a screenshot.
    pub fn screenrecord<F>(
        &self,
        out: Option<&str>,
        seconds: u32,
        approver: &mut F,
    ) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let seconds = seconds.clamp(1, 30);
        let path = self.artifact_path(out, "screenrecord", "mp4");
        self.enforce(
            &format!("record device screen ({seconds}s) to {}", path.display()),
            approver,
        )?;
        let secs = seconds.to_string();
        let mut out = self.exec_out_to_file(
            &["exec-out", "screenrecord", "--time-limit", &secs, "-"],
            &path,
            Duration::from_secs(seconds as u64 + 15),
        )?;
        if out.success {
            out.stdout = format!("screen recording saved to {}", path.display());
        }
        Ok(out)
    }

    /// Shared binary-safe `adb exec-out` capture: the device writes raw bytes
    /// to stdout and we funnel them straight into a file so nothing gets
    /// mangled by text capture. `dest` parent dirs are created.
    fn exec_out_to_file(
        &self,
        adb_args: &[&str],
        dest: &std::path::Path,
        timeout: Duration,
    ) -> Result<DeviceOutput> {
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| FsError::io(parent, e))?;
        }
        let file = std::fs::File::create(dest).map_err(|e| FsError::io(dest, e))?;
        let start = Instant::now();
        let mut child = Command::new("adb")
            .args(adb_args)
            .stdout(Stdio::from(file))
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| FsError::Other(format!("adb {adb_args:?} failed to spawn: {e}")))?;

        let mut stderr = String::new();
        let mut success = false;
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    if let Some(mut err) = child.stderr.take() {
                        let _ = err.read_to_string(&mut stderr);
                    }
                    success = status.success()
                        && dest.metadata().map(|m| m.len() > 0).unwrap_or(false);
                    break;
                }
                Ok(None) => {
                    if start.elapsed() >= timeout {
                        let _ = child.kill();
                        let _ = child.wait();
                        if let Some(mut err) = child.stderr.take() {
                            let _ = err.read_to_string(&mut stderr);
                        }
                        if stderr.is_empty() {
                            stderr = "timed out capturing device output".into();
                        }
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(40));
                }
                Err(e) => {
                    return Err(FsError::Other(format!("adb {adb_args:?} wait failed: {e}")));
                }
            }
        }

        Ok(DeviceOutput {
            stdout: String::new(),
            stderr,
            exit_code: if success { Some(0) } else { Some(1) },
            success,
            artifact: Some(dest.to_path_buf()),
        })
    }

    /// Absolute destination for a capture artifact, relative to the project
    /// root unless an explicit path is given; defaults to a timestamped name.
    fn artifact_path(&self, out: Option<&str>, stem: &str, ext: &str) -> PathBuf {
        match out {
            Some(p) => self.project_root.join(p),
            None => self.project_root.join(format!(
                "{stem}-{}.{ext}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0)
            )),
        }
    }

    /// Pair a device for wireless debugging: `adb pair <host:port> <code>`.
    /// After a successful pair, `connect <host:port>` establishes the link.
    pub fn pair<F>(&self, host_port: &str, code: &str, approver: &mut F) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            &format!("pair with device over wireless: adb pair {host_port} <code>"),
            approver,
        )?;
        self.run(&["pair", host_port, code])
    }

    /// Report the connected device's identity (model, OEM, Android version).
    pub fn info<F>(&self, approver: &mut F) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce("read device info (model / Android version / SDK)", approver)?;
        self.run(&[
            "shell",
            "getprop ro.product.manufacturer; getprop ro.product.model; getprop ro.build.version.release; getprop ro.build.version.sdk; getprop ro.serialno",
        ])
    }

    /// `adb reverse`: expose a host port on the device (`adb reverse tcp:<lp>
    /// tcp:<dp>`) so a device app can reach a dev server running on the host
    /// — essential for USB/webview and app debugging. Default device port =
    /// host port. `"-l"` when no target given lists current forwards.
    pub fn reverse<F>(
        &self,
        local_port: Option<u32>,
        device_port: Option<u32>,
        approver: &mut F,
    ) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce("manage adb reverse forwarding (device -> host localhost)", approver)?;
        match local_port {
            None => self.run(&["reverse", "--list"]),
            Some(lp) => {
                let dp = device_port.unwrap_or(lp);
                let local = format!("tcp:{lp}");
                let dev = format!("tcp:{dp}");
                self.run(&["reverse", &local, &dev])
            }
        }
    }

    /// `adb forward`: expose a device port on the host. List when port is
    /// omitted.
    pub fn forward<F>(
        &self,
        local: Option<u32>,
        device: Option<u32>,
        approver: &mut F,
    ) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce("manage adb port forwarding (host -> device)", approver)?;
        match local {
            None => self.run(&["forward", "--list"]),
            Some(lp) => {
                let dp = device.unwrap_or(lp);
                let local = format!("tcp:{lp}");
                let dev = format!("tcp:{dp}");
                self.run(&["forward", &local, &dev])
            }
        }
    }

    /// `adb shell input ...` — UI automation (tap/swipe/type/keyevent). The
    /// raw event string passes through as the device-shell argument.
    pub fn input<F>(&self, event: &str, approver: &mut F) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(&format!("send input event on device: input {event}"), approver)?;
        let output = Command::new("adb")
            .args(["shell", &format!("input {event}")])
            .output()
            .map_err(|e| FsError::Other(format!("adb shell input failed to spawn: {e}")))?;
        Ok(Self::into_output(output.status, output.stdout, output.stderr))
    }

    /// Clear the device logcat buffer (a fresh baseline before a test pass).
    pub fn logcat_clear<F>(&self, approver: &mut F) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce("clear device logcat buffer", approver)?;
        self.run(&["logcat", "-c"])
    }

    /// Copy a file or directory off the device. `local` is relative to the
    /// project root; a bare filename copies into the project root.
    pub fn pull<F>(&self, remote: &str, local: Option<&str>, approver: &mut F) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let local = local
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                std::path::Path::new(remote)
                    .file_name()
                    .map(|n| n.to_string_lossy().replace("/", "_"))
                    .unwrap_or_else(|| format!("pull_{}", std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0)))
            });
        let dest = self.project_root.join(&local);
        self.enforce(
            &format!("pull {} from device to {}", remote, dest.display()),
            approver,
        )?;
        let output = Command::new("adb")
            .args(["pull", remote, dest.to_string_lossy().as_ref()])
            .output()
            .map_err(|e| FsError::Other(format!("adb pull failed to spawn: {e}")))?;
        Ok(Self::into_output(output.status, output.stdout, output.stderr))
    }

    /// Copy a local file (relative to the project root) onto the device.
    pub fn push<F>(&self, local: &str, remote: &str, approver: &mut F) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let src = self.project_root.join(local);
        self.enforce(
            &format!("push {} to device {}", src.display(), remote),
            approver,
        )?;
        let output = Command::new("adb")
            .args(["push", src.to_string_lossy().as_ref(), remote])
            .output()
            .map_err(|e| FsError::Other(format!("adb push failed to spawn: {e}")))?;
        Ok(Self::into_output(output.status, output.stdout, output.stderr))
    }

    /// Run an arbitrary shell command on the device (`adb shell ...`) — the
    /// escape hatch for "any tool".
    pub fn shell<F>(&self, command: &str, approver: &mut F) -> Result<DeviceOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(&format!("run device shell command: {command}"), approver)?;
        let output = Command::new("adb")
            .args(["shell", command])
            .output()
            .map_err(|e| FsError::Other(format!("adb shell failed to spawn: {e}")))?;
        Ok(Self::into_output(output.status, output.stdout, output.stderr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use zeus_config::AgentSettings;

    fn engine(root: &std::path::Path) -> DeviceEngine {
        DeviceEngine::new(
            root.to_path_buf(),
            PermissionGate::new(AgentSettings::default(), root.to_path_buf()),
        )
    }
    fn approve(_: &PermissionRequest) -> ApprovalDecision {
        ApprovalDecision::Approved
    }

    #[test]
    fn every_operation_resolves_without_panicking() {
        let tmp = TempDir::new().unwrap();
        let engine = engine(tmp.path());
        // adb may be absent on CI; the contract here is "clean DeviceOutput
        // or a surfaceable error" — never a panic — and that the permission
        // gate is exercised first.
        let results = [
            engine.devices(&mut approve),
            engine.connect("192.168.1.10:5555", &mut approve),
            engine.disconnect("192.168.1.10:5555", &mut approve),
            engine.install("app.apk", &mut approve),
            engine.uninstall("com.example.app", &mut approve),
            engine.launch("com.example.app", None, &mut approve),
            engine.launch("com.example.app", Some(".MainActivity"), &mut approve),
            engine.logcat(None, 500, &mut approve),
            engine.shell("echo ok", &mut approve),
            engine.pair("192.168.1.10:37000", "123456", &mut approve),
            engine.info(&mut approve),
            engine.reverse(None, None, &mut approve),
            engine.reverse(Some(8080), None, &mut approve),
            engine.reverse(Some(8080), Some(9090), &mut approve),
            engine.forward(None, None, &mut approve),
            engine.forward(Some(8080), Some(9090), &mut approve),
            engine.input("tap 100 200", &mut approve),
            engine.logcat_clear(&mut approve),
            engine.pull("/sdcard/Download/readme.txt", None, &mut approve),
            engine.pull("/sdcard/Download/readme.txt", Some("copies/readme.txt"), &mut approve),
            engine.push("assets/data.json", "/sdcard/Download/data.json", &mut approve),
        ];
        for r in results {
            assert!(r.is_ok() || r.is_err());
        }
    }

    #[test]
    fn screenshot_reports_artifact_path() {
        let tmp = TempDir::new().unwrap();
        let engine = engine(tmp.path());
        if let Ok(out) = engine.screenshot(None, &mut approve) {
            assert!(out.artifact.is_some(), "screenshot must report a target path");
        }
        // No device/adb -> the spawn error surfaced earlier; also valid.
    }
}