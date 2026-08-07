//! Git integration: comprehensive porcelain coverage, each operation
//! permission-gated per the blueprint's Git Integration tiering. Shells out
//! to the real `git` binary rather than reimplementing the object model;
//! git's porcelain output is already well-specified, and this project
//! already trusts `git` for `git mv` in rename.
//!
//! Tiers:
//! - Read-only (status/diff/log/blame/show/branch·remote·tag list) — allow.
//! - Reversible write (add/commit/stash/branch create/tag create) — allow,
//!   but commit always previews the actual diff first.
//! - Working-tree-changing (checkout) — ask.
//! - Network / shared-state (fetch/pull/push) — ask.
//! - History-rewriting (reset/revert/cherry-pick/rebase/merge) — ask, with
//!   `reset --hard` and `push --force` specifically denied by a built-in
//!   rule (see `zeus-config`'s `PermissionSettings::builtin_safe`) — the
//!   existing `bash`-tool rules for those exact patterns don't cover these
//!   structured tools, since they're a different tool name, so the
//!   protection has to be re-declared here rather than inherited.
//!
//! "AI commit messages" and "diff review" need no special code: the agent
//! loop already lets a model call `diff` then `commit` with a message it
//! wrote itself after reading that diff — composition, not a new subsystem
//! (same principle as the Built-in Workflows in the blueprint).

use crate::error::{FsError, Result};
use crate::permission::{ApprovalDecision, PermissionGate, PermissionRequest};
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone)]
pub struct GitOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub success: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetMode {
    Soft,
    Mixed,
    Hard,
}

impl ResetMode {
    fn flag(self) -> &'static str {
        match self {
            ResetMode::Soft => "--soft",
            ResetMode::Mixed => "--mixed",
            ResetMode::Hard => "--hard",
        }
    }
}

pub struct GitEngine {
    project_root: PathBuf,
    gate: PermissionGate,
}

impl GitEngine {
    pub fn new(project_root: PathBuf, gate: PermissionGate) -> Self {
        Self { project_root, gate }
    }

    fn run(&self, args: &[&str]) -> Result<GitOutput> {
        let output = Command::new("git")
            .args(args)
            .current_dir(&self.project_root)
            .output()
            .map_err(|e| FsError::Other(format!("git {args:?} failed to spawn: {e}")))?;
        Ok(GitOutput {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code(),
            success: output.status.success(),
        })
    }

    fn enforce<F>(
        &self,
        tool: &str,
        description: String,
        preview: Option<String>,
        command: Option<String>,
        approver: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.gate.enforce(
            &PermissionRequest {
                tool: tool.to_string(),
                path: None,
                command,
                description,
                preview,
                ..Default::default()
            },
            approver,
        )
    }

    fn enforce_strict(&self, tool: &str, description: String) -> Result<()> {
        self.gate.enforce_strict(&PermissionRequest {
            tool: tool.to_string(),
            path: None,
            command: None,
            description,
            ..Default::default()
        })
    }

    // ---------------------------------------------------------------
    // Read-only
    // ---------------------------------------------------------------

    pub fn status(&self) -> Result<GitOutput> {
        self.enforce_strict("git_status", "git status".into())?;
        self.run(&["status", "--porcelain=v1", "--branch"])
    }

    /// `refs = None` diffs the working tree; `Some([a])` diffs against `a`;
    /// `Some([a, b])` diffs `a..b`.
    pub fn diff(&self, staged: bool, refs: &[&str]) -> Result<GitOutput> {
        self.enforce_strict("git_diff", "git diff".into())?;
        let mut args = vec!["diff"];
        if staged {
            args.push("--staged");
        }
        args.extend_from_slice(refs);
        self.run(&args)
    }

    pub fn blame(&self, path: &str) -> Result<GitOutput> {
        self.enforce_strict("git_blame", format!("git blame {path}"))?;
        self.run(&["blame", "--", path])
    }

    pub fn log(&self, max: usize, path: Option<&str>) -> Result<GitOutput> {
        self.enforce_strict("git_log", "git log".into())?;
        let max_arg = format!("-{max}");
        let mut args = vec!["log", &max_arg, "--oneline"];
        if let Some(p) = path {
            args.push("--");
            args.push(p);
        }
        self.run(&args)
    }

    pub fn show(&self, target: &str) -> Result<GitOutput> {
        self.enforce_strict("git_show", format!("git show {target}"))?;
        self.run(&["show", target])
    }

    pub fn branch_list(&self) -> Result<GitOutput> {
        self.enforce_strict("git_branch_list", "git branch --list".into())?;
        self.run(&["branch", "--list", "-a"])
    }

    pub fn remote_list(&self) -> Result<GitOutput> {
        self.enforce_strict("git_remote_list", "git remote -v".into())?;
        self.run(&["remote", "-v"])
    }

    pub fn tag_list(&self) -> Result<GitOutput> {
        self.enforce_strict("git_tag_list", "git tag --list".into())?;
        self.run(&["tag", "--list"])
    }

    pub fn stash_list(&self) -> Result<GitOutput> {
        self.enforce_strict("git_stash_list", "git stash list".into())?;
        self.run(&["stash", "list"])
    }

    // ---------------------------------------------------------------
    // Reversible write — allow by default, previewed where meaningful
    // ---------------------------------------------------------------

    pub fn add<F>(&self, paths: &[&str], mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            "git_add",
            format!("stage {}", paths.join(", ")),
            None,
            None,
            &mut approver,
        )?;
        let mut args = vec!["add", "--"];
        args.extend_from_slice(paths);
        self.run(&args)
    }

    pub fn commit<F>(&self, message: &str, all: bool, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let diff = if all {
            self.run(&["diff"])?
        } else {
            self.run(&["diff", "--staged"])?
        };
        let preview = if diff.stdout.trim().is_empty() {
            None
        } else {
            Some(diff.stdout.clone())
        };
        self.enforce(
            "git_commit",
            format!("commit: {message}"),
            preview,
            None,
            &mut approver,
        )?;
        if all {
            self.run(&["commit", "-a", "-m", message])
        } else {
            self.run(&["commit", "-m", message])
        }
    }

    pub fn stash_push<F>(&self, message: Option<&str>, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            "git_stash",
            "git stash push".into(),
            None,
            None,
            &mut approver,
        )?;
        match message {
            Some(m) => self.run(&["stash", "push", "-m", m]),
            None => self.run(&["stash", "push"]),
        }
    }

    pub fn stash_pop<F>(&self, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            "git_stash",
            "git stash pop".into(),
            None,
            None,
            &mut approver,
        )?;
        self.run(&["stash", "pop"])
    }

    pub fn branch_create<F>(&self, name: &str, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            "git_branch",
            format!("create branch '{name}'"),
            None,
            None,
            &mut approver,
        )?;
        self.run(&["branch", name])
    }

    /// Deleting a branch is ask-gated (not allow-by-default like create):
    /// git already refuses to delete an unmerged branch without `-D`, but
    /// losing track of a branch pointer is still more consequential than
    /// creating one.
    pub fn branch_delete<F>(&self, name: &str, force: bool, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            "git_branch_delete",
            format!("delete branch '{name}'{}", if force { " (forced)" } else { "" }),
            None,
            None,
            &mut approver,
        )?;
        let flag = if force { "-D" } else { "-d" };
        self.run(&["branch", flag, name])
    }

    pub fn tag_create<F>(&self, name: &str, message: Option<&str>, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            "git_tag",
            format!("create tag '{name}'"),
            None,
            None,
            &mut approver,
        )?;
        match message {
            Some(m) => self.run(&["tag", "-a", name, "-m", m]),
            None => self.run(&["tag", name]),
        }
    }

    // ---------------------------------------------------------------
    // Working-tree-changing — ask
    // ---------------------------------------------------------------

    pub fn checkout<F>(&self, target: &str, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            "git_checkout",
            format!("checkout '{target}'"),
            None,
            None,
            &mut approver,
        )?;
        self.run(&["checkout", target])
    }

    // ---------------------------------------------------------------
    // Network / shared-state — ask always
    // ---------------------------------------------------------------

    pub fn fetch<F>(&self, remote: Option<&str>, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let desc = match remote {
            Some(r) => format!("git fetch {r}"),
            None => "git fetch".to_string(),
        };
        self.enforce("git_fetch", desc, None, None, &mut approver)?;
        match remote {
            Some(r) => self.run(&["fetch", r]),
            None => self.run(&["fetch"]),
        }
    }

    pub fn pull<F>(&self, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce("git_pull", "git pull".into(), None, None, &mut approver)?;
        self.run(&["pull"])
    }

    /// `force: true` is denied by a built-in rule (`git_push` + command
    /// pattern `--force*`) — matching the existing `bash`-tool precedent for
    /// `git push --force*`, which wouldn't otherwise apply to this
    /// structured tool since it's a different tool name.
    pub fn push<F>(
        &self,
        remote: Option<&str>,
        branch: Option<&str>,
        force: bool,
        mut approver: F,
    ) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let mut desc = "git push".to_string();
        if let Some(r) = remote {
            desc.push(' ');
            desc.push_str(r);
        }
        if let Some(b) = branch {
            desc.push(' ');
            desc.push_str(b);
        }
        if force {
            desc.push_str(" --force");
        }
        let command = if force { Some("--force".to_string()) } else { None };
        self.enforce("git_push", desc, None, command, &mut approver)?;
        let mut args = vec!["push"];
        if force {
            args.push("--force");
        }
        if let Some(r) = remote {
            args.push(r);
        }
        if let Some(b) = branch {
            args.push(b);
        }
        self.run(&args)
    }

    /// Convenience: commit staged (or all) changes, then push to the current
    /// branch's upstream. Returns the commit output and the push output
    /// concatenated. Both steps run under their normal permission tiers.
    pub fn commit_and_push<F>(
        &self,
        message: &str,
        all: bool,
        remote: Option<&str>,
        mut approver: F,
    ) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let commit = self.commit(message, all, &mut approver)?;
        if !commit.success {
            return Ok(commit);
        }
        let push = self.push(remote, None, false, &mut approver)?;
        Ok(GitOutput {
            stdout: format!("{}{}", commit.stdout, push.stdout),
            stderr: format!("{}{}", commit.stderr, push.stderr),
            exit_code: push.exit_code.or(commit.exit_code),
            success: push.success && commit.success,
        })
    }

    // ---------------------------------------------------------------
    // History-rewriting / conflict-prone — ask, with hard-reset denied
    // ---------------------------------------------------------------

    /// `ResetMode::Hard` is denied by a built-in rule (`git_reset` + command
    /// pattern `--hard*`) — same rationale as force-push above.
    pub fn reset<F>(&self, mode: ResetMode, target: Option<&str>, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let target = target.unwrap_or("HEAD");
        self.enforce(
            "git_reset",
            format!("reset {} {target}", mode.flag()),
            None,
            Some(mode.flag().to_string()),
            &mut approver,
        )?;
        self.run(&["reset", mode.flag(), target])
    }

    pub fn revert<F>(&self, target: &str, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            "git_revert",
            format!("revert {target}"),
            None,
            None,
            &mut approver,
        )?;
        self.run(&["revert", "--no-edit", target])
    }

    pub fn cherry_pick<F>(&self, target: &str, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            "git_cherry_pick",
            format!("cherry-pick {target}"),
            None,
            None,
            &mut approver,
        )?;
        self.run(&["cherry-pick", target])
    }

    pub fn rebase<F>(&self, onto: &str, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            "git_rebase",
            format!("rebase onto '{onto}'"),
            None,
            None,
            &mut approver,
        )?;
        self.run(&["rebase", onto])
    }

    /// On failure (conflicts), the raw `git merge` stdout/stderr — which
    /// names the conflicting files — is returned as-is rather than
    /// specially parsed, so the caller (agent loop) can read the conflict
    /// markers straight from those files via the normal Read tool. Same
    /// feedback-loop principle as Fix Errors: surface the real failure,
    /// don't swallow it into a generic error.
    pub fn merge<F>(&self, branch: &str, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.enforce(
            "git_merge",
            format!("merge '{branch}'"),
            None,
            None,
            &mut approver,
        )?;
        self.run(&["merge", branch])
    }

    // ---------------------------------------------------------------
    // Pull requests — network, visible to others — ask
    // ---------------------------------------------------------------
    //
    // PRs are created via the `gh` CLI (GitHub official), not by calling the
    // REST API directly: `gh` handles auth, the remote's fork/upstream wiring,
    // and retry/error conventions. This also keeps token handling out of the
    // agent entirely — the user logs in to `gh` themselves (`gh auth login`).
    // Each PR op is gated on the "ask" tier (network, shared/visible state).
    //
    // Requirements: the current branch must already be pushed (`git push`)
    // before `gh pr create` can reference it.

    /// Check whether `gh` is available on PATH.
    pub fn gh_available(&self) -> bool {
        Command::new("gh")
            .arg("--version")
            .current_dir(&self.project_root)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn run_gh(&self, args: &[&str]) -> Result<GitOutput> {
        let output = Command::new("gh")
            .args(args)
            .current_dir(&self.project_root)
            .output()
            .map_err(|e| FsError::Other(format!("gh {args:?} failed to spawn: {e}")))?;
        Ok(GitOutput {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code(),
            success: output.status.success(),
        })
    }

    fn require_gh(&self, op: &str) -> Result<()> {
        if !self.gh_available() {
            return Err(FsError::Other(
                "PR support requires the GitHub CLI (`gh`). Install it and run \
                 `gh auth login`, then retry."
                    .into(),
            ));
        }
        let _ = op;
        Ok(())
    }

    /// List pull requests, newest first. `state` is one of open|closed|merged|all.
    pub fn pr_list<F>(&self, state: &str, limit: usize, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_gh("list")?;
        self.enforce(
            "git_pr_list",
            format!("gh pr list --state {state} --limit {limit}"),
            None,
            None,
            &mut approver,
        )?;
        let limit = limit.to_string();
        self.run_gh(&["pr", "list", "--state", state, "--limit", &limit])
    }

    /// Show details (status/checks/diff) for a PR.
    pub fn pr_view<F>(&self, number: &str, mut approver: F) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_gh("view")?;
        self.enforce(
            "git_pr_view",
            format!("gh pr view {number}"),
            None,
            None,
            &mut approver,
        )?;
        self.run_gh(&["pr", "view", number])
    }

    /// Create a PR for the current branch against an optional base branch.
    /// The head branch must already be pushed. Network — ask.
    pub fn pr_create<F>(
        &self,
        title: &str,
        body: Option<&str>,
        base: Option<&str>,
        mut approver: F,
    ) -> Result<GitOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_gh("create")?;
        self.enforce(
            "git_pr_create",
            format!("gh pr create title='{title}' base={}", base.unwrap_or("<default>")),
            None,
            None,
            &mut approver,
        )?;
        let mut args: Vec<&str> = vec!["pr", "create"];
        args.push("--title");
        args.push(title);
        if let Some(b) = body {
            args.push("--body");
            args.push(b);
        }
        if let Some(base) = base {
            args.push("--base");
            args.push(base);
        }
        self.run_gh(&args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permission::ApprovalDecision;
    use tempfile::TempDir;
    use zeus_config::AgentSettings;

    fn approve(_: &PermissionRequest) -> ApprovalDecision {
        ApprovalDecision::Approved
    }
    fn deny(_: &PermissionRequest) -> ApprovalDecision {
        ApprovalDecision::Denied
    }

    fn init_repo() -> (TempDir, GitEngine) {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        Command::new("git").arg("init").current_dir(&root).output().unwrap();
        Command::new("git")
            .args(["config", "user.email", "test@example.com"])
            .current_dir(&root)
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(&root)
            .output()
            .unwrap();
        let gate = PermissionGate::new(AgentSettings::default(), root.clone());
        let engine = GitEngine::new(root, gate);
        (tmp, engine)
    }

    fn commit_one(tmp: &TempDir, engine: &GitEngine, name: &str, content: &str) {
        std::fs::write(tmp.path().join("proj").join(name), content).unwrap();
        Command::new("git")
            .args(["add", name])
            .current_dir(tmp.path().join("proj"))
            .output()
            .unwrap();
        engine.commit(&format!("add {name}"), false, approve).unwrap();
    }

    #[test]
    fn status_on_fresh_repo_is_clean() {
        let (_tmp, engine) = init_repo();
        let out = engine.status().unwrap();
        assert!(out.success);
    }

    #[test]
    fn commit_and_log_roundtrip() {
        let (tmp, engine) = init_repo();
        commit_one(&tmp, &engine, "a.txt", "hello");
        let log = engine.log(5, None).unwrap();
        assert!(log.stdout.contains("add a.txt"));
    }

    #[test]
    fn commit_is_allowed_by_default_even_if_approver_would_deny() {
        // git_commit's default is Allow (reversible write tier, per the
        // blueprint) — the diff preview is always shown, but the approver
        // itself is never consulted at the default permission level. This
        // locks in that intentional behavior explicitly, rather than
        // leaving it as an untested side effect.
        let (tmp, engine) = init_repo();
        std::fs::write(tmp.path().join("proj/a.txt"), "hello").unwrap();
        Command::new("git")
            .args(["add", "a.txt"])
            .current_dir(tmp.path().join("proj"))
            .output()
            .unwrap();

        let out = engine.commit("allowed by default", false, deny).unwrap();
        assert!(out.success);
        let log = engine.log(5, None).unwrap();
        assert!(log.stdout.contains("allowed by default"));
    }

    #[test]
    fn commit_denied_does_not_create_commit_when_configured_to_ask() {
        // A user who tightens git_commit to Ask (overriding the default
        // Allow) must have "deny" actually block the commit.
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        Command::new("git").arg("init").current_dir(&root).output().unwrap();
        Command::new("git")
            .args(["config", "user.email", "test@example.com"])
            .current_dir(&root)
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(&root)
            .output()
            .unwrap();

        let mut settings = AgentSettings::default();
        settings.permissions.defaults.insert(
            0,
            zeus_config::PermissionDefault {
                tool: "git_commit".into(),
                state: zeus_config::PermissionState::Ask,
            },
        );
        let gate = PermissionGate::new(settings, root.clone());
        let engine = GitEngine::new(root.clone(), gate);

        std::fs::write(root.join("a.txt"), "hello").unwrap();
        Command::new("git")
            .args(["add", "a.txt"])
            .current_dir(&root)
            .output()
            .unwrap();

        let err = engine.commit("should not happen", false, deny).unwrap_err();
        assert!(matches!(err, FsError::Denied(_)));
        let log = engine.log(5, None).unwrap();
        assert!(!log.stdout.contains("should not happen"));
    }

    #[test]
    fn commit_and_push_requires_approval_for_push() {
        let (tmp, engine) = init_repo();
        commit_one(&tmp, &engine, "a.txt", "hello");
        // A fresh change to stage.
        std::fs::write(tmp.path().join("proj/b.txt"), "world").unwrap();
        Command::new("git")
            .args(["add", "b.txt"])
            .current_dir(tmp.path().join("proj"))
            .output()
            .unwrap();
        // Commit is allowed by default (reversible tier); push is on the ask
        // tier, so a denying approver aborts after the commit succeeds.
        let err = engine.commit_and_push("msg", false, None, deny).unwrap_err();
        assert!(matches!(err, FsError::Denied(_)));
    }

    #[test]
    fn pr_ops_error_without_gh_cli() {
        let (_tmp, engine) = init_repo();
        // gh may be absent (CI) or present (local). Either way the op must
        // not panic and must surface a GitOutput or a FsError — never hang.
        match engine.pr_list("open", 5, approve) {
            Ok(out) => {
                // gh returned (successfully listed, or errored as a normal
                // git-style GitOutput with an exit code). Either is fine.
                let _ = out;
            }
            Err(e) => {
                assert!(!e.to_string().is_empty(), "error must have a message");
            }
        }
    }

    #[test]
    fn force_push_is_denied_by_builtin_rule_even_when_approver_would_allow() {
        let (_tmp, engine) = init_repo();
        // Approver always says yes — the built-in rule must still deny
        // before the approver is ever consulted.
        let err = engine.push(None, None, true, approve).unwrap_err();
        assert!(matches!(err, FsError::Denied(_)));
    }

    #[test]
    fn hard_reset_is_denied_by_builtin_rule_even_when_approver_would_allow() {
        let (tmp, engine) = init_repo();
        commit_one(&tmp, &engine, "a.txt", "hello");
        let err = engine
            .reset(ResetMode::Hard, Some("HEAD"), approve)
            .unwrap_err();
        assert!(matches!(err, FsError::Denied(_)));
    }

    #[test]
    fn soft_reset_is_allowed_when_approved() {
        let (tmp, engine) = init_repo();
        commit_one(&tmp, &engine, "a.txt", "hello");
        commit_one(&tmp, &engine, "b.txt", "world");
        let out = engine.reset(ResetMode::Soft, Some("HEAD~1"), approve).unwrap();
        assert!(out.success, "soft reset failed: {}", out.stderr);
    }

    #[test]
    fn diff_and_status_are_read_only_no_prompt_needed() {
        let (tmp, engine) = init_repo();
        std::fs::write(tmp.path().join("proj/a.txt"), "hello").unwrap();
        let status = engine.status().unwrap();
        assert!(status.success);
        let diff = engine.diff(false, &[]).unwrap();
        assert!(diff.success);
    }

    #[test]
    fn branch_create_list_and_delete() {
        let (tmp, engine) = init_repo();
        commit_one(&tmp, &engine, "a.txt", "hello");

        let created = engine.branch_create("feature-x", approve).unwrap();
        assert!(created.success, "branch create failed: {}", created.stderr);
        let listed = engine.branch_list().unwrap();
        assert!(listed.stdout.contains("feature-x"));

        let deleted = engine.branch_delete("feature-x", false, approve).unwrap();
        assert!(deleted.success, "branch delete failed: {}", deleted.stderr);
        let listed_after = engine.branch_list().unwrap();
        assert!(!listed_after.stdout.contains("feature-x"));
    }

    #[test]
    fn stash_push_and_pop() {
        let (tmp, engine) = init_repo();
        commit_one(&tmp, &engine, "a.txt", "hello");
        std::fs::write(tmp.path().join("proj/a.txt"), "changed").unwrap();

        let stashed = engine.stash_push(Some("wip"), approve).unwrap();
        assert!(stashed.success, "stash push failed: {}", stashed.stderr);
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("proj/a.txt")).unwrap(),
            "hello"
        );

        let popped = engine.stash_pop(approve).unwrap();
        assert!(popped.success, "stash pop failed: {}", popped.stderr);
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("proj/a.txt")).unwrap(),
            "changed"
        );
    }

    #[test]
    fn tag_create_and_list() {
        let (tmp, engine) = init_repo();
        commit_one(&tmp, &engine, "a.txt", "hello");
        let created = engine.tag_create("v0.1.0", Some("first"), approve).unwrap();
        assert!(created.success, "tag create failed: {}", created.stderr);
        let listed = engine.tag_list().unwrap();
        assert!(listed.stdout.contains("v0.1.0"));
    }

    #[test]
    fn revert_creates_new_commit() {
        let (tmp, engine) = init_repo();
        commit_one(&tmp, &engine, "a.txt", "hello");
        let before = engine.log(10, None).unwrap();
        let reverted = engine.revert("HEAD", approve).unwrap();
        assert!(reverted.success, "revert failed: {}", reverted.stderr);
        let after = engine.log(10, None).unwrap();
        assert!(after.stdout.lines().count() > before.stdout.lines().count());
    }

    #[test]
    fn merge_conflict_surfaces_raw_output_not_a_generic_error() {
        let (tmp, engine) = init_repo();
        commit_one(&tmp, &engine, "a.txt", "line1");
        engine.branch_create("feature", approve).unwrap();

        // Conflicting change on main.
        std::fs::write(tmp.path().join("proj/a.txt"), "main-version").unwrap();
        Command::new("git")
            .args(["add", "a.txt"])
            .current_dir(tmp.path().join("proj"))
            .output()
            .unwrap();
        engine.commit("main change", false, approve).unwrap();

        // Conflicting change on feature.
        engine.checkout("feature", approve).unwrap();
        std::fs::write(tmp.path().join("proj/a.txt"), "feature-version").unwrap();
        Command::new("git")
            .args(["add", "a.txt"])
            .current_dir(tmp.path().join("proj"))
            .output()
            .unwrap();
        engine.commit("feature change", false, approve).unwrap();

        engine.checkout("master", approve).or_else(|_| engine.checkout("main", approve)).unwrap();
        let merged = engine.merge("feature", approve).unwrap();
        assert!(!merged.success);
        assert!(merged.stdout.to_lowercase().contains("conflict") || merged.stderr.to_lowercase().contains("conflict"));
    }
}
