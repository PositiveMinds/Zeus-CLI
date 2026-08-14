//! Git tool implementations for the agent tool layer.

use super::*;

impl ToolManager {
    // --- Git ---

    pub(super) fn do_git_diff(&self, args: &Value) -> Result<ToolResult> {
        let staged = args
            .get("staged")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let refs: Vec<String> = args
            .get("refs")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let refs_ref: Vec<&str> = refs.iter().map(|s| s.as_str()).collect();
        git_result(self.git.diff(staged, &refs_ref))
    }

    pub(super) fn do_git_blame(&self, args: &Value) -> Result<ToolResult> {
        let path = Self::str_arg(args, "path")?;
        git_result(self.git.blame(path))
    }

    pub(super) fn do_git_log(&self, args: &Value) -> Result<ToolResult> {
        let max = args.get("max").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        let path = args.get("path").and_then(|v| v.as_str());
        git_result(self.git.log(max, path))
    }

    pub(super) fn do_git_show(&self, args: &Value) -> Result<ToolResult> {
        let target = Self::str_arg(args, "target")?;
        git_result(self.git.show(target))
    }

    pub(super) fn do_git_add<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let paths: Vec<String> = args
            .get("paths")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if paths.is_empty() {
            return Err(AgentError::InvalidArguments {
                tool: "paths".into(),
                reason: "missing/empty 'paths'".into(),
            });
        }
        let paths_ref: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        git_result(self.git.add(&paths_ref, &mut *approver))
    }

    pub(super) fn do_git_commit<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let message = Self::str_arg(args, "message")?;
        let all = args.get("all").and_then(|v| v.as_bool()).unwrap_or(false);
        git_result(self.git.commit(message, all, &mut *approver))
    }

    pub(super) fn do_git_stash_push<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let message = args.get("message").and_then(|v| v.as_str());
        git_result(self.git.stash_push(message, &mut *approver))
    }

    pub(super) fn do_git_branch_create<F>(
        &self,
        args: &Value,
        approver: &mut F,
    ) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let name = Self::str_arg(args, "name")?;
        git_result(self.git.branch_create(name, &mut *approver))
    }

    pub(super) fn do_git_branch_delete<F>(
        &self,
        args: &Value,
        approver: &mut F,
    ) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let name = Self::str_arg(args, "name")?;
        let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
        git_result(self.git.branch_delete(name, force, &mut *approver))
    }

    pub(super) fn do_git_tag_create<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let name = Self::str_arg(args, "name")?;
        let message = args.get("message").and_then(|v| v.as_str());
        git_result(self.git.tag_create(name, message, &mut *approver))
    }

    pub(super) fn do_git_checkout<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let target = Self::str_arg(args, "target")?;
        git_result(self.git.checkout(target, &mut *approver))
    }

    pub(super) fn do_git_fetch<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let remote = args.get("remote").and_then(|v| v.as_str());
        git_result(self.git.fetch(remote, &mut *approver))
    }

    pub(super) fn do_git_push<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let remote = args.get("remote").and_then(|v| v.as_str());
        let branch = args.get("branch").and_then(|v| v.as_str());
        let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
        git_result(self.git.push(remote, branch, force, &mut *approver))
    }

    pub(super) fn do_git_reset<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let mode_str = Self::str_arg(args, "mode")?;
        let mode = match mode_str {
            "soft" => ResetMode::Soft,
            "mixed" => ResetMode::Mixed,
            "hard" => ResetMode::Hard,
            other => {
                return Err(AgentError::InvalidArguments {
                    tool: "mode".into(),
                    reason: format!("must be soft/mixed/hard, got '{other}'"),
                })
            }
        };
        let target = args.get("target").and_then(|v| v.as_str());
        git_result(self.git.reset(mode, target, &mut *approver))
    }

    pub(super) fn do_git_revert<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let target = Self::str_arg(args, "target")?;
        git_result(self.git.revert(target, &mut *approver))
    }

    pub(super) fn do_git_cherry_pick<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let target = Self::str_arg(args, "target")?;
        git_result(self.git.cherry_pick(target, &mut *approver))
    }

    pub(super) fn do_git_rebase<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let onto = Self::str_arg(args, "onto")?;
        git_result(self.git.rebase(onto, &mut *approver))
    }

    pub(super) fn do_git_merge<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let branch = Self::str_arg(args, "branch")?;
        git_result(self.git.merge(branch, &mut *approver))
    }
}
