//! Tool registry: bridges Phase 2 file operations + search + Phase 3
//! terminal execution into named tools the agent loop dispatches by name
//! with JSON-object arguments — this is the bridge layer the blueprint's
//! Agent Loop calls "Tool Manager".

use crate::background::BackgroundTaskRegistry;
use crate::error::{AgentError, Result};
use crate::hooks::{HookRunner, PreToolUseOutcome};
use crate::mcp::McpClient;
use crate::plugin::LoadedPlugin;
use crate::terminal::{CommandProfile, Sandbox, TerminalOptions, TerminalRunner};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;
use zeus_fs::{
    ApprovalDecision, CopyOptions, EditOptions, GitEngine, GitOutput, PermissionGate,
    PermissionRequest, ReadOptions, ResetMode, SearchOptions, Workspace, WriteOptions,
};
use zeus_provider::ToolSpec;

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }
    pub fn err(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// Tool specs advertised to the model. Kept in sync with `ToolManager`'s
/// `dispatch_with_approver` match arms below — every name here must have a
/// handler, and vice versa.
pub fn builtin_tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "read".into(),
            description: "Read a project file (line-numbered output).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"}
                },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "write".into(),
            description: "Create or overwrite a project file. Must read an existing file first."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": { "path": {"type": "string"}, "content": {"type": "string"} },
                "required": ["path", "content"]
            }),
        },
        ToolSpec {
            name: "edit".into(),
            description: "Targeted string replace in a file (must be read first).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"}
                },
                "required": ["path", "old_string", "new_string"]
            }),
        },
        ToolSpec {
            name: "delete".into(),
            description: "Delete a file or directory. Always requires user approval.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "path": {"type": "string"} },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "rename".into(),
            description: "Rename or move a file/directory.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "from": {"type": "string"}, "to": {"type": "string"} },
                "required": ["from", "to"]
            }),
        },
        ToolSpec {
            name: "copy".into(),
            description: "Copy a file.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "overwrite": {"type": "boolean"}
                },
                "required": ["from", "to"]
            }),
        },
        ToolSpec {
            name: "grep".into(),
            description: "Search file contents by regex.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "glob": {"type": "string"},
                    "ignore_case": {"type": "boolean"},
                    "max": {"type": "integer"}
                },
                "required": ["pattern"]
            }),
        },
        ToolSpec {
            name: "glob".into(),
            description: "Find files by glob pattern.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "pattern": {"type": "string"}, "max": {"type": "integer"} },
                "required": ["pattern"]
            }),
        },
        ToolSpec {
            name: "bash".into(),
            description: "Run a shell command in the project root (foreground, bounded) and wait for it to finish. Use for builds/tests/read-only commands. For a command that doesn't exit on its own (a dev server, `docker compose up`), set background=true instead of using this in foreground mode.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_secs": {"type": "integer"},
                    "background": {"type": "boolean", "description": "Run detached and return immediately with a task id, instead of waiting for exit."}
                },
                "required": ["command"]
            }),
        },
        ToolSpec {
            name: "bg_list".into(),
            description: "List background tasks started with bash(background=true) in this project, with their running/exited status.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "bg_output".into(),
            description: "Read the captured stdout/stderr so far for a background task by id.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "id": {"type": "integer"} },
                "required": ["id"]
            }),
        },
        ToolSpec {
            name: "bg_stop".into(),
            description: "Stop a running background task by id.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "id": {"type": "integer"} },
                "required": ["id"]
            }),
        },
        // --- Git: read-only ---
        ToolSpec {
            name: "git_status".into(),
            description: "git status (porcelain), with branch info.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "git_diff".into(),
            description: "git diff. staged=true for the index; refs=[\"a\"] diffs against a commit, refs=[\"a\",\"b\"] diffs a..b.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "staged": {"type": "boolean"},
                    "refs": {"type": "array", "items": {"type": "string"}}
                }
            }),
        },
        ToolSpec {
            name: "git_blame".into(),
            description: "git blame for a single file.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "path": {"type": "string"} },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "git_log".into(),
            description: "git log --oneline, optionally scoped to one path.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "max": {"type": "integer"}, "path": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "git_show".into(),
            description: "git show <commit-or-ref> — full diff/details for one commit.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "target": {"type": "string"} },
                "required": ["target"]
            }),
        },
        ToolSpec {
            name: "git_branch_list".into(),
            description: "List local and remote branches.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "git_remote_list".into(),
            description: "List configured remotes (git remote -v).".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "git_tag_list".into(),
            description: "List tags.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "git_stash_list".into(),
            description: "List stash entries.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        // --- Git: reversible write ---
        ToolSpec {
            name: "git_add".into(),
            description: "Stage one or more paths.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "paths": {"type": "array", "items": {"type": "string"}} },
                "required": ["paths"]
            }),
        },
        ToolSpec {
            name: "git_commit".into(),
            description: "Commit staged changes (or all tracked changes if all=true) with the given message. Read the diff first (git_diff) so the message actually reflects what changed.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "message": {"type": "string"}, "all": {"type": "boolean"} },
                "required": ["message"]
            }),
        },
        ToolSpec {
            name: "git_stash_push".into(),
            description: "Stash the working tree changes.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "message": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "git_stash_pop".into(),
            description: "Apply and drop the most recent stash entry.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "git_branch_create".into(),
            description: "Create a new branch at HEAD.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "name": {"type": "string"} },
                "required": ["name"]
            }),
        },
        ToolSpec {
            name: "git_branch_delete".into(),
            description: "Delete a branch. force=true uses -D (needed for an unmerged branch) instead of -d.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "name": {"type": "string"}, "force": {"type": "boolean"} },
                "required": ["name"]
            }),
        },
        ToolSpec {
            name: "git_tag_create".into(),
            description: "Create a tag, annotated if a message is given.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "name": {"type": "string"}, "message": {"type": "string"} },
                "required": ["name"]
            }),
        },
        // --- Git: working-tree-changing ---
        ToolSpec {
            name: "git_checkout".into(),
            description: "Check out an existing branch or commit.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "target": {"type": "string"} },
                "required": ["target"]
            }),
        },
        // --- Git: network / shared-state ---
        ToolSpec {
            name: "git_fetch".into(),
            description: "Fetch from a remote (or the default remote) without merging.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "remote": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "git_pull".into(),
            description: "git pull (fetch + merge/rebase per repo config).".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "git_push".into(),
            description: "git push. force=true is denied by a built-in safety rule regardless of approval — force-pushing needs an explicit, narrower rule in project settings.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "remote": {"type": "string"},
                    "branch": {"type": "string"},
                    "force": {"type": "boolean"}
                }
            }),
        },
        // --- Git: history-rewriting / conflict-prone ---
        ToolSpec {
            name: "git_reset".into(),
            description: "git reset. mode=\"hard\" is denied by a built-in safety rule regardless of approval (it discards working-tree changes) — use \"soft\" or \"mixed\" instead.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["soft", "mixed", "hard"]},
                    "target": {"type": "string"}
                },
                "required": ["mode"]
            }),
        },
        ToolSpec {
            name: "git_revert".into(),
            description: "Create a new commit that undoes the given commit (safer than reset — doesn't rewrite history).".into(),
            parameters: json!({
                "type": "object",
                "properties": { "target": {"type": "string"} },
                "required": ["target"]
            }),
        },
        ToolSpec {
            name: "git_cherry_pick".into(),
            description: "Apply the changes from one commit onto the current branch.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "target": {"type": "string"} },
                "required": ["target"]
            }),
        },
        ToolSpec {
            name: "git_rebase".into(),
            description: "Rebase the current branch onto another (rewrites history — use with care).".into(),
            parameters: json!({
                "type": "object",
                "properties": { "onto": {"type": "string"} },
                "required": ["onto"]
            }),
        },
        ToolSpec {
            name: "git_merge".into(),
            description: "Merge a branch into the current one. On conflict, the raw git output (naming the conflicting files) is returned — read those files to see the conflict markers.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "branch": {"type": "string"} },
                "required": ["branch"]
            }),
        },
    ]
}

/// Dispatches named tool calls against a workspace + terminal runner.
pub struct ToolManager {
    workspace: Workspace,
    terminal: TerminalRunner,
    background: BackgroundTaskRegistry,
    hooks: HookRunner,
    mcp_clients: Vec<McpClient>,
    plugins: Vec<LoadedPlugin>,
    git: GitEngine,
    cancel: Arc<AtomicBool>,
    /// Plan mode: read-only research/proposal, no mutating tool calls. Set
    /// via `set_plan_mode`; enforced centrally in `dispatch_with_approver`
    /// rather than per-tool, so it can't be bypassed by a tool that happens
    /// to be configured Allow in the permission settings.
    plan_mode: AtomicBool,
}

/// Tools that only observe state (files, git history, background task
/// status) — safe to run in Plan mode. Everything else (writes, git
/// mutations, `bash`, MCP/plugin calls, whose side effects zeus can't
/// characterize generically) is blocked while Plan mode is active.
fn is_read_only_tool(name: &str) -> bool {
    matches!(
        name,
        "read"
            | "grep"
            | "glob"
            | "bg_list"
            | "bg_output"
            | "git_status"
            | "git_diff"
            | "git_blame"
            | "git_log"
            | "git_show"
            | "git_branch_list"
            | "git_remote_list"
            | "git_tag_list"
            | "git_stash_list"
    )
}

/// MCP tool names exposed to the model are prefixed so they can't collide
/// with built-ins or across servers, and so dispatch can route back to the
/// right client.
fn mcp_tool_name(server: &str, tool: &str) -> String {
    format!("mcp__{server}__{tool}")
}

/// Same collision-avoidance/routing rationale as `mcp_tool_name`, for
/// native plugins.
fn plugin_tool_name(plugin: &str, tool: &str) -> String {
    format!("plugin__{plugin}__{tool}")
}

impl ToolManager {
    pub fn new(
        workspace: Workspace,
        terminal: TerminalRunner,
        background: BackgroundTaskRegistry,
        hooks: HookRunner,
        mcp_clients: Vec<McpClient>,
        plugins: Vec<LoadedPlugin>,
        cancel: Arc<AtomicBool>,
    ) -> Self {
        // Its own PermissionGate instance, same pattern `Workspace` already
        // uses internally for `files` vs. `search` (separate gates built
        // from the same settings + root, not a shared/cloned one).
        let git = GitEngine::new(
            workspace.project_root.clone(),
            PermissionGate::new(workspace.settings.clone(), workspace.project_root.clone()),
        );
        Self {
            workspace,
            terminal,
            background,
            hooks,
            mcp_clients,
            plugins,
            git,
            cancel,
            plan_mode: AtomicBool::new(false),
        }
    }

    pub fn set_plan_mode(&self, enabled: bool) {
        self.plan_mode.store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn plan_mode(&self) -> bool {
        self.plan_mode.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    pub fn background(&self) -> &BackgroundTaskRegistry {
        &self.background
    }

    pub fn hooks(&self) -> &HookRunner {
        &self.hooks
    }

    /// Built-in tool specs plus one per tool exposed by each connected MCP
    /// server (name-prefixed `mcp__<server>__<tool>`) and each loaded native
    /// plugin (`plugin__<plugin>__<tool>`). This is what the agent loop
    /// should advertise to the model — not `builtin_tool_specs()` alone.
    pub fn all_tool_specs(&self) -> Vec<ToolSpec> {
        let mut specs = builtin_tool_specs();
        for client in &self.mcp_clients {
            for tool in client.tools() {
                specs.push(ToolSpec {
                    name: mcp_tool_name(client.name(), &tool.name),
                    description: format!("[{}] {}", client.name(), tool.description),
                    parameters: tool.input_schema.clone(),
                });
            }
        }
        for plugin in &self.plugins {
            for tool in plugin.tools() {
                specs.push(ToolSpec {
                    name: plugin_tool_name(plugin.name(), &tool.name),
                    description: format!("[{}] {}", plugin.name(), tool.description),
                    parameters: tool.parameters.clone(),
                });
            }
        }
        specs
    }

    /// Execute a named tool call with JSON-object arguments. Permission
    /// "ask" prompts are routed through `approver`. Wrapped by the
    /// `pre-tool-use` hook (can block or rewrite the arguments) and the
    /// `post-tool-use` hook (its output, if any, is appended to the result
    /// so the model actually sees it — see the Hooks design note on
    /// diagnostics/test hooks).
    pub fn dispatch_with_approver<F>(
        &self,
        name: &str,
        arguments: &str,
        mut approver: F,
    ) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        if self.plan_mode() && !is_read_only_tool(name) {
            return Ok(ToolResult::err(format!(
                "blocked: Plan mode is active (read-only) — '{name}' would change something. \
                 Press Tab to switch to Build mode to make changes."
            )));
        }

        let arguments = match self.hooks.run_pre_tool_use(name, arguments) {
            PreToolUseOutcome::Block { reason } => {
                return Ok(ToolResult::err(format!(
                    "blocked by pre-tool-use hook: {reason}"
                )));
            }
            PreToolUseOutcome::Allow {
                modified_arguments: Some(modified),
            } => modified,
            PreToolUseOutcome::Allow {
                modified_arguments: None,
            } => arguments.to_string(),
        };
        let arguments = arguments.as_str();

        let result = self.dispatch_inner(name, arguments, &mut approver)?;

        Ok(match self.hooks.run_post_tool_use(
            name,
            arguments,
            &result.content,
            result.is_error,
        ) {
            Some(extra) => ToolResult {
                content: format!("{}\n\n[post-tool-use hook output]\n{extra}", result.content),
                is_error: result.is_error,
            },
            None => result,
        })
    }

    fn dispatch_inner<F>(&self, name: &str, arguments: &str, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let args: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
        match name {
            "read" => self.do_read(&args),
            "write" => self.do_write(&args, approver),
            "edit" => self.do_edit(&args, approver),
            "delete" => self.do_delete(&args, approver),
            "rename" => self.do_rename(&args, approver),
            "copy" => self.do_copy(&args, approver),
            "grep" => self.do_grep(&args),
            "glob" => self.do_glob(&args),
            "bash" => self.do_bash(&args, approver),
            "bg_list" => self.do_bg_list(),
            "bg_output" => self.do_bg_output(&args),
            "bg_stop" => self.do_bg_stop(&args),
            "git_status" => git_result(self.git.status()),
            "git_diff" => self.do_git_diff(&args),
            "git_blame" => self.do_git_blame(&args),
            "git_log" => self.do_git_log(&args),
            "git_show" => self.do_git_show(&args),
            "git_branch_list" => git_result(self.git.branch_list()),
            "git_remote_list" => git_result(self.git.remote_list()),
            "git_tag_list" => git_result(self.git.tag_list()),
            "git_stash_list" => git_result(self.git.stash_list()),
            "git_add" => self.do_git_add(&args, approver),
            "git_commit" => self.do_git_commit(&args, approver),
            "git_stash_push" => self.do_git_stash_push(&args, approver),
            "git_stash_pop" => git_result(self.git.stash_pop(&mut *approver)),
            "git_branch_create" => self.do_git_branch_create(&args, approver),
            "git_branch_delete" => self.do_git_branch_delete(&args, approver),
            "git_tag_create" => self.do_git_tag_create(&args, approver),
            "git_checkout" => self.do_git_checkout(&args, approver),
            "git_fetch" => self.do_git_fetch(&args, approver),
            "git_pull" => git_result(self.git.pull(&mut *approver)),
            "git_push" => self.do_git_push(&args, approver),
            "git_reset" => self.do_git_reset(&args, approver),
            "git_revert" => self.do_git_revert(&args, approver),
            "git_cherry_pick" => self.do_git_cherry_pick(&args, approver),
            "git_rebase" => self.do_git_rebase(&args, approver),
            "git_merge" => self.do_git_merge(&args, approver),
            other => {
                if let Some(rest) = other.strip_prefix("mcp__") {
                    self.do_mcp_call(rest, args, approver)
                } else if let Some(rest) = other.strip_prefix("plugin__") {
                    self.do_plugin_call(rest, args, approver)
                } else {
                    Err(AgentError::UnknownTool(other.to_string()))
                }
            }
        }
    }

    /// Dispatch `plugin__tool` (the part after the `plugin__` prefix) to the
    /// matching loaded native plugin. Permission-gated like MCP calls — a
    /// native plugin call is at least as consequential as an external
    /// server call (more so: it runs in-process, see the trust-boundary
    /// warning in `plugin.rs`), so it gets no less scrutiny.
    fn do_plugin_call<F>(&self, plugin_and_tool: &str, args: Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let Some((plugin_name, tool)) = plugin_and_tool.split_once("__") else {
            return Err(AgentError::UnknownTool(format!("plugin__{plugin_and_tool}")));
        };
        let Some(plugin) = self.plugins.iter().find(|p| p.name() == plugin_name) else {
            return Ok(ToolResult::err(format!(
                "no loaded plugin named '{plugin_name}'"
            )));
        };

        if let Err(e) = self.workspace.files.gate.enforce(
            &PermissionRequest {
                tool: format!("plugin_{plugin_name}"),
                path: None,
                command: None,
                description: format!("call plugin tool '{tool}' on '{plugin_name}': {args}"),
                ..Default::default()
            },
            approver,
        ) {
            return Ok(ToolResult::err(e.to_string()));
        }

        match plugin.call_tool(tool, &args.to_string()) {
            Ok(result) => Ok(ToolResult {
                content: result.content,
                is_error: result.is_error,
            }),
            Err(e) => Ok(ToolResult::err(format!(
                "plugin '{plugin_name}' tool '{tool}' failed: {e}"
            ))),
        }
    }

    /// Dispatch `server__tool` (the part after the `mcp__` prefix) to the
    /// matching connected client. Permission-gated the same way as `bash` —
    /// external, server-defined actions get their own tool-name category, so
    /// they default to "ask" via the generic no-rule fallback (same as any
    /// tool with no tailored default) rather than silently inheriting
    /// whatever the built-in `bash`/`write`/etc. defaults happen to be.
    fn do_mcp_call<F>(&self, server_and_tool: &str, args: Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let Some((server, tool)) = server_and_tool.split_once("__") else {
            return Err(AgentError::UnknownTool(format!("mcp__{server_and_tool}")));
        };
        let Some(client) = self.mcp_clients.iter().find(|c| c.name() == server) else {
            return Ok(ToolResult::err(format!(
                "no connected MCP server named '{server}'"
            )));
        };

        if let Err(e) = self.workspace.files.gate.enforce(
            &PermissionRequest {
                tool: format!("mcp_{server}"),
                path: None,
                command: None,
                description: format!("call MCP tool '{tool}' on server '{server}': {args}"),
                ..Default::default()
            },
            approver,
        ) {
            return Ok(ToolResult::err(e.to_string()));
        }

        match client.call_tool(tool, args) {
            Ok(result) => Ok(ToolResult {
                content: result.as_text(),
                is_error: result.is_error,
            }),
            Err(e) => Ok(ToolResult::err(format!(
                "mcp '{server}' tool '{tool}' failed: {e}"
            ))),
        }
    }

    fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
        args.get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::InvalidArguments {
                tool: key.into(),
                reason: format!("missing/invalid '{key}'"),
            })
    }

    fn do_read(&self, args: &Value) -> Result<ToolResult> {
        let path = Self::str_arg(args, "path")?;
        let offset = args.get("offset").and_then(|v| v.as_u64()).map(|v| v as usize);
        let limit = args.get("limit").and_then(|v| v.as_u64()).map(|v| v as usize);
        match self
            .workspace
            .files
            .read(Path::new(path), ReadOptions { offset, limit })
        {
            Ok(r) => Ok(ToolResult::ok(r.content)),
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_write<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let path = Self::str_arg(args, "path")?;
        let content = Self::str_arg(args, "content")?;
        match self.workspace.files.write(
            Path::new(path),
            content,
            WriteOptions::default(),
            &mut *approver,
        ) {
            Ok(()) => Ok(ToolResult::ok(format!("wrote {path}"))),
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_edit<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let path = Self::str_arg(args, "path")?;
        let old_string = Self::str_arg(args, "old_string")?.to_string();
        let new_string = Self::str_arg(args, "new_string")?.to_string();
        let replace_all = args
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        match self.workspace.files.edit(
            Path::new(path),
            EditOptions {
                old_string,
                new_string,
                replace_all,
            },
            &mut *approver,
        ) {
            Ok(n) => Ok(ToolResult::ok(format!("edited {path} ({n} replacement(s))"))),
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_delete<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let path = Self::str_arg(args, "path")?;
        match self.workspace.files.delete(Path::new(path), &mut *approver) {
            Ok(()) => Ok(ToolResult::ok(format!("deleted {path}"))),
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_rename<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let from = Self::str_arg(args, "from")?;
        let to = Self::str_arg(args, "to")?;
        match self
            .workspace
            .files
            .rename(Path::new(from), Path::new(to), &mut *approver)
        {
            Ok(()) => Ok(ToolResult::ok(format!("moved {from} -> {to}"))),
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_copy<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let from = Self::str_arg(args, "from")?;
        let to = Self::str_arg(args, "to")?;
        let overwrite = args
            .get("overwrite")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        match self.workspace.files.copy(
            Path::new(from),
            Path::new(to),
            CopyOptions { overwrite },
            &mut *approver,
        ) {
            Ok(()) => Ok(ToolResult::ok(format!("copied {from} -> {to}"))),
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_grep(&self, args: &Value) -> Result<ToolResult> {
        let pattern = Self::str_arg(args, "pattern")?.to_string();
        let glob = args
            .get("glob")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let case_insensitive = args
            .get("ignore_case")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let max_matches = args.get("max").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
        match self.workspace.search.grep(SearchOptions {
            pattern,
            glob,
            case_insensitive,
            max_matches,
            path: None,
        }) {
            Ok(hits) => {
                let text = hits
                    .iter()
                    .map(|h| format!("{}:{}:{}", h.path.display(), h.line, h.text))
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(ToolResult::ok(if text.is_empty() {
                    "(no matches)".to_string()
                } else {
                    text
                }))
            }
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_glob(&self, args: &Value) -> Result<ToolResult> {
        let pattern = Self::str_arg(args, "pattern")?;
        let max = args.get("max").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
        match self.workspace.search.glob(pattern, max) {
            Ok(hits) => {
                let text = hits
                    .iter()
                    .map(|h| h.path.display().to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(ToolResult::ok(if text.is_empty() {
                    "(no files)".to_string()
                } else {
                    text
                }))
            }
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_bash<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let command = Self::str_arg(args, "command")?;
        let background = args
            .get("background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if background {
            self.workspace
                .files
                .gate
                .enforce(
                    &PermissionRequest {
                        tool: "bash".into(),
                        path: None,
                        command: Some(command.to_string()),
                        description: format!("run as background task: {command}"),
                        ..Default::default()
                    },
                    &mut *approver,
                )
                .map_err(AgentError::Fs)?;
            return match self.background.spawn(command, &self.workspace.project_root) {
                Ok(id) => Ok(ToolResult::ok(format!(
                    "started background task id={id}: {command}\nUse bg_output with id={id} to check its output, bg_stop with id={id} to stop it."
                ))),
                Err(e) => Ok(ToolResult::err(e.to_string())),
            };
        }

        let timeout = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .map(Duration::from_secs)
            .or(Some(Duration::from_secs(120)));
        let opts = TerminalOptions {
            cwd: self.workspace.project_root.clone(),
            timeout,
            sandbox: Sandbox::RestrictedFs,
            profile: CommandProfile::Foreground,
            // See TerminalOptions::new's doc comment — PTY exit-detection is
            // unreliable on this setup, so the model-facing tool stays on
            // the well-proven piped path until that's root-caused.
            use_pty: false,
        };
        match self.terminal.run(
            command,
            &self.workspace.files.gate,
            opts,
            self.cancel.clone(),
            &mut *approver,
        ) {
            Ok(out) => {
                let text = format!(
                    "exit={:?} cancelled={} timed_out={}\n--- stdout ---\n{}--- stderr ---\n{}{}",
                    out.exit_code,
                    out.cancelled,
                    out.timed_out,
                    out.stdout,
                    out.stderr,
                    if out.truncated {
                        "\n(output truncated)"
                    } else {
                        ""
                    }
                );
                if out.exit_code == Some(0) && !out.cancelled && !out.timed_out {
                    Ok(ToolResult::ok(text))
                } else {
                    Ok(ToolResult::err(text))
                }
            }
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn u64_arg(args: &Value, key: &str) -> Result<u64> {
        args.get(key)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AgentError::InvalidArguments {
                tool: key.into(),
                reason: format!("missing/invalid '{key}'"),
            })
    }

    fn do_bg_list(&self) -> Result<ToolResult> {
        match self.background.list() {
            Ok(tasks) if tasks.is_empty() => Ok(ToolResult::ok("(no background tasks)")),
            Ok(tasks) => {
                let text = tasks
                    .iter()
                    .map(|(t, s)| format!("id={} status={:?} command={}", t.id, s, t.command))
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(ToolResult::ok(text))
            }
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_bg_output(&self, args: &Value) -> Result<ToolResult> {
        let id = Self::u64_arg(args, "id")?;
        let (stdout, stderr) = self.background.output(id);
        Ok(ToolResult::ok(format!(
            "--- stdout ---\n{stdout}--- stderr ---\n{stderr}"
        )))
    }

    fn do_bg_stop(&self, args: &Value) -> Result<ToolResult> {
        let id = Self::u64_arg(args, "id")?;
        match self.background.stop(id) {
            Ok(()) => Ok(ToolResult::ok(format!("stopped background task {id}"))),
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    // --- Git ---

    fn do_git_diff(&self, args: &Value) -> Result<ToolResult> {
        let staged = args.get("staged").and_then(|v| v.as_bool()).unwrap_or(false);
        let refs: Vec<String> = args
            .get("refs")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let refs_ref: Vec<&str> = refs.iter().map(|s| s.as_str()).collect();
        git_result(self.git.diff(staged, &refs_ref))
    }

    fn do_git_blame(&self, args: &Value) -> Result<ToolResult> {
        let path = Self::str_arg(args, "path")?;
        git_result(self.git.blame(path))
    }

    fn do_git_log(&self, args: &Value) -> Result<ToolResult> {
        let max = args.get("max").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        let path = args.get("path").and_then(|v| v.as_str());
        git_result(self.git.log(max, path))
    }

    fn do_git_show(&self, args: &Value) -> Result<ToolResult> {
        let target = Self::str_arg(args, "target")?;
        git_result(self.git.show(target))
    }

    fn do_git_add<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let paths: Vec<String> = args
            .get("paths")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
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

    fn do_git_commit<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let message = Self::str_arg(args, "message")?;
        let all = args.get("all").and_then(|v| v.as_bool()).unwrap_or(false);
        git_result(self.git.commit(message, all, &mut *approver))
    }

    fn do_git_stash_push<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let message = args.get("message").and_then(|v| v.as_str());
        git_result(self.git.stash_push(message, &mut *approver))
    }

    fn do_git_branch_create<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let name = Self::str_arg(args, "name")?;
        git_result(self.git.branch_create(name, &mut *approver))
    }

    fn do_git_branch_delete<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let name = Self::str_arg(args, "name")?;
        let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
        git_result(self.git.branch_delete(name, force, &mut *approver))
    }

    fn do_git_tag_create<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let name = Self::str_arg(args, "name")?;
        let message = args.get("message").and_then(|v| v.as_str());
        git_result(self.git.tag_create(name, message, &mut *approver))
    }

    fn do_git_checkout<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let target = Self::str_arg(args, "target")?;
        git_result(self.git.checkout(target, &mut *approver))
    }

    fn do_git_fetch<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let remote = args.get("remote").and_then(|v| v.as_str());
        git_result(self.git.fetch(remote, &mut *approver))
    }

    fn do_git_push<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let remote = args.get("remote").and_then(|v| v.as_str());
        let branch = args.get("branch").and_then(|v| v.as_str());
        let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
        git_result(self.git.push(remote, branch, force, &mut *approver))
    }

    fn do_git_reset<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
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

    fn do_git_revert<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let target = Self::str_arg(args, "target")?;
        git_result(self.git.revert(target, &mut *approver))
    }

    fn do_git_cherry_pick<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let target = Self::str_arg(args, "target")?;
        git_result(self.git.cherry_pick(target, &mut *approver))
    }

    fn do_git_rebase<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let onto = Self::str_arg(args, "onto")?;
        git_result(self.git.rebase(onto, &mut *approver))
    }

    fn do_git_merge<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let branch = Self::str_arg(args, "branch")?;
        git_result(self.git.merge(branch, &mut *approver))
    }
}

/// Render a `GitOutput` (or the permission/spawn error that prevented one)
/// as a `ToolResult` — a non-zero exit is a soft error visible to the model
/// (so it can read `git`'s own message and react), not a hard `Err` that
/// would abort the tool-call cycle. Matches the same convention already
/// used for `bash` and every other tool here.
fn git_result(result: zeus_fs::Result<GitOutput>) -> Result<ToolResult> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use zeus_config::{AgentSettings, Config, GlobalPaths, ProvidersFile};

    fn approve(_: &PermissionRequest) -> ApprovalDecision {
        ApprovalDecision::Approved
    }

    fn tool_manager(root: &Path) -> ToolManager {
        std::fs::create_dir_all(root).unwrap();
        let config = Config {
            global: GlobalPaths::from_root(root.join(".zeus-home")),
            project: None,
            settings: AgentSettings::default(),
            providers: ProvidersFile::default(),
            project_root: Some(root.to_path_buf()),
        };
        let workspace = Workspace::from_config(&config).unwrap();
        let terminal = TerminalRunner::new(root.join(".agent/checkpoints"));
        let background = BackgroundTaskRegistry::new(root.join(".agent/background"));
        let hooks = crate::hooks::HookRunner::new(root.join(".agent/hooks"), root.to_path_buf());
        ToolManager::new(
            workspace,
            terminal,
            background,
            hooks,
            Vec::new(),
            Vec::new(),
            Arc::new(AtomicBool::new(false)),
        )
    }

    fn tool_manager_with_mcp(root: &Path, mcp_clients: Vec<crate::mcp::McpClient>) -> ToolManager {
        std::fs::create_dir_all(root).unwrap();
        let config = Config {
            global: GlobalPaths::from_root(root.join(".zeus-home")),
            project: None,
            settings: AgentSettings::default(),
            providers: ProvidersFile::default(),
            project_root: Some(root.to_path_buf()),
        };
        let workspace = Workspace::from_config(&config).unwrap();
        let terminal = TerminalRunner::new(root.join(".agent/checkpoints"));
        let background = BackgroundTaskRegistry::new(root.join(".agent/background"));
        let hooks = crate::hooks::HookRunner::new(root.join(".agent/hooks"), root.to_path_buf());
        ToolManager::new(
            workspace,
            terminal,
            background,
            hooks,
            mcp_clients,
            Vec::new(),
            Arc::new(AtomicBool::new(false)),
        )
    }

    #[test]
    fn write_then_read_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let tm = tool_manager(&root);
        let r = tm
            .dispatch_with_approver("write", r#"{"path":"a.txt","content":"hello"}"#, approve)
            .unwrap();
        assert!(!r.is_error);
        let r = tm
            .dispatch_with_approver("read", r#"{"path":"a.txt"}"#, approve)
            .unwrap();
        assert!(r.content.contains("hello"));
    }

    #[test]
    fn plan_mode_blocks_mutating_tools_but_allows_read_only() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let tm = tool_manager(&root);
        tm.dispatch_with_approver("write", r#"{"path":"a.txt","content":"hello"}"#, approve)
            .unwrap();

        tm.set_plan_mode(true);
        assert!(tm.plan_mode());

        let blocked = tm
            .dispatch_with_approver(
                "write",
                r#"{"path":"a.txt","content":"changed"}"#,
                approve,
            )
            .unwrap();
        assert!(blocked.is_error);
        assert!(blocked.content.contains("Plan mode"));
        // The blocked call must not have actually touched the file.
        assert_eq!(std::fs::read_to_string(root.join("a.txt")).unwrap(), "hello");

        let read = tm.dispatch_with_approver("read", r#"{"path":"a.txt"}"#, approve).unwrap();
        assert!(!read.is_error);
        assert!(read.content.contains("hello"));

        tm.set_plan_mode(false);
        let write_again = tm
            .dispatch_with_approver(
                "write",
                r#"{"path":"a.txt","content":"changed"}"#,
                approve,
            )
            .unwrap();
        assert!(!write_again.is_error);
    }

    #[test]
    fn unknown_tool_errors() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let tm = tool_manager(&root);
        let err = tm
            .dispatch_with_approver("frobnicate", "{}", approve)
            .unwrap_err();
        assert!(matches!(err, AgentError::UnknownTool(_)));
    }

    #[test]
    fn bash_runs_and_denies_destructive() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let tm = tool_manager(&root);
        let r = tm
            .dispatch_with_approver("bash", r#"{"command":"echo hi"}"#, approve)
            .unwrap();
        assert!(!r.is_error);
        assert!(r.content.contains("hi"));

        let r2 = tm
            .dispatch_with_approver("bash", r#"{"command":"rm -rf /"}"#, approve)
            .unwrap();
        assert!(r2.is_error);
    }

    #[test]
    fn bash_background_spawns_and_is_listed_and_stoppable() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let tm = tool_manager(&root);
        let sleep_cmd = if cfg!(windows) {
            "ping -n 30 127.0.0.1 >NUL"
        } else {
            "sleep 30"
        };

        let started = tm
            .dispatch_with_approver(
                "bash",
                &format!(r#"{{"command":"{sleep_cmd}","background":true}}"#),
                approve,
            )
            .unwrap();
        assert!(!started.is_error);
        assert!(started.content.contains("started background task"));

        let listed = tm.dispatch_with_approver("bg_list", "{}", approve).unwrap();
        assert!(listed.content.contains("status=Running"));

        // Extract the id we were given and stop it via the tool, not the registry directly.
        let id = tm.background().list().unwrap()[0].0.id;
        let stopped = tm
            .dispatch_with_approver("bg_stop", &format!(r#"{{"id":{id}}}"#), approve)
            .unwrap();
        assert!(!stopped.is_error);
        assert!(tm.background().get(id).unwrap().is_none());
    }

    #[test]
    fn mcp_tool_is_advertised_and_dispatchable_end_to_end() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let script = crate::mcp::write_test_server(&root);
        let client = crate::mcp::McpClient::connect(
            "testsrv",
            crate::mcp::python_cmd(),
            &[script.display().to_string()],
            &root,
        )
        .unwrap();
        let tm = tool_manager_with_mcp(&root, vec![client]);

        // Advertised to the model with the server-prefixed name.
        let specs = tm.all_tool_specs();
        assert!(specs.iter().any(|s| s.name == "mcp__testsrv__echo"));

        // Dispatchable through the exact same path a real tool call takes.
        let ok = tm
            .dispatch_with_approver("mcp__testsrv__echo", r#"{"text":"hi"}"#, approve)
            .unwrap();
        assert!(!ok.is_error);
        assert_eq!(ok.content, "echo: hi");

        let failed = tm
            .dispatch_with_approver("mcp__testsrv__echo", r#"{"fail":true}"#, approve)
            .unwrap();
        assert!(failed.is_error);
        assert_eq!(failed.content, "deliberate failure");
    }

    #[test]
    fn mcp_call_denied_is_not_run() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let script = crate::mcp::write_test_server(&root);
        let client = crate::mcp::McpClient::connect(
            "testsrv",
            crate::mcp::python_cmd(),
            &[script.display().to_string()],
            &root,
        )
        .unwrap();
        let tm = tool_manager_with_mcp(&root, vec![client]);

        let denied = tm
            .dispatch_with_approver("mcp__testsrv__echo", r#"{"text":"hi"}"#, |_| {
                ApprovalDecision::Denied
            })
            .unwrap();
        assert!(denied.is_error);
        assert!(denied.content.contains("denied"));
    }

    #[test]
    fn every_tool_spec_has_a_handler() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let tm = tool_manager(&root);
        for spec in builtin_tool_specs() {
            let err = tm.dispatch_with_approver(&spec.name, "{}", approve);
            // Missing required args should surface as InvalidArguments, not
            // UnknownTool — proves the name is wired to a real handler.
            match err {
                Err(AgentError::UnknownTool(_)) => {
                    panic!("tool spec '{}' has no handler", spec.name)
                }
                _ => {}
            }
        }
    }

    #[test]
    fn git_tools_work_end_to_end_through_the_full_dispatch_path() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        std::process::Command::new("git").arg("init").current_dir(&root).output().unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@example.com"])
            .current_dir(&root)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(&root)
            .output()
            .unwrap();

        let tm = tool_manager(&root);

        // Real file, staged and committed through the tool dispatch layer —
        // not calling GitEngine directly — proving hooks/permission
        // wrapping and JSON argument parsing all work together, not just
        // the underlying engine in isolation.
        std::fs::write(root.join("a.txt"), "hello").unwrap();
        let add = tm
            .dispatch_with_approver("git_add", r#"{"paths":["a.txt"]}"#, approve)
            .unwrap();
        assert!(!add.is_error, "git_add failed: {}", add.content);

        let commit = tm
            .dispatch_with_approver("git_commit", r#"{"message":"initial commit"}"#, approve)
            .unwrap();
        assert!(!commit.is_error, "git_commit failed: {}", commit.content);

        let log = tm.dispatch_with_approver("git_log", "{}", approve).unwrap();
        assert!(!log.is_error);
        assert!(log.content.contains("initial commit"));

        let status = tm.dispatch_with_approver("git_status", "{}", approve).unwrap();
        assert!(!status.is_error);

        // Force-push must be denied even though the approver would allow —
        // proves the built-in rule reaches all the way through the tool
        // dispatch layer, not just the GitEngine unit tests.
        let force_push = tm
            .dispatch_with_approver("git_push", r#"{"force":true}"#, approve)
            .unwrap();
        assert!(force_push.is_error);

        // Hard reset likewise denied end to end.
        let hard_reset = tm
            .dispatch_with_approver("git_reset", r#"{"mode":"hard"}"#, approve)
            .unwrap();
        assert!(hard_reset.is_error);
    }
}
