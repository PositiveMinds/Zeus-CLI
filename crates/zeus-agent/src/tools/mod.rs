//! Tool registry: bridges file operations, search, terminal execution, and
//! external integrations into named tools the agent loop dispatches by name.
//!
//! This module is the bridge layer the blueprint's Agent Loop calls "Tool Manager".
//! It provides a unified interface for all tool operations the agent can perform.
//!
//! ## Tool Categories
//!
//! ### Core Tools (always available)
//! - **File operations**: `read`, `write`, `edit`, `delete`, `rename`, `copy`, `mkdir`, `listdir`
//! - **Search**: `grep`, `glob`, `code_symbols`, `code_defs`, `code_refs`, `code_graph`, `code_rename`
//! - **Terminal**: `bash`, `test`, `verify`
//! - **Web**: `web_fetch`, `web_search`, `browser`
//! - **Git**: `git_*` tools for status, diff, commit, branch, etc.
//! - **Background**: `bg_list`, `bg_output`, `bg_stop`, `bg_pause`, `bg_resume`
//! - **RAG**: `rag_search`, `rag_index` for retrieval-augmented generation
//! - **Skills**: `list_skills`, `read_skill`
//! - **Memory**: `memory`, `memory_write`
//! - **Other**: `todowrite`, `current_time`, `understand_repo`
//!
//! ### Platform Tools (conditional on CLI presence)
//! - GitHub (`gh_*`): issues, PRs, releases, workflows
//! - Supabase (`supabase_*`): projects, database, functions
//! - Vercel (`vercel_*`): projects, deploy, logs
//! - Docker (`docker_*`): containers, compose
//! - Kubernetes (`k8s_*`): pods, deployments
//! - Terraform (`tf_*`): plan, apply
//! - AWS, Azure, GCP, Helm, Fly, Railway, Render, Netlify, Firebase
//!
//! ### MCP Tools (from connected servers)
//! - Dynamically discovered from MCP servers via `mcp__<server>__<tool>` naming
//!
//! ### Native Plugin Tools
//! - From loaded native plugins via `plugin__<plugin>__<tool>` naming
//!
//! ## Permission System
//!
//! Tools are classified by their mutation risk:
//! - **Read-only**: Safe to run in Plan mode (file reads, search, git status, etc.)
//! - **Mutating**: Requires approval (file writes, bash execution, git commits, etc.)
//!
//! The `PermissionGate` enforces these rules based on project settings.
//!
//! ## Tool Dispatch
//!
//! Tools are dispatched via `ToolManager::dispatch_with_approver`:
//! 1. Pre-tool-use hooks can block or rewrite arguments
//! 2. Permission gate checks if the tool is allowed
//! 3. Tool implementation executes
//! 4. Post-tool-use hooks can append diagnostic output
//!
//! ## Background Tasks
//!
//! Long-running commands (dev servers, builds) can run as background tasks:
//! - Started via `bash` with `background=true`
//! - Monitored via `bg_list` and `bg_output`
//! - Paused/resumed/stopped via dedicated tools
//!
//! ## Code Intelligence
//!
//! - `code_index`: Build symbol index from tree-sitter AST parsing
//! - `code_symbols`: Find symbols by name
//! - `code_defs`: Find definitions
//! - `code_refs`: Find references via ripgrep
//! - `code_graph`: Call graph analysis (who calls what)
//! - `code_rename`: Propose rename plan (read-only)

mod git;
mod helpers;
mod platform;
mod specs;
#[cfg(test)]
mod tests;

use crate::background::BackgroundTaskRegistry;
use crate::error::{AgentError, Result};
use crate::hooks::{HookRunner, PreToolUseOutcome};
use crate::mcp::McpClient;
use crate::plugin::LoadedPlugin;
use crate::terminal::{CommandProfile, Sandbox, TerminalOptions, TerminalRunner};
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use zeus_fs::{
    filter_out_own_index, word_boundary, ApprovalDecision, CallEdge, CopyOptions, DeviceEngine,
    EditOptions, GitEngine, IndexEngine, PermissionGate, PermissionRequest, PlatformEngine,
    PlatformOutput, ReadOptions, ResetMode, SearchOptions, SymbolIndex, Workspace, WriteOptions,
};
use zeus_provider::{ModelProvider, ToolSpec};

// Re-export from sub-modules
pub(crate) use helpers::{detect_test_command, git_result, strip_html_with_tables};
pub use specs::builtin_tool_specs;
#[allow(unused_imports)] // re-exported for tools::tests
pub(crate) use specs::platform_cli_for;
pub use specs::platform_tool_specs;
pub(crate) use specs::PLATFORM_TOOLS;
use specs::{core_tool_specs, detect_platform_clis, filter_platform_specs};

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
    /// Vision-capable model attachments produced/returned by this tool call
    /// (e.g. `read_image`). Plumbed into the conversation by the agent loop.
    pub images: Vec<zeus_provider::ImagePart>,
}

impl ToolResult {
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            images: Vec::new(),
        }
    }
    pub fn err(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            images: Vec::new(),
        }
    }
}

/// Tools that only observe state (files, git history, background task
/// status) — safe to run in Plan mode. Everything else (writes, git
/// mutations, `bash`, MCP/plugin calls, whose side effects zeus can't
/// characterize generically) is blocked while Plan mode is active.
/// `pub(crate)` so personas and orchestration can classify steps the same
/// way Plan mode does (single source of truth for "does this mutate?").
pub(crate) fn is_read_only_tool(name: &str) -> bool {
    matches!(
        name,
        "read"
            | "read_multiple"
            | "grep"
            | "glob"
            | "listdir"
            | "web_fetch"
            | "web_search"
            |             "list_skills"
            | "read_skill"
|             "read_document"
            | "read_image"
            | "understand_repo"
            | "rag_search"
            | "memory"
            | "code_symbols"
            | "code_defs"
            | "code_refs"
            | "code_graph"
            | "code_rename"
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
            // Pure bookkeeping, no filesystem/process side effects — safe
            // to let a read-only Plan-mode turn use for progress tracking
            // too, same as the reference product's own `todowrite` tool.
            | "todowrite"
            // Pure clock read, no side effects — useful in Plan mode and to
            // delegated specialists ("what's today's date").
            | "current_time"
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

/// Dispatches named tool calls against a workspace + terminal runner.
pub struct ToolManager {
    workspace: Workspace,
    terminal: TerminalRunner,
    background: BackgroundTaskRegistry,
    hooks: HookRunner,
    mcp_clients: Vec<McpClient>,
    plugins: Vec<LoadedPlugin>,
    git: GitEngine,
    platform: PlatformEngine,
    device: DeviceEngine,
    cancel: Arc<AtomicBool>,
    /// Global skills dir (`~/.zeus/skills`), injected by the CLI so the tools
    /// can discover skills at both project and user scope.
    global_skills_dir: Option<PathBuf>,
    /// Plan mode: read-only research/proposal, no mutating tool calls. Set
    /// via `set_plan_mode`; enforced centrally in `dispatch_with_approver`
    /// rather than per-tool, so it can't be bypassed by a tool that happens
    /// to be configured Allow in the permission settings.
    plan_mode: AtomicBool,
    /// Cached repository fingerprint (repository understanding), shared with
    /// the Agent so the `understand_repo` tool doesn't rescan the tree.
    repo: Option<crate::analyze::RepoFingerprint>,
    /// Optional embeddings provider for `rag_index --embed`. Best-effort:
    /// when absent or unreachable the index is simply built without vectors.
    embedder: Option<Arc<dyn ModelProvider>>,
    /// Embedding model name to pass to `embedder` (usually the chat model;
    /// a provider may map it to its embedding model).
    embed_model: Option<String>,
    /// Which platform CLIs are on PATH, detected once and reused across
    /// `all_tool_specs` calls (every model round trip requests the list).
    available_clis: OnceLock<HashSet<String>>,
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
        let platform = PlatformEngine::new(
            workspace.project_root.clone(),
            PermissionGate::new(workspace.settings.clone(), workspace.project_root.clone()),
        );
        let device = DeviceEngine::new(
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
            platform,
            device,
            cancel,
            global_skills_dir: None,
            plan_mode: AtomicBool::new(false),
            repo: None,
            embedder: None,
            embed_model: None,
            available_clis: OnceLock::new(),
        }
    }

    /// Configure optional best-effort embeddings for `rag_index --embed`.
    /// Called once at startup with the session's provider/model; absent by
    /// default, which keeps the index keyword-only.
    pub fn set_embedding(&mut self, provider: Arc<dyn ModelProvider>, model: String) {
        self.embedder = Some(provider);
        self.embed_model = Some(model);
    }

    /// Share the cached repository fingerprint with the tool layer (used by
    /// `understand_repo` so it doesn't rescan the tree).
    pub fn set_repo(&mut self, repo: Option<crate::analyze::RepoFingerprint>) {
        self.repo = repo;
    }

    pub fn project_root(&self) -> PathBuf {
        self.workspace.project_root.clone()
    }

    /// Point the tools at the global skills dir (`~/.zeus/skills`). Project
    /// skills are discovered under `<project>/.agent/skills` automatically.
    pub fn set_global_skills_dir(&mut self, dir: Option<PathBuf>) {
        self.global_skills_dir = dir;
    }

    pub fn set_plan_mode(&self, enabled: bool) {
        self.plan_mode
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
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
        let mut specs = core_tool_specs();
        specs.extend(self.available_platform_tool_specs());
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

    /// Platform-CLI tools whose CLI binary is actually on PATH. Detected once
    /// and cached (`all_tool_specs` is called on every model round trip; a
    /// PATH re-scan per call would reintroduce the per-iteration overhead
    /// this filtering exists to remove).
    fn available_platform_tool_specs(&self) -> Vec<ToolSpec> {
        let present = self.available_clis.get_or_init(detect_platform_clis);
        filter_platform_specs(platform_tool_specs(), present)
    }

    /// Read-only subset of `all_tool_specs` — used for a `delegate`d
    /// specialist consultation, which is deliberately restricted to
    /// inspecting the workspace and never mutating it regardless of what
    /// that persona's own tool allow-list would otherwise permit: the
    /// primary agent stays the only thing that writes/edits/runs, a
    /// delegated specialist only ever gives it an informed opinion to act on.
    pub fn read_only_tool_specs(&self) -> Vec<ToolSpec> {
        self.all_tool_specs()
            .into_iter()
            .filter(|s| is_read_only_tool(&s.name))
            .collect()
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

        let result = match self.dispatch_inner(name, arguments, &mut approver) {
            Ok(r) => r,
            // The model called a tool with bad/missing arguments, or a
            // name that doesn't exist — its own mistake, and a recoverable
            // one: report it back as a normal (failed) tool result so the
            // model sees exactly what went wrong and can retry with
            // corrected arguments in the same turn, rather than one bad
            // call via `?` killing the entire turn outright with no way
            // for the model to self-correct. Anything else (`Provider`/`Fs`/`Terminal`/`Session`/`Io`) is a real system failure a
            // retry can't fix, and still aborts the turn as before.
            Err(e @ (AgentError::InvalidArguments { .. } | AgentError::UnknownTool(_))) => {
                ToolResult::err(e.to_string())
            }
            Err(e) => return Err(e),
        };

        Ok(
            match self
                .hooks
                .run_post_tool_use(name, arguments, &result.content, result.is_error)
            {
                Some(extra) => ToolResult {
                    content: format!("{}\n\n[post-tool-use hook output]\n{extra}", result.content),
                    is_error: result.is_error,
                    images: result.images,
                },
                None => result,
            },
        )
    }

    fn dispatch_inner<F>(&self, name: &str, arguments: &str, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let args: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
        match name {
            "todowrite" => self.do_todowrite(&args),
            "current_time" => self.do_current_time(&args),
            "read" => self.do_read(&args),
            "read_multiple" => self.do_read_multiple(&args),
            "write" => self.do_write(&args, approver),
            "edit" => self.do_edit(&args, approver),
            "delete" => self.do_delete(&args, approver),
            "rename" => self.do_rename(&args, approver),
            "copy" => self.do_copy(&args, approver),
            "grep" => self.do_grep(&args),
            "glob" => self.do_glob(&args),
            "mkdir" => self.do_mkdir(&args, approver),
            "listdir" => self.do_listdir(&args),
            "code_index" => self.do_code_index(&args, approver),
            "code_symbols" => self.do_code_symbols(&args),
            "code_defs" => self.do_code_defs(&args),
            "code_refs" => self.do_code_refs(&args),
            "code_graph" => self.do_code_graph(&args),
            "code_rename" => self.do_code_rename(&args),
            "bash" => self.do_bash(&args, approver),
            "test" => self.do_test(&args, approver),
            "verify" => self.do_verify(&args, approver),
            "browser" => self.do_browser(&args),
            "web_fetch" => self.do_web_fetch(&args),
            "web_search" => self.do_web_search(&args),
            "list_skills" => self.do_list_skills(&args),
            "read_skill" => self.do_read_skill(&args),
            "read_document" => self.do_read_document(&args),
            "read_image" => self.do_read_image(&args),
            "understand_repo" => self.do_understand_repo(&args),
            "rag_search" => self.do_rag_search(&args),
            "rag_index" => self.do_rag_index(&args, approver),
            "memory" => self.do_memory(&args),
            "memory_write" => self.do_memory_write(&args, approver),
            "device" => self.do_device(&args, approver),
            "bg_list" => self.do_bg_list(),
            "bg_output" => self.do_bg_output(&args),
            "bg_pause" => self.do_bg_pause(&args),
            "bg_resume" => self.do_bg_resume(&args),
            "bg_stop" => self.do_bg_stop(&args),
            "git_status" => helpers::git_result(self.git.status()),
            "git_diff" => self.do_git_diff(&args),
            "git_blame" => self.do_git_blame(&args),
            "git_log" => self.do_git_log(&args),
            "git_show" => self.do_git_show(&args),
            "git_branch_list" => helpers::git_result(self.git.branch_list()),
            "git_remote_list" => helpers::git_result(self.git.remote_list()),
            "git_tag_list" => helpers::git_result(self.git.tag_list()),
            "git_stash_list" => helpers::git_result(self.git.stash_list()),
            "git_add" => self.do_git_add(&args, approver),
            "git_commit" => self.do_git_commit(&args, approver),
            "git_stash_push" => self.do_git_stash_push(&args, approver),
            "git_stash_pop" => helpers::git_result(self.git.stash_pop(&mut *approver)),
            "git_branch_create" => self.do_git_branch_create(&args, approver),
            "git_branch_delete" => self.do_git_branch_delete(&args, approver),
            "git_tag_create" => self.do_git_tag_create(&args, approver),
            "git_checkout" => self.do_git_checkout(&args, approver),
            "git_fetch" => self.do_git_fetch(&args, approver),
            "git_pull" => helpers::git_result(self.git.pull(&mut *approver)),
            "git_push" => self.do_git_push(&args, approver),
            "git_reset" => self.do_git_reset(&args, approver),
            "git_revert" => self.do_git_revert(&args, approver),
            "git_cherry_pick" => self.do_git_cherry_pick(&args, approver),
            "git_rebase" => self.do_git_rebase(&args, approver),
            "git_merge" => self.do_git_merge(&args, approver),
            // Platform-CLI integrations route through a single dispatch entry
            // keyed off the shared `PLATFORM_TOOLS` registry (see above) —
            // names are asserted to match the spec list and the `do_platform`
            // match arms by the test suite.
            name if PLATFORM_TOOLS.contains(&name) => self.do_platform(name, &args, approver),
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

    // --- Tool implementations are in git.rs, platform.rs, and the impl blocks below ---

    fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
        args.get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::InvalidArguments {
                tool: key.into(),
                reason: format!("missing/invalid '{key}'"),
            })
    }

    fn opt_str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
        args.get(key).and_then(|v| v.as_str())
    }

    fn usize_arg(args: &Value, key: &str) -> Option<usize> {
        args.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
    }

    fn opt_bool_arg(args: &Value, key: &str) -> Option<bool> {
        args.get(key).and_then(|v| v.as_bool())
    }

    fn str_array_arg(args: &Value, key: &str) -> Vec<String> {
        args.get(key)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn u64_arg(args: &Value, key: &str) -> Result<u64> {
        args.get(key)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AgentError::InvalidArguments {
                tool: key.into(),
                reason: format!("missing/invalid '{key}'"),
            })
    }

    /// Dispatch `plugin__tool` (the part after the `plugin__` prefix) to the
    /// matching loaded native plugin. Permission-gated like MCP calls — a
    /// native plugin call is at least as consequential as an external
    /// server call (more so: it runs in-process, see the trust-boundary
    /// warning in `plugin.rs`), so it gets no less scrutiny.
    fn do_plugin_call<F>(
        &self,
        plugin_and_tool: &str,
        args: Value,
        approver: &mut F,
    ) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let Some((plugin_name, tool)) = plugin_and_tool.split_once("__") else {
            return Err(AgentError::UnknownTool(format!(
                "plugin__{plugin_and_tool}"
            )));
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
                images: Vec::new(),
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
    fn do_mcp_call<F>(
        &self,
        server_and_tool: &str,
        args: Value,
        approver: &mut F,
    ) -> Result<ToolResult>
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
                images: Vec::new(),
            }),
            Err(e) => Ok(ToolResult::err(format!(
                "mcp '{server}' tool '{tool}' failed: {e}"
            ))),
        }
    }

    // --- Core tool implementations ---

    /// The actual checklist state update happens one layer up, in
    /// `Agent::drive_turn` (it inspects this call's arguments after
    /// dispatch and emits `AgentEvent::TodosUpdated`) — `ToolManager` has
    /// no channel back to the UI, only validates the shape here and echoes
    /// a summary the model can see in its own tool-result.
    fn do_todowrite(&self, args: &Value) -> Result<ToolResult> {
        let Some(todos) = args.get("todos").and_then(|v| v.as_array()) else {
            return Ok(ToolResult::err("missing required \"todos\" array"));
        };
        for t in todos {
            if t.get("content")
                .and_then(|v| v.as_str())
                .is_none_or(str::is_empty)
            {
                return Ok(ToolResult::err(
                    "each todo needs a non-empty \"content\" string",
                ));
            }
            match t.get("status").and_then(|v| v.as_str()) {
                Some("pending" | "in_progress" | "completed") => {}
                _ => {
                    return Ok(ToolResult::err(
                        "each todo needs \"status\" to be one of pending/in_progress/completed",
                    ))
                }
            }
        }
        let done = todos
            .iter()
            .filter(|t| t.get("status").and_then(|v| v.as_str()) == Some("completed"))
            .count();
        Ok(ToolResult::ok(format!("{done}/{} done", todos.len())))
    }

    /// Current local date/time from the system clock, so the model never has
    /// to guess "today" from its training data — always fresh at call time.
    fn do_current_time(&self, _args: &Value) -> Result<ToolResult> {
        use chrono::Datelike;
        let now = chrono::Local::now();
        let off = now.offset().local_minus_utc();
        let (sign, off) = if off < 0 { ('-', -off) } else { ('+', off) };
        let content = format!(
            "Current date and time (local): {} — {}, {}\n\
             UTC offset: {sign}{:02}:{:02}\n\
             ISO week {} of year {}; day {} of {}",
            now.format("%Y-%m-%d %H:%M:%S"),
            now.format("%A"),
            now.format("%B %d, %Y"),
            off / 3600,
            (off % 3600) / 60,
            now.iso_week().week(),
            now.year(),
            now.ordinal(),
            if helpers::is_leap_year(now.year()) {
                366
            } else {
                365
            },
        );
        Ok(ToolResult::ok(content))
    }

    fn do_read(&self, args: &Value) -> Result<ToolResult> {
        let path = Self::str_arg(args, "path")?;
        let offset = args
            .get("offset")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        // Default-bounded read so an unguarded look at a huge generated file
        // can't fill the context — and the header below always states the
        // visible window, so a partial read is never mistaken for the file.
        let eff_limit = limit.unwrap_or(1500);
        let start = offset.unwrap_or(0);
        match self.workspace.files.read(
            Path::new(path),
            ReadOptions {
                offset,
                limit: Some(eff_limit),
            },
        ) {
            Ok(r) => {
                let visible_end = (start + eff_limit).min(r.total_lines);
                let header = if visible_end < r.total_lines {
                    format!(
                        "[read {path}: lines {}-{} of {} — NOT the whole file; pass offset={visible_end} to continue]\n",
                        r.start_line, visible_end, r.total_lines
                    )
                } else {
                    format!(
                        "[read {path}: full contents, lines {}-{} of {}]\n",
                        r.start_line, visible_end, r.total_lines
                    )
                };
                Ok(ToolResult::ok(format!("{header}{}", r.content)))
            }
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    /// Batch read: several paths in one call, each block bounded like `read`.
    /// A single missing/unreadable path is reported inline as an error block
    /// rather than failing the whole call, so the model can read a module
    /// plus its tests in one round-trip even when one file has moved.
    fn do_read_multiple(&self, args: &Value) -> Result<ToolResult> {
        let paths: Vec<String> = args
            .get("paths")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|p| p.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        if paths.is_empty() {
            return Ok(ToolResult::err(
                "read_multiple needs a `paths` array of file paths".to_string(),
            ));
        }
        const MAX_FILES: usize = 20;
        if paths.len() > MAX_FILES {
            return Ok(ToolResult::err(format!(
                "read_multiple accepts at most {MAX_FILES} paths, got {}",
                paths.len()
            )));
        }
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1500);
        let mut blocks = Vec::with_capacity(paths.len());
        for path in &paths {
            let eff_limit = limit.max(1);
            match self.workspace.files.read(
                Path::new(path),
                ReadOptions {
                    offset: None,
                    limit: Some(eff_limit),
                },
            ) {
                Ok(r) => {
                    let visible_end = eff_limit.min(r.total_lines);
                    let marker = if visible_end < r.total_lines {
                        format!(
                            " (lines {}-{} of {}, pass read offset={visible_end} for more)",
                            r.start_line, visible_end, r.total_lines
                        )
                    } else {
                        format!(
                            " (lines {}-{} of {})",
                            r.start_line, visible_end, r.total_lines
                        )
                    };
                    blocks.push(format!("=== {path}{marker} ===\n{}", r.content));
                }
                Err(e) => blocks.push(format!("--- {path}: {e} ---")),
            }
        }
        Ok(ToolResult::ok(blocks.join("\n")))
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
            Ok(n) => Ok(ToolResult::ok(format!(
                "edited {path} ({n} replacement(s))"
            ))),
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

    fn do_mkdir<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let path = Self::str_arg(args, "path")?;
        match self.workspace.files.mkdir(Path::new(path), &mut *approver) {
            Ok(()) => Ok(ToolResult::ok(format!("created directory {path}"))),
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_listdir(&self, args: &Value) -> Result<ToolResult> {
        let path = Self::str_arg(args, "path")?;
        let recursive = args
            .get("recursive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        match self.workspace.files.listdir(Path::new(path), recursive) {
            Ok(listing) => Ok(ToolResult::ok(listing)),
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
                let capped = max_matches > 0 && hits.len() >= max_matches;
                let mut text = hits
                    .iter()
                    .map(|h| format!("{}:{}:{}", h.path.display(), h.line, h.text))
                    .collect::<Vec<_>>()
                    .join("\n");
                if capped {
                    text.push_str(&format!(
                        "\n[truncated: hit the {max_matches}-match cap — MORE matches exist. Refine the pattern/glob or raise max before concluding anything is exhaustive.]"
                    ));
                }
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
                let capped = max > 0 && hits.len() >= max;
                let mut text = hits
                    .iter()
                    .map(|h| h.path.display().to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                if capped {
                    text.push_str(&format!(
                        "\n[truncated: hit the {max}-file cap — more files match. Narrow the pattern or raise max before treating this as the full list.]"
                    ));
                }
                Ok(ToolResult::ok(if text.is_empty() {
                    "(no files)".to_string()
                } else {
                    text
                }))
            }
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_code_index<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
        let root = self.workspace.project_root.clone();

        if !force {
            if let Ok(Some(idx)) = SymbolIndex::load(&root) {
                return Ok(ToolResult::ok(format!(
                    "index already exists: {} symbol(s) in {} file(s); pass force=true to rebuild",
                    idx.symbols.len(),
                    idx.scanned_files
                )));
            }
        }
        // Writes `.agent/index.json` below — needs the same gate every other
        // mutating tool goes through. Previously had none at all, and was
        // misclassified as read-only (letting it run even in Plan mode,
        // which is supposed to guarantee nothing changes); see
        // `is_read_only_tool`, where `"code_index"` has been removed.
        if let Err(e) = self.workspace.files.gate.enforce(
            &PermissionRequest {
                tool: "code_index".into(),
                path: Some(SymbolIndex::file_path(&root)),
                command: None,
                description: format!(
                    "build/refresh the code index at {}",
                    SymbolIndex::file_path(&root).display()
                ),
                ..Default::default()
            },
            &mut *approver,
        ) {
            return Ok(ToolResult::err(e.to_string()));
        }
        match IndexEngine::new(&root).scan() {
            Ok(idx) => match idx.save(&root) {
                Ok(_) => Ok(ToolResult::ok(format!(
                    "indexed {} symbol(s) in {} file(s) -> {}",
                    idx.symbols.len(),
                    idx.scanned_files,
                    SymbolIndex::file_path(&root).display()
                ))),
                Err(e) => Ok(ToolResult::err(format!("could not save index: {e}"))),
            },
            Err(e) => Ok(ToolResult::err(format!("scan failed: {e}"))),
        }
    }

    fn do_code_symbols(&self, args: &Value) -> Result<ToolResult> {
        let name = Self::str_arg(args, "name")?.to_string();
        self.report_index_query(&name)
    }

    fn do_code_defs(&self, args: &Value) -> Result<ToolResult> {
        let name = Self::str_arg(args, "name")?.to_string();
        let root = self.workspace.project_root.clone();
        match SymbolIndex::load(&root) {
            Ok(Some(idx)) => {
                let hits = idx.query(&name);
                if hits.is_empty() {
                    return Ok(ToolResult::ok(format!(
                        "no definition for '{name}' in the current index; the index is regex-based and best-effort, so absence is not proof the symbol doesn't exist. Verify with code_symbols or a targeted grep before concluding anything."
                    )));
                }
                let text = hits
                    .iter()
                    .map(|s| format!("{}:{}:{}  {}", s.file, s.line, s.kind, s.name))
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(ToolResult::ok(format!(
                    "{} definition(s) for '{name}':\n{text}",
                    hits.len()
                )))
            }
            Ok(None) => Ok(ToolResult::err("no index; run code_index first")),
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn report_index_query(&self, name: &str) -> Result<ToolResult> {
        let root = self.workspace.project_root.clone();
        match SymbolIndex::load(&root) {
            Ok(Some(idx)) => {
                let hits = idx.query(name);
                if hits.is_empty() {
                    return Ok(ToolResult::ok(format!(
                        "no symbols matching '{name}' in the index; the index is regex-based and best-effort, so absence is not proof of non-existence. Fall back to grep/glob for a definitive check."
                    )));
                }
                let text = hits
                    .iter()
                    .map(|s| format!("{:8} {}:{}  {}", s.kind, s.file, s.line, s.name))
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(ToolResult::ok(format!(
                    "{} match(es) for '{name}':\n{text}",
                    hits.len()
                )))
            }
            Ok(None) => Ok(ToolResult::err("no index; run code_index first")),
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_code_refs(&self, args: &Value) -> Result<ToolResult> {
        let name = Self::str_arg(args, "name")?;
        let max = args.get("max").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
        match self.workspace.search.grep(SearchOptions {
            pattern: word_boundary(name),
            glob: None,
            case_insensitive: false,
            max_matches: max,
            path: None,
        }) {
            Ok(hits) => {
                let capped = max > 0 && hits.len() >= max;
                let hits = filter_out_own_index(&self.workspace.project_root, hits);
                let mut text = hits
                    .iter()
                    .map(|h| format!("{}:{}:{}", h.path.display(), h.line, h.text))
                    .collect::<Vec<_>>()
                    .join("\n");
                if capped {
                    text.push_str(&format!(
                        "\n[truncated: hit the {max}-reference cap — MORE references may exist. Raise max or refine before treating this as exhaustive.]"
                    ));
                }
                Ok(ToolResult::ok(if text.is_empty() {
                    format!("no references to '{name}'")
                } else {
                    format!("{} reference(s) to '{name}':\n{text}", hits.len())
                }))
            }
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_code_graph(&self, args: &Value) -> Result<ToolResult> {
        let name = Self::str_arg(args, "name")?;
        let direction = args
            .get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("both");
        let root = self.workspace.project_root.clone();
        let idx = match SymbolIndex::load(&root) {
            Ok(Some(idx)) => idx,
            Ok(None) => return Ok(ToolResult::err("no index; run code_index first")),
            Err(e) => return Ok(ToolResult::err(e.to_string())),
        };

        let fmt_callers = |edges: &[&CallEdge]| -> String {
            edges
                .iter()
                .map(|c| format!("{}:{}  {} -> {name}", c.file, c.line, c.caller))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let fmt_callees = |edges: &[&CallEdge]| -> String {
            edges
                .iter()
                .map(|c| format!("{}:{}  {name} -> {}", c.file, c.line, c.callee))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let mut sections = Vec::new();
        if direction == "callers" || direction == "both" {
            let callers = idx.callers_of(name);
            sections.push(if callers.is_empty() {
                format!("no callers of '{name}' found")
            } else {
                format!(
                    "{} caller(s) of '{name}':\n{}",
                    callers.len(),
                    fmt_callers(&callers)
                )
            });
        }
        if direction == "callees" || direction == "both" {
            let callees = idx.callees_of(name);
            sections.push(if callees.is_empty() {
                format!("'{name}' calls nothing found in the graph")
            } else {
                format!(
                    "'{name}' calls {} function(s):\n{}",
                    callees.len(),
                    fmt_callees(&callees)
                )
            });
        }
        sections.push("Call graph is built from tree-sitter AST parsing of languages with a wired grammar; it misses dynamic dispatch/reflection and any language without a grammar, so absence is not proof of no callers/callees.".to_string());
        Ok(ToolResult::ok(sections.join("\n\n")))
    }

    fn do_code_rename(&self, args: &Value) -> Result<ToolResult> {
        let old = Self::str_arg(args, "old")?;
        let new = Self::str_arg(args, "new")?;
        let hits = match self.workspace.search.grep(SearchOptions {
            pattern: word_boundary(old),
            glob: None,
            case_insensitive: false,
            max_matches: 2000,
            path: None,
        }) {
            Ok(h) => h,
            Err(e) => return Ok(ToolResult::err(e.to_string())),
        };
        let hits = filter_out_own_index(&self.workspace.project_root, hits);
        if hits.is_empty() {
            return Ok(ToolResult::ok(format!("no references to '{old}'")));
        }
        let mut by_file: Vec<(std::path::PathBuf, Vec<usize>)> = Vec::new();
        for h in &hits {
            match by_file.iter().position(|(p, _)| *p == h.path) {
                Some(i) => by_file[i].1.push(h.line),
                None => by_file.push((h.path.clone(), vec![h.line])),
            }
        }
        let mut out = format!(
            "rename '{old}' -> '{new}': {} reference(s) in {} file(s)\n",
            hits.len(),
            by_file.len()
        );
        for (f, lines) in &by_file {
            let shown: Vec<String> = lines.iter().take(5).map(|l| l.to_string()).collect();
            let suffix = if lines.len() > 5 { ", …" } else { "" };
            out.push_str(&format!(
                "  {}: {} line(s) [{}]\n",
                f.display(),
                lines.len(),
                shown.join(", ") + suffix
            ));
        }
        out.push_str("Plan only — review and apply the edits yourself before they take effect.");
        Ok(ToolResult::ok(out))
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
            if let Err(e) = self.workspace.files.gate.enforce(
                &PermissionRequest {
                    tool: "bash".into(),
                    path: None,
                    command: Some(command.to_string()),
                    description: format!("run as background task: {command}"),
                    ..Default::default()
                },
                &mut *approver,
            ) {
                return Ok(ToolResult::err(e.to_string()));
            }
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

    /// Run the project's tests. Auto-detects the test command from common
    /// manifest files, or honors an explicit `command` override. Output is
    /// the same bounded format as `bash` plus a parsed pass/fail summary so
    /// the model gets a verdict without scraping raw logs.
    fn do_test<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let root = self.workspace.project_root.clone();
        let command = match Self::str_arg(args, "command") {
            Ok(c) if !c.trim().is_empty() => c.trim().to_string(),
            _ => match detect_test_command(&root) {
                Some(c) => c,
                None => {
                    return Ok(ToolResult::err(format!(
                        "couldn't auto-detect a test command in {}. Pass an explicit `command` (e.g. \"cargo test\" or \"pytest -q\").",
                        root.display()
                    )));
                }
            },
        };
        let timeout = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .map(Duration::from_secs)
            .or(Some(Duration::from_secs(300)));
        let opts = TerminalOptions {
            cwd: root.clone(),
            timeout,
            sandbox: Sandbox::RestrictedFs,
            profile: CommandProfile::Foreground,
            use_pty: false,
        };
        match self.terminal.run(
            &command,
            &self.workspace.files.gate,
            opts,
            self.cancel.clone(),
            &mut *approver,
        ) {
            Ok(out) => {
                let summary = helpers::summarize_test_output(&out.stdout);
                let text = format!(
                    "command: {command}\nexit={:?} cancelled={} timed_out={}\n[test summary]\n{summary}\n--- stdout ---\n{}--- stderr ---\n{}{}",
                    out.exit_code,
                    out.cancelled,
                    out.timed_out,
                    out.stdout,
                    out.stderr,
                    if out.truncated { "\n(output truncated)" } else { "" }
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

    /// Run the project's detected build/test commands (from the zeus-lang
    /// spec) to prove the code compiles and tests pass. An explicit
    /// `command` overrides detection; `steps` picks build/test/all.
    fn do_verify<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let root = self.workspace.project_root.clone();
        let spec = zeus_lang::detect_project(&root).map(zeus_lang::spec);
        let lang_name = spec
            .map(|s| s.display_name.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let join_cmds = |args: &[&'static str]| args.join(" ");
        let build_cmd = spec.map(|s| join_cmds(s.build)).filter(|c| !c.is_empty());
        let test_cmd = spec.map(|s| join_cmds(s.test)).filter(|c| !c.is_empty());

        let explicit = match Self::str_arg(args, "command") {
            Ok(c) if !c.trim().is_empty() => Some(c.trim().to_string()),
            _ => None,
        };
        let steps = Self::str_arg(args, "steps")
            .unwrap_or("all")
            .to_ascii_lowercase();
        let timeout = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(600));

        let mut to_run: Vec<String> = Vec::new();
        if let Some(c) = &explicit {
            to_run.push(c.clone());
        } else {
            match steps.as_str() {
                "build" => match &build_cmd {
                    Some(c) => to_run.push(c.clone()),
                    None => {
                        return Ok(ToolResult::err(format!(
                        "no build command configured for {lang_name} — pass an explicit `command`"
                    )))
                    }
                },
                "test" => match &test_cmd {
                    Some(c) => to_run.push(c.clone()),
                    None => {
                        return Ok(ToolResult::err(format!(
                        "no test command configured for {lang_name} — pass an explicit `command`"
                    )))
                    }
                },
                _ => {
                    if let Some(c) = &build_cmd {
                        to_run.push(c.clone());
                    }
                    if let Some(c) = &test_cmd {
                        to_run.push(c.clone());
                    }
                    if to_run.is_empty() {
                        return Ok(ToolResult::err(format!(
                            "couldn't detect any build or test command for this project \
                             (language: {lang_name}). Pass an explicit `command`."
                        )));
                    }
                }
            }
        }

        let mut report = String::new();
        let mut all_ok = true;
        for command in to_run {
            let opts = TerminalOptions {
                cwd: root.clone(),
                timeout: Some(timeout),
                sandbox: Sandbox::RestrictedFs,
                profile: CommandProfile::Foreground,
                use_pty: false,
            };
            match self.terminal.run(
                &command,
                &self.workspace.files.gate,
                opts,
                self.cancel.clone(),
                &mut *approver,
            ) {
                Ok(out) => {
                    let ok = out.exit_code == Some(0) && !out.cancelled && !out.timed_out;
                    all_ok &= ok;
                    report.push_str(&format!(
                        "> {command}\nexit={:?} cancelled={} timed_out={}\n--- stdout ---\n{}--- stderr ---\n{}{}\n",
                        out.exit_code,
                        out.cancelled,
                        out.timed_out,
                        out.stdout,
                        out.stderr,
                        if out.truncated { "\n(output truncated)" } else { "" }
                    ));
                }
                Err(e) => {
                    all_ok = false;
                    report.push_str(&format!("> {command}\nfailed to run: {e}\n"));
                }
            }
        }
        if all_ok {
            Ok(ToolResult::ok(report))
        } else {
            Ok(ToolResult::err(report))
        }
    }

    /// Open a URL in the default browser for visual verification of a
    /// running app. Launch-and-forget: spawns the platform opener and
    /// returns immediately — the browser window stays open on the user's
    /// machine while the agent keeps talking to them about what they see.
    fn do_browser(&self, args: &Value) -> Result<ToolResult> {
        let url = Self::str_arg(args, "url")?;
        let url = url.trim();
        match helpers::open_browser_url(url) {
            Ok(()) => Ok(ToolResult::ok(format!(
                "opened {url} in the default browser — the user is looking at it now. Readable Chrome DevTools-level DOM/inspection is not available from here; tell the user what to verify (layout, console errors, requests) and ask what they observe."
            ))),
            Err(e) => Ok(ToolResult::err(format!(
                "couldn't open {url}: {e}. On non-GUI/headless machines there may be no browser to launch."
            ))),
        }
    }

    /// Fetch a URL over HTTP(S) and return its content to the model — the
    /// actual web-scrape counterpart to `browser` (which only opens a page).
    /// Follows redirects, caps the body, and strips HTML to approximate text
    /// by default so the model gets readable content rather than raw markup.
    fn do_web_fetch(&self, args: &Value) -> Result<ToolResult> {
        let url = Self::str_arg(args, "url")?;
        let url = url.trim();
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Ok(ToolResult::err(format!(
                "'{url}' isn't an http(s) URL — web_fetch needs an absolute http:// or https:// address"
            )));
        }
        let max_chars = args
            .get("max_chars")
            .and_then(|v| v.as_u64())
            .unwrap_or(20_000) as usize;
        let selective = args
            .get("selective")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        if let Some(reason) = helpers::reject_web_target(url) {
            return Ok(ToolResult::err(format!("web_fetch refused: {reason}")));
        }

        let client = match reqwest::blocking::Client::builder()
            .user_agent("zeus-agent/0.1 (coding assistant; fetch-for-the-agent)")
            .timeout(std::time::Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
        {
            Ok(c) => c,
            Err(e) => return Ok(ToolResult::err(format!("http client init failed: {e}"))),
        };

        let resp = match client.get(url).send() {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult::err(format!(
                    "request failed for {url}: {e} (network unreachable or DNS/TLS error)"
                )))
            }
        };
        if let Some(reason) = helpers::reject_web_target(resp.url().as_str()) {
            return Ok(ToolResult::err(format!(
                "web_fetch refused after redirect: {reason}"
            )));
        }
        let status = resp.status();
        if !status.is_success() {
            return Ok(ToolResult::err(format!(
                "HTTP {status} for {url} — fetch only returns 2xx content"
            )));
        }
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();
        if content_type.contains("json")
            || content_type.contains("text")
            || content_type.contains("xml")
        {
            // fine
        } else {
            return Ok(ToolResult::err(format!(
                "refused to fetch {url}: content-type '{content_type}' isn't text/web content"
            )));
        }
        let body = match resp.text() {
            Ok(t) => t,
            Err(e) => return Ok(ToolResult::err(format!("body read failed for {url}: {e}"))),
        };
        let mut content = if selective && content_type.contains("html") {
            helpers::strip_html(&body)
        } else {
            body
        };
        if content.chars().count() > max_chars {
            content = content.chars().take(max_chars).collect::<String>();
            content.push_str("\n… (truncated, max_chars reached)");
        }
        Ok(ToolResult::ok(format!("# web_fetch {url}\n{content}")))
    }

    /// `web_search` — query a public web search endpoint and return the top
    /// result titles/URLs/snippets. Uses DuckDuckGo's keyless HTML search
    /// (fast, no account/API key), so it works out of the box; the model
    /// should `web_fetch` the most promising result for full content.
    fn do_web_search(&self, args: &Value) -> Result<ToolResult> {
        let query = Self::str_arg(args, "query")?.trim().to_string();
        if query.is_empty() {
            return Ok(ToolResult::err("web_search needs a non-empty `query`"));
        }
        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(6)
            .clamp(1, 10) as usize;

        let client = match reqwest::blocking::Client::builder()
            .user_agent("Mozilla/5.0 (zeus-agent; coding assistant search)")
            .timeout(std::time::Duration::from_secs(20))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
        {
            Ok(c) => c,
            Err(e) => return Ok(ToolResult::err(format!("http client init failed: {e}"))),
        };
        let endpoint = format!(
            "https://html.duckduckgo.com/html/?q={}",
            helpers::urlencode(&query)
        );
        let resp = match client.get(&endpoint).send() {
            Ok(r) => r,
            Err(e) => return Ok(ToolResult::err(format!("search request failed: {e}"))),
        };
        if !resp.status().is_success() {
            return Ok(ToolResult::err(format!(
                "search request returned HTTP {}",
                resp.status()
            )));
        }
        let html = match resp.text() {
            Ok(t) => t,
            Err(e) => return Ok(ToolResult::err(format!("search body read failed: {e}"))),
        };

        let mut results: Vec<(String, String, String)> = Vec::new();
        for chunk in html.split("result__a").skip(1) {
            if results.len() >= max_results {
                break;
            }
            let Some(href_start) = chunk.find("href=\"") else {
                continue;
            };
            let url = &chunk[href_start + 6
                ..chunk[href_start + 6..]
                    .find('"')
                    .map(|i| i + href_start + 6)
                    .unwrap_or(href_start + 6)];
            let Some(title_end) = chunk.find("</a>") else {
                continue;
            };
            let title = helpers::strip_html(&chunk[..title_end]);
            let snippet = chunk
                .find("result__snippet")
                .and_then(|s| {
                    let seg = &chunk[s..];
                    let o = seg.find(">").map(|o| o + 1);
                    o.map(|o| seg[o..seg.len().min(o + 400)].to_string())
                })
                .map(|s| helpers::strip_html(&s))
                .unwrap_or_default();
            let url_clean = url.trim_start_matches("//").to_string();
            results.push((
                title.trim().to_string(),
                url_clean,
                snippet.trim().to_string(),
            ));
        }

        if results.is_empty() {
            return Ok(ToolResult::err(
                "no results returned (network or provider issue; try again, or use web_fetch for a known URL)",
            ));
        }
        let mut out = format!("Web search results for: `{query}`\n");
        for (i, (title, url, snippet)) in results.iter().enumerate() {
            out.push_str(&format!(
                "\n{}. {title}\n   {url}\n   {}\n",
                i + 1,
                if snippet.is_empty() {
                    "(no snippet)".to_string()
                } else {
                    snippet.clone()
                }
            ));
        }
        out.push_str("\nUse web_fetch on the most relevant URL above for full content.");
        Ok(ToolResult::ok(out))
    }

    /// All discoverable skills (project > user > built-in), deduped by name
    /// with highest tier winning.
    fn all_skills(&self) -> Vec<crate::skills::Skill> {
        use crate::skills::{builtin_skill, discover_in_dir, Skill, SkillTier, BUILTIN_SKILLS};
        let mut by_name: std::collections::BTreeMap<String, Skill> =
            std::collections::BTreeMap::new();
        let project_dir = self.workspace.project_root.join(".agent").join("skills");
        let user_dir = self.global_skills_dir.clone();
        for tier in [SkillTier::Project, SkillTier::Global, SkillTier::Builtin] {
            let candidates: Vec<Skill> = match tier {
                SkillTier::Project => discover_in_dir(&project_dir, tier),
                SkillTier::Global => user_dir
                    .as_ref()
                    .map(|d| discover_in_dir(d, tier))
                    .unwrap_or_default(),
                SkillTier::Builtin => vec![],
            };
            for skill in candidates {
                by_name.entry(skill.name.clone()).or_insert(skill);
            }
        }
        for def in BUILTIN_SKILLS {
            by_name
                .entry(def.0.to_string())
                .or_insert_with(|| builtin_skill(def));
        }
        by_name.into_values().collect()
    }

    fn do_list_skills(&self, args: &Value) -> Result<ToolResult> {
        let search = Self::opt_str_arg(args, "search")
            .map(|s| s.to_lowercase())
            .unwrap_or_default();
        let skills = self.all_skills();
        if skills.is_empty() {
            return Ok(ToolResult::ok(
                "No skills installed. Create a `.agent/skills/<name>/SKILL.md` (project) or `~/.zeus/skills/<name>/SKILL.md` (user).",
            ));
        }
        let mut lines = Vec::new();
        for skill in skills {
            let hay =
                format!("{} {} {:?}", skill.name, skill.description, skill.tags).to_lowercase();
            if !search.is_empty() && !hay.contains(&search) {
                continue;
            }
            let tier = skill.tier.label();
            let tags = if skill.tags.is_empty() {
                String::new()
            } else {
                format!(" [{}]", skill.tags.join(", "))
            };
            lines.push(format!(
                "[{tier}] {name} — {desc}{tags}",
                name = skill.name,
                desc = if skill.description.is_empty() {
                    "(no description)"
                } else {
                    &skill.description
                },
                tags = tags,
            ));
        }
        if lines.is_empty() {
            Ok(ToolResult::ok(format!(
                "No skills match '{search}'. Run list_skills with no search to see everything."
            )))
        } else {
            Ok(ToolResult::ok(format!(
                "Available skills (call read_skill with the name to load one):\n{}",
                lines.join("\n")
            )))
        }
    }

    fn do_read_skill(&self, args: &Value) -> Result<ToolResult> {
        let name = Self::str_arg(args, "name")?.to_lowercase();
        use crate::skills::{read_skill_resource, skill_resources};
        let include_resources = Self::opt_bool_arg(args, "include_resources").unwrap_or(true);
        let recursive = Self::opt_bool_arg(args, "recursive").unwrap_or(true);
        let all = self.all_skills();
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut queue: std::collections::VecDeque<String> = std::collections::VecDeque::new();
        queue.push_back(name.clone());
        let mut ordered: Vec<crate::skills::Skill> = Vec::new();
        let mut missing: Vec<String> = Vec::new();
        while let Some(n) = queue.pop_front() {
            if !seen.insert(n.clone()) {
                continue;
            }
            match all.iter().find(|s| s.name == n) {
                Some(skill) => {
                    ordered.push(skill.clone());
                    if recursive {
                        for dep in &skill.depends_on {
                            if !seen.contains(dep) {
                                queue.push_back(dep.clone());
                            }
                        }
                    }
                }
                None => {
                    if n != name {
                        missing.push(n);
                    }
                }
            }
        }
        if ordered.is_empty() {
            let available: Vec<String> = all.iter().map(|s| s.name.clone()).collect();
            return Ok(ToolResult::err(format!(
                "unknown skill '{name}'. Available: {}",
                if available.is_empty() {
                    "(none)".to_string()
                } else {
                    available.join(", ")
                }
            )));
        }
        let mut out = String::new();
        for skill in ordered {
            out.push_str(&format!(
                "# skill: {} (tier: {})\n\n{}",
                skill.name,
                skill.tier.label(),
                skill.instructions
            ));
            if !skill.depends_on.is_empty() {
                out.push_str(&format!(
                    "\n*(composes: {})*\n",
                    skill.depends_on.join(", ")
                ));
            }
            if !skill.resources_are_empty() {
                let resources = skill_resources(&skill);
                if include_resources {
                    let mut inline = String::new();
                    for res in &resources {
                        if let Some(content) = read_skill_resource(&skill, res) {
                            inline.push_str(&format!("\n--- {res} ---\n{content}\n"));
                        }
                    }
                    out.push_str(&format!(
                        "\n## bundled resources ({})\n{}\n{}",
                        resources.join(", "),
                        resources.join(", "),
                        inline
                    ));
                } else {
                    out.push_str(&format!(
                        "\n## bundled resources ({})\n",
                        resources.join(", ")
                    ));
                }
            }
            out.push('\n');
        }
        if !missing.is_empty() {
            out.push_str(&format!(
                "\n*(note: depended-on skill(s) not found: {})*\n",
                missing.join(", ")
            ));
        }
        Ok(ToolResult::ok(out))
    }

    fn do_read_document(&self, args: &Value) -> Result<ToolResult> {
        let path = Self::str_arg(args, "path")?;
        let max_chars = Self::usize_arg(args, "max_chars")
            .unwrap_or(100_000)
            .max(1000);
        let root = self.workspace.project_root.clone();
        let resolved = match zeus_fs::resolve_in_project(&root, Path::new(path)) {
            Ok(p) => p,
            Err(e) => return Ok(ToolResult::err(e.to_string())),
        };
        match crate::docread::extract(&resolved, max_chars) {
            Ok(doc) => {
                let mut text =
                    format!("# {} — {}\n\n{}", resolved.display(), doc.summary, doc.text);
                if text.chars().count() > max_chars {
                    text = text.chars().take(max_chars).collect::<String>();
                    text.push_str("\n…(truncated by tool cap)");
                }
                Ok(ToolResult::ok(text))
            }
            Err(e) => Ok(ToolResult::err(format!(
                "could not extract {}: {e}",
                resolved.display()
            ))),
        }
    }

    fn do_read_image(&self, args: &Value) -> Result<ToolResult> {
        use base64::Engine;
        use zeus_provider::ImagePart;

        let path = Self::str_arg(args, "path")?;
        let root = self.workspace.project_root.clone();
        let resolved = match zeus_fs::resolve_in_project(&root, Path::new(path)) {
            Ok(p) => p,
            Err(e) => return Ok(ToolResult::err(e.to_string())),
        };
        let bytes = match std::fs::read(&resolved) {
            Ok(b) => b,
            Err(e) => {
                return Ok(ToolResult::err(format!(
                    "could not read {}: {e}",
                    resolved.display()
                )))
            }
        };
        let mime = helpers::image_mime(&resolved);
        let Some(mime) = mime else {
            return Ok(ToolResult::err(format!(
                "{} is not a supported image format (png/jpg/jpeg/gif/webp/bmp)",
                resolved.display()
            )));
        };
        if bytes.is_empty() {
            return Ok(ToolResult::err(format!("{} is empty", resolved.display())));
        }
        let data_base64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let kb = bytes.len() as f64 / 1024.0;
        Ok(ToolResult {
            content: format!(
                "Read image {} ({mime}, {kb:.0} KiB). The image data itself is attached to this message — describe what you see and use it as the design source.",
                resolved.display(),
            ),
            is_error: false,
            images: vec![ImagePart { mime_type: mime.to_string(), data_base64 }],
        })
    }

    fn do_understand_repo(&self, args: &Value) -> Result<ToolResult> {
        let topic = Self::str_arg(args, "topic").unwrap_or_default();
        let root = self.project_root();
        let fp = match &self.repo {
            Some(fp) => fp.clone(),
            None => crate::project::load_or_analyze(&root),
        };
        let text = if topic.trim().is_empty() {
            format!(
                "Repository understanding:\n{}",
                fp.banner_lines().join("\n")
            )
        } else {
            fp.render(topic)
        };
        Ok(ToolResult::ok(text))
    }

    fn do_rag_search(&self, args: &Value) -> Result<ToolResult> {
        let query = Self::opt_str_arg(args, "query")
            .unwrap_or_default()
            .to_string();
        let k = args
            .get("k")
            .and_then(|v| v.as_u64())
            .unwrap_or(5)
            .clamp(1, 20) as usize;
        if query.trim().is_empty() {
            return Ok(ToolResult::err("query must not be empty"));
        }
        let root = self.workspace.project_root.clone();
        let index = match zeus_rag::PersistedRagIndex::load(&root) {
            Some(persisted) if persisted.is_fresh() => persisted.into_index(),
            _ => zeus_rag::RagIndex::from_project(&root, 800, 80),
        };
        if index.is_empty() {
            return Ok(ToolResult::ok("no source files to search"));
        }
        let hits = index.search(&query, k);
        if hits.is_empty() {
            return Ok(ToolResult::ok(format!(
                "no chunks matched '{query}' (searched {} chunk(s) in {} file(s)); try different wording or grep for exact strings",
                index.len(),
                zeus_rag::chunker::source_files(&root).len()
            )));
        }
        let lines: Vec<String> = hits
            .iter()
            .map(|h| {
                let path = h
                    .chunk
                    .path
                    .strip_prefix(&root)
                    .unwrap_or(&h.chunk.path)
                    .display();
                format!("[{:.0}%] {}:\n{}", h.score * 100.0, path, h.chunk.text)
            })
            .collect();
        Ok(ToolResult::ok(format!(
            "top {} match(es) for '{query}':\n\n{}",
            hits.len(),
            lines.join("\n\n")
        )))
    }

    fn do_rag_index<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let force = args.get("force").and_then(|v| v.as_bool()).unwrap_or(false);
        let embed = args.get("embed").and_then(|v| v.as_bool()).unwrap_or(false);
        let root = self.workspace.project_root.clone();
        let path = zeus_rag::PersistedRagIndex::file_path(&root);

        let mut persisted = zeus_rag::PersistedRagIndex::load(&root);
        if !force {
            if let Some(p) = persisted.as_ref() {
                if p.is_fresh() && (!embed || p.has_vectors()) {
                    return Ok(ToolResult::ok(format!(
                        "index already exists and is fresh: {} chunk(s) in {} file(s); pass force=true to rebuild",
                        p.documents.len(), p.stamps.len()
                    )));
                }
            }
        }

        if let Err(e) = self.workspace.files.gate.enforce(
            &PermissionRequest {
                tool: "rag_index".into(),
                path: Some(path.clone()),
                command: None,
                description: format!("build/refresh the RAG chunk index at {}", path.display()),
                ..Default::default()
            },
            &mut *approver,
        ) {
            return Ok(ToolResult::err(e.to_string()));
        }

        let mut index = if let Some(mut p) = persisted.take() {
            if !force {
                p.refresh(800, 80);
            }
            p.into_index()
        } else {
            zeus_rag::RagIndex::from_project(&root, 800, 80)
        };

        if index.is_empty() {
            return Ok(ToolResult::ok("no source files to index"));
        }

        let mut notes = Vec::new();
        if embed {
            match self.embed_index(&mut index) {
                Some(n) if n > 0 => notes.push(format!("embedded {n} chunk(s)")),
                _ => notes.push("no embedding provider reachable; index kept keyword-only".into()),
            }
        }

        let persisted = zeus_rag::PersistedRagIndex::from_index(&index);
        match persisted.save(&root) {
            Ok(_) => {
                let mut msg = format!(
                    "indexed {} chunk(s) in {} file(s) -> {}",
                    index.len(),
                    persisted.stamps.len(),
                    path.display()
                );
                if !notes.is_empty() {
                    msg.push_str("; ");
                    msg.push_str(&notes.join("; "));
                }
                Ok(ToolResult::ok(msg))
            }
            Err(e) => Ok(ToolResult::err(format!("could not save index: {e}"))),
        }
    }

    fn embed_index(&self, index: &mut zeus_rag::RagIndex) -> Option<usize> {
        let provider = self.embedder.as_ref()?;
        let model = self.embed_model.as_ref()?;
        let handle = match tokio::runtime::Handle::try_current() {
            Ok(h) => h,
            Err(_) => {
                tracing::warn!(
                    "no tokio runtime available for embeddings; index kept keyword-only"
                );
                return Some(0);
            }
        };
        let provider = provider.clone();
        let model = model.clone();
        let mut work = index.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        handle.spawn(async move {
            let res = work.embed_all(&*provider, &model, 32).await;
            let _ = tx.send((work, res));
        });
        match rx.recv() {
            Ok((done, Ok(n))) => {
                if n > 0 {
                    if let Some(v) = done.vectors {
                        index.set_vectors(v);
                    }
                }
                Some(n)
            }
            Ok((_, Err(e))) => {
                tracing::warn!(err = %e, "embedding failed; index kept keyword-only");
                Some(0)
            }
            Err(_) => {
                tracing::warn!("embedding task did not complete; index kept keyword-only");
                Some(0)
            }
        }
    }

    fn do_memory(&self, args: &Value) -> Result<ToolResult> {
        let action = Self::str_arg(args, "action")?.to_ascii_lowercase();
        let root = self.project_root();
        match action.as_str() {
            "list" => {
                let idx = crate::project::memory_index(&root);
                if idx.is_empty() {
                    return Ok(ToolResult::ok(
                        "No long-term memory yet. Use `memory_write` to persist a decision/convention/gotcha across sessions.",
                    ));
                }
                let lines: Vec<String> = idx
                    .iter()
                    .map(|(n, first)| format!("- {n}: {first}"))
                    .collect();
                Ok(ToolResult::ok(format!(
                    ".agent/memory/ notes ({}):\n{}",
                    idx.len(),
                    lines.join("\n")
                )))
            }
            "read" => {
                let name = Self::str_arg(args, "name")?;
                match crate::project::memory_read(&root, name) {
                    Some(body) => Ok(ToolResult::ok(format!(".agent/memory/{name}.md:\n{body}"))),
                    None => Ok(ToolResult::err(format!("no memory note named `{name}`"))),
                }
            }
            other => Ok(ToolResult::err(format!(
                "unknown memory action `{other}` (expected list|read)"
            ))),
        }
    }

    fn do_memory_write<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let name = Self::str_arg(args, "name")?;
        let content = Self::str_arg(args, "content")?.to_string();
        let global = args
            .get("global")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let path_name = match crate::project::safe_memory_name(name) {
            Some(safe) => safe,
            None => {
                return Ok(ToolResult::err(
                    "invalid memory name (letters, digits, `-`, `_`)",
                ))
            }
        };
        if global {
            // Write to global memory at ~/.zeus/memory/
            let dir = dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".zeus")
                .join("memory");
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join(format!("{path_name}.md"));
            match std::fs::write(&path, &content) {
                Ok(()) => Ok(ToolResult::ok(format!(
                    "wrote global memory: {path_name}.md"
                ))),
                Err(e) => Ok(ToolResult::err(e.to_string())),
            }
        } else {
            let rel = format!(".agent/memory/{path_name}.md");
            match self.workspace.files.write(
                Path::new(&rel),
                &content,
                WriteOptions::default(),
                &mut *approver,
            ) {
                Ok(()) => Ok(ToolResult::ok(format!(
                    "wrote .agent/memory/{path_name}.md"
                ))),
                Err(e) => Ok(ToolResult::err(e.to_string())),
            }
        }
    }

    fn do_device<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let action = Self::str_arg(args, "action")?.to_ascii_lowercase();
        let device = &self.device;
        if !matches!(
            action.as_str(),
            "devices"
                | "connect"
                | "disconnect"
                | "install"
                | "uninstall"
                | "launch"
                | "screenshot"
                | "screenrecord"
                | "logcat"
                | "logcat_clear"
                | "shell"
                | "pair"
                | "info"
                | "reverse"
                | "forward"
                | "input"
                | "pull"
                | "push"
        ) {
            return Ok(ToolResult::err(format!(
                "unknown device action '{action}' — use one of: devices, connect, disconnect, install, uninstall, launch, screenshot, screenrecord, logcat, logcat_clear, shell, pair, info, reverse, forward, input, pull, push"
            )));
        }

        let opt_str = |key: &str| Self::opt_str_arg(args, key).map(|s| s.to_string());
        let req_str = |key: &str| {
            Self::str_arg(args, key)
                .map(|s| s.to_string())
                .map_err(|_| AgentError::InvalidArguments {
                    tool: "device".into(),
                    reason: format!("action '{action}' requires '{key}'"),
                })
        };

        let result = match action.as_str() {
            "devices" => device.devices(&mut *approver),
            "connect" => device.connect(&req_str("target")?, &mut *approver),
            "disconnect" => device.disconnect(&req_str("target")?, &mut *approver),
            "install" => device.install(&req_str("path")?, &mut *approver),
            "uninstall" => device.uninstall(&req_str("package")?, &mut *approver),
            "launch" => device.launch(
                &req_str("package")?,
                opt_str("activity").as_deref(),
                &mut *approver,
            ),
            "screenshot" => device.screenshot(opt_str("out").as_deref(), &mut *approver),
            "screenrecord" => {
                let seconds = args.get("seconds").and_then(|v| v.as_u64()).unwrap_or(10) as u32;
                device.screenrecord(opt_str("out").as_deref(), seconds, &mut *approver)
            }
            "logcat" => {
                let max = args
                    .get("max_lines")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(200) as usize;
                device.logcat(opt_str("filter").as_deref(), max, &mut *approver)
            }
            "logcat_clear" => device.logcat_clear(&mut *approver),
            "shell" => device.shell(&req_str("command")?, &mut *approver),
            "pair" => device.pair(&req_str("host_port")?, &req_str("code")?, &mut *approver),
            "info" => device.info(&mut *approver),
            "reverse" => {
                let local = args
                    .get("local_port")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                let dev = args
                    .get("device_port")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                device.reverse(local, dev, &mut *approver)
            }
            "forward" => {
                let local = args
                    .get("local_port")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                let dev = args
                    .get("device_port")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                device.forward(local, dev, &mut *approver)
            }
            "input" => device.input(&req_str("event")?, &mut *approver),
            "pull" => device.pull(
                &req_str("remote")?,
                opt_str("out").as_deref(),
                &mut *approver,
            ),
            "push" => device.push(&req_str("out")?, &req_str("remote")?, &mut *approver),
            _ => unreachable!("validated above"),
        };

        match result {
            Ok(out) => Ok(helpers::device_result(out)),
            Err(e) => Ok(ToolResult::err(format!(
                "device action '{action}' failed: {e}"
            ))),
        }
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

    fn do_bg_pause(&self, args: &Value) -> Result<ToolResult> {
        let id = Self::u64_arg(args, "id")?;
        match self.background.pause(id) {
            Ok(()) => Ok(ToolResult::ok(format!(
                "paused background task {id}; resume it later with bg_resume"
            ))),
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    fn do_bg_resume(&self, args: &Value) -> Result<ToolResult> {
        let id = Self::u64_arg(args, "id")?;
        match self.background.resume(id) {
            Ok(()) => Ok(ToolResult::ok(format!("resumed background task {id}"))),
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }
}
