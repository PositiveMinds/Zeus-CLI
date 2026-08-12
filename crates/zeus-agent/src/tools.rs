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
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;
use zeus_fs::{
    ApprovalDecision, CopyOptions, DeviceEngine, EditOptions, GitEngine, GitOutput, IndexEngine,
    PermissionGate, PermissionRequest, PlatformEngine, PlatformOutput, ReadOptions, ResetMode,
    SearchOptions, SymbolIndex, Workspace, WriteOptions, filter_out_own_index, word_boundary,
};
use zeus_provider::ToolSpec;

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

/// Tool specs advertised to the model. Kept in sync with `ToolManager`'s
/// `dispatch_with_approver` match arms below — every name here must have a
/// handler, and vice versa.
pub fn builtin_tool_specs() -> Vec<ToolSpec> {
    let mut specs = vec![
        ToolSpec {
            name: "todowrite".into(),
            description: "Replace your own progress checklist for this session with the given list — call this whenever you break a request into multiple steps, and again every time a step's status changes. You own this list entirely: pass the FULL list every time (not a diff), including items already completed. Mark exactly one item in_progress at a time (the one you're actively working on), never more; mark an item completed only once you've actually verified it, not just attempted it. Skip this tool for a single trivial action.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "Short imperative description of the step"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                            },
                            "required": ["content", "status"]
                        }
                    }
                },
                "required": ["todos"]
            }),
        },
        ToolSpec {
            name: "read".into(),
            description: "Read a project file (line-numbered output). The result is prefixed with the exact line window shown (e.g. lines 1-500 of 3200) — if it says the file continues, pass offset=<next line> to keep reading; never treat a partial read as the whole file.".into(),
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
            description: "Stop a running (or paused) background task by id.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "id": {"type": "integer"} },
                "required": ["id"]
            }),
        },
        ToolSpec {
            name: "bg_pause".into(),
            description: "Suspend a running background task in place (freezes it without killing the process); resume it later with bg_resume using the same id.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "id": {"type": "integer"} },
                "required": ["id"]
            }),
        },
        ToolSpec {
            name: "bg_resume".into(),
            description: "Continue a previously-paused background task, exactly where it stopped.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "id": {"type": "integer"} },
                "required": ["id"]
            }),
        },
        // --- Verification: tests + visual (browser) ---
        ToolSpec {
            name: "test".into(),
            description: "Run the project's test suite. Auto-detects the test runner from the repo (cargo test / npm|pnpm|yarn test / python -m pytest / go test / make test); pass an explicit `command` to override when a targeted run is needed (single test, extra flags). Bounded by timeout_secs (default 300). Returns the exit code plus a parsed pass/fail summary — treat a nonzero exit as a failing suite and read the stderr below it.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Explicit test command to run instead of auto-detection"},
                    "timeout_secs": {"type": "integer"}
                }
            }),
        },
        ToolSpec {
            name: "browser".into(),
            description: "Open a URL in the user's default web browser so the running app can be visually inspected. Use AFTER starting a dev server (bash background=true + bg_output). Accepts http(s):// URLs and localhost:port-style addresses (http:// scheme is added automatically for bare host:port). The human looks at the page — tell them what to check and ask what they see.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "url": {"type": "string"} },
                "required": ["url"]
            }),
        },
        ToolSpec {
            name: "web_fetch".into(),
            description: "Fetch a URL over HTTP(S) and return its content as text. Use to scrape docs, read an API/endpoint response, download raw source, or inspect a web page the model needs to act on (the browser tool just opens it for a human — web_fetch returns the actual content here). max_chars caps the returned body (default 20000); selective=true strips HTML to approximate markdown text instead of returning raw HTML. Errors on non-2xx status and on obviously non-text content. Requires network access.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Cap on returned characters (default 20000)"},
                    "selective": {"type": "boolean", "description": "Strip HTML tags to text (default true)"}
                },
                "required": ["url"]
            }),
        },
        ToolSpec {
            name: "web_search".into(),
            description: "Search the web and return the top result titles, URLs, and snippets. Use when you need current or external information (latest library versions, API docs, third-party package details, known issues) rather than relying on possibly-stale training knowledge. `query` is the search string; `max_results` caps the returned results (default 6). Then call web_fetch on the most promising URL for full content. Requires network access.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results to return (default 10)"}
                },
                "required": ["query"]
            }),
        },
        ToolSpec {
            name: "list_skills".into(),
            description: "List available skills (project `<project>/.agent/skills`, user `~/.zeus/skills`, and built-ins). Skills are just-in-time expertise packages — SKILL.md directories with instructions and bundled resources. Returns <tier> name — description plus tags. Call read_skill before acting on a skill you intend to use.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "search": {"type": "string", "description": "Optional substring to filter names/descriptions/tags by"}
                }
            }),
        },
ToolSpec {
            name: "read_skill".into(),
            description: "Load a skill's full SKILL.md instructions into context by name. Use when a listed skill is relevant to the current task — it returns markdown instructions plus any bundled resource file names (which can then be read directly from the skill directory via the read tool). The skill's instructions may change HOW you approach the task, so read the full body, not just the description.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill name as printed by list_skills"},
                    "include_resources": {"type": "boolean", "description": "Also return bundled resource file contents (default true)"},
                    "recursive": {"type": "boolean", "description": "Also load depends_on skills the skill composes (default true)"}
                },
                "required": ["name"]
            }),
        },
ToolSpec {
            name: "read_document".into(),
            description: "Extract text from binary/office documents for the model to act on: PDF, DOCX, XLSX (each worksheet as a row grid), PPTX (slides). Also handles plain-text formats via the read tool. max_chars caps returned text (default 20000). Use instead of read for .pdf/.docx/.pptx/.xlsx files — read would return binary garbage for those. Returns unsupported/missing files as errors. For scanned/image PDFs (no text layer) it errors and you should use read_image + the ui-design skill.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or project-relative path to the document"},
                    "max_chars": {"type": "integer", "description": "Cap on returned characters (default 100000)"}
                },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "read_image".into(),
            description: "Read a local image file so a vision-capable model can SEE it (the binary image data is attached to the message). Supports PNG, JPEG, GIF, WEBP, BMP. Use for screenshots, UI mockups/design images, diagrams, scanned docs — anything you must inspect visually or recreate/design from. The companion text result states the resolved path and dimensions hint if known. For scanned PDFs (no text layer) pair with the ui-design + document-reading skills.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or project-relative path to the image file"}
                },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "understand_repo".into(),
            description: "Repository understanding: returns a deterministic snapshot of the project (language stack, frameworks, package manager, database, entry points, build/test commands, git status) plus — when a `topic` is given (e.g. \"authentication\") — a list of existing files/modules whose names relate to that topic. Read this or a targeted grep BEFORE writing new code, so you reuse existing modules instead of duplicating them.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Optional subject to find existing related code for"}
                }
            }),
        },
        ToolSpec {
            name: "memory".into(),
            description: "Long-term project memory under .agent/memory/: `list` shows note names + first lines; `read <name>` returns a note's full body. Used to persist decisions, conventions, and gotchas across sessions. Read before large/unknown tasks; check what the project already decided before exploring anew.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "read"]},
                    "name": {"type": "string", "description": "note name (required for read)"}
                },
                "required": ["action"]
            }),
        },
        ToolSpec {
            name: "memory_write".into(),
            description: "Write a long-term project memory note under .agent/memory/<name>.md (name: letters/digits/-/_). Content is a short markdown plan/decision/gotcha you want to persist across sessions. Overwrites the note if it exists. Ask the user first before writing non-obvious memories.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["name", "content"]
            }),
        },
        ToolSpec {
            name: "device".into(),
            description: "Test on an Android device via adb — over USB debugging or wireless (adb connect). Actions: devices (list USB+wireless), connect <host:port> (wireless debug), disconnect <host:port>, install <apk_path>, uninstall <package>, launch <package> [activity] (start the app), screenshot [out] (PNG into the project), screenrecord [out] [seconds] (MP4 screen capture, 1-30s, default 10), logcat [filter] [max_lines] (bounded crash/console dump), logcat_clear (reset the buffer), shell <command> (arbitrary device shell — the escape hatch), pair <host_port> <code> (wireless pairing), info (model / Android version / SDK), reverse [local_port] [device_port] (expose a host port on the device — needed for app/webview debugging), forward [local_port] [device_port] (expose a device port on the host), input <event> (UI automation: tap/swipe/keyevent/type), pull <remote> [out] (copy a file off the device), push <out> <remote> (copy a file onto the device). Requires the Android platform-tools `adb` on PATH and a device authorized for debugging.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["devices", "connect", "disconnect", "install", "uninstall", "launch", "screenshot", "screenrecord", "logcat", "logcat_clear", "shell", "pair", "info", "reverse", "forward", "input", "pull", "push"]},
                    "target": {"type": "string", "description": "host:port for connect/disconnect"},
                    "path": {"type": "string", "description": "APK path for install"},
                    "package": {"type": "string", "description": "app package for uninstall/launch"},
                    "activity": {"type": "string", "description": "optional activity (relative or fully-qualified) for launch"},
                    "command": {"type": "string", "description": "device shell command for action=shell"},
                    "filter": {"type": "string", "description": "logcat filter for action=logcat"},
                    "max_lines": {"type": "integer", "description": "logcat tail length (default 200)"},
                    "out": {"type": "string", "description": "output path relative to project root (screenshot/screenrecord/pull) or local file to push"},
                    "seconds": {"type": "integer", "description": "screenrecord duration in seconds (1-30, default 10)"},
                    "host_port": {"type": "string", "description": "host:port for wireless pairing"},
                    "code": {"type": "string", "description": "6-digit pairing code for action=pair"},
                    "local_port": {"type": "integer", "description": "host-side port for reverse/forward"},
                    "device_port": {"type": "integer", "description": "device-side port for reverse/forward"},
                    "event": {"type": "string", "description": "input event for action=input, e.g. 'tap 540 1200' or 'swipe 100 500 300 500 200' or 'text hello' or 'keyevent 4'"},
                    "remote": {"type": "string", "description": "device path for pull/push"}
                },
                "required": ["action"]
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
        // --- Phase 6: Code Intelligence (database-free symbol index) ---
        ToolSpec {
            name: "code_index".into(),
            description: "Scan the project's source files and write .agent/index.json (symbol index). Run before code_symbols/code_defs when no index exists yet.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "force": {"type": "boolean"} }
            }),
        },
        ToolSpec {
            name: "code_symbols".into(),
            description: "Look up symbols (functions/structs/classes/enums/...) in the project index by name (substring, case-insensitive). Returns kind, file, line.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "name": {"type": "string"} },
                "required": ["name"]
            }),
        },
        ToolSpec {
            name: "code_defs".into(),
            description: "Go-to-definition: same as code_symbols but reports the matching definitions only, suitable for 'where is X defined?'.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "name": {"type": "string"} },
                "required": ["name"]
            }),
        },
        ToolSpec {
            name: "code_refs".into(),
            description: "Find references to a symbol across the project (and configured extra project roots) via ripgrep. Word-boundary matching, file:line:text output.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "max": {"type": "integer"}
                },
                "required": ["name"]
            }),
        },
        ToolSpec {
            name: "code_rename".into(),
            description: "Propose a reference-update plan for renaming symbol `old` to `new` (word-boundary). Reports each file and the affected lines. It never writes — applying the edits is left to a separate review step.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "old": {"type": "string"},
                    "new": {"type": "string"}
                },
                "required": ["old", "new"]
            }),
        },
    ];
    specs.extend(platform_tool_specs());
    specs
}

/// Tool specs for the platform-CLI integrations (gh/supabase/vercel/aws/…).
/// Kept separate so the file stays navigable. Names must match the
/// `do_platform` dispatch arms exactly.
pub fn platform_tool_specs() -> Vec<ToolSpec> {
    vec![
        // --- GitHub ---
        ToolSpec {
            name: "gh_issue_list".into(),
            description: "List GitHub issues (state=open/closed).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "state": {"type": "string"},
                    "limit": {"type": "integer"},
                    "label": {"type": "string"}
                }
            }),
        },
        ToolSpec {
            name: "gh_issue_view".into(),
            description: "View a single GitHub issue.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "number": {"type": "string"} },
                "required": ["number"]
            }),
        },
        ToolSpec {
            name: "gh_issue_create".into(),
            description: "Create a GitHub issue (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "label": {"type": "string"}
                },
                "required": ["title"]
            }),
        },
        ToolSpec {
            name: "gh_issue_close".into(),
            description: "Close a GitHub issue (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": { "number": {"type": "string"} },
                "required": ["number"]
            }),
        },
        ToolSpec {
            name: "gh_pr_list".into(),
            description: "List GitHub pull requests (state=open/closed).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "state": {"type": "string"},
                    "limit": {"type": "integer"}
                }
            }),
        },
        ToolSpec {
            name: "gh_pr_view".into(),
            description: "View a single GitHub pull request.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "number": {"type": "string"} },
                "required": ["number"]
            }),
        },
        ToolSpec {
            name: "gh_pr_create".into(),
            description: "Create a GitHub pull request (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "base": {"type": "string"}
                },
                "required": ["title"]
            }),
        },
        ToolSpec {
            name: "gh_pr_merge".into(),
            description: "Merge a GitHub pull request (requires approval). method=merge/squash/rebase.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "number": {"type": "string"},
                    "method": {"type": "string"},
                    "delete_branch": {"type": "boolean"}
                },
                "required": ["number"]
            }),
        },
        ToolSpec {
            name: "gh_release_list".into(),
            description: "List GitHub releases.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "limit": {"type": "integer"} }
            }),
        },
        ToolSpec {
            name: "gh_release_create".into(),
            description: "Create a GitHub release (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "tag": {"type": "string"},
                    "title": {"type": "string"},
                    "notes": {"type": "string"}
                },
                "required": ["tag"]
            }),
        },
        ToolSpec {
            name: "gh_workflow_list".into(),
            description: "List GitHub Actions workflows.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "gh_workflow_run".into(),
            description: "Trigger a GitHub Actions workflow (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "workflow": {"type": "string"},
                    "ref": {"type": "string"}
                },
                "required": ["workflow"]
            }),
        },
        ToolSpec {
            name: "gh_run_list".into(),
            description: "List GitHub Actions runs.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "workflow": {"type": "string"},
                    "limit": {"type": "integer"}
                }
            }),
        },
        // --- Supabase ---
        ToolSpec {
            name: "supabase_login".into(),
            description: "Log in to Supabase (opens browser, requires approval).".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "supabase_link".into(),
            description: "Link the project to a Supabase remote (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": { "project_ref": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "supabase_projects_list".into(),
            description: "List Supabase projects.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "supabase_status".into(),
            description: "Show local Supabase dev service status.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "supabase_db_push".into(),
            description: "Push local migrations to the linked remote database (requires approval).".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "supabase_db_diff".into(),
            description: "Generate a DB diff against the linked remote.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "schema": {"type": "string"},
                    "linked": {"type": "boolean"}
                }
            }),
        },
        ToolSpec {
            name: "supabase_functions_list".into(),
            description: "List Supabase Edge Functions.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "supabase_functions_deploy".into(),
            description: "Deploy a Supabase Edge Function (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "function": {"type": "string"},
                    "project_ref": {"type": "string"},
                    "no_verify_jwt": {"type": "boolean"}
                },
                "required": ["function"]
            }),
        },
        // --- Vercel ---
        ToolSpec {
            name: "vercel_whoami".into(),
            description: "Show the logged-in Vercel user.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "vercel_projects_list".into(),
            description: "List Vercel projects.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "vercel_env_list".into(),
            description: "List Vercel environment variables.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "env": {"type": "string"},
                    "project": {"type": "string"}
                }
            }),
        },
        ToolSpec {
            name: "vercel_deploy".into(),
            description: "Deploy to Vercel (requires approval). prod=true deploys to production.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "prod": {"type": "boolean"},
                    "target": {"type": "string"},
                    "project": {"type": "string"}
                }
            }),
        },
        ToolSpec {
            name: "vercel_logs".into(),
            description: "Show Vercel deployment logs.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "deployment": {"type": "string"},
                    "project": {"type": "string"},
                    "follow": {"type": "boolean"}
                }
            }),
        },
        // --- Docker ---
        ToolSpec {
            name: "docker_ps".into(),
            description: "List docker containers (all=true includes stopped).".into(),
            parameters: json!({
                "type": "object",
                "properties": { "all": {"type": "boolean"} }
            }),
        },
        ToolSpec {
            name: "docker_images".into(),
            description: "List docker images.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "docker_compose_up".into(),
            description: "docker compose up (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "services": {"type": "array", "items": {"type": "string"}},
                    "detached": {"type": "boolean"},
                    "build": {"type": "boolean"}
                }
            }),
        },
        ToolSpec {
            name: "docker_compose_down".into(),
            description: "docker compose down (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": { "volumes": {"type": "boolean"} }
            }),
        },
        ToolSpec {
            name: "docker_compose_logs".into(),
            description: "docker compose logs.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "follow": {"type": "boolean"}
                }
            }),
        },
        // --- Kubernetes ---
        ToolSpec {
            name: "k8s_get".into(),
            description: "kubectl get resources (pods/services/deployments/…).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "resource": {"type": "string"},
                    "name": {"type": "string"},
                    "namespace": {"type": "string"},
                    "all_namespaces": {"type": "boolean"}
                },
                "required": ["resource"]
            }),
        },
        ToolSpec {
            name: "k8s_logs".into(),
            description: "kubectl logs for a pod.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pod": {"type": "string"},
                    "container": {"type": "string"},
                    "namespace": {"type": "string"},
                    "follow": {"type": "boolean"}
                },
                "required": ["pod"]
            }),
        },
        ToolSpec {
            name: "k8s_apply".into(),
            description: "kubectl apply -f a manifest (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "namespace": {"type": "string"}
                },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "k8s_rollout_status".into(),
            description: "kubectl rollout status for a deployment.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "resource": {"type": "string"},
                    "namespace": {"type": "string"}
                },
                "required": ["resource"]
            }),
        },
        // --- Terraform ---
        ToolSpec {
            name: "tf_init".into(),
            description: "terraform init (requires approval).".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "tf_validate".into(),
            description: "terraform validate.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "tf_plan".into(),
            description: "terraform plan (optionally -out=<file>).".into(),
            parameters: json!({
                "type": "object",
                "properties": { "out": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "tf_apply".into(),
            description: "terraform apply (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "plan_file": {"type": "string"},
                    "auto_approve": {"type": "boolean"}
                }
            }),
        },
        // --- CircleCI ---
        ToolSpec {
            name: "circleci_validate".into(),
            description: "circleci config validate.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "config": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "circleci_builds".into(),
            description: "List CircleCI builds for a project.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "project": {"type": "string"},
                    "branch": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["project"]
            }),
        },
        // --- AWS ---
        ToolSpec {
            name: "aws_whoami".into(),
            description: "Show the active AWS identity (sts get-caller-identity).".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "aws_s3_ls".into(),
            description: "List S3 buckets or objects under a prefix.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "path": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "aws_s3_sync".into(),
            description: "Sync files to/from S3 (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "dest": {"type": "string"}
                },
                "required": ["source", "dest"]
            }),
        },
        ToolSpec {
            name: "aws_ecr_login".into(),
            description: "Print an ECR docker login token (requires approval).".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "aws_lambda_list".into(),
            description: "List AWS Lambda functions.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "aws_lambda_invoke".into(),
            description: "Invoke an AWS Lambda function (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "function": {"type": "string"},
                    "payload": {"type": "string"}
                },
                "required": ["function"]
            }),
        },
        ToolSpec {
            name: "aws_ecs_list_clusters".into(),
            description: "List ECS clusters.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "aws_ecs_force_deploy".into(),
            description: "Force a new deployment of an ECS service (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "cluster": {"type": "string"},
                    "service": {"type": "string"}
                },
                "required": ["cluster", "service"]
            }),
        },
        // --- AWS SAM / CloudFormation ---
        ToolSpec {
            name: "sam_build".into(),
            description: "sam build (requires approval).".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "sam_deploy".into(),
            description: "sam deploy (requires approval). guided=true for interactive prompts.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "guided": {"type": "boolean"},
                    "stack_name": {"type": "string"}
                }
            }),
        },
        ToolSpec {
            name: "cloudformation_describe".into(),
            description: "aws cloudformation describe-stacks for a stack.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "stack": {"type": "string"} },
                "required": ["stack"]
            }),
        },
        ToolSpec {
            name: "cloudformation_deploy".into(),
            description: "aws cloudformation deploy a template to a stack (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "template": {"type": "string"},
                    "stack": {"type": "string"}
                },
                "required": ["template", "stack"]
            }),
        },
        // --- Azure ---
        ToolSpec {
            name: "az_whoami".into(),
            description: "Show the active Azure account.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "az_webapp_list".into(),
            description: "List Azure App Service web apps.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "az_webapp_deploy".into(),
            description: "Deploy to an Azure web app (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "resource_group": {"type": "string"},
                    "source": {"type": "string"}
                },
                "required": ["name", "resource_group", "source"]
            }),
        },
        ToolSpec {
            name: "az_functionapp_deploy".into(),
            description: "Deploy an Azure Functions app (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "resource_group": {"type": "string"},
                    "source": {"type": "string"}
                },
                "required": ["name", "resource_group", "source"]
            }),
        },
        // --- Google Cloud ---
        ToolSpec {
            name: "gcloud_whoami".into(),
            description: "Show the active gcloud config/account.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "gcloud_app_deploy".into(),
            description: "Deploy to Google App Engine (requires approval).".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "gcloud_run_deploy".into(),
            description: "Deploy a container image to Cloud Run (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "image": {"type": "string"},
                    "region": {"type": "string"}
                },
                "required": ["service", "image"]
            }),
        },
        ToolSpec {
            name: "gcloud_run_services".into(),
            description: "List Cloud Run services.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        // --- Helm ---
        ToolSpec {
            name: "helm_list".into(),
            description: "helm list releases.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "namespace": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "helm_status".into(),
            description: "helm status for a release.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "release": {"type": "string"},
                    "namespace": {"type": "string"}
                },
                "required": ["release"]
            }),
        },
        ToolSpec {
            name: "helm_install".into(),
            description: "helm install a chart (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "release": {"type": "string"},
                    "chart": {"type": "string"},
                    "namespace": {"type": "string"}
                },
                "required": ["release", "chart"]
            }),
        },
        ToolSpec {
            name: "helm_upgrade".into(),
            description: "helm upgrade a release (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "release": {"type": "string"},
                    "chart": {"type": "string"},
                    "namespace": {"type": "string"}
                },
                "required": ["release", "chart"]
            }),
        },
        ToolSpec {
            name: "helm_uninstall".into(),
            description: "helm uninstall a release (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "release": {"type": "string"},
                    "namespace": {"type": "string"}
                },
                "required": ["release"]
            }),
        },
        // --- Fly.io ---
        ToolSpec {
            name: "fly_whoami".into(),
            description: "Show the logged-in Fly.io user.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "fly_apps_list".into(),
            description: "List Fly.io apps.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "fly_deploy".into(),
            description: "Deploy to Fly.io (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                    "app": {"type": "string"}
                }
            }),
        },
        ToolSpec {
            name: "fly_status".into(),
            description: "Show Fly.io app status.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "app": {"type": "string"} },
                "required": ["app"]
            }),
        },
        // --- Railway ---
        ToolSpec {
            name: "railway_whoami".into(),
            description: "Show the logged-in Railway user.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "railway_status".into(),
            description: "Show Railway project status.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "railway_up".into(),
            description: "Deploy to Railway (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": { "detach": {"type": "boolean"} }
            }),
        },
        // --- Render ---
        ToolSpec {
            name: "render_whoami".into(),
            description: "Show the logged-in Render user.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "render_services".into(),
            description: "List Render services.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "render_deploy".into(),
            description: "Trigger a deploy for a Render service (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": { "service_id": {"type": "string"} },
                "required": ["service_id"]
            }),
        },
        // --- Netlify ---
        ToolSpec {
            name: "netlify_whoami".into(),
            description: "Show the logged-in Netlify user.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "netlify_sites".into(),
            description: "List Netlify sites.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "netlify_deploy".into(),
            description: "Deploy to Netlify (requires approval). prod=true deploys to production.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dir": {"type": "string"},
                    "prod": {"type": "boolean"},
                    "site": {"type": "string"}
                },
                "required": ["dir"]
            }),
        },
        // --- Firebase ---
        ToolSpec {
            name: "firebase_projects".into(),
            description: "List Firebase projects.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "firebase_deploy".into(),
            description: "Deploy to Firebase Hosting / Functions (requires approval).".into(),
            parameters: json!({
                "type": "object",
                "properties": { "only": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "firebase_functions".into(),
            description: "List Firebase Functions.".into(),
            parameters: json!({ "type": "object", "properties": {} }),
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
            | "web_fetch"
            | "web_search"
            |             "list_skills"
            | "read_skill"
|             "read_document"
            | "read_image"
            | "understand_repo"
            | "memory"
            | "code_index"
            | "code_symbols"
            | "code_defs"
            | "code_refs"
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
        }
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
            // for the model to self-correct. Anything else (`Provider`/
            // `Fs`/`Terminal`/`Session`/`Io`) is a real system failure a
            // retry can't fix, and still aborts the turn as before.
            Err(e @ (AgentError::InvalidArguments { .. } | AgentError::UnknownTool(_))) => {
                ToolResult::err(e.to_string())
            }
            Err(e) => return Err(e),
        };

        Ok(match self.hooks.run_post_tool_use(
            name,
            arguments,
            &result.content,
            result.is_error,
        ) {
            Some(extra) => ToolResult {
                content: format!("{}\n\n[post-tool-use hook output]\n{extra}", result.content),
                is_error: result.is_error,
                images: result.images,
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
            "todowrite" => self.do_todowrite(&args),
            "read" => self.do_read(&args),
            "write" => self.do_write(&args, approver),
            "edit" => self.do_edit(&args, approver),
            "delete" => self.do_delete(&args, approver),
            "rename" => self.do_rename(&args, approver),
            "copy" => self.do_copy(&args, approver),
            "grep" => self.do_grep(&args),
            "glob" => self.do_glob(&args),
            "code_index" => self.do_code_index(&args),
            "code_symbols" => self.do_code_symbols(&args),
            "code_defs" => self.do_code_defs(&args),
            "code_refs" => self.do_code_refs(&args),
            "code_rename" => self.do_code_rename(&args),
            "bash" => self.do_bash(&args, approver),
            "test" => self.do_test(&args, approver),
            "browser" => self.do_browser(&args),
            "web_fetch" => self.do_web_fetch(&args),
            "web_search" => self.do_web_search(&args),
            "list_skills" => self.do_list_skills(&args),
            "read_skill" => self.do_read_skill(&args),
            "read_document" => self.do_read_document(&args),
            "read_image" => self.do_read_image(&args),
            "understand_repo" => self.do_understand_repo(&args),
            "memory" => self.do_memory(&args),
            "memory_write" => self.do_memory_write(&args, approver),
            "device" => self.do_device(&args, approver),
            "bg_list" => self.do_bg_list(),
            "bg_output" => self.do_bg_output(&args),
            "bg_pause" => self.do_bg_pause(&args),
            "bg_resume" => self.do_bg_resume(&args),
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
            "gh_issue_list" => self.do_platform("gh_issue_list", &args, approver),
            "gh_issue_view" => self.do_platform("gh_issue_view", &args, approver),
            "gh_issue_create" => self.do_platform("gh_issue_create", &args, approver),
            "gh_issue_close" => self.do_platform("gh_issue_close", &args, approver),
            "gh_pr_list" => self.do_platform("gh_pr_list", &args, approver),
            "gh_pr_view" => self.do_platform("gh_pr_view", &args, approver),
            "gh_pr_create" => self.do_platform("gh_pr_create", &args, approver),
            "gh_pr_merge" => self.do_platform("gh_pr_merge", &args, approver),
            "gh_release_list" => self.do_platform("gh_release_list", &args, approver),
            "gh_release_create" => self.do_platform("gh_release_create", &args, approver),
            "gh_workflow_list" => self.do_platform("gh_workflow_list", &args, approver),
            "gh_workflow_run" => self.do_platform("gh_workflow_run", &args, approver),
            "gh_run_list" => self.do_platform("gh_run_list", &args, approver),
            "supabase_login" => self.do_platform("supabase_login", &args, approver),
            "supabase_link" => self.do_platform("supabase_link", &args, approver),
            "supabase_projects_list" => self.do_platform("supabase_projects_list", &args, approver),
            "supabase_status" => self.do_platform("supabase_status", &args, approver),
            "supabase_db_push" => self.do_platform("supabase_db_push", &args, approver),
            "supabase_db_diff" => self.do_platform("supabase_db_diff", &args, approver),
            "supabase_functions_list" => self.do_platform("supabase_functions_list", &args, approver),
            "supabase_functions_deploy" => self.do_platform("supabase_functions_deploy", &args, approver),
            "vercel_whoami" => self.do_platform("vercel_whoami", &args, approver),
            "vercel_projects_list" => self.do_platform("vercel_projects_list", &args, approver),
            "vercel_env_list" => self.do_platform("vercel_env_list", &args, approver),
            "vercel_deploy" => self.do_platform("vercel_deploy", &args, approver),
            "vercel_logs" => self.do_platform("vercel_logs", &args, approver),
            "docker_ps" => self.do_platform("docker_ps", &args, approver),
            "docker_images" => self.do_platform("docker_images", &args, approver),
            "docker_compose_up" => self.do_platform("docker_compose_up", &args, approver),
            "docker_compose_down" => self.do_platform("docker_compose_down", &args, approver),
            "docker_compose_logs" => self.do_platform("docker_compose_logs", &args, approver),
            "k8s_get" => self.do_platform("k8s_get", &args, approver),
            "k8s_logs" => self.do_platform("k8s_logs", &args, approver),
            "k8s_apply" => self.do_platform("k8s_apply", &args, approver),
            "k8s_rollout_status" => self.do_platform("k8s_rollout_status", &args, approver),
            "tf_init" => self.do_platform("tf_init", &args, approver),
            "tf_validate" => self.do_platform("tf_validate", &args, approver),
            "tf_plan" => self.do_platform("tf_plan", &args, approver),
            "tf_apply" => self.do_platform("tf_apply", &args, approver),
            "circleci_validate" => self.do_platform("circleci_validate", &args, approver),
            "circleci_builds" => self.do_platform("circleci_builds", &args, approver),
            "aws_whoami" => self.do_platform("aws_whoami", &args, approver),
            "aws_s3_ls" => self.do_platform("aws_s3_ls", &args, approver),
            "aws_s3_sync" => self.do_platform("aws_s3_sync", &args, approver),
            "aws_ecr_login" => self.do_platform("aws_ecr_login", &args, approver),
            "aws_lambda_list" => self.do_platform("aws_lambda_list", &args, approver),
            "aws_lambda_invoke" => self.do_platform("aws_lambda_invoke", &args, approver),
            "aws_ecs_list_clusters" => self.do_platform("aws_ecs_list_clusters", &args, approver),
            "aws_ecs_force_deploy" => self.do_platform("aws_ecs_force_deploy", &args, approver),
            "sam_build" => self.do_platform("sam_build", &args, approver),
            "sam_deploy" => self.do_platform("sam_deploy", &args, approver),
            "cloudformation_describe" => self.do_platform("cloudformation_describe", &args, approver),
            "cloudformation_deploy" => self.do_platform("cloudformation_deploy", &args, approver),
            "az_whoami" => self.do_platform("az_whoami", &args, approver),
            "az_webapp_list" => self.do_platform("az_webapp_list", &args, approver),
            "az_webapp_deploy" => self.do_platform("az_webapp_deploy", &args, approver),
            "az_functionapp_deploy" => self.do_platform("az_functionapp_deploy", &args, approver),
            "gcloud_whoami" => self.do_platform("gcloud_whoami", &args, approver),
            "gcloud_app_deploy" => self.do_platform("gcloud_app_deploy", &args, approver),
            "gcloud_run_deploy" => self.do_platform("gcloud_run_deploy", &args, approver),
            "gcloud_run_services" => self.do_platform("gcloud_run_services", &args, approver),
            "helm_list" => self.do_platform("helm_list", &args, approver),
            "helm_status" => self.do_platform("helm_status", &args, approver),
            "helm_install" => self.do_platform("helm_install", &args, approver),
            "helm_upgrade" => self.do_platform("helm_upgrade", &args, approver),
            "helm_uninstall" => self.do_platform("helm_uninstall", &args, approver),
            "fly_whoami" => self.do_platform("fly_whoami", &args, approver),
            "fly_apps_list" => self.do_platform("fly_apps_list", &args, approver),
            "fly_deploy" => self.do_platform("fly_deploy", &args, approver),
            "fly_status" => self.do_platform("fly_status", &args, approver),
            "railway_whoami" => self.do_platform("railway_whoami", &args, approver),
            "railway_status" => self.do_platform("railway_status", &args, approver),
            "railway_up" => self.do_platform("railway_up", &args, approver),
            "render_whoami" => self.do_platform("render_whoami", &args, approver),
            "render_services" => self.do_platform("render_services", &args, approver),
            "render_deploy" => self.do_platform("render_deploy", &args, approver),
            "netlify_whoami" => self.do_platform("netlify_whoami", &args, approver),
            "netlify_sites" => self.do_platform("netlify_sites", &args, approver),
            "netlify_deploy" => self.do_platform("netlify_deploy", &args, approver),
            "firebase_projects" => self.do_platform("firebase_projects", &args, approver),
            "firebase_deploy" => self.do_platform("firebase_deploy", &args, approver),
            "firebase_functions" => self.do_platform("firebase_functions", &args, approver),
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
                images: Vec::new(),
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

    fn opt_str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
        args.get(key).and_then(|v| v.as_str())
    }

    fn usize_arg(args: &Value, key: &str) -> Option<usize> {
        args.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    fn opt_bool_arg(args: &Value, key: &str) -> Option<bool> {
        args.get(key).and_then(|v| v.as_bool())
    }

    fn str_array_arg(args: &Value, key: &str) -> Vec<String> {
        args.get(key)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default()
    }

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
            if t.get("content").and_then(|v| v.as_str()).is_none_or(str::is_empty) {
                return Ok(ToolResult::err("each todo needs a non-empty \"content\" string"));
            }
            match t.get("status").and_then(|v| v.as_str()) {
                Some("pending" | "in_progress" | "completed") => {}
                _ => return Ok(ToolResult::err(
                    "each todo needs \"status\" to be one of pending/in_progress/completed",
                )),
            }
        }
        let done = todos
            .iter()
            .filter(|t| t.get("status").and_then(|v| v.as_str()) == Some("completed"))
            .count();
        Ok(ToolResult::ok(format!("{done}/{} done", todos.len())))
    }

    fn do_read(&self, args: &Value) -> Result<ToolResult> {
        let path = Self::str_arg(args, "path")?;
        let offset = args.get("offset").and_then(|v| v.as_u64()).map(|v| v as usize);
        let limit = args.get("limit").and_then(|v| v.as_u64()).map(|v| v as usize);
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

    fn do_code_index(&self, args: &Value) -> Result<ToolResult> {
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
                let summary = summarize_test_output(&out.stdout);
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

    /// Open a URL in the default browser for visual verification of a
    /// running app. Launch-and-forget: spawns the platform opener and
    /// returns immediately — the browser window stays open on the user's
    /// machine while the agent keeps talking to them about what they see.
    fn do_browser(&self, args: &Value) -> Result<ToolResult> {
        let url = Self::str_arg(args, "url")?;
        let url = url.trim();
        match open_browser_url(url) {
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
        // Reject inner-IP/loopback targets just like `browser` does, so the
        // tool can't be pointed at the user's local services.
        if let Some(reason) = reject_web_target(url) {
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
        if content_type.contains("json") || content_type.contains("text") || content_type.contains("xml") {
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
            strip_html(&body)
        } else {
            body
        };
        if content.chars().count() > max_chars {
            content = content.chars().take(max_chars).collect::<String>();
            content.push_str("\n… (truncated, max_chars reached)");
        }
        Ok(ToolResult::ok(format!(
            "# web_fetch {url}\n{content}"
        )))
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
            urlencode(&query)
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

        // DuckDuckGo HTML results: <a class="result__a" href="URL">title</a>
        // and <a class="result__snippet" ...>snippet</a>.
        let mut results: Vec<(String, String, String)> = Vec::new();
        for (_i, chunk) in html.split("result__a").skip(1).enumerate() {
            if results.len() >= max_results {
                break;
            }
            let Some(href_start) = chunk.find("href=\"") else { continue };
            let url = &chunk[href_start + 6..chunk[href_start + 6..].find('"').map(|i| i + href_start + 6).unwrap_or(href_start + 6)];
            let Some(title_end) = chunk.find("</a>") else { continue };
            let title = strip_html(&chunk[..title_end]);
            let snippet = chunk
                .find("result__snippet")
                .and_then(|s| {
                    let seg = &chunk[s..];
                    let o = seg.find(">").map(|o| o + 1);
                    o.map(|o| seg[o..seg.len().min(o + 400)].to_string())
                })
                .map(|s| strip_html(&s))
                .unwrap_or_default();
            let url_clean = url.trim_start_matches("//").to_string();
            results.push((title.trim().to_string(), url_clean, snippet.trim().to_string()));
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
                if snippet.is_empty() { "(no snippet)".to_string() } else { snippet.clone() }
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
                SkillTier::Builtin => vec![], // built-ins are registered below
            };
            for skill in candidates {
                // Higher tiers already inserted win; lower tiers don't overwrite.
                by_name.entry(skill.name.clone()).or_insert(skill);
            }
        }
        // Built-in skills ship as static data, always last in precedence.
        for def in BUILTIN_SKILLS {
            by_name
                .entry(def.0.to_string())
                .or_insert_with(|| builtin_skill(def));
        }
        by_name.into_values().collect()
    }

    /// `list_skills` — the model's browseable catalog of available skills.
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
            let hay = format!(
                "{} {} {:?}",
                skill.name, skill.description, skill.tags
            )
            .to_lowercase();
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

    /// `read_skill` — load a skill's SKILL.md body (+ bundled resources),
    /// and optionally its `depends_on` chain so one call can compose a whole
    /// workflow (e.g. database → backend → frontend → security → testing).
    fn do_read_skill(&self, args: &Value) -> Result<ToolResult> {
        let name = Self::str_arg(args, "name")?.to_lowercase();
        use crate::skills::{read_skill_resource, skill_resources};
        let include_resources = Self::opt_bool_arg(args, "include_resources").unwrap_or(true);
        let recursive = Self::opt_bool_arg(args, "recursive").unwrap_or(true);
        let all = self.all_skills();
        // Resolve the skill plus its dependency closure (BFS, cycle-safe).
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

    /// `read_document` — extract text from office/binaries so the model can
    /// read specs, reports, spreadsheets and slide decks.
    fn do_read_document(&self, args: &Value) -> Result<ToolResult> {
        let path = Self::str_arg(args, "path")?;
        let max_chars = Self::usize_arg(args, "max_chars").unwrap_or(100_000).max(1000);
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

    /// `read_image` — expose a local image's bytes to a vision-capable model.
    /// The binary data rides along on the ToolResult so the agent loop can
    /// attach it as a multimodal image part on the next request.
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
            Err(e) => return Ok(ToolResult::err(format!("could not read {}: {e}", resolved.display()))),
        };
        // Only raster formats are safe to hand to a vision model.
        let mime = image_mime(&resolved);
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

    /// `understand_repo` — deterministic project understanding + (optionally)
    /// existing files relevant to a subject. No model call; the fingerprint
    /// is cached on the agent and shared so repeated calls are cheap.
    fn do_understand_repo(&self, args: &Value) -> Result<ToolResult> {
        let topic = Self::str_arg(args, "topic").unwrap_or_default();
        let root = self.project_root();
        let fp = match &self.repo {
            Some(fp) => fp.clone(),
            None => crate::project::load_or_analyze(&root),
        };
        let text = if topic.trim().is_empty() {
            format!("Repository understanding:\n{}", fp.banner_lines().join("\n"))
        } else {
            fp.render(&topic)
        };
        Ok(ToolResult::ok(text))
    }

    /// `memory` — list or read a long-term project memory note.
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
                    None => Ok(ToolResult::err(format!(
                        "no memory note named `{name}`"
                    ))),
                }
            }
            other => Ok(ToolResult::err(format!(
                "unknown memory action `{other}` (expected list|read)"
            ))),
        }
    }

    /// `memory_write` — persist a long-term project memory note.
    fn do_memory_write<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let name = Self::str_arg(args, "name")?;
        let content = Self::str_arg(args, "content")?.to_string();
        let path_name = match crate::project::safe_memory_name(name) {
            Some(safe) => safe,
            None => {
                return Ok(ToolResult::err(
                    "invalid memory name (letters, digits, `-`, `_`)",
                ))
            }
        };
        let rel = format!(".agent/memory/{path_name}.md");
        match self.workspace.files.write(
            Path::new(&rel),
            &content,
            WriteOptions::default(),
            &mut *approver,
        ) {
            Ok(()) => Ok(ToolResult::ok(format!("wrote .agent/memory/{path_name}.md"))),
            Err(e) => Ok(ToolResult::err(e.to_string())),
        }
    }

    /// Drive an attached Android device/emulator through `adb` — USB or
    /// wireless. Individual operations (list/connect/install/launch/logcat/
    /// screenshot/shell) are implemented in `DeviceEngine`; this layer parses
    /// the tool arguments and formats the result for the model.
    fn do_device<F>(&self, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let action = Self::str_arg(args, "action")?
            .to_ascii_lowercase();
        let device = &self.device;
        // A no-op enum check so unknown actions are rejected before any adb
        // call (and before the permission prompt).
        if !matches!(
            action.as_str(),
            "devices" | "connect" | "disconnect" | "install" | "uninstall" | "launch" | "screenshot"
                | "screenrecord" | "logcat" | "logcat_clear" | "shell" | "pair" | "info" | "reverse"
                | "forward" | "input" | "pull" | "push"
        ) {
            return Ok(ToolResult::err(format!(
                "unknown device action '{action}' — use one of: devices, connect, disconnect, install, uninstall, launch, screenshot, screenrecord, logcat, logcat_clear, shell, pair, info, reverse, forward, input, pull, push"
            )));
        }

        let opt_str = |key: &str| Self::opt_str_arg(args, key).map(|s| s.to_string());
        let req_str = |key: &str| {
            Self::str_arg(args, key).map(|s| s.to_string()).map_err(|_| {
                AgentError::InvalidArguments {
                    tool: "device".into(),
                    reason: format!("action '{action}' requires '{key}'"),
                }
            })
        };

        let result = match action.as_str() {
            "devices" => device.devices(&mut *approver),
            "connect" => device.connect(&req_str("target")?, &mut *approver),
            "disconnect" => device.disconnect(&req_str("target")?, &mut *approver),
            "install" => device.install(&req_str("path")?, &mut *approver),
            "uninstall" => device.uninstall(&req_str("package")?, &mut *approver),
            "launch" => device.launch(&req_str("package")?, opt_str("activity").as_deref(), &mut *approver),
            "screenshot" => device.screenshot(opt_str("out").as_deref(), &mut *approver),
            "screenrecord" => {
                let seconds = args.get("seconds").and_then(|v| v.as_u64()).unwrap_or(10) as u32;
                device.screenrecord(opt_str("out").as_deref(), seconds, &mut *approver)
            }
            "logcat" => {
                let max = args.get("max_lines").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
                device.logcat(opt_str("filter").as_deref(), max, &mut *approver)
            }
            "logcat_clear" => device.logcat_clear(&mut *approver),
            "shell" => device.shell(&req_str("command")?, &mut *approver),
            "pair" => device.pair(&req_str("host_port")?, &req_str("code")?, &mut *approver),
            "info" => device.info(&mut *approver),
            "reverse" => {
                let local = args.get("local_port").and_then(|v| v.as_u64()).map(|v| v as u32);
                let dev = args.get("device_port").and_then(|v| v.as_u64()).map(|v| v as u32);
                device.reverse(local, dev, &mut *approver)
            }
            "forward" => {
                let local = args.get("local_port").and_then(|v| v.as_u64()).map(|v| v as u32);
                let dev = args.get("device_port").and_then(|v| v.as_u64()).map(|v| v as u32);
                device.forward(local, dev, &mut *approver)
            }
            "input" => device.input(&req_str("event")?, &mut *approver),
            "pull" => device.pull(&req_str("remote")?, opt_str("out").as_deref(), &mut *approver),
            "push" => device.push(&req_str("out")?, &req_str("remote")?, &mut *approver),
            _ => unreachable!("validated above"),
        };

        match result {
            Ok(out) => Ok(device_result(out)),
            Err(e) => Ok(ToolResult::err(format!("device action '{action}' failed: {e}"))),
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
            Ok(()) => Ok(ToolResult::ok(format!(
                "resumed background task {id}"
            ))),
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

    /// Dispatch a platform-CLI tool (gh/supabase/vercel/docker/kubectl/
    /// terraform/circleci) to the matching `PlatformEngine` method. Arguments
    /// are read from the JSON tool args; read-only ops ignore the approver.
    fn do_platform<F>(&self, name: &str, args: &Value, approver: &mut F) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let count = |k: &str| Self::usize_arg(args, k).unwrap_or(20);
        let result = match name {
            "gh_issue_list" => {
                let label = Self::opt_str_arg(args, "label");
                let state = Self::opt_str_arg(args, "state").unwrap_or("open");
                platform_result(self.platform.gh_issue_list(state, count("limit"), label))
            }
            "gh_issue_view" => {
                let n = Self::str_arg(args, "number")?;
                platform_result(self.platform.gh_issue_view(n))
            }
            "gh_issue_create" => {
                let title = Self::str_arg(args, "title")?;
                let body = Self::opt_str_arg(args, "body");
                let label = Self::opt_str_arg(args, "label");
                platform_result(
                    self.platform
                        .gh_issue_create(title, body, label, &mut *approver),
                )
            }
            "gh_issue_close" => {
                let n = Self::str_arg(args, "number")?;
                platform_result(self.platform.gh_issue_close(n, &mut *approver))
            }
            "gh_pr_list" => {
                let state = Self::opt_str_arg(args, "state").unwrap_or("open");
                platform_result(self.platform.gh_pr_list(state, count("limit")))
            }
            "gh_pr_view" => {
                let n = Self::str_arg(args, "number")?;
                platform_result(self.platform.gh_pr_view(n))
            }
            "gh_pr_create" => {
                let title = Self::str_arg(args, "title")?;
                let body = Self::opt_str_arg(args, "body");
                let base = Self::opt_str_arg(args, "base");
                platform_result(
                    self.platform
                        .gh_pr_create(title, body, base, &mut *approver),
                )
            }
            "gh_pr_merge" => {
                let n = Self::str_arg(args, "number")?;
                let method = Self::opt_str_arg(args, "method");
                let del = Self::opt_bool_arg(args, "delete_branch").unwrap_or(false);
                platform_result(self.platform.gh_pr_merge(n, method, del, &mut *approver))
            }
            "gh_release_list" => platform_result(self.platform.gh_release_list(count("limit"))),
            "gh_release_create" => {
                let tag = Self::str_arg(args, "tag")?;
                let title = Self::opt_str_arg(args, "title");
                let notes = Self::opt_str_arg(args, "notes");
                platform_result(self.platform.gh_release_create(tag, title, notes, &mut *approver))
            }
            "gh_workflow_list" => platform_result(self.platform.gh_workflow_list()),
            "gh_workflow_run" => {
                let wf = Self::str_arg(args, "workflow")?;
                let r = Self::opt_str_arg(args, "ref");
                platform_result(self.platform.gh_workflow_run(wf, r, &mut *approver))
            }
            "gh_run_list" => {
                let wf = Self::opt_str_arg(args, "workflow");
                platform_result(self.platform.gh_run_list(wf, count("limit")))
            }
            "supabase_login" => platform_result(self.platform.supabase_login(&mut *approver)),
            "supabase_link" => {
                let pr = Self::opt_str_arg(args, "project_ref");
                platform_result(self.platform.supabase_link(pr, &mut *approver))
            }
            "supabase_projects_list" => platform_result(self.platform.supabase_projects_list()),
            "supabase_status" => platform_result(self.platform.supabase_status()),
            "supabase_db_push" => platform_result(self.platform.supabase_db_push(&mut *approver)),
            "supabase_db_diff" => {
                let schema = Self::opt_str_arg(args, "schema");
                let linked = Self::opt_bool_arg(args, "linked").unwrap_or(false);
                platform_result(self.platform.supabase_db_diff(schema, linked))
            }
            "supabase_functions_list" => platform_result(self.platform.supabase_functions_list()),
            "supabase_functions_deploy" => {
                let f = Self::str_arg(args, "function")?;
                let pr = Self::opt_str_arg(args, "project_ref");
                let nvj = Self::opt_bool_arg(args, "no_verify_jwt").unwrap_or(false);
                platform_result(
                    self.platform
                        .supabase_functions_deploy(f, pr, nvj, &mut *approver),
                )
            }
            "vercel_whoami" => platform_result(self.platform.vercel_whoami()),
            "vercel_projects_list" => platform_result(self.platform.vercel_projects_list()),
            "vercel_env_list" => {
                let env = Self::opt_str_arg(args, "env");
                let project = Self::opt_str_arg(args, "project");
                platform_result(self.platform.vercel_env_list(env, project))
            }
            "vercel_deploy" => {
                let prod = Self::opt_bool_arg(args, "prod").unwrap_or(false);
                let target = Self::opt_str_arg(args, "target");
                let project = Self::opt_str_arg(args, "project");
                platform_result(self.platform.vercel_deploy(prod, target, project, &mut *approver))
            }
            "vercel_logs" => {
                let dep = Self::opt_str_arg(args, "deployment");
                let project = Self::opt_str_arg(args, "project");
                let follow = Self::opt_bool_arg(args, "follow").unwrap_or(false);
                platform_result(self.platform.vercel_logs(dep, project, follow))
            }
            "docker_ps" => {
                let all = Self::opt_bool_arg(args, "all").unwrap_or(false);
                platform_result(self.platform.docker_ps(all))
            }
            "docker_images" => platform_result(self.platform.docker_images()),
            "docker_compose_up" => {
                let services = Self::str_array_arg(args, "services");
                let detached = Self::opt_bool_arg(args, "detached").unwrap_or(false);
                let build = Self::opt_bool_arg(args, "build").unwrap_or(false);
                platform_result(
                    self.platform
                        .docker_compose_up(services, detached, build, &mut *approver),
                )
            }
            "docker_compose_down" => {
                let volumes = Self::opt_bool_arg(args, "volumes").unwrap_or(false);
                platform_result(self.platform.docker_compose_down(volumes, &mut *approver))
            }
            "docker_compose_logs" => {
                let service = Self::opt_str_arg(args, "service");
                let follow = Self::opt_bool_arg(args, "follow").unwrap_or(false);
                platform_result(self.platform.docker_compose_logs(service, follow))
            }
            "k8s_get" => {
                let resource = Self::str_arg(args, "resource")?;
                let n = Self::opt_str_arg(args, "name");
                let ns = Self::opt_str_arg(args, "namespace");
                let an = Self::opt_bool_arg(args, "all_namespaces").unwrap_or(false);
                platform_result(self.platform.k8s_get(resource, n, ns, an))
            }
            "k8s_logs" => {
                let pod = Self::str_arg(args, "pod")?;
                let c = Self::opt_str_arg(args, "container");
                let ns = Self::opt_str_arg(args, "namespace");
                let follow = Self::opt_bool_arg(args, "follow").unwrap_or(false);
                platform_result(self.platform.k8s_logs(pod, c, ns, follow))
            }
            "k8s_apply" => {
                let path = Self::str_arg(args, "path")?;
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(self.platform.k8s_apply(path, ns, &mut *approver))
            }
            "k8s_rollout_status" => {
                let resource = Self::str_arg(args, "resource")?;
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(self.platform.k8s_rollout_status(resource, ns))
            }
            "tf_init" => platform_result(self.platform.tf_init(&mut *approver)),
            "tf_validate" => platform_result(self.platform.tf_validate()),
            "tf_plan" => {
                let out = Self::opt_str_arg(args, "out");
                platform_result(self.platform.tf_plan(out))
            }
            "tf_apply" => {
                let plan = Self::opt_str_arg(args, "plan_file");
                let aa = Self::opt_bool_arg(args, "auto_approve").unwrap_or(false);
                platform_result(self.platform.tf_apply(plan, aa, &mut *approver))
            }
            "circleci_validate" => {
                let cfg = Self::opt_str_arg(args, "config");
                platform_result(self.platform.circleci_validate(cfg))
            }
            "circleci_builds" => {
                let project = Self::str_arg(args, "project")?;
                let branch = Self::opt_str_arg(args, "branch");
                platform_result(self.platform.circleci_builds(project, branch, count("limit")))
            }
            "aws_whoami" => platform_result(self.platform.aws_whoami()),
            "aws_s3_ls" => {
                let path = Self::opt_str_arg(args, "path");
                platform_result(self.platform.aws_s3_ls(path))
            }
            "aws_s3_sync" => {
                let source = Self::str_arg(args, "source")?;
                let dest = Self::str_arg(args, "dest")?;
                platform_result(self.platform.aws_s3_sync(source, dest, &mut *approver))
            }
            "aws_ecr_login" => platform_result(self.platform.aws_ecr_login(&mut *approver)),
            "aws_lambda_list" => platform_result(self.platform.aws_lambda_list()),
            "aws_lambda_invoke" => {
                let function = Self::str_arg(args, "function")?;
                let payload = Self::opt_str_arg(args, "payload");
                platform_result(self.platform.aws_lambda_invoke(function, payload, &mut *approver))
            }
            "aws_ecs_list_clusters" => platform_result(self.platform.aws_ecs_list_clusters()),
            "aws_ecs_force_deploy" => {
                let cluster = Self::str_arg(args, "cluster")?;
                let service = Self::str_arg(args, "service")?;
                platform_result(self.platform.aws_ecs_force_deploy(cluster, service, &mut *approver))
            }
            "sam_build" => platform_result(self.platform.sam_build(&mut *approver)),
            "sam_deploy" => {
                let guided = Self::opt_bool_arg(args, "guided").unwrap_or(false);
                let stack_name = Self::opt_str_arg(args, "stack_name");
                platform_result(self.platform.sam_deploy(guided, stack_name, &mut *approver))
            }
            "cloudformation_describe" => {
                let stack = Self::str_arg(args, "stack")?;
                platform_result(self.platform.cloudformation_describe(stack))
            }
            "cloudformation_deploy" => {
                let template = Self::str_arg(args, "template")?;
                let stack = Self::str_arg(args, "stack")?;
                platform_result(self.platform.cloudformation_deploy(template, stack, &mut *approver))
            }
            "az_whoami" => platform_result(self.platform.az_whoami()),
            "az_webapp_list" => platform_result(self.platform.az_webapp_list()),
            "az_webapp_deploy" => {
                let name = Self::str_arg(args, "name")?;
                let rg = Self::str_arg(args, "resource_group")?;
                let source = Self::str_arg(args, "source")?;
                platform_result(self.platform.az_webapp_deploy(name, rg, source, &mut *approver))
            }
            "az_functionapp_deploy" => {
                let name = Self::str_arg(args, "name")?;
                let rg = Self::str_arg(args, "resource_group")?;
                let source = Self::str_arg(args, "source")?;
                platform_result(self.platform.az_functionapp_deploy(name, rg, source, &mut *approver))
            }
            "gcloud_whoami" => platform_result(self.platform.gcloud_whoami()),
            "gcloud_app_deploy" => platform_result(self.platform.gcloud_app_deploy(&mut *approver)),
            "gcloud_run_deploy" => {
                let service = Self::str_arg(args, "service")?;
                let image = Self::str_arg(args, "image")?;
                let region = Self::opt_str_arg(args, "region");
                platform_result(self.platform.gcloud_run_deploy(service, image, region, &mut *approver))
            }
            "gcloud_run_services" => platform_result(self.platform.gcloud_run_services()),
            "helm_list" => {
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(self.platform.helm_list(ns))
            }
            "helm_status" => {
                let release = Self::str_arg(args, "release")?;
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(self.platform.helm_status(release, ns))
            }
            "helm_install" => {
                let release = Self::str_arg(args, "release")?;
                let chart = Self::str_arg(args, "chart")?;
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(self.platform.helm_install(release, chart, ns, &mut *approver))
            }
            "helm_upgrade" => {
                let release = Self::str_arg(args, "release")?;
                let chart = Self::str_arg(args, "chart")?;
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(self.platform.helm_upgrade(release, chart, ns, &mut *approver))
            }
            "helm_uninstall" => {
                let release = Self::str_arg(args, "release")?;
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(self.platform.helm_uninstall(release, ns, &mut *approver))
            }
            "fly_whoami" => platform_result(self.platform.fly_whoami()),
            "fly_apps_list" => platform_result(self.platform.fly_apps_list()),
            "fly_deploy" => {
                let image = Self::opt_str_arg(args, "image");
                let app = Self::opt_str_arg(args, "app");
                platform_result(self.platform.fly_deploy(image, app, &mut *approver))
            }
            "fly_status" => {
                let app = Self::str_arg(args, "app")?;
                platform_result(self.platform.fly_status(app))
            }
            "railway_whoami" => platform_result(self.platform.railway_whoami()),
            "railway_status" => platform_result(self.platform.railway_status()),
            "railway_up" => {
                let detach = Self::opt_bool_arg(args, "detach").unwrap_or(false);
                platform_result(self.platform.railway_up(detach, &mut *approver))
            }
            "render_whoami" => platform_result(self.platform.render_whoami()),
            "render_services" => platform_result(self.platform.render_services()),
            "render_deploy" => {
                let service_id = Self::str_arg(args, "service_id")?;
                platform_result(self.platform.render_deploy(service_id, &mut *approver))
            }
            "netlify_whoami" => platform_result(self.platform.netlify_whoami()),
            "netlify_sites" => platform_result(self.platform.netlify_sites()),
            "netlify_deploy" => {
                let dir = Self::str_arg(args, "dir")?;
                let prod = Self::opt_bool_arg(args, "prod").unwrap_or(false);
                let site = Self::opt_str_arg(args, "site");
                platform_result(self.platform.netlify_deploy(dir, prod, site, &mut *approver))
            }
            "firebase_projects" => platform_result(self.platform.firebase_projects()),
            "firebase_deploy" => {
                let only = Self::opt_str_arg(args, "only");
                platform_result(self.platform.firebase_deploy(only, &mut *approver))
            }
            "firebase_functions" => platform_result(self.platform.firebase_functions()),
            _ => return Err(AgentError::UnknownTool(name.to_string())),
        };
        result
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

/// Same convention as `git_result` for the platform-CLI engines.
fn platform_result(result: zeus_fs::Result<PlatformOutput>) -> Result<ToolResult> {
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

/// Same convention as `git_result`/`platform_result` for the adb-backed
/// device engine. `DeviceOutput.success` is false when the command exits
/// nonzero OR the capture itself failed (no device, timeout) — in both cases
/// zeus must present it as an error so the model can react, not shrug.
fn device_result(out: zeus_fs::DeviceOutput) -> ToolResult {
    let artifact = out
        .artifact
        .as_ref()
        .map(|p| format!("\nartifact: {}", p.display()))
        .unwrap_or_default();
    let text = format!(
        "exit={:?}\n--- stdout ---\n{}--- stderr ---\n{}{}",
        out.exit_code, out.stdout, out.stderr, artifact
    );
    if out.success {
        ToolResult::ok(text)
    } else {
        ToolResult::err(text)
    }
}

/// Detect the most likely test command for a project by looking at its
/// manifests. Best-effort; the tool falls back to an explicit override when
/// nothing matches.
pub(crate) fn detect_test_command(root: &Path) -> Option<String> {
    let dir = |name: &str| root.join(name);
    // Ordered by likelihood/portability. `cargo test` and `go test ./...`
    // are the two that never need an extra runner installed.
    if dir("Cargo.toml").is_file() {
        return Some("cargo test".into());
    }
    if dir("go.mod").is_file() {
        return Some("go test ./...".into());
    }
    if dir("pyproject.toml").is_file() {
        return Some("python -m pytest -q".into());
    }
    if dir("package.json").is_file() {
        if dir("pnpm-lock.yaml").is_file() || dir("pnpm-workspace.yaml").is_file() {
            return Some("pnpm test".into());
        }
        if dir("yarn.lock").is_file() {
            return Some("yarn test".into());
        }
        return Some("npm test".into());
    }
    if dir("Makefile").is_file() {
        return Some("make test".into());
    }
    if dir("Gemfile").is_file() {
        return Some("bundle exec rspec".into());
    }
    None
}

/// Pull the handful of verdict lines (e.g. `test result: ok. 12 passed; 0
/// failed`, `12 passed in 1.2s`, `Done in 1.1s`) out of raw runner output so
/// the model gets a compact summary instead of a wall of dots.
fn summarize_test_output(stdout: &str) -> String {
    let mut seen: Vec<&str> = Vec::new();
    for raw in stdout.lines() {
        let line = raw.trim();
        let interesting = line.starts_with("test result:")
            || line.starts_with("ok ")
            || line.starts_with("FAIL")
            || (line.contains("passed") && line.contains("failed"))
            || line.contains(" passed in ")
            || line.contains("Tests:")
            || line.contains("Test Suites:")
            || line.contains("Error:")
            || line.starts_with("no test data")
            || line.starts_with("All tests passed");
        if interesting && !seen.contains(&line) {
            seen.push(line);
        }
        if seen.len() >= 8 {
            break;
        }
    }
    if seen.is_empty() {
        "(no summary lines captured)".to_string()
    } else {
        seen.join("\n")
    }
}

/// Launch the platform's default browser opener for `url`, launch-and-forget.
/// Rejects non-`{http,https,file}://` (and scheme-less `host:port`) targets so
/// a stray string can't be misinterpreted as a shell flag or command.
fn open_browser_url(url: &str) -> std::io::Result<()> {
    let url = url.trim();
    if !(url.starts_with("http://")
        || url.starts_with("https://")
        || url.starts_with("file://")
        || url.starts_with("localhost:")
        || url.starts_with("127.0.0.1:")
        || (url.contains('.') && !url.contains(' ') && !url.starts_with('-')))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("'{url}' isn't a usable web URL — expect something like http://localhost:5173"),
        ));
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .arg("/C")
            .arg("start")
            .arg("")
            .arg(url)
            .spawn()
            .map(|_| ())?;
    }
    #[cfg(not(target_os = "windows"))]
    {
        #[cfg(target_os = "macos")]
        let mut cmd = std::process::Command::new("open");
        #[cfg(all(not(target_os = "macos"), target_os = "linux"))]
        let mut cmd = std::process::Command::new("xdg-open");
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let mut cmd = std::process::Command::new("xdg-open");
        cmd.arg(url).spawn().map(|_| ())?;
    }
    Ok(())
}

/// Map an image file extension to its MIME type; `None` for non-raster
/// formats that a vision model cannot ingest.
fn image_mime(path: &Path) -> Option<&'static str> {
    let ext = path.extension().and_then(|e| e.to_str())?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" | "jpe" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "bmp" => Some("image/bmp"),
        _ => None,
    }
}

/// Returns `Some(reason)` if `url` points at a loopback/private target that
/// `web_fetch` should refuse to scrape (the fetch tool retrieves content for
/// the model, so pointing it at the user's local services would leak them).
fn reject_web_target(url: &str) -> Option<String> {
    let host = url
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("")
        .trim();
    let host = host.rsplit('@').next().unwrap_or("").trim();
    let host = host.trim_matches(|c| c == '[' || c == ']'); // IPv6 brackets
    if host.is_empty() {
        return Some("no host in url".into());
    }
    for bad in [
        "localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0", "169.254.169.254", "metadata.google.internal",
    ] {
        if host == bad {
            return Some(format!("'{host}' resolves to the loopback/metadata services"));
        }
    }
    None
}

/// Public alias for the doc-extraction module to reuse.
pub(crate) fn strip_html_pub(html: &str) -> String {
    strip_html(html)
}

/// Crude-but-effective HTML → text: drops scripts/styles/head, then tags,
/// then decodes common entities and collapses whitespace. Good enough for
/// scraping docs/pages into something the model can read.
fn strip_html(html: &str) -> String {
    let mut clipped: String = html.to_string();
    for (open_tag, close_tag) in [("<script", "</script"), ("<style", "</style")] {
        let mut buffer = String::with_capacity(clipped.len());
        let mut rest = clipped.as_str();
        while let Some(start) = rest.find(open_tag) {
            buffer.push_str(&rest[..start]);
            rest = &rest[start..];
            match rest.find(close_tag) {
                Some(end) => rest = &rest[end..],
                None => break,
            }
        }
        buffer.push_str(rest);
        clipped = buffer;
    }
    let without_tags = clipped;
    let mut text = String::with_capacity(without_tags.len());
    for seg in without_tags.split('<') {
        match seg.find('>') {
            Some(idx) if !seg[..idx].trim().is_empty() => text.push('\n'),
            _ => {}
        }
        if let Some(idx) = seg.find('>') {
            text.push_str(&seg[idx + 1..]);
        } else {
            text.push_str(seg);
        }
    }
    for (entity, ch) in [
        ("&amp;", '&'),
        ("&lt;", '<'),
        ("&gt;", '>'),
        ("&quot;", '"'),
        ("&#39;", '\''),
        ("&nbsp;", ' '),
    ] {
        text = text.replace(entity, &ch.to_string());
    }
    let text = text.replace('\r', "");
    text.split('\n')
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Percent-encode a string for use inside a URL query value.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
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
        // Calling an unknown tool is the model's own mistake, and
        // recoverable — it comes back as a failed `ToolResult` (so the
        // model sees the mistake and can retry) rather than a hard `Err`
        // that would kill the whole turn with no chance to self-correct.
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let tm = tool_manager(&root);
        let result = tm.dispatch_with_approver("frobnicate", "{}", approve).unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("frobnicate"));
    }

    #[test]
    fn builtin_skills_are_listed_and_readable() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let tm = tool_manager(&root);
        // list_skills includes all shipped built-ins.
        let listed = tm
            .dispatch_with_approver("list_skills", "{}", approve)
            .unwrap();
        assert!(!listed.is_error, "{}", listed.content);
        assert!(listed.content.contains("build-app"));
        assert!(listed.content.contains("database"));
        assert!(listed.content.contains("ui-design"));
        // Search narrows the catalog.
        let searched = tm
            .dispatch_with_approver("list_skills", r#"{"search":"xlsx"}"#, approve)
            .unwrap();
        assert!(!searched.is_error);
        assert!(searched.content.contains("document-reading"));
        assert!(!searched.content.contains("build-app"));
        // read_skill loads instructions.
        let read = tm
            .dispatch_with_approver("read_skill", r#"{"name":"git-workflows"}"#, approve)
            .unwrap();
        assert!(!read.is_error, "{}", read.content);
        assert!(read.content.contains("Before committing"));
        // Unknown skill errors.
        let missing = tm
            .dispatch_with_approver("read_skill", r#"{"name":"nope"}"#, approve)
            .unwrap();
        assert!(missing.is_error);
    }

    #[test]
    fn read_skill_recursively_composes_depends_on_chain() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let tm = tool_manager(&root);
        // build-app composes project-orientation, database, api, frontend,
        // security, qa-testing, documentation — a single read_skill call
        // loads the whole chain.
        let r = tm
            .dispatch_with_approver("read_skill", r#"{"name":"build-app","recursive":true}"#, approve)
            .unwrap();
        assert!(!r.is_error, "{}", r.content);
        assert!(r.content.contains("skill: build-app"));
        assert!(r.content.contains("skill: database"));
        assert!(r.content.contains("skill: api"));
        assert!(r.content.contains("skill: frontend"));
        assert!(r.content.contains("skill: security"));
        assert!(r.content.contains("skill: qa-testing"));
        assert!(r.content.contains("skill: documentation"));
        assert!(r.content.contains("skill: project-orientation"));
    }

    #[test]
    fn project_skill_shadows_builtin() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(root.join(".agent/skills/database")).unwrap();
        std::fs::write(
            root.join(".agent/skills/database/SKILL.md"),
            "---\nname: database\ndescription: PROJECT-OVERRIDE\n---\n# Project DB rules",
        )
        .unwrap();
        let tm = tool_manager(&root);
        let read = tm
            .dispatch_with_approver("read_skill", r#"{"name":"database"}"#, approve)
            .unwrap();
        assert!(!read.is_error);
        assert!(!read.content.contains("Design schemas, SQL, and migrations"));
        assert!(read.content.contains("Project DB rules"));
        assert!(read.content.contains("skill: database (tier: project)"));
    }

    #[test]
    fn read_document_extracts_text_and_errors_on_binary() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("notes.md"), "# Notes\n\nplain markdown text here").unwrap();
        let tm = tool_manager(&root);

        let r = tm
            .dispatch_with_approver("read_document", r#"{"path":"notes.md"}"#, approve)
            .unwrap();
        assert!(!r.is_error, "{}", r.content);
        assert!(r.content.contains("plain markdown text here"));

        let missing = tm
            .dispatch_with_approver("read_document", r#"{"path":"nope.pdf"}"#, approve)
            .unwrap();
        assert!(missing.is_error);
    }

    #[test]
    fn read_image_attaches_base64_bytes() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        // 1x1 transparent PNG.
        let png = base64::engine::general_purpose::STANDARD
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
            .unwrap();
        std::fs::write(root.join("pixel.png"), &png).unwrap();
        let tm = tool_manager(&root);

        let r = tm
            .dispatch_with_approver("read_image", r#"{"path":"pixel.png"}"#, approve)
            .unwrap();
        assert!(!r.is_error, "{}", r.content);
        assert_eq!(r.images.len(), 1);
        assert_eq!(r.images[0].mime_type, "image/png");
        assert!(!r.images[0].data_base64.is_empty());

        // Non-image / missing paths error cleanly.
        let bad = tm
            .dispatch_with_approver("read_image", r#"{"path":"pixel.txt"}"#, approve)
            .unwrap();
        assert!(bad.is_error);
        let missing = tm
            .dispatch_with_approver("read_image", r#"{"path":"absent.png"}"#, approve)
            .unwrap();
        assert!(missing.is_error);
    }

    #[test]
    fn understand_repo_reports_stack_and_relevance() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(root.join("src/auth")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname=\"x\"\n[dependencies]\naxum=\"0.7\"\nsqlx=\"0.8\"\n",
        )
        .unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(root.join("src/auth/login.rs"), "pub fn login() {}").unwrap();
        let tm = tool_manager(&root);

        let r = tm
            .dispatch_with_approver("understand_repo", r#"{"topic":"authentication"}"#, approve)
            .unwrap();
        assert!(!r.is_error, "{}", r.content);
        assert!(r.content.contains("Repository understanding"));
        assert!(r.content.contains("Axum"));
        assert!(r.content.contains("authentication") || r.content.contains("auth"));

        let no_topic = tm
            .dispatch_with_approver("understand_repo", "{}", approve)
            .unwrap();
        assert!(!no_topic.is_error);
        assert!(no_topic.content.contains("Rust"));
    }

    #[test]
    fn urlencode_encodes_query() {
        assert_eq!(urlencode("offline sync"), "offline+sync");
        assert_eq!(urlencode("a&b?"), "a%26b%3F");
        assert_eq!(urlencode("rust"), "rust");
    }

    #[test]
    fn web_search_rejects_empty_query() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let tm = tool_manager(&root);
        let r = tm
            .dispatch_with_approver("web_search", r#"{"query":""}"#, approve)
            .unwrap();
        assert!(r.is_error);
        assert!(r.content.contains("non-empty"));
        // Missing `query` is the model's own mistake, and recoverable —
        // it comes back as a failed `ToolResult` (so the model sees the
        // mistake and can retry) rather than a hard dispatch error that
        // would kill the whole turn with no chance to self-correct.
        let missing = tm.dispatch_with_approver("web_search", "{}", approve).unwrap();
        assert!(missing.is_error, "missing `query` should surface as a failed tool result");
    }

#[test]
    fn web_search_is_read_only_tool() {
        assert!(is_read_only_tool("web_search"), "web_search must run in plan mode");
        assert!(is_read_only_tool("web_fetch"));
        assert!(!is_read_only_tool("bash"));
    }

    #[test]
    fn memory_tools_list_read_write() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let tm = tool_manager(&root);
        let approve = |_p: &PermissionRequest| ApprovalDecision::Approved;

        let empty = tm
            .dispatch_with_approver("memory", r#"{"action":"list"}"#, approve)
            .unwrap();
        assert!(!empty.is_error);
        assert!(empty.content.contains("No long-term memory"));

        let w = tm
            .dispatch_with_approver("memory_write", r#"{"name":"auth","content":"token-based auth"}"#, approve)
            .unwrap();
        assert!(!w.is_error, "{}", w.content);
        let path = root.join(".agent/memory/auth.md");
        assert!(path.exists());

        let list = tm
            .dispatch_with_approver("memory", r#"{"action":"list"}"#, approve)
            .unwrap();
        assert!(list.content.contains("auth"));

        let read = tm
            .dispatch_with_approver("memory", r#"{"action":"read","name":"auth"}"#, approve)
            .unwrap();
        assert!(read.content.contains("token-based"));

        let bad_name = tm
            .dispatch_with_approver("memory_write", r#"{"name":"BAD NAME","content":"x"}"#, approve)
            .unwrap();
        assert!(bad_name.is_error);
    }

    #[test]
    fn memory_tools_blocked_in_plan_mode() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let tm = tool_manager(&root);
        tm.set_plan_mode(true);
        let approve = |_p: &PermissionRequest| ApprovalDecision::Approved;
        let r = tm
            .dispatch_with_approver("memory_write", r#"{"name":"x","content":"y"}"#, approve)
            .unwrap();
        assert!(r.is_error, "memory_write must be blocked in plan mode");
    }

    #[test]
    fn code_intel_tools_round_trip() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("lib.rs"),
            "pub struct Foo {}\nimpl Foo { pub fn bar(&self) {} }\nfn use_it(f: &Foo) -> u32 { 0 }\n",
        )
        .unwrap();

        let tm = tool_manager(&root);

        // Build the index (force so a stale one can't short-circuit).
        let r = tm
            .dispatch_with_approver("code_index", r#"{"force":true}"#, approve)
            .unwrap();
        assert!(!r.is_error, "{}", r.content);
        assert!(r.content.contains("indexed"));

        // Fresh run without force reports the cached index.
        let r = tm
            .dispatch_with_approver("code_index", "{}", approve)
            .unwrap();
        assert!(!r.is_error);
        assert!(r.content.contains("already exists"));

        // Symbols lookup.
        let r = tm
            .dispatch_with_approver("code_symbols", r#"{"name":"Foo"}"#, approve)
            .unwrap();
        assert!(!r.is_error);
        assert!(r.content.contains("Foo") && r.content.contains("lib.rs"));

        // Refs (word-boundary) find all three occurrences.
        let r = tm
            .dispatch_with_approver("code_refs", r#"{"name":"Foo"}"#, approve)
            .unwrap();
        assert!(!r.is_error);
        assert!(r.content.contains("3 reference(s)"), "got: {}", r.content);
    }

    #[test]
    fn code_verbose_rename_reports_plan_only() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("lib.rs"), "fn alpha() { leap_alpha(); }\n").unwrap();

        let tm = tool_manager(&root);
        let r = tm
            .dispatch_with_approver(
                "code_rename",
                r#"{"old":"alpha","new":"omega"}"#,
                approve,
            )
            .unwrap();
        assert!(!r.is_error);
        assert!(r.content.contains("rename 'alpha' -> 'omega'"));
        assert!(r.content.contains("Plan only"));
        // Rename must never write.
        assert!(std::fs::read_to_string(root.join("lib.rs")).unwrap().contains("fn alpha()"));
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
            let result = tm.dispatch_with_approver(&spec.name, "{}", approve).unwrap();
            // `dispatch_with_approver` now soft-fails both InvalidArguments
            // and UnknownTool into `Ok(ToolResult::err(...))` instead of
            // returning `Err`, so `Err(AgentError::UnknownTool(_))` can no
            // longer surface here at all — checking for it (the old form of
            // this test) would pass unconditionally regardless of whether a
            // spec has a real handler. Check the error text `dispatch_inner`
            // actually produces for an unmatched name instead: missing
            // required args on a *wired* tool surfaces as some other
            // message ("missing/invalid '...'" etc.), never this one.
            assert!(
                !(result.is_error && result.content.starts_with("unknown tool:")),
                "tool spec '{}' has no handler: {}",
                spec.name,
                result.content
            );
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

    #[test]
    fn detect_test_command_maps_manifests() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("Cargo.toml"), "").unwrap();
        assert_eq!(detect_test_command(root).as_deref(), Some("cargo test"));

        std::fs::remove_file(root.join("Cargo.toml")).unwrap();
        std::fs::write(root.join("go.mod"), "").unwrap();
        assert_eq!(detect_test_command(root).as_deref(), Some("go test ./..."));

        std::fs::remove_file(root.join("go.mod")).unwrap();
        std::fs::write(root.join("pyproject.toml"), "").unwrap();
        assert_eq!(
            detect_test_command(root).as_deref(),
            Some("python -m pytest -q")
        );

        std::fs::remove_file(root.join("pyproject.toml")).unwrap();
        std::fs::write(root.join("package.json"), "{}").unwrap();
        assert_eq!(detect_test_command(root).as_deref(), Some("npm test"));

        std::fs::write(root.join("pnpm-lock.yaml"), "").unwrap();
        assert_eq!(detect_test_command(root).as_deref(), Some("pnpm test"));

        std::fs::remove_file(root.join("pnpm-lock.yaml")).unwrap();
        std::fs::write(root.join("yarn.lock"), "").unwrap();
        assert_eq!(detect_test_command(root).as_deref(), Some("yarn test"));
    }

    #[test]
    fn detect_test_command_none_when_no_manifest() {
        let tmp = TempDir::new().unwrap();
        assert_eq!(detect_test_command(tmp.path()), None);
    }

    #[test]
    fn summarize_test_output_picks_verdict_lines() {
        let out = "\n  Compiling zeus v0.1.0\n\nrunning 4 tests\n..s....\n\ntest result: ok. 4 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out\n\nrunning 1 test\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n";
        let summary = summarize_test_output(out);
        assert!(summary.contains("test result: ok"), "{summary}");
        assert!(summary.contains("4 passed"), "{summary}");
        assert!(!summary.contains("running 4"), "{summary}");
    }

    #[test]
    fn test_tool_runs_with_explicit_command() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let tm = tool_manager(&root);
        let cmd = if cfg!(windows) {
            r#"powershell -NoProfile -Command "Write-Output 'test result: ok. 1 passed; 0 failed'""#
        } else {
            "echo test result: ok. 1 passed; 0 failed"
        };
        let args = serde_json::json!({ "command": cmd });
        let r = tm
            .dispatch_with_approver("test", &args.to_string(), approve)
            .unwrap();
        assert!(!r.is_error, "{}", r.content);
        assert!(r.content.contains("1 passed"), "{}", r.content);
    }

    #[test]
    fn test_tool_without_command_reports_detection_failure() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let tm = tool_manager(&root);
        let r = tm.dispatch_with_approver("test", "{}", approve).unwrap();
        assert!(r.is_error, "{}", r.content);
        assert!(r.content.contains("auto-detect"), "{}", r.content);
    }

    #[test]
    fn browser_rejects_bad_url_and_blocks_in_plan_mode() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let tm = tool_manager(&root);

        // A path-ish target isn't a web URL and must not be handed as-is to
        // the opener (argument-injection guard — never spawn in this test).
        let bad = tm
            .dispatch_with_approver("browser", r#"{"url":"C:/Windows/System32"}"#, approve)
            .unwrap();
        assert!(bad.is_error, "{}", bad.content);

        // Plan mode blocks it even for a plausible http URL.
        tm.set_plan_mode(true);
        let blocked = tm
            .dispatch_with_approver("browser", r#"{"url":"http://localhost:5173"}"#, approve)
            .unwrap();
        assert!(blocked.is_error, "{}", blocked.content);
        assert!(blocked.content.contains("Plan mode"), "{}", blocked.content);
        tm.set_plan_mode(false);
    }
}
