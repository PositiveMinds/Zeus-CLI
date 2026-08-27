//! Tool spec definitions (what the model sees) and platform CLI detection.

use std::collections::HashSet;
use std::path::PathBuf;
use zeus_provider::ToolSpec;

/// Tool specs advertised to the model. Kept in sync with `ToolManager`'s
/// `dispatch_with_approver` match arms below — every name here must have a
/// handler, and vice versa.
pub(crate) fn core_tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "todowrite".into(),
            description: "Replace your own progress checklist for this session with the given list — call this whenever you break a request into multiple steps, and again every time a step's status changes. You own this list entirely: pass the FULL list every time (not a diff), including items already completed. Mark exactly one item in_progress at a time (the one you're actively working on), never more; mark an item completed only once you've actually verified it, not just attempted it. Skip this tool for a single trivial action.".into(),
            parameters: serde_json::json!({
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
            name: "current_time".into(),
            description: "Return the current local date and time (with UTC offset, ISO week, and day of year). Use whenever the user asks what date/time it is, or whenever a task depends on the current date (naming files with today's date, scheduling, 'is X released yet' comparisons). Always fresh — never assume a date or time from your training data; the value is read from the clock at call time.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "read".into(),
            description: "Read a project file (line-numbered output). The result is prefixed with the exact line window shown (e.g. lines 1-500 of 3200) — if it says the file continues, pass offset=<next line> to keep reading; never treat a partial read as the whole file.".into(),
            parameters: serde_json::json!({
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
            name: "read_multiple".into(),
            description: "Read several project files in one call. `paths` is a JSON array of strings (up to 20, each read with its own `limit`, default 1500 lines). Each file is returned as a separate block headed with `=== path ===`; a missing file yields a `--- path: <error> ---` block instead of failing the whole call. Use when you need several related files (a module, its types, its tests) at once — one round-trip instead of N reads.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 20
                    },
                    "limit": {"type": "integer"}
                },
                "required": ["paths"]
            }),
        },
        ToolSpec {
            name: "write".into(),
            description: "Create or overwrite a project file. Must read an existing file first."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "path": {"type": "string"}, "content": {"type": "string"} },
                "required": ["path", "content"]
            }),
        },
        ToolSpec {
            name: "edit".into(),
            description: "Targeted string replace in a file — you MUST read the file first. `old_string` must match the file's text exactly; include enough surrounding context (unique lines) so it matches once. Multiple matches are rejected unless `replace_all` is set — if you get an 'ambiguous' error, re-read the file and widen `old_string` with neighboring lines. The change goes through the approval prompt as a diff you can apply or reject. Stale files (changed on disk since your last read) are refused — re-read and retry.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string", "description": "Exact text to replace; must match uniquely (or set replace_all)"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of erroring on multiple matches"}
                },
                "required": ["path", "old_string", "new_string"]
            }),
        },
        ToolSpec {
            name: "delete".into(),
            description: "Delete a file or directory. Always requires user approval.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "path": {"type": "string"} },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "rename".into(),
            description: "Rename or move a file/directory.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "from": {"type": "string"}, "to": {"type": "string"} },
                "required": ["from", "to"]
            }),
        },
        ToolSpec {
            name: "copy".into(),
            description: "Copy a file.".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "pattern": {"type": "string"}, "max": {"type": "integer"} },
                "required": ["pattern"]
            }),
        },
        ToolSpec {
            name: "mkdir".into(),
            description: "Create a directory (and any missing parents) inside the project. Use to scaffold project structure (e.g. src/components, public/assets, models/) before writing files into it — `write` also creates parents automatically, but empty directories need this.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "path": {"type": "string"} },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "listdir".into(),
            description: "List a directory's immediate contents (files and subdirectories, one per line; directories show a trailing '/'). Pass recursive=true for a full tree — the fast way to analyze a project's structure before reading specific files.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"}
                },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "bash".into(),
            description: "Run a shell command in the project root (foreground, bounded) and wait for it to finish. Use for builds/tests/read-only commands. For a command that doesn't exit on its own (a dev server, `docker compose up`), set background=true instead of using this in foreground mode.".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "bg_output".into(),
            description: "Read the captured stdout/stderr so far for a background task by id.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "id": {"type": "integer"} },
                "required": ["id"]
            }),
        },
        ToolSpec {
            name: "bg_stop".into(),
            description: "Stop a running (or paused) background task by id.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "id": {"type": "integer"} },
                "required": ["id"]
            }),
        },
        ToolSpec {
            name: "bg_pause".into(),
            description: "Suspend a running background task in place (freezes it without killing the process); resume it later with bg_resume using the same id.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "id": {"type": "integer"} },
                "required": ["id"]
            }),
        },
        ToolSpec {
            name: "bg_resume".into(),
            description: "Continue a previously-paused background task, exactly where it stopped.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "id": {"type": "integer"} },
                "required": ["id"]
            }),
        },
        // --- Verification: tests + visual (browser) ---
        ToolSpec {
            name: "test".into(),
            description: "Run the project's test suite. Auto-detects the test runner from the repo (cargo test / npm|pnpm|yarn test / python -m pytest / go test / make test); pass an explicit `command` to override when a targeted run is needed (single test, extra flags). Bounded by timeout_secs (default 300). Returns the exit code plus a parsed pass/fail summary — treat a nonzero exit as a failing suite and read the stderr below it.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Explicit test command to run instead of auto-detection"},
                    "timeout_secs": {"type": "integer"}
                }
            }),
        },
        ToolSpec {
            name: "verify".into(),
            description: "Verify the project compiles/builds (and tests pass) using the build and test commands detected for the project's language (cargo build, go build, npm run build, dotnet build, tsc, ...). Runs build then test by default; `steps` narrows to just \"build\" or \"test\", and an explicit `command` overrides detection entirely. A nonzero exit means verification failed — read the stderr below it. Bounded by timeout_secs (default 600). Use this after writing or editing code to prove it still compiles.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Explicit command to run instead of the detected build/test"},
                    "steps": {"type": "string", "enum": ["build", "test", "all"], "description": "Which steps to run (default all)"},
                    "timeout_secs": {"type": "integer"}
                }
            }),
        },
        ToolSpec {
            name: "browser".into(),
            description: "Open a URL in the user's default web browser so the running app can be visually inspected. Use AFTER starting a dev server (bash background=true + bg_output). Accepts http(s):// URLs and localhost:port-style addresses (http:// scheme is added automatically for bare host:port). The human looks at the page — tell them what to check and ask what they see.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "url": {"type": "string"} },
                "required": ["url"]
            }),
        },
        ToolSpec {
            name: "web_fetch".into(),
            description: "Fetch a URL over HTTP(S) and return its content as text. Use to scrape docs, read an API/endpoint response, download raw source, or inspect a web page the model needs to act on (the browser tool just opens it for a human — web_fetch returns the actual content here). max_chars caps the returned body (default 20000); selective=true strips HTML to approximate markdown text instead of returning raw HTML. Errors on non-2xx status and on obviously non-text content. Requires network access.".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "search": {"type": "string", "description": "Optional substring to filter names/descriptions/tags by"}
                }
            }),
        },
        ToolSpec {
            name: "read_skill".into(),
            description: "Load a skill's full SKILL.md instructions into context by name. Use when a listed skill is relevant to the current task — it returns markdown instructions plus any bundled resource file names (which can then be read directly from the skill directory via the read tool). The skill's instructions may change HOW you approach the task, so read the full body, not just the description.".into(),
            parameters: serde_json::json!({
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
            description: "Extract text from binary/office documents for the model to act on: PDF, DOCX, XLSX (each worksheet as a row grid), PPTX (slides), EPUB (chapters in reading order). HTML is also handled with tables rendered as pipe-delimited row grids. Also handles plain-text formats via the read tool. max_chars caps returned text (default 20000). Use instead of read for .pdf/.docx/.pptx/.xlsx/.epub/.html files — read would return binary garbage for those. Returns unsupported/missing files as errors. For scanned/image PDFs (no text layer) it errors and you should use read_image + the ui-design skill.".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Optional subject to find existing related code for"}
                }
            }),
        },
        ToolSpec {
            name: "rag_search".into(),
            description: "Keyword-based retrieval over the project's source files: chunks each file and ranks chunks against the query with BM25-style term weights (no model call, read-only, works offline). Use when you need to find code that is about a concept but may not contain the exact identifier/string you would grep for — e.g. \"where is connection retry handled\" or \"which code touches rate limiting\". Returns the top-k matching chunks with file paths. For exact-string lookup prefer grep; for symbol names use code_symbols.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The concept/terms to find code for"},
                    "k": {"type": "integer", "description": "How many top chunks to return (default 5)"}
                },
                "required": ["query"]
            }),
        },
        ToolSpec {
            name: "rag_index".into(),
            description: "Build or refresh the persistent RAG chunk index at .agent/rag_index.json: chunks every source file (same walk as rag_search) and saves the index to disk so later rag_search calls reuse it instead of re-chunking the whole project. Uses the current file set; a later rag_search rebuilds automatically if the index is stale. Pass force=true to rebuild unconditionally; pass embed=true to also embed every chunk with the configured provider (best-effort: if no embeddings are available the index stays keyword-only).".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "force": {"type": "boolean", "description": "Rebuild even if a fresh index already exists (default false)"},
                    "embed": {"type": "boolean", "description": "Also embed every chunk with the configured provider (best-effort, default false)"}
                }
            }),
        },
        ToolSpec {
            name: "memory".into(),
            description: "Long-term project memory under .agent/memory/: `list` shows note names + first lines; `read <name>` returns a note's full body. Used to persist decisions, conventions, and gotchas across sessions. Read before large/unknown tasks; check what the project already decided before exploring anew.".into(),
            parameters: serde_json::json!({
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
            description: "Write a long-term memory note. By default writes to .agent/memory/<name>.md (project-scoped). Set global=true to write to ~/.zeus/memory/<name>.md (shared across all projects). Content is a short markdown plan/decision/gotcha you want to persist across sessions. Set category to classify the note (decision/gotcha/convention/todo). Overwrites the note if it exists. Ask the user first before writing non-obvious memories.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                    "global": {"type": "boolean", "description": "Write to global memory (~/.zeus/memory/) instead of project memory (default false)"},
                    "category": {"type": "string", "enum": ["decision", "gotcha", "convention", "todo"], "description": "Category for the memory note (default: none)"}
                },
                "required": ["name", "content"]
            }),
        },
        ToolSpec {
            name: "device".into(),
            description: "Test on an Android device via adb — over USB debugging or wireless (adb connect). Actions: devices (list USB+wireless), connect <host:port> (wireless debug), disconnect <host:port>, install <apk_path>, uninstall <package>, launch <package> [activity] (start the app), screenshot [out] (PNG into the project), screenrecord [out] [seconds] (MP4 screen capture, 1-30s, default 10), logcat [filter] [max_lines] (bounded crash/console dump), logcat_clear (reset the buffer), shell <command> (arbitrary device shell — the escape hatch), pair <host_port> <code> (wireless pairing), info (model / Android version / SDK), reverse [local_port] [device_port] (expose a host port on the device — needed for app/webview debugging), forward [local_port] [device_port] (expose a device port on the host), input <event> (UI automation: tap/swipe/keyevent/type), pull <remote> [out] (copy a file off the device), push <out> <remote> (copy a file onto the device). Requires the Android platform-tools `adb` on PATH and a device authorized for debugging.".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "git_diff".into(),
            description: "git diff. staged=true for the index; refs=[\"a\"] diffs against a commit, refs=[\"a\",\"b\"] diffs a..b.".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "path": {"type": "string"} },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "git_log".into(),
            description: "git log --oneline, optionally scoped to one path.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "max": {"type": "integer"}, "path": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "git_show".into(),
            description: "git show <commit-or-ref> — full diff/details for one commit.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "target": {"type": "string"} },
                "required": ["target"]
            }),
        },
        ToolSpec {
            name: "git_branch_list".into(),
            description: "List local and remote branches.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "git_remote_list".into(),
            description: "List configured remotes (git remote -v).".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "git_tag_list".into(),
            description: "List tags.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "git_stash_list".into(),
            description: "List stash entries.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        // --- Git: reversible write ---
        ToolSpec {
            name: "git_add".into(),
            description: "Stage one or more paths.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "paths": {"type": "array", "items": {"type": "string"}} },
                "required": ["paths"]
            }),
        },
        ToolSpec {
            name: "git_commit".into(),
            description: "Commit staged changes (or all tracked changes if all=true) with the given message. Read the diff first (git_diff) so the message actually reflects what changed.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "message": {"type": "string"}, "all": {"type": "boolean"} },
                "required": ["message"]
            }),
        },
        ToolSpec {
            name: "git_stash_push".into(),
            description: "Stash the working tree changes.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "message": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "git_stash_pop".into(),
            description: "Apply and drop the most recent stash entry.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "git_branch_create".into(),
            description: "Create a new branch at HEAD.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "name": {"type": "string"} },
                "required": ["name"]
            }),
        },
        ToolSpec {
            name: "git_branch_delete".into(),
            description: "Delete a branch. force=true uses -D (needed for an unmerged branch) instead of -d.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "name": {"type": "string"}, "force": {"type": "boolean"} },
                "required": ["name"]
            }),
        },
        ToolSpec {
            name: "git_tag_create".into(),
            description: "Create a tag, annotated if a message is given.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "name": {"type": "string"}, "message": {"type": "string"} },
                "required": ["name"]
            }),
        },
        // --- Git: working-tree-changing ---
        ToolSpec {
            name: "git_checkout".into(),
            description: "Check out an existing branch or commit.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "target": {"type": "string"} },
                "required": ["target"]
            }),
        },
        // --- Git: network / shared-state ---
        ToolSpec {
            name: "git_fetch".into(),
            description: "Fetch from a remote (or the default remote) without merging.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "remote": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "git_pull".into(),
            description: "git pull (fetch + merge/rebase per repo config).".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "git_push".into(),
            description: "git push. force=true is denied by a built-in safety rule regardless of approval — force-pushing needs an explicit, narrower rule in project settings.".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "target": {"type": "string"} },
                "required": ["target"]
            }),
        },
        ToolSpec {
            name: "git_cherry_pick".into(),
            description: "Apply the changes from one commit onto the current branch.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "target": {"type": "string"} },
                "required": ["target"]
            }),
        },
        ToolSpec {
            name: "git_rebase".into(),
            description: "Rebase the current branch onto another (rewrites history — use with care).".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "onto": {"type": "string"} },
                "required": ["onto"]
            }),
        },
        ToolSpec {
            name: "git_merge".into(),
            description: "Merge a branch into the current one. On conflict, the raw git output (naming the conflicting files) is returned — read those files to see the conflict markers.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "branch": {"type": "string"} },
                "required": ["branch"]
            }),
        },
        // --- Phase 6: Code Intelligence (database-free symbol index) ---
        ToolSpec {
            name: "code_index".into(),
            description: "Scan the project's source files and write .agent/index.json (symbol index). Run before code_symbols/code_defs when no index exists yet.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "force": {"type": "boolean"} }
            }),
        },
        ToolSpec {
            name: "code_symbols".into(),
            description: "Look up symbols (functions/structs/classes/enums/...) in the project index by name (substring, case-insensitive). Returns kind, file, line.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "name": {"type": "string"} },
                "required": ["name"]
            }),
        },
        ToolSpec {
            name: "code_defs".into(),
            description: "Go-to-definition: same as code_symbols but reports the matching definitions only, suitable for 'where is X defined?'.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "name": {"type": "string"} },
                "required": ["name"]
            }),
        },
        ToolSpec {
            name: "code_refs".into(),
            description: "Find references to a symbol across the project (and configured extra project roots) via ripgrep. Word-boundary matching, file:line:text output.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "max": {"type": "integer"}
                },
                "required": ["name"]
            }),
        },
        ToolSpec {
            name: "code_graph".into(),
            description: "Call graph: who calls this symbol, and/or what does it call. Built from tree-sitter AST parsing (not text search), so it only covers languages with a wired grammar (rust/python/go/js/ts/c/cpp/java/cs/rb) and misses dynamic dispatch/reflection. Run code_index first.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "direction": {"type": "string", "enum": ["callers", "callees", "both"]}
                },
                "required": ["name"]
            }),
        },
        ToolSpec {
            name: "code_rename".into(),
            description: "Propose a reference-update plan for renaming symbol `old` to `new` (word-boundary). Reports each file and the affected lines. It never writes — applying the edits is left to a separate review step.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "old": {"type": "string"},
                    "new": {"type": "string"}
                },
                "required": ["old", "new"]
            }),
        },
    ]
}

/// Every built-in tool (core + platform) — the full registry. Used by the
/// dispatch-completeness test and re-exported for tooling. This is
/// deliberately *not* the list the model sees: `ToolManager::all_tool_specs`
/// filters platform tools down to those whose CLI is actually on PATH (see
/// `platform_cli_for`/`filter_platform_specs`), so a small/free model isn't
/// asked to weigh 80+ deployment tools it can't use — a 147-tool list is
/// enough to make a small model burn every tool-call iteration without ever
/// converging to a final answer (the "(no final answer after N tool calls)"
/// failure, and the per-turn latency that makes simple tasks feel slow).
pub fn builtin_tool_specs() -> Vec<ToolSpec> {
    let mut specs = core_tool_specs();
    specs.extend(platform_tool_specs());
    specs
}

/// The single source of truth for platform-CLI tool names. `dispatch_inner`
/// routes any name in this list to `do_platform`, and the test suite asserts
/// this list matches both `platform_tool_specs()` and the `do_platform`
/// match arms — so adding a platform tool in one place without the others
/// is a test failure, not a silent drift.
pub const PLATFORM_TOOLS: &[&str] = &[
    "gh_issue_list",
    "gh_issue_view",
    "gh_issue_create",
    "gh_issue_close",
    "gh_pr_list",
    "gh_pr_view",
    "gh_pr_create",
    "gh_pr_merge",
    "gh_release_list",
    "gh_release_create",
    "gh_workflow_list",
    "gh_workflow_run",
    "gh_run_list",
    "supabase_login",
    "supabase_link",
    "supabase_projects_list",
    "supabase_status",
    "supabase_db_push",
    "supabase_db_diff",
    "supabase_functions_list",
    "supabase_functions_deploy",
    "vercel_whoami",
    "vercel_projects_list",
    "vercel_env_list",
    "vercel_deploy",
    "vercel_logs",
    "docker_ps",
    "docker_images",
    "docker_compose_up",
    "docker_compose_down",
    "docker_compose_logs",
    "k8s_get",
    "k8s_logs",
    "k8s_apply",
    "k8s_rollout_status",
    "tf_init",
    "tf_validate",
    "tf_plan",
    "tf_apply",
    "circleci_validate",
    "circleci_builds",
    "aws_whoami",
    "aws_s3_ls",
    "aws_s3_sync",
    "aws_ecr_login",
    "aws_lambda_list",
    "aws_lambda_invoke",
    "aws_ecs_list_clusters",
    "aws_ecs_force_deploy",
    "sam_build",
    "sam_deploy",
    "cloudformation_describe",
    "cloudformation_deploy",
    "az_whoami",
    "az_webapp_list",
    "az_webapp_deploy",
    "az_functionapp_deploy",
    "gcloud_whoami",
    "gcloud_app_deploy",
    "gcloud_run_deploy",
    "gcloud_run_services",
    "helm_list",
    "helm_status",
    "helm_install",
    "helm_upgrade",
    "helm_uninstall",
    "fly_whoami",
    "fly_apps_list",
    "fly_deploy",
    "fly_status",
    "railway_whoami",
    "railway_status",
    "railway_up",
    "render_whoami",
    "render_services",
    "render_deploy",
    "netlify_whoami",
    "netlify_sites",
    "netlify_deploy",
    "firebase_projects",
    "firebase_deploy",
    "firebase_functions",
];

/// Tool specs for the platform-CLI integrations (gh/supabase/vercel/aws/…).
/// Kept separate so the file stays navigable. Names must match the
/// `do_platform` dispatch arms exactly.
pub fn platform_tool_specs() -> Vec<ToolSpec> {
    vec![
        // --- GitHub ---
        ToolSpec {
            name: "gh_issue_list".into(),
            description: "List GitHub issues (state=open/closed).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "number": {"type": "string"} },
                "required": ["number"]
            }),
        },
        ToolSpec {
            name: "gh_issue_create".into(),
            description: "Create a GitHub issue (requires approval).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "number": {"type": "string"} },
                "required": ["number"]
            }),
        },
        ToolSpec {
            name: "gh_pr_list".into(),
            description: "List GitHub pull requests (state=open/closed).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "number": {"type": "string"} },
                "required": ["number"]
            }),
        },
        ToolSpec {
            name: "gh_pr_create".into(),
            description: "Create a GitHub pull request (requires approval).".into(),
            parameters: serde_json::json!({
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
            description:
                "Merge a GitHub pull request (requires approval). method=merge/squash/rebase."
                    .into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "limit": {"type": "integer"} }
            }),
        },
        ToolSpec {
            name: "gh_release_create".into(),
            description: "Create a GitHub release (requires approval).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "gh_workflow_run".into(),
            description: "Trigger a GitHub Actions workflow (requires approval).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "supabase_link".into(),
            description: "Link the project to a Supabase remote (requires approval).".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "project_ref": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "supabase_projects_list".into(),
            description: "List Supabase projects.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "supabase_status".into(),
            description: "Show local Supabase dev service status.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "supabase_db_push".into(),
            description: "Push local migrations to the linked remote database (requires approval)."
                .into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "supabase_db_diff".into(),
            description: "Generate a DB diff against the linked remote.".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "supabase_functions_deploy".into(),
            description: "Deploy a Supabase Edge Function (requires approval).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "vercel_projects_list".into(),
            description: "List Vercel projects.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "vercel_env_list".into(),
            description: "List Vercel environment variables.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "env": {"type": "string"},
                    "project": {"type": "string"}
                }
            }),
        },
        ToolSpec {
            name: "vercel_deploy".into(),
            description: "Deploy to Vercel (requires approval). prod=true deploys to production."
                .into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "all": {"type": "boolean"} }
            }),
        },
        ToolSpec {
            name: "docker_images".into(),
            description: "List docker images.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "docker_compose_up".into(),
            description: "docker compose up (requires approval).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "volumes": {"type": "boolean"} }
            }),
        },
        ToolSpec {
            name: "docker_compose_logs".into(),
            description: "docker compose logs.".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "tf_validate".into(),
            description: "terraform validate.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "tf_plan".into(),
            description: "terraform plan (optionally -out=<file>).".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "out": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "tf_apply".into(),
            description: "terraform apply (requires approval).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "config": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "circleci_builds".into(),
            description: "List CircleCI builds for a project.".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "aws_s3_ls".into(),
            description: "List S3 buckets or objects under a prefix.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "path": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "aws_s3_sync".into(),
            description: "Sync files to/from S3 (requires approval).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "aws_lambda_list".into(),
            description: "List AWS Lambda functions.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "aws_lambda_invoke".into(),
            description: "Invoke an AWS Lambda function (requires approval).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "aws_ecs_force_deploy".into(),
            description: "Force a new deployment of an ECS service (requires approval).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "sam_deploy".into(),
            description: "sam deploy (requires approval). guided=true for interactive prompts."
                .into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "stack": {"type": "string"} },
                "required": ["stack"]
            }),
        },
        ToolSpec {
            name: "cloudformation_deploy".into(),
            description: "aws cloudformation deploy a template to a stack (requires approval)."
                .into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "az_webapp_list".into(),
            description: "List Azure App Service web apps.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "az_webapp_deploy".into(),
            description: "Deploy to an Azure web app (requires approval).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "gcloud_app_deploy".into(),
            description: "Deploy to Google App Engine (requires approval).".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "gcloud_run_deploy".into(),
            description: "Deploy a container image to Cloud Run (requires approval).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        // --- Helm ---
        ToolSpec {
            name: "helm_list".into(),
            description: "helm list releases.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "namespace": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "helm_status".into(),
            description: "helm status for a release.".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "fly_apps_list".into(),
            description: "List Fly.io apps.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "fly_deploy".into(),
            description: "Deploy to Fly.io (requires approval).".into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "app": {"type": "string"} },
                "required": ["app"]
            }),
        },
        // --- Railway ---
        ToolSpec {
            name: "railway_whoami".into(),
            description: "Show the logged-in Railway user.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "railway_status".into(),
            description: "Show Railway project status.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "railway_up".into(),
            description: "Deploy to Railway (requires approval).".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "detach": {"type": "boolean"} }
            }),
        },
        // --- Render ---
        ToolSpec {
            name: "render_whoami".into(),
            description: "Show the logged-in Render user.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "render_services".into(),
            description: "List Render services.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "render_deploy".into(),
            description: "Trigger a deploy for a Render service (requires approval).".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "service_id": {"type": "string"} },
                "required": ["service_id"]
            }),
        },
        // --- Netlify ---
        ToolSpec {
            name: "netlify_whoami".into(),
            description: "Show the logged-in Netlify user.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "netlify_sites".into(),
            description: "List Netlify sites.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "netlify_deploy".into(),
            description: "Deploy to Netlify (requires approval). prod=true deploys to production."
                .into(),
            parameters: serde_json::json!({
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
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "firebase_deploy".into(),
            description: "Deploy to Firebase Hosting / Functions (requires approval).".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "only": {"type": "string"} }
            }),
        },
        ToolSpec {
            name: "firebase_functions".into(),
            description: "List Firebase Functions.".into(),
            parameters: serde_json::json!({ "type": "object", "properties": {} }),
        },
    ]
}

/// The CLI binary a platform tool needs on PATH before it's worth
/// advertising to the model. `None` for core (non-platform) tools — those
/// are always advertised. Tools whose CLI isn't present would fail at
/// dispatch anyway, and carrying their specs is pure dead weight on every
/// model request: ~80 deployment tools is a big enough list to slow
/// time-to-first-token and push small/free models into degenerate retries
/// and tool-call exhaustion. Dispatch still accepts these names regardless —
/// this only controls what the model is nudged toward.
pub(crate) fn platform_cli_for(name: &str) -> Option<&'static str> {
    let (prefix, _) = name.split_once('_')?;
    Some(match prefix {
        "gh" => "gh",
        "supabase" => "supabase",
        "vercel" => "vercel",
        "docker" => "docker",
        "k8s" => "kubectl",
        "tf" => "terraform",
        "circleci" => "circleci",
        "aws" | "cloudformation" => "aws",
        "sam" => "sam",
        "az" => "az",
        "gcloud" => "gcloud",
        "helm" => "helm",
        "fly" => "flyctl",
        "railway" => "railway",
        "render" => "render",
        "netlify" => "netlify",
        "firebase" => "firebase",
        _ => return None,
    })
}

/// Drop platform-tool specs whose CLI isn't in `present`. Core tools have no
/// CLI mapping (`platform_cli_for` returns `None`) and always survive.
pub(crate) fn filter_platform_specs(
    specs: Vec<ToolSpec>,
    present: &HashSet<String>,
) -> Vec<ToolSpec> {
    specs
        .into_iter()
        .filter(|s| platform_cli_for(&s.name).is_none_or(|cli| present.contains(cli)))
        .collect()
}

/// Which platform CLIs are on PATH. Pure PATH walk with filesystem existence
/// checks — no process spawns — so it's cheap enough to call from
/// `all_tool_specs` even without caching (`ToolManager` caches it anyway).
/// Windows checks the `.exe/.cmd/.bat/.com` variants.
pub(crate) fn detect_platform_clis() -> HashSet<String> {
    const CLIS: &[&str] = &[
        "gh",
        "supabase",
        "vercel",
        "docker",
        "kubectl",
        "terraform",
        "circleci",
        "aws",
        "sam",
        "az",
        "gcloud",
        "helm",
        "flyctl",
        "railway",
        "render",
        "netlify",
        "firebase",
    ];
    CLIS.iter()
        .filter(|c| cli_on_path(c))
        .map(|s| (*s).to_string())
        .collect()
}

pub(super) fn cli_on_path(cli: &str) -> bool {
    let Some(path_var) = std::env::var_os("PATH") else {
        return false;
    };
    let mut names = vec![cli.to_string()];
    if cfg!(windows) {
        names.extend([
            format!("{cli}.exe"),
            format!("{cli}.cmd"),
            format!("{cli}.bat"),
            format!("{cli}.com"),
        ]);
    }
    for dir in std::env::split_paths(&path_var) {
        for name in &names {
            let full = if dir.as_os_str().is_empty() {
                PathBuf::from(name)
            } else {
                dir.join(name)
            };
            if full.is_file() {
                return true;
            }
        }
    }
    false
}
