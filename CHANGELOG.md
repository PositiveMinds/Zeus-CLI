# Changelog

All notable changes to Zeus are tracked here. `zeus update` prints the notes
for the newest release before asking to install it; the full history lives in
this file. Versions follow `major.minor.patch`; each entry covers the
user-visible changes since the previous release.

## 0.2.0

- **New providers**: Azure OpenAI, Vertex AI, AWS Bedrock, OpenRouter,
  and Moonshot — with native auth (API key, OAuth2, SigV4) and
  config-only passthrough.
- **Orchestration overhaul**: dependency-aware step scheduling, retry
  with rollback, streaming plan generation, scored persona matching,
  and mid-execution revision signals.
- **File ops hardening**: atomic writes (temp+rename), per-path locking,
  symlink detection, file size limits, `.gitignore` respect in
  `listdir`, directory creation preview, and bulk-edit rollback.
- **Improved diffs**: context lines, original line numbers, hunk-based
  output, and edit range preview.
- **Context awareness**: content-based file probing, key file snippets,
  global memory (`~/.zeus/memory/`), stale-docs warning, recent git
  commits, and probe directory weighting.
- **UX/UI polish**: clipboard flash confirmation, undo/redo feedback,
  `Ctrl+O` hint in empty state, context window fix on model switch.
- **Internal**: `ReadTracker` upgraded from `Mutex` to `RwLock` for
  concurrent reads.


## 0.1.10

- **Sessions housekeeping**: `zeus sessions remove|rm <id>`, `prune
  --older-than <days>`, and `label <id> <name>` (empty name clears the
  label). The TUI `/sessions` picker shows labels, and `v` on a session
  opens it read-only for browsing (scroll with arrow/paging keys, Esc back
  to the live chat).
- **Whole-word approvals**: permission prompts now read `approve / session /
  cancel` (or accept `yes`/`y`, `session`/`s`) instead of bare y/n.
- **`zeus doctor` live checks**: each configured provider now performs a
  real model-list call (12s timeout) so dead API keys, exhausted credits,
  and down local servers show up in the Providers table instead of as a
  later auth error mid-chat.
- **`zeus update` release notes**: `zeus update`/`--check` prints what's new
  in the pending release fetched from GitHub before installing.
- **`zeus bg logs` exit codes**: a background task's exit status is now
  propagated to `zeus bg logs <id>`'s own exit code, so scripts can react
  to a failed dev-server/build the way they would if it ran in the
  foreground.
- **`/export` everywhere**: the current conversation can be exported to
  Markdown from the REPL (`/export`) and the TUI, including finished
  sessions via `zeus sessions export <id>`.

## 0.1.9

- **Session storage**: `zeus sessions list`, `show <id>` (terminal
  transcript), and `export <id>` (Markdown) — saved conversations persist
  across restarts.
- **Background tasks**: `zeus bg start`, `logs`, `follow`, `kill` — long-
  running processes run detached with log capture and lifecycle management.
- **`zeus doctor`**: startup health check that verifies provider
  configuration and key presence.
- **TUI session picker**: browse and resume past sessions from within the
  TUI interface.

## 0.1.8

- **File uploads**: `ctrl+o` opens a filesystem browser to stage files for
  the agent; `/upload` attaches them to the conversation.
- **Chat failover**: automatic fallback to a secondary provider when the
  primary is unreachable; the failover provider's own model is sent on
  retry.
- **Shell completion**: bash/zsh/fish tab-completion for zeus subcommands
  and flags.
- **Session auto-resume**: reopening the TUI after a crash or restart
  offers to resume the last active session.
- **`read_document` EPUB/HTML tables**: extract text from EPUB chapters
  and render HTML tables as pipe-delimited rows; clearer binary-file hint
  when `read` hits a non-text file.
- **Provider retry**: transient HTTP/network failures are retried with
  exponential backoff; zip extraction guarded against zip-slip path
  traversal.

## 0.1.7

- **`code_graph` tool**: real call graph (who-calls-who) built from
  tree-sitter AST parsing across 9 languages — not text search, so it
  catches indirect calls that grep would miss.
- **TUI audit fixes**: help discoverability, reduced-motion support,
  performance improvements, and `/bg` task visibility enhancements.
- **Auto mode on plan execution**: confirming a `/plan` result now
  switches to Auto mode automatically.
- **`understand_repo` improvements**: deterministic project fingerprint
  cached per session; "no relevant modules" nudge for the model to verify
  with grep before writing new files.

## 0.1.6

- **Copy-on-select**: `ctrl+c` copies the current selection to the
  system clipboard in the TUI.
- **`/suggest` context**: the `/suggest` command accepts an optional
  context string to anchor recommendations to recently-completed work.
- **Adaptive agent loop**: the tool-call budget extends dynamically when
  the model keeps making novel calls, and repeated identical calls stop
  the turn early — faster convergence on simple tasks, no artificial cap
  on legitimate multi-step work.

## 0.1.5

- **Resumable update downloads**: `zeus update` resumes interrupted
  downloads via HTTP Range requests instead of restarting from zero.
- **Completion notification**: a notification is shown when a background
  update download finishes.

## 0.1.4

- **Current-directory scoping**: `zeus` now scopes itself to the real
  current directory, never widening to a false broader environment when
  launched from a subdirectory or symlink.

## 0.1.3

- **`zeus update` self-replace**: the update command now replaces the
  binary in place regardless of install location (npm global, system PATH,
  custom directory).
- **Windows gate fix**: `ProcessEntry32W`'s `Default` impl gated behind
  `#[cfg(windows)]` to fix non-Windows compilation.

## 0.1.2

- **Local model management**: `zeus pull hf` downloads GGUF models from
  Hugging Face (resumable via HTTP Range); `zeus serve` auto-downloads and
  runs a llama.cpp server with zero separate setup.
- **Ollama registry support**: download models from Ollama's registry
  alongside Hugging Face.
- **Background task improvements**: `DETACHED_PROCESS` on Windows so `bg
  run` returns immediately; suspend/resume freezes the whole process tree
  including worker children.
- **Gemini fixes**: `thought_signature` echoed on tool-call follow-ups;
  default model updated from deprecated `gemini-2.0-flash` to
  `gemini-3.6-flash`; TOML `base_url` aligned.
- **Bash tool**: full shell execution with bounded timeout, sandbox, and
  permission gate; background mode for long-running processes.
- **Windows PID liveness**: guard against PID reuse via process creation
  time so `bg follow` doesn't hang past exit on loaded machines.
- **Provider robustness**: retry transient HTTP failures; guard zip
  extraction against zip-slip; bare OpenRouter catalog IDs handled.
