# zeus

Database-free AI coding agent. **The filesystem is the source of truth.**
Published on npm as [`zeus-code`](https://www.npmjs.com/package/zeus-code).

## Features

- **Terminal-native TUI** — a full ratatui chat interface: streaming replies,
  syntax-highlighted diffs (including a side-by-side `/diff` view) and code
  blocks, a command palette (`/` or `ctrl+p`), `@model` autocomplete,
  multiline input, theme presets, message queuing while a turn is in flight,
  and session search/resume.
- **Three agent modes** — `Plan` (read-only research), `Build` (full tool
  access), and `Auto` (a continuous tool-calling loop that keeps going until
  the request is genuinely done), cycled with Tab.
- **43 specialist personas across ~6 departments**, reachable in Auto mode
  via a model-invoked `delegate` tool or a full `/plan`-driven orchestrated
  run, or directly with `/agents`; an approved-but-interrupted plan resumes
  where it left off instead of re-planning from scratch.
- **Provider-agnostic** — Anthropic, OpenAI, Gemini, Grok, DeepSeek,
  OpenRouter, OpenCodeZen, and local runners (Ollama, LM Studio, llama.cpp),
  with in-app key entry, a live model picker, and automatic fallback to
  whatever local server is actually reachable.
- **Local model management** — download GGUF models straight from Hugging
  Face (`zeus pull hf`, resumable via HTTP `Range` on a dropped connection)
  or Ollama's registry, then serve one directly through an auto-downloaded
  llama.cpp server (`zeus serve`) with zero separate setup.
- **Filesystem-first safety core** — permission gates, path containment,
  checkpoints + rewind, and a database-free `.agent/` project state instead
  of a hidden daemon or cloud session store.
- **Git, code intelligence, and RAG search** — 24-operation git integration
  with AI commit messages and PR support, a tree-sitter-backed symbol index
  (definitions/references/rename across 9 languages), and hybrid
  keyword+vector search over the codebase for "find code about X" queries.
- **Language/framework aware** — detects a project's stack across 41
  languages and 13 frameworks to ground the agent's context, and can
  scaffold a real, buildable starter project from scratch for 38 of them.
- **Extensive DevOps tool integration** — Docker, Kubernetes, Terraform,
  GitHub, AWS/Azure/GCP, Vercel, Supabase, and a dozen more platform CLIs are
  wired in as first-class tools alongside the core file/git/search set.
- **Background orchestration** — `zeus bg` runs long tasks (including a full
  orchestrated `--auto` run) detached, so you can check on, pause, resume,
  or stop them independently of the TUI.
- **Android device testing** — drive a real device or emulator over `adb`
  directly from the agent: install/launch, screenshot/screenrecord, logcat,
  UI input automation, and file push/pull.
- **Self-diagnosing** — `zeus doctor` checks every configured provider's
  readiness (key presence, or live reachability for local runners), not
  just the default one.

## Status

Phases 1–3 (CLI/config/logging/provider abstraction, permissions/filesystem
ops/checkpoints/search, and the core agent loop) are done. Multi-agent
orchestration, cloud providers, code intelligence, RAG search, scaffold
detection/generation, local model management, and DevOps tool integration
have all since landed too — see Features above for what's actually shipped
rather than a phase-by-phase breakdown.

## Install

Prebuilt binaries for Windows, macOS (Apple Silicon + Intel), and Linux x86_64
are attached to each [release](https://github.com/PositiveMinds/Zeus-CLI-releases).
The binaries are published from the private source repo to this public mirror, so
no Rust toolchain is required to install.

**npm** (any platform, if you already have Node — this is the cleanest option):
```bash
npm install -g zeus-code
```
Installs a small wrapper that fetches the right prebuilt binary for your OS/arch
as an optional dependency (same trick esbuild/swc/opencode use) — see
[`npm/README.md`](./npm/README.md) for how the packages are built and published.

**PowerShell** (Windows 10/11):
```powershell
irm https://raw.githubusercontent.com/PositiveMinds/Zeus-CLI-releases/main/install.ps1 | iex
```

**cmd** (Windows):
```batch
curl -L https://raw.githubusercontent.com/PositiveMinds/Zeus-CLI-releases/main/install.bat | cmd
```

Either Windows installer puts `zeus` in `%LOCALAPPDATA%\zeus` and adds it to your
user PATH. Pin a specific version with `$env:ZEUS_VERSION = "0.1.0"` before running
the PowerShell installer.

**macOS / Linux**:
```bash
curl -fsSL https://raw.githubusercontent.com/PositiveMinds/Zeus-CLI-releases/main/install.sh | sh
```
Puts `zeus` in `~/.local/share/zeus/bin` and prints a one-time PATH hint. Pin a
specific version with `ZEUS_VERSION=0.1.0 curl ... | sh`.

**From source** (all platforms, requires Rust):
```bash
cargo install --git https://github.com/PositiveMinds/Zeus-CLI
```

Verify with `zeus doctor`.

## Quick start

```bash
# Build
cargo build --release

# Initialize global home (~/.zeus) + project .agent/
cargo run -p zeus-cli -- init
cargo run -p zeus-cli -- init --with-project

# Inspect
cargo run -p zeus-cli -- doctor
cargo run -p zeus-cli -- config show

# One-shot chat with your configured provider (e.g. local Ollama)
cargo run -p zeus-cli -- chat "hello" --provider ollama

# Safe file ops (permission prompts; use --yes to auto-approve this process only)
cargo run -p zeus-cli -- write notes.txt "hello" --yes
cargo run -p zeus-cli -- read notes.txt
cargo run -p zeus-cli -- edit notes.txt hello world --yes
cargo run -p zeus-cli -- grep "world" --glob "*.txt"
cargo run -p zeus-cli -- glob "**/*.rs"
cargo run -p zeus-cli -- checkpoints
cargo run -p zeus-cli -- rewind <turn-id>
```

Binary name: `zeus` (package `zeus-cli`). Run `zeus --help` for the full
command list — code intelligence (`zeus codeint`), RAG search
(`zeus ragindex`), project scaffolding (`zeus project scaffold --list`),
local model serving (`zeus serve`), and Android device testing all have
their own subcommands beyond what's shown here.

## Layout

```text
~/.zeus/
├── config.toml
├── providers.toml
├── settings.toml
├── sessions/  memory/  cache/  logs/  plugins/  commands/  prompts/

<project>/.agent/
├── settings.toml           # shared (checked in)
├── settings.local.toml     # personal (gitignored)
├── memory.md  tasks.json  index.json
├── hooks/  commands/  checkpoints/
```

**Config resolution** (highest last): builtin safe defaults → global `settings.toml` → project `settings.toml` → `settings.local.toml`.

Override home for tests: `ZEUS_HOME=/tmp/zeus-test`.

## Workspace crates

```text
crates/
  zeus-cli/       # binary: TUI + one-shot subcommands
  zeus-agent/     # agent loop, tools, orchestration, personas
  zeus-config/    # paths + layered TOML
  zeus-logging/   # tracing + file logs
  zeus-provider/  # ModelProvider trait + real backends
  zeus-fs/        # permission gate, file ops, checkpoints, search, device/git
  zeus-lang/      # language/framework detection + scaffolding
  zeus-rag/       # database-free chunk index + hybrid search
```

## Safety model (Phase 2)

- **Permission Gate**: allow / ask / deny per tool, path glob, and command pattern.
- **Delete** always asks (no silent-allow tier).
- **Path containment**: operations cannot escape the project root.
- **Must-read-before-write** for existing files; stale-hash checks refuse silent clobber.
- **Checkpoints**: every mutating op snapshots prior state under `.agent/checkpoints/<turn-id>/`.
- Session auto-approve (`--yes` or interactive `s`) is process-only — never persisted.

## Cloud providers & modes

Besides a local Ollama/LM-Studio-style provider, zeus
supports cloud LLMs via OpenAI-compatible and native Anthropic routes:

- **OpenAI-compatible**: OpenAI, Grok (x.ai), OpenRouter, OpenCode Zen,
  DeepSeek, Gemini — configured under `providers.toml`
  (`kind = "openai_compat"`, `base_url`, `api_key_env`, `default_model`,
  optional `headers`).
- **Anthropic native** (`kind = "anthropic"`, `/v1/messages`, `x-api-key`).

API keys come from an env var named by `api_key_env`, or an embedded
`header` value; a missing key surfaces a clear error instead of silently
failing at runtime.

## Agent modes

`Tab` (TUI) or `/mode build|plan|auto` toggles how a turn runs:

- **build** — a single tool-using turn (default).
- **plan** — one read-only planning turn, no file writes.
- **auto** — plan, then execute each planned step through the orchestrator.

## Multi-agent orchestration

`/plan` breaks a goal into subtasks and dispatches each to a specialist
`Persona` (an Architect, Backend, QA, … modeled loosely on an "AI company").
After the steps complete, a matching **reviewer** persona runs one read-only
review pass over the combined result.

- `/agents` lists the specialist roster grouped by department (`/agents count`
  for the total).
- **Safe bounded parallelism**: consecutive read-only steps run as concurrent
  headless provider calls (cap via `max_parallel_read_steps`); file-mutating
  steps stay sequential to avoid edit races.
- **Custom personas**: drop `*.toml` files in `~/.zeus/personas/` to extend or
  shadow the built-in roster (pure prompt data, loaded once at startup).

## Tests

```bash
cargo test --workspace
```

## Coding standards

- Rust, async-first, test-driven
- Provider-agnostic, plugin-first
- No application database
- Destructive actions always go through the Permission Gate
