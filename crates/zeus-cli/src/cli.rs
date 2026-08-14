//! Clap CLI definitions for zeus.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use zeus_fs::ResetMode;

#[derive(Debug, Parser)]
#[command(
    name = "zeus",
    version,
    about = "Database-free AI coding agent — filesystem is the source of truth",
    long_about = None
)]
pub struct Cli {
    /// Override project root (defaults to auto-detect from cwd)
    #[arg(long = "project-root", global = true, value_name = "PATH")]
    pub project_root: Option<PathBuf>,

    /// Log level override (trace|debug|info|warn|error)
    #[arg(long, global = true, env = "ZEUS_LOG")]
    pub log_level: Option<String>,

    /// Auto-approve permission prompts for this process only (never persisted)
    #[arg(long, global = true)]
    pub yes: bool,

    /// No subcommand: start an interactive REPL session instead of a
    /// one-shot command.
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Initialize global (~/.zeus) and optional project (.agent) config
    Init {
        /// Also create .agent/ in the current (or --project-root) directory
        #[arg(long)]
        with_project: bool,
    },

    /// Show resolved configuration
    Config {
        #[command(subcommand)]
        action: ConfigCmd,
    },

    /// One-shot chat with the configured provider
    Chat {
        /// User message
        message: String,
        /// Provider name from providers.toml
        #[arg(long)]
        provider: Option<String>,
        /// Model override
        #[arg(long)]
        model: Option<String>,
        /// Disable streaming (single response)
        #[arg(long)]
        no_stream: bool,
    },

    /// List models from a provider
    Models {
        /// Provider name (default: settings.model.provider)
        #[arg(long)]
        provider: Option<String>,
        /// Scan local model files across the system instead of querying a
        /// provider's API — finds downloaded-but-not-yet-served models too.
        #[arg(long)]
        local: bool,
        /// Copy a model file from the `--local` listing into the zeus model
        /// library (~/.zeus/models), matched on the exact path shown.
        /// Pair with `--move` to relocate it instead of copying.
        #[arg(long, value_name = "PATH")]
        import: Option<PathBuf>,
        /// With `--import`, move the model file instead of copying it.
        #[arg(long = "move")]
        relocate: bool,
    },

    /// Run one turn through the full Agent Loop (tool-calling, context
    /// compaction, session persistence) rather than a raw one-shot chat.
    Agent {
        /// User message
        message: String,
        /// Provider name from providers.toml
        #[arg(long)]
        provider: Option<String>,
        /// Model override
        #[arg(long)]
        model: Option<String>,
        /// Resume an existing session id instead of starting a new one
        #[arg(long)]
        session: Option<String>,
        /// Resume the most recently used saved session instead of starting a
        /// new one (ignored if `--session` is also given).
        #[arg(long)]
        resume: bool,
        /// Plan mode: research the request read-only, persist a structured
        /// plan to .agent/tasks.json, and exit WITHOUT executing anything.
        #[arg(long)]
        plan: bool,
        /// Auto mode: run the goal as a full orchestrated plan — split into
        /// steps, execute each as its own tool-using turn, run a lead-reviewer
        /// gate, and finish. Auto-approves every gate when combined with
        /// `--yes`; the headless counterpart of `/plan` + Auto mode. Designed
        /// to be spawned in the background (`zeus bg orchestrate`).
        #[arg(long)]
        auto: bool,
        /// Run a named multi-specialist workflow (from .agent/workflows or
        /// ~/.zeus/workflows) against the message as its goal.
        #[arg(long, value_name = "NAME")]
        workflow: Option<String>,
    },

    /// List saved sessions (id, message count, last user message)
    Sessions,

    /// Check for (and optionally install) a newer zeus release
    Update {
        /// Only report whether a newer version is available; don't install it
        #[arg(long)]
        check: bool,
    },

    /// Set or list provider API keys (~/.zeus/keys.toml) without opening a
    /// session — the one-shot equivalent of the REPL's `/provider key`
    Key {
        #[command(subcommand)]
        action: KeyCmd,
    },

    /// Count tokens for a message (context-budget helper)
    Tokens {
        message: String,
        #[arg(long)]
        provider: Option<String>,
        #[arg(long)]
        model: Option<String>,
    },

    /// Read a project file (line-numbered)
    Read {
        path: PathBuf,
        #[arg(long)]
        offset: Option<usize>,
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Write/create a project file
    Write {
        path: PathBuf,
        /// Content to write (use - for stdin)
        content: String,
    },

    /// Targeted string replace in a file
    Edit {
        path: PathBuf,
        old: String,
        new: String,
        #[arg(long)]
        replace_all: bool,
    },

    /// Delete a file or directory (always asks unless --yes)
    Rm { path: PathBuf },

    /// Rename or move a file/directory (git mv-aware)
    Mv { from: PathBuf, to: PathBuf },

    /// Copy a file
    Cp {
        from: PathBuf,
        to: PathBuf,
        #[arg(long)]
        overwrite: bool,
    },

    /// Pattern-based search+replace across multiple files (previews scope, then one approval to apply)
    BulkEdit {
        /// Files or directories to search within
        roots: Vec<PathBuf>,
        #[arg(long)]
        old: String,
        #[arg(long)]
        new: String,
        #[arg(long)]
        replace_all: bool,
        /// Only show which files would change; don't apply
        #[arg(long)]
        dry_run: bool,
    },

    /// Search file contents (regex)
    Grep {
        pattern: String,
        #[arg(long)]
        glob: Option<String>,
        #[arg(long, short = 'i')]
        ignore_case: bool,
        #[arg(long, default_value_t = 50)]
        max: usize,
        #[arg(long)]
        path: Option<PathBuf>,
    },

    /// Find files by glob
    Glob {
        pattern: String,
        #[arg(long, default_value_t = 100)]
        max: usize,
    },

    /// Restore files from a checkpoint turn
    Rewind { turn_id: String },

    /// List checkpoint turns
    Checkpoints,

    /// Print version and capability summary
    Doctor,

    /// Manage background (long-running/detached) commands
    Bg {
        #[command(subcommand)]
        action: BgCmd,
    },

    /// Git operations, permission-gated per the blueprint's tiering
    Git {
        #[command(subcommand)]
        action: GitCmd,
    },

    /// Download a model — no browser needed
    Pull {
        #[command(subcommand)]
        source: PullCmd,
    },

    /// Serve a local GGUF model via llama.cpp: `zeus serve <model-name>` or
    /// `zeus serve <repo>/<file.gguf>` (auto-downloads if missing).
    Serve {
        /// Model name from [settings.llamacpp.models], or `repo/file.gguf`
        model: Option<String>,
    },

    /// Manage user-defined slash command templates (.agent/commands/,
    /// ~/.zeus/commands/) — distinct from the built-in REPL /help etc.
    UserCommand {
        #[command(subcommand)]
        action: UserCommandCmd,
    },

    /// Code Intelligence — database-free symbol index & reference tools.
    /// Build `.agent/index.json` then query definitions/references.
    Codeint {
        #[command(subcommand)]
        action: CodeintCmd,
    },

    /// RAG index — database-free semantic retrieval over source files.
    /// Build/refresh `.agent/rag_index.json`, optionally embedding chunks,
    /// then hybrid-search it from the terminal.
    Ragindex {
        #[command(subcommand)]
        action: RagindexCmd,
    },

    /// Language support — detect a project's language, show its standard
    /// dev commands (build / test / lint / format), scaffold a minimal
    /// buildable skeleton, or format a file / project.
    Project {
        #[command(subcommand)]
        action: ProjectCmd,
    },
}

/// Sub-actions for `zeus project`.
#[derive(Debug, Subcommand)]
pub enum ProjectCmd {
    /// Detect the primary language of the project root (or fail).
    Detect,
    /// Show the project language and its standard build / test / lint /
    /// format commands.
    Commands {
        /// Restrict to a specific language instead of auto-detecting.
        lang: Option<String>,
    },
    /// Scaffold a minimal, buildable project skeleton.
    Scaffold {
        /// Language (or file extension) to scaffold — e.g. "rust", "ts",
        /// "go", "c#".
        lang: String,
        /// Project / module name (used for package names, classes, etc.).
        name: String,
    },
    /// Format a single source file (per-language formatter) — or the whole
    /// project with no path. Requires the language's formatter on PATH.
    Format {
        /// Target file. Omit to run the project-wide formatter.
        path: Option<PathBuf>,
    },
}

/// Sub-actions for `zeus codeint` (Phase 6 — database-free index).
#[derive(Debug, Subcommand)]
pub enum CodeintCmd {
    /// Scan the project's source files and write `.agent/index.json`
    Index {
        /// Rebuild even if a fresh-enough index already exists.
        #[arg(long)]
        force: bool,
    },
    /// Find definitions in the index by symbol name (substring,
    /// case-insensitive).
    Find {
        /// Symbol name (or prefix) to search for.
        name: String,
    },
    /// Go-to-definition — same as `find` but shows the resolved primary
    /// definition for the symbol.
    Defs { name: String },
    /// Find references to a symbol across the project (and any configured
    /// extra project roots) via ripgrep.
    Refs {
        name: String,
        /// Only touch files matching this glob.
        #[arg(long)]
        glob: Option<String>,
        /// Case-insensitive reference search.
        #[arg(long, short = 'i')]
        ignore_case: bool,
        /// Cap number of reported hits.
        #[arg(long, default_value_t = 200)]
        max: usize,
    },
    /// Propose a reference-update plan for renaming `old` -> `new`
    /// (word-boundary). Prints the per-file proposal; applying writes is
    /// intentionally left to a review step because it mutates many files.
    Rename {
        old: String,
        #[arg(long)]
        new: String,
    },
}

/// Sub-actions for `zeus ragindex` (database-free RAG, no SQL).
#[derive(Debug, Subcommand)]
pub enum RagindexCmd {
    /// Scan source files and write `.agent/rag_index.json`. When a stale
    /// index exists it is refreshed incrementally — only changed files are
    /// re-chunked; untouched chunks are kept.
    Index {
        /// Rebuild from scratch even if a fresh index already exists.
        #[arg(long)]
        force: bool,
        /// Embed every chunk with the configured provider (best-effort:
        /// without a reachable embeddings-capable provider the index stays
        /// keyword-only).
        #[arg(long)]
        embed: bool,
        /// Provider name from providers.toml (used with --embed).
        #[arg(long)]
        provider: Option<String>,
        /// Model override (used with --embed).
        #[arg(long)]
        model: Option<String>,
    },
    /// Hybrid keyword + vector search over a fresh index. Prints the top
    /// matching chunks; a stale or missing index is reported instead of
    /// re-chunking silently.
    Search {
        query: String,
        /// Number of results to return.
        #[arg(long, default_value_t = 5)]
        k: usize,
    },
}

#[derive(Debug, Subcommand)]
pub enum KeyCmd {
    /// Set a provider's API key. Omit <NAME> to pick one from a numbered
    /// list instead of typing it; omit <KEY> to be prompted with input
    /// hidden (recommended interactively) rather than passing it inline.
    Set {
        /// Provider name — see `zeus key list` or providers.toml. Omit to
        /// choose from a list instead.
        name: Option<String>,
        key: Option<String>,
    },
    /// Show which configured providers have a key set
    List,
}

#[derive(Debug, Subcommand)]
pub enum UserCommandCmd {
    /// List available commands (project shadows global of the same name)
    List,
    /// Print a command's template content
    Show { name: String },
    /// Create or overwrite a command template. content: use "-" to read from stdin.
    Add {
        name: String,
        content: String,
        /// Write to ~/.zeus/commands/ instead of .agent/commands/
        #[arg(long)]
        global: bool,
    },
    /// Delete a command template
    Remove {
        name: String,
        /// Delete from ~/.zeus/commands/ instead of .agent/commands/
        #[arg(long)]
        global: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum PullCmd {
    /// Pull via Ollama's own registry (what `ollama pull` uses) — once
    /// pulled it's auto-detected via `zeus models --provider ollama`.
    Ollama { model: String },
    /// Download a file directly from a Hugging Face repo into ~/.zeus/models/
    /// (or a configured extra dir) — auto-detected via `zeus models --local`.
    Hf { repo: String, file: String },
}

#[derive(Debug, Subcommand)]
pub enum BgCmd {
    /// Start a command detached; it keeps running after this process exits
    Run { command: String },
    /// Run a goal as a full orchestrated plan in the background: the goal is
    /// dispatched to a detached headless `zeus agent --auto` process (all
    /// gates auto-approved), so you keep your prompt while the workforce
    /// plans, executes, and lead-reviews. Track with `zeus bg list` /
    /// `zeus bg output <id>` / `zeus bg stop <id>`.
    Orchestrate {
        /// The goal for the orchestrated run.
        goal: String,
        /// Run a named workflow (from .agent/workflows or ~/.zeus/workflows)
        /// instead of the auto planner.
        #[arg(long)]
        workflow: Option<String>,
    },
    /// List background tasks and their running/paused/exited status
    List,
    /// Print captured stdout/stderr for a background task
    Output { id: u64 },
    /// Suspend a running task in place (resume continues it where it left off)
    Pause { id: u64 },
    /// Continue a previously-paused task
    Resume { id: u64 },
    /// Stop a running background task
    Stop { id: u64 },
}

#[derive(Debug, Subcommand)]
pub enum GitCmd {
    /// git status (porcelain + branch info)
    Status,
    /// git diff (working tree, or --staged for the index)
    Diff {
        #[arg(long)]
        staged: bool,
        refs: Vec<String>,
    },
    /// git log --oneline
    Log {
        #[arg(long, default_value_t = 20)]
        max: usize,
        path: Option<String>,
    },
    /// git show <target>
    Show { target: String },
    /// git blame <path>
    Blame { path: String },
    /// List local and remote branches
    Branches,
    /// List configured remotes
    Remotes,
    /// List tags
    Tags,
    /// List stash entries
    Stashes,
    /// Stage one or more paths
    Add { paths: Vec<String> },
    /// Commit staged changes (or all tracked changes with --all)
    Commit {
        message: String,
        #[arg(long)]
        all: bool,
    },
    /// Stash the working tree
    Stash {
        #[arg(long)]
        message: Option<String>,
    },
    /// Apply and drop the most recent stash entry
    StashPop,
    /// Create a branch at HEAD
    BranchCreate { name: String },
    /// Delete a branch (--force for an unmerged branch)
    BranchDelete {
        name: String,
        #[arg(long)]
        force: bool,
    },
    /// Create a tag, annotated if --message is given
    TagCreate {
        name: String,
        #[arg(long)]
        message: Option<String>,
    },
    /// Check out an existing branch or commit
    Checkout { target: String },
    /// Fetch from a remote without merging
    Fetch { remote: Option<String> },
    /// git pull
    Pull,
    /// git push (--force is denied by a built-in safety rule)
    Push {
        remote: Option<String>,
        branch: Option<String>,
        #[arg(long)]
        force: bool,
    },
    /// Commit (all changes by default) and push to the current branch's upstream
    CommitAndPush {
        message: String,
        #[arg(short, long)]
        all: bool,
        #[arg(long)]
        remote: Option<String>,
    },
    /// git reset (mode=hard is denied by a built-in safety rule)
    Reset {
        #[arg(value_enum)]
        mode: ResetModeArg,
        target: Option<String>,
    },
    /// Create a new commit that undoes the given commit
    Revert { target: String },
    /// Apply the changes from one commit onto the current branch
    CherryPick { target: String },
    /// Rebase the current branch onto another (rewrites history)
    Rebase { onto: String },
    /// Merge a branch into the current one
    Merge { branch: String },
    /// List pull requests (needs `gh` CLI)
    PrList {
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long, default_value = "open")]
        state: String,
    },
    /// Show a pull request (needs `gh` CLI)
    PrView { number: String },
    /// Create a pull request for the current branch (needs `gh` CLI; branch must be pushed)
    PrCreate {
        title: String,
        #[arg(long)]
        body: Option<String>,
        #[arg(long)]
        base: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum ResetModeArg {
    Soft,
    Mixed,
    Hard,
}

impl From<ResetModeArg> for ResetMode {
    fn from(a: ResetModeArg) -> Self {
        match a {
            ResetModeArg::Soft => ResetMode::Soft,
            ResetModeArg::Mixed => ResetMode::Mixed,
            ResetModeArg::Hard => ResetMode::Hard,
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum ConfigCmd {
    /// Print merged settings and paths
    Show {
        /// Emit JSON-ish debug format
        #[arg(long)]
        debug: bool,
    },
    /// Print path to global home
    Path,
    /// Read a dotted key (e.g. "model.provider") from one settings.toml
    /// layer — project if active, else global. This is *not* the fully
    /// merged view `config show` prints; it's the raw on-disk value in
    /// that one file.
    Get { key: String },
    /// Write a dotted key to one settings.toml layer (project if active
    /// and --global isn't passed, else global), creating the file/table
    /// path as needed. Value type is inferred: "true"/"false" -> bool,
    /// parses as an integer or float -> number, else a string.
    Set {
        key: String,
        value: String,
        /// Write to the global settings.toml even inside a project
        #[arg(long)]
        global: bool,
    },
}
