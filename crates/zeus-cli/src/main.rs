//! zeus — database-free AI coding agent CLI.

mod clipboard;
mod decor;
mod highlight;
mod tui;
mod ui;
mod update;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use std::io::{self, IsTerminal, Read as IoRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tracing::{error, info, warn};
use zeus_agent::{
    discover_workflows, load_custom_personas, personas_by_department, Agent, AgentEvent,
    AgentOptions, BackgroundTaskRegistry, ContextManager, ExpandResult, HookRunner, McpClient,
    SessionStore, SlashCommands, TerminalRunner, ToolManager, TurnResult,
};
use zeus_config::{Config, KeysFile};
use zeus_fs::{filter_out_own_index, word_boundary, IndexEngine, SymbolIndex};
use zeus_fs::{
    ApprovalDecision, CopyOptions, EditOptions, GitEngine, PermissionGate, PermissionRequest,
    ReadOptions, ResetMode, SearchOptions, Workspace, WriteOptions,
};
use zeus_logging::{init as init_logging, LoggingOptions};
use zeus_provider::{
    create_default, create_provider, ChatRequest, Message, ModelProvider, StreamEvent,
    UnconfiguredProvider,
};

#[derive(Debug, Parser)]
#[command(
    name = "zeus",
    version,
    about = "Database-free AI coding agent — filesystem is the source of truth",
    long_about = None
)]
struct Cli {
    /// Override project root (defaults to auto-detect from cwd)
    #[arg(long = "project-root", global = true, value_name = "PATH")]
    project_root: Option<PathBuf>,

    /// Log level override (trace|debug|info|warn|error)
    #[arg(long, global = true, env = "ZEUS_LOG")]
    log_level: Option<String>,

    /// Auto-approve permission prompts for this process only (never persisted)
    #[arg(long, global = true)]
    yes: bool,

    /// No subcommand: start an interactive REPL session instead of a
    /// one-shot command.
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
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
enum ProjectCmd {
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
enum CodeintCmd {
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

#[derive(Debug, Subcommand)]
enum KeyCmd {
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
enum UserCommandCmd {
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
enum PullCmd {
    /// Pull via Ollama's own registry (what `ollama pull` uses) — once
    /// pulled it's auto-detected via `zeus models --provider ollama`.
    Ollama { model: String },
    /// Download a file directly from a Hugging Face repo into ~/.zeus/models/
    /// (or a configured extra dir) — auto-detected via `zeus models --local`.
    Hf { repo: String, file: String },
}

#[derive(Debug, Subcommand)]
enum BgCmd {
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
enum GitCmd {
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
enum ResetModeArg {
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
enum ConfigCmd {
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

fn main() {
    // Run the async entrypoint on a thread with a large stack. clap's
    // command construction plus the debug build's fat frames otherwise exceed
    // the OS default 1 MiB main-thread stack, producing a stack overflow.
    let result = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .name("zeus-main".into())
        .spawn(|| {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("build runtime");
            rt.block_on(async_main());
        })
        .expect("spawn zeus-main")
        .join();

    if result.is_err() {
        std::process::exit(101);
    }
}

async fn async_main() {
    if let Err(e) = run().await {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    if matches!(cli.command, Some(Commands::Init { .. })) {
        let _ = init_logging(LoggingOptions {
            level: cli.log_level.clone().unwrap_or_else(|| "info".into()),
            file: false,
            logs_dir: None,
            console: true,
        });
        return cmd_init(&cli).await;
    }

    let config = load_config(&cli)?;
    let level = cli
        .log_level
        .clone()
        .unwrap_or_else(|| config.settings.logging.level.clone());
    // No subcommand + a real terminal on both ends is exactly the condition
    // `cmd_repl` uses to hand off to the raw-mode TUI (see `tui::run` below)
    // — a stray log line written to stderr mid-session corrupts that
    // screen, so the console layer must be off before the TUI ever starts.
    // File logging (if configured) still captures everything either way.
    let entering_tui =
        cli.command.is_none() && io::stdin().is_terminal() && io::stdout().is_terminal();
    let _ = init_logging(LoggingOptions {
        level,
        file: config.settings.logging.file,
        logs_dir: Some(config.global.logs.clone()),
        console: !entering_tui,
    });

    match cli.command {
        None => cmd_repl(&config, cli.yes).await,
        Some(Commands::Init { .. }) => unreachable!(),
        Some(Commands::Config { action }) => cmd_config(&config, action),
        Some(Commands::Chat {
            message,
            provider,
            model,
            no_stream,
        }) => cmd_chat(&config, message, provider, model, !no_stream).await,
        Some(Commands::Models {
            provider,
            local,
            import,
            relocate,
        }) => cmd_models(&config, provider, local, import, relocate).await,
        Some(Commands::Agent {
            message,
            provider,
            model,
            session,
            resume,
            plan,
            auto,
            workflow,
        }) => {
            cmd_agent(
                &config, message, provider, model, session, resume, plan, auto, workflow, cli.yes,
            )
            .await
        }
        Some(Commands::Sessions) => cmd_sessions(&config),
        Some(Commands::Update { check }) => update::cmd_update(check).await,
        Some(Commands::Key { action }) => cmd_key(&config, action),
        Some(Commands::Tokens {
            message,
            provider,
            model,
        }) => cmd_tokens(&config, message, provider, model).await,
        Some(Commands::Read {
            path,
            offset,
            limit,
        }) => cmd_read(&config, path, offset, limit),
        Some(Commands::Write { path, content }) => cmd_write(&config, path, content, cli.yes),
        Some(Commands::Edit {
            path,
            old,
            new,
            replace_all,
        }) => cmd_edit(&config, path, old, new, replace_all, cli.yes),
        Some(Commands::Rm { path }) => cmd_rm(&config, path, cli.yes),
        Some(Commands::Mv { from, to }) => cmd_mv(&config, from, to, cli.yes),
        Some(Commands::Cp {
            from,
            to,
            overwrite,
        }) => cmd_cp(&config, from, to, overwrite, cli.yes),
        Some(Commands::BulkEdit {
            roots,
            old,
            new,
            replace_all,
            dry_run,
        }) => cmd_bulk_edit(&config, roots, old, new, replace_all, dry_run, cli.yes),
        Some(Commands::Grep {
            pattern,
            glob,
            ignore_case,
            max,
            path,
        }) => cmd_grep(&config, pattern, glob, ignore_case, max, path),
        Some(Commands::Glob { pattern, max }) => cmd_glob(&config, pattern, max),
        Some(Commands::Rewind { turn_id }) => cmd_rewind(&config, turn_id),
        Some(Commands::Checkpoints) => cmd_checkpoints(&config),
        Some(Commands::Doctor) => cmd_doctor(&config).await,
        Some(Commands::Bg { action }) => cmd_bg(&config, action),
        Some(Commands::Git { action }) => cmd_git(&config, action, cli.yes),
        Some(Commands::Pull { source }) => cmd_pull(&config, source).await,
        Some(Commands::Serve { model }) => cmd_serve(&config, model).await,
        Some(Commands::UserCommand { action }) => cmd_user_commands(&config, action, cli.yes),
        Some(Commands::Codeint { action }) => cmd_codeint(&config, action),
        Some(Commands::Project { action }) => cmd_project(&config, action),
    }
}

fn load_config(cli: &Cli) -> Result<Config> {
    let config = if let Some(root) = &cli.project_root {
        Config::load(Some(root.as_path())).context("failed to load config")?
    } else {
        Config::load_from_cwd().context("failed to load config")?
    };
    // Custom specialist personas from ~/.zeus/personas/*.toml (once, then
    // cached globally for the process).
    load_custom_personas(&config.global.personas);
    Ok(config)
}

fn workspace(config: &Config) -> Result<Workspace> {
    Workspace::from_config(config).map_err(|e| anyhow::anyhow!(e))
}

fn approver(yes: bool) -> impl FnMut(&PermissionRequest) -> ApprovalDecision {
    move |req: &PermissionRequest| {
        if yes {
            eprintln!("[auto-approve] {}", req.description);
            if let Some(preview) = &req.preview {
                eprintln!(
                    "{}",
                    if highlight::looks_like_diff(preview) {
                        highlight::ansi_diff(preview)
                    } else {
                        preview.clone()
                    }
                );
            }
            return ApprovalDecision::Approved;
        }
        if let Some(preview) = &req.preview {
            eprintln!(
                "{}",
                if highlight::looks_like_diff(preview) {
                    highlight::ansi_diff(preview)
                } else {
                    preview.clone()
                }
            );
        }
        eprint!("Allow {}? [y/N/s(session)] ", req.description);
        let _ = io::stderr().flush();
        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_err() {
            return ApprovalDecision::Denied;
        }
        match line.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => ApprovalDecision::Approved,
            "s" | "session" => ApprovalDecision::ApprovedForSession,
            _ => ApprovalDecision::Denied,
        }
    }
}

async fn cmd_init(cli: &Cli) -> Result<()> {
    let global = Config::init_global().context("init global home")?;
    println!("Initialized global home: {}", global.root.display());
    println!("  config:    {}", global.config_toml.display());
    println!("  settings:  {}", global.settings_toml.display());
    println!("  providers: {}", global.providers_toml.display());
    println!("  logs:      {}", global.logs.display());

    let with_project = matches!(cli.command, Some(Commands::Init { with_project: true }));
    if with_project {
        let root = cli
            .project_root
            .clone()
            .unwrap_or(std::env::current_dir().context("cwd")?);
        let proj = Config::init_project(&root).context("init project")?;
        println!("Initialized project agent dir: {}", proj.root.display());
        println!("  settings: {}", proj.settings_toml.display());
        println!("  memory:   {}", proj.memory_md.display());
        println!("  tasks:    {}", proj.tasks_json.display());
    }
    Ok(())
}

fn cmd_config(config: &Config, action: ConfigCmd) -> Result<()> {
    match action {
        ConfigCmd::Path => {
            println!("{}", config.global.root.display());
            Ok(())
        }
        ConfigCmd::Show { debug } => {
            println!("global_home: {}", config.global.root.display());
            if let Some(root) = &config.project_root {
                println!("project_root: {}", root.display());
            } else {
                println!("project_root: (none)");
            }
            println!("model.provider: {}", config.settings.model.provider);
            println!("model.model: {}", config.settings.model.model);
            println!("logging.level: {}", config.settings.logging.level);
            println!(
                "context.compact_threshold: {}",
                config.settings.context.compact_threshold
            );
            println!(
                "permissions.defaults: {} entries",
                config.settings.permissions.defaults.len()
            );
            println!(
                "permissions.rules: {} entries",
                config.settings.permissions.rules.len()
            );
            println!("providers: {}", config.providers.providers.len());
            for name in config.providers.providers.keys() {
                let kind = &config.providers.providers[name].kind;
                println!("  - {name} (kind={kind})");
            }
            if debug {
                println!("\n# settings debug\n{:#?}", config.settings);
            }
            Ok(())
        }
        ConfigCmd::Get { key } => {
            let path = settings_file_path(config, false);
            let doc = load_toml_or_empty(&path)?;
            let parts: Vec<&str> = key.split('.').collect();
            match get_toml_path(&doc, &parts) {
                Some(v) => println!("{v}"),
                None => println!("(not set in {})", path.display()),
            }
            Ok(())
        }
        ConfigCmd::Set { key, value, global } => {
            let path = settings_file_path(config, global);
            let mut doc = load_toml_or_empty(&path)?;
            let parts: Vec<&str> = key.split('.').collect();
            set_toml_path(&mut doc, &parts, parse_toml_scalar(&value));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).context("create settings dir")?;
            }
            let text = toml::to_string_pretty(&doc).context("serialize settings")?;
            std::fs::write(&path, text).context("write settings file")?;
            println!("set {key} = {value}  ({})", path.display());
            Ok(())
        }
    }
}

/// Which settings.toml file `config get`/`config set` operate on: the
/// project's (checked-in, shared) file if a project is active and `global`
/// wasn't forced, else the global one. These commands read/write *one*
/// on-disk layer directly (as raw TOML), not the fully merged view `config
/// show` prints — simpler, and correct for "I want to change this project's
/// checked-in setting" without needing typed structs for every field.
fn settings_file_path(config: &Config, global: bool) -> PathBuf {
    if !global {
        if let Some(project) = &config.project {
            return project.settings_toml.clone();
        }
    }
    config.global.settings_toml.clone()
}

fn load_toml_or_empty(path: &Path) -> Result<toml::Value> {
    if !path.exists() {
        return Ok(toml::Value::Table(toml::value::Table::new()));
    }
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn get_toml_path<'a>(root: &'a toml::Value, parts: &[&str]) -> Option<&'a toml::Value> {
    let mut current = root;
    for part in parts {
        current = current.get(part)?;
    }
    Some(current)
}

fn set_toml_path(root: &mut toml::Value, parts: &[&str], value: toml::Value) {
    if !root.is_table() {
        *root = toml::Value::Table(toml::value::Table::new());
    }
    let table = root.as_table_mut().expect("just ensured it's a table");
    if parts.len() == 1 {
        table.insert(parts[0].to_string(), value);
        return;
    }
    let entry = table
        .entry(parts[0].to_string())
        .or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
    set_toml_path(entry, &parts[1..], value);
}

/// Infer a TOML scalar type from a plain CLI string: "true"/"false" -> bool,
/// parses as an integer or float -> number, else a plain string.
fn parse_toml_scalar(s: &str) -> toml::Value {
    if let Ok(b) = s.parse::<bool>() {
        return toml::Value::Boolean(b);
    }
    if let Ok(i) = s.parse::<i64>() {
        return toml::Value::Integer(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return toml::Value::Float(f);
    }
    toml::Value::String(s.to_string())
}

/// Resolve a provider by name. When the caller didn't explicitly request
/// one (`provider: None`, using the configured default), and that default
/// is a local server kind (Ollama/LM Studio/llama.cpp) that isn't actually
/// running right now, this auto-detects another local server that *is*
/// running rather than failing with an opaque connection error. If none are
/// reachable it returns a clear, actionable error. An explicit `--provider`
/// is always respected as-is; auto-detection only kicks in for the default.
async fn resolve_provider(
    config: &Config,
    provider: Option<String>,
) -> Result<std::sync::Arc<dyn ModelProvider>> {
    let explicit = provider.is_some();
    let name = provider.unwrap_or_else(|| config.settings.model.provider.clone());

    if !explicit {
        if let Some(cfg) = config.providers.get(&name) {
            if !zeus_provider::is_provider_reachable(cfg, std::time::Duration::from_millis(800))
                .await
            {
                // Default provider is llama.cpp but no server is running yet:
                // auto-download the llama-server binary + model file and launch
                // one, so `zeus chat`/the TUI "just works" for local models.
                if cfg.kind == "llamacpp" {
                    if let Some(entry) = zeus_provider::resolve_local_model(
                        &config.settings.llamacpp,
                        &config.settings.model.model,
                    ) {
                        match zeus_provider::serve(
                            &config.settings.llamacpp,
                            &entry,
                            &config.global,
                        )
                        .await
                        {
                            Ok(server) => {
                                info!(origin = %server.origin, model = %entry.name, "auto-started llama.cpp local server");
                                return create_provider(&name, &config.providers)
                                    .or_else(|_| create_default(&name, &config.providers))
                                    .with_context(|| {
                                        format!("failed to create provider '{name}'")
                                    });
                            }
                            Err(e) => {
                                info!(model = %entry.name, error = %e, "auto-start of llama.cpp failed; falling back")
                            }
                        }
                    }
                }
                match zeus_provider::detect_local_provider(&config.providers).await {
                    Some(detected) => {
                        if detected != name {
                            info!(from = %name, to = %detected, "default provider unreachable; auto-detected a running local server instead");
                        }
                        return create_provider(&detected, &config.providers).with_context(|| {
                            format!("failed to create auto-detected provider '{detected}'")
                        });
                    }
                    None => {
                        info!(provider = %name, "no reachable model server detected");
                        return Err(anyhow::anyhow!(
                            "no model provider is reachable. Start a local server (ollama/lmstudio/llamacpp) or configure a cloud provider and set its API key, then run `zeus config set core.default.provider <name>` (or pass `--provider <name>`)"
                        ));
                    }
                }
            }
        }
    }

    match create_provider(&name, &config.providers) {
        Ok(p) => Ok(p),
        Err(_) => create_default(&name, &config.providers)
            .with_context(|| format!("failed to create provider '{name}'")),
    }
}

/// Resolve `model` against the provider's real model list: a configured
/// default like "llama3.2" commonly doesn't exactly match what's actually
/// pulled (e.g. Ollama tags it "llama3.2:3b"), which otherwise surfaces as a
/// confusing 404 deep in the first chat request. If the exact name isn't
/// found but a tag-suffixed variant is (Ollama's "name:tag" convention),
/// use that instead of failing outright. Returns the resolved model name
/// and the fetched list (so callers that also need model metadata, e.g.
/// context window, don't have to call `list_models` a second time).
async fn resolve_model(
    provider: &dyn ModelProvider,
    model: String,
) -> (String, Option<Vec<zeus_provider::ModelInfo>>) {
    let models_list = provider.list_models().await.ok();
    let resolved = match &models_list {
        Some(models) if !models.iter().any(|m| m.id == model) => models
            .iter()
            .find(|m| m.id.starts_with(&format!("{model}:")))
            .map(|m| {
                info!(configured = %model, resolved = %m.id, "configured model not found exactly; using a matching locally available tag instead");
                m.id.clone()
            })
            .unwrap_or(model),
        _ => model,
    };
    (resolved, models_list)
}

/// Best-effort model list across every configured provider, grouped by
/// Describe every configured provider for `/provider` (and the TUI's
/// provider picker): name, kind, default model, and whether it's usable
/// right now — a cloud provider needs its `api_key_env` key present,
/// a local kind is assumed ready (the model picker live-probes reachability
/// separately). Backs the slash command so `/provider` lists *everything*
/// that's configured, not just the ones currently reachable.
pub(crate) fn describe_providers(config: &Config) -> Vec<String> {
    let mut names: Vec<&String> = config.providers.providers.keys().collect();
    names.sort();
    let stored = KeysFile::load(&config.global.keys_toml).unwrap_or_default();
    let mut out = Vec::new();
    for name in names {
        let Some(cfg) = config.providers.get(name) else {
            continue;
        };
        let model = cfg.default_model.as_deref().unwrap_or("");
        let local = matches!(cfg.kind.as_str(), "ollama" | "lmstudio" | "llamacpp");
        let status = if local {
            "local".to_string()
        } else if stored.get(name).is_some() || cfg.headers.contains_key("Authorization") {
            "key stored".to_string()
        } else if let Some(var) = &cfg.api_key_env {
            if std::env::var(var).map(|k| !k.is_empty()).unwrap_or(false) {
                "key set".to_string()
            } else {
                format!("no key (${var})")
            }
        } else {
            "ready".to_string()
        };
        out.push(format!(
            "  {name:<11} kind={:<10} model={:<18} {status}",
            cfg.kind, model
        ));
    }
    out
}

/// Persist the default provider (and optionally model) to the active
/// settings.toml layer — so a `/provider` switch survives restarts. Returns
/// the path written, for the confirm message.
pub(crate) fn persist_default_provider(
    config: &Config,
    provider: &str,
    model: Option<&str>,
) -> Result<PathBuf> {
    let path = settings_file_path(config, false);
    let mut doc = load_toml_or_empty(&path)?;
    set_toml_path(
        &mut doc,
        &["model", "provider"],
        toml::Value::String(provider.to_string()),
    );
    if let Some(m) = model {
        set_toml_path(
            &mut doc,
            &["model", "model"],
            toml::Value::String(m.to_string()),
        );
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).context("create settings dir")?;
    }
    let text = toml::to_string_pretty(&doc).context("serialize settings")?;
    std::fs::write(&path, text).context("write settings file")?;
    Ok(path)
}

/// provider_name — backs the `/model` picker's multi-provider view. A
/// provider that's unreachable, misconfigured, or slow to respond (bounded
/// by a short timeout) is silently skipped rather than blocking the whole
/// picker on one bad entry, same "best effort" spirit as MCP server connect.
///
/// All providers are probed concurrently (not one after another) — with a
/// handful of configured providers and a 3s-per-provider timeout, a
/// sequential scan could take 10+ seconds if several are unreachable, which
/// reads as a frozen picker. Run together, the whole scan takes as long as
/// the single slowest provider (worst case ~3s) instead of their sum.
pub(crate) async fn list_models_by_provider(
    config: &Config,
) -> Vec<(String, Vec<zeus_provider::ModelInfo>)> {
    let mut names: Vec<&String> = config.providers.providers.keys().collect();
    names.sort();

    let fetches = names.into_iter().map(|name| async move {
        let provider = zeus_provider::create_provider(name, &config.providers).ok()?;
        let fetch = tokio::time::timeout(std::time::Duration::from_secs(3), provider.list_models());
        match fetch.await {
            Ok(Ok(models)) if !models.is_empty() => Some((name.clone(), models)),
            _ => None,
        }
    });
    futures::future::join_all(fetches)
        .await
        .into_iter()
        .flatten()
        .collect()
}

async fn cmd_chat(
    config: &Config,
    message: String,
    provider: Option<String>,
    model: Option<String>,
    stream: bool,
) -> Result<()> {
    let provider = resolve_provider(config, provider).await?;
    let model = model.unwrap_or_else(|| {
        config
            .providers
            .get(provider.id())
            .and_then(|c| c.default_model.clone())
            .unwrap_or_else(|| config.settings.model.model.clone())
    });
    let (model, _) = resolve_model(&*provider, model).await;

    info!(provider = provider.id(), %model, "chat");

    let request = ChatRequest::new(
        model.clone(),
        vec![
            Message::system("You are zeus, a helpful coding assistant. Be concise and accurate."),
            Message::user(message),
        ],
    );

    if stream {
        let mut s = provider.stream(request).await.context("stream start")?;
        let stdout = io::stdout();
        let mut out = stdout.lock();
        while let Some(ev) = s.next().await {
            match ev.context("stream event")? {
                StreamEvent::TextDelta { text } => {
                    write!(out, "{text}")?;
                    out.flush()?;
                }
                StreamEvent::ToolCallDelta { name, .. } => {
                    if let Some(n) = name {
                        write!(out, "\n[tool_call: {n}]")?;
                        out.flush()?;
                    }
                }
                StreamEvent::Done {
                    finish_reason,
                    usage,
                } => {
                    writeln!(out)?;
                    eprintln!(
                        "— finish={finish_reason:?} tokens={}/{} (prompt/completion)",
                        usage.prompt_tokens, usage.completion_tokens
                    );
                }
            }
        }
    } else {
        let resp = provider.chat(request).await.context("chat")?;
        println!("{}", resp.message.content);
        eprintln!(
            "— finish={:?} tokens={}/{}",
            resp.finish_reason, resp.usage.prompt_tokens, resp.usage.completion_tokens
        );
    }
    Ok(())
}

async fn cmd_models(
    config: &Config,
    provider: Option<String>,
    local: bool,
    import: Option<PathBuf>,
    relocate: bool,
) -> Result<()> {
    if local || import.is_some() {
        return cmd_local_models(config, import.as_deref(), relocate);
    }

    let provider = resolve_provider(config, provider).await?;
    let models = provider.list_models().await.context("list_models")?;
    if models.is_empty() {
        bail!("no models returned by provider {}", provider.id());
    }
    println!("provider: {}", provider.id());
    for m in models {
        match m.context_window {
            Some(w) => println!("  {} — {} (ctx={w})", m.id, m.name),
            None => println!("  {} — {}", m.id, m.name),
        }
    }
    println!("prompt_cache: {}", provider.supports_prompt_cache());
    Ok(())
}

/// Scan the system for local model files (or import one into the library).
/// A model is "imported" by copying it into `~/.zeus/models`; `--move`
/// relocates it instead. `import` indexes into the same listing shown by
/// `zeus models --local`.
fn cmd_local_models(config: &Config, import: Option<&Path>, relocate: bool) -> Result<()> {
    let extra_dirs: Vec<PathBuf> = config
        .settings
        .extra_model_dirs
        .iter()
        .map(PathBuf::from)
        .collect();
    let found = zeus_provider::scan_system_models(&config.global.models, &extra_dirs);

    if let Some(want) = import {
        let Some(model) = found.iter().find(|m| m.path == want) else {
            bail!(
                "no scanned model at '{}' — pick the full path from `zeus models --local`",
                want.display()
            );
        };
        return cmd_import_model(config, model, relocate);
    }

    if found.is_empty() {
        println!("(no local model files found)");
        println!("scanned: {}", config.global.models.display());
        println!("hint: drop a .gguf/.safetensors under Downloads/Desktop/Documents or add `extra_model_dirs` to extend the scan.");
        return Ok(());
    }

    println!("{} model files on this system:", found.len());
    for f in &found {
        let size_mb = f.size_bytes as f64 / (1024.0 * 1024.0);
        println!("{:>10.1} MB  {}  [{}]", size_mb, f.path.display(), f.source);
    }
    println!();
    println!("import into the zeus library:  zeus models --import \"<path from above>\"   (add --move to relocate)");
    Ok(())
}

fn cmd_import_model(
    config: &Config,
    model: &zeus_provider::LocalModelFile,
    relocate: bool,
) -> Result<()> {
    let dest_dir = config.global.models.clone();
    let size_mb = model.size_bytes as f64 / (1024.0 * 1024.0);
    println!(
        "{} {} ({size_mb:.1} MB) -> {}",
        if relocate { "moving" } else { "copying" },
        model.path.display(),
        dest_dir.display()
    );
    let saved = zeus_provider::import_model_file(model, &dest_dir, relocate)
        .context("import model into library")?;
    let name = saved.file_name().map(|n| n.to_string_lossy().into_owned());
    println!("done — {}", saved.display());
    if let Some(name) = name {
        println!("use it with:  zeus serve {name}");
    }
    Ok(())
}

async fn cmd_pull(config: &Config, source: PullCmd) -> Result<()> {
    match source {
        PullCmd::Ollama { model } => {
            let cfg =
                config
                    .providers
                    .get("ollama")
                    .cloned()
                    .unwrap_or(zeus_config::ProviderConfig {
                        kind: "ollama".into(),
                        base_url: Some("http://127.0.0.1:11434".into()),
                        api_key_env: None,
                        default_model: None,
                        headers: Default::default(),
                        embeddings: true,
                        prompt_cache: false,
                    });
            let base_url = cfg
                .base_url
                .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());
            let provider = zeus_provider::OllamaProvider::new("ollama", base_url);
            provider
                .pull(&model, |status| println!("{status}"))
                .await
                .context("ollama pull")?;
            println!("done — pulled '{model}', now visible via `zeus models --provider ollama`");
            Ok(())
        }
        PullCmd::Hf { repo, file } => {
            let dest_dir = config.global.models.clone();
            println!("downloading {repo}/{file} -> {}", dest_dir.display());
            let mut last_pct_reported = 0u64;
            let path =
                zeus_provider::download_hf_file(&repo, &file, &dest_dir, |downloaded, total| {
                    if let Some(total) = total {
                        if total > 0 {
                            let pct = downloaded * 100 / total;
                            if pct >= last_pct_reported + 10 || pct == 100 {
                                last_pct_reported = pct;
                                println!("  {pct}% ({downloaded}/{total} bytes)");
                            }
                        }
                    }
                })
                .await
                .context("hugging face download")?;
            println!(
                "done — saved to {}, now visible via `zeus models --local`",
                path.display()
            );
            Ok(())
        }
    }
}

/// `zeus serve` — ensure a local GGUF model is on disk and serve it via
/// llama.cpp, auto-downloading the `llama-server` binary on first use. The
/// server runs detached (keeps going after this command exits).
async fn cmd_serve(config: &Config, model: Option<String>) -> Result<()> {
    let requested = model.unwrap_or_else(|| config.settings.model.model.clone());
    let entry = if requested.contains('/') {
        // Treat as a `repo/file` GGUF download.
        let split = requested.splitn(2, '/').collect::<Vec<_>>();
        if split.len() != 2 || split[0].is_empty() || split[1].is_empty() {
            bail!("usage: zeus serve <model-name>  or  zeus serve <repo>/<file.gguf>");
        }
        zeus_config::LocalModelEntry {
            name: requested.clone(),
            repo: split[0].to_string(),
            file: split[1].to_string(),
        }
    } else {
        match zeus_provider::resolve_local_model(&config.settings.llamacpp, &requested) {
            Some(e) => e,
            None => bail!(
                "no local model '{requested}' — add it under [settings.llamacpp.models], pass `repo/file`, or see `zeus serve llama3.2`"
            ),
        }
    };

    let lcpp = &config.settings.llamacpp;
    let server = zeus_provider::serve(lcpp, &entry, &config.global)
        .await
        .with_context(|| format!("failed to start llama.cpp for '{requested}'"))?;
    if server.pid == 0 {
        println!("reusing llama.cpp already running at {}", server.origin);
    } else {
        println!(
            "serving '{}' via llama.cpp at {} (pid {})",
            entry.name, server.origin, server.pid
        );
        println!(
            "model file: {}",
            config.global.models.join(&entry.file).display()
        );
    }
    println!("logs: {}/llamacpp.stderr.log", config.global.logs.display());
    println!(
        "connect with `zeus chat --provider llamacpp --model {}` or via `/provider` in the TUI",
        entry.file
    );
    Ok(())
}

/// Best-effort connect to every MCP server in `config.settings.mcp_servers`:
/// a server that fails to start (bad command, crashes during handshake)
/// logs a warning and is skipped rather than failing the whole agent turn —
/// one misconfigured server shouldn't take down every other tool.
fn connect_configured_mcp_servers(
    config: &Config,
    project_root: &std::path::Path,
) -> Vec<McpClient> {
    config
        .settings
        .mcp_servers
        .iter()
        .filter_map(
            |s| match McpClient::connect(&s.name, &s.command, &s.args, project_root) {
                Ok(client) => {
                    info!(server = %s.name, tools = client.tools().len(), "connected MCP server");
                    Some(client)
                }
                Err(e) => {
                    error!(server = %s.name, ?e, "failed to connect MCP server; skipping");
                    None
                }
            },
        )
        .collect()
}

/// Render the saved-session listing (shared by `zeus sessions` and the
/// REPL `/sessions` command).
fn render_sessions(config: &Config) -> Result<String> {
    let store = SessionStore::new(config.global.sessions.clone());
    let summaries = store.summaries().context("list sessions")?;
    if summaries.is_empty() {
        return Ok(
            "no saved sessions yet — run a turn (zeus agent \"...\") to create one.".to_string(),
        );
    }
    let mut lines = Vec::new();
    for s in &summaries {
        lines.push(format!(
            "{:<38} {} messages  {}",
            s.id,
            s.message_count,
            if s.last_user.is_empty() {
                "(no user message yet)".to_string()
            } else {
                format!("— {}\"", s.last_user)
            }
        ));
    }
    Ok(lines.join("\n"))
}

/// Resolve which session to continue: explicit `--session`, else the most
/// recently used one when `--resume`, else a brand-new session.
fn resolve_session(config: &Config, explicit: Option<String>, resume: bool) -> Option<String> {
    if let Some(id) = explicit {
        return Some(id);
    }
    if resume {
        let store = SessionStore::new(config.global.sessions.clone());
        if let Ok(Some(id)) = store.most_recent() {
            eprintln!("resuming most recent session: {id}");
            return Some(id);
        }
        eprintln!("no saved session found — starting a new one");
    }
    None
}

/// List saved sessions (id, message count, last user message).
fn cmd_sessions(config: &Config) -> Result<()> {
    let text = render_sessions(config)?;
    print!("{}", text);
    println!();
    Ok(())
}

/// `zeus key set/list` — the one-shot equivalent of the REPL's `/provider
/// key`, for setting a key without opening a full interactive session
/// first (the terminal-native version of "a popup that asks for a key":
/// one focused prompt, takes the key, exits — Zeus has no GUI layer to put
/// an actual dialog in).
fn cmd_key(config: &Config, action: KeyCmd) -> Result<()> {
    match action {
        KeyCmd::Set { name, key } => {
            let name = match name {
                Some(n) => n,
                None => prompt_choose_provider(config)?,
            };
            let key = match key {
                Some(k) => k,
                None => {
                    match read_hidden_line(&format!("Paste {name} API key (input hidden): "))? {
                        Some(k) if !k.trim().is_empty() => k.trim().to_string(),
                        _ => bail!("cancelled — no key set"),
                    }
                }
            };
            save_provider_key(config, &name, &key)?;
            println!(
                "key saved for '{name}' in {} — persistent across restarts.",
                config.global.keys_toml.display()
            );
            Ok(())
        }
        KeyCmd::List => {
            for line in describe_providers(config) {
                println!("{line}");
            }
            Ok(())
        }
    }
}

/// Lists configured providers with a ready/needs-key marker and prompts for
/// a number — the "which provider" step before `zeus key set` asks for its
/// key, for when you'd rather pick from a list than type the exact name.
fn prompt_choose_provider(config: &Config) -> Result<String> {
    let mut names: Vec<&String> = config.providers.providers.keys().collect();
    names.sort();
    if names.is_empty() {
        bail!("no providers configured — see providers.toml");
    }
    let stored = KeysFile::load(&config.global.keys_toml).unwrap_or_default();
    println!("Select a provider:");
    for (i, name) in names.iter().enumerate() {
        let cfg = config.providers.get(name);
        let local = cfg
            .map(|c| matches!(c.kind.as_str(), "ollama" | "lmstudio" | "llamacpp"))
            .unwrap_or(false);
        let ready = local
            || stored.get(name).is_some()
            || cfg
                .and_then(|c| c.api_key_env.as_ref())
                .map(|var| std::env::var(var).map(|k| !k.is_empty()).unwrap_or(false))
                .unwrap_or(false);
        let marker = if ready { "●" } else { "◌" };
        println!("  {:>2}) {marker} {name}", i + 1);
    }
    print!("Enter a number: ");
    io::stdout().flush().ok();
    let mut line = String::new();
    io::stdin().read_line(&mut line).context("read selection")?;
    let choice: usize = line.trim().parse().context("not a number")?;
    names
        .get(choice.wrapping_sub(1))
        .map(|n| n.to_string())
        .ok_or_else(|| anyhow::anyhow!("no provider numbered {choice}"))
}

/// Build a fully-wired `Agent` (provider, tools, context, session) — shared
/// by the one-shot `agent` subcommand and the interactive REPL so both go
/// through identical setup.
async fn build_agent(
    config: &Config,
    provider_name: Option<String>,
    model: Option<String>,
    session: Option<String>,
) -> Result<Agent> {
    let provider = resolve_provider(config, provider_name).await?;
    let model = model.unwrap_or_else(|| {
        config
            .providers
            .get(provider.id())
            .and_then(|c| c.default_model.clone())
            .unwrap_or_else(|| config.settings.model.model.clone())
    });
    build_agent_with_provider(config, provider, model, session).await
}

/// Variant used by the interactive REPL/TUI when no provider is ready at
/// startup: it wires an `UnconfiguredProvider` placeholder so the UI still
/// launches and the user can set providers/keys in-app via `/provider`.
async fn build_agent_unconfigured(config: &Config) -> Result<Agent> {
    let provider: std::sync::Arc<dyn ModelProvider> = std::sync::Arc::new(UnconfiguredProvider {
        requested: Some(config.settings.model.provider.clone()),
    });
    build_agent_with_provider(config, provider, config.settings.model.model.clone(), None).await
}

/// Construct an Agent around an already-resolved provider — shared by the
/// normal and fallback paths.
async fn build_agent_with_provider(
    config: &Config,
    provider: std::sync::Arc<dyn ModelProvider>,
    model: String,
    session: Option<String>,
) -> Result<Agent> {
    let (model, models_list) = resolve_model(&*provider, model).await;
    let window = models_list
        .and_then(|models| models.into_iter().find(|m| m.id == model))
        .and_then(|m| m.context_window)
        .unwrap_or(128_000);

    let ws = workspace(config)?;
    let project_root = ws.project_root.clone();
    let terminal = TerminalRunner::new(project_root.join(".agent/terminal"));
    let background = BackgroundTaskRegistry::new(project_root.join(".agent/background"));
    let hooks = HookRunner::new(project_root.join(".agent/hooks"), project_root.clone());
    let mcp_clients = connect_configured_mcp_servers(config, &project_root);
    let plugins = zeus_agent::load_all_plugins(&config.global.plugins);
    let mut tools = ToolManager::new(
        ws,
        terminal,
        background,
        hooks,
        mcp_clients,
        plugins,
        Arc::new(AtomicBool::new(false)),
    );
    tools.set_global_skills_dir(Some(config.global.skills.clone()));
    let context = ContextManager::new(
        window,
        config.settings.context.compact_threshold,
        config.settings.context.keep_recent_turns,
    );
    let sessions = SessionStore::new(config.global.sessions.clone());
    let session_id = session.unwrap_or_else(zeus_agent::new_session_id);
    let mut state = sessions.load(&session_id).context("load session")?;
    // A brand-new session gets one steering message so a small/local model
    // doesn't reflexively reach for tools (observed: calling git_status for
    // plain greetings, burning the tool-iteration budget on nothing). This
    // is deliberately silent about git specifically: the user works and
    // initializes git at their own convenience, not because zeus checked
    // for a repo up front.
    if state.messages.is_empty() {
        state.messages.push(system_prompt(config));
        if let Some(survey) = build_project_survey(config) {
            state.messages.push(Message::system(survey));
        }
    }

    Ok(Agent::new(
        provider,
        tools,
        context,
        sessions,
        state,
        AgentOptions {
            model,
            max_tool_iterations: 8,
            temperature: config.settings.model.temperature,
            // Bounds worst-case reply latency — otherwise an ungrounded
            // ramble (especially on slow CPU-bound local inference) keeps
            // generating for as long as the model's context window allows.
            // `model.max_tokens` in settings.toml overrides this.
            max_tokens: Some(config.settings.model.max_tokens.unwrap_or(1024)),
            max_parallel_read_steps: 2,
            tasks_file: config.project.as_ref().map(|p| p.tasks_json.clone()),
        },
    ))
}

/// The agent's standing instructions (system prompt): identity, tool
/// discipline, and the anti-hallucination grounding rules. New sessions get
/// this as their leading system message; resumed sessions keep their own.
fn system_prompt(_config: &Config) -> Message {
    Message::system(
        "You are zeus, a helpful coding assistant with access to file, git, terminal, \
             and search tools. Default to replying in plain text with no tool call at all. \
             Only call a tool when the user's message clearly requires one — e.g. they ask you \
             to read, write, or search a specific file, run a command, or inspect git history. \
             A greeting, thank-you, or general question (\"hi\", \"hello\", \"thanks\", \"how are \
             you\") gets a plain-text reply and nothing else — never a file write, a git command, \
             or any other tool call just to have done something. When genuinely unsure whether a \
             tool is needed, don't call one; ask the user what they want instead. Don't check git \
             status or assume this is a git repository; plenty of valid work happens outside one, \
             and the user will run `git init` (or `zeus git ...`) themselves whenever they want \
             version control.\n\
             \n\
             GROUNDING RULES (never violate these; they exist to keep you from hallucinating):\n\
             1. Never claim a file's contents, a file's existence, a git state, a tool's output, \
             or a command's result unless you actually ran the tool that produced that fact in this \
             conversation. If you haven't run it, either run it now or explicitly say you don't \
             know — do not guess.\n\
             2. Only assert something as fact if it directly quotes a tool result you received. \
             If you must summarize or extrapolate, clearly mark it as inference rather than fact.\n\
             3. Never invent diffs, test passes/failures, error output, line numbers, or file paths \
             you have not observed. If a tool errored, report the error you actually got.\n\
             4. If a task requires current information (file contents, test results, command \
             output, git history) that you cannot produce from an actual tool call available to \
             you, say you need to run the relevant tool rather than fabricating the answer.\n\
             5. When you are not certain, say so plainly. It is always better to admit a gap or \
             run a tool than to produce a confident but wrong answer.",
    )
}

/// A factual, bounded snapshot of the project the agent is operating in,
/// injected into *new* sessions as a system message. Purpose: ground the
/// model in real project facts at session start so it doesn't hallucinate
/// structure (guessing manifests, frameworks, layouts) — everything here is
/// actually walked from disk and explicitly labeled as such. Kept deliberately
/// small and capped: on a huge tree it enumerates only the top level plus a
/// depth-limited walk, and never reads file bodies.
fn build_project_survey(config: &Config) -> Option<String> {
    let root = config.project_root.as_ref()?;
    let name = root
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| root.display().to_string());

    let mut top_dirs: Vec<String> = Vec::new();
    let mut top_files: Vec<String> = Vec::new();
    let mut entries = 0usize;

    // Bounded recursive walk: enumerate top-level entries fully, and stop
    // after a modest depth/count so startup never crawls a giant tree.
    let skipped = |f: &str| {
        matches!(
            f,
            ".git"
                | "node_modules"
                | "target"
                | ".agent"
                | ".zeus"
                | ".venv"
                | "__pycache__"
                | "dist"
                | "build"
                | ".next"
                | ".cache"
        )
    };
    fn walk(
        dir: &Path,
        depth: usize,
        top_dirs: &mut Vec<String>,
        top_files: &mut Vec<String>,
        entries: &mut usize,
        skipped: &impl Fn(&str) -> bool,
    ) {
        if depth > 6 || *entries >= 500 {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in rd.flatten() {
            if *entries >= 500 {
                return;
            }
            let fname = entry.file_name().to_string_lossy().into_owned();
            if skipped(&fname) {
                continue;
            }
            let Ok(ft) = entry.file_type() else { continue };
            if depth == 0 {
                if ft.is_dir() {
                    top_dirs.push(fname);
                } else if ft.is_file() {
                    top_files.push(fname);
                }
            }
            *entries += 1;
            if ft.is_dir() {
                walk(
                    &entry.path(),
                    depth + 1,
                    top_dirs,
                    top_files,
                    entries,
                    skipped,
                );
            }
        }
    }
    walk(
        root,
        0,
        &mut top_dirs,
        &mut top_files,
        &mut entries,
        &skipped,
    );

    let mut lines = vec![
        "PROJECT SURVEY (facts observed from the filesystem, not guesses — verify anything you rely on)".to_string(),
        format!("workspace name: {name}"),
        format!("project root: {}", root.display()),
    ];
    if !top_files.is_empty() {
        top_files.sort();
        lines.push(format!("top-level files: {}", top_files.join(", ")));
    }
    if !top_dirs.is_empty() {
        top_dirs.sort();
        lines.push(format!("top-level directories: {}", top_dirs.join(", ")));
    }
    lines.push(format!("entries scanned (capped at 500): {entries}"));

    let survey = lines.join("\n");
    if survey.chars().count() > 8_000 {
        let truncated: String = survey.chars().take(8_000).collect();
        return Some(format!("{truncated}\n[survey truncated at 8000 chars]"));
    }
    Some(survey)
}

/// Print one `AgentEvent` to stdout/stderr — shared by the one-shot `agent`
/// subcommand and the REPL so both render turns identically.
/// Printed once before each turn starts, so `print_agent_event` itself can
/// stay a stateless per-event renderer (it has no notion of "first delta").
fn print_turn_header() {
    println!(
        "{}",
        ui::styled(ui::assistant_marker_style(), "● assistant")
    );
}

fn print_agent_event(ev: AgentEvent) {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    match ev {
        AgentEvent::TextDelta(t) => {
            let _ = write!(out, "{t}");
            let _ = out.flush();
        }
        AgentEvent::ToolCallStarted {
            name, arguments, ..
        } => {
            let _ = writeln!(
                out,
                "\n{}",
                ui::styled(ui::tool_style(), &format!("⚙ {name} {arguments}"))
            );
        }
        AgentEvent::ToolCallFinished {
            name,
            is_error,
            result,
            ..
        } => {
            let label = if is_error {
                ui::styled(ui::error_style(), &format!("✗ {name} failed"))
            } else {
                ui::styled(ui::tool_style(), &format!("✓ {name}"))
            };
            let _ = writeln!(out, "{label}\n{result}");
        }
        AgentEvent::Compacted(c) => {
            eprintln!(
                "{}",
                ui::styled(
                    ui::dim_style(),
                    &format!("(compacted {} earlier message(s))", c.removed_messages)
                )
            );
        }
        AgentEvent::Cancelled => {
            eprintln!("{}", ui::styled(ui::warn_style(), "(cancelled)"));
        }
        AgentEvent::Done => {}
        AgentEvent::TodosUpdated { todos } => {
            let done = todos.iter().filter(|t| t.status == "completed").count();
            eprintln!(
                "{}",
                ui::styled(
                    ui::dim_style(),
                    &format!("(todos: {done}/{} done)", todos.len())
                )
            );
        }
        AgentEvent::PlanGenerated { steps } => {
            let roster = steps
                .iter()
                .map(|s| match s.persona.as_deref() {
                    Some(p) => format!("{} [{}]", s.description, p),
                    None => s.description.clone(),
                })
                .collect::<Vec<_>>()
                .join(" → ");
            eprintln!(
                "{}",
                ui::styled(
                    ui::dim_style(),
                    &format!("plan · {} step(s): {roster}", steps.len())
                )
            );
        }
        AgentEvent::PlanStepStarted { step } => {
            eprintln!(
                "{}",
                ui::styled(
                    ui::warn_style(),
                    &format!("{}. {}", step.id, step.description)
                )
            );
        }
        AgentEvent::PlanReviewed { persona, report } => {
            let _ = writeln!(
                out,
                "{}",
                ui::styled(
                    ui::tool_style(),
                    &format!(
                        "◆ review ({persona}): {}",
                        report.chars().take(300).collect::<String>()
                    )
                )
            );
        }
        AgentEvent::PlanStepDone { step, summary } => {
            let _ = writeln!(
                out,
                "{}",
                ui::styled(
                    ui::tool_style(),
                    &format!("✓ step {} · {}: {}", step.id, step.description, summary)
                )
            );
        }
        AgentEvent::PlanStepDeclined { step } => {
            let _ = writeln!(
                out,
                "{}",
                ui::styled(
                    ui::dim_style(),
                    &format!("⊘ step {} declined · {}", step.id, step.description)
                )
            );
        }
        AgentEvent::OrchestrationDone { summary } => {
            let _ = writeln!(
                out,
                "\n{}",
                ui::styled(ui::assistant_marker_style(), "● plan complete")
            );
            let _ = writeln!(out, "{summary}");
        }
        AgentEvent::OrchestrationRevision { report } => {
            let _ = writeln!(
                out,
                "\n{}",
                ui::styled(ui::warn_style(), "⊘ lead reviewer did NOT accept")
            );
            let _ = writeln!(out, "{report}");
        }
        AgentEvent::WorkflowStarted {
            id,
            description,
            phases,
        } => {
            let roster = phases
                .iter()
                .map(|p| format!("{} [{}]", p.prompt, p.persona))
                .collect::<Vec<_>>()
                .join(" → ");
            let _ = writeln!(
                out,
                "{}",
                ui::styled(
                    ui::assistant_marker_style(),
                    &format!("● workflow '{id}' — {description}\n   {roster}")
                )
            );
        }
        AgentEvent::WorkflowPhaseStarted { name, persona } => {
            let _ = writeln!(
                out,
                "{}",
                ui::styled(ui::warn_style(), &format!("▶ {}", name))
            );
            eprintln!(
                "{}",
                ui::styled(ui::dim_style(), &format!("   as {persona}"))
            );
        }
        AgentEvent::WorkflowPhaseDone {
            name,
            persona,
            summary,
        } => {
            let _ = writeln!(
                out,
                "{}",
                ui::styled(
                    ui::tool_style(),
                    &format!(
                        "✓ phase · {name} [{persona}]: {}",
                        summary.chars().take(300).collect::<String>()
                    )
                )
            );
        }
        AgentEvent::WorkflowDone { summary } => {
            let _ = writeln!(
                out,
                "\n{}",
                ui::styled(ui::assistant_marker_style(), "● workflow complete")
            );
            let _ = writeln!(out, "{summary}");
        }
        AgentEvent::RepoAnalyzed { stack, relevance } => {
            eprintln!("{}", ui::styled(ui::dim_style(), "Analyzing repository..."));
            for line in stack.lines() {
                eprintln!("{}", ui::styled(ui::assistant_marker_style(), line));
            }
            if !relevance.is_empty() {
                let _ = writeln!(out, "{}", ui::styled(ui::tool_style(), &relevance));
            }
        }
        AgentEvent::RepoRelevanceUpdated { relevance } => {
            let _ = writeln!(out, "{}", ui::styled(ui::tool_style(), &relevance));
        }
        AgentEvent::OrientationSaved { docs } => {
            let mut written = Vec::new();
            if docs.architecture {
                written.push(".agent/architecture.md");
            }
            if docs.conventions {
                written.push(".agent/conventions.md");
            }
            if written.is_empty() {
                eprintln!(
                    "{}",
                    ui::styled(
                        ui::warn_style(),
                        "orientation docs could not be extracted (model didn't emit markers)"
                    )
                );
            } else {
                let _ = writeln!(
                    out,
                    "{}",
                    ui::styled(
                        ui::assistant_marker_style(),
                        &format!("wrote {}", written.join(", "))
                    )
                );
            }
        }
        AgentEvent::ReviewUncommitted { persona, report } => {
            let _ = writeln!(
                out,
                "{}",
                ui::styled(
                    ui::assistant_marker_style(),
                    &format!("review ({persona}) complete")
                )
            );
            let _ = writeln!(out, "{report}");
        }
        AgentEvent::FeaturesSuggested { report } => {
            let _ = writeln!(
                out,
                "{}",
                ui::styled(ui::assistant_marker_style(), "suggestions")
            );
            let _ = writeln!(out, "{report}");
        }
    }
}

/// Expand a message if it's a recognized slash command, printing which one
/// was used. Unchanged messages (including an unrecognized `/word`) pass
/// through as-is.
fn expand_slash_command(config: &Config, message: String) -> String {
    let commands = SlashCommands::new(
        config.project.as_ref().map(|p| p.commands.clone()),
        config.global.commands.clone(),
    );
    match commands.expand(&message) {
        ExpandResult::Unchanged => message,
        ExpandResult::Expanded { command, rendered } => {
            eprintln!("[slash command: /{command}]");
            rendered
        }
    }
}

#[allow(clippy::too_many_arguments)] // one flag per CLI option; matches clap's flat shape
async fn cmd_agent(
    config: &Config,
    message: String,
    provider_name: Option<String>,
    model: Option<String>,
    session: Option<String>,
    resume: bool,
    plan: bool,
    auto: bool,
    workflow: Option<String>,
    yes: bool,
) -> Result<()> {
    let message = expand_slash_command(config, message);
    let session = resolve_session(config, session, resume);
    let mut agent = build_agent(config, provider_name, model, session).await?;

    print_turn_header();
    let is_workflow = workflow.is_some();
    let result = if let Some(name) = workflow {
        // `--workflow <name>`: run a named multi-specialist pipeline.
        let workflows = discover_workflows(config.project_root.as_deref(), &config.global.root);
        let wf = workflows
            .iter()
            .find(|w| w.id == name)
            .with_context(|| format!("no workflow named '{name}' (run zeus /workflows to list)"))?;
        agent
            .run_workflow(&message, wf, print_agent_event, approver(yes))
            .await
            .map(|summary| TurnResult {
                final_text: summary,
                tool_calls: 0,
                cancelled: false,
                usage: Default::default(),
            })
    } else if auto {
        // `--auto`: full orchestrated run — plan, execute, lead-reviewer gate.
        agent
            .orchestrate(&message, print_agent_event, approver(yes))
            .await
            .map(|(summary, usage)| TurnResult {
                final_text: summary,
                tool_calls: 0,
                cancelled: false,
                usage,
            })
    } else if plan {
        // `--plan`: research read-only, persist .agent/tasks.json, don't
        // execute. Plan mode is forced on so no tool call can mutate.
        agent.set_plan_mode(true);
        agent
            .plan_turn(&message, print_agent_event, approver(yes))
            .await
    } else {
        agent
            .run_turn(&message, print_agent_event, approver(yes))
            .await
    }
    .context("agent turn")?;

    writeln!(io::stdout())?;
    if plan {
        if let Some(project) = config.project.as_ref() {
            eprintln!("plan saved to {}", project.tasks_json.display());
        }
        eprintln!(
            "— plan only: nothing was executed. Switch to auto mode and approve to run the plan."
        );
    }
    if auto || is_workflow {
        eprintln!("— orchestrated run finished.");
    }
    eprintln!(
        "— session={} tool_calls={} cancelled={}",
        agent.session_id(),
        result.tool_calls,
        result.cancelled
    );
    Ok(())
}

/// Interactive chat mode: `zeus` with no subcommand. Builds one `Agent` and
/// reuses it across every line typed, so conversation context carries over
/// between messages within the session — each one-shot `zeus agent` call
/// instead starts fresh unless `--session` is passed explicitly.
const REPL_BUILTIN_COMMANDS: &[(&str, &str)] = &[
    ("help", "show this list"),
    (
        "clear",
        "start a fresh session (new session id, empty context)",
    ),
    ("new", "start a fresh session (alias for /clear)"),
    (
        "compact",
        "force context compaction now, even under the auto threshold",
    ),
    ("autocompact", "toggle auto-compaction: /autocompact on|off"),
    (
        "context",
        "show token usage against the model's context window",
    ),
    ("diff", "show uncommitted changes: /diff or /diff staged"),
    (
        "undo",
        "revert file changes made this session (confirm with /undo confirm)",
    ),
    (
        "settings",
        "view/change display settings: reduced_motion, notify, accent <#hex>|reset",
    ),
    (
        "model",
        "switch model (opens a picker), or /model <name> directly",
    ),
    (
        "provider",
        "list providers, switch (<name>), or set a key: /provider key <name> (prompts, hidden)",
    ),
    (
        "mode",
        "set agent mode: /mode build|plan|auto (Tab also cycles)",
    ),
    (
        "plan",
        "plan-only turn: research a goal read-only, persist the plan, don't execute",
    ),
    (
        "understand",
        "read-only repository scan: what already exists for a topic (/understand auth)",
    ),
    (
        "orient",
        "write .agent/architecture.md + .agent/conventions.md (read-only scan)",
    ),
    (
        "review",
        "read-only review of uncommitted changes (copy passes, verdict at the end)",
    ),
    (
        "suggest",
        "read-only next-feature recommendations grounded in what already exists",
    ),
    (
        "workflow",
        "run a named multi-specialist pipeline: /workflow <name> <goal>",
    ),
    (
        "workflows",
        "list available workflows from .agent/workflows and ~/.zeus/workflows",
    ),
    (
        "bg",
        "run the workforce in the background (/bg <goal>); manage with `zeus bg ...`",
    ),
    ("session", "show the current session id"),
    (
        "sessions",
        "list saved sessions (opens a resume picker in the TUI)",
    ),
    (
        "agents",
        "list the specialist-agents roster grouped by department (/agents count)",
    ),
    (
        "copy",
        "copy the last assistant reply to the system clipboard",
    ),
];

fn print_repl_help_lines() -> String {
    let mut lines = vec!["Built-in commands:".to_string()];
    for (name, desc) in REPL_BUILTIN_COMMANDS {
        lines.push(format!("  /{name:<17}{desc}"));
    }
    lines.push("  exit, quit, ^D     leave the REPL".to_string());
    lines.push(
        "Anything else starting with '/' is looked up as a user-defined command in .agent/commands/ or ~/.zeus/commands/."
            .to_string(),
    );
    lines.join("\n")
}

fn print_repl_help() {
    println!("{}", print_repl_help_lines());
}

/// Persists `key` for `name` to `~/.zeus/keys.toml` (owner-only permissions,
/// see `KeysFile::save`) and applies it to the running process's env
/// immediately. Shared by both the inline (`/provider key <name> <KEY>`,
/// convenient for scripting but echoes the key to the terminal) and
/// hidden-prompt (`/provider key <name>`, masked input) forms.
fn save_provider_key(config: &Config, name: &str, key: &str) -> Result<()> {
    let cfg = config.providers.get(name).ok_or_else(|| {
        anyhow::anyhow!("unknown provider '{name}' — see `zeus key list` or /provider for the list")
    })?;
    let mut keys = KeysFile::load(&config.global.keys_toml).context("read key store")?;
    keys.keys.insert(name.to_string(), key.to_string());
    keys.save(&config.global.keys_toml)
        .context("save key store")?;
    if let Some(var) = &cfg.api_key_env {
        std::env::set_var(var, key);
    }
    Ok(())
}

/// Reads a line with terminal echo suppressed, printing `•` per keystroke
/// instead of the character typed — the plain-REPL equivalent of the TUI's
/// masked key-entry modal (`tui::mask_secret`). Returns `None` if the user
/// cancels with Esc or Ctrl-C rather than pressing Enter.
fn read_hidden_line(prompt: &str) -> io::Result<Option<String>> {
    use crossterm::event::{read, Event, KeyCode, KeyEventKind, KeyModifiers};
    use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

    print!("{prompt}");
    io::stdout().flush()?;
    enable_raw_mode()?;
    let mut buf = String::new();
    let outcome = loop {
        match read()? {
            Event::Key(k) if k.kind != KeyEventKind::Release => match k.code {
                KeyCode::Enter => break Some(buf),
                KeyCode::Esc => break None,
                KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => break None,
                KeyCode::Backspace => {
                    if buf.pop().is_some() {
                        print!("\u{8} \u{8}");
                        io::stdout().flush()?;
                    }
                }
                KeyCode::Char(c) => {
                    buf.push(c);
                    print!("•");
                    io::stdout().flush()?;
                }
                _ => {}
            },
            _ => {}
        }
    };
    disable_raw_mode()?;
    println!();
    Ok(outcome)
}

/// The `/settings` slash command (plain REPL): view or change persisted
/// display settings (`~/.zeus/settings.toml`, Global layer) without
/// hand-editing the file. The plain REPL doesn't use the wordmark/bell
/// these gate, so a change here just needs saving — the TUI's own
/// `"settings"` match arm additionally applies it live via `theme::`.
fn handle_settings_slash(arg: &str, config: &Config) {
    let parts: Vec<&str> = arg.split_whitespace().collect();
    let path = &config.global.settings_toml;
    match parts.as_slice() {
        [] => {
            println!("reduced_motion:       {}", config.settings.reduced_motion);
            println!(
                "notify_on_completion: {}",
                config.settings.notify_on_completion
            );
            println!(
                "accent_color:         {}",
                config
                    .settings
                    .accent_color
                    .as_deref()
                    .unwrap_or("(default violet)")
            );
            println!("Use /settings reduced_motion on|off, /settings notify on|off, /settings accent <#hex>|reset.");
        }
        ["reduced_motion", v @ ("on" | "off")] => {
            let on = *v == "on";
            match zeus_config::set_reduced_motion(path, on) {
                Ok(()) => println!("reduced_motion: {on} (takes effect next launch)"),
                Err(e) => eprintln!("couldn't save setting: {e}"),
            }
        }
        ["notify", v @ ("on" | "off")] => {
            let on = *v == "on";
            match zeus_config::set_notify_on_completion(path, on) {
                Ok(()) => println!("notify_on_completion: {on} (takes effect next launch)"),
                Err(e) => eprintln!("couldn't save setting: {e}"),
            }
        }
        ["accent", "reset"] => match zeus_config::set_accent_color(path, None) {
            Ok(()) => println!("accent_color reset to default (takes effect next launch)"),
            Err(e) => eprintln!("couldn't save setting: {e}"),
        },
        ["accent", hex] => match zeus_config::set_accent_color(path, Some((*hex).to_string())) {
            Ok(()) => println!("accent_color: {hex} (takes effect next launch)"),
            Err(e) => eprintln!("couldn't save setting: {e}"),
        },
        _ => eprintln!(
            "usage: /settings [reduced_motion on|off] [notify on|off] [accent <#hex>|reset]"
        ),
    }
}

/// The `/provider` slash command (plain REPL): list all configured providers,
/// switch the active one (persisting the choice), or set a cloud API key —
/// either inline (scriptable, echoes to the terminal) or via a hidden-input
/// prompt when no key is given on the line.
async fn handle_provider_slash(arg: &str, config: &Config, agent: &mut Agent) {
    let parts: Vec<&str> = arg.split_whitespace().collect();
    match parts.as_slice() {
        // `/provider` — show current + full configured roster.
        [] => {
            println!(
                "current provider: {} / model: {}",
                agent.provider_id(),
                agent.model()
            );
            println!("Configured providers:");
            for line in describe_providers(config) {
                println!("{line}");
            }
            println!("Use /provider <name> to switch, /provider key <name> <KEY> to set a key.");
        }
        // `/provider key <name> <KEY>` — persist the key to ~/.zeus/keys.toml
        // and apply it to the in-memory provider config right away. Kept for
        // scripting; it echoes the key to the terminal, so `/provider key
        // <name>` (below) is the better default for interactive use.
        ["key", name, key] => match save_provider_key(config, name, key) {
            Ok(()) => println!(
                "key saved for '{name}' in {} — persistent across restarts.",
                config.global.keys_toml.display()
            ),
            Err(e) => eprintln!("{e:#}"),
        },
        // `/provider key <name>` — no key on the line, so prompt for one
        // with echo suppressed instead of requiring it in cleartext.
        ["key", name] => {
            match read_hidden_line(&format!("Paste {name} API key (input hidden): ")) {
                Ok(Some(key)) if !key.trim().is_empty() => {
                    match save_provider_key(config, name, key.trim()) {
                        Ok(()) => println!(
                            "key saved for '{name}' in {} — persistent across restarts.",
                            config.global.keys_toml.display()
                        ),
                        Err(e) => eprintln!("{e:#}"),
                    }
                }
                Ok(_) => eprintln!("cancelled — no key set"),
                Err(e) => eprintln!("couldn't read key: {e}"),
            }
        }
        // `/provider key` with no args — usage hint, not a provider switch.
        ["key"] => {
            eprintln!("usage: /provider key <name> (prompts, hidden) or /provider key <name> <KEY>")
        }
        // `/provider <name>` — switch the active provider + persist default.
        [name] => match create_provider(name, &config.providers) {
            Ok(handle) => {
                agent.set_provider(handle);
                let model = config
                    .providers
                    .get(name)
                    .and_then(|c| c.default_model.clone())
                    .unwrap_or_else(|| agent.model().to_string());
                agent.set_model(model.clone());
                match persist_default_provider(config, name, Some(&model)) {
                    Ok(path) => println!(
                        "switched to provider: {name} (model: {model}) — saved to {}",
                        path.display()
                    ),
                    Err(e) => {
                        eprintln!("switched to provider {name}, but saving default failed: {e:#}")
                    }
                }
            }
            Err(e) => eprintln!("couldn't switch to '{name}': {e:#}"),
        },
        _ => eprintln!("usage: /provider | /provider <name> | /provider key <name> <KEY>"),
    }
}

/// Every slash name the REPL currently recognizes, with a one-line
/// description — built-in meta-commands plus user-defined
/// `.agent/commands/*.md` templates (project then global) — used to drive
/// the live dropdown so typing `/` actually surfaces what's available
/// instead of requiring the user to already know it.
fn known_slash_commands(config: &Config) -> Vec<(String, String)> {
    let mut names: Vec<(String, String)> = REPL_BUILTIN_COMMANDS
        .iter()
        .map(|(n, d)| (n.to_string(), d.to_string()))
        .collect();
    if let Some(project) = &config.project {
        if let Ok(found) = list_md_names(&project.commands) {
            names.extend(found.into_iter().map(|n| (n, "user command".to_string())));
        }
    }
    if let Ok(found) = list_md_names(&config.global.commands) {
        names.extend(found.into_iter().map(|n| (n, "user command".to_string())));
    }
    names
}

/// Interactive entry point: a full opencode/Claude-Code-style TUI when
/// stdin+stdout are a real terminal, or the plain line-by-line REPL below
/// for anything else (piped input, scripted invocations, tests) — the TUI's
/// raw-mode/alternate-screen handling only makes sense against a real
/// console.
/// Build an agent, falling back to the `UnconfiguredProvider` setup-mode
/// variant when no provider is ready — used by interactive sessions so they
/// can always launch/clear without a configured key.
pub(crate) async fn build_agent_repl(config: &Config) -> Result<Agent> {
    build_agent_repl_with(config, None, None).await
}

/// Same as `build_agent_repl`, but for `/new` and `/clear` — those rebuild
/// the agent from the *already-running* session, not a fresh process, so
/// they must target whatever provider/model that session is actually using
/// rather than `config.settings.model.*`. Persisting a provider switch
/// (`persist_default_provider`) only writes to disk; it doesn't mutate the
/// in-memory `Config` this process loaded at startup (which is passed
/// around as `&Config`, not `&mut Config`, everywhere). Without this,
/// setting up a provider mid-session and then running `/new` would rebuild
/// against the stale startup default — which is likely still key-less —
/// and silently drop back to the unconfigured placeholder even though the
/// provider you just configured is sitting right there in `AppState`.
pub(crate) async fn build_agent_repl_with(
    config: &Config,
    provider: Option<String>,
    model: Option<String>,
) -> Result<Agent> {
    match build_agent(config, provider, model, None).await {
        Ok(a) => Ok(a),
        Err(err) => {
            warn!(error = %err, "no provider ready; launching interactive session in setup mode");
            build_agent_unconfigured(config).await
        }
    }
}

/// Same as `build_agent_repl_with`, but loads an existing session's
/// conversation history (`ConversationState`) instead of starting fresh —
/// backs the TUI's `/sessions` picker. Note: this restores the *context*
/// the model continues from, not a replay of old messages into the visible
/// transcript — the same behavior `zeus agent --session <id> --resume`
/// already has at the plain-REPL/one-shot level, just reachable from a
/// picker instead of needing the id typed out.
pub(crate) async fn build_agent_repl_with_session(
    config: &Config,
    provider: Option<String>,
    model: Option<String>,
    session_id: String,
) -> Result<Agent> {
    match build_agent(config, provider, model, Some(session_id)).await {
        Ok(a) => Ok(a),
        Err(err) => {
            warn!(error = %err, "no provider ready; launching interactive session in setup mode");
            build_agent_unconfigured(config).await
        }
    }
}

async fn cmd_repl(config: &Config, yes: bool) -> Result<()> {
    let agent = build_agent_repl(config).await?;
    if config.project_root.is_some() {
        agent.set_plan_mode(true);
    }
    if io::stdin().is_terminal() && io::stdout().is_terminal() {
        return tui::run(config, agent, yes).await;
    }
    run_plain_repl(config, agent, yes).await
}

async fn run_plain_repl(config: &Config, mut agent: Agent, yes: bool) -> Result<()> {
    println!(
        "{}",
        ui::styled(
            ui::prompt_style(),
            &format!("zeus interactive session — session={}", agent.session_id())
        )
    );
    println!("Type a message and press Enter. `exit`, `quit`, or Ctrl-D to leave; Ctrl-C cancels an in-flight turn.");
    println!(
        "Resume this session later with: zeus agent --session {} \"...\"",
        agent.session_id()
    );

    let mut last_reply = String::new();

    loop {
        print!("\n{}", ui::styled(ui::prompt_style(), "zeus❯ "));
        io::stdout().flush().ok();
        let mut buf = String::new();
        let bytes_read = io::stdin().read_line(&mut buf).context("read stdin")?;
        if bytes_read == 0 {
            // EOF (Ctrl-D / piped input exhausted).
            println!("\n(end of input, exiting)");
            break;
        }
        let line = buf.trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "exit" | "quit" | ":q") {
            break;
        }

        // Built-in REPL meta-commands — intercepted before the user-defined
        // `.agent/commands/*.md` template expansion below, and never sent
        // to the model themselves (unlike those templates, which expand
        // into a message). A fixed reserved set, same idea as Claude Code's
        // own /help, /clear, /compact, /model.
        if let Some(rest) = line.strip_prefix('/') {
            let mut parts = rest.splitn(2, char::is_whitespace);
            let cmd = parts.next().unwrap_or("");
            let arg = parts.next().unwrap_or("").trim();
            let mut handled = true;
            match cmd {
                "help" => print_repl_help(),
                "clear" => {
                    let (provider, model) =
                        (agent.provider_id().to_string(), agent.model().to_string());
                    agent = build_agent_repl_with(config, Some(provider), Some(model)).await?;
                    println!("cleared — new session={}", agent.session_id());
                }
                "new" => {
                    let (provider, model) =
                        (agent.provider_id().to_string(), agent.model().to_string());
                    agent = build_agent_repl_with(config, Some(provider), Some(model)).await?;
                    println!("new session started — session={}", agent.session_id());
                }
                "compact" => match agent.compact_now().await {
                    Ok(result) if result.compacted => println!(
                        "compacted — removed {} earlier message(s)",
                        result.removed_messages
                    ),
                    Ok(_) => println!("nothing to compact yet (not enough history)"),
                    Err(e) => eprintln!("compact failed: {e:#}"),
                },
                "autocompact" => match arg {
                    "on" => {
                        agent.set_auto_compact(true);
                        println!("auto-compaction: on");
                    }
                    "off" => {
                        agent.set_auto_compact(false);
                        println!("auto-compaction: off");
                    }
                    _ => println!(
                        "auto-compaction: {} — use /autocompact on|off",
                        if agent.auto_compact() { "on" } else { "off" }
                    ),
                },
                "context" => match agent.context_usage().await {
                    Ok(u) => {
                        let approx = if u.approximate { "~" } else { "" };
                        println!(
                            "{approx}{} / {} tokens ({} messages)",
                            u.tokens, u.window, u.message_count
                        );
                    }
                    Err(e) => eprintln!("context lookup failed: {e:#}"),
                },
                "diff" => {
                    let git = git_engine_for_agent(config, &agent);
                    let staged = arg.eq_ignore_ascii_case("staged");
                    match git.diff(staged, &[]) {
                        Ok(out) if out.stdout.trim().is_empty() => println!("(no changes)"),
                        Ok(out) => print!("{}", highlight::ansi_diff(&out.stdout)),
                        Err(e) => eprintln!("diff failed: {e}"),
                    }
                }
                "undo" => {
                    let ws = agent.workspace();
                    let turn_id = ws.files.turn_id.clone();
                    let snaps = ws
                        .files
                        .checkpoints
                        .load_snapshots(&turn_id)
                        .unwrap_or_default();
                    if snaps.is_empty() {
                        println!("(nothing to undo this session)");
                    } else if arg.eq_ignore_ascii_case("confirm") {
                        match ws.files.checkpoints.restore(&turn_id, &ws.project_root) {
                            Ok(n) => println!("reverted {n} file change(s) made this session"),
                            Err(e) => eprintln!("undo failed: {e}"),
                        }
                    } else {
                        println!(
                            "this will revert {} file change(s) made since this session started — run `/undo confirm` to proceed",
                            snaps.len()
                        );
                    }
                }
                "copy" => {
                    if last_reply.is_empty() {
                        eprintln!("nothing to copy yet — send a message first");
                    } else {
                        match clipboard::copy(&last_reply) {
                            Ok(()) => println!(
                                "copied {} char(s) to clipboard",
                                last_reply.chars().count()
                            ),
                            Err(e) => eprintln!("copy failed: {e}"),
                        }
                    }
                }
                "model" => {
                    if arg.is_empty() {
                        println!("current model: {}", agent.model());
                    } else {
                        agent.set_model(arg.to_string());
                        println!("switched to model: {arg}");
                    }
                }
                "provider" => handle_provider_slash(arg, config, &mut agent).await,
                "settings" => handle_settings_slash(arg, config),
                "session" => println!("session={}", agent.session_id()),
                "sessions" => match render_sessions(config) {
                    Ok(text) => println!("{text}"),
                    Err(e) => eprintln!("couldn't list sessions: {e:#}"),
                },
                "agents" => {
                    if arg.eq_ignore_ascii_case("count") {
                        let pools = personas_by_department();
                        let total: usize = pools.iter().map(|(_, list)| list.len()).sum();
                        println!("{total} specialist agents");
                    } else {
                        println!("Specialist agent pool (grouped by department):");
                        for (dept, people) in personas_by_department() {
                            print!("  {dept}: ");
                            for (i, p) in people.iter().enumerate() {
                                if i > 0 {
                                    print!(", ");
                                }
                                print!("{} ({})", p.id, p.role);
                            }
                            println!();
                        }
                    }
                }
                "mode" => match arg.to_ascii_lowercase().as_str() {
                    "" => {
                        let m = if agent.auto_mode() {
                            "auto"
                        } else if agent.plan_mode() {
                            "plan"
                        } else {
                            "build"
                        };
                        println!("mode: {m}");
                    }
                    other => {
                        agent.set_plan_mode(other == "plan");
                        agent.set_auto_mode(other == "auto");
                        println!("mode: {other}");
                    }
                },
                "plan" => {
                    if arg.is_empty() {
                        eprintln!("usage: /plan <goal to plan>");
                    } else {
                        agent.set_plan_mode(true);
                        print_turn_header();
                        match agent.plan_turn(arg, print_agent_event, approver(yes)).await {
                            Ok(_) => {
                                writeln!(io::stdout())?;
                                if let Some(project) = config.project.as_ref() {
                                    eprintln!("plan saved to {}", project.tasks_json.display());
                                }
                                eprintln!(
                                    "— plan-only turn; switch to /mode auto and approve to execute."
                                );
                            }
                            Err(e) => eprintln!(
                                "\n{}",
                                ui::styled(ui::error_style(), &format!("plan failed: {e:#}"))
                            ),
                        }
                    }
                }
                "understand" => {
                    if arg.is_empty() {
                        eprintln!("usage: /understand <topic> — e.g. /understand authentication");
                    } else {
                        print_turn_header();
                        if let Err(e) = agent
                            .understand_topic(arg, print_agent_event, approver(yes))
                            .await
                        {
                            eprintln!(
                                "\n{}",
                                ui::styled(ui::error_style(), &format!("understand failed: {e:#}"))
                            );
                        }
                    }
                }
                "orient" => {
                    print_turn_header();
                    if let Err(e) = agent.orient_turn(print_agent_event, approver(yes)).await {
                        eprintln!(
                            "\n{}",
                            ui::styled(ui::error_style(), &format!("orient failed: {e:#}"))
                        );
                    }
                }
                "review" => {
                    print_turn_header();
                    if let Err(e) = agent.review_turn(print_agent_event, approver(yes)).await {
                        eprintln!(
                            "\n{}",
                            ui::styled(ui::error_style(), &format!("review failed: {e:#}"))
                        );
                    }
                }
                "suggest" => {
                    print_turn_header();
                    if let Err(e) = agent.suggest_turn(print_agent_event, approver(yes)).await {
                        eprintln!(
                            "\n{}",
                            ui::styled(ui::error_style(), &format!("suggest failed: {e:#}"))
                        );
                    }
                }
                "workflow" | "wf" => {
                    // /workflow <name> <goal...> — run a declarative
                    // multi-specialist pipeline from .agent/workflows or
                    // ~/.zeus/workflows.
                    let mut parts = arg.splitn(2, char::is_whitespace);
                    let name = parts.next().unwrap_or("").trim();
                    let goal = parts.next().unwrap_or("").trim();
                    if name.is_empty() || goal.is_empty() {
                        eprintln!("usage: /workflow <name> <goal> — e.g. /workflow build-backend 'add a health endpoint'");
                        eprintln!("  (/workflows lists available workflows)");
                    } else {
                        let workflows =
                            discover_workflows(config.project_root.as_deref(), &config.global.root);
                        match workflows.iter().find(|w| w.id == name) {
                            Some(wf) => {
                                print_turn_header();
                                if let Err(e) = agent
                                    .run_workflow(goal, wf, print_agent_event, approver(yes))
                                    .await
                                {
                                    eprintln!(
                                        "\n{}",
                                        ui::styled(
                                            ui::error_style(),
                                            &format!("workflow failed: {e:#}")
                                        )
                                    );
                                }
                            }
                            None => {
                                eprintln!(
                                    "{}",
                                    ui::styled(
                                        ui::warn_style(),
                                        &format!("no workflow named '{name}'")
                                    )
                                );
                                eprintln!("run /workflows to list available workflows");
                            }
                        }
                    }
                }
                "workflows" => {
                    let workflows =
                        discover_workflows(config.project_root.as_deref(), &config.global.root);
                    if workflows.is_empty() {
                        eprintln!(
                            "no workflows found. Create `<project>/.agent/workflows/<name>.toml` or `~/.zeus/workflows/<name>.toml`."
                        );
                    } else {
                        eprintln!("workflows:");
                        for wf in workflows {
                            eprintln!(
                                "  {} — {} ({} phase(s))",
                                ui::styled(ui::assistant_marker_style(), &wf.id),
                                wf.description,
                                wf.phases.len()
                            );
                        }
                    }
                }
                "bg" => {
                    // /bg <goal> [--workflow <name>] — run the workforce in the
                    // background: detach a headless `agent --auto` run so you
                    // keep your prompt while it plans, executes, and reviews.
                    let mut parts = arg.splitn(2, char::is_whitespace);
                    let rest = parts.next().unwrap_or("").trim();
                    if rest.is_empty() {
                        eprintln!("usage: /bg <goal> — run an orchestrated plan in the background");
                        eprintln!("  /bg list | output <id> | stop <id>  manage background tasks");
                        eprintln!("  append `@@workflow:<name>` to run a named workflow");
                    } else if matches!(rest, "list" | "output" | "stop") {
                        eprintln!("  for background tasks use the `zeus bg ...` subcommand:");
                        eprintln!("  zeus bg list · zeus bg output <id> · zeus bg stop <id>");
                    } else {
                        let (goal, workflow) = match rest.rsplit_once("@@workflow:") {
                            Some((g, name)) => (g.trim(), Some(name.trim())),
                            None => (rest, None),
                        };
                        match spawn_bg_orchestrate(config, goal, workflow) {
                            Ok(id) => {
                                println!(
                                    "{}",
                                    ui::styled(
                                        ui::assistant_marker_style(),
                                        &format!("● background orchestration started id={id}")
                                    )
                                );
                                println!("goal: {goal}");
                                println!(
                                    "follow: zeus bg output {id}   |   stop: zeus bg stop {id}"
                                );
                            }
                            Err(e) => eprintln!(
                                "\n{}",
                                ui::styled(ui::error_style(), &format!("bg spawn failed: {e:#}"))
                            ),
                        }
                    }
                }
                _ => handled = false,
            }
            if handled {
                continue;
            }
        }

        let message = expand_slash_command(config, line.to_string());

        // Ctrl-C during a turn cancels *that turn* via the agent's existing
        // cancellation mechanism (run_turn notices it internally and winds
        // down cleanly — persisting state, emitting AgentEvent::Cancelled —
        // rather than this racing/dropping the turn future ourselves, which
        // would skip that cleanup). Only armed while a turn is in flight:
        // Ctrl-C at an idle prompt falls through to the OS's normal
        // SIGINT handling (process exit), which is the expected REPL-at-rest
        // behavior.
        let cancel_tx = agent.cancel_handle();
        let watcher = tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                let _ = cancel_tx.send(true);
            }
        });

        print_turn_header();
        let result = match agent
            .run_turn(&message, print_agent_event, approver(yes))
            .await
        {
            Ok(result) => result,
            Err(e) => {
                watcher.abort();
                eprintln!(
                    "\n{}",
                    ui::styled(ui::error_style(), &format!("turn failed: {e:#}"))
                );
                continue;
            }
        };
        watcher.abort();
        last_reply = result.final_text.clone();

        writeln!(io::stdout())?;
        if result.cancelled {
            eprintln!(
                "{}",
                ui::styled(
                    ui::warn_style(),
                    &format!("— cancelled (tool_calls={})", result.tool_calls)
                )
            );
        }
    }

    println!(
        "Resume this session later with: zeus agent --session {} \"...\"",
        agent.session_id()
    );
    Ok(())
}

async fn cmd_tokens(
    config: &Config,
    message: String,
    provider: Option<String>,
    model: Option<String>,
) -> Result<()> {
    let provider = resolve_provider(config, provider).await?;
    let model = model.unwrap_or_else(|| config.settings.model.model.clone());
    let resp = provider
        .count_tokens(zeus_provider::TokenCountRequest {
            model,
            messages: vec![Message::user(message)],
            tools: vec![],
        })
        .await
        .context("count_tokens")?;
    println!("tokens={} approximate={}", resp.tokens, resp.approximate);
    Ok(())
}

fn cmd_read(
    config: &Config,
    path: PathBuf,
    offset: Option<usize>,
    limit: Option<usize>,
) -> Result<()> {
    let ws = workspace(config)?;
    let result = ws
        .files
        .read(&path, ReadOptions { offset, limit })
        .context("read")?;
    print!("{}", result.content);
    eprintln!(
        "— {} lines (showing from line {})",
        result.total_lines, result.start_line
    );
    Ok(())
}

fn cmd_write(config: &Config, path: PathBuf, content: String, yes: bool) -> Result<()> {
    let ws = workspace(config)?;
    let body = if content == "-" {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        buf
    } else {
        content
    };
    ws.files
        .write(&path, &body, WriteOptions::default(), approver(yes))
        .context("write")?;
    println!("wrote {}", path.display());
    Ok(())
}

fn cmd_edit(
    config: &Config,
    path: PathBuf,
    old: String,
    new: String,
    replace_all: bool,
    yes: bool,
) -> Result<()> {
    let ws = workspace(config)?;
    // Ensure model/user has "read" semantics for safety rule.
    let _ = ws
        .files
        .read(&path, ReadOptions::default())
        .context("read before edit")?;
    let n = ws
        .files
        .edit(
            &path,
            EditOptions {
                old_string: old,
                new_string: new,
                replace_all,
            },
            approver(yes),
        )
        .context("edit")?;
    println!("edited {} ({} replacement(s))", path.display(), n);
    Ok(())
}

fn cmd_rm(config: &Config, path: PathBuf, yes: bool) -> Result<()> {
    let ws = workspace(config)?;
    ws.files.delete(&path, approver(yes)).context("delete")?;
    println!("deleted {}", path.display());
    Ok(())
}

fn cmd_mv(config: &Config, from: PathBuf, to: PathBuf, yes: bool) -> Result<()> {
    let ws = workspace(config)?;
    ws.files.rename(&from, &to, approver(yes)).context("mv")?;
    println!("moved {} -> {}", from.display(), to.display());
    Ok(())
}

fn cmd_cp(config: &Config, from: PathBuf, to: PathBuf, overwrite: bool, yes: bool) -> Result<()> {
    let ws = workspace(config)?;
    ws.files
        .copy(&from, &to, CopyOptions { overwrite }, approver(yes))
        .context("cp")?;
    println!("copied {} -> {}", from.display(), to.display());
    Ok(())
}

fn cmd_bulk_edit(
    config: &Config,
    roots: Vec<PathBuf>,
    old: String,
    new: String,
    replace_all: bool,
    dry_run: bool,
    yes: bool,
) -> Result<()> {
    let ws = workspace(config)?;
    let plan = ws
        .files
        .bulk_edit_plan(&roots, &old, &new, replace_all)
        .context("bulk edit plan")?;
    if plan.files.is_empty() {
        println!("no files match");
        return Ok(());
    }
    println!("would modify {} file(s):", plan.files.len());
    for f in &plan.files {
        println!("  {}", f.display());
    }
    if dry_run {
        return Ok(());
    }
    let result = ws
        .files
        .bulk_edit_apply(&plan, approver(yes))
        .context("bulk edit apply")?;
    println!("modified {} file(s)", result.modified.len());
    if !result.skipped.is_empty() {
        println!("skipped {} file(s):", result.skipped.len());
        for (f, err) in &result.skipped {
            println!("  {}: {}", f.display(), err);
        }
    }
    Ok(())
}

fn cmd_grep(
    config: &Config,
    pattern: String,
    glob: Option<String>,
    ignore_case: bool,
    max: usize,
    path: Option<PathBuf>,
) -> Result<()> {
    let ws = workspace(config)?;
    let hits = ws
        .search
        .grep(SearchOptions {
            pattern,
            glob,
            case_insensitive: ignore_case,
            max_matches: max,
            path,
        })
        .context("grep")?;
    for h in &hits {
        let prefix = h
            .project
            .as_ref()
            .map(|p| format!("[{p}] "))
            .unwrap_or_default();
        println!("{prefix}{}:{}:{}", h.path.display(), h.line, h.text);
    }
    eprintln!("— {} match(es)", hits.len());
    Ok(())
}

fn cmd_glob(config: &Config, pattern: String, max: usize) -> Result<()> {
    let ws = workspace(config)?;
    let hits = ws.search.glob(&pattern, max).context("glob")?;
    for h in &hits {
        let prefix = h
            .project
            .as_ref()
            .map(|p| format!("[{p}] "))
            .unwrap_or_default();
        println!("{prefix}{}", h.path.display());
    }
    eprintln!("— {} file(s)", hits.len());
    Ok(())
}

fn cmd_codeint(config: &Config, action: CodeintCmd) -> Result<()> {
    let ws = workspace(config)?;
    let root = ws.project_root.clone();

    match action {
        CodeintCmd::Index { force } => {
            if !force {
                if let Ok(Some(idx)) = SymbolIndex::load(&root) {
                    println!(
                        "index already fresh: {} symbol(s) in {} file(s) — use --force to rebuild",
                        idx.symbols.len(),
                        idx.scanned_files
                    );
                    return Ok(());
                }
            }
            let idx = IndexEngine::new(&root).scan().context("scan symbols")?;
            idx.save(&root).context("save symbol index")?;
            println!(
                "indexed {} symbol(s) in {} file(s) -> {}",
                idx.symbols.len(),
                idx.scanned_files,
                SymbolIndex::file_path(&root).display()
            );
        }
        CodeintCmd::Find { name } | CodeintCmd::Defs { name } => {
            let idx = load_symbol_index(&root)?;
            let hits = idx.query(&name);
            if hits.is_empty() {
                println!("no symbols matching '{name}'");
            } else {
                println!("{} match(es) for '{name}':", hits.len());
                for s in &hits {
                    println!("{:10} {}:{}  {}", s.kind, s.file, s.line, s.name);
                }
            }
        }
        CodeintCmd::Refs {
            name,
            glob,
            ignore_case,
            max,
        } => {
            let hits = ws
                .search
                .grep(SearchOptions {
                    pattern: word_boundary(&name),
                    glob,
                    case_insensitive: ignore_case,
                    max_matches: max,
                    path: None,
                })
                .context("find references")?;
            let hits = filter_out_own_index(&root, hits);
            println!("{} reference(s) to '{name}':", hits.len());
            for h in &hits {
                let prefix = h
                    .project
                    .as_ref()
                    .map(|p| format!("[{p}] "))
                    .unwrap_or_default();
                println!("{prefix}{}:{}:{}", h.path.display(), h.line, h.text);
            }
        }
        CodeintCmd::Rename { old, new } => {
            let hits = ws
                .search
                .grep(SearchOptions {
                    pattern: word_boundary(&old),
                    glob: None,
                    case_insensitive: false,
                    max_matches: 2000,
                    path: None,
                })
                .context("scan rename references")?;
            let hits = filter_out_own_index(&root, hits);
            if hits.is_empty() {
                println!("no references to '{old}' found");
                return Ok(());
            }
            // Group hits by file so the plan is actionable per file.
            let mut by_file: Vec<(PathBuf, Vec<usize>)> = Vec::new();
            for h in &hits {
                match by_file.iter().position(|(p, _)| *p == h.path) {
                    Some(i) => by_file[i].1.push(h.line),
                    None => by_file.push((h.path.clone(), vec![h.line])),
                }
            }
            println!(
                "rename '{old}' -> '{new}': {} reference(s) in {} file(s)",
                hits.len(),
                by_file.len()
            );
            for (f, lines) in &by_file {
                let shown: Vec<String> = lines.iter().take(5).map(|l| l.to_string()).collect();
                let suffix = if lines.len() > 5 { ", …" } else { "" };
                println!(
                    "  {}: {} line(s) [{}]",
                    f.display(),
                    lines.len(),
                    shown.join(", ") + suffix
                );
            }
            println!("plan only — applying the edit is left to a review step.");
        }
    }
    Ok(())
}

fn load_symbol_index(root: &Path) -> Result<SymbolIndex> {
    SymbolIndex::load(root)
        .context("load symbol index")?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no index at {}; run `zeus codeint index` first",
                SymbolIndex::file_path(root).display()
            )
        })
}

fn cmd_project(config: &Config, action: ProjectCmd) -> Result<()> {
    let ws = workspace(config)?;
    let root = ws.project_root.clone();
    match action {
        ProjectCmd::Detect => match zeus_lang::detect_project(&root) {
            Some(lang) => {
                println!("{}", zeus_lang::spec(lang).display_name);
                Ok(())
            }
            None => bail!(
                "could not detect a supported language for {} — run `zeus project commands --help` to list them",
                root.display()
            ),
        },
        ProjectCmd::Commands { lang } => {
            let lang = match lang {
                Some(name) => zeus_lang::Language::from_name(&name).ok_or_else(|| {
                    anyhow::anyhow!("unknown language '{name}' — try a name like rust, ts, go, c#")
                })?,
                None => zeus_lang::detect_project(&root).ok_or_else(|| {
                    anyhow::anyhow!("cannot detect language for {}; pass a language name", root.display())
                })?,
            };
            print_lang_commands(lang, &root);
            Ok(())
        }
        ProjectCmd::Scaffold { lang, name } => {
            let lang = zeus_lang::Language::from_name(&lang).ok_or_else(|| {
                anyhow::anyhow!("unknown language '{lang}' — try `zeus project scaffold --list` for choices")
            })?;
            let target = std::env::current_dir().context("current dir")?.join(&name);
            if target.exists() {
                bail!("{} already exists — pick a different name", target.display());
            }
            let written = zeus_lang::scaffold_project(lang, &name, &target).context("scaffold")?;
            println!(
                "scaffolded {} project '{name}' into {}:",
                zeus_lang::spec(lang).display_name,
                target.display()
            );
            for p in &written {
                println!("  created {}", p.display());
            }
            Ok(())
        }
        ProjectCmd::Format { path } => match path {
            Some(path) => format_one(&root, &path),
            None => format_project(&root),
        },
    }
}

/// Resolve the language for `target`: a file's own extension wins, then the
/// enclosing project root's detection, then the spec table.
fn resolve_lang_for(root: &Path, target: &Path) -> Result<zeus_lang::Language> {
    if let Some(lang) = zeus_lang::detect_source(target) {
        return Ok(lang);
    }
    if let Some(lang) = zeus_lang::detect_project(root) {
        return Ok(lang);
    }
    bail!(
        "cannot detect a language for {} (not a known source file)",
        target.display()
    )
}

fn format_one(root: &Path, path: &Path) -> Result<()> {
    let target = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    if !target.is_file() {
        bail!("{} is not a file", target.display());
    }
    let lang = resolve_lang_for(root, &target)?;
    let spec = zeus_lang::spec(lang);
    if spec.format.is_empty() {
        bail!("no formatter configured for {}", spec.display_name);
    }
    if spec.format_style == zeus_lang::FormatStyle::Project {
        println!(
            "{} uses a project-wide formatter; run `zeus project format` (no path) instead",
            spec.display_name
        );
        return Ok(());
    }
    let args = expand_format_args(spec.format, &target);
    run_argv(&args, root)
}

fn format_project(root: &Path) -> Result<()> {
    let lang = zeus_lang::detect_project(root)
        .ok_or_else(|| anyhow::anyhow!("cannot detect language in {}", root.display()))?;
    let spec = zeus_lang::spec(lang);
    if spec.format.is_empty() {
        bail!("no formatter configured for {}", spec.display_name);
    }
    let args: Vec<String> = spec.format.iter().map(|a| a.to_string()).collect();
    run_argv(&args, root)
}

/// Substitute `{file}` in a per-file format command argv.
fn expand_format_args(template: &[&'static str], target: &Path) -> Vec<String> {
    let path = target.to_string_lossy().into_owned();
    template
        .iter()
        .map(|a| a.replace(zeus_lang::FILE_PLACEHOLDER, path.as_str()))
        .collect()
}

/// Spawn `args[0] args[1..]` with cwd `root`, streaming the program's
/// stdout/stderr through to the terminal and propagating its exit status.
fn run_argv(args: &[String], root: &Path) -> Result<()> {
    let Some((prog, rest)) = args.split_first() else {
        bail!("empty command");
    };
    let status = std::process::Command::new(prog)
        .args(rest)
        .current_dir(root)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run '{prog}': {e}"))?;
    if !status.success() {
        bail!("{prog} exited with {}", status.code().unwrap_or(-1));
    }
    Ok(())
}

fn print_lang_commands(lang: zeus_lang::Language, root: &Path) {
    let s = zeus_lang::spec(lang);
    let fmt_vec = |args: &[&'static str]| -> String {
        if args.is_empty() {
            "(none)".to_string()
        } else {
            args.join(" ")
        }
    };
    println!("project:  {}", root.display());
    println!("language: {} ({})", s.display_name, s.exts.join(", "));
    println!("  build:  {}", fmt_vec(s.build));
    println!("  test:   {}", fmt_vec(s.test));
    println!("  lint:   {}", fmt_vec(s.lint));
    println!("  format: {}", fmt_vec(s.format));
}

fn cmd_rewind(config: &Config, turn_id: String) -> Result<()> {
    let ws = workspace(config)?;
    let n = ws
        .files
        .checkpoints
        .restore(&turn_id, &ws.project_root)
        .context("rewind")?;
    println!("restored {n} snapshot(s) from turn {turn_id}");
    Ok(())
}

fn cmd_checkpoints(config: &Config) -> Result<()> {
    let ws = workspace(config)?;
    let turns = ws
        .files
        .checkpoints
        .list_turns()
        .context("list checkpoints")?;
    if turns.is_empty() {
        println!("(no checkpoints)");
        return Ok(());
    }
    for t in turns {
        println!(
            "{}  files={}  {}",
            t.turn_id,
            t.file_count,
            t.path.display()
        );
    }
    Ok(())
}

fn user_command_dir(config: &Config, global: bool) -> Result<PathBuf> {
    if global {
        return Ok(config.global.commands.clone());
    }
    config
        .project
        .as_ref()
        .map(|p| p.commands.clone())
        .ok_or_else(|| {
            anyhow::anyhow!("no project active; pass --global or run `zeus init --with-project`")
        })
}

fn cmd_user_commands(config: &Config, action: UserCommandCmd, yes: bool) -> Result<()> {
    match action {
        UserCommandCmd::List => {
            let mut seen = std::collections::HashSet::new();
            if let Some(project) = &config.project {
                for name in list_md_names(&project.commands)? {
                    println!("{name}  [project]");
                    seen.insert(name);
                }
            }
            for name in list_md_names(&config.global.commands)? {
                if seen.contains(&name) {
                    println!("{name}  [global, shadowed by project]");
                } else {
                    println!("{name}  [global]");
                }
            }
            Ok(())
        }
        UserCommandCmd::Show { name } => {
            // Project shadows global, same lookup order as SlashCommands::expand.
            if let Some(project) = &config.project {
                let p = project.commands.join(format!("{name}.md"));
                if p.exists() {
                    println!("{}", std::fs::read_to_string(&p)?);
                    return Ok(());
                }
            }
            let p = config.global.commands.join(format!("{name}.md"));
            if p.exists() {
                println!("{}", std::fs::read_to_string(&p)?);
                return Ok(());
            }
            bail!("no command named '{name}' (checked project and global commands/)");
        }
        UserCommandCmd::Add {
            name,
            content,
            global,
        } => {
            let dir = user_command_dir(config, global)?;
            std::fs::create_dir_all(&dir).context("create commands dir")?;
            let body = if content == "-" {
                let mut buf = String::new();
                io::stdin().read_to_string(&mut buf)?;
                buf
            } else {
                content
            };
            let path = dir.join(format!("{name}.md"));
            std::fs::write(&path, body).context("write command template")?;
            println!("wrote {}", path.display());
            Ok(())
        }
        UserCommandCmd::Remove { name, global } => {
            let dir = user_command_dir(config, global)?;
            let path = dir.join(format!("{name}.md"));
            if !path.exists() {
                bail!("no such command: {}", path.display());
            }
            // Outside any project's FileEngine jurisdiction when --global
            // (that engine's path containment would — correctly — refuse a
            // path outside the project root), so this asks directly rather
            // than routing through the Permission Gate.
            if !yes {
                eprint!("Delete {}? [y/N] ", path.display());
                io::stderr().flush().ok();
                let mut line = String::new();
                io::stdin().read_line(&mut line)?;
                if !matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
                    println!("cancelled");
                    return Ok(());
                }
            }
            std::fs::remove_file(&path).context("remove command template")?;
            println!("removed {}", path.display());
            Ok(())
        }
    }
}

fn list_md_names(dir: &Path) -> Result<Vec<String>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("md") {
            if let Some(stem) = path.file_stem() {
                names.push(stem.to_string_lossy().into_owned());
            }
        }
    }
    names.sort();
    Ok(names)
}

/// Detached headless orchestration: re-invoke this binary as a background
/// `agent --auto [--workflow]` run. Uses an argv-based spawn (not a shell
/// wrapper) so the goal text survives untouched — nothing is quoted or
/// expanded. Shared by `zeus bg orchestrate` and the REPL `/bg` command.
pub(crate) fn spawn_bg_orchestrate(
    config: &Config,
    goal: &str,
    workflow: Option<&str>,
) -> anyhow::Result<u64> {
    let ws = workspace(config)?;
    let registry = BackgroundTaskRegistry::new(ws.project_root.join(".agent/background"));
    let exe = std::env::current_exe().context("resolve own executable")?;
    let mut args: Vec<String> = vec!["agent".into(), goal.into(), "--yes".into(), "--auto".into()];
    if let Some(name) = workflow {
        args.push("--workflow".into());
        args.push(name.to_string());
    }
    registry
        .spawn_argv(&exe.to_string_lossy(), args, &ws.project_root)
        .context("spawn background orchestration")
}

fn cmd_bg(config: &Config, action: BgCmd) -> Result<()> {
    let ws = workspace(config)?;
    let registry = BackgroundTaskRegistry::new(ws.project_root.join(".agent/background"));
    match action {
        BgCmd::Run { command } => {
            let id = registry
                .spawn(&command, &ws.project_root)
                .context("spawn background task")?;
            println!("started background task id={id}: {command}");
            Ok(())
        }
        BgCmd::Orchestrate { goal, workflow } => {
            let id = spawn_bg_orchestrate(config, &goal, workflow.as_deref())?;
            println!("started background orchestration id={id}");
            let goal_flat = goal.replace('\n', " ");
            println!("  goal:     {goal_flat}");
            println!(
                "  workflow: {}",
                workflow.as_deref().unwrap_or("(auto-plan)")
            );
            println!("  follow:  zeus bg output {id}   |   stop: zeus bg stop {id}");
            Ok(())
        }
        BgCmd::List => {
            let tasks = registry.list().context("list background tasks")?;
            if tasks.is_empty() {
                println!("(no background tasks)");
                return Ok(());
            }
            for (t, status) in tasks {
                println!("{}  {:?}  pid={}  {}", t.id, status, t.pid, t.command);
            }
            Ok(())
        }
        BgCmd::Output { id } => {
            let (stdout, stderr) = registry.output(id);
            println!("--- stdout ---\n{stdout}--- stderr ---\n{stderr}");
            Ok(())
        }
        BgCmd::Pause { id } => {
            registry.pause(id).context("pause background task")?;
            println!("paused background task {id}");
            Ok(())
        }
        BgCmd::Resume { id } => {
            registry.resume(id).context("resume background task")?;
            println!("resumed background task {id}");
            Ok(())
        }
        BgCmd::Stop { id } => {
            registry.stop(id).context("stop background task")?;
            println!("stopped background task {id}");
            Ok(())
        }
    }
}

fn git_engine(config: &Config) -> Result<GitEngine> {
    let ws = workspace(config)?;
    let gate = PermissionGate::new(config.settings.clone(), ws.project_root.clone());
    Ok(GitEngine::new(ws.project_root, gate))
}

/// Build a `GitEngine` bound to `agent`'s already-live project root — used by
/// the `/diff` REPL and TUI commands, which need git access without
/// re-deriving a whole `Workspace` (that would mint a fresh, unused
/// checkpoint turn directory on every call).
fn git_engine_for_agent(config: &Config, agent: &Agent) -> GitEngine {
    let root = agent.workspace().project_root.clone();
    let gate = PermissionGate::new(config.settings.clone(), root.clone());
    GitEngine::new(root, gate)
}

/// Print a `GitOutput`'s stdout, and on failure also print stderr and
/// return a non-zero-exit error (so scripting against `zeus git ...` can
/// rely on the process exit code, same as running `git` directly).
fn report_git(out: zeus_fs::GitOutput) -> Result<()> {
    print!("{}", out.stdout);
    if !out.success {
        eprint!("{}", out.stderr);
        bail!("git exited with {:?}", out.exit_code);
    }
    Ok(())
}

fn cmd_git(config: &Config, action: GitCmd, yes: bool) -> Result<()> {
    let engine = git_engine(config)?;
    match action {
        GitCmd::Status => report_git(engine.status()?),
        GitCmd::Diff { staged, refs } => {
            let refs_ref: Vec<&str> = refs.iter().map(|s| s.as_str()).collect();
            report_git(engine.diff(staged, &refs_ref)?)
        }
        GitCmd::Log { max, path } => report_git(engine.log(max, path.as_deref())?),
        GitCmd::Show { target } => report_git(engine.show(&target)?),
        GitCmd::Blame { path } => report_git(engine.blame(&path)?),
        GitCmd::Branches => report_git(engine.branch_list()?),
        GitCmd::Remotes => report_git(engine.remote_list()?),
        GitCmd::Tags => report_git(engine.tag_list()?),
        GitCmd::Stashes => report_git(engine.stash_list()?),
        GitCmd::Add { paths } => {
            let paths_ref: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
            report_git(engine.add(&paths_ref, approver(yes))?)
        }
        GitCmd::Commit { message, all } => {
            report_git(engine.commit(&message, all, approver(yes))?)
        }
        GitCmd::Stash { message } => {
            report_git(engine.stash_push(message.as_deref(), approver(yes))?)
        }
        GitCmd::StashPop => report_git(engine.stash_pop(approver(yes))?),
        GitCmd::BranchCreate { name } => report_git(engine.branch_create(&name, approver(yes))?),
        GitCmd::BranchDelete { name, force } => {
            report_git(engine.branch_delete(&name, force, approver(yes))?)
        }
        GitCmd::TagCreate { name, message } => {
            report_git(engine.tag_create(&name, message.as_deref(), approver(yes))?)
        }
        GitCmd::Checkout { target } => report_git(engine.checkout(&target, approver(yes))?),
        GitCmd::Fetch { remote } => report_git(engine.fetch(remote.as_deref(), approver(yes))?),
        GitCmd::Pull => report_git(engine.pull(approver(yes))?),
        GitCmd::Push {
            remote,
            branch,
            force,
        } => report_git(engine.push(remote.as_deref(), branch.as_deref(), force, approver(yes))?),
        GitCmd::CommitAndPush {
            message,
            all,
            remote,
        } => report_git(engine.commit_and_push(&message, all, remote.as_deref(), approver(yes))?),
        GitCmd::Reset { mode, target } => {
            report_git(engine.reset(mode.into(), target.as_deref(), approver(yes))?)
        }
        GitCmd::Revert { target } => report_git(engine.revert(&target, approver(yes))?),
        GitCmd::CherryPick { target } => report_git(engine.cherry_pick(&target, approver(yes))?),
        GitCmd::Rebase { onto } => report_git(engine.rebase(&onto, approver(yes))?),
        GitCmd::Merge { branch } => report_git(engine.merge(&branch, approver(yes))?),
        GitCmd::PrList { limit, state } => {
            report_git(engine.pr_list(&state, limit, approver(yes))?)
        }
        GitCmd::PrView { number } => report_git(engine.pr_view(&number, approver(yes))?),
        GitCmd::PrCreate { title, body, base } => {
            report_git(engine.pr_create(&title, body.as_deref(), base.as_deref(), approver(yes))?)
        }
    }
}

/// One provider's health, checked without necessarily constructing a real
/// client — local providers (ollama/lmstudio/llamacpp) get an actual
/// reachability probe rather than being assumed "ready" just because their
/// `kind` matches, since a stopped local server is exactly the kind of
/// thing `zeus doctor` exists to catch.
struct ProviderHealth {
    ok: bool,
    detail: String,
}

async fn provider_health(cfg: &zeus_config::ProviderConfig) -> ProviderHealth {
    if matches!(cfg.kind.as_str(), "ollama" | "lmstudio" | "llamacpp") {
        let Some(base) = &cfg.base_url else {
            return ProviderHealth {
                ok: false,
                detail: "local provider has no base_url configured".to_string(),
            };
        };
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(1500))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                return ProviderHealth {
                    ok: false,
                    detail: format!("couldn't build http client: {e}"),
                }
            }
        };
        // Any response at all (2xx, 404, whatever) means something is
        // listening — that's the actual question, not whether the root
        // path happens to be a valid route for this server.
        match client.get(base).send().await {
            Ok(_) => ProviderHealth {
                ok: true,
                detail: format!("reachable at {base}"),
            },
            Err(e) => ProviderHealth {
                ok: false,
                detail: format!("unreachable at {base} ({e}) — is it running?"),
            },
        }
    } else if cfg.headers.contains_key("Authorization") {
        ProviderHealth {
            ok: true,
            detail: "auth header configured".to_string(),
        }
    } else {
        match &cfg.api_key_env {
            Some(var) => match std::env::var(var) {
                Ok(k) if !k.is_empty() => ProviderHealth {
                    ok: true,
                    detail: format!("key set via ${var}"),
                },
                _ => ProviderHealth {
                    ok: false,
                    detail: format!("no key — set ${var}"),
                },
            },
            None => ProviderHealth {
                ok: true,
                detail: "no key required".to_string(),
            },
        }
    }
}

async fn cmd_doctor(config: &Config) -> Result<()> {
    println!("zeus — Foundation + Safety Core");
    println!("  version:      {}", env!("CARGO_PKG_VERSION"));
    println!("  global_home:  {}", config.global.root.display());
    println!(
        "  project:      {}",
        config
            .project_root
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".into())
    );
    println!(
        "  default:      {} / {}",
        config.settings.model.provider, config.settings.model.model
    );
    println!();
    println!("Phase 1 — Foundation:");
    println!("  [x] CLI (one-shot subcommands + interactive `zeus` chat mode)");
    println!("  [x] Config (global / project / local layering)");
    println!("  [x] Logging (console + ~/.zeus/logs)");
    println!("  [x] Provider abstraction (chat/stream/list/embeddings/count_tokens)");
    println!("  [x] Providers: ollama, lmstudio, llamacpp");
    println!("  [x] Cloud providers: openai, grok, openrouter, opencodezen, deepseek, gemini (OpenAI-compatible), anthropic");
    println!("  [x] Local-provider auto-detection + reachability fallback");
    println!();
    println!("Phase 2 — Safety Core:");
    println!("  [x] Permission gate (allow/ask/deny)");
    println!("  [x] Path containment");
    println!("  [x] File ops (read/write/edit/delete/rename/copy/move/bulk)");
    println!("  [x] Checkpoints + rewind");
    println!("  [x] Search (grep/glob, cross-project roots)");
    println!();
    println!("Phase 3 — Execution:");
    println!("  [x] Agent loop (tool-calling, context compaction, session persistence)");
    println!("  [x] Terminal execution (piped by default; PTY implemented, opt-in only — see known issue)");
    println!("  [x] Background tasks (zeus bg run/list/output/stop)");
    println!();
    println!("Phase 4 — Extensibility:");
    println!("  [x] Hooks, slash commands, MCP client, native plugin SDK");
    println!();
    println!("Phase 5 — Git & Review:");
    println!("  [x] Git integration (zeus git ..., 24 operations, tiered permissions)");
    println!("  [x] AI commit messages + diff review (composed from git_diff + git_commit — no special code needed)");
    println!("  [x] PR support (git pr create/list/view via gh CLI)");
    println!();
    println!("Phase 6 — Code Intelligence:");
    println!("  [x] Database-free symbol index (.agent/index.json) — zeus codeint index");
    println!("  [x] Find definitions / go-to-definition (zeus codeint find|defs)");
    println!("  [x] Find references via ripgrep, cross-project roots (zeus codeint refs)");
    println!("  [x] Rename proposal (word-boundary reference plan; apply is review-gated)");
    println!("  Next: tree-sitter parsing + LSP manager (definition/diagnostics/format)");
    println!();
    println!("Phase 7 — Plan Mode:");
    println!("  [x] -plan one-shot and /plan slash command (read-only research, no execution)");
    println!("  [x] Structured plan persisted to .agent/tasks.json (`.agent/tasks.json`)");
    println!("  [x] Review-before-execute gate in Auto mode (plan_execute approval prompt)");
    println!("  Next: plan diff preview + per-step status in the tasks.json UI");
    println!();
    println!("Phase 8 — Test & Visual Verification:");
    println!("  [x] test tool: auto-detect + run the repo test suite (cargo/npm/pnpm/yarn/pytest/go/make), parsed pass/fail summary");
    println!("  [x] browser tool: open a URL in the default browser for visual checks + background dev servers");
    println!("  [x] device tool: adb-driven app testing over USB debug / wireless (connect, install, launch, logcat, screenshot, shell)");
    println!();
    let default_provider = config.settings.model.provider.clone();

    println!("Providers:");
    let mut names: Vec<&String> = config.providers.providers.keys().collect();
    names.sort();
    let mut default_ready = false;
    for name in &names {
        let cfg = &config.providers.providers[*name];
        let health = provider_health(cfg).await;
        if **name == default_provider {
            default_ready = health.ok;
        }
        let icon = if health.ok { "✓" } else { "✗" };
        let marker = if **name == default_provider {
            " (default)"
        } else {
            ""
        };
        println!(
            "  [{icon}] {name}{marker} — {}: {}",
            cfg.kind, health.detail
        );
    }
    println!();

    if !default_ready {
        bail!(
            "default provider '{default_provider}' isn't ready — see the Providers list above \
             (missing key, or a local server that isn't running)."
        );
    }

    // Stronger than the health check above: actually constructs the
    // client, catching config-shape errors (a malformed base_url, an
    // unsupported `kind`) the lighter per-provider check above can't see.
    match create_default(&default_provider, &config.providers) {
        Ok(_) => println!("doctor: default provider '{default_provider}' constructs OK"),
        Err(e) => {
            error!(provider = %default_provider, error = ?e, "default provider failed");
            bail!("default provider '{default_provider}' failed to construct: {e}");
        }
    }
    Ok(())
}
