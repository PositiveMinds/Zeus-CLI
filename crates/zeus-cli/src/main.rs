//! zeus — database-free AI coding agent CLI.

mod tui;
mod ui;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use zeus_agent::{
    load_custom_personas, personas_by_department, Agent, AgentEvent, AgentOptions,
    BackgroundTaskRegistry, ContextManager, ExpandResult, HookRunner, McpClient, SessionStore,
    SlashCommands, TerminalRunner, ToolManager,
};
use zeus_config::Config;
use zeus_fs::{
    ApprovalDecision, CopyOptions, EditOptions, GitEngine, PermissionGate, PermissionRequest,
    ReadOptions, ResetMode, SearchOptions, Workspace, WriteOptions,
};
use zeus_logging::{init as init_logging, LoggingOptions};
use zeus_provider::{
    create_default, create_provider, ChatRequest, Message, ModelProvider, StreamEvent,
};
use std::io::{self, IsTerminal, Read as IoRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tracing::{error, info};

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
        /// Scan local model files instead of querying a provider's API —
        /// finds downloaded-but-not-yet-served models too.
        #[arg(long)]
        local: bool,
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
    Rm {
        path: PathBuf,
    },

    /// Rename or move a file/directory (git mv-aware)
    Mv {
        from: PathBuf,
        to: PathBuf,
    },

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
    Rewind {
        turn_id: String,
    },

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

    /// Manage user-defined slash command templates (.agent/commands/,
    /// ~/.zeus/commands/) — distinct from the built-in REPL /help etc.
    Commands {
        #[command(subcommand)]
        action: UserCommandCmd,
    },
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
    /// List background tasks and their running/exited status
    List,
    /// Print captured stdout/stderr for a background task
    Output { id: u64 },
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

#[tokio::main]
async fn main() {
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
        });
        return cmd_init(&cli).await;
    }

    let config = load_config(&cli)?;
    let level = cli
        .log_level
        .clone()
        .unwrap_or_else(|| config.settings.logging.level.clone());
    let _ = init_logging(LoggingOptions {
        level,
        file: config.settings.logging.file,
        logs_dir: Some(config.global.logs.clone()),
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
        Some(Commands::Models { provider, local }) => cmd_models(&config, provider, local).await,
        Some(Commands::Agent {
            message,
            provider,
            model,
            session,
        }) => cmd_agent(&config, message, provider, model, session, cli.yes).await,
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
        Some(Commands::Doctor) => cmd_doctor(&config),
        Some(Commands::Bg { action }) => cmd_bg(&config, action),
        Some(Commands::Git { action }) => cmd_git(&config, action, cli.yes),
        Some(Commands::Pull { source }) => cmd_pull(&config, source).await,
        Some(Commands::Commands { action }) => cmd_user_commands(&config, action, cli.yes),
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
                eprintln!("{preview}");
            }
            return ApprovalDecision::Approved;
        }
        if let Some(preview) = &req.preview {
            eprintln!("{preview}");
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
            if !zeus_provider::is_provider_reachable(cfg, std::time::Duration::from_millis(800)).await
            {
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
/// provider name — backs the `/model` picker's multi-provider view. A
/// provider that's unreachable, misconfigured, or slow to respond (bounded
/// by a short timeout) is silently skipped rather than blocking the whole
/// picker on one bad entry, same "best effort" spirit as MCP server connect.
pub(crate) async fn list_models_by_provider(
    config: &Config,
) -> Vec<(String, Vec<zeus_provider::ModelInfo>)> {
    let mut names: Vec<&String> = config.providers.providers.keys().collect();
    names.sort();

    let mut groups = Vec::new();
    for name in names {
        let Ok(provider) = zeus_provider::create_provider(name, &config.providers) else {
            continue;
        };
        let fetch = tokio::time::timeout(std::time::Duration::from_secs(3), provider.list_models());
        if let Ok(Ok(models)) = fetch.await {
            if !models.is_empty() {
                groups.push((name.clone(), models));
            }
        }
    }
    groups
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
            Message::system(
                "You are zeus, a helpful coding assistant. Be concise and accurate.",
            ),
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

async fn cmd_models(config: &Config, provider: Option<String>, local: bool) -> Result<()> {
    if local {
        let extra_dirs: Vec<PathBuf> = config
            .settings
            .extra_model_dirs
            .iter()
            .map(PathBuf::from)
            .collect();
        let found = zeus_provider::scan_local_models(&config.global.models, &extra_dirs);
        if found.is_empty() {
            println!("(no local model files found)");
            println!("scanned: {}", config.global.models.display());
            return Ok(());
        }
        for f in found {
            let size_mb = f.size_bytes as f64 / (1024.0 * 1024.0);
            println!(
                "{}  {:.1} MB  [{}]",
                f.path.display(),
                size_mb,
                f.source
            );
        }
        return Ok(());
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

async fn cmd_pull(config: &Config, source: PullCmd) -> Result<()> {
    match source {
        PullCmd::Ollama { model } => {
            let cfg = config
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
            let path = zeus_provider::download_hf_file(&repo, &file, &dest_dir, |downloaded, total| {
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

/// Best-effort connect to every MCP server in `config.settings.mcp_servers`:
/// a server that fails to start (bad command, crashes during handshake)
/// logs a warning and is skipped rather than failing the whole agent turn —
/// one misconfigured server shouldn't take down every other tool.
fn connect_configured_mcp_servers(config: &Config, project_root: &std::path::Path) -> Vec<McpClient> {
    config
        .settings
        .mcp_servers
        .iter()
        .filter_map(|s| {
            match McpClient::connect(&s.name, &s.command, &s.args, project_root) {
                Ok(client) => {
                    info!(server = %s.name, tools = client.tools().len(), "connected MCP server");
                    Some(client)
                }
                Err(e) => {
                    error!(server = %s.name, ?e, "failed to connect MCP server; skipping");
                    None
                }
            }
        })
        .collect()
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
    let tools = ToolManager::new(
        ws,
        terminal,
        background,
        hooks,
        mcp_clients,
        plugins,
        Arc::new(AtomicBool::new(false)),
    );
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
        state.messages.push(Message::system(
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
        ));
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
        },
    ))
}

/// Print one `AgentEvent` to stdout/stderr — shared by the one-shot `agent`
/// subcommand and the REPL so both render turns identically.
/// Printed once before each turn starts, so `print_agent_event` itself can
/// stay a stateless per-event renderer (it has no notion of "first delta").
fn print_turn_header() {
    println!("{}", ui::styled(ui::assistant_marker_style(), "● assistant"));
}

fn print_agent_event(ev: AgentEvent) {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    match ev {
        AgentEvent::TextDelta(t) => {
            let _ = write!(out, "{t}");
            let _ = out.flush();
        }
        AgentEvent::ToolCallStarted { name, arguments, .. } => {
            let _ = writeln!(
                out,
                "\n{}",
                ui::styled(ui::tool_style(), &format!("⚙ {name} {arguments}"))
            );
        }
        AgentEvent::ToolCallFinished { name, is_error, result, .. } => {
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
        AgentEvent::PlanGenerated { steps } => {
            eprintln!(
                "{}",
                ui::styled(
                    ui::dim_style(),
                    &format!(
                        "plan · {} step(s): {}",
                        steps.len(),
                        steps
                            .iter()
                            .map(|s| s.description.clone())
                            .collect::<Vec<_>>()
                            .join(" → ")
                    )
                )
            );
        }
        AgentEvent::PlanStepStarted { step } => {
            eprintln!(
                "{}",
                ui::styled(ui::warn_style(), &format!("{}. {}", step.id, step.description))
            );
        }
        AgentEvent::PlanReviewed { persona, report } => {
            let _ = writeln!(
                out,
                "{}",
                ui::styled(
                    ui::tool_style(),
                    &format!("◆ review ({persona}): {}", report.chars().take(300).collect::<String>())
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
        AgentEvent::OrchestrationDone { summary } => {
            let _ = writeln!(out, "\n{}", ui::styled(ui::assistant_marker_style(), "● plan complete"));
            let _ = writeln!(out, "{summary}");
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

async fn cmd_agent(
    config: &Config,
    message: String,
    provider_name: Option<String>,
    model: Option<String>,
    session: Option<String>,
    yes: bool,
) -> Result<()> {
    let message = expand_slash_command(config, message);
    let mut agent = build_agent(config, provider_name, model, session).await?;

    print_turn_header();
    let result = agent
        .run_turn(&message, print_agent_event, approver(yes))
        .await
        .context("agent turn")?;

    writeln!(io::stdout())?;
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
    ("clear", "start a fresh session (new session id, empty context)"),
    ("new", "start a fresh session (alias for /clear)"),
    ("compact", "force context compaction now, even under the auto threshold"),
    ("autocompact", "toggle auto-compaction: /autocompact on|off"),
    ("context", "show token usage against the model's context window"),
    ("model", "switch model (opens a picker), or /model <name> directly"),
    ("mode", "set agent mode: /mode build|plan|auto (Tab also cycles)"),
    ("session", "show the current session id"),
    ("agents", "list the specialist-agents roster grouped by department (/agents count)"),
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
async fn cmd_repl(config: &Config, yes: bool) -> Result<()> {
    let agent = build_agent(config, None, None, None).await?;
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
                    agent = build_agent(config, None, None, None).await?;
                    println!("cleared — new session={}", agent.session_id());
                }
                "new" => {
                    agent = build_agent(config, None, None, None).await?;
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
                "model" => {
                    if arg.is_empty() {
                        println!("current model: {}", agent.model());
                    } else {
                        agent.set_model(arg.to_string());
                        println!("switched to model: {arg}");
                    }
                }
                "session" => println!("session={}", agent.session_id()),
                "agents" => {
                    if arg.to_ascii_lowercase() == "count" {
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
                "mode" => {
                    match arg.to_ascii_lowercase().as_str() {
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
        let result = match agent.run_turn(&message, print_agent_event, approver(yes)).await {
            Ok(result) => result,
            Err(e) => {
                watcher.abort();
                eprintln!("\n{}", ui::styled(ui::error_style(), &format!("turn failed: {e:#}")));
                continue;
            }
        };
        watcher.abort();

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
    println!(
        "tokens={} approximate={}",
        resp.tokens, resp.approximate
    );
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
        .read(
            &path,
            ReadOptions { offset, limit },
        )
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
        .write(
            &path,
            &body,
            WriteOptions::default(),
            approver(yes),
        )
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
    ws.files
        .delete(&path, approver(yes))
        .context("delete")?;
    println!("deleted {}", path.display());
    Ok(())
}

fn cmd_mv(config: &Config, from: PathBuf, to: PathBuf, yes: bool) -> Result<()> {
    let ws = workspace(config)?;
    ws.files
        .rename(&from, &to, approver(yes))
        .context("mv")?;
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
        println!(
            "{prefix}{}:{}:{}",
            h.path.display(),
            h.line,
            h.text
        );
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
    let turns = ws.files.checkpoints.list_turns().context("list checkpoints")?;
    if turns.is_empty() {
        println!("(no checkpoints)");
        return Ok(());
    }
    for t in turns {
        println!("{}  files={}  {}", t.turn_id, t.file_count, t.path.display());
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
        .ok_or_else(|| anyhow::anyhow!("no project active; pass --global or run `zeus init --with-project`"))
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

fn cmd_bg(config: &Config, action: BgCmd) -> Result<()> {
    let ws = workspace(config)?;
    let registry = BackgroundTaskRegistry::new(ws.project_root.join(".agent/background"));
    match action {
        BgCmd::Run { command } => {
            let id = registry.spawn(&command, &ws.project_root).context("spawn background task")?;
            println!("started background task id={id}: {command}");
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
        GitCmd::Commit { message, all } => report_git(engine.commit(&message, all, approver(yes))?),
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
        GitCmd::Push { remote, branch, force } => report_git(engine.push(
            remote.as_deref(),
            branch.as_deref(),
            force,
            approver(yes),
        )?),
        GitCmd::CommitAndPush { message, all, remote } => report_git(engine.commit_and_push(
            &message,
            all,
            remote.as_deref(),
            approver(yes),
        )?),
        GitCmd::Reset { mode, target } => {
            report_git(engine.reset(mode.into(), target.as_deref(), approver(yes))?)
        }
        GitCmd::Revert { target } => report_git(engine.revert(&target, approver(yes))?),
        GitCmd::CherryPick { target } => report_git(engine.cherry_pick(&target, approver(yes))?),
        GitCmd::Rebase { onto } => report_git(engine.rebase(&onto, approver(yes))?),
        GitCmd::Merge { branch } => report_git(engine.merge(&branch, approver(yes))?),
        GitCmd::PrList { limit, state } => report_git(engine.pr_list(&state, limit, approver(yes))?),
        GitCmd::PrView { number } => report_git(engine.pr_view(&number, approver(yes))?),
        GitCmd::PrCreate { title, body, base } => {
            report_git(engine.pr_create(&title, body.as_deref(), base.as_deref(), approver(yes))?)
        }
    }
}

fn cmd_doctor(config: &Config) -> Result<()> {
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
    println!("  [x] Cloud providers: openai, grok, openrouter, opencodezen, gemini (OpenAI-compatible), anthropic");
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
    println!("Next: Phase 6 — Code Intelligence");
    let default_provider = config.settings.model.provider.clone();
    match create_default(&default_provider, &config.providers) {
        Ok(_) => println!("doctor: provider '{default_provider}' OK"),
        Err(e) => {
            error!(provider = %default_provider, error = ?e, "default provider failed");
            bail!("default provider '{default_provider}' failed: {e}");
        }
    }
    Ok(())
}
