//! zeus — database-free AI coding agent CLI.

mod catalog;
mod clipboard;
mod decor;
mod highlight;
mod tui;
mod ui;
mod update;

use anyhow::{bail, Context, Result};
use clap::Parser;
use futures::StreamExt;
use std::io::{self, IsTerminal, Read as IoRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tracing::{error, info, warn};
use zeus_agent::{
    discover_workflows, personas_by_department, Agent, AgentEvent, AgentOptions,
    BackgroundTaskRegistry, ContextManager, ConversationState, ExpandResult, HookRunner, McpClient,
    SessionStore, SlashCommands, TerminalRunner, ToolManager, TurnResult,
};
use zeus_config::{Config, KeysFile};
use zeus_fs::{filter_out_own_index, word_boundary, IndexEngine, SymbolIndex};
use zeus_fs::{
    CopyOptions, EditOptions, GitEngine, PermissionGate, ReadOptions, SearchOptions, WriteOptions,
};
use zeus_logging::{init as init_logging, LoggingOptions};
use zeus_provider::{
    create_default, create_provider, ChatRequest, Message, ModelProvider, StreamEvent,
    UnconfiguredProvider,
};

mod cli;
mod config;
pub use cli::{
    BgCmd, Cli, CodeintCmd, Commands, ConfigCmd, GitCmd, KeyCmd, ProjectCmd, PullCmd, RagindexCmd,
    ResetModeArg, SessionsCmd, UserCommandCmd,
};
use config::{
    approver, get_toml_path, load_config, load_toml_or_empty, parse_toml_scalar, set_toml_path,
    settings_file_path, workspace,
};

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
        None => cmd_repl(&config, cli.yes, cli.fresh).await,
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
        Some(Commands::Sessions { action }) => match action {
            Some(SessionsCmd::List) | None => cmd_sessions(&config),
            Some(SessionsCmd::Export { id, output }) => cmd_sessions_export(&config, &id, output),
        },
        Some(Commands::Update { check }) => {
            update::cmd_update(check, config.settings.notify_on_completion).await
        }
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
        Some(Commands::Upload { paths, to, dry_run }) => cmd_upload(&config, paths, to, dry_run),
        Some(Commands::Completion { shell }) => {
            use clap::CommandFactory;
            let mut cmd = crate::Cli::command();
            clap_complete::generate(shell.to_clap(), &mut cmd, "zeus", &mut io::stdout());
            Ok(())
        }
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
        Some(Commands::Ragindex { action }) => cmd_ragindex(&config, action).await,
        Some(Commands::Project { action }) => {
            cmd_project(&config, action, cli.project_root.as_deref())
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
        ensure_agent_gitignored(&root)?;
    }
    Ok(())
}

/// If `root` is a git repo, make sure `.agent/` is ignored — uploaded files
/// (`.agent/uploads/`) can be secrets, and a fresh project's `.gitignore`
/// won't have the rule `zeus` itself relies on.
fn ensure_agent_gitignored(root: &std::path::Path) -> Result<()> {
    if !root.join(".git").exists() {
        return Ok(());
    }
    let gitignore = root.join(".gitignore");
    let existing = std::fs::read_to_string(&gitignore).unwrap_or_default();
    let already = existing.lines().any(|l| {
        let t = l.trim();
        t == ".agent" || t == ".agent/" || t == "**/.agent" || t == "**/.agent/" || t == "/.agent"
    });
    if already {
        return Ok(());
    }
    let mut out = existing;
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(".agent/\n");
    std::fs::write(&gitignore, out).with_context(|| format!("write {}", gitignore.display()))?;
    println!("  gitignore: added `.agent/` to {}", gitignore.display());
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
                            "no model provider is reachable. Start a local server (ollama/lmstudio/llamacpp) or configure a cloud provider and set its API key, then run `zeus config set model.provider <name>` (or pass `--provider <name>`)"
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

/// If `err` reads like an out-of-credits / billing failure from a cloud
/// provider (the OpenRouter 402 we've seen in the wild), return a concrete
/// "switch to a provider that has a key" hint — listing exactly which
/// configured providers are usable right now. `None` for anything else, so
/// callers fall through to the generic error handling untouched.
/// Providers other than `exclude` that currently have a usable key: a stored
/// key, an embedded Authorization header, a non-empty api_key_env variable,
/// or a local kind that needs no key at all. Sorted for stable output.
fn providers_with_keys(config: &Config, exclude: &str) -> Vec<String> {
    let stored = KeysFile::load(&config.global.keys_toml).unwrap_or_default();
    let mut names: Vec<String> = config
        .providers
        .providers
        .iter()
        .filter(|(name, cfg)| {
            if name.as_str() == exclude {
                return false; // it's the one that just failed
            }
            let local = matches!(cfg.kind.as_str(), "ollama" | "lmstudio" | "llamacpp");
            local
                || stored.get(name).is_some()
                || cfg.headers.contains_key("Authorization")
                || cfg
                    .api_key_env
                    .as_ref()
                    .map(|var| std::env::var(var).map(|k| !k.is_empty()).unwrap_or(false))
                    .unwrap_or(false)
        })
        .map(|(name, _)| name.clone())
        .collect();
    names.sort();
    names
}

pub(crate) fn credit_failure_hint(config: &Config, err: &str) -> Option<String> {
    let lower = err.to_lowercase();
    let is_credit = [
        "402",
        "insufficient credits",
        "out of credits",
        "credit limit",
        "more credits",
        "billing",
        "payment required",
    ]
    .iter()
    .any(|k| lower.contains(k));
    if !is_credit {
        return None;
    }
    let with_key = providers_with_keys(config, &config.settings.model.provider);
    if with_key.is_empty() {
        return Some(
            "this provider is out of credits — add credits at its billing page, or set a key \
             for another provider with `zeus key set <name>` (see `zeus key list`)"
                .to_string(),
        );
    }
    Some(format!(
        "this provider is out of credits — {} has a key. Switch with `/provider {}` \
         (TUI) or `zeus chat --provider {}` (one-shot).",
        with_key.join(", "),
        with_key[0],
        with_key[0]
    ))
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
        let models = match zeus_provider::create_provider(name, &config.providers) {
            Ok(provider) => {
                let fetch =
                    tokio::time::timeout(std::time::Duration::from_secs(3), provider.list_models());
                match fetch.await {
                    Ok(Ok(models)) if !models.is_empty() => Some(models),
                    // Live probe failed (no key, server down, timeout) — fall
                    // back to the curated catalog so the provider still shows
                    // up grouped in the picker with as many models as possible.
                    _ => catalog::known_models(name),
                }
            }
            Err(_) => catalog::known_models(name),
        };
        models.map(|m| (name.clone(), m))
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
            Message::system(format!(
                "You are zeus, a helpful coding assistant. Be concise and accurate. \
                 Today's date is {today}; the current local time is {time_now}.",
                today = current_date(),
                time_now = current_time(),
            )),
            Message::user(message),
        ],
    );

    run_chat_with_failover(config, &*provider, model, request, stream).await
}

/// Whether a provider error is a good candidate for failing over to another
/// configured provider: out-of-credits (402 / billing language) or a rate
/// limit (429). Contract errors (400/401/403/404) and 5xx are not — 5xx is
/// provider-specific and usually affects all endpoints at once.
fn failover_relevant(err: &anyhow::Error) -> bool {
    let Some(provider_err) = err.downcast_ref::<zeus_provider::ProviderError>() else {
        return false;
    };
    match provider_err {
        zeus_provider::ProviderError::Http { status, message } => {
            *status == 402 || *status == 429 || {
                let lower = message.to_lowercase();
                lower.contains("credit")
                    || lower.contains("billing")
                    || lower.contains("quota")
                    || lower.contains("rate limit")
            }
        }
        _ => false,
    }
}

/// Send one chat request, streaming or not.
async fn attempt_chat(
    provider: &dyn ModelProvider,
    request: ChatRequest,
    stream: bool,
) -> anyhow::Result<()> {
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

/// One-shot `chat`: send with the requested provider, and if it bounces with
/// out-of-credits or a rate limit, silently fail over to the next configured
/// provider that has a key (mirroring `credit_failure_hint`'s candidate set).
async fn run_chat_with_failover(
    config: &Config,
    primary: &dyn ModelProvider,
    primary_model: String,
    request: ChatRequest,
    stream: bool,
) -> Result<()> {
    match attempt_chat(primary, request.clone(), stream).await {
        Ok(()) => Ok(()),
        Err(err) if failover_relevant(&err) => {
            let candidates = providers_with_keys(config, primary.id());
            if candidates.is_empty() {
                return Err(err).context(format!(
                    "no other provider with a key — add credits to {} or run `zeus chat --provider <name>`",
                    primary.id()
                ));
            }
            let mut last = err;
            for name in candidates {
                let Ok(prov) = create_provider(&name, &config.providers) else {
                    continue;
                };
                let model = config
                    .providers
                    .get(&name)
                    .and_then(|c| c.default_model.clone())
                    .unwrap_or_else(|| primary_model.clone());
                eprintln!(
                    "{} is out of credits / rate-limited — failing over to {} ({model})",
                    primary.id(),
                    name
                );
                // Rebuild the request with the failover provider's own default
                // model — the primary's model ID is provider-specific (e.g.
                // OpenCode Zen's `deepseek-v4-flash-free` means nothing to
                // OpenRouter) and would otherwise 400.
                let mut swapped = request.clone();
                swapped.model = model.clone();
                match attempt_chat(&*prov, swapped, stream).await {
                    Ok(()) => return Ok(()),
                    Err(e) => last = e,
                }
            }
            Err(last).context(format!(
                "all failover providers failed (from {})",
                primary.id()
            ))
        }
        Err(err) => Err(err).context("chat failed"),
    }
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
                            let Some(pct) = downloaded
                                .checked_mul(100)
                                .and_then(|n| n.checked_div(total))
                            else {
                                return;
                            };
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
        // Treat as a `repo/file` GGUF download. HF repo IDs themselves are
        // `org/model`, so split on the LAST slash — `org/model/file.gguf`
        // must yield repo=`org/model`, file=`file.gguf`.
        let Some((repo, file)) = requested.rsplit_once('/') else {
            bail!("usage: zeus serve <model-name>  or  zeus serve <repo>/<file.gguf>");
        };
        if repo.is_empty() || file.is_empty() {
            bail!("usage: zeus serve <model-name>  or  zeus serve <repo>/<file.gguf>");
        }
        zeus_config::LocalModelEntry {
            name: requested.clone(),
            repo: repo.to_string(),
            file: file.to_string(),
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

/// `zeus sessions export <id> [--output <file>]` — render a saved
/// conversation as Markdown so it can be shared, archived, or grepped
/// outside the TUI.
fn cmd_sessions_export(config: &Config, id: &str, output: Option<PathBuf>) -> Result<()> {
    let store = SessionStore::new(config.global.sessions.clone());
    let state = store.load(id).context("load session")?;
    if state.messages.is_empty() {
        bail!("session {id} has no messages to export");
    }
    let md = render_session_markdown(&state);
    let path = output.unwrap_or_else(|| PathBuf::from(format!("{id}.md")));
    std::fs::write(&path, md)?;
    println!("exported session {id} to {}", path.display());
    Ok(())
}

/// Render a conversation as a readable Markdown transcript.
fn render_session_markdown(state: &ConversationState) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Session {}\n\n", state.session_id));
    for msg in &state.messages {
        match msg.role {
            zeus_provider::Role::System => {
                out.push_str(&format!("> **system**\n\n{}\n\n", msg.content.trim()));
            }
            zeus_provider::Role::User => {
                out.push_str(&format!("## User\n\n{}\n\n", msg.content.trim()));
                if !msg.images.is_empty() {
                    out.push_str(&format!(
                        "*[{} image attachment(s) omitted]*\n\n",
                        msg.images.len()
                    ));
                }
            }
            zeus_provider::Role::Assistant => {
                out.push_str("## Assistant\n\n");
                if !msg.content.trim().is_empty() {
                    out.push_str(&format!("{}\n\n", msg.content.trim()));
                }
                for call in &msg.tool_calls {
                    out.push_str(&format!(
                        "```\n[call] {} ({})\n{}\n```\n\n",
                        call.name, call.id, call.arguments
                    ));
                }
                if !msg.images.is_empty() {
                    out.push_str(&format!(
                        "*[{} image attachment(s) omitted]*\n\n",
                        msg.images.len()
                    ));
                }
            }
            zeus_provider::Role::Tool => {
                out.push_str(&format!(
                    "## Tool{} result\n\n{}\n\n",
                    msg.tool_call_id
                        .as_deref()
                        .map(|id| format!(" ({id})"))
                        .unwrap_or_default(),
                    msg.content.trim()
                ));
            }
        }
    }
    out
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
    tools.set_embedding(provider.clone(), model.clone());
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
        if let Some(survey) = build_project_survey(config.project_root.as_deref()) {
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
            max_tool_iterations: 16,
            temperature: config.settings.model.temperature,
            // Bounds worst-case reply latency — otherwise an ungrounded
            // ramble (especially on slow CPU-bound local inference) keeps
            // generating for as long as the model's context window allows.
            // `model.max_tokens` in settings.toml overrides this.
            max_tokens: Some(config.settings.model.max_tokens.unwrap_or(1024)),
            max_parallel_read_steps: config.settings.max_parallel_read_steps.unwrap_or(2),
            tasks_file: config.project.as_ref().map(|p| p.tasks_json.clone()),
        },
    ))
}

/// The agent's standing instructions (system prompt): identity, tool
/// discipline, and the anti-hallucination grounding rules. New sessions get
/// this as their leading system message; resumed sessions keep their own
/// (their stale date is harmless — the agent can always call the
/// `current_time` tool for a fresh reading, which is also how a session that
/// runs past midnight keeps an accurate sense of "today").
fn system_prompt(_config: &Config) -> Message {
    Message::system(format!(
        "You are zeus, a helpful coding assistant with access to file, git, terminal, \
             and search tools. Today's date is {today}, and the current local time is \
             {time_now}. For any date/time question, answer directly from these facts or, \
             when precision matters or the session has been running a while, call the \
             current_time tool — never guess a date from your training data.\n\
             \n\
             Default to replying in plain text with no tool call at all. \
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
        today = current_date(),
        time_now = current_time(),
    ))
}

/// `2026-08-18` — the session-start date, stamped into the system prompt.
fn current_date() -> String {
    chrono::Local::now().format("%Y-%m-%d").to_string()
}

/// `HH:MM:SS` (local) at session start, so a "what time is it" right after
/// opening zeus is answerable without a tool round-trip.
fn current_time() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

/// A factual, bounded snapshot of the project the agent is operating in,
/// injected into *new* sessions as a system message. Purpose: ground the
/// model in real project facts at session start so it doesn't hallucinate
/// structure (guessing manifests, frameworks, layouts) — everything here is
/// actually walked from disk and explicitly labeled as such. Kept deliberately
/// small and capped: on a huge tree it enumerates only the top level plus a
/// depth-limited walk, and never reads file bodies.
fn build_project_survey(project_root: Option<&Path>) -> Option<String> {
    let root = project_root?;
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
    // Detected dev stack from the language/framework tables. These are the
    // exact commands `verify`/`test` would run — prefer them over guessing.
    if let Some(lang) = zeus_lang::detect_project(root) {
        let s = zeus_lang::spec(lang);
        let fmt = |args: &[&'static str]| {
            if args.is_empty() {
                "(none)".to_string()
            } else {
                args.join(" ")
            }
        };
        lines.push(format!("detected language: {}", s.display_name));
        lines.push(format!("  build:  {}", fmt(s.build)));
        lines.push(format!("  test:   {}", fmt(s.test)));
        lines.push(format!("  lint:   {}", fmt(s.lint)));
        lines.push(format!("  format: {}", fmt(s.format)));
    }
    if let Some(fw) = zeus_lang::Framework::detect_framework(root) {
        let fs = zeus_lang::framework_spec(fw);
        lines.push(format!(
            "detected framework: {} (base {})",
            fs.display_name,
            zeus_lang::spec(fs.base).display_name
        ));
    }
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
    ("theme", "switch color theme: dark, light, or high-contrast"),
    (
        "mouse",
        "TUI only: /mouse off disables mouse capture so your terminal's native click-drag text selection works; /mouse on restores click/scroll/chips",
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
        "read-only next-feature recommendations grounded in what exists (/suggest [context])",
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
        "run the workforce in the background (/bg <goal>); /bg list|output <id>|pause <id>|resume <id>|stop <id> to manage (TUI: in-session; plain REPL: via `zeus bg ...`)",
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
    (
        "upload",
        "copy files (anywhere on disk) into .agent/uploads/ and tell the agent to read them: /upload [--to SUBDIR] <path> [path ...]",
    ),
    (
        "uploads",
        "list what's currently staged under .agent/uploads/: /uploads",
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

/// The `/theme` slash command (plain REPL): view or persist the TUI color
/// theme preset. The plain REPL itself has no chrome to repaint, so a
/// change here just saves — it applies live inside the TUI (`tui.rs`
/// applies it immediately via `theme::set_theme` and saves the same way).
fn handle_theme_slash(arg: &str, config: &Config) {
    let path = &config.global.settings_toml;
    let names = tui::theme::ThemeKind::ALL
        .iter()
        .map(|k| k.label())
        .collect::<Vec<_>>()
        .join(", ");
    match arg.trim() {
        "" => println!(
            "theme: {} (available: {names})",
            config.settings.theme.as_deref().unwrap_or("dark")
        ),
        name => match tui::theme::ThemeKind::from_label(name) {
            Some(_) => match zeus_config::set_theme(path, Some(name.to_string())) {
                Ok(()) => println!("theme: {name} (takes effect next TUI launch)"),
                Err(e) => eprintln!("couldn't save setting: {e}"),
            },
            None => eprintln!("'{name}' isn't a theme — try {names}"),
        },
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

async fn cmd_repl(config: &Config, yes: bool, fresh: bool) -> Result<()> {
    // Auto-resume the most recently used saved session unless the user asked
    // for a fresh one with `--new`. A crashed/killed session therefore picks
    // right back up where it left off (state files are written atomically,
    // so a corrupt file still resolves to a clean empty session rather than
    // a hard launch failure).
    let store = SessionStore::new(config.global.sessions.clone());
    if !fresh {
        if let Ok(Some(id)) = store.most_recent() {
            let agent = build_agent_repl_with_session(config, None, None, id).await?;
            if config.project_root.is_some() {
                agent.set_plan_mode(true);
            }
            if io::stdin().is_terminal() && io::stdout().is_terminal() {
                return tui::run(config, agent, yes).await;
            }
            return run_plain_repl(config, agent, yes).await;
        }
    }
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

        // A successful /upload copies the files and then falls through to a
        // normal turn whose message tells the agent what arrived — so this
        // override is declared at loop scope, not inside the slash block.
        let mut upload_message: Option<String> = None;

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
                "upload" => {
                    let (to, paths, parse_err) = parse_upload_args(arg);
                    if let Some(e) = parse_err {
                        eprintln!("upload failed: {e}");
                    } else if paths.is_empty() {
                        eprintln!(
                            "usage: /upload [--to SUBDIR] <path> [path ...] — copies files \
                             (anywhere on disk) into .agent/uploads/ and tells the agent to read them; \
                             quote paths with spaces: /upload \"my file.png\""
                        );
                    } else {
                        match upload_files(config, &paths, to.as_deref(), false) {
                            Ok(report) if !report.uploaded.is_empty() => {
                                for rel in &report.uploaded {
                                    println!("uploaded: {rel}");
                                }
                                for w in &report.warnings {
                                    println!("warning: {w}");
                                }
                                upload_message = Some(format!(
                                    "The user uploaded the following file(s). Read each one as \
                                     appropriate — read for text, read_image for images, \
                                     read_document for PDF/office docs:\n{}",
                                    report.uploaded.join("\n")
                                ));
                                handled = false; // send the turn below
                            }
                            Ok(_) => eprintln!("no files uploaded"),
                            Err(e) => eprintln!("upload failed: {e:#}"),
                        }
                    }
                }
                "uploads" => match arg.split_whitespace().next() {
                    Some("rm") => {
                        let rel = arg["rm".len()..].trim();
                        match delete_upload(config, rel) {
                            Ok(n) => println!("removed {n} item(s) from .agent/uploads"),
                            Err(e) => eprintln!("remove failed: {e:#}"),
                        }
                    }
                    _ => match list_uploads(config) {
                        Ok(entries) if entries.is_empty() => {
                            println!("no uploads staged yet — use /upload <path> to add files");
                        }
                        Ok(entries) => {
                            println!("staged uploads:");
                            for e in entries {
                                if e.size == 0 {
                                    println!("  {}  (dir)", e.rel);
                                } else {
                                    println!("  {}  {}", e.rel, human_size(e.size));
                                }
                            }
                            println!("use `/uploads rm <rel-path>` to remove one");
                        }
                        Err(e) => eprintln!("uploads listing failed: {e:#}"),
                    },
                },
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
                        match persist_default_provider(config, &config.settings.model.provider, Some(agent.model())) {
                            Ok(path) => println!(
                                "switched to model: {arg} — saved to {}",
                                path.display()
                            ),
                            Err(e) => eprintln!("switched to model, but saving default failed: {e:#}"),
                        }
                    }
                }
                "provider" => handle_provider_slash(arg, config, &mut agent).await,
                "settings" => handle_settings_slash(arg, config),
                "theme" => handle_theme_slash(arg, config),
                "mouse" => println!(
                    "mouse capture toggling only applies to the interactive TUI — this plain REPL session never enables it"
                ),
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
                    if let Err(e) = agent
                        .suggest_turn(arg, print_agent_event, approver(yes))
                        .await
                    {
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
                    //
                    // Only the *first word* of `arg` decides which case this
                    // is (a bare "list"/"output"/"stop", vs. an actual goal);
                    // the goal itself must come from the full trimmed `arg`,
                    // not that first word alone — `splitn(2, ...).next()`
                    // used to be used for both, which silently spawned an
                    // orchestration for just "build" out of "build a login
                    // page", discarding the rest of the goal.
                    let full_arg = arg.trim();
                    let mut parts = full_arg.splitn(2, char::is_whitespace);
                    let first_word = parts.next().unwrap_or("");
                    if full_arg.is_empty() {
                        eprintln!("usage: /bg <goal> — run an orchestrated plan in the background");
                        eprintln!("  /bg list | output <id> | stop <id>  manage background tasks");
                        eprintln!("  append `@@workflow:<name>` to run a named workflow");
                    } else if matches!(first_word, "list" | "output" | "stop") {
                        eprintln!("  for background tasks use the `zeus bg ...` subcommand:");
                        eprintln!("  zeus bg list · zeus bg output <id> · zeus bg stop <id>");
                    } else {
                        let (goal, workflow) = match full_arg.rsplit_once("@@workflow:") {
                            Some((g, name)) => (g.trim(), Some(name.trim())),
                            None => (full_arg, None),
                        };
                        match spawn_bg_orchestrate(config, goal, workflow, None) {
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

        let message = match upload_message {
            Some(m) => m,
            None => expand_slash_command(config, line.to_string()),
        };

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
                if let Some(hint) = credit_failure_hint(config, &format!("{e:#}")) {
                    eprintln!("{}", ui::styled(ui::warn_style(), &hint));
                }
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

/// `zeus upload <paths...>`: copy files/directories (from anywhere on disk)
/// into `<project>/.agent/uploads/` and print their project-relative paths.
/// The agent can only read inside the project root (path containment), so
/// this is the "give zeus a file that lives elsewhere" ergonomic.
fn cmd_upload(
    config: &Config,
    paths: Vec<PathBuf>,
    to: Option<String>,
    dry_run: bool,
) -> Result<()> {
    let raw: Vec<String> = paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let report = upload_files(config, &raw, to.as_deref(), dry_run)?;
    let verb = if dry_run { "would stage" } else { "uploaded" };
    for rel in &report.uploaded {
        println!("{verb}: {rel}");
    }
    for w in &report.warnings {
        println!("warning: {w}");
    }
    if report.uploaded.is_empty() {
        println!("nothing to upload");
        return Ok(());
    }
    if dry_run {
        println!("(dry run — nothing was copied; drop --dry-run to stage these)");
    } else {
        println!(
            "staged under .agent/uploads — the agent reads these like any project file \
             (read for text, read_image for images, read_document for PDF/office docs)."
        );
    }
    Ok(())
}

/// Result of a successful upload: the staged project-relative paths plus any
/// non-fatal notes (e.g. "this looks binary, so the agent can't read it as
/// text"). Warnings are printed to the user but don't stop the upload.
#[derive(Debug, PartialEq)]
pub(crate) struct UploadReport {
    pub uploaded: Vec<String>,
    pub warnings: Vec<String>,
}

/// Per-upload safety cap — a single top-level path (file or whole tree) may
/// stage at most this many bytes, so a stray multi-GB directory can't be
/// pulled into the project by accident.
const MAX_UPLOAD_BYTES: u64 = 512 * 1024 * 1024;

/// Copy one or more paths (files or directories) from anywhere on disk into
/// `<project>/.agent/uploads/`, returning each item's project-relative path
/// (forward slashes, so the agent and the user see the same spelling). All
/// inputs are validated *before* any copy, so a typo'd path doesn't leave
/// earlier uploads half-staged. Symlinks are never followed, and nothing
/// over `MAX_UPLOAD_BYTES` is staged. A plain local copy — nothing leaves
/// the machine. Used by both `zeus upload` and the `/upload` slash command.
/// With `dry_run` the destinations are computed and returned but nothing is
/// copied or created.
pub(crate) fn upload_files(
    config: &Config,
    paths: &[String],
    to: Option<&str>,
    dry_run: bool,
) -> Result<UploadReport> {
    let root = config.project_root.clone().ok_or_else(|| {
        anyhow::anyhow!("no project root — run zeus inside a project (or pass --project-root)")
    })?;
    let mut dest_dir = root.join(".agent").join("uploads");
    if let Some(sub) = to {
        // Keep the subdir a single plain name — never a path — so a value
        // like `..` or `a/b` can't point the upload somewhere surprising.
        if sub.is_empty() || sub == "." || sub == ".." || sub.contains(['/', '\\']) {
            bail!("--to must be a plain subdirectory name (no slashes): got '{sub}'");
        }
        dest_dir = dest_dir.join(sub);
    }
    if !dry_run {
        std::fs::create_dir_all(&dest_dir)
            .with_context(|| format!("create upload directory {}", dest_dir.display()))?;
    }

    // Validate every path before copying any of them.
    let mut sources: Vec<PathBuf> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    for raw in paths {
        let p = raw.trim().trim_matches(['"', '\'']);
        if p.is_empty() {
            continue;
        }
        let src = PathBuf::from(p);
        let src = if src.is_absolute() {
            src
        } else {
            std::env::current_dir()
                .context("read current directory")?
                .join(src)
        };
        if !src.exists() {
            errors.push(format!("'{p}' does not exist"));
            continue;
        }
        sources.push(src);
    }
    if !errors.is_empty() {
        bail!("upload aborted — {}", errors.join("; "));
    }

    // Pre-flight the whole set (symlinks + sizes) so a bad pick aborts
    // before a single byte is copied.
    let mut symlinks: Vec<String> = Vec::new();
    let mut oversized: Vec<String> = Vec::new();
    for src in &sources {
        let size = walk_upload_tree(src, &mut symlinks)?;
        if size > MAX_UPLOAD_BYTES {
            oversized.push(format!(
                "'{}' is {:.1} GiB — the upload cap is {:.0} MiB",
                src.display(),
                size as f64 / (1024.0 * 1024.0 * 1024.0),
                MAX_UPLOAD_BYTES as f64 / (1024.0 * 1024.0)
            ));
        }
    }
    if !symlinks.is_empty() {
        bail!(
            "upload aborted — symlinks are never followed: {}",
            symlinks.join("; ")
        );
    }
    if !oversized.is_empty() {
        bail!("upload aborted — {}", oversized.join("; "));
    }

    let mut uploaded = Vec::new();
    let mut warnings = Vec::new();
    for src in sources {
        let name = src
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .ok_or_else(|| anyhow::anyhow!("can't derive a file name from '{}'", src.display()))?;
        // Idempotency: re-uploading a file whose staged copy is byte-identical
        // (or a directory tree that already matches) is a no-op — no `-1`
        // duplicate is created and nothing is clobbered. Only a *changed*
        // source gets deduped into `name-1`, `name-2`, …
        let first_dest = dest_dir.join(&name);
        if upload_is_unchanged(&src, &first_dest) {
            let rel = first_dest
                .strip_prefix(&root)
                .map(|r| r.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|_| first_dest.display().to_string());
            uploaded.push(rel);
            continue;
        }
        let dest = unique_dest(&dest_dir, &name);
        if !dry_run {
            if src.is_dir() {
                copy_dir_recursive(&src, &dest)
                    .with_context(|| format!("upload directory {}", src.display()))?;
            } else {
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("create {}", parent.display()))?;
                }
                std::fs::copy(&src, &dest).with_context(|| format!("upload {}", src.display()))?;
            }
        }
        let rel = dest
            .strip_prefix(&root)
            .map(|r| r.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| dest.display().to_string());
        uploaded.push(rel);
        if !dry_run && !src.is_dir() {
            if let Some(w) = upload_warning_for(&dest) {
                warnings.push(w);
            }
        }
    }
    Ok(UploadReport { uploaded, warnings })
}

/// Split a `/upload` command line into `(to, paths, error)`. Handles a
/// `--to <subdir>` flag anywhere in the line and double-quoted paths that
/// contain spaces — `/upload --to designs "/my files/a b.png" c.png`.
/// The returned paths are de-quoted; `upload_files` re-trims defensively.
pub(crate) fn parse_upload_args(arg: &str) -> (Option<String>, Vec<String>, Option<String>) {
    let mut to = None;
    let mut paths = Vec::new();
    let mut error = None;
    let cs: Vec<char> = arg.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        while i < cs.len() && cs[i].is_whitespace() {
            i += 1;
        }
        if i >= cs.len() {
            break;
        }
        let mut tok = String::new();
        if cs[i] == '"' {
            i += 1;
            let mut closed = false;
            while i < cs.len() {
                if cs[i] == '"' {
                    closed = true;
                    i += 1;
                    break;
                }
                tok.push(cs[i]);
                i += 1;
            }
            if !closed {
                error = Some("unterminated quote in path".to_string());
            }
        } else {
            while i < cs.len() && !cs[i].is_whitespace() {
                tok.push(cs[i]);
                i += 1;
            }
        }
        if tok == "--to" {
            while i < cs.len() && cs[i].is_whitespace() {
                i += 1;
            }
            let mut sub = String::new();
            if i < cs.len() && cs[i] == '"' {
                i += 1;
                while i < cs.len() && cs[i] != '"' {
                    sub.push(cs[i]);
                    i += 1;
                }
                if i < cs.len() {
                    i += 1;
                }
            } else {
                while i < cs.len() && !cs[i].is_whitespace() {
                    sub.push(cs[i]);
                    i += 1;
                }
            }
            if sub.is_empty() {
                error = Some("--to needs a value".to_string());
            } else {
                to = Some(sub);
            }
        } else if !tok.is_empty() && error.is_none() {
            paths.push(tok);
        }
        if error.is_some() {
            break;
        }
    }
    (to, paths, error)
}

/// One staged upload, project-relative with forward slashes (dirs get a
/// trailing `/` and a `size` of 0).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct UploadEntry {
    pub rel: String,
    pub size: u64,
}

/// The `<project>/.agent/uploads` directory, erroring when there's no
/// project root.
pub(crate) fn uploads_dir(config: &Config) -> Result<std::path::PathBuf> {
    let root = config
        .project_root
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no project root — run zeus inside a project"))?;
    Ok(root.join(".agent").join("uploads"))
}

/// List everything currently staged under `<project>/.agent/uploads/`.
/// Empty when nothing is staged yet. Backs the `/uploads` slash command.
pub(crate) fn list_uploads(config: &Config) -> Result<Vec<UploadEntry>> {
    let dir = uploads_dir(config)?;
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let root = config.project_root.clone().unwrap();
    let mut out = Vec::new();
    let mut stack = vec![dir];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).with_context(|| format!("read {}", d.display()))? {
            let entry = entry.context("read directory entry")?;
            let p = entry.path();
            let rel = p
                .strip_prefix(&root)
                .map(|r| r.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|_| p.display().to_string());
            if entry.file_type().context("stat directory entry")?.is_dir() {
                out.push(UploadEntry {
                    rel: format!("{rel}/"),
                    size: 0,
                });
                stack.push(p);
            } else {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                out.push(UploadEntry { rel, size });
            }
        }
    }
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok(out)
}

/// Delete a staged upload (file, or directory tree). The `rel` path is
/// validated to stay inside `.agent/uploads/` so it can't escape. Returns
/// the number of filesystem items removed.
pub(crate) fn delete_upload(config: &Config, rel: &str) -> Result<u64> {
    let dir = uploads_dir(config)?;
    let rel = rel.trim().trim_matches(['"', '\'']).trim_start_matches('/');
    // Accept both the listing's project-relative form (`.agent/uploads/x`)
    // and a path already relative to the uploads dir (`x`).
    let rel = rel
        .strip_prefix(".agent/uploads/")
        .or_else(|| rel.strip_prefix(".agent/uploads"))
        .unwrap_or(rel);
    if rel.is_empty() || rel == "." || rel == ".." {
        bail!("usage: /uploads rm <rel-path> — pick one from the /uploads listing");
    }
    let dest = dir.join(rel);
    let within = dest
        .strip_prefix(&dir)
        .ok()
        .and_then(|p| p.components().next())
        .map(|c| {
            !matches!(
                c,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        })
        .unwrap_or(false);
    if !within {
        bail!("refusing to delete outside .agent/uploads: {rel}");
    }
    if !dest.exists() {
        bail!("no such upload: {rel} (run /uploads to see the current listing)");
    }
    let meta =
        std::fs::symlink_metadata(&dest).with_context(|| format!("stat {}", dest.display()))?;
    let n = if meta.is_dir() {
        let count = std::fs::read_dir(&dest)
            .map(|rd| rd.flatten().count())
            .unwrap_or(0)
            + 1;
        std::fs::remove_dir_all(&dest).with_context(|| format!("remove {}", dest.display()))?;
        count as u64
    } else {
        std::fs::remove_file(&dest).with_context(|| format!("remove {}", dest.display()))?;
        1
    };
    Ok(n)
}

/// Human-readable size, e.g. `14 B`, `12.3 KiB`, `4.5 MiB`.
pub(crate) fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

/// Walk a source (file or whole tree), returning its total size in bytes and
/// recording any symlinks found. Symlinks are never followed — they're
/// reported so the caller can abort before copying anything.
fn walk_upload_tree(src: &Path, symlinks: &mut Vec<String>) -> Result<u64> {
    let meta = std::fs::symlink_metadata(src).with_context(|| format!("stat {}", src.display()))?;
    if meta.file_type().is_symlink() {
        symlinks.push(src.display().to_string());
        return Ok(0);
    }
    if meta.is_file() {
        return Ok(meta.len());
    }
    let mut total = 0u64;
    for entry in std::fs::read_dir(src).with_context(|| format!("read {}", src.display()))? {
        let entry = entry.context("read directory entry")?;
        total += walk_upload_tree(&entry.path(), symlinks)?;
    }
    Ok(total)
}

/// A non-fatal note for one staged file: if it's a binary blob that isn't a
/// known image or office-document format, the agent can't `read` it as text —
/// say which tool it should use instead.
fn upload_warning_for(dest: &Path) -> Option<String> {
    let binary = match std::fs::File::open(dest) {
        Ok(mut f) => {
            let mut buf = [0u8; 8192];
            let n = std::io::Read::read(&mut f, &mut buf).unwrap_or(0);
            buf[..n].contains(&0)
        }
        Err(_) => return None,
    };
    if !binary {
        return None;
    }
    let ext = dest
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    let handled_by_read_image = matches!(
        ext.as_str(),
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "avif" | "heic" | "ico"
    );
    let handled_by_read_document = matches!(
        ext.as_str(),
        "pdf" | "docx" | "doc" | "xlsx" | "xls" | "pptx" | "ppt" | "odt" | "ods" | "odp"
    );
    if handled_by_read_image || handled_by_read_document {
        return None;
    }
    Some(format!(
        "'{}' looks like a binary file — the agent can't 'read' it as text; use read_image (images) or read_document (PDF/office docs).",
        dest.display()
    ))
}

/// Pick a non-colliding destination name, appending `-1`, `-2`, … before the
/// extension when `name` is already taken (an upload never overwrites).
fn unique_dest(dir: &Path, name: &str) -> PathBuf {
    let candidate = dir.join(name);
    if !candidate.exists() {
        return candidate;
    }
    let (stem, ext) = match Path::new(name).extension().and_then(|e| e.to_str()) {
        Some(ext) => {
            let stem = Path::new(name)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| name.to_string());
            (stem, format!(".{ext}"))
        }
        None => (name.to_string(), String::new()),
    };
    for i in 1.. {
        let candidate = dir.join(format!("{stem}-{i}{ext}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("loop always returns or the name space is exhausted")
}

/// True when the staged copy at `dest` already matches `src` — byte-identical
/// for files, structurally identical (same relative paths, dirs, and file
/// sizes) for directories. The basis of `/upload` idempotency.
fn upload_is_unchanged(src: &Path, dest: &Path) -> bool {
    if !dest.exists() {
        return false;
    }
    if src.is_dir() {
        if !dest.is_dir() {
            return false;
        }
        let mut a = Vec::new();
        let mut b = Vec::new();
        collect_upload_tree(src, std::path::Path::new(""), &mut a);
        collect_upload_tree(dest, std::path::Path::new(""), &mut b);
        a.sort_by(|x, y| x.0.cmp(&y.0));
        b.sort_by(|x, y| x.0.cmp(&y.0));
        a == b
    } else {
        match (std::fs::metadata(src), std::fs::metadata(dest)) {
            (Ok(a), Ok(b)) if a.len() == b.len() => {
                matches!((std::fs::read(src), std::fs::read(dest)), (Ok(x), Ok(y)) if x == y)
            }
            _ => false,
        }
    }
}

/// Recursively collect `(relative path, is_dir, size)` for every entry under
/// `dir`. Used to compare two directory trees without reading file bytes.
fn collect_upload_tree(dir: &Path, rel: &Path, out: &mut Vec<(PathBuf, bool, u64)>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            let child_rel = rel.join(e.file_name());
            if p.is_dir() {
                out.push((child_rel.clone(), true, 0));
                collect_upload_tree(&p, &child_rel, out);
            } else {
                let len = p.metadata().map(|m| m.len()).unwrap_or(0);
                out.push((child_rel, false, len));
            }
        }
    }
}

/// Recursively copy a directory tree. Symlinks are never followed — the
/// pre-flight in `upload_files` already rejects them, so hitting one here is
/// a race and is treated as fatal rather than silently copied.
fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest).with_context(|| format!("create {}", dest.display()))?;
    for entry in std::fs::read_dir(src).with_context(|| format!("read {}", src.display()))? {
        let entry = entry.context("read directory entry")?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        let ft = entry.file_type().context("stat directory entry")?;
        if ft.is_symlink() {
            bail!("symlinks are never followed: {}", from.display());
        }
        if ft.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to).with_context(|| format!("copy {}", from.display()))?;
        }
    }
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
    let roots = if roots.is_empty() {
        vec![ws.project_root.clone()]
    } else {
        roots
    };
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
            guard_against_home_root(&root)?;
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

/// Refuse to build a full index over the user's home directory. `find_project_root`
/// falls back to (or walks up to) the home dir when a plain directory under it has
/// no `.git`/`.agent` marker, which turns `index` into a walk of the entire home
/// tree — effectively a hang. Fail loudly with a fix hint instead.
fn guard_against_home_root(root: &Path) -> Result<()> {
    if let Some(home) = dirs::home_dir() {
        if root == home {
            return Err(anyhow::anyhow!(
                "refusing to index your home directory ({}): run this from a project directory \
                 instead (or add a .git/.agent marker there to pin the project root)",
                home.display()
            ));
        }
    }
    Ok(())
}

async fn cmd_ragindex(config: &Config, action: RagindexCmd) -> Result<()> {
    let ws = workspace(config)?;
    let root = ws.project_root.clone();

    match action {
        RagindexCmd::Index {
            force,
            embed,
            provider,
            model,
        } => {
            guard_against_home_root(&root)?;
            // Fast path: a fresh index that already satisfies the request.
            let mut persisted = zeus_rag::PersistedRagIndex::load(&root);
            if !force {
                if let Some(p) = persisted.as_ref() {
                    if p.is_fresh() && (!embed || p.has_vectors()) {
                        println!(
                            "index already fresh: {} chunk(s) in {} file(s) — use --force to rebuild",
                            p.documents.len(),
                            p.stamps.len()
                        );
                        return Ok(());
                    }
                }
            }

            // Stale index -> incremental refresh; force or no index -> full walk.
            let mut index = if let Some(mut p) = persisted.take() {
                if !force {
                    p.refresh(800, 80);
                }
                p.into_index()
            } else {
                zeus_rag::RagIndex::from_project(&root, 800, 80)
            };
            if index.is_empty() {
                println!("no source files to index");
                return Ok(());
            }

            if embed {
                match resolve_provider(config, provider).await {
                    Ok(provider) => {
                        let requested = model.unwrap_or_else(|| config.settings.model.model.clone());
                        let (model, _) = resolve_model(&*provider, requested).await;
                        let embedded = index
                            .embed_all(&*provider, &model, 32)
                            .await
                            .unwrap_or_else(|e| {
                                eprintln!("warning: embedding failed ({e}); index kept keyword-only");
                                0
                            });
                        if embedded > 0 {
                            println!("embedded {embedded} chunk(s)");
                        } else {
                            println!("no embeddings produced; index kept keyword-only");
                        }
                    }
                    Err(e) => eprintln!(
                        "warning: {e}; index kept keyword-only — start a local server or configure a cloud provider to enable embeddings"
                    ),
                }
            }

            let persisted = zeus_rag::PersistedRagIndex::from_index(&index);
            persisted.save(&root).context("save RAG index")?;
            println!(
                "indexed {} chunk(s) in {} file(s) -> {}",
                index.len(),
                persisted.stamps.len(),
                zeus_rag::PersistedRagIndex::file_path(&root).display()
            );
        }
        RagindexCmd::Search { query, k } => {
            let index = match zeus_rag::PersistedRagIndex::load(&root) {
                Some(p) if p.is_fresh() => p.into_index(),
                _ => {
                    println!(
                        "no fresh index at {}; run `zeus ragindex index` first",
                        zeus_rag::PersistedRagIndex::file_path(&root).display()
                    );
                    return Ok(());
                }
            };
            if index.is_empty() {
                println!("index is empty; run `zeus ragindex index` first");
                return Ok(());
            }
            let hits = index.search(&query, k.clamp(1, 20));
            if hits.is_empty() {
                println!("no chunks matched '{query}'");
                return Ok(());
            }
            println!("top {} match(es) for '{query}':", hits.len());
            for h in &hits {
                let path = h
                    .chunk
                    .path
                    .strip_prefix(&root)
                    .unwrap_or(&h.chunk.path)
                    .display();
                println!("[{:.0}%] {}:", h.score * 100.0, path);
                for line in h.chunk.text.lines() {
                    println!("  {line}");
                }
                println!();
            }
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

/// Where `zeus project scaffold` places a new project: an explicit
/// `--project-root` wins; otherwise the current working directory — never the
/// walked-up project root, so scaffolding from inside an existing repo doesn't
/// accidentally dump the new project into the ancestor repo.
fn scaffold_base(explicit_root: Option<&Path>, cwd: &Path) -> PathBuf {
    explicit_root
        .map(Path::to_path_buf)
        .unwrap_or_else(|| cwd.to_path_buf())
}

fn cmd_project(config: &Config, action: ProjectCmd, explicit_root: Option<&Path>) -> Result<()> {
    let ws = workspace(config)?;
    let root = ws.project_root.clone();
    match action {
        ProjectCmd::Detect => {
            let fw = zeus_lang::Framework::detect_framework(&root);
            match zeus_lang::detect_project(&root) {
                Some(lang) => {
                    println!("language: {}", zeus_lang::spec(lang).display_name);
                    if let Some(fw) = fw {
                        println!(
                            "framework: {} (base {})",
                            zeus_lang::framework_spec(fw).display_name,
                            zeus_lang::spec(zeus_lang::framework_spec(fw).base).display_name
                        );
                    }
                    Ok(())
                }
                None => {
                    if let Some(fw) = fw {
                        println!("framework: {}", zeus_lang::framework_spec(fw).display_name);
                        return Ok(());
                    }
                    bail!(
                        "could not detect a supported language or framework for {} — run `zeus project commands --help` to list them",
                        root.display()
                    )
                }
            }
        }
        ProjectCmd::Commands { lang } => {
            let lang = match lang {
                Some(name) => {
                    if let Some(lang) = zeus_lang::Language::from_name(&name) {
                        lang
                    } else if let Some(fw) = zeus_lang::Framework::from_name(&name) {
                        zeus_lang::framework_spec(fw).base
                    } else {
                        bail!("unknown language or framework '{name}' — try a name like rust, ts, go, c#, react, django")
                    }
                }
                None => zeus_lang::detect_project(&root).ok_or_else(|| {
                    anyhow::anyhow!(
                        "cannot detect language for {}; pass a language name",
                        root.display()
                    )
                })?,
            };
            print_lang_commands(lang, &root);
            Ok(())
        }
        ProjectCmd::Scaffold { list, lang, name } => {
            if list {
                print_scaffold_choices();
                return Ok(());
            }
            let (Some(lang), Some(name)) = (lang, name) else {
                bail!(
                    "scaffold needs a language and a project name — try `zeus project scaffold --list` to see choices"
                );
            };
            let lang = lang.as_str();
            let name = name.as_str();
            // A scaffold creates a *new* project, so it lands in the current
            // working directory — not the walked-up project root (an ancestor
            // repo's root would otherwise swallow the new project). Only an
            // explicit `--project-root` redirects it elsewhere.
            let base = scaffold_base(
                explicit_root,
                &std::env::current_dir().unwrap_or_else(|_| root.clone()),
            );
            let target = base.join(name);
            if target.exists() {
                bail!(
                    "{} already exists — pick a different name",
                    target.display()
                );
            }
            let written = if let Some(fw) = zeus_lang::Framework::from_name(lang) {
                zeus_lang::scaffold_framework(fw, name, &target)
                    .context("scaffold")?
                    .iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
            } else {
                let lang = zeus_lang::Language::from_name(lang).ok_or_else(|| {
                    anyhow::anyhow!(
                        "unknown language or framework '{lang}' — try `zeus project scaffold --list` for choices"
                    )
                })?;
                zeus_lang::scaffold_project(lang, name, &target)
                    .context("scaffold")?
                    .iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
            };
            println!(
                "scaffolded {lang} project '{name}' into {}:",
                target.display()
            );
            for p in &written {
                println!("  created {p}");
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

/// Print every scaffoldable language and framework with the exact name to
/// pass to `zeus project scaffold`.
fn print_scaffold_choices() {
    println!("languages:");
    for lang in zeus_lang::available_scaffold_languages() {
        let key = format!("{lang:?}").to_ascii_lowercase();
        println!("  {key:<14} {}", zeus_lang::spec(lang).display_name);
    }
    println!("frameworks:");
    for fw in zeus_lang::Framework::ALL {
        let key = format!("{fw:?}").to_ascii_lowercase();
        let s = zeus_lang::framework_spec(*fw);
        println!(
            "  {key:<14} {} (base {})",
            s.display_name,
            zeus_lang::spec(s.base).display_name
        );
    }
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

/// Build the argv for a detached headless orchestration (`zeus agent <goal>
/// --yes --auto [--workflow <name>] [--model <model>]`). The args go through
/// `spawn_argv` (no shell wrapper) so the goal text survives untouched.
fn bg_orchestrate_argv(goal: &str, workflow: Option<&str>, model: Option<&str>) -> Vec<String> {
    let mut args: Vec<String> = vec!["agent".into(), goal.into(), "--yes".into(), "--auto".into()];
    if let Some(name) = workflow {
        args.push("--workflow".into());
        args.push(name.to_string());
    }
    if let Some(m) = model {
        args.push("--model".into());
        args.push(m.to_string());
    }
    args
}

/// Spawn a goal as a detached headless orchestration (the `bg orchestrate`
/// wrapper) so the goal text survives untouched — nothing is quoted or
/// expanded. Shared by `zeus bg orchestrate` and the REPL `/bg` command.
pub(crate) fn spawn_bg_orchestrate(
    config: &Config,
    goal: &str,
    workflow: Option<&str>,
    model: Option<&str>,
) -> anyhow::Result<u64> {
    let ws = workspace(config)?;
    let registry = BackgroundTaskRegistry::new(ws.project_root.join(".agent/background"));
    let exe = std::env::current_exe().context("resolve own executable")?;
    let args = bg_orchestrate_argv(goal, workflow, model);
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
        BgCmd::Orchestrate {
            goal,
            workflow,
            model,
        } => {
            let id = spawn_bg_orchestrate(config, &goal, workflow.as_deref(), model.as_deref())?;
            println!("started background orchestration id={id}");
            let goal_flat = goal.replace('\n', " ");
            println!("  goal:     {goal_flat}");
            println!(
                "  workflow: {}",
                workflow.as_deref().unwrap_or("(auto-plan)")
            );
            if let Some(m) = &model {
                println!("  model:    {m}");
            }
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
        BgCmd::Logs { id } => {
            registry.follow(id).context("follow background task")?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_chat_subcommand() {
        let cli = Cli::try_parse_from(["zeus", "chat", "hello world", "--model", "m1"]).unwrap();
        match cli.command.unwrap() {
            Commands::Chat {
                message,
                model,
                no_stream,
                ..
            } => {
                assert_eq!(message, "hello world");
                assert_eq!(model.as_deref(), Some("m1"));
                assert!(!no_stream);
            }
            other => panic!("expected Chat, got {other:?}"),
        }
    }

    #[test]
    fn cli_global_flags_are_recognized() {
        let cli =
            Cli::try_parse_from(["zeus", "--yes", "--project-root", "/tmp/x", "init"]).unwrap();
        assert!(cli.yes);
        assert_eq!(
            cli.project_root.as_deref(),
            Some(std::path::Path::new("/tmp/x"))
        );
        assert!(matches!(cli.command, Some(Commands::Init { .. })));
    }

    #[test]
    fn cli_rejects_unknown_subcommand() {
        assert!(Cli::try_parse_from(["zeus", "frobnicate"]).is_err());
    }

    #[test]
    fn expand_format_args_substitutes_file_placeholder() {
        let target = std::path::Path::new("/repo/src/main.rs");
        let out = expand_format_args(&["cargo", "fmt", "--", zeus_lang::FILE_PLACEHOLDER], target);
        assert_eq!(out, vec!["cargo", "fmt", "--", "/repo/src/main.rs"]);
    }

    #[test]
    fn cli_parses_git_subcommand() {
        let cli = Cli::try_parse_from(["zeus", "git", "status"]).unwrap();
        assert!(matches!(cli.command, Some(Commands::Git { .. })));
    }

    #[test]
    fn cli_parses_read_flags() {
        let cli =
            Cli::try_parse_from(["zeus", "read", "file.txt", "--offset", "10", "--limit", "5"])
                .unwrap();
        match cli.command.unwrap() {
            Commands::Read {
                path,
                offset,
                limit,
            } => {
                assert_eq!(path, PathBuf::from("file.txt"));
                assert_eq!(offset, Some(10));
                assert_eq!(limit, Some(5));
            }
            other => panic!("expected Read, got {other:?}"),
        }
    }

    #[test]
    fn cli_parses_ragindex_subcommands() {
        let cli = Cli::try_parse_from([
            "zeus",
            "ragindex",
            "index",
            "--force",
            "--embed",
            "--provider",
            "ollama",
            "--model",
            "all-minilm",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::Ragindex {
                action: RagindexCmd::Index { force, embed, .. },
            } => {
                assert!(force);
                assert!(embed);
            }
            other => panic!("expected ragindex index, got {other:?}"),
        }

        let cli = Cli::try_parse_from(["zeus", "ragindex", "search", "retry", "-k", "3"]).unwrap();
        match cli.command.unwrap() {
            Commands::Ragindex {
                action: RagindexCmd::Search { query, k },
            } => {
                assert_eq!(query, "retry");
                assert_eq!(k, 3);
            }
            other => panic!("expected ragindex search, got {other:?}"),
        }
    }

    #[test]
    fn guard_refuses_home_root_but_allows_projects() {
        if let Some(home) = dirs::home_dir() {
            assert!(guard_against_home_root(&home).is_err());
        }
        let proj = std::path::Path::new("C:\\some\\project");
        assert!(guard_against_home_root(proj).is_ok());
    }

    #[test]
    fn project_survey_injects_detected_stack_and_command_lines() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path().join("webapp");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("package.json"),
            r#"{ "name": "webapp", "version": "0.1.0", "scripts": { "build": "tsc" } }"#,
        )
        .unwrap();
        std::fs::write(root.join("src/main.ts"), "export const hello = () => 1;\n").unwrap();

        let survey = build_project_survey(Some(&root)).unwrap();
        assert!(survey.contains("workspace name: webapp"), "{survey}");
        assert!(survey.contains("detected language: TypeScript"), "{survey}");
        // The injected commands must be the ones `verify`/`test` would run,
        // so the agent trusts them instead of guessing.
        assert!(survey.contains("build:  tsc -p ."), "{survey}");
        assert!(survey.contains("test:   npm test"), "{survey}");
        assert!(survey.contains("entries scanned"), "{survey}");

        assert!(build_project_survey(None).is_none());
    }

    /// A scaffold lands in the current working directory when no explicit
    /// `--project-root` was given — never in a walked-up ancestor repo root.
    #[test]
    fn scaffold_base_prefers_cwd_over_walked_up_root() {
        let cwd = std::path::Path::new("work/src/lib");
        let repo_root = std::path::Path::new("work");
        assert_eq!(
            scaffold_base(None, cwd),
            cwd.to_path_buf(),
            "no explicit root -> cwd"
        );
        assert_eq!(
            scaffold_base(Some(repo_root), cwd),
            repo_root.to_path_buf(),
            "explicit --project-root wins over cwd"
        );
    }

    /// `zeus bg orchestrate` accepts `--model` and it reaches the detached
    /// `agent` argv verbatim (so a bare OpenRouter ID like
    /// `deepseek/deepseek-chat` passes through untouched).
    #[test]
    fn bg_orchestrate_argv_includes_model_flag() {
        let args =
            bg_orchestrate_argv("write readme", Some("ship"), Some("deepseek/deepseek-chat"));
        assert_eq!(args[0], "agent");
        assert!(args.windows(2).any(|w| w == ["--workflow", "ship"]));
        assert!(args
            .windows(2)
            .any(|w| w == ["--model", "deepseek/deepseek-chat"]));
        assert!(!args.windows(2).any(|w| w == ["--model"] && w[1].is_empty()));

        let plain = bg_orchestrate_argv("write readme", None, None);
        assert!(!plain.windows(2).any(|w| w[0] == "--model"));
    }

    #[test]
    fn cli_parses_bg_orchestrate_model_flag() {
        let cli = Cli::try_parse_from([
            "zeus",
            "bg",
            "orchestrate",
            "write readme",
            "--model",
            "deepseek/deepseek-chat",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::Bg {
                action: BgCmd::Orchestrate { model, .. },
            } => assert_eq!(model.as_deref(), Some("deepseek/deepseek-chat")),
            other => panic!("expected bg orchestrate, got {other:?}"),
        }
    }

    #[test]
    fn cli_parses_upload_subcommand() {
        let cli = Cli::try_parse_from([
            "zeus",
            "upload",
            "C:/Users/me/Desktop/shot.png",
            "C:/Users/me/Documents/report.pdf",
            "--to",
            "designs",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Commands::Upload { paths, to, dry_run } => {
                assert_eq!(paths.len(), 2);
                assert_eq!(to.as_deref(), Some("designs"));
                assert!(!dry_run);
            }
            other => panic!("expected Upload, got {other:?}"),
        }
    }

    #[test]
    fn upload_files_copies_into_project_and_dedupes() {
        use tempfile::TempDir;
        use zeus_config::{AgentSettings, GlobalPaths, ProvidersFile};

        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();

        // Source files live OUTSIDE the project root — that's the whole point
        // of upload: the agent can only read inside the project, so upload
        // stages the file there for it.
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("shot.png"), b"png-bytes").unwrap();
        std::fs::write(outside.join("report.pdf"), b"pdf-bytes").unwrap();
        std::fs::create_dir_all(outside.join("assets")).unwrap();
        std::fs::write(outside.join("assets/logo.svg"), b"<svg/>").unwrap();

        let config = Config {
            global: GlobalPaths::from_root(root.join(".zeus-home")),
            project: None,
            settings: AgentSettings::default(),
            providers: ProvidersFile::default(),
            project_root: Some(root.clone()),
        };

        let report = upload_files(
            &config,
            &[
                outside.join("shot.png").to_string_lossy().into_owned(),
                outside.join("report.pdf").to_string_lossy().into_owned(),
                outside.join("assets").to_string_lossy().into_owned(),
            ],
            None,
            false,
        )
        .unwrap();

        assert_eq!(
            report.uploaded,
            vec![
                ".agent/uploads/shot.png".to_string(),
                ".agent/uploads/report.pdf".to_string(),
                ".agent/uploads/assets".to_string(),
            ]
        );
        assert!(
            report.warnings.is_empty(),
            "text/svg/png uploads shouldn't warn: {:?}",
            report.warnings
        );
        assert_eq!(
            std::fs::read(root.join(".agent/uploads/shot.png")).unwrap(),
            b"png-bytes"
        );
        assert!(root.join(".agent/uploads/assets/logo.svg").exists());

        // Re-uploading a byte-identical file is a no-op (idempotent) — the
        // staged copy already matches, so no `-1` duplicate is created.
        let second = upload_files(
            &config,
            &[outside.join("shot.png").to_string_lossy().into_owned()],
            None,
            false,
        )
        .unwrap();
        assert_eq!(second.uploaded, vec![".agent/uploads/shot.png".to_string()]);

        // Only a *changed* source gets deduped — never clobbered.
        std::fs::write(outside.join("shot.png"), "png-bytes-v2").unwrap();
        let second = upload_files(
            &config,
            &[outside.join("shot.png").to_string_lossy().into_owned()],
            None,
            false,
        )
        .unwrap();
        assert_eq!(
            second.uploaded,
            vec![".agent/uploads/shot-1.png".to_string()]
        );

        // --to stages under a named subdirectory.
        let sub = upload_files(
            &config,
            &[outside.join("shot.png").to_string_lossy().into_owned()],
            Some("designs"),
            false,
        )
        .unwrap();
        assert_eq!(
            sub.uploaded,
            vec![".agent/uploads/designs/shot.png".to_string()]
        );

        // A missing path aborts cleanly (and before copying anything).
        let err = upload_files(
            &config,
            &["C:/definitely/missing.txt".to_string()],
            None,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("does not exist"));

        // --to refuses path separators so it can't escape .agent/uploads.
        let err = upload_files(&config, &[], Some("../evil"), false).unwrap_err();
        assert!(err.to_string().contains("plain subdirectory"));

        // /uploads lists what's staged (dirs with a trailing slash + sizes).
        let listed = list_uploads(&config).unwrap();
        let rels: Vec<&str> = listed.iter().map(|e| e.rel.as_str()).collect();
        assert!(rels.contains(&".agent/uploads/report.pdf"));
        assert!(rels.contains(&".agent/uploads/assets/"));
        assert!(rels.contains(&".agent/uploads/shot-1.png"));
        let pdf = listed
            .iter()
            .find(|e| e.rel == ".agent/uploads/report.pdf")
            .unwrap();
        assert_eq!(pdf.size, 9, "pdf-bytes is 9 bytes");
        assert!(listed
            .iter()
            .any(|e| e.rel == ".agent/uploads/shot.png" && e.size == 9));

        // Dry-run computes the same destinations without touching the disk.
        let dry = upload_files(
            &config,
            &[outside.join("shot.png").to_string_lossy().into_owned()],
            None,
            true,
        )
        .unwrap();
        assert_eq!(dry.uploaded, vec![".agent/uploads/shot-2.png".to_string()]);
        assert!(
            dry.warnings.is_empty(),
            "dry run must not sniff staged files"
        );
        assert!(
            !root.join(".agent/uploads/shot-2.png").exists(),
            "dry run copies nothing"
        );
    }

    #[test]
    fn uploads_rm_deletes_only_inside_uploads() {
        use tempfile::TempDir;
        use zeus_config::{AgentSettings, GlobalPaths, ProvidersFile};

        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("a.txt"), "aaa").unwrap();

        let config = Config {
            global: GlobalPaths::from_root(root.join(".zeus-home")),
            project: None,
            settings: AgentSettings::default(),
            providers: ProvidersFile::default(),
            project_root: Some(root.clone()),
        };

        upload_files(
            &config,
            &[outside.join("a.txt").to_string_lossy().into_owned()],
            None,
            false,
        )
        .unwrap();
        assert!(root.join(".agent/uploads/a.txt").exists());

        // An escape attempt is refused.
        let err = delete_upload(&config, "../evil").unwrap_err();
        assert!(err.to_string().contains("refusing"));

        // A directory tree is removed recursively.
        std::fs::create_dir_all(root.join(".agent/uploads/bundle")).unwrap();
        std::fs::write(root.join(".agent/uploads/bundle/x.txt"), "x").unwrap();
        let n = delete_upload(&config, ".agent/uploads/bundle").unwrap();
        assert_eq!(n, 2);
        assert!(!root.join(".agent/uploads/bundle").exists());

        let n = delete_upload(&config, ".agent/uploads/a.txt").unwrap();
        assert_eq!(n, 1);
        assert!(!root.join(".agent/uploads/a.txt").exists());

        // Unknown rel errors cleanly.
        assert!(delete_upload(&config, ".agent/uploads/nope.txt").is_err());
        assert!(delete_upload(&config, "").is_err());
    }

    #[test]
    fn human_size_formats_bytes() {
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(14), "14 B");
        assert_eq!(human_size(1024), "1.0 KiB");
        assert_eq!(human_size(12 * 1024 + 512), "12.5 KiB");
        assert_eq!(human_size(5 * 1024 * 1024), "5.0 MiB");
    }

    #[test]
    fn upload_files_stages_pdf_and_warns_on_unknown_binary() {
        use tempfile::TempDir;
        use zeus_config::{AgentSettings, GlobalPaths, ProvidersFile};

        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();

        // A plausible minimal PDF header (binary content, but read_document
        // handles it, so upload must NOT warn about it).
        std::fs::write(
            outside.join("report.pdf"),
            b"%PDF-1.4\n1 0 obj\n<< >>\nendobj\n%%EOF",
        )
        .unwrap();
        // Unknown binary — NUL bytes with no doc/image extension → warn.
        std::fs::write(outside.join("tool.bin"), [0u8; 64]).unwrap();
        // Known image — binary but read_image handles it → no warning.
        std::fs::write(outside.join("pic.png"), b"\x89PNG\r\n\x1a\nfake").unwrap();

        let config = Config {
            global: GlobalPaths::from_root(root.join(".zeus-home")),
            project: None,
            settings: AgentSettings::default(),
            providers: ProvidersFile::default(),
            project_root: Some(root.clone()),
        };

        let report = upload_files(
            &config,
            &[
                outside.join("report.pdf").to_string_lossy().into_owned(),
                outside.join("tool.bin").to_string_lossy().into_owned(),
                outside.join("pic.png").to_string_lossy().into_owned(),
            ],
            None,
            false,
        )
        .unwrap();

        assert_eq!(report.uploaded.len(), 3);
        // Byte-identical staging (the whole point of upload).
        assert_eq!(
            std::fs::read(root.join(".agent/uploads/report.pdf")).unwrap(),
            b"%PDF-1.4\n1 0 obj\n<< >>\nendobj\n%%EOF"
        );
        assert_eq!(
            std::fs::read(root.join(".agent/uploads/tool.bin")).unwrap(),
            [0u8; 64]
        );
        // Exactly one warning: the unknown binary. PDF and PNG are known.
        assert_eq!(report.warnings.len(), 1, "warnings: {:?}", report.warnings);
        assert!(report.warnings[0].contains("tool.bin"));
        assert!(report.warnings[0].contains("read_image"));
    }

    #[test]
    fn parse_upload_args_handles_quotes_and_to_flag() {
        let (to, paths, err) = parse_upload_args("--to designs \"my file a.png\" b.png");
        assert!(err.is_none());
        assert_eq!(to.as_deref(), Some("designs"));
        assert_eq!(paths, vec!["my file a.png", "b.png"]);

        // No --to, plain paths, quotes with spaces preserved.
        let (to, paths, err) = parse_upload_args("--to\"nope\" \"a b.txt\" c.txt");
        // The first token is `--to"nope"` — not the flag — so it's a path.
        assert!(err.is_none());
        assert_eq!(to, None);
        assert_eq!(paths.len(), 3);

        let (to, paths, err) = parse_upload_args("\"unterminated");
        assert!(err.is_some());
        assert!(paths.is_empty());
        assert_eq!(to, None);

        let (_, paths, err) = parse_upload_args("--to");
        assert!(err.is_some());
        assert!(paths.is_empty());
    }

    #[test]
    fn credit_failure_hint_suggests_provider_with_key() {
        use tempfile::TempDir;
        use zeus_config::{AgentSettings, GlobalPaths, ProvidersFile};

        let tmp = TempDir::new().unwrap();
        let global = GlobalPaths::from_root(tmp.path().join(".zeus-home"));
        std::fs::create_dir_all(global.keys_toml.parent().unwrap()).unwrap();
        // Stored key for gemini (the live-machine setup: openrouter keyed but
        // out of credits, gemini keyed and working).
        std::fs::write(&global.keys_toml, "[keys]\ngemini = \"gk-abc\"\n").unwrap();

        let config = Config {
            global: global.clone(),
            project: None,
            settings: AgentSettings::default(),
            providers: ProvidersFile {
                providers: std::collections::HashMap::from([
                    (
                        "openrouter".into(),
                        zeus_config::ProviderConfig {
                            kind: "openrouter".into(),
                            base_url: None,
                            api_key_env: None,
                            default_model: None,
                            headers: std::collections::HashMap::new(),
                            embeddings: false,
                            prompt_cache: false,
                        },
                    ),
                    (
                        "gemini".into(),
                        zeus_config::ProviderConfig {
                            kind: "gemini".into(),
                            base_url: None,
                            api_key_env: None,
                            default_model: None,
                            headers: std::collections::HashMap::new(),
                            embeddings: false,
                            prompt_cache: false,
                        },
                    ),
                ]),
            },
            project_root: None,
        };

        let hint = credit_failure_hint(&config, "HTTP 402 Payment Required — out of credits");
        let hint = hint.expect("a 402 must produce a credit hint");
        assert!(hint.contains("gemini"), "hint: {hint}");
        assert!(
            !hint.contains("openrouter"),
            "current provider excluded: {hint}"
        );

        // A non-credit failure (e.g. network) must NOT produce the hint.
        assert!(credit_failure_hint(&config, "connection reset by peer").is_none());
    }

    #[test]
    fn failover_relevant_detects_credits_and_rate_limits_only() {
        let wrap = |status: u16, msg: &str| -> anyhow::Error {
            anyhow::Error::new(zeus_provider::ProviderError::Http {
                status,
                message: msg.into(),
            })
        };
        assert!(failover_relevant(&wrap(402, "out of credits")));
        assert!(failover_relevant(&wrap(429, "Too Many Requests")));
        assert!(failover_relevant(&wrap(
            200,
            "you have exceeded your rate limit"
        )));
        assert!(!failover_relevant(&wrap(404, "model not found")));
        assert!(!failover_relevant(&wrap(401, "unauthorized")));
        assert!(!failover_relevant(&wrap(503, "service unavailable")));
        assert!(!failover_relevant(&anyhow::anyhow!(
            "connection reset by peer"
        )));
    }

    #[test]
    fn ensure_agent_gitignored_appends_rule() {
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join(".git")).unwrap();

        ensure_agent_gitignored(tmp.path()).unwrap();
        let ignore = std::fs::read_to_string(tmp.path().join(".gitignore")).unwrap();
        assert!(
            ignore.lines().any(|l| l.trim() == ".agent/"),
            "gitignore: {ignore}"
        );

        // Idempotent — a second pass must not duplicate the rule.
        ensure_agent_gitignored(tmp.path()).unwrap();
        let ignore = std::fs::read_to_string(tmp.path().join(".gitignore")).unwrap();
        let count = ignore.lines().filter(|l| l.trim() == ".agent/").count();
        assert_eq!(count, 1, "gitignore: {ignore}");

        // A non-repo dir gets no .gitignore at all.
        let bare = TempDir::new().unwrap();
        ensure_agent_gitignored(bare.path()).unwrap();
        assert!(!bare.path().join(".gitignore").exists());
    }

    #[test]
    fn upload_files_rejects_symlinks_when_creation_is_possible() {
        use tempfile::TempDir;
        use zeus_config::{AgentSettings, GlobalPaths, ProvidersFile};

        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        let target = outside.join("real.txt");
        std::fs::write(&target, "secret outside the project").unwrap();
        let link = outside.join("link.txt");

        #[cfg(unix)]
        let _ = std::os::unix::fs::symlink(&target, &link);
        #[cfg(windows)]
        let _ = std::os::windows::fs::symlink_file(&target, &link);

        // Symlink creation needs privilege/dev-mode on Windows; when it fails
        // we can't exercise the guard, so skip quietly.
        let is_link = std::fs::symlink_metadata(&link)
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false);
        if !is_link {
            return;
        }

        let config = Config {
            global: GlobalPaths::from_root(root.join(".zeus-home")),
            project: None,
            settings: AgentSettings::default(),
            providers: ProvidersFile::default(),
            project_root: Some(root.clone()),
        };

        let err =
            upload_files(&config, &[link.to_string_lossy().into_owned()], None, false).unwrap_err();
        assert!(
            err.to_string().contains("symlink"),
            "expected a symlink refusal, got: {err}"
        );
    }

    #[test]
    fn render_session_markdown_covers_all_roles() {
        use zeus_provider::{ImagePart, Role, ToolCall};
        let mut state = ConversationState::new("sess-1");
        state.messages = vec![
            Message::system("you are helpful"),
            Message::user("hello"),
            Message::assistant("hi there"),
            Message {
                role: Role::Assistant,
                content: String::new(),
                tool_call_id: None,
                tool_calls: vec![ToolCall {
                    id: "call-1".into(),
                    name: "read".into(),
                    arguments: "{\"path\": \"x\"}".into(),
                    extra_content: None,
                }],
                images: Vec::new(),
            },
            Message {
                role: Role::Tool,
                content: "file contents".into(),
                tool_call_id: Some("call-1".into()),
                tool_calls: Vec::new(),
                images: Vec::new(),
            },
            Message::user_with_images(
                "look at this",
                vec![ImagePart {
                    mime_type: "image/png".into(),
                    data_base64: "AAAA".into(),
                }],
            ),
        ];
        let md = render_session_markdown(&state);
        assert!(md.starts_with("# Session sess-1"));
        assert!(md.contains("> **system**"));
        assert!(md.contains("you are helpful"));
        assert!(md.contains("## User"));
        assert!(md.contains("hello"));
        assert!(md.contains("## Assistant"));
        assert!(md.contains("hi there"));
        assert!(md.contains("[call] read (call-1)"));
        assert!(md.contains("## Tool (call-1) result"));
        assert!(md.contains("file contents"));
        assert!(md.contains("1 image attachment(s) omitted"));
    }

    #[test]
    fn render_session_markdown_empty_state_has_header_only() {
        let state = ConversationState::new("sess-2");
        assert_eq!(render_session_markdown(&state), "# Session sess-2\n\n");
    }
}
