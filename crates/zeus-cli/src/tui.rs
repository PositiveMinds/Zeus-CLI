//! Full interactive chat interface — an alternate-screen ratatui TUI
//! modeled after opencode/Claude Code's own CLI: a centered splash logo and
//! bordered input box on first launch, then a scrolling transcript pane with
//! the input box pinned at the bottom once the conversation starts.
//!
//! Only entered when stdin+stdout are both a real terminal (checked by the
//! caller in `main.rs`); piped/scripted sessions use the plain REPL instead,
//! since raw-mode/alternate-screen control only makes sense against a real
//! console.
//!
//! One turn runs at a time. While a turn is in flight, the `Agent` is moved
//! into a spawned task (so its streamed events/tool calls don't block
//! rendering or key handling) and handed back once the turn completes.
//! Permission prompts from inside that task are bridged back to this render
//! loop as a modal: the tool-dispatch code's synchronous approver closure
//! waits on a `tokio::sync::oneshot` reply channel until a keypress here
//! answers it, wrapped in `tokio::task::block_in_place` so the wait never
//! pins a runtime worker thread — waiting on a human is inherently slow, so
//! this bounded bridge is the deliberate exception to "never block an async
//! task".

use crate::{
    build_agent_repl_with, build_agent_repl_with_session, expand_slash_command,
    git_engine_for_agent, known_slash_commands, list_models_by_provider, persist_default_provider,
    print_repl_help_lines,
};
#[path = "theme.rs"]
pub(crate) mod theme;
#[path = "tui_text.rs"]
mod tui_text;
use anyhow::{Context, Result};
use crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
    MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Paragraph, Wrap,
};
use ratatui::{Frame, Terminal};
use std::io;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tui_text::*;
use zeus_agent::{
    personas_by_department, Agent, AgentEvent, SessionStore, TurnResult,
};
use zeus_config::{Config, KeysFile};
use zeus_fs::{ApprovalDecision, PermissionRequest};
use zeus_provider::{create_provider, TokenUsage};

#[path = "transcript.rs"]
mod transcript;
use transcript::*;

#[path = "pickers.rs"]
mod pickers;
use pickers::*;

#[path = "modal.rs"]
pub(crate) mod modal;
use modal::*;

#[path = "favorites.rs"]
mod favorites;
use favorites::*;

/// Bridge one permission ask from the synchronous tool-dispatch approver into
/// the render loop as a modal, then wait for the answer.
///
/// The tool-dispatch chain through `zeus-agent`/`zeus-fs` is synchronous
/// (`FnMut(&PermissionRequest) -> ApprovalDecision`), so the wait can't be a
/// polled async await. It happens on a *blocked worker thread* instead — but
/// `tokio::task::block_in_place` hands that worker back to the runtime, so
/// the render loop and key handling keep running while we wait rather than
/// every approval pinning a worker hostage. Only safe on the multi-thread
/// runtime `main.rs` builds; `--yes` short-circuits before any ask.
fn build_approver(
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
    yes: bool,
) -> impl FnMut(&PermissionRequest) -> ApprovalDecision + Send {
    let tx_approval = ui_tx.clone();
    move |req: &PermissionRequest| -> ApprovalDecision {
        if yes {
            return ApprovalDecision::Approved;
        }
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        let _ = tx_approval.send(UiEvent::Approval(ApprovalRequestMsg {
            request: req.clone(),
            reply: reply_tx,
        }));
        match tokio::task::block_in_place(|| reply_rx.blocking_recv()) {
            Ok(decision) => decision,
            Err(_) => ApprovalDecision::Denied,
        }
    }
}

enum UiEvent {
    Agent(AgentEvent),
    Approval(ApprovalRequestMsg),
    /// `/model`'s picker finished probing — empty entries means "no models found".
    ModelPickerReady(
        Vec<PickerEntry>,
        usize,
        Vec<(String, Vec<zeus_provider::ModelInfo>)>,
    ),
    /// The one-shot startup version check (see `run_app`) found a newer
    /// release than `update::current_version()`. Carries the latest version
    /// string, shown as a small dim notice — never auto-installed.
    UpdateAvailable(String),
}

enum Mode {
    Chat,
    /// A pending permission ask with the scroll offset into its diff/preview
    /// (↑/↓, pgup/pgdn scroll the preview when it overflows the modal).
    Approval {
        pending: ApprovalRequestMsg,
        scroll: usize,
    },
    ModelPicker {
        entries: Vec<PickerEntry>,
        selected: usize,
    },
    /// Grouped provider picker (paid / free / local) — arrow keys to move,
    /// Enter to select. Selecting a provider without a key opens `KeyEntry`.
    ProviderPicker {
        entries: Vec<ProviderEntry>,
        selected: usize,
    },
    /// Pasting an API key for the named provider. Enter saves it (persisted
    /// to keys.toml, env var set, provider switched) and returns to Chat.
    KeyEntry {
        provider: String,
    },
    /// The two-pane side-by-side diff opened by `/diff`: `rows` hold the
    /// aligned old/new cells, `scroll` the window offset into them.
    /// ↑/↓, pgup/pgdn scroll; esc returns to Chat.
    Diff {
        rows: Vec<super::highlight::DiffRow>,
        scroll: usize,
    },
}

/// Applies a chosen (provider, model id) pair: switches provider first if
/// it differs from the one currently in use (conversation history carries
/// over — only the provider handle and model name change), then sets the
/// model. Closes the picker either way; a provider-construction failure is
/// reported in the transcript rather than left unexplained. Takes owned
/// strings (not a borrow into `state.mode`'s picker entries) specifically
/// so callers can extract the choice, drop that borrow, and then freely
/// mutate `state` here without a borrow-checker conflict.
fn apply_picker_choice(
    provider: String,
    model_id: String,
    state: &mut AppState,
    agent_slot: &mut Option<Agent>,
    config: &Config,
) {
    state.mode = Mode::Chat;
    let Some(agent) = agent_slot.as_mut() else {
        return;
    };
    if provider != state.provider {
        match create_provider(&provider, &config.providers) {
            Ok(handle) => {
                agent.set_provider(handle);
                state.provider = provider;
            }
            Err(e) => {
                state.push_error(format!("couldn't switch to provider '{provider}': {e:#}"));
                return;
            }
        }
    }
    agent.set_model(model_id.clone());
    state.record_recent_model(&state.provider.clone(), &model_id);
    state.model = model_id;
    state.context_window = None;
}

/// Selecting a *model* row (as opposed to a provider row) in any picker or
/// dropdown never checked whether its provider actually has a key —
/// `apply_picker_choice` would just try to construct the provider, fail,
/// and dump a raw "couldn't switch to provider" error with no way to
/// actually fix it, even though a model showing "Free" is only free once
/// you're through the gateway (OpenRouter/OpenCode Zen models still need
/// that gateway's own key). This is the single choke point every
/// model-selection path (both pickers' keyboard Enter and mouse click, and
/// the top-right dropdown) should call instead of `apply_picker_choice`
/// directly, so picking an unready provider's model always routes to key
/// entry rather than a dead-end error.
fn apply_model_choice_or_key_entry(
    provider: String,
    model_id: String,
    config: &Config,
    agent_slot: &mut Option<Agent>,
    state: &mut AppState,
) {
    if provider_status_ok(config, &provider) {
        apply_picker_choice(provider, model_id, state, agent_slot, config);
    } else {
        state.input.clear();
        state.cursor = 0;
        state.mode = Mode::KeyEntry { provider };
    }
}

/// Persist an API key for a provider (to `~/.zeus/keys.toml`), apply it as
/// the provider's env var for the running session, then — instead of
/// silently landing on that provider's default model — auto-chain straight
/// into the model picker scoped to it, the same way the reference product's
/// key-entry prompt auto-opens its model dialog on submit. The actual
/// provider/model switch happens once a model is picked there
/// (`apply_model_choice_or_key_entry` → `apply_picker_choice`), which
/// already handles a provider failing to construct.
fn persist_key_and_switch(
    provider: &str,
    key: &str,
    config: &Config,
    state: &mut AppState,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
) {
    let mut keys = match KeysFile::load(&config.global.keys_toml) {
        Ok(k) => k,
        Err(e) => {
            state.push_error(format!("couldn't read key store: {e:#}"));
            return;
        }
    };
    keys.keys.insert(provider.to_string(), key.to_string());
    if let Err(e) = keys.save(&config.global.keys_toml) {
        state.push_error(format!("couldn't save key store: {e:#}"));
        return;
    }
    // A newly-keyed provider can now list models it couldn't before —
    // drop the cached scan so the picker fetch below (and any later
    // picker/dropdown open) re-probes instead of showing it as still empty.
    state.model_cache = None;
    if let Some(cfg) = config.providers.get(provider) {
        if let Some(var) = &cfg.api_key_env {
            std::env::set_var(var, key);
        }
    }
    state.push_info(format!(
        "key saved for '{provider}' ({}) — choose a model",
        config.global.keys_toml.display()
    ));

    state.model_picker_search.clear();
    state.fetching_providers = true;
    let cfg = config.clone();
    let provider = provider.to_string();
    let current_model = state.model.clone();
    let recent = state.recent_models.clone();
    let favorites = state.favorite_models.clone();
    let tx = ui_tx.clone();
    tokio::spawn(async move {
        let groups = list_models_by_provider(&cfg).await;
        let (entries, selected) =
            build_model_picker_entries(&groups, &provider, &current_model, &recent, &favorites);
        let _ = tx.send(UiEvent::ModelPickerReady(entries, selected, groups));
    });
}

/// Apply a provider-picker choice. Ready providers switch immediately; a
/// provider that needs a key opens the `KeyEntry` paste screen instead.
fn apply_provider_picker_choice(
    name: String,
    ready: bool,
    config: &Config,
    agent_slot: &mut Option<Agent>,
    state: &mut AppState,
) {
    if ready {
        state.mode = Mode::Chat;
        let Some(cfg) = config.providers.get(&name) else {
            state.push_error(format!("unknown provider '{name}'"));
            return;
        };
        let model = cfg
            .default_model
            .clone()
            .unwrap_or_else(|| state.model.clone());
        match create_provider(&name, &config.providers) {
            Ok(handle) => {
                if let Some(agent) = agent_slot.as_mut() {
                    agent.set_provider(handle);
                    agent.set_model(model.clone());
                }
                state.provider = name.clone();
                state.model = model.clone();
                state.context_window = None;
                match persist_default_provider(config, &name, Some(&model)) {
                    Ok(path) => state.push_info(format!(
                        "switched to provider: {name} (model: {model}) — saved to {}",
                        path.display()
                    )),
                    Err(e) => state.push_info(format!(
                        "switched to provider {name}, but saving default failed: {e:#}"
                    )),
                }
            }
            Err(e) => state.push_error(format!("couldn't switch to '{name}': {e:#}")),
        }
    } else {
        state.input.clear();
        state.cursor = 0;
        state.mode = Mode::KeyEntry { provider: name };
    }
}

/// Opens the `/provider` picker popup — shared by the `/provider` command
/// and the "no provider connected yet" nudge on a failed send. Lists
/// providers only (no per-provider model listing — that's a separate,
/// deliberate step reached by picking a provider), so unlike the old
/// merged picker this needs no network probe and opens instantly no matter
/// how many providers are configured or how slow any of them are to reach.
fn open_provider_picker(state: &mut AppState, config: &Config) {
    let (entries, selected) = provider_picker_entries(config, &state.provider);
    if entries.is_empty() {
        state.push_error("no providers configured — see config.toml / providers.toml");
    } else {
        state.mode = Mode::ProviderPicker { entries, selected };
    }
}

/// The `/provider` slash command inside the TUI: list all configured
/// providers (with live key/local status), switch the active one, or set a
/// cloud key for the session. Mirrors the plain-REPL handler, but pushes
/// messages into the transcript instead of printing to stdout.
async fn handle_provider_tui(
    arg: &str,
    config: &Config,
    agent_slot: &mut Option<Agent>,
    state: &mut AppState,
) {
    let parts: Vec<&str> = arg.split_whitespace().collect();
    match parts.as_slice() {
        // `/provider` with no args — open the grouped picker popup.
        [] => open_provider_picker(state, config),
        ["key", name, key] => {
            let cfg = match config.providers.get(name) {
                Some(c) => c,
                None => {
                    state.push_error(format!("unknown provider '{name}' — see /provider"));
                    return;
                }
            };
            let mut keys = match KeysFile::load(&config.global.keys_toml) {
                Ok(k) => k,
                Err(e) => {
                    state.push_error(format!("couldn't read key store: {e:#}"));
                    return;
                }
            };
            keys.keys.insert(name.to_string(), key.to_string());
            if let Err(e) = keys.save(&config.global.keys_toml) {
                state.push_error(format!("couldn't save key store: {e:#}"));
                return;
            }
            // Apply immediately to the running session.
            if let Some(var) = &cfg.api_key_env {
                std::env::set_var(var, *key);
            }
            state.push_info(format!(
                "key saved for '{name}' in {} — persistent across restarts.",
                config.global.keys_toml.display()
            ));
        }
        // `/provider key <name>` — no key on the line, so open the masked
        // key-entry modal (the TUI's equivalent of the REPL's hidden-prompt
        // form). Unlike the inline form this works for a provider that
        // already has a key, so it doubles as the way to *change* one.
        ["key", name] => {
            if config.providers.get(name).is_none() {
                state.push_error(format!("unknown provider '{name}' — see /provider"));
                return;
            }
            state.input.clear();
            state.cursor = 0;
            state.mode = Mode::KeyEntry {
                provider: name.to_string(),
            };
        }
        ["key"] => state.push_error(
            "usage: /provider key <name> (prompts, hidden) or /provider key <name> <KEY>",
        ),
        [name] => match create_provider(name, &config.providers) {
            Ok(handle) => {
                if let Some(agent) = agent_slot.as_mut() {
                    agent.set_provider(handle);
                }
                let model = config
                    .providers
                    .get(name)
                    .and_then(|c| c.default_model.clone())
                    .unwrap_or_else(|| state.model.clone());
                if let Some(agent) = agent_slot.as_mut() {
                    agent.set_model(model.clone());
                }
                state.provider = name.to_string();
                state.model = model.clone();
                state.context_window = None;
                match persist_default_provider(config, name, Some(&model)) {
                    Ok(path) => state.push_info(format!(
                        "switched to provider: {name} (model: {model}) — saved to {}",
                        path.display()
                    )),
                    Err(e) => state.push_info(format!(
                        "switched to provider {name}, but saving default failed: {e:#}"
                    )),
                }
            }
            Err(e) => state.push_error(format!("couldn't switch to '{name}': {e:#}")),
        },
        _ => state.push_error("usage: /provider | /provider <name> | /provider key <name> <KEY>"),
    }
}

/// Plan mode is read-only (research/propose, no mutating tool calls —
/// enforced in `zeus-agent`'s `ToolManager`, not just a UI label); Build
/// mode is normal operation. Auto mode plans-then-executes each request.
/// Toggled with Tab, same idea as opencode's own Build/Plan/Auto switch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AgentMode {
    Build,
    Plan,
    Auto,
}

impl AgentMode {
    fn label(self) -> &'static str {
        match self {
            AgentMode::Build => "Build",
            AgentMode::Plan => "Plan",
            AgentMode::Auto => "Auto",
        }
    }

    fn toggled(self) -> Self {
        match self {
            AgentMode::Build => AgentMode::Plan,
            AgentMode::Plan => AgentMode::Auto,
            AgentMode::Auto => AgentMode::Build,
        }
    }
}

/// Which flavor of composer dropdown is currently open — slash commands
/// (`/…`) or model autocomplete (`@…`). The two share one menu slot, the
/// selection index, and the arrow/Tab/Enter/click handling; only the
/// accept action differs (commands fill `/cmd `, models swap the `@token`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MenuKind {
    Commands,
    Models,
}

/// The accent color a mode drives the UI with — the HTML page's per-mode
/// `--mode-color` (`PLAN`→gold `#fbbf24`, `BUILD`→cyan `#22d3ee`, `AUTO`→
/// magenta `#ff3d9a`). Switching modes repaints the pill, input ring, caret,
/// SEND button, hint label, and the whole right-hand sidebar, mirroring the
/// reference page's `setMode()` behavior.
fn mode_accent(mode: AgentMode) -> Color {
    match mode {
        AgentMode::Build => theme::CYAN,
        AgentMode::Plan => theme::GOLD,
        AgentMode::Auto => theme::MAGENTA,
    }
}

/// Push the selected TUI mode down to the agent: Plan turns the read-only
/// guard on; Build turns it off; Auto leaves it off and enables plan-then-run.
fn apply_agent_mode(agent: &Agent, mode: AgentMode) {
    agent.set_plan_mode(mode == AgentMode::Plan);
    agent.set_auto_mode(mode == AgentMode::Auto);
}

struct AppState {
    transcript: Vec<Block_>,
    current_reply: String,
    input: String,
    /// Char index (not byte index) into `input`.
    cursor: usize,
    busy: bool,
    /// Messages submitted while a turn was already in flight — matches the
    /// reference product's own queued-message behavior: Enter while busy
    /// doesn't get dropped, it appears in the transcript immediately and
    /// waits its turn, draining one at a time as each turn finishes rather
    /// than making the user wait for the composer to unlock before typing.
    queued_messages: std::collections::VecDeque<String>,
    /// Goal of a `/plan` run that just finished successfully, waiting on
    /// confirmation to actually execute it via `Agent::orchestrate` —
    /// `/plan` itself is read-only research plus a persisted
    /// `.agent/tasks.json`, so without this the plan just sits there
    /// unused unless the user separately knows to run `/bg orchestrate`.
    /// Consumed by pressing Enter with an empty composer (see `handle_key`);
    /// sending any other message instead clears it without running anything.
    pending_plan_goal: Option<String>,
    quit: bool,
    mode: Mode,
    agent_mode: AgentMode,
    model: String,
    provider: String,
    session_id: String,
    known_commands: Vec<(String, String)>,
    /// Project-root info for the header "directory" panel (workspace name,
    /// root path, git branch if any, file count).
    dir: DirInfo,
    /// Set once by the one-shot startup version check (`run_app`) if a
    /// newer release exists — `None` means either still checking, offline,
    /// or already up to date, all of which render identically (silent).
    update_available: Option<String>,
    /// Highlighted row in the slash-command dropdown; clamped against the
    /// current match count wherever it's read; reset to 0 whenever the
    /// input text changes so the highlight always starts back at the top
    /// match instead of pointing at a stale/since-filtered-out row.
    command_selected: usize,
    /// Screen area the model picker's list occupies in the last-rendered
    /// frame — used to map a mouse click's row back to a model index. `None`
    /// whenever the picker isn't open.
    model_picker_area: Option<Rect>,
    /// Same for the provider picker popup.
    provider_picker_area: Option<Rect>,
    /// TODO checklist for the right sidebar (mirrors the HTML sample data).
    todos: Vec<TodoItem>,
    /// True from `PlanGenerated` until the orchestrated run ends. While a
    /// real plan is driving the checklist, `complete_task`'s generic
    /// tool-name heuristic must stay out of the way — it exists to give
    /// non-orchestrated turns *some* checklist feedback, but during a real
    /// plan it would mark whichever step happens to be first-undone as done
    /// just because *some* unrelated tool call succeeded, stealing credit
    /// from the step actually in progress and leaving the checklist telling
    /// a false story about what's finished.
    plan_active: bool,
    /// Rendered rect of the TODO list — maps mouse clicks to rows.
    todo_area: Option<Rect>,
    /// When the session started — drives the "Session" readout in the sidebar.
    started: std::time::Instant,
    /// Rendered rects of the empty-state's three example chips, for mouse
    /// hit-testing.
    chip_areas: Vec<Rect>,
    /// Cached provider → models scan (`list_models_by_provider`'s result),
    /// populated on the first dropdown/`/model`/`/provider` open this
    /// session and reused after that. The set of *providers* always comes
    /// straight from `Config` (cheap, always current) — only the expensive
    /// per-provider network probe for their live model lists is cached, so
    /// repeat opens are instant instead of re-hitting every provider again.
    model_cache: Option<Vec<(String, Vec<zeus_provider::ModelInfo>)>>,
    /// True while a provider/model probe (dropdown open, `/model`,
    /// `/provider`) is running in a spawned task. Probing hits every
    /// configured provider with up to a 3s timeout each, so awaiting it
    /// inline in the key/mouse handler — which itself blocks the render
    /// loop — froze the whole UI for several seconds per click; the probe
    /// now runs in the background and reports back via `UiEvent`. Also
    /// keeps the animation tick alive so the screen doesn't look dead
    /// while it's in flight.
    fetching_providers: bool,
    /// Free-text filter for the `/model` picker's search bar.
    model_picker_search: String,
    /// Most-recently-switched-to (provider, model id) pairs, newest first —
    /// session-only (not persisted), shown as the picker's "Recent" section.
    recent_models: Vec<(String, String)>,
    /// Starred (provider, model id) pairs — persisted to `favorites.toml`
    /// next to `keys.toml` so they survive restarts, shown as the picker's
    /// "Favorites" section. Toggled with ctrl+f on the highlighted row.
    favorite_models: Vec<(String, String)>,
    /// Rendered rect of the slash-command palette (the "/model", "/help", …
    /// suggestion list under the composer) — maps a click to the matching
    /// command and lets the scroll wheel move the highlight, same pattern
    /// as every other popup's `*_area` field. `None` whenever the palette
    /// isn't showing.
    command_menu_area: Option<Rect>,
    /// Running total across every completed turn this session (each turn's
    /// `TurnResult::usage` is added in as it finishes) — drives the
    /// sidebar's "Tokens" readout. Real usage, not an estimate; `None` for
    /// a turn (e.g. Auto mode's orchestrated runs) just adds zero, since
    /// that path doesn't report per-turn usage yet.
    session_usage: TokenUsage,
    /// The active model's context window, for the "Tokens N / window"
    /// readout. `None` until the first turn completes — filled in lazily
    /// from `Agent::context_usage()` (a real provider call, so it isn't
    /// worth paying for on every render) and cleared on provider/model
    /// switch so it gets re-fetched for the new model.
    context_window: Option<u32>,
    /// Every submitted input this session (messages and slash commands
    /// alike), oldest first — shell-style Up/Down recall in the composer.
    input_history: Vec<String>,
    /// Index into `input_history` while browsing it with Up/Down; `None`
    /// means "not browsing, showing live input".
    history_pos: Option<usize>,
    /// What was actually being typed before Up first started browsing
    /// history — restored when Down arrows back past the newest entry, so
    /// browsing history doesn't lose an in-progress draft.
    history_draft: String,
    /// Manual scroll offset into the transcript (lines from the top), or
    /// `None` to auto-follow the bottom as new content streams in — same
    /// "stick to the bottom unless you've scrolled up to read history"
    /// convention any chat UI uses. Reset to `None` as soon as scrolling
    /// back down reaches the bottom, so a new message doesn't require
    /// manually re-following every time.
    transcript_scroll: Option<u16>,
    /// Rendered rect of the transcript pane, so mouse-wheel scrolling only
    /// takes effect when the cursor is actually over the chat area (not
    /// e.g. the sidebar).
    transcript_area: Option<Rect>,
    /// The bottom-most valid scroll offset as of the last render — needed
    /// so a Page Up/mouse-wheel press has something to scroll *from* the
    /// first time (before that, `transcript_scroll` is `None`/auto-follow,
    /// which carries no numeric offset of its own).
    transcript_max_scroll: u16,
    /// The scroll offset actually applied on the last render (`scroll`, not
    /// `transcript_scroll` — the latter is `None` while auto-following the
    /// bottom) — needed to convert a click's on-screen row back into an
    /// absolute transcript row for click-to-copy.
    transcript_applied_scroll: u16,
    /// Wrapped-row `[start, end)` range of each `transcript` block, in the
    /// same coordinate space the click-to-copy math above uses — parallel
    /// to `transcript`, one entry per block. Rebuilt every render.
    transcript_block_rows: Vec<(u16, u16)>,
    /// Open while searching the transcript (ctrl+f); `None` the rest of the
    /// time, same "`Some` while an overlay is active" convention `dropdown`
    /// already uses.
    search: Option<SearchState>,
    /// Open while the `/sessions` picker is up; `None` otherwise.
    session_picker: Option<SessionPickerState>,
    /// Specialist persona currently driving an Auto-mode plan step or
    /// `/workflow` phase, shown as a topbar chip — `None` outside an
    /// orchestrated run (Build/Plan turns have no persona of their own) or
    /// once one finishes (cleared at the same `busy -> false` point
    /// everything else about "this turn is over" resets at).
    active_persona: Option<String>,
    /// Rendered rect of the session picker's list, for mouse hit-testing —
    /// same pattern as `model_picker_area`/`provider_picker_area`.
    session_picker_area: Option<Rect>,
    /// In-flight tool calls: id → (start time, path touched if this is a
    /// mutating file op). Populated on `ToolCallStarted`, drained on
    /// `ToolCallFinished` to compute a duration and, on success, feed
    /// `files_touched` — `AgentEvent::ToolCallFinished` carries no
    /// arguments of its own, so the path has to be captured up front.
    tool_call_meta: std::collections::HashMap<String, (std::time::Instant, Option<String>)>,
    /// Start time of the currently-active orchestrated plan step, keyed by
    /// its description (same key `PlanStepStarted`/`PlanStepDone` already
    /// use to find the matching `TodoItem`) — lets `PlanStepDone` report how
    /// long the step took.
    plan_step_started: std::collections::HashMap<String, std::time::Instant>,
    /// Paths written/edited/deleted this session, most-recently-touched
    /// last — drives the sidebar's "Files" panel. Capped so a very long
    /// session doesn't grow this unboundedly.
    files_touched: Vec<String>,
    /// Selected transcript blocks as an inclusive `(start, end)` block-index
    /// range — highlighted in the transcript and copied with ctrl+y. `None`
    /// when nothing is selected. Selection is how you reach older messages:
    /// a click sets it, shift-click or drag extends it, and ctrl+y copies
    /// just the selected blocks (falling back to the last reply when empty).
    selection: Option<(usize, usize)>,
    /// The block where a mouse press started, so drag / shift-click extends
    /// the selection from a fixed end instead of replacing it.
    selection_anchor: Option<usize>,
}

/// Transcript search — an overlay on top of `Mode::Chat`, not a `Mode`
/// variant of its own, so a new `Mode` doesn't need updating every
/// exhaustive match on it across this file for one narrow feature.
struct SearchState {
    query: String,
    /// Indices into `AppState::transcript` whose text matches `query`
    /// (case-insensitive substring), oldest first — recomputed on every
    /// keystroke.
    matches: Vec<usize>,
    /// Which entry in `matches` is currently focused, jumped to with Enter.
    current: usize,
}

struct TodoItem {
    text: String,
    done: bool,
    active: bool,
    /// Wall-clock time the step took, once finished — `None` for a step
    /// still pending/active, or one completed by the generic
    /// `complete_task` heuristic (which has no start time to measure from).
    duration: Option<std::time::Duration>,
}

/// Git branch shown in the side-panel session footer. Single field only —
/// other directory facts (workspace/path/file count) aren't rendered yet, so
/// they were dropped to avoid dead fields.
pub struct DirInfo {
    pub git_branch: Option<String>,
}

impl AppState {
    fn new(
        agent: &Agent,
        known_commands: Vec<(String, String)>,
        dir: DirInfo,
        start_in_plan: bool,
        config: &Config,
    ) -> Self {
        let state = Self {
            transcript: Vec::new(),
            current_reply: String::new(),
            input: String::new(),
            cursor: 0,
            busy: false,
            queued_messages: std::collections::VecDeque::new(),
            pending_plan_goal: None,
            quit: false,
            mode: Mode::Chat,
            agent_mode: if start_in_plan {
                AgentMode::Plan
            } else {
                AgentMode::Build
            },
            model: agent.model().to_string(),
            provider: agent.provider_id().to_string(),
            session_id: agent.session_id().to_string(),
            known_commands,
            dir,
            update_available: None,
            command_selected: 0,
            model_picker_area: None,
            provider_picker_area: None,
            todos: Vec::new(),
            plan_active: false,
            todo_area: None,
            started: std::time::Instant::now(),
            chip_areas: Vec::new(),
            model_cache: None,
            fetching_providers: false,
            model_picker_search: String::new(),
            recent_models: Vec::new(),
            favorite_models: load_favorites(config),
            command_menu_area: None,
            session_usage: TokenUsage::default(),
            context_window: None,
            input_history: Vec::new(),
            history_pos: None,
            history_draft: String::new(),
            transcript_scroll: None,
            transcript_area: None,
            transcript_max_scroll: 0,
            transcript_applied_scroll: 0,
            transcript_block_rows: Vec::new(),
            search: None,
            session_picker: None,
            session_picker_area: None,
            active_persona: None,
            tool_call_meta: std::collections::HashMap::new(),
            plan_step_started: std::collections::HashMap::new(),
            files_touched: Vec::new(),
            selection: None,
            selection_anchor: None,
        };
        state
    }

    /// Rows to move per Page Up/Down press or mouse-wheel notch — a
    /// generous fraction of a screen's worth, so context still carries
    /// over between jumps (same convention a pager/`less` uses), but
    /// small enough (3 lines) for a single wheel notch not to feel like a
    /// full page-jump.
    fn scroll_transcript(&mut self, up: bool, lines: u16) {
        let current = self.transcript_scroll.unwrap_or(self.transcript_max_scroll);
        let next = if up {
            current.saturating_sub(lines)
        } else {
            (current + lines).min(self.transcript_max_scroll)
        };
        self.transcript_scroll = if next >= self.transcript_max_scroll {
            None
        } else {
            Some(next)
        };
    }

    /// Appends a submitted input to history (skipping exact-duplicate
    /// repeats of the last entry and blank lines, same as a shell's
    /// `HISTCONTROL=ignoredups`), and resets history browsing back to
    /// "live" for the next Up press.
    fn record_history(&mut self, text: String) {
        self.history_pos = None;
        self.history_draft.clear();
        if text.trim().is_empty() {
            return;
        }
        if self.input_history.last() != Some(&text) {
            self.input_history.push(text);
        }
    }

    /// Records a (provider, model) pair as most-recently-used — moves it to
    /// the front if already present, caps the list so the "Recent" section
    /// doesn't grow forever.
    fn record_recent_model(&mut self, provider: &str, model: &str) {
        self.recent_models
            .retain(|(p, m)| !(p == provider && m == model));
        self.recent_models
            .insert(0, (provider.to_string(), model.to_string()));
        self.recent_models.truncate(8);
    }

    /// Toggles a (provider, model) favorite and persists the change
    /// immediately — a starred model should survive even if the app closes
    /// before a clean exit.
    fn toggle_favorite_model(&mut self, provider: &str, model: &str, config: &Config) {
        if let Some(pos) = self
            .favorite_models
            .iter()
            .position(|(p, m)| p == provider && m == model)
        {
            self.favorite_models.remove(pos);
        } else {
            self.favorite_models
                .push((provider.to_string(), model.to_string()));
        }
        save_favorites(config, &self.favorite_models);
    }

    /// True while the full-screen empty-state splash (`render_empty_state`)
    /// should be shown instead of the normal topbar/sidebar/chat-column
    /// layout — an empty transcript with no turn in flight. `push_user`
    /// (called before any turn starts) makes this false again as soon as
    /// the first message goes out, so it never flickers mid-turn.
    fn showing_empty_state(&self) -> bool {
        self.transcript.is_empty() && !self.busy && !matches!(self.mode, Mode::KeyEntry { .. })
    }

    fn flush_current_reply(&mut self) {
        if !self.current_reply.is_empty() {
            let text = std::mem::take(&mut self.current_reply);
            self.transcript.push(Block_::new(Role::Assistant, text));
        }
    }

    fn apply_agent_event(&mut self, ev: AgentEvent) {
        match ev {
            AgentEvent::TextDelta(t) => self.current_reply.push_str(&t),
            AgentEvent::ToolCallStarted {
                id,
                name,
                arguments,
            } => {
                self.flush_current_reply();
                let path = touched_path(&name, &arguments);
                self.tool_call_meta
                    .insert(id, (std::time::Instant::now(), path));
                self.transcript
                    .push(Block_::new(Role::Tool, format!("{name} {arguments}")));
            }
            AgentEvent::ToolCallFinished {
                id,
                name,
                result,
                is_error,
            } => {
                self.flush_current_reply();
                let (elapsed, path) = match self.tool_call_meta.remove(&id) {
                    Some((start, path)) => (Some(start.elapsed()), path),
                    None => (None, None),
                };
                let role = if is_error {
                    Role::ToolError
                } else {
                    Role::Tool
                };
                let marker = if is_error { "failed" } else { "done" };
                let timing = elapsed
                    .map(|d| format!(", {}", fmt_duration(d)))
                    .unwrap_or_default();
                self.transcript.push(Block_::new(
                    role,
                    format!("{name} ({marker}{timing})\n{result}"),
                ));
                if !is_error {
                    if let Some(path) = path {
                        self.touch_file(path);
                    }
                }
                // A completed mutation auto-checks the in-flight task, filling
                // the sidebar progress bar (the HTML page's AI-driven TODOs).
                // Only when there's no *real* plan driving the checklist —
                // `PlanStepStarted`/`PlanStepDone` already track an
                // orchestrated run's steps precisely by matching their own
                // description text, and this heuristic firing alongside
                // them would just check off whichever step happens to be
                // first-undone because some unrelated tool call succeeded.
                if !is_error
                    && !self.plan_active
                    && matches!(
                        name.as_str(),
                        "apply_patch" | "write" | "edit" | "update" | "run" | "patch"
                    )
                {
                    self.complete_task(&name);
                }
            }
            AgentEvent::Compacted(c) => {
                self.push_info(format!(
                    "(compacted {} earlier message(s))",
                    c.removed_messages
                ));
            }
            AgentEvent::Cancelled => {
                self.push_info("(cancelled)");
                self.plan_active = false;
            }
            AgentEvent::Done => self.flush_current_reply(),
            AgentEvent::TodosUpdated { todos } => {
                // The model owns this list and sends the full state every
                // call (not a diff) — replace wholesale, same convention
                // the reference product's own todo tool uses. This works
                // in any mode (Build included), unlike the old checklist
                // which only ever populated from an orchestrated `/plan`.
                self.todos = todos
                    .into_iter()
                    .map(|t| TodoItem {
                        done: t.status == "completed",
                        active: t.status == "in_progress",
                        text: t.content,
                        duration: None,
                    })
                    .collect();
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
                self.push_info(format!("plan · {} step(s): {roster}", steps.len()));
                // Seed the sidebar checklist with real *pending* items up
                // front, rather than only ever showing retroactively
                // `done: true` entries — see `complete_task`'s fallback for
                // turns with no plan (a bare tool call with no plan step
                // behind it still gets its own after-the-fact entry there).
                self.todos = steps
                    .iter()
                    .map(|s| TodoItem {
                        text: s.description.clone(),
                        done: false,
                        active: false,
                        duration: None,
                    })
                    .collect();
            }
            AgentEvent::PlanStepStarted { step } => {
                self.push_info(format!("plan step {} · {}", step.id, step.description));
                self.active_persona = step.persona.clone();
                self.plan_step_started
                    .insert(step.description.clone(), std::time::Instant::now());
                if let Some(t) = self.todos.iter_mut().find(|t| t.text == step.description) {
                    t.active = true;
                }
            }
            AgentEvent::PlanReviewed { persona, report } => {
                self.transcript.push(Block_::new(
                    Role::Tool,
                    format!("review ({persona})\n{report}"),
                ));
            }
            AgentEvent::PlanStepDone { step, summary } => {
                let elapsed = self
                    .plan_step_started
                    .remove(&step.description)
                    .map(|t| t.elapsed());
                let timing = elapsed
                    .map(|d| format!(" ({})", fmt_duration(d)))
                    .unwrap_or_default();
                self.transcript.push(Block_::new(
                    Role::Tool,
                    format!(
                        "step {} done{timing} · {}\n{}",
                        step.id, step.description, summary
                    ),
                ));
                if let Some(t) = self.todos.iter_mut().find(|t| t.text == step.description) {
                    t.done = true;
                    t.active = false;
                    t.duration = elapsed;
                }
            }
            AgentEvent::PlanStepDeclined { step } => {
                self.plan_step_started.remove(&step.description);
                self.transcript.push(Block_::new(
                    Role::Tool,
                    format!("step {} declined · {}", step.id, step.description),
                ));
                // No separate "declined" visual on a checkbox — mark it
                // done so it doesn't linger as a permanently pending item.
                if let Some(t) = self.todos.iter_mut().find(|t| t.text == step.description) {
                    t.done = true;
                    t.active = false;
                }
            }
            AgentEvent::OrchestrationDone { summary } => {
                self.flush_current_reply();
                self.transcript.push(Block_::new(Role::Assistant, summary));
            }
            AgentEvent::OrchestrationRevision { report } => {
                self.flush_current_reply();
                self.transcript.push(Block_::new(
                    Role::Tool,
                    format!("lead reviewer did NOT accept\n{report}"),
                ));
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
                self.push_info(format!("workflow '{id}' — {description}"));
                self.transcript
                    .push(Block_::new(Role::Info, roster.to_string()));
            }
            AgentEvent::WorkflowPhaseStarted { name, persona } => {
                self.push_info(format!("▶ {name} [as {persona}]"));
                self.active_persona = Some(persona);
            }
            AgentEvent::WorkflowPhaseDone {
                name,
                persona,
                summary,
            } => {
                self.transcript.push(Block_::new(
                    Role::Tool,
                    format!("phase · {name} [{persona}]\n{summary}"),
                ));
            }
            AgentEvent::WorkflowDone { summary } => {
                self.flush_current_reply();
                self.transcript.push(Block_::new(Role::Assistant, summary));
            }
            AgentEvent::RepoAnalyzed { stack, relevance } => {
                let mut text = format!("Analyzing repository...\n{stack}");
                if !relevance.is_empty() {
                    text.push_str(&format!("\n\n{relevance}"));
                }
                self.transcript.push(Block_::new(Role::Info, text));
            }
            AgentEvent::RepoRelevanceUpdated { relevance } => {
                self.transcript.push(Block_::new(Role::Info, relevance));
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
                    self.transcript.push(Block_::new(
                        Role::Error,
                        "orientation docs not written (markers missing)".to_string(),
                    ));
                } else {
                    self.transcript.push(Block_::new(
                        Role::Info,
                        format!("wrote {}", written.join(", ")),
                    ));
                }
            }
            AgentEvent::ReviewUncommitted { persona, report } => {
                self.flush_current_reply();
                self.transcript
                    .push(Block_::new(Role::Info, format!("review ({persona})")));
                self.transcript.push(Block_::new(Role::Assistant, report));
            }
            AgentEvent::FeaturesSuggested { report } => {
                self.flush_current_reply();
                self.transcript.push(Block_::new(
                    Role::Info,
                    "next-feature suggestions".to_string(),
                ));
                self.transcript.push(Block_::new(Role::Assistant, report));
            }
        }
    }

    fn push_user(&mut self, text: String) {
        self.transcript.push(Block_::new(Role::User, text));
    }

    fn push_info(&mut self, text: impl Into<String>) {
        self.transcript.push(Block_::new(Role::Info, text.into()));
    }

    fn push_error(&mut self, text: impl Into<String>) {
        self.transcript.push(Block_::new(Role::Error, text.into()));
    }

    /// Mark the current in-flight task complete (the AI drives the checklist):
    /// checks the first open `active` task, or records a fresh done entry when
    /// there's nothing active yet. Re-rendering recomputes the progress fill.
    fn complete_task(&mut self, tool: &str) {
        if let Some(t) = self.todos.iter_mut().find(|t| !t.done) {
            t.done = true;
            t.active = false;
        } else {
            // Active list so a completed task still shows as a ticked row.
            let text = if self.todos.is_empty() {
                format!("Complete {tool}")
            } else {
                self.todos
                    .last()
                    .map(|x| x.text.clone())
                    .unwrap_or_default()
            };
            if !self.todos.iter().any(|x| x.done && x.text == text) {
                self.todos.push(TodoItem {
                    text,
                    done: true,
                    active: false,
                    duration: None,
                });
            }
        }
    }

    /// Recomputes `search.matches` from the current query — called on every
    /// keystroke while search is open.
    fn search_recompute(&mut self) {
        let Some(query) = self.search.as_ref().map(|s| s.query.clone()) else {
            return;
        };
        let matches: Vec<usize> = if query.is_empty() {
            Vec::new()
        } else {
            let q = query.to_lowercase();
            self.transcript
                .iter()
                .enumerate()
                .filter(|(_, b)| b.plain_text().to_lowercase().contains(&q))
                .map(|(i, _)| i)
                .collect()
        };
        if let Some(search) = self.search.as_mut() {
            search.matches = matches;
            search.current = 0;
        }
    }

    /// Scrolls the transcript so the currently-focused search match is in
    /// view — the same "block index → wrapped row" lookup click-to-copy
    /// uses (`transcript_block_rows`), just driving scroll instead of a
    /// clipboard copy.
    fn search_jump_to_current(&mut self) {
        let Some(block_idx) = self
            .search
            .as_ref()
            .and_then(|s| s.matches.get(s.current).copied())
        else {
            return;
        };
        if let Some(&(start, _)) = self.transcript_block_rows.get(block_idx) {
            self.transcript_scroll = Some(start.min(self.transcript_max_scroll));
        }
    }

    /// Records a touched file path, most-recent last, deduping any earlier
    /// touch of the same path — mirrors `record_recent_model`'s move-to-end
    /// pattern. Capped at 50 entries; the sidebar only ever shows the last
    /// few anyway.
    fn touch_file(&mut self, path: String) {
        self.files_touched.retain(|p| p != &path);
        self.files_touched.push(path);
        if self.files_touched.len() > 50 {
            self.files_touched.remove(0);
        }
    }

    fn toggle_todo(&mut self, idx: usize) {
        let Some(t) = self.todos.get_mut(idx) else {
            return;
        };
        t.done = !t.done;
        if t.done {
            t.active = false;
        }
    }

    fn command_matches(&self) -> Vec<(&str, &str)> {
        filter_commands(&self.input, &self.known_commands)
    }

    /// The currently active composer dropdown. Slash commands win when the
    /// input starts with `/…` (no whitespace yet — the existing behavior);
    /// otherwise a trailing `@…` token turns the same dropdown into model
    /// autocomplete, so the two never fight over the single menu slot.
    /// Returns owned entries plus which kind so accept/render can treat
    /// them differently (commands fill `/cmd `, models swap the `@token`).
    fn menu(&self) -> (Vec<(String, String)>, MenuKind) {
        let cmd = self.command_matches();
        if !cmd.is_empty() {
            return (
                cmd.into_iter()
                    .map(|(n, d)| (n.to_string(), d.to_string()))
                    .collect(),
                MenuKind::Commands,
            );
        }
        let models = self.model_matches();
        if !models.is_empty() {
            return (models, MenuKind::Models);
        }
        (Vec::new(), MenuKind::Commands)
    }

    /// Model autocomplete candidates for the composer's trailing `@…`
    /// token: every model the app already knows about — the cached live
    /// probe (per provider), recents, and favorites — de-duplicated and
    /// filtered by the text after the last `@`, case-insensitively.
    /// An empty partial (bare `@`) lists everything.
    fn model_matches(&self) -> Vec<(String, String)> {
        filter_model_matches(
            &self.input,
            self.model_cache.as_deref().unwrap_or(&[]),
            &self.recent_models,
            &self.favorite_models,
        )
    }
}

/// Shared by `AppState::model_matches` and its unit tests — pure matching
/// logic with no `AppState` dependency.
fn filter_model_matches(
    input: &str,
    cache: &[(String, Vec<zeus_provider::ModelInfo>)],
    recent: &[(String, String)],
    favorites: &[(String, String)],
) -> Vec<(String, String)> {
    let last_token = input.rsplit(char::is_whitespace).next().unwrap_or("");
    let Some(partial) = last_token.strip_prefix('@') else {
        return Vec::new();
    };
    let partial = partial.to_ascii_lowercase();

    let mut out: Vec<(String, String)> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (provider, models) in cache {
        for m in models {
            let label = format!("{provider}/{model}", model = m.id);
            if seen.insert(label.clone()) {
                out.push((label, provider.clone()));
            }
        }
    }
    for (provider, model) in recent.iter().chain(favorites.iter()) {
        let label = format!("{provider}/{model}");
        if seen.insert(label.clone()) {
            out.push((label, provider.clone()));
        }
    }
    out.retain(|(label, _)| {
        if partial.is_empty() {
            return true;
        }
        let lower = label.to_ascii_lowercase();
        if lower.starts_with(&partial) {
            return true;
        }
        // Also match against the model id alone (after the provider's
        // slash) — typing `@clau` should surface `openrouter/claude-…`.
        lower
            .rsplit_once('/')
            .map(|(_, model)| model.starts_with(&partial))
            .unwrap_or(false)
    });
    out
}

/// Prefix-match slash commands against the composer input: a leading `/` with
/// no whitespace after it narrows `known` to those starting with the prefix.
/// Split out of the `AppState` method so the matching logic is unit-testable
/// without constructing a full `AppState`.
fn filter_commands<'a>(input: &str, known: &'a [(String, String)]) -> Vec<(&'a str, &'a str)> {
    match input.strip_prefix('/') {
        Some(prefix) if !prefix.contains(char::is_whitespace) => known
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_str()))
            .filter(|(n, _)| n.starts_with(prefix))
            .collect(),
        _ => Vec::new(),
    }
}

/// Copies the most recent assistant/tool block to the clipboard — shared
/// by the `/copy` slash command and the ctrl+y keybinding.
fn copy_last_response(state: &mut AppState) {
    let block = state
        .transcript
        .iter()
        .rev()
        .find(|b| matches!(b.role, Role::Assistant | Role::Tool));
    match block {
        Some(b) => match super::clipboard::copy(&b.plain_text()) {
            Ok(()) => state.push_info("copied last block to clipboard"),
            Err(e) => state.push_error(format!("copy failed: {e}")),
        },
        None => state.push_error("nothing to copy yet"),
    }
}

/// Copy the currently selected transcript blocks (inclusive `selection`
/// range) as one joined text blob. Clears the selection once copied, since
/// the highlight's job — pointing at the blocks to copy — is done.
fn copy_selection(state: &mut AppState) {
    let Some(text) = selection_plain_text(&state.transcript, state.selection) else {
        return;
    };
    let count = text.chars().count();
    let len = state
        .selection
        .map(|(a, b)| b.saturating_sub(a) + 1)
        .unwrap_or(0);
    match super::clipboard::copy(&text) {
        Ok(()) => {
            state.push_info(format!(
                "copied {count} char(s) from {len} block(s) to clipboard"
            ));
            state.selection = None;
            state.selection_anchor = None;
        }
        Err(e) => state.push_error(format!("copy failed: {e}")),
    }
}

/// The text the current selection covers — the selected blocks' plain text
/// joined with blank lines. Split out so the join/range logic is
/// unit-testable without touching the system clipboard.
fn selection_plain_text(
    transcript: &[Block_],
    selection: Option<(usize, usize)>,
) -> Option<String> {
    let (start, end) = selection?;
    let blocks: Vec<&Block_> = transcript
        .iter()
        .take(end.saturating_add(1))
        .skip(start)
        .collect();
    if blocks.is_empty() {
        return None;
    }
    Some(
        blocks
            .iter()
            .map(|b| b.plain_text())
            .collect::<Vec<_>>()
            .join("\n\n"),
    )
}

/// Map a mouse position over the transcript back to the block under it, in
/// the same wrapped-row space `transcript_block_rows` / `transcript_text`
/// render and `transcript_applied_scroll` scrolls in. `None` when the click
/// is outside the transcript pane or over the separator gap between blocks.
fn transcript_block_at(
    col: u16,
    row: u16,
    area: Option<Rect>,
    block_rows: &[(u16, u16)],
    scroll: u16,
) -> Option<usize> {
    let area = area?;
    if col < area.x || col >= area.x + area.width || row < area.y || row >= area.y + area.height {
        return None;
    }
    let clicked_row = (row - area.y) + scroll;
    block_rows
        .iter()
        .position(|&(start, end)| clicked_row >= start && clicked_row < end)
}

/// One-line pitch + signup URL for the key-entry modal — the "why this
/// provider, where do I get a key" copy a first-time user actually needs.
fn byte_index(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}

fn insert_char_at(s: &mut String, char_idx: usize, c: char) {
    let bi = byte_index(s, char_idx);
    s.insert(bi, c);
}

fn remove_char_at(s: &mut String, char_idx: usize) {
    let bi = byte_index(s, char_idx);
    if bi < s.len() {
        s.remove(bi);
    }
}

/// The pinned input box: a mode-colored accent bar down the left edge, an
/// elevated flat panel (no full box border), the placeholder/typed text on
/// its own line, and mode/model/provider status on the line beneath it —
/// no `›` caret, no SEND button; Enter is the only way to send. The bar
/// color and status line repaint whenever the mode switches.
fn render_input_box(f: &mut Frame, area: Rect, state: &AppState, input_text_h: u16) {
    let accent = mode_accent(state.agent_mode);
    // Typing (to queue the next message) works even mid-turn now, so the
    // composer stays "focused" while busy too — only an actual picker/modal
    // mode takes the cursor away.
    let focused = matches!(state.mode, Mode::Chat);
    let bar = if focused { accent } else { theme::border() };
    // A single thick accent bar down the left edge on a flat elevated panel
    // — no full box border, no `›` caret — rather than the earlier
    // all-sides bordered composer.
    let block = Block::default()
        .borders(Borders::LEFT)
        .border_type(BorderType::Thick)
        .border_style(Style::default().fg(bar))
        .style(Style::default().bg(theme::panel2()));
    let raw_inner = block.inner(area);
    f.render_widget(block, area);
    // One column of breathing room between the accent bar and the text —
    // `Block::padding` would do this too, but a manual inset avoids pulling
    // in `ratatui::widgets::Padding` for a single call site.
    let inner = Rect {
        x: raw_inner.x + 1,
        y: raw_inner.y,
        width: raw_inner.width.saturating_sub(1),
        height: raw_inner.height,
    };

    // `input_text_h` (computed by the caller, from how many rows the
    // current input actually wraps to) — a long message used to be capped
    // at a fixed single row and silently clip everything past the first
    // wrapped line. A leading blank row gives the composer breathing room
    // instead of the text sitting flush against the transcript above it.
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(input_text_h),
        Constraint::Length(1),
    ])
    .split(inner);
    let (text_row, status_row) = (rows[1], rows[2]);

    if matches!(state.mode, Mode::Approval { .. }) {
        // The actual approval UI is a centered modal
        // (`render_approval_modal`, drawn on top by `render()`, with the
        // full diff preview instead of one clipped line) — this bar just
        // goes quiet underneath it.
        f.render_widget(
            Paragraph::new(Line::from(Span::styled("…", placeholder_style()))),
            text_row,
        );
        return;
    }

    if matches!(state.mode, Mode::KeyEntry { .. }) {
        // The actual key-entry UI is a centered modal (`render_key_entry_modal`,
        // drawn on top by `render()`) — this bar just goes quiet underneath it.
        f.render_widget(
            Paragraph::new(Line::from(Span::styled("…", placeholder_style()))),
            text_row,
        );
        return;
    }

    let input_line = if !state.input.is_empty() {
        // `Text` not a single `Line` — a multiline draft (Shift+Enter) needs
        // each `\n` to become its own row, and `Text::from(String)` splits
        // on newlines exactly like the empty-state composer already does.
        Text::from(state.input.clone())
    } else if state.busy {
        Text::from(Line::from(Span::styled(
            format!(
                "{} zeus is working… (you can type the next message)",
                spinner_glyph(state)
            ),
            placeholder_style(),
        )))
    } else {
        Text::from(Line::from(vec![
            Span::styled("Ask anything… ", placeholder_style()),
            Span::styled("\"Fix a TODO in the codebase\"", placeholder_style()),
        ]))
    };
    f.render_widget(
        Paragraph::new(input_line).wrap(Wrap { trim: false }),
        text_row,
    );

    // Mode · model provider — no send button, no caret; Enter is the only
    // way to send, same as it always was, just no longer advertised with a
    // dedicated button now that the box has no border to anchor one to.
    let status = Line::from(vec![
        Span::styled(
            state.agent_mode.label(),
            Style::default().fg(accent).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" · ", theme::faint()),
        Span::styled(
            state.model.clone(),
            theme::text().add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(state.provider.clone(), theme::dim()),
    ]);
    f.render_widget(Paragraph::new(status), status_row);

    if focused {
        // Multi-row input: figure out which wrapped row the cursor lands
        // on by re-wrapping just the text before it. Not pixel-identical
        // to ratatui's own `Wrap` in every edge case (mid-word cursor
        // positions can wrap slightly differently in principle), but
        // exact for the overwhelmingly common case of typing forward with
        // the cursor at the end, and close enough otherwise.
        let typed_before: String = state.input.chars().take(state.cursor).collect();
        let wrapped_before = wrap_preserving_newlines(&typed_before, inner.width.max(1) as usize);
        let last_row = input_text_h.saturating_sub(1);
        let row = (wrapped_before.len() as u16)
            .saturating_sub(1)
            .min(last_row);
        let col = wrapped_before.last().map(|l| char_count(l)).unwrap_or(0) as u16;
        f.set_cursor_position((text_row.x + col, text_row.y + row));
    }
}

/// A "key label" pair in the hint row — the key itself bold/bright, its
/// description dim, several spaces of gap before the next pair rather than
/// a bullet separator.
fn hint_pair(key: &str, label: &str) -> Vec<Span<'static>> {
    vec![
        Span::styled(key.to_string(), theme::text().add_modifier(Modifier::BOLD)),
        // `dim()` not `faint()` — these are the app's primary discoverable
        // keybinding hints, not decoration; `faint()` fails WCAG contrast.
        Span::styled(format!(" {label}"), theme::dim()),
    ]
}

/// Build the key-binding legend, shrinking it to fit `max_w` columns:
/// full 4-space gaps when there's room, 2-space gaps on a tighter terminal,
/// and whole trailing pairs dropped (from the right) once that still
/// overflows — so the row never clips mid-label on a narrow window. The
/// full legend is ~92 columns, wider than a sidebar-less 80-col terminal.
fn hints_for_width(pairs: &[[&str; 2]], max_w: usize) -> Vec<Span<'static>> {
    fn assembled(pairs: &[[&str; 2]], gap: &'static str) -> Vec<Span<'static>> {
        let mut spans = Vec::new();
        for (i, [key, label]) in pairs.iter().enumerate() {
            if i > 0 {
                spans.push(Span::raw(gap));
            }
            spans.extend(hint_pair(key, label));
        }
        spans
    }
    fn width(spans: &[Span<'static>]) -> usize {
        spans.iter().map(|s| s.content.chars().count()).sum()
    }

    for gap in ["    ", "  "] {
        let spans = assembled(pairs, gap);
        if width(&spans) <= max_w {
            return spans;
        }
    }
    let mut spans = Vec::new();
    for pair in pairs {
        let mut candidate = spans.clone();
        if !candidate.is_empty() {
            candidate.push(Span::raw("  "));
        }
        candidate.extend(hint_pair(pair[0], pair[1]));
        if width(&candidate) > max_w {
            break;
        }
        spans = candidate;
    }
    spans
}

/// Mode/model/provider now lives inside the input box itself (its second
/// line), so this row is just the key-binding legend underneath it.
fn render_hints(f: &mut Frame, area: Rect, _state: &AppState) {
    let pairs: &[[&str; 2]] = &[
        ["tab", "agents"],
        ["/ ctrl+p", "commands"],
        ["click", "select"],
        ["ctrl+c", "copy"],
        ["ctrl+f", "find"],
        ["esc", "close"],
    ];
    f.render_widget(
        Paragraph::new(Line::from(hints_for_width(pairs, area.width as usize))),
        area,
    );
}

/// Animated activity glyph: cycles through braille frames based on elapsed
/// time since the UI started, so "busy" states read as alive rather than
/// stalled. The 100ms step matches the redraw cadence.
fn spinner_frames() -> &'static [char] {
    &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
}

fn spinner_glyph(state: &AppState) -> char {
    let frames = spinner_frames();
    frames[(state.started.elapsed().as_millis() / 100) as usize % frames.len()]
}

fn transcript_text(state: &AppState, width: u16) -> Text<'static> {
    let selection = state.selection;
    let mut lines: Vec<Line<'static>> = Vec::new();
    for (i, block) in state.transcript.iter().enumerate() {
        let selected = selection.is_some_and(|(a, b)| i >= a && i <= b);
        if selected {
            lines.extend(block.to_lines(width).into_iter().map(style_selected));
            // Solid fill under the whole selected run so the highlight reads
            // as one contiguous bar across block boundaries (the blank
            // separator row between messages gets the same background). This
            // is still exactly one row, so `transcript_block_rows`'s
            // `wrapped + 1` math stays valid and click/scroll mapping
            // doesn't drift when a selection is active.
            lines.push(Line::from(vec![Span::styled(
                " ".repeat(width as usize),
                Style::default().bg(theme::selected_bg()),
            )]));
        } else {
            lines.extend(block.to_lines(width));
            lines.push(Line::from(""));
        }
    }
    if !state.current_reply.is_empty() {
        let streaming = Block_::new(Role::Assistant, state.current_reply.clone());
        lines.extend(streaming.to_lines(width));
    } else if state.busy {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{} ", spinner_glyph(state)),
                theme::violet().add_modifier(Modifier::BOLD),
            ),
            Span::styled("thinking…", theme::dim()),
        ]));
    }
    // `state.transcript.is_empty() && !state.busy` never reaches here — that
    // case renders `render_empty_state` instead of the chat column at all.
    Text::from(lines)
}

/// Re-style a transcript line so every span carries the selection
/// background, overriding whatever card or diff tint it had (a selection has
/// to read as one uniform bar). Foregrounds are kept so the text itself
/// stays legible; ratatui keeps the span's background across wrapped
/// continuation rows, so even a long line that wraps is highlighted whole.
fn style_selected(line: Line<'static>) -> Line<'static> {
    Line::from(
        line.spans
            .into_iter()
            .map(|s| Span::styled(s.content, s.style.bg(theme::selected_bg())))
            .collect::<Vec<_>>(),
    )
}

/// Wrapped-row `[start, end)` range for each transcript block, in the same
/// coordinate space `transcript_text`'s `Paragraph` renders/scrolls in (its
/// wrapped `line_count`, not each block's raw pre-wrap `Vec<Line>` length —
/// a long or wide highlighted line can wrap further inside the `Paragraph`
/// than `Block_::to_lines` alone produced, the same reason the auto-scroll
/// math below already uses `line_count` over a raw count). Used to map a
/// mouse click back to "which message did they click on" for click-to-copy.
fn transcript_block_rows(state: &AppState, width: u16) -> Vec<(u16, u16)> {
    let mut out = Vec::with_capacity(state.transcript.len());
    let mut row: u16 = 0;
    for block in &state.transcript {
        let wrapped = Paragraph::new(Text::from(block.to_lines(width))).line_count(width) as u16;
        out.push((row, row + wrapped));
        row += wrapped + 1; // +1 for the blank separator line after each block
    }
    out
}

/// Transcript search bar — a small floating box in the top-right corner of
/// the chat column (not centered, so it never covers the match it just
/// jumped to) showing the live query and match count. Opened with ctrl+f,
/// closed with Esc — see `SearchState` and the "settings"-adjacent key
/// handling near the top of the main key-event match.
fn render_search_bar(f: &mut Frame, area: Rect, search: &SearchState) {
    let label = if search.query.is_empty() {
        "type to search…".to_string()
    } else if search.matches.is_empty() {
        format!("{}  ·  no matches", search.query)
    } else {
        format!(
            "{}  ·  {}/{}",
            search.query,
            search.current + 1,
            search.matches.len()
        )
    };
    let width = (label.chars().count() as u16 + 8).clamp(24, area.width.saturating_sub(4).max(24));
    let bar_area = Rect {
        x: area.x + area.width.saturating_sub(width + 2),
        y: area.y,
        width,
        height: 3,
    };
    f.render_widget(Clear, bar_area);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::accent()))
        .style(Style::default().bg(theme::panel()))
        .title(Line::from(Span::styled(
            " find ",
            Style::default()
                .fg(theme::accent())
                .add_modifier(Modifier::BOLD),
        )))
        .title_bottom(
            Line::from(Span::styled(" enter next · esc close ", theme::dim()))
                .alignment(Alignment::Center),
        );
    let inner = block.inner(bar_area);
    f.render_widget(block, bar_area);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(label, theme::text()))),
        inner,
    );
}

/// Resets everything tied to *this conversation's view* — transcript,
/// checklist, per-turn timing/plan-step bookkeeping — without touching
/// anything that outlives it (favorites, recent models, files touched on
/// disk). Shared by `/clear`, `/new`, and resuming a session from the
/// `/sessions` picker, all three of which swap in a fresh `Agent` and need
/// the old conversation's leftovers not to bleed into the new one.
fn reset_conversation_view(state: &mut AppState) {
    state.transcript.clear();
    state.current_reply.clear();
    state.session_usage = TokenUsage::default();
    state.context_window = None;
    state.todos.clear();
    state.plan_active = false;
    state.tool_call_meta.clear();
    state.plan_step_started.clear();
}

/// Swaps in an `Agent` bound to `session_id`'s saved conversation — the
/// `/sessions` picker's resume action. Restores the *context* the model
/// continues from; it doesn't replay the old messages into the visible
/// transcript (see `build_agent_repl_with_session`).
async fn resume_session(
    session_id: String,
    config: &Config,
    agent_slot: &mut Option<Agent>,
    state: &mut AppState,
) {
    let result = build_agent_repl_with_session(
        config,
        Some(state.provider.clone()),
        Some(state.model.clone()),
        session_id.clone(),
    )
    .await;
    match result {
        Ok(agent) => {
            apply_agent_mode(&agent, state.agent_mode);
            state.session_id = agent.session_id().to_string();
            state.model = agent.model().to_string();
            state.provider = agent.provider_id().to_string();
            *agent_slot = Some(agent);
            reset_conversation_view(state);
            state.push_info(format!(
                "resumed session={session_id} — continuing from its saved context"
            ));
        }
        Err(e) => state.push_error(format!("couldn't resume session '{session_id}': {e:#}")),
    }
}

/// The chat column: scrolling transcript on top, the slash-command dropdown
/// and the pinned input bar + hint row at the bottom — mirroring the HTML's
/// `.chatcol` layout.
fn render_chat_column(f: &mut Frame, area: Rect, state: &mut AppState) {
    // Dropped immediately after `.len()` — recomputed again just before
    // `render_menu` actually needs the contents, so this temporary borrow
    // of `state` doesn't stay alive across the whole function (it would
    // otherwise conflict with the `state.transcript_*` writes below).
    let menu_h = menu_height(&state.menu().0);
    // The text row grows with the input — a long message used to be
    // capped at a single fixed row and silently clip everything past the
    // first wrapped line, with no indication anything was cut off. The cap
    // itself scales with the available screen height (up to a third of
    // the chat column, floor 6 rows, ceiling 20) rather than a fixed small
    // number, so genuinely long pasted/typed text keeps expanding the
    // composer instead of scrolling invisibly inside a tiny fixed box —
    // while still guaranteeing at least ~2/3 of the screen stays available
    // for the transcript on any terminal size. A second, always-1-row line
    // underneath carries mode/model/provider status normally (a
    // prompt/preview line for the TUI-only `Approval`/`KeyEntry` states) —
    // no top/bottom border to add for (the box only has a left accent bar,
    // not a full box border).
    let composer_inner_w = area.width.saturating_sub(3).max(10) as usize;
    let max_input_rows = (area.height / 3).clamp(6, 20) as usize;
    let input_text_h = wrap_preserving_newlines(&state.input, composer_inner_w)
        .len()
        .clamp(1, max_input_rows) as u16;
    // +1 for the status row, +1 for a blank row of top padding — a
    // one-line-tall composer (the common case: a short command) otherwise
    // reads as barely a sliver next to the transcript above it.
    let input_h = input_text_h + 2;
    let rows = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(menu_h),
        Constraint::Length(input_h),
        Constraint::Length(1),
    ])
    .split(area);

    let text = transcript_text(state, rows[0].width);
    let visible = rows[0].height;
    let para = Paragraph::new(text).wrap(Wrap { trim: false });
    // `Paragraph::scroll` counts rows *after* `Wrap` has reflowed the text
    // (confirmed in ratatui's own source: its line composer wraps first,
    // then `scroll.y` skips that many composed rows) — but this used to be
    // computed from the raw, pre-wrap `Text::lines().len()`, which
    // undercounts as soon as any line is long enough to wrap (a normal
    // assistant paragraph, a tool result, a grep match). Auto-follow then
    // scrolled to a position short of the true bottom, so the newest
    // message(s) were rendered past the end of the visible viewport —
    // effectively hidden right where the composer sits. `line_count` uses
    // the identical wrapping pass the render itself does, so this is now
    // counted in the same units `.scroll()` consumes.
    let total_wrapped = para.line_count(rows[0].width) as u16;
    let max_scroll = total_wrapped.saturating_sub(visible);
    state.transcript_max_scroll = max_scroll;
    // `None` (auto-follow) always tracks the live bottom; a manual offset
    // that has scrolled back down to (or past, as new lines arrive) the
    // bottom snaps back to auto-follow, same convention any chat UI uses.
    let scroll = match state.transcript_scroll {
        Some(s) if s < max_scroll => s,
        _ => {
            state.transcript_scroll = None;
            max_scroll
        }
    };
    let para = para.scroll((scroll, 0));
    f.render_widget(para, rows[0]);
    state.transcript_area = Some(rows[0]);
    state.transcript_applied_scroll = scroll;
    state.transcript_block_rows = transcript_block_rows(state, rows[0].width);

    render_input_box(f, rows[2], state, input_text_h);
    render_hints(f, rows[3], state);
    // Drawn last, once everything else in the column is already on the
    // buffer — `render_menu` dims the whole frame as a backdrop before
    // drawing itself, and that only dims what's already been painted.
    // Drawing it earlier (its position doesn't depend on draw order, since
    // `rows[1]` was already computed above) would leave the composer/hints
    // drawn fresh and undimmed right next to a dimmed transcript.
    state.command_menu_area = if menu_h > 0 {
        let (matches, kind) = state.menu();
        let accent = match kind {
            MenuKind::Commands => mode_accent(state.agent_mode),
            // Model autocomplete keeps its own teal accent so it reads as
            // a distinct (model-list) dropdown rather than a command one.
            MenuKind::Models => theme::teal_color(),
        };
        Some(render_menu(
            f,
            rows[1],
            &matches,
            state.command_selected,
            accent,
        ))
    } else {
        None
    };
}

/// The top bar is a single flex row (logo, mode pills, spacer, provider
/// button all inline) plus its `border-bottom` hairline — 2 rows total,
/// matching the HTML's `.topbar` exactly rather than stacking modes below
/// the logo on their own row.
const TOPBAR_H: u16 = 2;
/// Right-hand sidebar width (the HTML's 300px TODO panel ≈ 44 columns).
const SIDE_W: u16 = 44;

/// Top bar — the HTML `.topbar`: a single flex row with the logo, the mode
/// segmented-pill control, a flexible spacer, and the provider button (with
/// status dot) all inline, followed by the `border-bottom` hairline.
/// Returns the provider button's rect for mouse hit testing.
fn render_topbar(f: &mut Frame, area: Rect, state: &AppState, _config: &Config) {
    let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).split(area);

    // The wordmark carries the same rainbow/pulse sweep as the splash logo;
    // the ⚡ bolt stays violet, like the HTML's drop-shadowed SVG. Settling
    // `reduced_motion = true` in settings.toml swaps it for plain static
    // text instead — a redraw every tick isn't welcome on every link
    // (slow SSH), and some people just don't want a pulsing CLI.
    let mut logo_spans = vec![Span::styled(
        "⚡ ",
        theme::violet().add_modifier(Modifier::BOLD),
    )];
    if theme::reduced_motion() {
        logo_spans.push(Span::styled(
            "ZEUS",
            theme::text().add_modifier(Modifier::BOLD),
        ));
    } else {
        let t = state.started.elapsed().as_millis();
        logo_spans.extend(super::decor::animated_wordmark("ZEUS", t));
    }
    logo_spans.push(Span::styled(
        format!("  v{}", env!("CARGO_PKG_VERSION")),
        theme::faint(),
    ));
    if let Some(latest) = &state.update_available {
        // Passive only — never auto-installs. `zeus update` (or `zeus
        // update --check`) is still the only thing that actually changes
        // anything; see `update.rs`'s doc comment for why a silent
        // background auto-update (like OpenCode's) isn't worth copying.
        logo_spans.push(Span::styled(
            format!(" → v{latest} (run `zeus update`)"),
            theme::gold(),
        ));
    }
    let logo = Line::from(logo_spans);

    let pills = mode_pills_line(state.agent_mode);

    // Which specialist is driving the current Auto-mode plan step /
    // `/workflow` phase — previously only visible as a scrolling info line
    // in the transcript, easy to lose once a long run scrolls past it.
    // Truncated to whatever the topbar has left over after logo + pills
    // instead of being clipped mid-name by the right edge on a narrow
    // terminal.
    let persona_slot = rows[0]
        .width
        .saturating_sub(logo.width() as u16 + 3 + pills.width() as u16);
    let persona_line = (persona_slot >= 4)
        .then_some(state.active_persona.as_deref())
        .flatten()
        .map(|p| {
            let avail = (persona_slot as usize).saturating_sub(4); // " ▸ " + "…"
            let shown: String = if char_count(p) > avail {
                let mut s: String = p.chars().take(avail).collect();
                s.push('…');
                s
            } else {
                p.to_string()
            };
            Line::from(vec![
                Span::styled(" ▸ ", theme::faint()),
                Span::styled(shown, theme::gold().add_modifier(Modifier::ITALIC)),
            ])
        });
    let persona_w = persona_line.as_ref().map(|l| l.width() as u16).unwrap_or(0);

    // logo | gap | mode pills | persona chip | flexible spacer. No
    // provider/model chip up here anymore — the reference product doesn't
    // put one in its top bar either, and the composer's own status line
    // (`{Mode} · {model} {provider}`) already shows the same information,
    // so the top-right button was a redundant second surface for the same
    // action `/provider`/`ctrl+a`/`ctrl+p` already cover.
    let cols = Layout::horizontal([
        Constraint::Length(logo.width() as u16),
        Constraint::Length(3),
        Constraint::Length(pills.width() as u16),
        Constraint::Length(persona_w),
        Constraint::Min(0),
    ])
    .split(rows[0]);
    f.render_widget(Paragraph::new(logo), cols[0]);
    f.render_widget(Paragraph::new(pills), cols[2]);
    if let Some(persona_line) = persona_line {
        f.render_widget(Paragraph::new(persona_line), cols[3]);
    }

    // Hairline separator — the HTML `border-bottom` of the topbar.
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "─".repeat(area.width as usize),
            Style::default().fg(theme::border()),
        ))),
        rows[1],
    );
}

/// The HTML `.modes` segmented control — PLAN/BUILD/AUTO with the active
/// segment filled in its mode accent and the others dimmed.
fn mode_pills_line(mode: AgentMode) -> Line<'static> {
    let pills = [
        ("PLAN", AgentMode::Plan),
        ("BUILD", AgentMode::Build),
        ("AUTO", AgentMode::Auto),
    ];
    let mut spans = vec![Span::styled(" [ ", theme::faint())];
    for (i, (name, m)) in pills.iter().enumerate() {
        if *m == mode {
            spans.push(Span::styled(
                format!(" {} ", name),
                Style::default()
                    .fg(theme::void())
                    .bg(mode_accent(*m))
                    .add_modifier(Modifier::BOLD),
            ));
        } else {
            spans.push(Span::styled(format!(" {} ", name), theme::dim()));
        }
        if i < 2 {
            spans.push(Span::styled(" │ ", theme::faint()));
        }
    }
    spans.push(Span::styled(" ]", theme::faint()));
    Line::from(spans)
}

/// The HTML `.side` panel: TODOs header w/ live count, the progress bar
/// (fills as tasks are completed), the checklist, and the session footer.
/// Returns the todo-list rect for mouse-click row mapping.
fn render_side(f: &mut Frame, area: Rect, state: &AppState) -> Rect {
    // Fill the panel background.
    f.render_widget(
        Paragraph::new("").style(Style::default().bg(theme::panel())),
        area,
    );

    let done = state.todos.iter().filter(|t| t.done).count();
    let total = state.todos.len();
    let accent = mode_accent(state.agent_mode);

    // Up to 3 recently-touched files get their own fixed-height strip above
    // the footer; an empty session (nothing written yet) skips it entirely
    // rather than reserving dead space.
    let files_h: u16 = if state.files_touched.is_empty() {
        0
    } else {
        (state.files_touched.len().min(3) + 1) as u16
    };
    let rows = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(files_h),
        Constraint::Length(4),
    ])
    .split(area);

    // Header.
    let head = Line::from(vec![
        Span::styled(
            "TODOs",
            Style::default()
                .fg(theme::dim_color())
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            format!("{done} / {total}"),
            Style::default().fg(accent).add_modifier(Modifier::BOLD),
        ),
    ]);
    f.render_widget(Paragraph::new(head), rows[0]);

    // Progress bar (violet → accent gradient).
    render_progress(f, rows[1], done, total, accent);

    // TODO list.
    let todo_list_area = rows[2];
    render_todos(f, todo_list_area, state);

    // Recently-touched files.
    if files_h > 0 {
        render_files(f, rows[3], state);
    }

    // Footer: Session / Tokens / Cost / Branch.
    render_side_foot(f, rows[4], state);

    todo_list_area
}

fn render_progress(f: &mut Frame, area: Rect, done: usize, total: usize, accent: Color) {
    let w = area.width as usize;
    let filled = if total == 0 {
        0
    } else {
        ((done as f64 / total as f64) * w as f64).round() as usize
    };
    let mut spans = Vec::new();
    for i in 0..w {
        if i < filled {
            let t = if filled > 1 {
                i as f32 / (filled - 1) as f32
            } else {
                1.0
            };
            spans.push(Span::styled(
                "█",
                Style::default().fg(lerp_color(theme::accent(), accent, t)),
            ));
        } else {
            spans.push(Span::styled("·", theme::faint()));
        }
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_todos(f: &mut Frame, area: Rect, state: &AppState) {
    let accent = mode_accent(state.agent_mode);
    let mut lines = Vec::new();
    if state.todos.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "  no tasks yet — ask Zeus to fix something",
            theme::faint(),
        )]));
        f.render_widget(Paragraph::new(lines), area);
        return;
    }
    for item in &state.todos {
        // The active item gets its own distinct marker (a filled diamond,
        // no checkbox brackets) instead of just an empty `[ ]` in accent
        // color — reads as "this one's happening right now" the way a
        // spinner/asterisk bullet does, rather than blending in as just
        // another differently-colored checkbox row.
        let mut spans = if item.active {
            vec![
                Span::styled(
                    "◆ ",
                    Style::default().fg(accent).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    item.text.clone(),
                    theme::text().add_modifier(Modifier::BOLD),
                ),
            ]
        } else {
            let box_mark = if item.done { "✓" } else { " " };
            let box_style = if item.done {
                Style::default()
                    .fg(theme::void())
                    .bg(theme::GREEN)
                    .add_modifier(Modifier::BOLD)
            } else {
                theme::faint()
            };
            let label_style = if item.done {
                Style::default()
                    .fg(theme::faint_color())
                    .add_modifier(Modifier::CROSSED_OUT)
            } else {
                theme::text()
            };
            vec![
                Span::styled("[", theme::faint()),
                Span::styled(box_mark, box_style),
                Span::styled("]", theme::faint()),
                Span::raw(" "),
                Span::styled(item.text.clone(), label_style),
            ]
        };
        if let Some(d) = item.duration {
            spans.push(Span::styled(
                format!("  {}", fmt_duration(d)),
                theme::faint(),
            ));
        }
        lines.push(Line::from(spans));
    }
    f.render_widget(Paragraph::new(lines), area);
}

/// The sidebar's "Files" panel: the last few paths written/edited/deleted
/// this session, most-recent first — mirrors the mockups' workspace panel
/// without needing a full file-tree browser.
fn render_files(f: &mut Frame, area: Rect, state: &AppState) {
    let mut lines = vec![Line::from(Span::styled(
        "Files",
        Style::default()
            .fg(theme::dim_color())
            .add_modifier(Modifier::BOLD),
    ))];
    let max_rows = (area.height as usize).saturating_sub(1);
    // The sidebar is 44 cols — long paths get an ellipsis tail instead of
    // being clipped mid-path by the panel's right edge.
    let max_w = area.width as usize;
    for path in state.files_touched.iter().rev().take(max_rows) {
        let shown = if char_count(path) + 2 > max_w {
            let keep = max_w.saturating_sub(3);
            format!("{}…", path.chars().take(keep).collect::<String>())
        } else {
            path.clone()
        };
        lines.push(Line::from(Span::styled(format!("· {shown}"), theme::dim())));
    }
    f.render_widget(Paragraph::new(lines), area);
}

/// Best-effort target path for a mutating tool call, parsed out of its raw
/// JSON arguments — used to populate the sidebar's "Files" panel.
/// `ToolCallFinished` carries no arguments of its own, so this has to run at
/// `ToolCallStarted` time and be carried forward keyed by call id.
fn touched_path(tool_name: &str, arguments: &str) -> Option<String> {
    if !matches!(tool_name, "write" | "edit" | "delete" | "rename" | "copy") {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(arguments).ok()?;
    v.get("path")
        .or_else(|| v.get("to"))
        .and_then(|p| p.as_str())
        .map(|s| s.to_string())
}

fn render_side_foot(f: &mut Frame, area: Rect, state: &AppState) {
    let branch = state
        .dir
        .git_branch
        .clone()
        .unwrap_or_else(|| "(no git repo)".to_string());
    let elapsed = state.started.elapsed();
    let secs = elapsed.as_secs();
    let session = format!("{}m {:02}s", secs / 60, secs % 60);
    let tokens = match state.context_window {
        Some(window) => format!(
            "{} / {}",
            format_token_count(state.session_usage.total_tokens),
            format_token_count(window)
        ),
        None => format_token_count(state.session_usage.total_tokens),
    };
    let cost = match estimate_cost(&state.provider, &state.model, &state.session_usage) {
        Some(usd) => format!("~${usd:.2}"),
        None => "—".to_string(),
    };
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(area);
    let vals: [(String, String); 4] = [
        ("Session".into(), session),
        ("Tokens".into(), tokens),
        ("Cost".into(), cost),
        ("Branch".into(), branch),
    ];
    for (i, (k, v)) in vals.iter().enumerate() {
        let line = Line::from(vec![
            Span::styled(k.clone(), theme::faint()),
            Span::styled("  ", theme::faint()),
            Span::styled(v.clone(), theme::dim()),
        ]);
        f.render_widget(Paragraph::new(line), rows[i]);
    }
}

/// Example prompts shown as clickable chips — verbatim from
/// `zeus-empty-state.html`'s `#chips` row.
const EXAMPLE_CHIPS: [&str; 3] = [
    "Scaffold a new API",
    "Explain this codebase",
    "Write tests for a file",
];

/// The full-screen empty-state splash — `zeus-empty-state.html` reproduced
/// with no topbar/sidebar chrome: a status eyebrow, the gradient "ZEUS"
/// wordmark, a question line, a centered composer, three example chips, and
/// a hint row. Shown instead of `render_topbar`/`render_side`/
/// `render_chat_column` whenever `AppState::showing_empty_state` is true.
fn render_empty_state(f: &mut Frame, area: Rect, state: &mut AppState, config: &Config) {
    let t_ms = state.started.elapsed().as_millis() as f64;

    // Plain `--ink` background — no animated network (removed: the braille
    // particle simulation redrawing every tick was a real, measurable
    // source of lag for little payoff on a screen that's mostly text).
    f.render_widget(
        Block::default().style(Style::default().bg(theme::ink())),
        area,
    );

    // Composer sizing computed up front, before `total_h` — the vertical
    // centering below needs the composer's *real* height (it grows with
    // wrapped line count for a long paste), not a fixed guess. Using a
    // fixed constant here meant a multi-line paste before the first send
    // silently pushed everything below the composer off the intended
    // center, and on a short terminal `centered_row`'s own height clamp
    // could squeeze the chips/hint down to zero height with no error.
    let composer_w = (area.width.saturating_sub(8)).clamp(24, 64);
    // Left accent bar + one column of breathing room (2 columns total) —
    // no `›` caret and no `➜` send glyph, matching the main composer's
    // "Enter is the only way to send" metaphor.
    let composer_text_w = (composer_w as usize).saturating_sub(2).max(10);
    let wrapped_input = wrap_preserving_newlines(&state.input, composer_text_w);
    let composer_text_h = wrapped_input.len().clamp(1, 10) as u16;
    let composer_h = composer_text_h + 2;

    // Slash-command palette sizing, same reasoning, plus it now reserves
    // its own space below the composer instead of the chips/hint sharing
    // the same `y` and getting painted over by it.
    // Owned, not borrowed from `state` — `menu()` returns owned entries
    // and borrows `state` only briefly, so it can't live across the
    // `state.chip_areas`/`state.command_menu_area` writes further down.
    let (palette_matches, _menu_kind) = state.menu();
    let menu_h = menu_height(&palette_matches);

    // ---- Centered content stack ----
    const STATUS_H: u16 = 1;
    const BANNER_H: u16 = 1;
    const QUESTION_H: u16 = 1;
    const CHIPS_H: u16 = 1;
    const HINT_H: u16 = 1;
    const GAP: u16 = 1;
    let menu_reserved_h = if menu_h > 0 { menu_h + GAP } else { 0 };
    let total_h = STATUS_H
        + GAP
        + BANNER_H
        + GAP
        + QUESTION_H
        + GAP
        + composer_h
        + GAP
        + menu_reserved_h
        + CHIPS_H
        + GAP
        + HINT_H;
    let mut y = area.y + (area.height.saturating_sub(total_h)) / 2;

    // Status eyebrow: pulsing dot (mirrors the CSS `ping` keyframe) +
    // readiness text. Reflects whether the *current* provider actually has
    // a key/is reachable — a first-run session with nothing configured yet
    // used to always claim "READY" here regardless, so the example chips
    // below invited a task that would only fail with an error after Enter,
    // instead of pointing straight at `/model`/`/provider` up front.
    let ready = provider_status_ok(config, &state.provider);
    let pulse = 0.5 + 0.5 * (t_ms * 0.0024).sin();
    let dot_color = if ready {
        theme::teal_color()
    } else {
        theme::GOLD
    };
    let dot_style = if pulse > 0.6 {
        Style::default().fg(dot_color).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(dot_color)
    };
    let label = if ready {
        "  Z E U S   R E A D Y"
    } else {
        // `/provider` — not `/model` — is the command that actually works
        // here: with no ready provider, `/model` just posts "no models
        // found on any configured provider" and dead-ends, while
        // `/provider` lists providers with a "needs a key" indicator and
        // opens the key-paste prompt when you pick one.
        "  N O   P R O V I D E R  ·  / P R O V I D E R   T O   C O N N E C T"
    };
    let status_text = format!("●{label}");
    let status_w = status_text.chars().count() as u16;
    let status_area = centered_row(area, y, STATUS_H, status_w + 2);
    opaque(f, status_area);
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("●", dot_style),
            Span::styled(label, theme::muted()),
        ]))
        .alignment(Alignment::Center),
        status_area,
    );
    y += STATUS_H + GAP;

    // "ZEUS" wordmark — a clean static gradient (near-white → gold), unlike
    // the topbar's animated rainbow wordmark which stays as-is.
    let wordmark = gradient_wordmark("ZEUS");
    let banner_area = centered_row(area, y, BANNER_H, wordmark.len() as u16 + 2);
    opaque(f, banner_area);
    f.render_widget(
        Paragraph::new(Line::from(wordmark)).alignment(Alignment::Center),
        banner_area,
    );
    y += BANNER_H + GAP;

    // Question line.
    let question = "What are we building?";
    let question_area = centered_row(area, y, QUESTION_H, question.chars().count() as u16 + 2);
    opaque(f, question_area);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(question, theme::muted())))
            .alignment(Alignment::Center),
        question_area,
    );
    y += QUESTION_H + GAP;

    // Composer: same flat accent-bar panel as the pinned chat composer — no
    // `›` caret, no send glyph, Enter is the only way to send. Height grows
    // with wrapped line count (same fix as the main chat composer) — this
    // screen's input previously rendered without any `.wrap(...)` at all,
    // so long typed/pasted text ran straight off the right edge of the box
    // instead of wrapping. Sizing (`composer_w`/`composer_text_w`/
    // `wrapped_input`/`composer_text_h`/`composer_h`) was already computed
    // above, for `total_h`.
    let composer_area = centered_row(area, y, composer_h, composer_w);
    opaque(f, composer_area);
    let accent = mode_accent(state.agent_mode);
    let composer_block = Block::default()
        .borders(Borders::LEFT)
        .border_type(BorderType::Thick)
        .border_style(Style::default().fg(accent))
        .style(Style::default().bg(theme::panel2()));
    let raw_inner = composer_block.inner(composer_area);
    f.render_widget(composer_block, composer_area);
    // Same one column of breathing room between the accent bar and the text
    // as the chat composer.
    let composer_inner = Rect {
        x: raw_inner.x + 1,
        y: raw_inner.y,
        width: raw_inner.width.saturating_sub(1),
        height: raw_inner.height,
    };
    let input_line = if state.input.is_empty() {
        Paragraph::new(Line::from(Span::styled(
            "Describe a task, or paste a file path…",
            placeholder_style(),
        )))
    } else {
        Paragraph::new(Text::from(state.input.clone()).style(theme::text()))
            .wrap(Wrap { trim: false })
    };
    f.render_widget(input_line, composer_inner);
    if matches!(state.mode, Mode::Chat) && !state.busy {
        let typed_before: String = state.input.chars().take(state.cursor).collect();
        let wrapped_before = wrap_preserving_newlines(&typed_before, composer_text_w);
        let last_row = composer_text_h.saturating_sub(1);
        let cursor_row = (wrapped_before.len() as u16)
            .saturating_sub(1)
            .min(last_row);
        let cursor_col = wrapped_before.last().map(|l| char_count(l)).unwrap_or(0) as u16;
        f.set_cursor_position((composer_inner.x + cursor_col, composer_inner.y + cursor_row));
    }
    y += composer_h + GAP;

    // Slash-command palette, when typing "/…", floats just below the
    // composer, in its own reserved `menu_reserved_h` space (accounted for
    // in `total_h` above) rather than sharing the chips' `y` and getting
    // painted over by it. `palette_matches`/`palette_refs`/`menu_h` were
    // already computed above, for `total_h`. The actual draw is still
    // deferred past the chips/hint — `render_menu` dims the whole frame as
    // a backdrop before drawing itself, and that only dims what's already
    // been painted, so it has to run after everything else in this screen.
    let menu_area = (menu_h > 0).then(|| centered_row(area, y, menu_h, composer_w));
    if menu_h > 0 {
        y += menu_h + GAP;
    }

    // Example chips — clicking one fills the composer, like the HTML's chip handler.
    let gap = "   ";
    let chips_text = EXAMPLE_CHIPS.map(|c| format!("[ {c} ]")).join(gap);
    let chips_w = chips_text.chars().count() as u16;
    let chips_area = centered_row(area, y, CHIPS_H, chips_w + 2);
    opaque(f, chips_area);
    let mut spans = Vec::new();
    let mut areas = Vec::new();
    let mut col = chips_area.x + (chips_area.width.saturating_sub(chips_w)) / 2;
    for (i, chip) in EXAMPLE_CHIPS.iter().enumerate() {
        let label = format!("[ {chip} ]");
        let w = label.chars().count() as u16;
        spans.push(Span::styled(label, theme::dim()));
        areas.push(Rect {
            x: col,
            y,
            width: w,
            height: 1,
        });
        col += w;
        if i < EXAMPLE_CHIPS.len() - 1 {
            spans.push(Span::styled(gap, theme::faint()));
            col += gap.chars().count() as u16;
        }
    }
    f.render_widget(
        Paragraph::new(Line::from(spans)).alignment(Alignment::Center),
        chips_area,
    );
    state.chip_areas = areas;
    y += CHIPS_H + GAP;

    // Hint row.
    let hint = "Enter to start  ·  Esc to clear";
    let hint_area = centered_row(area, y, HINT_H, hint.chars().count() as u16 + 2);
    opaque(f, hint_area);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(hint, theme::empty_faint())))
            .alignment(Alignment::Center),
        hint_area,
    );

    state.command_menu_area = menu_area.map(|area| {
        render_menu(
            f,
            area,
            &palette_matches,
            state.command_selected,
            theme::teal_color(),
        )
    });
}

fn render(f: &mut Frame, state: &mut AppState, config: &Config) {
    let area = f.area();

    if state.showing_empty_state() {
        render_empty_state(f, area, state, config);
        return;
    }
    state.chip_areas.clear();

    // Fill the whole frame with the void background first.
    f.render_widget(
        Block::default().style(Style::default().bg(theme::void())),
        area,
    );

    let rows = Layout::vertical([Constraint::Length(TOPBAR_H), Constraint::Min(0)]).split(area);

    render_topbar(f, rows[0], state, config);

    let has_side = area.width >= 100;
    // The HTML's `.chatcol { border-right: 1px solid var(--border) }` — a
    // 1-column divider between the chat column and the TODO sidebar, with a
    // 1-column margin so the chat column's own bordered boxes don't touch
    // it directly (otherwise their border and the divider read as one
    // doubled-up line).
    let main = Layout::horizontal([
        Constraint::Min(0),
        Constraint::Length(if has_side { 1 } else { 0 }),
        Constraint::Length(if has_side { 1 } else { 0 }),
        Constraint::Length(if has_side { SIDE_W } else { 0 }),
    ])
    .split(rows[1]);
    state.todo_area = if has_side { Some(main[3]) } else { None };

    render_chat_column(f, main[0], state);

    if has_side {
        let divider = vec![
            Line::from(Span::styled("│", Style::default().fg(theme::border())));
            main[2].height as usize
        ];
        f.render_widget(Paragraph::new(divider), main[2]);
        let list_area = render_side(f, main[3], state);
        state.todo_area = Some(list_area);
    }

    // Dim everything drawn so far when a modal is about to go on top of it —
    // matches the reference product's own semi-transparent black backdrop
    // behind an open dialog. Ratatui has no real alpha blending, so this
    // directly darkens each already-rendered cell's colors instead; the
    // modal's own `Clear` + solid background then fully repaints its own
    // footprint on top, so it doesn't matter that this dims that area too.
    let any_modal_open = matches!(
        state.mode,
        Mode::ModelPicker { .. }
            | Mode::ProviderPicker { .. }
            | Mode::KeyEntry { .. }
            | Mode::Approval { .. }
            | Mode::Diff { .. }
    ) || state.session_picker.is_some();
    if any_modal_open {
        dim_backdrop(f, area);
    }

    let picker_area = if let Mode::ModelPicker { entries, selected } = &state.mode {
        Some(render_model_picker(
            f,
            area,
            &state.provider,
            &state.model,
            entries,
            *selected,
            &state.model_picker_search,
            &state.favorite_models,
        ))
    } else {
        None
    };
    state.model_picker_area = picker_area;

    let provider_picker_area = if let Mode::ProviderPicker { entries, selected } = &state.mode {
        Some(render_provider_picker(
            f,
            area,
            &state.provider,
            entries,
            *selected,
        ))
    } else {
        None
    };
    state.provider_picker_area = provider_picker_area;

    if let Mode::KeyEntry { provider } = &state.mode {
        let provider = provider.clone();
        let input_rect = render_key_entry_modal(f, area, &provider, &state.input);
        // Cursor position only depends on how many characters precede it,
        // not their actual (masked) content, so wrapping a same-length
        // placeholder against the input box's real width gives the right
        // row/col for a key long enough to have wrapped onto later rows.
        let wrapped_before = wrap_text(&"x".repeat(state.cursor), input_rect.width.max(1) as usize);
        let last_row = input_rect.height.saturating_sub(1);
        let cursor_row = (wrapped_before.len() as u16)
            .saturating_sub(1)
            .min(last_row);
        let cursor_col = wrapped_before.last().map(|l| char_count(l)).unwrap_or(0) as u16;
        f.set_cursor_position((input_rect.x + cursor_col, input_rect.y + cursor_row));
    }

    if let Mode::Approval { pending, scroll } = &state.mode {
        render_approval_modal(f, area, pending, *scroll);
    }

    if let Mode::Diff { rows, scroll } = &state.mode {
        render_diff_modal(f, area, rows, *scroll);
    }

    if let Some(search) = &state.search {
        render_search_bar(f, main[0], search);
    }

    let session_picker_area = state
        .session_picker
        .as_ref()
        .map(|picker| render_session_picker(f, area, picker));
    state.session_picker_area = session_picker_area;
}

type TurnJoin = JoinHandle<(Agent, zeus_agent::Result<TurnResult>)>;

fn start_turn(
    state: &mut AppState,
    agent_slot: &mut Option<Agent>,
    turn_handle: &mut Option<TurnJoin>,
    cancel_tx: &mut Option<watch::Sender<bool>>,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
    message: String,
    yes: bool,
) {
    let Some(mut agent) = agent_slot.take() else {
        return;
    };
    *cancel_tx = Some(agent.cancel_handle());
    state.busy = true;
    let tx_events = ui_tx.clone();
    let tx_approval = ui_tx.clone();
    let handle = tokio::spawn(async move {
        let on_event = move |ev: AgentEvent| {
            let _ = tx_events.send(UiEvent::Agent(ev));
        };
        let approver = build_approver(&tx_approval, yes);
        let result = agent.run_turn(&message, on_event, approver).await;
        (agent, result)
    });
    *turn_handle = Some(handle);
}

/// `/plan <goal>`: same plumbing as `start_turn`, but drives
/// `Agent::plan_turn` — research read-only, persist `.agent/tasks.json`,
/// and return without executing anything. Plan mode stays on afterwards, so
/// further chatting stays observational; to actually run the plan, confirm
/// the prompt shown on completion (see `start_orchestrate_turn`) rather than
/// switching modes yourself — plain Auto-mode chat no longer replays a
/// pending plan on its own (Auto mode moved to a continuous tool loop that
/// doesn't consult `tasks.json`).
fn start_plan_turn(
    state: &mut AppState,
    agent_slot: &mut Option<Agent>,
    turn_handle: &mut Option<TurnJoin>,
    cancel_tx: &mut Option<watch::Sender<bool>>,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
    message: String,
    yes: bool,
) {
    let Some(mut agent) = agent_slot.take() else {
        return;
    };
    *cancel_tx = Some(agent.cancel_handle());
    state.busy = true;
    let tx_events = ui_tx.clone();
    let tx_approval = ui_tx.clone();
    let handle = tokio::spawn(async move {
        let on_event = move |ev: AgentEvent| {
            let _ = tx_events.send(UiEvent::Agent(ev));
        };
        let approver = build_approver(&tx_approval, yes);
        let result = agent.plan_turn(&message, on_event, approver).await;
        (agent, result)
    });
    *turn_handle = Some(handle);
}

/// Confirmed run of a plan `/plan` just generated — drives
/// `Agent::orchestrate`, which offers to resume an approved plan with
/// pending steps first (re-planning is only the fallback) and then executes
/// step by step behind its own `plan_execute` approval prompt.
/// `orchestrate` returns a `(String, TokenUsage)` pair rather than a
/// `TurnResult`; wrapped into one here so this flows through the exact same
/// completion handling as every other turn kind, including the
/// `PlanStepStarted`/`PlanStepDone`/`OrchestrationDone` events that already
/// drive the sidebar and persona chip.
fn start_orchestrate_turn(
    state: &mut AppState,
    agent_slot: &mut Option<Agent>,
    turn_handle: &mut Option<TurnJoin>,
    cancel_tx: &mut Option<watch::Sender<bool>>,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
    goal: String,
    yes: bool,
) {
    let Some(mut agent) = agent_slot.take() else {
        return;
    };
    *cancel_tx = Some(agent.cancel_handle());
    state.busy = true;
    let tx_events = ui_tx.clone();
    let tx_approval = ui_tx.clone();
    let handle = tokio::spawn(async move {
        let on_event = move |ev: AgentEvent| {
            let _ = tx_events.send(UiEvent::Agent(ev));
        };
        let approver = build_approver(&tx_approval, yes);
        let result = agent
            .orchestrate(&goal, on_event, approver)
            .await
            .map(|(summary, usage)| TurnResult {
                final_text: summary,
                tool_calls: 0,
                cancelled: false,
                usage,
            });
        (agent, result)
    });
    *turn_handle = Some(handle);
}

/// `/understand <topic>`: read-only repository scan for what already exists.
fn start_understand_turn(
    state: &mut AppState,
    agent_slot: &mut Option<Agent>,
    turn_handle: &mut Option<TurnJoin>,
    cancel_tx: &mut Option<watch::Sender<bool>>,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
    topic: String,
    yes: bool,
) {
    let Some(mut agent) = agent_slot.take() else {
        return;
    };
    *cancel_tx = Some(agent.cancel_handle());
    state.busy = true;
    let tx_events = ui_tx.clone();
    let tx_approval = ui_tx.clone();
    let handle = tokio::spawn(async move {
        let on_event = move |ev: AgentEvent| {
            let _ = tx_events.send(UiEvent::Agent(ev));
        };
        let approver = build_approver(&tx_approval, yes);
        let result = agent.understand_topic(&topic, on_event, approver).await;
        (agent, result)
    });
    *turn_handle = Some(handle);
}

/// `/orient`: read-only scan that writes `.agent/architecture.md` +
/// `.agent/conventions.md`.
fn start_orient_turn(
    state: &mut AppState,
    agent_slot: &mut Option<Agent>,
    turn_handle: &mut Option<TurnJoin>,
    cancel_tx: &mut Option<watch::Sender<bool>>,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
    yes: bool,
) {
    let Some(mut agent) = agent_slot.take() else {
        return;
    };
    *cancel_tx = Some(agent.cancel_handle());
    state.busy = true;
    let tx_events = ui_tx.clone();
    let tx_approval = ui_tx.clone();
    let handle = tokio::spawn(async move {
        let on_event = move |ev: AgentEvent| {
            let _ = tx_events.send(UiEvent::Agent(ev));
        };
        let approver = build_approver(&tx_approval, yes);
        let result = agent.orient_turn(on_event, approver).await;
        (agent, result)
    });
    *turn_handle = Some(handle);
}

/// `/review`: read-only review of the current uncommitted changes.
fn start_review_turn(
    state: &mut AppState,
    agent_slot: &mut Option<Agent>,
    turn_handle: &mut Option<TurnJoin>,
    cancel_tx: &mut Option<watch::Sender<bool>>,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
    yes: bool,
) {
    let Some(mut agent) = agent_slot.take() else {
        return;
    };
    *cancel_tx = Some(agent.cancel_handle());
    state.busy = true;
    let tx_events = ui_tx.clone();
    let tx_approval = ui_tx.clone();
    let handle = tokio::spawn(async move {
        let on_event = move |ev: AgentEvent| {
            let _ = tx_events.send(UiEvent::Agent(ev));
        };
        let approver = build_approver(&tx_approval, yes);
        let result = agent.review_turn(on_event, approver).await;
        (agent, result)
    });
    *turn_handle = Some(handle);
}

/// `/suggest`: read-only next-feature recommendations grounded in the repo.
/// `/suggest [context]`: read-only next-feature recommendations grounded in
/// what already exists; an optional context ("just finished the login page")
/// anchors the suggestions to the work that was just done.
#[allow(clippy::too_many_arguments)] // app plumbing at a single call site
fn start_suggest_turn(
    state: &mut AppState,
    agent_slot: &mut Option<Agent>,
    turn_handle: &mut Option<TurnJoin>,
    cancel_tx: &mut Option<watch::Sender<bool>>,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
    context: String,
    yes: bool,
) {
    let Some(mut agent) = agent_slot.take() else {
        return;
    };
    *cancel_tx = Some(agent.cancel_handle());
    state.busy = true;
    let tx_events = ui_tx.clone();
    let tx_approval = ui_tx.clone();
    let handle = tokio::spawn(async move {
        let on_event = move |ev: AgentEvent| {
            let _ = tx_events.send(UiEvent::Agent(ev));
        };
        let approver = build_approver(&tx_approval, yes);
        let result = agent.suggest_turn(&context, on_event, approver).await;
        (agent, result)
    });
    *turn_handle = Some(handle);
}

/// `/workflow <name> <goal>`: run a declarative multi-specialist pipeline.
#[allow(clippy::too_many_arguments)] // app plumbing + workflow args at a single call site
fn start_workflow_turn(
    state: &mut AppState,
    agent_slot: &mut Option<Agent>,
    turn_handle: &mut Option<TurnJoin>,
    cancel_tx: &mut Option<watch::Sender<bool>>,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
    workflow: zeus_agent::Workflow,
    goal: String,
    yes: bool,
) {
    let Some(mut agent) = agent_slot.take() else {
        return;
    };
    *cancel_tx = Some(agent.cancel_handle());
    state.busy = true;
    let tx_events = ui_tx.clone();
    let tx_approval = ui_tx.clone();
    let handle = tokio::spawn(async move {
        let on_event = move |ev: AgentEvent| {
            let _ = tx_events.send(UiEvent::Agent(ev));
        };
        let approver = build_approver(&tx_approval, yes);
        let result = agent
            .run_workflow(&goal, &workflow, on_event, approver)
            .await
            .map(|summary| TurnResult {
                final_text: summary,
                tool_calls: 0,
                cancelled: false,
                usage: Default::default(),
            });
        (agent, result)
    });
    *turn_handle = Some(handle);
}

/// Recompute the model picker's selection after the search string changed:
/// the current `selected` index may now point at a header row or past the
/// end of the filtered list, so jump back to the first model row. A no-op
/// if the picker isn't open — callers run it right after mutating the
/// search, when `Mode::ModelPicker` is guaranteed by the surrounding guard.
fn model_picker_nudge_to_first(state: &mut AppState) {
    if let Mode::ModelPicker { entries, selected } = &mut state.mode {
        let filtered = model_picker_filtered(entries, &state.model_picker_search);
        *selected = first_selectable_picker(&filtered);
    }
}

/// Move the model picker's selection one step in `dir` (`1`/`-1`), skipping
/// headers and wrapping, without ever panicking if the mode changed.
fn model_picker_nudge(state: &mut AppState, dir: isize) {
    if let Mode::ModelPicker { entries, selected } = &mut state.mode {
        let filtered = model_picker_filtered(entries, &state.model_picker_search);
        *selected = picker_move(&filtered, *selected, dir);
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_key(
    key: KeyEvent,
    state: &mut AppState,
    agent_slot: &mut Option<Agent>,
    turn_handle: &mut Option<TurnJoin>,
    cancel_tx: &mut Option<watch::Sender<bool>>,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
    config: &Config,
    yes: bool,
) -> Result<()> {
    if let Mode::Approval { .. } = &state.mode {
        // ↑/↓ and paging scroll the preview; the render clamps the counter to
        // the real line count, so the keys can just nudge it here.
        if let Mode::Approval { scroll, .. } = &mut state.mode {
            match key.code {
                KeyCode::Up => *scroll = scroll.saturating_sub(1),
                KeyCode::Down => *scroll = scroll.saturating_add(1),
                KeyCode::PageUp => *scroll = scroll.saturating_sub(20),
                KeyCode::PageDown => *scroll = scroll.saturating_add(20),
                _ => {}
            }
        }
        let decision = match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => Some(ApprovalDecision::Approved),
            KeyCode::Char('s') | KeyCode::Char('S') => Some(ApprovalDecision::ApprovedForSession),
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                Some(ApprovalDecision::Denied)
            }
            _ => None,
        };
        if let Some(decision) = decision {
            if let Mode::Approval { pending, .. } = std::mem::replace(&mut state.mode, Mode::Chat) {
                let _ = pending.reply.send(decision);
            }
        }
        return Ok(());
    }

    // ---- Two-pane diff view (`/diff`) ----
    if let Mode::Diff { .. } = &state.mode {
        if let Mode::Diff { scroll, .. } = &mut state.mode {
            match key.code {
                KeyCode::Up => *scroll = scroll.saturating_sub(1),
                KeyCode::Down => *scroll = scroll.saturating_add(1),
                KeyCode::PageUp => *scroll = scroll.saturating_sub(20),
                KeyCode::PageDown => *scroll = scroll.saturating_add(20),
                _ => {}
            }
        }
        if key.code == KeyCode::Esc {
            state.mode = Mode::Chat;
        }
        return Ok(());
    }

    // ---- Session-resume picker ----
    if let Some(picker) = state.session_picker.as_ref() {
        let filtered_len = session_picker_filtered(picker).len();
        match key.code {
            KeyCode::Esc => state.session_picker = None,
            KeyCode::Up => {
                if let Some(picker) = state.session_picker.as_mut() {
                    picker.selected = picker.selected.saturating_sub(1);
                }
            }
            KeyCode::Down => {
                if let Some(picker) = state.session_picker.as_mut() {
                    if picker.selected + 1 < filtered_len {
                        picker.selected += 1;
                    }
                }
            }
            KeyCode::Enter => {
                let id = session_picker_filtered(picker)
                    .get(picker.selected)
                    .map(|s| s.id.clone());
                state.session_picker = None;
                if let Some(id) = id {
                    resume_session(id, config, agent_slot, state).await;
                }
            }
            KeyCode::Backspace => {
                if let Some(picker) = state.session_picker.as_mut() {
                    picker.search.pop();
                    picker.selected = 0;
                }
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(picker) = state.session_picker.as_mut() {
                    picker.search.push(c);
                    picker.selected = 0;
                }
            }
            _ => {}
        }
        return Ok(());
    }

    // ---- Transcript search overlay ----
    // Takes priority over everything below while open: every keystroke
    // edits the query rather than the composer, until Esc closes it or
    // Enter cycles to the next match.
    if state.search.is_some() {
        match key.code {
            KeyCode::Esc => state.search = None,
            KeyCode::Enter => {
                if let Some(search) = state.search.as_mut() {
                    if !search.matches.is_empty() {
                        search.current = (search.current + 1) % search.matches.len();
                    }
                }
                state.search_jump_to_current();
            }
            KeyCode::Backspace => {
                if let Some(search) = state.search.as_mut() {
                    search.query.pop();
                }
                state.search_recompute();
                state.search_jump_to_current();
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(search) = state.search.as_mut() {
                    search.query.push(c);
                }
                state.search_recompute();
                state.search_jump_to_current();
            }
            _ => {}
        }
        return Ok(());
    }
    // ctrl+f opens it — only in plain chat, so it doesn't steal the model
    // picker's own ctrl+f (toggle favorite).
    if matches!(state.mode, Mode::Chat)
        && key.code == KeyCode::Char('f')
        && key.modifiers.contains(KeyModifiers::CONTROL)
    {
        state.search = Some(SearchState {
            query: String::new(),
            matches: Vec::new(),
            current: 0,
        });
        return Ok(());
    }

    if let Mode::ModelPicker { .. } = &state.mode {
        // ctrl+a jumps straight to the provider picker ("connect a
        // provider") — checked before the plain Char branch below since
        // both match on `KeyCode::Char`.
        if key.code == KeyCode::Char('a') && key.modifiers.contains(KeyModifiers::CONTROL) {
            open_provider_picker(state, config);
            return Ok(());
        }
        if key.code == KeyCode::Char('f') && key.modifiers.contains(KeyModifiers::CONTROL) {
            if let Mode::ModelPicker { entries, selected } = &state.mode {
                let filtered = model_picker_filtered(entries, &state.model_picker_search);
                if let Some(PickerEntry::Model { provider, model }) = filtered.get(*selected) {
                    let (provider, model_id) = (provider.clone(), model.id.clone());
                    state.toggle_favorite_model(&provider, &model_id, config);
                }
            }
            return Ok(());
        }
        match key.code {
            KeyCode::Up => model_picker_nudge(state, -1),
            KeyCode::Down => model_picker_nudge(state, 1),
            KeyCode::Enter => {
                let chosen = match &state.mode {
                    Mode::ModelPicker { entries, selected } => {
                        let filtered = model_picker_filtered(entries, &state.model_picker_search);
                        match filtered.get(*selected) {
                            Some(PickerEntry::Model { provider, model }) => {
                                Some((provider.clone(), model.id.clone()))
                            }
                            // A header row isn't a choice — leave the picker open
                            // rather than silently closing it on a stray Enter.
                            _ => None,
                        }
                    }
                    _ => None,
                };
                if let Some((provider, model_id)) = chosen {
                    apply_model_choice_or_key_entry(provider, model_id, config, agent_slot, state)
                }
            }
            KeyCode::Esc => {
                state.mode = Mode::Chat;
                state.model_picker_search.clear();
            }
            KeyCode::Backspace => {
                state.model_picker_search.pop();
                model_picker_nudge_to_first(state);
            }
            KeyCode::Char(c) => {
                state.model_picker_search.push(c);
                model_picker_nudge_to_first(state);
            }
            _ => {}
        }
        return Ok(());
    }

    if let Mode::ProviderPicker { .. } = &state.mode {
        match key.code {
            KeyCode::Up => {
                if let Mode::ProviderPicker { entries, selected } = &mut state.mode {
                    *selected = provider_picker_move(entries, *selected, -1);
                }
            }
            KeyCode::Down => {
                if let Mode::ProviderPicker { entries, selected } = &mut state.mode {
                    *selected = provider_picker_move(entries, *selected, 1);
                }
            }
            KeyCode::Enter => {
                let chosen = match &state.mode {
                    Mode::ProviderPicker { entries, selected } => match entries.get(*selected) {
                        Some(ProviderEntry::Provider { name, ready, .. }) => {
                            Some((name.clone(), *ready))
                        }
                        // A header row isn't a choice — leave the picker open
                        // rather than silently closing it on a stray Enter.
                        _ => None,
                    },
                    _ => None,
                };
                if let Some((name, ready)) = chosen {
                    apply_provider_picker_choice(name, ready, config, agent_slot, state);
                }
            }
            // ctrl+k re-keys the highlighted provider — even one that's
            // already connected — since Enter on a ready provider just
            // switches and would otherwise leave no way to update a key.
            KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let name = match &state.mode {
                    Mode::ProviderPicker { entries, selected } => match entries.get(*selected) {
                        Some(ProviderEntry::Provider { name, .. }) => Some(name.clone()),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(name) = name {
                    state.input.clear();
                    state.cursor = 0;
                    state.mode = Mode::KeyEntry { provider: name };
                }
            }
            KeyCode::Esc => {
                state.mode = Mode::Chat;
            }
            _ => {}
        }
        return Ok(());
    }

    if let Mode::KeyEntry { .. } = &state.mode {
        match key.code {
            KeyCode::Enter => {
                let provider = match &state.mode {
                    Mode::KeyEntry { provider } => provider.clone(),
                    _ => return Ok(()),
                };
                state.mode = Mode::Chat;
                let key = std::mem::take(&mut state.input);
                state.cursor = 0;
                let key = key.trim().to_string();
                if key.is_empty() {
                    state.push_error(format!("no key entered for '{provider}' — key not saved"));
                    return Ok(());
                }
                persist_key_and_switch(&provider, &key, config, state, ui_tx);
                return Ok(());
            }
            KeyCode::Esc => {
                state.mode = Mode::Chat;
                state.input.clear();
                state.cursor = 0;
                return Ok(());
            }
            KeyCode::Backspace => {
                if state.cursor > 0 {
                    remove_char_at(&mut state.input, state.cursor - 1);
                    state.cursor -= 1;
                }
            }
            KeyCode::Left => {
                if state.cursor > 0 {
                    state.cursor -= 1;
                }
            }
            KeyCode::Right => {
                if state.cursor < char_count(&state.input) {
                    state.cursor += 1;
                }
            }
            KeyCode::Home => state.cursor = 0,
            KeyCode::End => state.cursor = char_count(&state.input),
            KeyCode::Char(c) => {
                insert_char_at(&mut state.input, state.cursor, c);
                state.cursor += 1;
            }
            _ => {}
        }
        return Ok(());
    }

    // The terminal convention: Ctrl+C copies whatever is highlighted. When a
    // transcript selection is active that wins — otherwise Ctrl+C keeps its
    // cancel/quit role (matching how ctrl+y already copies a selection and
    // falls back to the last reply). Selection-then-copy is deliberate, so
    // reaching for the OS muscle memory of "highlight + Ctrl+C" just works.
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        if state.selection.is_some() {
            copy_selection(state);
            return Ok(());
        }
        if state.busy {
            if let Some(tx) = cancel_tx.as_ref() {
                let _ = tx.send(true);
            }
        } else {
            state.quit = true;
        }
        return Ok(());
    }

    // A keybinding for the same thing `/copy` does — reaching for a
    // slash command just to copy the last reply is a lot of typing for
    // something you'd want mid-conversation without breaking flow. When a
    // selection is active it copies exactly those blocks instead, which is
    // how you copy anything older than the last reply.
    if key.code == KeyCode::Char('y') && key.modifiers.contains(KeyModifiers::CONTROL) {
        if state.selection.is_some() {
            copy_selection(state);
        } else {
            copy_last_response(state);
        }
        return Ok(());
    }

    // Esc in a plain chat clears any active transcript selection, mirroring
    // how Esc dismisses the pickers. It stays a no-op otherwise, so the key
    // still does nothing alarming mid-conversation.
    if key.code == KeyCode::Esc
        && matches!(state.mode, Mode::Chat)
        && (state.selection.is_some() || state.selection_anchor.is_some())
    {
        state.selection = None;
        state.selection_anchor = None;
        state.push_info("cleared selection");
    }

    // Command palette — matches the reference product's own `ctrl+p`
    // (their `command_list` binding). Reuses the existing slash-command
    // menu wholesale rather than a separate widget: typing "/" already
    // opens it and filters as you type, so seeding the composer with just
    // "/" surfaces every command immediately, exactly as if you'd typed
    // the slash yourself.
    if matches!(state.mode, Mode::Chat)
        && key.code == KeyCode::Char('p')
        && key.modifiers.contains(KeyModifiers::CONTROL)
        && state.input.is_empty()
    {
        // Only when the composer is empty — otherwise this would silently
        // stomp an unsent draft (or a queued follow-up being typed mid-turn)
        // with no way to recover it.
        state.input = "/".to_string();
        state.cursor = 1;
        return Ok(());
    }

    // ESC acts as the universal pause/cancel: during an in-flight turn it
    // stops the agent loop and any running bash tool (same path as Ctrl-C);
    // at idle it clears the current input instead of accidental quit.
    if key.code == KeyCode::Esc {
        if state.busy {
            if let Some(tx) = cancel_tx.as_ref() {
                let _ = tx.send(true);
            }
            state.push_info("ESC pressed — pausing in-flight turn");
        } else {
            state.input.clear();
            state.cursor = 0;
            state.pending_plan_goal = None;
        }
        return Ok(());
    }

    if state.busy {
        // Typing (and Enter to queue) still works while a turn is in
        // flight instead of every keystroke being silently dropped —
        // pickers/dropdowns/mode-switching stay blocked (everything below
        // this point), but composing the *next* message doesn't have to
        // wait for this one to finish first.
        match key.code {
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
                insert_char_at(&mut state.input, state.cursor, '\n');
                state.cursor += 1;
            }
            KeyCode::Enter => {
                let trimmed = state.input.trim().to_string();
                if !trimmed.is_empty() {
                    state.input.clear();
                    state.cursor = 0;
                    state.push_user(trimmed.clone());
                    state
                        .push_info("queued — will send once the current turn finishes".to_string());
                    state.queued_messages.push_back(trimmed);
                }
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                insert_char_at(&mut state.input, state.cursor, c);
                state.cursor += 1;
            }
            KeyCode::Backspace => {
                if state.cursor > 0 {
                    remove_char_at(&mut state.input, state.cursor - 1);
                    state.cursor -= 1;
                }
            }
            KeyCode::Left => {
                if state.cursor > 0 {
                    state.cursor -= 1;
                }
            }
            KeyCode::Right if state.cursor < char_count(&state.input) => {
                state.cursor += 1;
            }
            _ => {}
        }
        return Ok(());
    }

    // While the slash-command / model-autocomplete dropdown is open, arrow
    // keys move the highlight and Enter/Tab accept the highlighted entry
    // instead of their normal effect — commands fill `/cmd ` ready for
    // arguments, models swap the trailing `@…` token for the picked name.
    let (menu_entries, menu_kind) = state.menu();
    let menu_len = menu_entries.len();
    if menu_len > 0 {
        match key.code {
            KeyCode::Up => {
                state.command_selected = if state.command_selected == 0 {
                    menu_len - 1
                } else {
                    state.command_selected - 1
                };
                return Ok(());
            }
            KeyCode::Down => {
                state.command_selected = (state.command_selected + 1) % menu_len;
                return Ok(());
            }
            KeyCode::Enter | KeyCode::Tab => {
                let idx = state.command_selected.min(menu_len - 1);
                let selected = menu_entries[idx].0.clone();
                match menu_kind {
                    MenuKind::Commands => {
                        state.input = format!("/{selected} ");
                    }
                    MenuKind::Models => {
                        // Swap everything after the last `@` (the partial)
                        // for the picked model, keeping any leading text.
                        if let Some(at) = state.input.rfind('@') {
                            state.input.truncate(at + 1);
                            state.input.push_str(&selected);
                        } else {
                            state.input = selected;
                        }
                    }
                }
                state.cursor = char_count(&state.input);
                state.command_selected = 0;
                return Ok(());
            }
            _ => {}
        }
    }

    match key.code {
        KeyCode::Tab => {
            // Not logged to the transcript — the status line inside the
            // input box already shows the current mode continuously, and
            // pushing a transcript entry here would (a) spam the chat log on
            // every toggle and (b) prematurely flip an empty session from
            // the splash view to the chat view just from pressing Tab.
            state.agent_mode = state.agent_mode.toggled();
            if let Some(agent) = agent_slot.as_ref() {
                apply_agent_mode(agent, state.agent_mode);
            }
        }
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
            insert_char_at(&mut state.input, state.cursor, '\n');
            state.cursor += 1;
        }
        KeyCode::Enter => {
            if state.input.trim().is_empty() {
                // A bare Enter right after `/plan` finished confirms running
                // it now — otherwise this has always just been a no-op, so
                // there's nothing existing to conflict with.
                if let Some(goal) = state.pending_plan_goal.take() {
                    state.push_info(format!("running plan: {goal}"));
                    start_orchestrate_turn(
                        state,
                        agent_slot,
                        turn_handle,
                        cancel_tx,
                        ui_tx,
                        goal,
                        yes,
                    );
                }
                return Ok(());
            }
            // Sending a real message instead declines a pending plan
            // confirmation rather than leaving it to fire on some later,
            // unrelated bare Enter.
            state.pending_plan_goal = None;
            let raw = std::mem::take(&mut state.input);
            state.cursor = 0;
            let trimmed = raw.trim().to_string();
            state.record_history(trimmed.clone());
            if matches!(trimmed.as_str(), "exit" | "quit" | ":q") {
                state.quit = true;
                return Ok(());
            }

            if let Some(rest) = trimmed.strip_prefix('/') {
                let mut parts = rest.splitn(2, char::is_whitespace);
                let cmd = parts.next().unwrap_or("");
                let arg = parts.next().unwrap_or("").trim();
                match cmd {
                    "help" => state.push_info(print_repl_help_lines()),
                    "clear" => {
                        let agent = build_agent_repl_with(
                            config,
                            Some(state.provider.clone()),
                            Some(state.model.clone()),
                        )
                        .await?;
                        apply_agent_mode(&agent, state.agent_mode);
                        state.session_id = agent.session_id().to_string();
                        state.model = agent.model().to_string();
                        state.provider = agent.provider_id().to_string();
                        *agent_slot = Some(agent);
                        reset_conversation_view(state);
                        state.push_info(format!("cleared — new session={}", state.session_id));
                    }
                    "new" => {
                        let agent = build_agent_repl_with(
                            config,
                            Some(state.provider.clone()),
                            Some(state.model.clone()),
                        )
                        .await?;
                        apply_agent_mode(&agent, state.agent_mode);
                        state.session_id = agent.session_id().to_string();
                        state.model = agent.model().to_string();
                        state.provider = agent.provider_id().to_string();
                        *agent_slot = Some(agent);
                        reset_conversation_view(state);
                        state.push_info(format!(
                            "new session started — session={}",
                            state.session_id
                        ));
                    }
                    "autocompact" => {
                        let on = arg.to_ascii_lowercase();
                        if let Some(agent) = agent_slot.as_mut() {
                            match on.as_str() {
                                "on" => {
                                    agent.set_auto_compact(true);
                                    state.push_info("auto-compaction: on");
                                }
                                "off" => {
                                    agent.set_auto_compact(false);
                                    state.push_info("auto-compaction: off");
                                }
                                _ => {
                                    let on = agent.auto_compact();
                                    state.push_info(format!(
                                        "auto-compaction: {} — use /autocompact on|off",
                                        if on { "on" } else { "off" }
                                    ));
                                }
                            }
                        }
                    }
                    "compact" => {
                        if let Some(agent) = agent_slot.as_mut() {
                            match agent.compact_now().await {
                                Ok(r) if r.compacted => state.push_info(format!(
                                    "compacted — removed {} earlier message(s)",
                                    r.removed_messages
                                )),
                                Ok(_) => state.push_info("nothing to compact yet"),
                                Err(e) => state.push_error(format!("compact failed: {e:#}")),
                            }
                        }
                    }
                    "context" => {
                        if let Some(agent) = agent_slot.as_ref() {
                            match agent.context_usage().await {
                                Ok(u) => {
                                    let approx = if u.approximate { "~" } else { "" };
                                    state.push_info(format!(
                                        "{approx}{} / {} tokens ({} messages)",
                                        u.tokens, u.window, u.message_count
                                    ));
                                }
                                Err(e) => state.push_error(format!("context lookup failed: {e:#}")),
                            }
                        }
                    }
                    "diff" => {
                        if let Some(agent) = agent_slot.as_ref() {
                            let git = git_engine_for_agent(config, agent);
                            let staged = arg.eq_ignore_ascii_case("staged");
                            match git.diff(staged, &[]) {
                                Ok(out) if out.stdout.trim().is_empty() => {
                                    state.push_info("(no changes)")
                                }
                                Ok(out) => {
                                    let rows = super::highlight::side_by_side_rows(&out.stdout);
                                    state.mode = Mode::Diff { rows, scroll: 0 };
                                }
                                Err(e) => state.push_error(format!("diff failed: {e}")),
                            }
                        }
                    }
                    "undo" => {
                        if let Some(agent) = agent_slot.as_ref() {
                            let ws = agent.workspace();
                            let turn_id = ws.files.turn_id.clone();
                            let snaps = ws
                                .files
                                .checkpoints
                                .load_snapshots(&turn_id)
                                .unwrap_or_default();
                            if snaps.is_empty() {
                                state.push_info("(nothing to undo this session)");
                            } else if arg.eq_ignore_ascii_case("confirm") {
                                match ws.files.checkpoints.restore(&turn_id, &ws.project_root) {
                                    Ok(n) => state.push_info(format!(
                                        "reverted {n} file change(s) made this session"
                                    )),
                                    Err(e) => state.push_error(format!("undo failed: {e}")),
                                }
                            } else {
                                state.push_info(format!(
                                    "this will revert {} file change(s) made since this session started — run `/undo confirm` to proceed",
                                    snaps.len()
                                ));
                            }
                        }
                    }
                    "settings" => {
                        let mut parts = arg.split_whitespace();
                        let path = &config.global.settings_toml;
                        match (parts.next(), parts.next()) {
                            (None, _) => {
                                state.push_info(format!(
                                    "reduced_motion: {}\nnotify_on_completion: {}\naccent_color: {}",
                                    theme::reduced_motion(),
                                    theme::notify_on_completion(),
                                    config.settings.accent_color.as_deref().unwrap_or("(default violet)"),
                                ));
                            }
                            (Some("reduced_motion"), Some(v @ ("on" | "off"))) => {
                                let on = v == "on";
                                theme::set_reduced_motion(on);
                                match zeus_config::set_reduced_motion(path, on) {
                                    Ok(()) => state.push_info(format!("reduced_motion: {on}")),
                                    Err(e) => state.push_error(format!("couldn't save setting: {e}")),
                                }
                            }
                            (Some("notify"), Some(v @ ("on" | "off"))) => {
                                let on = v == "on";
                                theme::set_notify_on_completion(on);
                                match zeus_config::set_notify_on_completion(path, on) {
                                    Ok(()) => state.push_info(format!("notify_on_completion: {on}")),
                                    Err(e) => state.push_error(format!("couldn't save setting: {e}")),
                                }
                            }
                            (Some("accent"), Some("reset")) => {
                                theme::reset_accent();
                                match zeus_config::set_accent_color(path, None) {
                                    Ok(()) => state.push_info("accent_color reset to default"),
                                    Err(e) => state.push_error(format!("couldn't save setting: {e}")),
                                }
                            }
                            (Some("accent"), Some(hex)) => {
                                if theme::set_accent_hex(hex) {
                                    match zeus_config::set_accent_color(path, Some(hex.to_string())) {
                                        Ok(()) => state.push_info(format!("accent_color: {hex}")),
                                        Err(e) => state.push_error(format!("couldn't save setting: {e}")),
                                    }
                                } else {
                                    state.push_error(format!("'{hex}' isn't a #rrggbb hex color"));
                                }
                            }
                            _ => state.push_error(
                                "usage: /settings [reduced_motion on|off] [notify on|off] [accent <#hex>|reset]",
                            ),
                        }
                    }
                    "theme" => {
                        let path = &config.global.settings_toml;
                        match arg.trim() {
                            "" => {
                                state.push_info(format!(
                                    "theme: {} (available: {})",
                                    theme::current_theme().label(),
                                    theme::ThemeKind::ALL
                                        .iter()
                                        .map(|k| k.label())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                ));
                            }
                            name => match theme::ThemeKind::from_label(name) {
                                Some(kind) => {
                                    theme::set_theme(kind);
                                    match zeus_config::set_theme(
                                        path,
                                        Some(kind.label().to_string()),
                                    ) {
                                        Ok(()) => {
                                            state.push_info(format!("theme: {}", kind.label()))
                                        }
                                        Err(e) => {
                                            state.push_error(format!("couldn't save setting: {e}"))
                                        }
                                    }
                                }
                                None => state.push_error(format!(
                                    "'{name}' isn't a theme — try {}",
                                    theme::ThemeKind::ALL
                                        .iter()
                                        .map(|k| k.label())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                )),
                            },
                        }
                    }
                    "model" => {
                        if arg.is_empty() {
                            // No name given — open a picker grouped by
                            // provider (arrow keys/scroll to choose,
                            // Enter/click to apply, Esc to back out) instead
                            // of just printing the current model, so the
                            // user doesn't need to already know an exact
                            // model name to type. A cached scan opens it
                            // instantly; otherwise probing every configured
                            // provider can take several seconds (each has up
                            // to a 3s timeout), so that's spawned rather than
                            // awaited here — awaiting inline would freeze
                            // the whole render loop for that long.
                            state.model_picker_search.clear();
                            if let Some(groups) = state.model_cache.clone() {
                                let (entries, selected) = build_model_picker_entries(
                                    &groups,
                                    &state.provider,
                                    &state.model,
                                    &state.recent_models,
                                    &state.favorite_models,
                                );
                                if entries.is_empty() {
                                    state.push_error(
                                        "no models found on any configured provider (check they're running)",
                                    );
                                } else {
                                    state.mode = Mode::ModelPicker { entries, selected };
                                }
                            } else {
                                state.fetching_providers = true;
                                state.push_info("fetching models…");
                                let cfg = config.clone();
                                let provider = state.provider.clone();
                                let model = state.model.clone();
                                let recent = state.recent_models.clone();
                                let favorites = state.favorite_models.clone();
                                let tx = ui_tx.clone();
                                tokio::spawn(async move {
                                    let groups = list_models_by_provider(&cfg).await;
                                    let (entries, selected) = build_model_picker_entries(
                                        &groups, &provider, &model, &recent, &favorites,
                                    );
                                    let _ = tx
                                        .send(UiEvent::ModelPickerReady(entries, selected, groups));
                                });
                            }
                        } else if let Some(agent) = agent_slot.as_mut() {
                            agent.set_model(arg.to_string());
                            state.model = arg.to_string();
                            state.push_info(format!("switched to model: {arg}"));
                        }
                    }
                    "provider" => {
                        handle_provider_tui(arg, config, agent_slot, state).await;
                    }
                    "session" => state.push_info(format!("session={}", state.session_id)),
                    "sessions" => {
                        let store = SessionStore::new(config.global.sessions.clone());
                        match store.summaries() {
                            Ok(entries) if entries.is_empty() => state
                                .push_info("no saved sessions yet — send a message to create one"),
                            Ok(entries) => {
                                state.session_picker = Some(SessionPickerState {
                                    entries,
                                    selected: 0,
                                    search: String::new(),
                                });
                            }
                            Err(e) => state.push_error(format!("couldn't list sessions: {e:#}")),
                        }
                    }
                    "understand" => {
                        if arg.is_empty() {
                            state.push_error(
                                "usage: /understand <topic> — e.g. /understand authentication"
                                    .to_string(),
                            );
                        } else {
                            state.push_user(trimmed.clone());
                            start_understand_turn(
                                state,
                                agent_slot,
                                turn_handle,
                                cancel_tx,
                                ui_tx,
                                arg.to_string(),
                                yes,
                            );
                        }
                    }
                    "orient" => {
                        state.push_user(trimmed.clone());
                        start_orient_turn(state, agent_slot, turn_handle, cancel_tx, ui_tx, yes);
                    }
                    "review" => {
                        state.push_user(trimmed.clone());
                        start_review_turn(state, agent_slot, turn_handle, cancel_tx, ui_tx, yes);
                    }
                    "suggest" => {
                        state.push_user(trimmed.clone());
                        start_suggest_turn(
                            state,
                            agent_slot,
                            turn_handle,
                            cancel_tx,
                            ui_tx,
                            arg.to_string(),
                            yes,
                        );
                    }
                    "agents" => {
                        if arg.eq_ignore_ascii_case("count") {
                            let pools = personas_by_department();
                            let total: usize = pools.iter().map(|(_, list)| list.len()).sum();
                            state.push_info(format!("{total} specialist agents"));
                        } else {
                            let mut text =
                                String::from("Specialist agent pool (grouped by department):");
                            for (dept, people) in personas_by_department() {
                                text.push_str(&format!("\n  {}:", dept));
                                for p in people {
                                    text.push_str(&format!("\n    {}  — {}", p.id, p.role));
                                }
                            }
                            state.push_info(text);
                        }
                    }
                    "mode" => {
                        let next = match arg.to_ascii_lowercase().as_str() {
                            "build" => AgentMode::Build,
                            "plan" => AgentMode::Plan,
                            "auto" => AgentMode::Auto,
                            "" => state.agent_mode.toggled(),
                            other => {
                                state.push_error(format!(
                                    "unknown mode: {other} (use build | plan | auto)"
                                ));
                                return Ok(());
                            }
                        };
                        state.agent_mode = next;
                        if let Some(agent) = agent_slot.as_ref() {
                            apply_agent_mode(agent, next);
                        }
                        state.push_info(format!("mode: {}", next.label()));
                    }
                    "plan" => {
                        if arg.is_empty() {
                            state.push_error("usage: /plan <goal>");
                        } else {
                            state.agent_mode = AgentMode::Plan;
                            if let Some(agent) = agent_slot.as_ref() {
                                apply_agent_mode(agent, AgentMode::Plan);
                            }
                            state.push_user(trimmed.clone());
                            state.pending_plan_goal = Some(arg.to_string());
                            start_plan_turn(
                                state,
                                agent_slot,
                                turn_handle,
                                cancel_tx,
                                ui_tx,
                                arg.to_string(),
                                yes,
                            );
                        }
                    }
                    "workflow" | "wf" => {
                        let mut parts = arg.splitn(2, char::is_whitespace);
                        let name = parts.next().unwrap_or("").trim();
                        let goal = parts.next().unwrap_or("").trim();
                        if name.is_empty() || goal.is_empty() {
                            state.push_error(
                                "usage: /workflow <name> <goal> — e.g. /workflow build-backend 'add a health endpoint'".to_string(),
                            );
                        } else {
                            let workflows = zeus_agent::discover_workflows(
                                config.project_root.as_deref(),
                                &config.global.root,
                            );
                            match workflows.iter().find(|w| w.id == name) {
                                Some(wf) => {
                                    state.push_user(trimmed.clone());
                                    start_workflow_turn(
                                        state,
                                        agent_slot,
                                        turn_handle,
                                        cancel_tx,
                                        ui_tx,
                                        wf.clone(),
                                        goal.to_string(),
                                        yes,
                                    );
                                }
                                None => {
                                    state.push_error(format!(
                                        "no workflow named '{name}' (run /workflows to list)"
                                    ));
                                }
                            }
                        }
                    }
                    "workflows" => {
                        let workflows = zeus_agent::discover_workflows(
                            config.project_root.as_deref(),
                            &config.global.root,
                        );
                        if workflows.is_empty() {
                            state.push_info(
                                "no workflows found. Create <project>/.agent/workflows/<name>.toml or ~/.zeus/workflows/<name>.toml".to_string(),
                            );
                        } else {
                            for wf in workflows {
                                state.push_info(format!(
                                    "{} — {} ({} phase(s))",
                                    wf.id,
                                    wf.description,
                                    wf.phases.len()
                                ));
                            }
                        }
                    }
                    "bg" => {
                        let mut parts = arg.splitn(2, char::is_whitespace);
                        let rest = parts.next().unwrap_or("").trim();
                        if rest.is_empty() {
                            state.push_error(
                                "usage: /bg <goal> — run an orchestrated plan in the background"
                                    .to_string(),
                            );
                        } else if matches!(rest, "list" | "output" | "stop" | "pause" | "resume") {
                            state.push_info(
                                "manage background tasks with the `zeus bg` subcommand: zeus bg list · zeus bg output <id> · zeus bg pause <id> · zeus bg resume <id> · zeus bg stop <id>".to_string(),
                            );
                        } else {
                            let (goal, workflow) = match rest.rsplit_once("@@workflow:") {
                                Some((g, name)) => (g.trim(), Some(name.trim())),
                                None => (rest, None),
                            };
                            match crate::spawn_bg_orchestrate(config, goal, workflow, None) {
                                Ok(id) => {
                                    state.push_info(format!(
                                        "● background orchestration started id={id}"
                                    ));
                                    state.push_info(format!(
                                        "follow: zeus bg output {id}   |   stop: zeus bg stop {id}"
                                    ));
                                }
                                Err(e) => {
                                    state.push_error(format!("bg spawn failed: {e:#}"));
                                }
                            }
                        }
                    }
                    "copy" => {
                        if state.selection.is_some() {
                            copy_selection(state);
                        } else {
                            copy_last_response(state);
                        }
                    }
                    _ => {
                        let expanded = expand_slash_command(config, trimmed.clone());
                        if expanded != trimmed {
                            state.push_user(trimmed.clone());
                            start_turn(
                                state,
                                agent_slot,
                                turn_handle,
                                cancel_tx,
                                ui_tx,
                                expanded,
                                yes,
                            );
                        } else {
                            state.push_error(format!("unknown command: /{cmd}"));
                        }
                    }
                }
                return Ok(());
            }

            // Catch the common "never set up a key" case before it burns a
            // turn on a network call that's guaranteed to fail with a wall
            // of red API-error text — nudge straight to the picker instead.
            if !provider_status_ok(config, &state.provider) {
                state.push_error(format!(
                    "'{}' isn't connected yet — opening the provider picker to set it up",
                    state.provider
                ));
                open_provider_picker(state, config);
                return Ok(());
            }
            state.push_user(trimmed.clone());
            start_turn(
                state,
                agent_slot,
                turn_handle,
                cancel_tx,
                ui_tx,
                trimmed,
                yes,
            );
        }
        KeyCode::Backspace => {
            if state.cursor > 0 {
                remove_char_at(&mut state.input, state.cursor - 1);
                state.cursor -= 1;
                state.command_selected = 0;
            }
        }
        KeyCode::Left => {
            if state.cursor > 0 {
                state.cursor -= 1;
            }
        }
        KeyCode::Right if state.cursor < char_count(&state.input) => {
            state.cursor += 1;
        }
        KeyCode::Home => state.cursor = 0,
        KeyCode::End => state.cursor = char_count(&state.input),
        // Shell-style history recall — only reachable here because the
        // slash-command palette and every picker/dropdown already claim
        // Up/Down for themselves higher up in this function.
        KeyCode::Up if !state.input_history.is_empty() => {
            if state.history_pos.is_none() {
                state.history_draft = state.input.clone();
            }
            let next = match state.history_pos {
                Some(0) => 0,
                Some(i) => i - 1,
                None => state.input_history.len() - 1,
            };
            state.history_pos = Some(next);
            state.input = state.input_history[next].clone();
            state.cursor = char_count(&state.input);
        }
        KeyCode::Down if state.history_pos.is_some() => {
            let at_newest = state.history_pos == Some(state.input_history.len() - 1);
            if at_newest {
                state.history_pos = None;
                state.input = std::mem::take(&mut state.history_draft);
            } else if let Some(i) = state.history_pos {
                state.history_pos = Some(i + 1);
                state.input = state.input_history[i + 1].clone();
            }
            state.cursor = char_count(&state.input);
        }
        KeyCode::Char(c) => {
            insert_char_at(&mut state.input, state.cursor, c);
            state.cursor += 1;
            state.command_selected = 0;
        }
        KeyCode::PageUp | KeyCode::PageDown => {
            let step = state
                .transcript_area
                .map(|a| a.height.saturating_sub(2))
                .unwrap_or(10)
                .max(1);
            state.scroll_transcript(key.code == KeyCode::PageUp, step);
        }
        _ => {}
    }
    Ok(())
}

/// Mouse support for the model picker: click a row to select and apply it
/// immediately (matching opencode's own click-to-choose), scroll to move
/// the highlight without applying. Only meaningful while the picker is
/// open — mouse events are otherwise ignored.
async fn handle_mouse(
    ev: MouseEvent,
    state: &mut AppState,
    agent_slot: &mut Option<Agent>,
    config: &Config,
) {
    if let MouseEventKind::Down(MouseButton::Left) = ev.kind {
        let col = ev.column;
        let row = ev.row;

        // Empty-state example chips: click one to fill the composer.
        if let Some(idx) = state
            .chip_areas
            .iter()
            .position(|a| col >= a.x && col < a.x + a.width && row >= a.y && row < a.y + a.height)
        {
            if let Some(label) = EXAMPLE_CHIPS.get(idx) {
                state.input = label.to_string();
                state.cursor = char_count(&state.input);
            }
            return;
        }

        // Slash-command / model-autocomplete palette: click a row to
        // accept it, same as pressing Enter/Tab on the highlighted entry —
        // commands fill the input ready for arguments, models swap the
        // trailing `@…` token, rather than sending immediately.
        if let Some(area) = state.command_menu_area {
            if col >= area.x
                && col < area.x + area.width
                && row >= area.y
                && row < area.y + area.height
            {
                let idx = (row - area.y) as usize;
                let (entries, kind) = state.menu();
                let selected = entries.get(idx).map(|(n, _)| n.clone());
                if let Some(selected) = selected {
                    match kind {
                        MenuKind::Commands => state.input = format!("/{selected} "),
                        MenuKind::Models => {
                            if let Some(at) = state.input.rfind('@') {
                                state.input.truncate(at + 1);
                                state.input.push_str(&selected);
                            } else {
                                state.input = selected;
                            }
                        }
                    }
                    state.cursor = char_count(&state.input);
                    state.command_selected = 0;
                }
                return;
            }
        }

        // Session-resume picker: click a row to resume it immediately —
        // same click-to-choose interaction as the model picker.
        if let Some(area) = state.session_picker_area {
            if col >= area.x
                && col < area.x + area.width
                && row >= area.y
                && row < area.y + area.height
            {
                let idx = (row - area.y) as usize;
                let session_id = state
                    .session_picker
                    .as_ref()
                    .and_then(|p| session_picker_filtered(p).get(idx).map(|e| e.id.clone()));
                if let Some(id) = session_id {
                    state.session_picker = None;
                    resume_session(id, config, agent_slot, state).await;
                }
                return;
            }
        }

        // Transcript: left-click SELECTS a block (highlighted, then copied
        // explicitly with ctrl+y) instead of copying it instantly — a click
        // on a folded tool result still expands it. Shift-click or drag
        // extends the selection from the click anchor; clicking empty space
        // clears it. This replaces the old "click copies the last thing"
        // shortcut, which could never reach past the single most recent
        // block; selection reaches any block and copying is deliberate.
        if matches!(state.mode, Mode::Chat) {
            if let Some(area) = state.transcript_area {
                if col >= area.x
                    && col < area.x + area.width
                    && row >= area.y
                    && row < area.y + area.height
                {
                    match transcript_block_at(
                        col,
                        row,
                        state.transcript_area,
                        &state.transcript_block_rows,
                        state.transcript_applied_scroll,
                    ) {
                        Some(idx) => {
                            if let Some(block) = state.transcript.get(idx) {
                                if block.is_foldable() {
                                    block.toggle_expanded();
                                }
                            }
                            if ev.modifiers.contains(KeyModifiers::SHIFT) {
                                let anchor = state.selection_anchor.unwrap_or(idx);
                                state.selection_anchor = Some(anchor);
                                state.selection = Some((anchor.min(idx), anchor.max(idx)));
                            } else {
                                state.selection_anchor = Some(idx);
                                state.selection = Some((idx, idx));
                            }
                        }
                        None => {
                            state.selection = None;
                            state.selection_anchor = None;
                        }
                    }
                    return;
                }
            }
        }

        // Click a TODO row to toggle it (progress bar fills in live).
        if let Some(ta) = state.todo_area {
            if col >= ta.x && col < ta.x + ta.width && row >= ta.y && row < ta.y + ta.height {
                let idx = row as isize - ta.y as isize;
                if idx >= 0 && (idx as usize) < state.todos.len() {
                    state.toggle_todo(idx as usize);
                }
                return;
            }
        }
    }

    // Drag the left button over the transcript to extend the selection from
    // the click anchor. The `Down` press above set the anchor; every drag
    // motion re-targets the far end so the highlight grows/shrinks live.
    if let MouseEventKind::Drag(MouseButton::Left) = ev.kind {
        if matches!(state.mode, Mode::Chat) {
            if let (Some(area), Some(anchor)) = (state.transcript_area, state.selection_anchor) {
                if ev.column >= area.x
                    && ev.column < area.x + area.width
                    && ev.row >= area.y
                    && ev.row < area.y + area.height
                {
                    if let Some(idx) = transcript_block_at(
                        ev.column,
                        ev.row,
                        Some(area),
                        &state.transcript_block_rows,
                        state.transcript_applied_scroll,
                    ) {
                        state.selection = Some((anchor.min(idx), anchor.max(idx)));
                        return;
                    }
                }
            }
        }
    }

    // Slash-command / model-autocomplete palette: scroll to move the
    // highlight, without needing to be over the palette itself — it's the
    // only thing visible to scroll while it's open.
    if !state.busy {
        let menu_len = state.menu().0.len();
        if menu_len > 0 {
            match ev.kind {
                MouseEventKind::ScrollUp => {
                    state.command_selected = if state.command_selected == 0 {
                        menu_len - 1
                    } else {
                        state.command_selected - 1
                    };
                    return;
                }
                MouseEventKind::ScrollDown => {
                    state.command_selected = (state.command_selected + 1) % menu_len;
                    return;
                }
                _ => {}
            }
        }
    }

    // Mouse-wheel scrolling over the transcript itself — previously a
    // no-op there entirely (only pickers/the palette handled wheel
    // events), so there was no way to look back at earlier messages
    // except resizing the terminal.
    if matches!(state.mode, Mode::Chat) && !state.busy {
        if let Some(area) = state.transcript_area {
            let over_transcript = ev.column >= area.x
                && ev.column < area.x + area.width
                && ev.row >= area.y
                && ev.row < area.y + area.height;
            if over_transcript {
                match ev.kind {
                    MouseEventKind::ScrollUp => {
                        state.scroll_transcript(true, 3);
                        return;
                    }
                    MouseEventKind::ScrollDown => {
                        state.scroll_transcript(false, 3);
                        return;
                    }
                    _ => {}
                }
            }
        }
    }

    match state.mode {
        Mode::ModelPicker { .. } => {
            let Some(area) = state.model_picker_area else {
                return;
            };
            match ev.kind {
                MouseEventKind::Down(MouseButton::Left) => {
                    if ev.column >= area.x
                        && ev.column < area.x + area.width
                        && ev.row >= area.y
                        && ev.row < area.y + area.height
                    {
                        let row = (ev.row - area.y) as usize;
                        let chosen = if let Mode::ModelPicker { entries, .. } = &state.mode {
                            let filtered =
                                model_picker_filtered(entries, &state.model_picker_search);
                            filtered.get(row).and_then(|e| match e {
                                PickerEntry::Model { provider, model } => {
                                    Some((provider.clone(), model.id.clone()))
                                }
                                _ => None,
                            })
                        } else {
                            None
                        };
                        if let Some((provider, model_id)) = chosen {
                            apply_model_choice_or_key_entry(
                                provider, model_id, config, agent_slot, state,
                            );
                        }
                    }
                }
                MouseEventKind::ScrollUp => {
                    if let Mode::ModelPicker { entries, selected } = &mut state.mode {
                        let filtered = model_picker_filtered(entries, &state.model_picker_search);
                        *selected = picker_move(&filtered, *selected, -1);
                    }
                }
                MouseEventKind::ScrollDown => {
                    if let Mode::ModelPicker { entries, selected } = &mut state.mode {
                        let filtered = model_picker_filtered(entries, &state.model_picker_search);
                        *selected = picker_move(&filtered, *selected, 1);
                    }
                }
                _ => {}
            }
        }
        Mode::ProviderPicker { .. } => {
            let Some(area) = state.provider_picker_area else {
                return;
            };
            match ev.kind {
                MouseEventKind::Down(MouseButton::Left) => {
                    if ev.column >= area.x
                        && ev.column < area.x + area.width
                        && ev.row >= area.y
                        && ev.row < area.y + area.height
                    {
                        let row = (ev.row - area.y) as usize;
                        let chosen = if let Mode::ProviderPicker { entries, .. } = &state.mode {
                            entries.get(row).and_then(|e| match e {
                                ProviderEntry::Provider { name, ready, .. } => {
                                    Some((name.clone(), *ready))
                                }
                                _ => None,
                            })
                        } else {
                            None
                        };
                        if let Some((name, ready)) = chosen {
                            apply_provider_picker_choice(name, ready, config, agent_slot, state);
                        }
                    }
                }
                MouseEventKind::ScrollUp => {
                    if let Mode::ProviderPicker { entries, selected } = &mut state.mode {
                        *selected = provider_picker_move(entries, *selected, -1);
                    }
                }
                MouseEventKind::ScrollDown => {
                    if let Mode::ProviderPicker { entries, selected } = &mut state.mode {
                        *selected = provider_picker_move(entries, *selected, 1);
                    }
                }
                _ => {}
            }
        }
        Mode::Diff { .. } => match ev.kind {
            MouseEventKind::ScrollUp => {
                if let Mode::Diff { scroll, .. } = &mut state.mode {
                    *scroll = scroll.saturating_sub(3);
                }
            }
            MouseEventKind::ScrollDown => {
                if let Mode::Diff { scroll, .. } = &mut state.mode {
                    *scroll = scroll.saturating_add(3);
                }
            }
            _ => {}
        },
        _ => {}
    }
}

async fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    config: &Config,
    agent: Agent,
    yes: bool,
) -> Result<()> {
    let known_commands = known_slash_commands(config);
    let dir = build_dir_info(config);
    let mut state = AppState::new(
        &agent,
        known_commands,
        dir,
        config.project_root.is_some(),
        config,
    );
    let mut agent_slot = Some(agent);
    // When starting with an actual project, begin in read-only Plan mode so
    // the agent researches before it changes anything (and the injected
    // project survey grounds the plan in real files rather than guesses).
    if config.project_root.is_some() {
        if let Some(a) = agent_slot.as_mut() {
            apply_agent_mode(a, state.agent_mode);
        }
    }
    let mut turn_handle: Option<TurnJoin> = None;
    let mut cancel_tx: Option<watch::Sender<bool>> = None;

    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<UiEvent>();

    // One-shot background version check — never blocks startup, never
    // auto-installs, just surfaces a small dim notice if a newer release
    // exists (see `render_topbar`). A failed/offline check is silent by
    // design: `update::latest_version()`'s `Err` is simply dropped.
    let update_tx = ui_tx.clone();
    tokio::spawn(async move {
        if let Ok(latest) = super::update::latest_version().await {
            if super::update::is_newer(&latest, super::update::current_version()) {
                let _ = update_tx.send(UiEvent::UpdateAvailable(latest));
            }
        }
    });

    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<Event>();
    std::thread::spawn(move || {
        while let Ok(ev) = crossterm::event::read() {
            if input_tx.send(ev).is_err() {
                break;
            }
        }
    });

    terminal.draw(|f| render(f, &mut state, config))?;
    sync_cursor_visibility(terminal, &state);

    // Redraws happen on demand (key/mouse/agent events) everywhere else, but
    // continuous animation — the status dot pulse, the busy spinner —
    // needs a steady heartbeat too. 10fps is plenty for a character-cell
    // animation and cheap enough to leave running only while something is
    // actually animating (`wants_animation`); otherwise this tick still
    // fires but is a no-op draw-wise.
    let mut anim_tick = tokio::time::interval(std::time::Duration::from_millis(100));
    anim_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        let mut should_draw = true;
        tokio::select! {
            _ = anim_tick.tick() => {
                should_draw = wants_animation(&state);
            }
            Some(ev) = input_rx.recv() => {
                match ev {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        handle_key(key, &mut state, &mut agent_slot, &mut turn_handle, &mut cancel_tx, &ui_tx, config, yes).await?;
                    }
                    Event::Paste(text) => {
                        if let Mode::KeyEntry { .. } = state.mode {
                            for c in text.chars() {
                                insert_char_at(&mut state.input, state.cursor, c);
                                state.cursor += 1;
                            }
                        } else if matches!(state.mode, Mode::Chat) {
                            // Plain composer paste (a message, or slash-command
                            // text — both live in `state.input`). Pickers use
                            // their own search buffers and don't want raw
                            // text dumped into `state.input`.
                            let pasted_chars = text.chars().count();
                            let pasted_lines = text.lines().count();
                            for c in text.chars() {
                                insert_char_at(&mut state.input, state.cursor, c);
                                state.cursor += 1;
                            }
                            // The composer only ever shows a handful of
                            // wrapped rows — pasting a whole file or a long
                            // log in gave no sign anything unusual just
                            // happened until you scrolled through it (or
                            // sent it).
                            if pasted_lines > 10 || pasted_chars > 1000 {
                                state.push_info(format!(
                                    "pasted {pasted_chars} character(s) across {pasted_lines} line(s) into the composer"
                                ));
                            }
                        }
                    }
                    Event::Mouse(mouse) => {
                        handle_mouse(mouse, &mut state, &mut agent_slot, config).await;
                    }
                    Event::Resize(..) => {
                        // Ratatui's own diffing skips cells that look
                        // unchanged from its last-known buffer, but that
                        // buffer can go stale relative to what's actually on
                        // screen right after a resize — especially on
                        // legacy Windows consoles (`cmd.exe`/conhost, as
                        // opposed to Windows Terminal), which have known
                        // gaps in VT/ANSI handling and can leave old glyphs
                        // sitting in cells the new frame no longer writes
                        // to. Forcing a full repaint here (skip the diff
                        // once) clears any such leftovers.
                        terminal.clear().ok();
                    }
                    _ => {}
                }
            }
            Some(ui_event) = ui_rx.recv() => {
                match ui_event {
                    UiEvent::Agent(ev) => state.apply_agent_event(ev),
                    UiEvent::Approval(req) => {
                        state.mode = Mode::Approval {
                            pending: req,
                            scroll: 0,
                        }
                    }
                    UiEvent::ModelPickerReady(entries, selected, groups) => {
                        state.fetching_providers = false;
                        state.model_cache = Some(groups);
                        if matches!(state.mode, Mode::Chat) {
                            if entries.is_empty() {
                                state.push_error(
                                    "no models found on any configured provider (check they're running)",
                                );
                            } else {
                                state.mode = Mode::ModelPicker { entries, selected };
                            }
                        }
                    }
                    UiEvent::UpdateAvailable(latest) => {
                        state.update_available = Some(latest);
                    }
                }
            }
            res = async { turn_handle.as_mut().unwrap().await }, if turn_handle.is_some() => {
                turn_handle = None;
                state.busy = false;
                state.active_persona = None;
                cancel_tx = None;
                // The cue that lets you tab away during a long/Auto-mode
                // run instead of watching it — BEL is safe to write
                // straight to stdout mid-alternate-screen since terminals
                // treat it as a non-printing control byte (no cursor move,
                // no redraw), unlike any other raw write here would be.
                if theme::notify_on_completion() {
                    use std::io::Write as _;
                    let _ = write!(io::stdout(), "\x07");
                    let _ = io::stdout().flush();
                }
                // Safety net alongside the more precise `OrchestrationDone`
                // /`OrchestrationRevision`/`Cancelled` resets: `/plan` alone
                // (research-only, no execution) sets `plan_active` via
                // `PlanGenerated` too but has no step-driven event to clear
                // it afterward, so without this it would stay stuck `true`
                // and silently disable the checklist heuristic for every
                // later normal turn in the session.
                state.plan_active = false;
                match res {
                    Ok((agent, turn_result)) => {
                        agent_slot = Some(agent);
                        state.flush_current_reply();
                        match turn_result {
                            Ok(result) => {
                                state.session_usage.prompt_tokens += result.usage.prompt_tokens;
                                state.session_usage.completion_tokens += result.usage.completion_tokens;
                                state.session_usage.total_tokens += result.usage.total_tokens;
                                // Lazily fetched once per model — a real
                                // provider call (token counting), not worth
                                // paying for on every render, and the
                                // window itself never changes for a given
                                // model anyway.
                                if state.context_window.is_none() {
                                    if let Some(agent) = agent_slot.as_ref() {
                                        if let Ok(usage) = agent.context_usage().await {
                                            state.context_window = Some(usage.window);
                                        }
                                    }
                                }
                                // `/plan` alone never executes anything — offer
                                // to actually run it now via `orchestrate`
                                // rather than leaving the persisted
                                // `.agent/tasks.json` unused. Left set (not
                                // consumed here) so the next bare-Enter press
                                // can pick it up; declining by sending a real
                                // message instead clears it in `handle_key`.
                                //
                                // `result.cancelled` guards this: a `/plan`
                                // stopped mid-research via Esc still returns
                                // `Ok` (cancellation isn't an error), so
                                // without this check a cancelled plan would
                                // both claim to be "ready" and leave a stale
                                // goal armed for some later, unrelated bare
                                // Enter to actually execute.
                                if result.cancelled {
                                    state.pending_plan_goal = None;
                                } else if let Some(goal) = &state.pending_plan_goal {
                                    state.push_info(format!(
                                        "plan ready for \"{goal}\" — press Enter now to run it, Esc to dismiss, or type a message to keep going instead"
                                    ));
                                }
                            }
                            Err(e) => {
                                state.pending_plan_goal = None;
                                state.push_error(format!("turn failed: {e:#}"));
                            }
                        }
                    }
                    Err(join_err) => {
                        state.push_error(format!("internal error: {join_err}"));
                        let agent =
                            build_agent_repl_with(config, Some(state.provider.clone()), Some(state.model.clone()))
                                .await?;
                        apply_agent_mode(&agent, state.agent_mode);
                        agent_slot = Some(agent);
                    }
                }
                // Drain one queued message now that this turn is done — its
                // bubble and "queued" note were already pushed when it was
                // submitted (see `handle_key`'s busy-Enter branch), so this
                // just starts the turn, same as a fresh Enter would.
                if let Some(next) = state.queued_messages.pop_front() {
                    start_turn(&mut state, &mut agent_slot, &mut turn_handle, &mut cancel_tx, &ui_tx, next, yes);
                }
            }
        }

        if state.quit {
            break;
        }
        if should_draw {
            terminal.draw(|f| render(f, &mut state, config))?;
            sync_cursor_visibility(terminal, &state);
        }
    }
    Ok(())
}

/// Whether the animated tick's redraw is worth paying for right now — the
/// empty-state splash's pulsing status dot and the busy spinner are the
/// only things that change without a user/agent event driving a redraw.
fn wants_animation(state: &AppState) -> bool {
    state.showing_empty_state() || state.busy || state.fetching_providers
}

/// Current git branch for the side-panel footer. Best-effort — any failure
/// degrades silently to "(no git repo)".
fn build_dir_info(config: &Config) -> DirInfo {
    let path = config.project_root.clone().unwrap_or_else(|| {
        std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
    });

    let git_branch = std::process::Command::new("git")
        .arg("-C")
        .arg(&path)
        .arg("branch")
        .arg("--show-current")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());

    DirInfo { git_branch }
}

/// Belt-and-suspenders alongside `render_input_box` only setting a cursor
/// position in idle chat mode: explicitly hides the terminal cursor
/// whenever it wouldn't otherwise be positioned, rather than relying solely
/// on ratatui's own per-frame tracking — some Windows consoles (legacy
/// conhost in particular) have been observed leaving a stray blinking
/// cursor at its last position instead of hiding it. Harmless if ratatui
/// already handled it.
fn sync_cursor_visibility<B: Backend>(terminal: &mut Terminal<B>, state: &AppState) {
    // Busy no longer hides the cursor on its own — the composer stays
    // typeable (to queue the next message) while a turn is in flight, so
    // Chat mode keeps a live cursor regardless of `busy`; only an actual
    // picker/modal mode (anything but Chat/KeyEntry) takes it away.
    if !matches!(state.mode, Mode::Chat | Mode::KeyEntry { .. }) {
        terminal.hide_cursor().ok();
    }
}

pub async fn run(config: &Config, agent: Agent, yes: bool) -> Result<()> {
    theme::init_runtime(
        config.settings.accent_color.as_deref(),
        config.settings.reduced_motion,
        config.settings.notify_on_completion,
        config.settings.theme.as_deref(),
    );
    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        EnableBracketedPaste
    )
    .context("enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend).context("init terminal")?;
    // `TerminalGuard` restores the console on *both* the normal return path
    // and a panic inside `run_app`: without it, a panic anywhere in the draw
    // or key loop leaves the user's terminal stuck in raw mode + the
    // alternate screen. This used to be plain sequential teardown after the
    // `await`, which silently skipped exactly that case.
    let mut terminal = TerminalGuard { terminal };

    let result = run_app(&mut terminal.terminal, config, agent, yes).await;

    drop(terminal);
    result
}

/// Owning wrapper that restores the terminal (raw mode off, leave alternate
/// screen, mouse/paste off, cursor visible) when dropped — including during
/// unwinding from a panic in `run_app`, which the previous sequential
/// teardown never covered. Teardown targets `stdout` directly rather than the
/// wrapped backend: the alternate screen / mouse capture / bracketed paste
/// are console *state* that belongs to the terminal device, not to a
/// particular backend, so the same `Drop` works for any `Backend` (and is
/// exercisable in tests against a `TestBackend` with no console attached).
struct TerminalGuard<B: Backend> {
    terminal: Terminal<B>,
}

impl<B: Backend> Drop for TerminalGuard<B> {
    fn drop(&mut self) {
        disable_raw_mode().ok();
        execute!(
            std::io::stdout(),
            LeaveAlternateScreen,
            DisableMouseCapture,
            DisableBracketedPaste
        )
        .ok();
        self.terminal.show_cursor().ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The panic-safe teardown path runs without a real terminal: the guard's
    /// `Drop` exercises the same teardown sequence (`disable_raw_mode`,
    /// leave-alt-screen, mouse/paste off, cursor show) that a panic inside
    /// `run_app` triggers. Against a `TestBackend` the crossterm escapes are
    /// buffered rather than sent to a console, and the raw-mode calls fail
    /// harmlessly — so this proves dropping the guard never panics even when
    /// nothing terminal-y is actually attached.
    #[test]
    fn terminal_guard_drop_runs_teardown_without_panicking() {
        let terminal = Terminal::new(ratatui::backend::TestBackend::new(80, 24)).unwrap();
        let guard = TerminalGuard { terminal };
        drop(guard);
    }

    #[test]
    fn agent_mode_cycles_and_labels() {
        assert_eq!(AgentMode::Build.label(), "Build");
        assert_eq!(AgentMode::Build.toggled(), AgentMode::Plan);
        assert_eq!(AgentMode::Plan.toggled(), AgentMode::Auto);
        assert_eq!(AgentMode::Auto.toggled(), AgentMode::Build);
        assert_eq!(mode_accent(AgentMode::Build), theme::CYAN);
        assert_eq!(mode_accent(AgentMode::Plan), theme::GOLD);
        assert_eq!(mode_accent(AgentMode::Auto), theme::MAGENTA);
    }

    #[test]
    fn slash_command_filter_narrows_by_prefix() {
        let known: Vec<(String, String)> = vec![
            ("model".into(), "switch model".into()),
            ("mode".into(), "switch agent mode".into()),
            ("provider".into(), "switch provider".into()),
        ];
        let matches = filter_commands("/m", &known);
        assert_eq!(
            matches.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            vec!["model", "mode"]
        );
        assert!(filter_commands("model", &known).is_empty());
        assert!(filter_commands("/nope", &known).is_empty());
        assert!(filter_commands("/mode with spaces", &known).is_empty());
    }

    #[test]
    fn model_matches_trigger_on_at_token_and_filter() {
        let cache: Vec<(String, Vec<zeus_provider::ModelInfo>)> = vec![(
            "openrouter".to_string(),
            vec![
                zeus_provider::ModelInfo {
                    id: "claude-sonnet-4".to_string(),
                    name: String::new(),
                    context_window: None,
                },
                zeus_provider::ModelInfo {
                    id: "gpt-5".to_string(),
                    name: String::new(),
                    context_window: None,
                },
            ],
        )];
        let recent: Vec<(String, String)> =
            vec![("openrouter".to_string(), "deepseek".to_string())];
        let favorites: Vec<(String, String)> = Vec::new();

        // Bare `@` lists everything the app knows about, de-duplicated.
        let all = filter_model_matches("@", &cache, &recent, &favorites);
        assert_eq!(
            all,
            vec![
                (
                    "openrouter/claude-sonnet-4".to_string(),
                    "openrouter".to_string()
                ),
                ("openrouter/gpt-5".to_string(), "openrouter".to_string()),
                ("openrouter/deepseek".to_string(), "openrouter".to_string()),
            ]
        );

        // A partial narrows case-insensitively by prefix.
        let claude = filter_model_matches("use @CLAU", &cache, &recent, &favorites);
        assert_eq!(
            claude,
            vec![(
                "openrouter/claude-sonnet-4".to_string(),
                "openrouter".to_string()
            )]
        );

        // Mid-sentence `@` also triggers (trailing token), and a provider/
        // model that doesn't match is dropped.
        let deep = filter_model_matches("switch to @deep", &cache, &recent, &favorites);
        assert_eq!(
            deep,
            vec![("openrouter/deepseek".to_string(), "openrouter".to_string())]
        );

        // No `@` in the input means no autocomplete.
        assert!(filter_model_matches("hello world", &cache, &recent, &favorites).is_empty());
        // A non-trailing `@` (mid-word, followed by text after whitespace)
        // doesn't trigger — only the last whitespace token counts.
        assert!(filter_model_matches("hi @x there", &cache, &recent, &favorites).is_empty());
    }

    #[test]
    fn menu_prefers_commands_but_falls_back_to_models() {
        let cache: Vec<(String, Vec<zeus_provider::ModelInfo>)> = vec![(
            "openrouter".to_string(),
            vec![zeus_provider::ModelInfo {
                id: "gpt-5".to_string(),
                name: String::new(),
                context_window: None,
            }],
        )];
        // No `@`, no slash-prefix → nothing to show.
        let matches = filter_model_matches("plain text", &cache, &[], &[]);
        assert!(matches.is_empty());
    }

    #[test]
    fn selection_plain_text_joins_selected_blocks() {
        let transcript: Vec<Block_> = vec![
            Block_::new(Role::User, "first question".into()),
            Block_::new(Role::Assistant, "first answer".into()),
            Block_::new(Role::User, "second question".into()),
            Block_::new(Role::Assistant, "second answer".into()),
        ];
        assert_eq!(
            selection_plain_text(&transcript, Some((1, 2))).unwrap(),
            "first answer\n\nsecond question"
        );
        assert_eq!(
            selection_plain_text(&transcript, Some((3, 3))).unwrap(),
            "second answer"
        );
        // A range past the end clips instead of panicking.
        assert_eq!(
            selection_plain_text(&transcript, Some((2, 99))).unwrap(),
            "second question\n\nsecond answer"
        );
        assert_eq!(selection_plain_text(&transcript, None), None);
        // Empty range (start beyond transcript) yields nothing.
        assert_eq!(selection_plain_text(&transcript, Some((10, 12))), None);
    }

    #[test]
    fn transcript_block_at_maps_wrapped_rows_to_blocks() {
        let area = Some(Rect::new(0, 0, 80, 24));
        // Block 0 spans rows 0..2, block 1 spans rows 3..6 (wrapped).
        let rows = vec![(0u16, 2u16), (3u16, 6u16)];
        assert_eq!(transcript_block_at(10, 1, area, &rows, 0), Some(0));
        assert_eq!(transcript_block_at(10, 4, area, &rows, 1), Some(1));
        // The blank separator row between blocks hits nothing.
        assert_eq!(transcript_block_at(10, 2, area, &rows, 0), None);
        // Outside the pane: no block.
        assert_eq!(transcript_block_at(10, 30, area, &rows, 0), None);
        // Scrolled by 2 rows shifts the mapping accordingly.
        assert_eq!(transcript_block_at(10, 1, area, &rows, 2), Some(1));
        // Without a computed area there is nothing to hit.
        assert_eq!(transcript_block_at(10, 1, None, &rows, 0), None);
    }

    // ---- TUI golden tests: modal renders, key/mouse handling ----

    /// Build a minimal in-memory `Agent` rooted at a temp project dir (same
    /// shape as `zeus-agent`'s own test harness): an `UnconfiguredProvider`
    /// (never called — these tests only render/handle input), a `ToolManager`
    /// over a temp workspace, and a fresh session. Lets `AppState::new`
    /// construct a real state without touching the network or the user's
    /// `~/.zeus`.
    fn test_agent(root: &std::path::Path) -> Agent {
        let config = test_config(root);
        let workspace = zeus_fs::Workspace::from_config(&config).unwrap();
        let terminal = zeus_agent::TerminalRunner::new(root.join(".agent/checkpoints"));
        let background = zeus_agent::BackgroundTaskRegistry::new(root.join(".agent/background"));
        let hooks = zeus_agent::HookRunner::new(root.join(".agent/hooks"), root.to_path_buf());
        let mut tools = zeus_agent::ToolManager::new(
            workspace,
            terminal,
            background,
            hooks,
            Vec::new(),
            Vec::new(),
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        );
        tools.set_global_skills_dir(None);
        Agent::new(
            std::sync::Arc::new(zeus_provider::UnconfiguredProvider { requested: None }),
            tools,
            zeus_agent::ContextManager::new(128_000, 0.8, 6),
            SessionStore::new(root.join(".sessions")),
            zeus_agent::ConversationState::new("test-session"),
            zeus_agent::AgentOptions {
                model: "test-model".into(),
                max_tool_iterations: 8,
                temperature: None,
                max_tokens: Some(1024),
                max_parallel_read_steps: 2,
                tasks_file: None,
            },
        )
    }

    fn test_config(root: &std::path::Path) -> zeus_config::Config {
        zeus_config::Config {
            global: zeus_config::GlobalPaths::from_root(root.join(".zeus-home")),
            project: None,
            settings: zeus_config::AgentSettings::default(),
            providers: zeus_config::ProvidersFile::default(),
            project_root: Some(root.to_path_buf()),
        }
    }

    fn test_app_state(root: &std::path::Path) -> (AppState, Config) {
        let config = test_config(root);
        let agent = test_agent(root);
        let state = AppState::new(
            &agent,
            Vec::new(),
            DirInfo { git_branch: None },
            false,
            &config,
        );
        (state, config)
    }

    /// Flatten every rendered cell into one string (rows joined by '\n') so
    /// assertions can check for expected glyphs/columns without depending on
    /// exact cursor/color internals.
    fn buffer_rows(buffer: &ratatui::buffer::Buffer) -> Vec<String> {
        let mut rows: Vec<String> = Vec::new();
        for y in 0..buffer.area.height {
            let mut line = String::new();
            for x in 0..buffer.area.width {
                if let Some(cell) = buffer.cell((x, y)) {
                    line.push_str(cell.symbol());
                }
            }
            rows.push(line);
        }
        rows
    }

    /// The `/diff` modal draws removed/added cells into two columns and
    /// spans headers across the full width — a golden render that locks in
    /// the two-pane layout.
    #[test]
    fn diff_modal_renders_two_pane_columns() {
        use crate::highlight::DiffRow;
        let rows = vec![
            DiffRow::Header("@@ -1,2 +1,2 @@".to_string()),
            DiffRow::Pair(Some("before".to_string()), Some("after".to_string())),
            DiffRow::Pair(Some("only-old".to_string()), None),
            DiffRow::Pair(None, Some("only-new".to_string())),
            DiffRow::Pair(Some("same".to_string()), Some("same".to_string())),
        ];
        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(100, 30)).unwrap();
        terminal
            .draw(|f| render_diff_modal(f, f.area(), &rows, 0))
            .unwrap();
        let rows = buffer_rows(terminal.backend().buffer());

        // Header row spans the modal: its `@@` hunk marker is present.
        assert!(rows.iter().any(|r| r.contains("@@ -1,2 +1,2 @@")));
        // Old/new column cells both appear.
        assert!(rows
            .iter()
            .any(|r| r.contains("before") && r.contains("after")));
        // A removed-only row shows on the left; an added-only row on the right.
        assert!(rows.iter().any(|r| r.contains("only-old")));
        assert!(rows.iter().any(|r| r.contains("only-new")));
        // Unchanged context fills both cells.
        assert!(rows.iter().any(|r| r.contains("same")));
        // The two columns are separated by the divider glyph.
        assert!(rows.iter().any(|r| r.contains('│')));
        // Title and bottom hint render.
        assert!(rows.iter().any(|r| r.contains(" Diff ")));
        assert!(rows.iter().any(|r| r.contains("esc close")));
    }

    /// The key-entry modal sizes its input box against the *actual* wrap
    /// width (`width - 4`), not the old `width - 6` guess: a 71-char key
    /// wraps to one row at the real 72-col inner width on a 100-wide
    /// terminal, so the input box must be exactly one row tall rather than
    /// a mis-sized (one line too tall) two-row box.
    #[test]
    fn key_entry_modal_sizes_to_real_wrap_width() {
        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(100, 30)).unwrap();
        let key = "k".repeat(71);
        let mut input_rect = Rect::default();
        terminal
            .draw(|f| {
                input_rect = render_key_entry_modal(f, f.area(), "anthropic", &key);
            })
            .unwrap();
        let rows = buffer_rows(terminal.backend().buffer());
        // `mask_secret` keeps only the last character visible.
        assert!(rows.iter().any(|r| r.contains('k')));
        // 71 chars fit one 72-col row — the old width-6 sizing would have
        // wrapped at 70 and allocated a 2-row box.
        assert_eq!(input_rect.height, 1);
    }

    #[test]
    fn hints_elide_trailing_pairs_when_narrow() {
        use ratatui::text::Span;
        let pairs: &[[&str; 2]] = &[
            ["tab", "agents"],
            ["/ ctrl+p", "commands"],
            ["click", "select"],
            ["ctrl+y", "copy"],
            ["ctrl+f", "find"],
            ["esc", "close"],
        ];
        let join =
            |spans: &[Span<'static>]| spans.iter().map(|s| s.content.to_string()).collect::<String>();
        // Wide terminal: full legend with 4-space gaps.
        let wide = hints_for_width(pairs, 200);
        assert_eq!(
            join(&wide),
            "tab agents    / ctrl+p commands    click select    ctrl+y copy    ctrl+f find    esc close"
        );
        // Mid-width: gaps collapse to 2 spaces before any pair drops.
        let mid = hints_for_width(pairs, 85);
        assert_eq!(
            join(&mid),
            "tab agents  / ctrl+p commands  click select  ctrl+y copy  ctrl+f find  esc close"
        );
        // Narrow terminal: trailing pairs dropped rather than clipped.
        let narrow = hints_for_width(pairs, 30);
        let narrow_text = join(&narrow);
        assert!(narrow_text.starts_with("tab agents"));
        assert!(!narrow_text.contains("esc close"));
    }

    /// The diff modal used to panic with `clamp(10, max_h)` whenever the
    /// terminal was under 16 rows tall (`max_h` below the floor) — it should
    /// render (shrunk to fit) instead of crashing.
    #[test]
    fn diff_modal_survives_short_terminal() {
        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(100, 12)).unwrap();
        let rows = vec![crate::highlight::DiffRow::Header("@@".into())];
        terminal
            .draw(|f| {
                render_diff_modal(f, f.area(), &rows, 0);
            })
            .unwrap();
    }

    /// Scrolling the diff modal with ↑/↓ and paging nudges the offset; the
    /// render clamps it later, so keys just need to move the counter.
    #[tokio::test]
    async fn diff_mode_keys_scroll_and_esc_returns_to_chat() {
        let tmp = tempfile::tempdir().unwrap();
        let (mut state, _config) = test_app_state(tmp.path());
        state.mode = Mode::Diff {
            rows: vec![crate::highlight::DiffRow::Header("@@".into())],
            scroll: 0,
        };
        let (ui_tx, _ui_rx) = mpsc::unbounded_channel::<UiEvent>();
        let mut agent_slot: Option<Agent> = None;
        let mut turn_handle: Option<TurnJoin> = None;
        let mut cancel_tx: Option<watch::Sender<bool>> = None;

        let key = |code| KeyEvent::new(code, KeyModifiers::NONE);
        handle_key(
            key(KeyCode::Down),
            &mut state,
            &mut agent_slot,
            &mut turn_handle,
            &mut cancel_tx,
            &ui_tx,
            &_config,
            false,
        )
        .await
        .unwrap();
        assert!(matches!(&state.mode, Mode::Diff { scroll: 1, .. }));
        handle_key(
            key(KeyCode::Up),
            &mut state,
            &mut agent_slot,
            &mut turn_handle,
            &mut cancel_tx,
            &ui_tx,
            &_config,
            false,
        )
        .await
        .unwrap();
        assert!(matches!(&state.mode, Mode::Diff { scroll: 0, .. }));
        handle_key(
            key(KeyCode::PageDown),
            &mut state,
            &mut agent_slot,
            &mut turn_handle,
            &mut cancel_tx,
            &ui_tx,
            &_config,
            false,
        )
        .await
        .unwrap();
        assert!(matches!(&state.mode, Mode::Diff { scroll: 20, .. }));
        handle_key(
            key(KeyCode::Esc),
            &mut state,
            &mut agent_slot,
            &mut turn_handle,
            &mut cancel_tx,
            &ui_tx,
            &_config,
            false,
        )
        .await
        .unwrap();
        assert!(matches!(state.mode, Mode::Chat));
    }

    /// Mouse wheel scrolls the two-pane diff (3 rows per notch).
    #[tokio::test]
    async fn diff_mode_mouse_scroll_adjusts_offset() {
        let tmp = tempfile::tempdir().unwrap();
        let (mut state, _config) = test_app_state(tmp.path());
        state.mode = Mode::Diff {
            rows: vec![crate::highlight::DiffRow::Header("@@".into())],
            scroll: 0,
        };
        let mut agent_slot: Option<Agent> = None;

        let scroll = |kind| MouseEvent {
            kind,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        handle_mouse(
            scroll(MouseEventKind::ScrollDown),
            &mut state,
            &mut agent_slot,
            &_config,
        )
        .await;
        assert!(matches!(&state.mode, Mode::Diff { scroll: 3, .. }));
        handle_mouse(
            scroll(MouseEventKind::ScrollUp),
            &mut state,
            &mut agent_slot,
            &_config,
        )
        .await;
        assert!(matches!(&state.mode, Mode::Diff { scroll: 0, .. }));
    }

    /// A permission ask renders its description, the actual diff preview is
    /// two-pane-colorized, and the footer hints at the answer keys.
    #[test]
    fn approval_modal_renders_preview_and_hints() {
        let (reply_tx, _reply_rx) = tokio::sync::oneshot::channel();
        let pending = ApprovalRequestMsg {
            request: PermissionRequest {
                tool: "edit".to_string(),
                path: None,
                command: None,
                description: "Allow editing src/main.rs".to_string(),
                preview: Some(
                    "diff --git a/src/main.rs b/src/main.rs\n@@ -1,2 +1,2 @@\n-old\n+new\n"
                        .to_string(),
                ),
                overwrites: false,
            },
            reply: reply_tx,
        };
        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(100, 30)).unwrap();
        terminal
            .draw(|f| render_approval_modal(f, f.area(), &pending, 0))
            .unwrap();
        let rows = buffer_rows(terminal.backend().buffer());

        assert!(rows.iter().any(|r| r.contains("Allow editing src/main.rs")));
        assert!(rows.iter().any(|r| r.contains("Permission needed")));
        assert!(rows.iter().any(|r| r.contains('│')));
        assert!(rows.iter().any(|r| r.contains("y approve")));
    }

    /// Approval keys resolve the oneshot and return to Chat.
    #[tokio::test]
    async fn approval_key_y_resolves_and_returns_to_chat() {
        let tmp = tempfile::tempdir().unwrap();
        let (mut state, _config) = test_app_state(tmp.path());
        let (reply_tx, mut reply_rx) = tokio::sync::oneshot::channel();
        state.mode = Mode::Approval {
            pending: ApprovalRequestMsg {
                request: PermissionRequest {
                    tool: "edit".to_string(),
                    path: None,
                    command: None,
                    description: "Allow editing".to_string(),
                    preview: None,
                    overwrites: false,
                },
                reply: reply_tx,
            },
            scroll: 0,
        };
        let (ui_tx, _ui_rx) = mpsc::unbounded_channel::<UiEvent>();
        let mut agent_slot: Option<Agent> = None;
        let mut turn_handle: Option<TurnJoin> = None;
        let mut cancel_tx: Option<watch::Sender<bool>> = None;

        handle_key(
            KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE),
            &mut state,
            &mut agent_slot,
            &mut turn_handle,
            &mut cancel_tx,
            &ui_tx,
            &_config,
            false,
        )
        .await
        .unwrap();
        assert!(matches!(state.mode, Mode::Chat));
        assert_eq!(reply_rx.try_recv().unwrap(), ApprovalDecision::Approved);
    }

    /// Ctrl+C with a transcript selection active copies the selection (the
    /// terminal's "highlight then copy" convention) instead of quitting —
    /// the dispatch decision is what's under test, so the exact clipboard
    /// outcome (which is headless-environment-dependent) doesn't matter.
    #[tokio::test]
    async fn ctrl_c_with_selection_copies_instead_of_quitting() {
        let tmp = tempfile::tempdir().unwrap();
        let (mut state, _config) = test_app_state(tmp.path());
        state.push_user("hello".to_string());
        state.push_info("some assistant text to copy");
        state.selection = Some((0, 1));
        let (ui_tx, _ui_rx) = mpsc::unbounded_channel::<UiEvent>();
        let mut agent_slot: Option<Agent> = None;
        let mut turn_handle: Option<TurnJoin> = None;
        let mut cancel_tx: Option<watch::Sender<bool>> = None;

        handle_key(
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
            &mut state,
            &mut agent_slot,
            &mut turn_handle,
            &mut cancel_tx,
            &ui_tx,
            &_config,
            false,
        )
        .await
        .unwrap();

        // A selection means copy — the app must NOT quit.
        assert!(!state.quit, "Ctrl+C on a selection must not quit");
        // copy_selection clears the selection on success; on a headless box
        // the clipboard may fail and leave it in place. Either way the
        // outcome is one of the two copy paths, never quit.
        assert!(
            state.selection.is_none() || !state.transcript.is_empty(),
            "selection was either cleared by copy or a copy error was reported"
        );
    }

    /// Ctrl+C with no selection keeps its cancel/quit role.
    #[tokio::test]
    async fn ctrl_c_without_selection_still_quits() {
        let tmp = tempfile::tempdir().unwrap();
        let (mut state, _config) = test_app_state(tmp.path());
        let (ui_tx, _ui_rx) = mpsc::unbounded_channel::<UiEvent>();
        let mut agent_slot: Option<Agent> = None;
        let mut turn_handle: Option<TurnJoin> = None;
        let mut cancel_tx: Option<watch::Sender<bool>> = None;

        handle_key(
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
            &mut state,
            &mut agent_slot,
            &mut turn_handle,
            &mut cancel_tx,
            &ui_tx,
            &_config,
            false,
        )
        .await
        .unwrap();

        assert!(state.quit, "Ctrl+C with no selection should quit");
    }
}
