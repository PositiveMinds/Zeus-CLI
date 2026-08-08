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
//! blocks on a `std::sync::mpsc` reply channel until a keypress here answers
//! it — an intentional, bounded exception to "never block an async task",
//! justified because waiting on a human is inherently slow anyway.

use crate::{
    build_agent_repl, expand_slash_command, known_slash_commands, list_models_by_provider,
    persist_default_provider, print_repl_help_lines,
};
use anyhow::{Context, Result};
use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
    MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use std::io;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use zeus_agent::{personas_by_department, Agent, AgentEvent, TurnResult};
use zeus_config::{Config, KeysFile};
use zeus_fs::{ApprovalDecision, PermissionRequest};
use zeus_provider::{create_provider, ModelInfo};

enum Role {
    User,
    Assistant,
    Tool,
    ToolError,
    Info,
    Error,
}

impl Role {
    /// Avatar glyph shown before the message body for the first line only
    /// (mirrors the zeus-cli.html `YOUR`/`⚡` avatar chips).
    fn marker(&self) -> &'static str {
        match self {
            Role::User => "YOU ",
            Role::Assistant => "⚡ ",
            Role::Tool => "◆ ",
            Role::ToolError => "✗ ",
            Role::Info => "· ",
            Role::Error => "✗ ",
        }
    }

    fn marker_style(&self) -> Style {
        match self {
            Role::User => theme::dim(),
            Role::Assistant => theme::violet().add_modifier(Modifier::BOLD),
            Role::Tool => theme::cyan(),
            Role::ToolError => theme::red(),
            Role::Info => theme::faint(),
            Role::Error => theme::red(),
        }
    }

    fn text_style(&self) -> Style {
        match self {
            Role::User => theme::user_text(),
            Role::Assistant => theme::text(),
            Role::Tool => theme::cyan(),
            Role::ToolError => theme::red(),
            Role::Info => theme::dim(),
            Role::Error => theme::red(),
        }
    }
}

struct Block_ {
    role: Role,
    /// Plain body text. When `lines` is empty, `to_lines` derives the rows
    /// from this (splitting on newlines); otherwise `lines` wins (used for
    /// the seeded demo transcript with tool-call chips and diff blocks).
    text: String,
    /// Pre-built styled rows (empty for plain text blocks).
    lines: Vec<Vec<Span<'static>>>,
}

impl Block_ {
    fn new(role: Role, text: String) -> Self {
        Self { role, text, lines: Vec::new() }
    }

    fn rich(role: Role, lines: Vec<Vec<Span<'static>>>) -> Self {
        Self { role, text: String::new(), lines }
    }

    /// Plain text of this block: the stored `text` when present, otherwise the
    /// concatenated content of its styled spans (e.g. a copied diff).
    fn plain_text(&self) -> String {
        if !self.text.is_empty() {
            return self.text.clone();
        }
        self.lines
            .iter()
            .map(|spans| {
                spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Indent applied to continuation lines so they clear the avatar marker.
    fn pad() -> &'static str {
        "    "
    }

    fn to_lines(&self) -> Vec<Line<'static>> {
        let marker_style = self.role.marker_style();
        let marker = self.role.marker();
        let text_style = self.role.text_style();
        let mut raw_lines: Vec<&str> = if self.lines.is_empty() {
            self.text.lines().collect()
        } else {
            Vec::new()
        };
        if raw_lines.is_empty() && self.lines.is_empty() {
            raw_lines.push("");
        }

        let mut out: Vec<Line<'static>> = Vec::new();
        if !self.lines.is_empty() {
            for (i, spans) in self.lines.iter().enumerate() {
                let mut line_spans = Vec::new();
                if i == 0 {
                    line_spans.push(Span::styled(marker, marker_style));
                    if self.role == Role::Assistant {
                        // violet left rule like the HTML bubble's border-left
                        line_spans.push(Span::styled("▍", theme::violet().add_modifier(Modifier::BOLD)));
                        line_spans.push(Span::raw(" "));
                    }
                } else {
                    line_spans.push(Span::styled(Self::pad(), theme::faint()));
                }
                line_spans.extend(spans.iter().cloned());
                out.push(Line::from(line_spans));
            }
            return out;
        }

        // Assistant replies may embed fenced code blocks — tokenize and color
        // them instead of emitting everything in a single flat span.
        let highlighted: Option<Vec<Vec<Span<'static>>>> = (self.role == Role::Assistant)
            .then(|| super::highlight::markdown_lines(&self.text, text_style));
        if let Some(lines) = highlighted {
            if !lines.is_empty() {
                for (i, spans) in lines.iter().enumerate() {
                    let mut line_spans = Vec::new();
                    if i == 0 {
                        line_spans.push(Span::styled(marker, marker_style));
                        line_spans.push(Span::styled("▍", theme::violet().add_modifier(Modifier::BOLD)));
                        line_spans.push(Span::raw(" "));
                    } else {
                        line_spans.push(Span::styled(Self::pad(), theme::faint()));
                    }
                    line_spans.extend(spans.iter().cloned());
                    out.push(Line::from(line_spans));
                }
                return out;
            }
        }

        for (i, l) in raw_lines.iter().enumerate() {
            let mut spans = Vec::new();
            if i == 0 {
                spans.push(Span::styled(marker, marker_style));
                if self.role == Role::Assistant {
                    spans.push(Span::styled("▍", theme::violet().add_modifier(Modifier::BOLD)));
                    spans.push(Span::raw(" "));
                }
            } else {
                spans.push(Span::styled(Self::pad(), theme::faint()));
            }
            spans.push(Span::styled(l.to_string(), text_style));
            out.push(Line::from(spans));
        }
        out
    }
}

/// A temporary seeded preview: the demo conversation from `zeus-cli.html`,
/// reproduced verbatim (messages, tool-call chips, and a diff block) so the
/// bubble design can be confirmed in the TUI. **Seeding preview only** — the
/// intent is to remove this once the layout is signed off.
fn demo_transcript() -> Vec<Block_> {
    let dblk = Color::Rgb(0x0a, 0x0c, 0x12);
    let dadd = Color::Rgb(0x0c, 0x12, 0x0e);
    let ddel = Color::Rgb(0x0c, 0x10, 0x10);
    let dchip = Color::Rgb(0x0a, 0x16, 0x1f); // faint cyan ≈ --cyan at 7% on --void

    // <code> spans render cyan, matching `.bubble code { color: var(--cyan) }`.
    let text = |s: &str| Span::styled(s.to_string(), theme::text());
    let code = |s: &str| Span::styled(s.to_string(), theme::cyan());
    // A tool-call chip: cyan "●" + args on a faint-cyan pill, like `.toolcall`.
    let chip = |args: &str| {
        vec![
            Span::styled(
                "● ",
                Style::default().fg(theme::CYAN).add_modifier(Modifier::BOLD).bg(dchip),
            ),
            Span::styled(args.to_string(), Style::default().fg(theme::CYAN).bg(dchip)),
        ]
    };

    let mut blocks = Vec::new();

    blocks.push(Block_::new(
        Role::User,
        "the /api/users endpoint is throwing a 500 on pagination past page 3. can you find it and fix it?"
            .into(),
    ));

    // Assistant #1 with tool chips, inline code, and a diff block.
    let asst1: Vec<Vec<Span<'static>>> = vec![
        vec![text("Looking at the route handler and the query builder now.")],
        chip("search_files(\"pagination\", scope: \"src/api/users\")"),
        chip("read_file(\"src/api/users/handler.ts\")"),
        vec![
            text("Found it — "),
            code("offset"),
            text(" is computed as "),
            code("page * limit"),
            text(" instead of "),
            code("(page - 1) * limit"),
            text(", so once you pass the third page it skips past the end of a small result set and the count query throws. Here's the fix:"),
        ],
        // diffblock
        vec![
            Span::styled("  src/api/users/handler.ts", Style::default().fg(theme::DIM).bg(dblk)),
            Span::styled("       @@ -18,7 +18,7 @@", Style::default().fg(theme::DIM).bg(dblk)),
        ],
        { let dctx = |s: &str| vec![Span::styled(s.to_string(), Style::default().fg(theme::DIM).bg(dblk))]; dctx("  const limit = Number(query.limit) || 20;") },
        { let ddell = |s: &str| vec![Span::styled(s.to_string(), Style::default().fg(theme::RED).bg(ddel))]; ddell("- const offset = page * limit;") },
        { let daddl = |s: &str| vec![Span::styled(s.to_string(), Style::default().fg(theme::GREEN).bg(dadd))]; daddl("+ const offset = (page - 1) * limit;") },
        { let dctx = |s: &str| vec![Span::styled(s.to_string(), Style::default().fg(theme::DIM).bg(dblk))]; dctx("  const rows = await db.users.findMany({ take: limit, skip: offset });") },
        vec![text(
            "Added it to today's TODOs on the right so it doesn't get lost before you review the diff.",
        )],
    ];
    blocks.push(Block_::rich(Role::Assistant, asst1));

    blocks.push(Block_::new(
        Role::User,
        "good catch. apply it and run the tests".into(),
    ));

    // asst2: two tool calls then a short reply.
    let asst2: Vec<Vec<Span<'static>>> = vec![
        chip("apply_patch(\"src/api/users/handler.ts\")"),
        chip("run(\"npm test -- users.pagination\")"),
        vec![text(
            "All 6 pagination tests pass, including the new edge case for a partial last page. Want me to open a commit for this, or keep it staged for a bigger PR?",
        )],
    ];
    blocks.push(Block_::rich(Role::Assistant, asst2));

    blocks
}

impl PartialEq for Role {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (Role::User, Role::User)
                | (Role::Assistant, Role::Assistant)
                | (Role::Tool, Role::Tool)
                | (Role::ToolError, Role::ToolError)
                | (Role::Info, Role::Info)
                | (Role::Error, Role::Error)
        )
    }
}

/// A pending permission ask, bridged out of the spawned turn task.
struct ApprovalRequestMsg {
    request: PermissionRequest,
    reply: std::sync::mpsc::Sender<ApprovalDecision>,
}

enum UiEvent {
    Agent(AgentEvent),
    Approval(ApprovalRequestMsg),
}

enum Mode {
    Chat,
    Approval(ApprovalRequestMsg),
    ModelPicker { entries: Vec<PickerEntry>, selected: usize },
    /// Grouped provider picker (paid / free / local) — arrow keys to move,
    /// Enter to select. Selecting a provider without a key opens `KeyEntry`.
    ProviderPicker { entries: Vec<ProviderEntry>, selected: usize },
    /// Pasting an API key for the named provider. Enter saves it (persisted
    /// to keys.toml, env var set, provider switched) and returns to Chat.
    KeyEntry { provider: String },
}

/// One row in the model picker: a non-selectable provider-group header, or
/// a selectable model belonging to that provider. Kept as a flat list (with
/// header rows navigation skips over) rather than nested groups, so a
/// single `ListState`/`selected` index still works for both keyboard and
/// mouse selection.
enum PickerEntry {
    Header(String),
    Model { provider: String, model: ModelInfo },
}

/// One row in the provider picker: a non-selectable group header (paid /
/// free / local), a selectable provider, or a selectable model belonging to
/// that provider. Flat list so a single index drives keyboard and mouse
/// selection.
#[derive(Clone)]
enum ProviderEntry {
    Header(String),
    Provider {
        name: String,
        kind: String,
        model: String,
        /// True when the provider can be used right now (local kind, stored
        /// key, or env key present) — false means "needs a key" and Enter
        /// jumps to `KeyEntry` instead of switching.
        ready: bool,
    },
    Model {
        provider: String,
        model: ModelInfo,
        /// Tagged free (heuristic on the model id) vs paid.
        free: bool,
    },
}

/// Moves `selected` one step in the given direction (`1` or `-1`), skipping
/// over header rows, wrapping around the ends. Safe as long as `entries`
/// contains at least one `Model` row (always true by construction — a
/// header is only ever pushed alongside its models).
fn picker_move(entries: &[PickerEntry], selected: usize, dir: isize) -> usize {
    let len = entries.len() as isize;
    if len == 0 {
        return 0;
    }
    let mut idx = selected as isize;
    loop {
        idx = (idx + dir).rem_euclid(len);
        if matches!(entries[idx as usize], PickerEntry::Model { .. }) {
            return idx as usize;
        }
    }
}

/// Same navigation for the provider picker, skipping its group headers but
/// allowing both provider and model rows to be selected.
fn provider_picker_move(entries: &[ProviderEntry], selected: usize, dir: isize) -> usize {
    let len = entries.len() as isize;
    if len == 0 {
        return 0;
    }
    let mut idx = selected as isize;
    loop {
        idx = (idx + dir).rem_euclid(len);
        if !matches!(entries[idx as usize], ProviderEntry::Header(_)) {
            return idx as usize;
        }
    }
}

/// Move the dropdown highlight one row toward the end, skipping group
/// headers (not selectable). Wraps around the list.
fn dropdown_next_selectable(entries: &[ProviderEntry], selected: usize) -> usize {
    if entries.is_empty() {
        return 0;
    }
    let mut idx = selected;
    for _ in 0..entries.len() {
        idx = (idx + 1) % entries.len();
        if !matches!(entries.get(idx), Some(ProviderEntry::Header(_))) {
            return idx;
        }
    }
    selected
}

/// Move the dropdown highlight one row toward the start, skipping group
/// headers (not selectable). Wraps around the list.
fn dropdown_prev_selectable(entries: &[ProviderEntry], selected: usize) -> usize {
    if entries.is_empty() {
        return 0;
    }
    let mut idx = selected;
    for _ in 0..entries.len() {
        idx = (idx + entries.len() - 1) % entries.len();
        if !matches!(entries.get(idx), Some(ProviderEntry::Header(_))) {
            return idx;
        }
    }
    selected
}

/// Applies the chosen model: switches provider first if it differs from the
/// one currently in use (conversation history carries over — only the
/// provider handle and model name change), then sets the model. Closes the
/// picker either way; a provider-construction failure is reported in the
/// transcript rather than left unexplained.
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
    state.model = model_id;
}

/// Persist an API key for a provider (to `~/.zeus/keys.toml`), apply it as
/// the provider's env var for the running session, then switch the agent to
/// that provider's default model. Pushes a transcript line on success/failure.
fn persist_key_and_switch(
    provider: &str,
    key: &str,
    config: &Config,
    agent_slot: &mut Option<Agent>,
    state: &mut AppState,
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
    if let Some(cfg) = config.providers.get(provider) {
        if let Some(var) = &cfg.api_key_env {
            std::env::set_var(var, key);
        }
    }
    match create_provider(provider, &config.providers) {
        Ok(handle) => {
            if let Some(agent) = agent_slot.as_mut() {
                agent.set_provider(handle);
            }
            let model = config
                .providers
                .get(provider)
                .and_then(|c| c.default_model.clone())
                .unwrap_or_else(|| state.model.clone());
            if let Some(agent) = agent_slot.as_mut() {
                agent.set_model(model.clone());
            }
            state.provider = provider.to_string();
            state.model = model.clone();
            let saved = config.global.keys_toml.display();
            match persist_default_provider(config, provider, Some(&model)) {
                Ok(path) => state.push_info(format!(
                    "key saved for '{provider}' ({saved}) — switched to {provider} / {model} (default saved to {})",
                    path.display()
                )),
                Err(e) => state.push_info(format!(
                    "key saved for '{provider}' ({saved}) — switched to {provider} / {model}, but saving default failed: {e:#}"
                )),
            }
        }
        Err(e) => state.push_error(format!("couldn't switch to '{provider}': {e:#}")),
    }
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

/// Apply a pick inside the top-right provider dropdown. Ready providers/model
/// switch immediately (dropdown closes); a provider that still needs a key
/// keeps the dropdown open but swaps it into the inline API-key entry view.
fn dropdown_apply(
    state: &mut AppState,
    agent_slot: &mut Option<Agent>,
    config: &Config,
    entry: &ProviderEntry,
) {
    match entry {
        ProviderEntry::Provider { name, ready, .. } => {
            if *ready {
                apply_provider_picker_choice(name.clone(), true, config, agent_slot, state);
                state.dropdown = None;
            } else {
                open_dropdown_key(state, name);
            }
        }
        ProviderEntry::Model { provider, model, .. } => {
            if provider_status_ok(config, provider) {
                apply_picker_choice(provider.clone(), model.id.clone(), state, agent_slot, config);
                state.dropdown = None;
            } else {
                open_dropdown_key(state, provider);
            }
        }
        ProviderEntry::Header(_) => state.dropdown = None,
    }
}

/// Swap the dropdown's list for its inline key-entry view for `provider`
/// (preserving the entry list so Esc can step back out).
fn open_dropdown_key(state: &mut AppState, provider: &str) {
    if let Some(d) = state.dropdown.as_mut() {
        d.keying = Some(KeyingState { provider: provider.to_string(), key: String::new() });
        d.selected = 0;
    }
}

/// Free-vs-paid tag for a fetched model id. Generous free heuristic — matches
/// the common free/low-cost tiers across providers (opencodezen's
/// deepseek-v4-flash-free, gemini flash, gpt mini, lite/nano variants, and
/// openrouter's `:free` suffixes) so as many genuinely free models as possible
/// surface as green in the picker.
fn is_free_model(id: &str) -> bool {
    let id = id.to_ascii_lowercase();
    const FREE_SUBSTRINGS: &[&str] = &[
        "free", "flash", "mini", "lite", "nano", "tiny", "light", "small", "1b", "3b", "8b",
    ];
    FREE_SUBSTRINGS.iter().any(|s| id.contains(s))
}

/// Build grouped provider-picker entries: Local, Free, then Paid. Each group
/// gets a header row; providers carry their kind, default model, and whether
/// they're immediately usable (local kind, stored key, or env key set). The
/// caller's current provider's row is preselected. Reachable providers have
/// their real models appended underneath (tagged free/paid); providers that
/// can't list models (no key, server down) still show as a switchable row.
async fn provider_picker_entries(
    config: &Config,
    current: &str,
    current_model: &str,
) -> (Vec<ProviderEntry>, usize) {
    let group_of = |kind: &str| -> &'static str {
        if matches!(kind, "ollama" | "lmstudio" | "llamacpp") {
            "Local"
        } else if kind == "opencodezen" {
            "Free"
        } else {
            "Paid"
        }
    };
    let mut names: Vec<&String> = config.providers.providers.keys().collect();
    names.sort();
    let models = list_models_by_provider(config).await;
    let mut entries = Vec::new();
    let mut selected = 0;
    for group in ["Local", "Free", "Paid"] {
        let members: Vec<&&String> = names
            .iter()
            .filter(|n| {
                config
                    .providers
                    .get(n)
                    .map(|c| group_of(&c.kind) == group)
                    .unwrap_or(false)
            })
            .collect();
        if members.is_empty() {
            continue;
        }
        entries.push(ProviderEntry::Header(group.to_string()));
        for name in members {
            let Some(cfg) = config.providers.get(name) else { continue };
            let ready = provider_status_ok(config, name);
            if name.as_str() == current && ready {
                selected = entries.len();
            }
            entries.push(ProviderEntry::Provider {
                name: name.to_string(),
                kind: cfg.kind.clone(),
                model: cfg.default_model.clone().unwrap_or_default(),
                ready,
            });
            // Models for this provider, if reachable, right under its row.
            if let Some((_, provider_models)) =
                models.iter().find(|(n, _)| n == (name as &String))
            {
                for m in provider_models {
                    if m.id == current_model && name.as_str() == current {
                        selected = entries.len();
                    }
                    entries.push(ProviderEntry::Model {
                        provider: name.to_string(),
                        model: m.clone(),
                        free: is_free_model(&m.id),
                    });
                }
            }
        }
    }
    (entries, selected)
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
        [] => {
            let (entries, selected) =
                provider_picker_entries(config, &state.provider, &state.model).await;
            if entries.is_empty() {
                state.push_error("no providers configured — see config.toml / providers.toml");
            } else {
                state.mode = Mode::ProviderPicker { entries, selected };
            }
        }
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
        ["key"] => state.push_error("usage: /provider key <name> <KEY>"),
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
#[derive(Clone, Copy, PartialEq, Eq)]
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

/// A mode chip (filled with the mode's accent, dark label) for the segmented
/// control and the input status line.
fn mode_chip(mode: AgentMode) -> Style {
    Style::default()
        .fg(theme::VOID)
        .bg(mode_accent(mode))
        .add_modifier(Modifier::BOLD)
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
    /// Bare-provider dropdown (top-right corner, like the HTML page) — same
    /// `ProviderEntry` shape as the `/provider` picker but rendered as a
    /// compact popover under the provider button. `None` when hidden.
    dropdown: Option<DropdownState>,
    /// TODO checklist for the right sidebar (mirrors the HTML sample data).
    todos: Vec<TodoItem>,
    /// The provider-picker dropdown has a free-text search bar, like the
    /// HTML `.provider-search` box.
    drop_search: String,
    /// Rendered rect of the TODO list — maps mouse clicks to rows.
    todo_area: Option<Rect>,
    /// Rendered rect of the top-right provider button — maps a click to
    /// opening/closing the dropdown.
    provider_btn_area: Option<Rect>,
    /// When the session started — drives the "Session" readout in the sidebar.
    started: std::time::Instant,
}

struct DropdownState {
    entries: Vec<ProviderEntry>,
    selected: usize,
    /// Where the popover gets drawn. `None` until the first render computes it.
    area: Option<Rect>,
    /// When set, the dropdown swaps its list for an inline API-key entry for
    /// the given provider (still inside the same popover), instead of kicking
    /// the user out to the full-screen `KeyEntry` mode.
    keying: Option<KeyingState>,
}

struct KeyingState {
    provider: String,
    key: String,
}

struct TodoItem {
    text: String,
    done: bool,
    active: bool,
}

/// Builds the compact provider/model rows for the top-right dropdown,
/// regrouping configured providers so the current one leads, then the rest in
/// alphabetical order — same scanning approach as the full `/provider` picker.
async fn build_dropdown_entries(
    config: &Config,
    current: &str,
    _current_model: &str,
) -> Vec<ProviderEntry> {
    let mut names: Vec<&String> = config.providers.providers.keys().collect();
    names.sort();
    if let Some(pos) = names.iter().position(|n| n.as_str() == current) {
        let cur = names.remove(pos);
        names.insert(0, cur);
    }
    let models = list_models_by_provider(config).await;
    let mut entries = Vec::new();
    let mut any = false;
    for name in names {
        let Some(cfg) = config.providers.get(name) else { continue };
        let ready = provider_status_ok(config, name);
        any = true;
        entries.push(ProviderEntry::Header(name.to_string()));
        entries.push(ProviderEntry::Provider {
            name: name.to_string(),
            kind: cfg.kind.clone(),
            model: cfg.default_model.clone().unwrap_or_default(),
            ready,
        });
        if let Some((_, provider_models)) = models.iter().find(|(n, _)| n == (name as &String)) {
            for m in provider_models {
                entries.push(ProviderEntry::Model {
                    provider: name.to_string(),
                    model: m.clone(),
                    free: is_free_model(&m.id),
                });
            }
        }
    }
    if any {
        entries.push(ProviderEntry::Header("manage providers — /provider".to_string()));
    }
entries
}

/// Git branch shown in the side-panel session footer. Single field only —
/// other directory facts (workspace/path/file count) aren't rendered yet, so
/// they were dropped to avoid dead fields.
pub struct DirInfo {
    pub git_branch: Option<String>,
}

impl AppState {
    fn new(agent: &Agent, known_commands: Vec<(String, String)>, dir: DirInfo, start_in_plan: bool) -> Self {
        let state = Self {
            transcript: demo_transcript(),
            current_reply: String::new(),
            input: String::new(),
            cursor: 0,
            busy: false,
            quit: false,
            mode: Mode::Chat,
            agent_mode: if start_in_plan { AgentMode::Plan } else { AgentMode::Build },
            model: agent.model().to_string(),
            provider: agent.provider_id().to_string(),
            session_id: agent.session_id().to_string(),
            known_commands,
            dir,
            command_selected: 0,
            model_picker_area: None,
            provider_picker_area: None,
            dropdown: None,
            todos: Vec::new(),
            drop_search: String::new(),
            todo_area: None,
            provider_btn_area: None,
            started: std::time::Instant::now(),
        };
        state
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
            AgentEvent::ToolCallStarted { name, arguments, .. } => {
                self.flush_current_reply();
                self.transcript.push(Block_::new(
                    Role::Tool,
                    format!("{name} {arguments}"),
                ));
            }
            AgentEvent::ToolCallFinished { name, result, is_error, .. } => {
                self.flush_current_reply();
                let role = if is_error { Role::ToolError } else { Role::Tool };
                let marker = if is_error { "failed" } else { "done" };
                self.transcript.push(Block_::new(
                    role,
                    format!("{name} ({marker})\n{result}"),
                ));
                // A completed mutation auto-checks the in-flight task, filling
                // the sidebar progress bar (the HTML page's AI-driven TODOs).
                if !is_error && matches!(name.as_str(), "apply_patch" | "write" | "edit" | "update" | "run" | "patch") {
                    self.complete_task(&name);
                }
            }
            AgentEvent::Compacted(c) => {
                self.push_info(format!("(compacted {} earlier message(s))", c.removed_messages));
            }
            AgentEvent::Cancelled => self.push_info("(cancelled)"),
            AgentEvent::Done => self.flush_current_reply(),
            AgentEvent::PlanGenerated { steps } => {
                self.push_info(format!(
                    "plan · {} step(s): {}",
                    steps.len(),
                    steps
                        .iter()
                        .map(|s| s.description.clone())
                        .collect::<Vec<_>>()
                        .join(" → ")
                ));
            }
            AgentEvent::PlanStepStarted { step } => {
                self.push_info(format!("plan step {} · {}", step.id, step.description));
            }
            AgentEvent::PlanReviewed { persona, report } => {
                self.transcript.push(Block_::new(
                    Role::Tool,
                    format!("review ({persona})\n{report}"),
                ));
            }
            AgentEvent::PlanStepDone { step, summary } => {
                self.transcript.push(Block_::new(
                    Role::Tool,
                    format!("step {} done · {}\n{}", step.id, step.description, summary),
                ));
            }
            AgentEvent::PlanStepDeclined { step } => {
                self.transcript.push(Block_::new(
                    Role::Tool,
                    format!("step {} declined · {}", step.id, step.description),
                ));
            }
            AgentEvent::OrchestrationDone { summary } => {
                self.flush_current_reply();
                self.transcript.push(Block_::new(Role::Assistant, summary));
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
                self.todos.last().map(|x| x.text.clone()).unwrap_or_default()
            };
            if !self.todos.iter().any(|x| x.done && x.text == text) {
                self.todos.push(TodoItem { text, done: true, active: false });
            }
        }
    }

    fn toggle_todo(&mut self, idx: usize) {
        let Some(t) = self.todos.get_mut(idx) else { return };
        t.done = !t.done;
        if t.done {
            t.active = false;
        }
    }

    fn command_matches(&self) -> Vec<(&str, &str)> {
        match self.input.strip_prefix('/') {
            Some(prefix) if !prefix.contains(char::is_whitespace) => self
                .known_commands
                .iter()
                .map(|(n, d)| (n.as_str(), d.as_str()))
                .filter(|(n, _)| n.starts_with(prefix))
                .collect(),
            _ => Vec::new(),
        }
    }
}

fn char_count(s: &str) -> usize {
    s.chars().count()
}

fn byte_index(s: &str, char_idx: usize) -> usize {
    s.char_indices().nth(char_idx).map(|(i, _)| i).unwrap_or(s.len())
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

/// The six-letter-row "ZEUS" wordmark, hacker-green block letters — a static
/// logo (no animation; an earlier Matrix-rain splash was replaced after
/// feedback that the animation felt out of place) styled after opencode's
/// blocky splash wordmark.
/// Warm-to-cool "AI" gradient (orange → pink → purple → blue → cyan),
/// matching the sweep across Antigravity CLI's logo — `t` in `0.0..=1.0`.
fn placeholder_style() -> Style {
    theme::faint()
}

/// zeus-cli.html palette — variables taken 1:1 from the reference page's
/// `:root` block (`--void` … `--red`) plus the mode accent colors, so the
/// TUI reproduces the HTML mockup exactly.
mod theme {
    use ratatui::style::{Color, Style};

    /// `--void`: page background (`#07080d`).
    pub const VOID: Color = Color::Rgb(0x07, 0x08, 0x0d);
    /// `--panel`: panel background (`#0d1017`).
    pub const PANEL: Color = Color::Rgb(0x0d, 0x10, 0x17);
    /// `--panel-2`: slightly lighter panel (`#11141c`).
    pub const PANEL2: Color = Color::Rgb(0x11, 0x14, 0x1c);
    /// `--elevated`: dropdown/elevated surface (`#161a24`).
    pub const ELEVATED: Color = Color::Rgb(0x16, 0x1a, 0x24);
    /// `--border`: borders (`#20242f`).
    pub const BORDER: Color = Color::Rgb(0x20, 0x24, 0x2f);
    /// `--border-soft`: soft borders (`#181c26`).
    pub const BORDER_SOFT: Color = Color::Rgb(0x18, 0x1c, 0x26);
    /// `--text`: primary text (`#e7e9f2`).
    pub const TEXT: Color = Color::Rgb(0xe7, 0xe9, 0xf2);
    /// `--text-dim`: secondary text (`#767c8f`).
    pub const DIM: Color = Color::Rgb(0x76, 0x7c, 0x8f);
    /// `--text-faint`: tertiary text (`#454a59`).
    pub const FAINT: Color = Color::Rgb(0x45, 0x4a, 0x59);
    /// `--violet`: brand accent (`#a855ff`).
    pub const VIOLET: Color = Color::Rgb(0xa8, 0x55, 0xff);
    /// `--violet-dim`: muted violet (`#6d2fb8`).
    pub const VIOLET_DIM: Color = Color::Rgb(0x6d, 0x2f, 0xb8);
    /// `--cyan` (`#22d3ee`) — Build mode accent + tool calls.
    pub const CYAN: Color = Color::Rgb(0x22, 0xd3, 0xee);
    /// `--gold` (`#fbbf24`) — Plan mode accent.
    pub const GOLD: Color = Color::Rgb(0xfb, 0xbf, 0x24);
    /// `--magenta` (`#ff3d9a`) — Auto mode accent.
    pub const MAGENTA: Color = Color::Rgb(0xff, 0x3d, 0x9a);
    /// `--green` (`#34d399`).
    pub const GREEN: Color = Color::Rgb(0x34, 0xd3, 0x99);
    /// `--red` (`#ff5f6d`).
    pub const RED: Color = Color::Rgb(0xff, 0x5f, 0x6d);

    pub fn violet() -> Style {
        Style::default().fg(VIOLET)
    }
    pub fn cyan() -> Style {
        Style::default().fg(CYAN)
    }
    pub fn gold() -> Style {
        Style::default().fg(GOLD)
    }
    pub fn green() -> Style {
        Style::default().fg(GREEN)
    }
    pub fn red() -> Style {
        Style::default().fg(RED)
    }
    pub fn text() -> Style {
        Style::default().fg(TEXT)
    }
    pub fn dim() -> Style {
        Style::default().fg(DIM)
    }
    pub fn faint() -> Style {
        Style::default().fg(FAINT)
    }
    /// User bubble text (`#c7cadb`).
    pub fn user_text() -> Style {
        Style::default().fg(Color::Rgb(0xc7, 0xca, 0xdb))
    }
}

fn status_style() -> Style {
    theme::dim()
}

/// Violet border used by the centered picker popups.
fn border_style() -> Style {
    theme::violet()
}

fn menu_height(matches: &[(&str, &str)]) -> u16 {
    if matches.is_empty() {
        0
    } else {
        matches.len().min(8) as u16 + 2
    }
}

/// Slash-command dropdown — command names in violet bold, descriptions dim,
/// highlight bar on the selected row (mirrors the HTML `.palette-item`
/// rows with their gold command labels).
fn render_menu(f: &mut Frame, area: Rect, matches: &[(&str, &str)], selected: usize) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::BORDER))
        .style(Style::default().bg(theme::ELEVATED));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let name_width = matches.iter().map(|(n, _)| n.len()).max().unwrap_or(0).max(8);
    let items: Vec<ListItem> = matches
        .iter()
        .map(|(name, desc)| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("/{name:<name_width$}"), theme::violet().add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled(desc.to_string(), placeholder_style()),
            ]))
        })
        .collect();
    let list = List::new(items).highlight_style(
        Style::default().fg(theme::GOLD).add_modifier(Modifier::BOLD),
    );
    let mut list_state = ListState::default();
    list_state.select(Some(selected.min(matches.len().saturating_sub(1))));
    f.render_stateful_widget(list, inner, &mut list_state);
}

/// The pinned input box — the HTML `.inputbar`: a mode-colored caret `›`,
/// the placeholder "Message Zeus, or type / for commands…", and a filled
/// mode-accent SEND button on the right. The border + caret + button all
/// repaint when the mode switches.
fn render_input_box(f: &mut Frame, area: Rect, state: &AppState) {
    let accent = mode_accent(state.agent_mode);
    let focused = !state.busy && matches!(state.mode, Mode::Chat);
    let border = if focused { accent } else { theme::BORDER };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border))
        .style(Style::default().bg(theme::PANEL));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).split(inner);

    if let Mode::Approval(pending) = &state.mode {
        let line = Line::from(vec![
            Span::styled("⚠ ", theme::gold().add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("Allow {}?", pending.request.description),
                theme::text().add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("[y] approve · [n] deny · [s] session", theme::faint()),
        ]);
        f.render_widget(Paragraph::new(line), rows[0]);
        let preview = pending.request.preview.as_deref().unwrap_or("");
        let preview_lines: Vec<Line> = if super::highlight::looks_like_diff(preview) {
            super::highlight::diff_lines(preview, placeholder_style())
                .into_iter()
                .map(Line::from)
                .collect()
        } else {
            vec![Line::from(Span::styled(
                preview.lines().next().unwrap_or(""),
                placeholder_style(),
            ))]
        };
        let preview_paragraph = preview_lines.first().cloned();
        if let Some(p) = preview_paragraph {
            f.render_widget(Paragraph::new(p), rows[1]);
        }
        return;
    }

    if let Mode::KeyEntry { provider } = &state.mode {
        let line = Line::from(vec![
            Span::styled("🔑 ", theme::gold().add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("paste API key for {provider}:"),
                theme::text().add_modifier(Modifier::BOLD),
            ),
        ]);
        f.render_widget(Paragraph::new(line), rows[0]);
        let key_line = if state.input.is_empty() {
            Line::from(Span::styled(
                "sk-… (ctrl+v to paste, then enter)",
                placeholder_style(),
            ))
        } else {
            Line::from(Span::raw(state.input.clone()))
        };
        f.render_widget(Paragraph::new(key_line), rows[1]);
        return;
    }

    let caret = Span::styled(
        "› ",
        Style::default().fg(accent).add_modifier(Modifier::BOLD),
    );
    let input_line = if state.busy {
        Line::from(vec![
            caret,
            Span::styled(
                format!("{} zeus is working…", spinner_glyph(state)),
                placeholder_style(),
            ),
        ])
    } else if state.input.is_empty() {
        Line::from(vec![
            caret,
            Span::styled("Message Zeus, or type / for commands…", placeholder_style()),
        ])
    } else {
        Line::from(vec![caret, Span::raw(state.input.clone())])
    };

    let send_style = if state.busy {
        Style::default().fg(theme::FAINT).bg(theme::BORDER_SOFT)
    } else {
        Style::default().fg(theme::VOID).bg(accent).add_modifier(Modifier::BOLD)
    };
    let send = Line::from(vec![Span::styled(" SEND ", send_style)]);

    let cols = Layout::horizontal([Constraint::Min(0), Constraint::Length(6)]).split(rows[0]);
    f.render_widget(Paragraph::new(input_line), cols[0]);
    f.render_widget(Paragraph::new(send), cols[1]);

    let status = Line::from(vec![
        Span::styled(format!(" {} ", state.agent_mode.label().to_uppercase()), mode_chip(state.agent_mode)),
        Span::styled(" · ", status_style()),
        Span::styled(state.model.clone(), theme::text()),
        Span::styled(" · ", status_style()),
        Span::styled(state.provider.clone(), theme::dim()),
        Span::styled(format!(" · session={}", state.session_id), theme::faint()),
    ]);
    f.render_widget(Paragraph::new(status), rows[1]);

    if focused {
        // Cursor sits just past the `› ` caret.
        let base = 2u16;
        let cursor_col = base + char_count(&state.input.chars().take(state.cursor).collect::<String>()) as u16;
        f.set_cursor_position((inner.x + cursor_col, inner.y));
    }
}

fn render_hints(f: &mut Frame, area: Rect, state: &AppState) {
    let accent = mode_accent(state.agent_mode);
    let left = Line::from(vec![Span::styled(
        "⏎ send · ⇧⏎ newline · / commands · esc close",
        theme::faint(),
    )]);
    let right_text = format!(
        "mode: {} · {} · {}",
        state.agent_mode.label().to_uppercase(),
        state.model,
        state.provider
    );
    let right = Line::from(vec![
        Span::styled("mode: ", theme::faint()),
        Span::styled(
            state.agent_mode.label().to_uppercase(),
            Style::default().fg(accent).add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" · {} · {}", state.model, state.provider), theme::dim()),
    ]);
    let parts = Layout::horizontal([
        Constraint::Min(0),
        Constraint::Length(right_text.chars().count() as u16 + 2),
    ])
    .split(area);
    f.render_widget(Paragraph::new(left), parts[0]);
    f.render_widget(Paragraph::new(right), parts[1]);
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

fn transcript_text(state: &AppState) -> Text<'static> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    for block in &state.transcript {
        lines.extend(block.to_lines());
        lines.push(Line::from(""));
    }
    if !state.current_reply.is_empty() {
        let streaming = Block_::new(Role::Assistant, state.current_reply.clone());
        lines.extend(streaming.to_lines());
    } else if state.busy {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{} ", spinner_glyph(state)),
                theme::violet().add_modifier(Modifier::BOLD),
            ),
            Span::styled("thinking…", theme::dim()),
        ]));
    } else if state.transcript.is_empty() {
        // Empty-session splash: the animated rainbow/pulse ZEUS wordmark.
        let t = state.started.elapsed().as_millis();
        lines.extend(
            super::decor::animated_banner(t)
                .into_iter()
                .map(Line::from),
        );
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            "· let me build with you ·",
            theme::faint(),
        )]));
    }
    Text::from(lines)
}

/// The chat column: scrolling transcript on top, the slash-command dropdown
/// and the pinned input bar + hint row at the bottom — mirroring the HTML's
/// `.chatcol` layout.
fn render_chat_column(f: &mut Frame, area: Rect, state: &AppState, matches: &[(&str, &str)]) {
    let menu_h = menu_height(matches);
    let rows = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(menu_h),
        Constraint::Length(2),
        Constraint::Length(1),
    ])
    .split(area);

    let text = transcript_text(state);
    let total_lines = text.lines.len() as u16;
    let visible = rows[0].height;
    let scroll = total_lines.saturating_sub(visible);
    let para = Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    f.render_widget(para, rows[0]);

    if menu_h > 0 {
        render_menu(f, rows[1], matches, state.command_selected);
    }
    render_input_box(f, rows[2], state);
    render_hints(f, rows[3], state);
}

/// Centers a `width`x`height` box within `area` (clamped to fit).
fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    }
}

/// A centered modal listing the current provider's available models —
/// arrow keys or a mouse click/scroll to navigate, Enter or a click to
/// select, Esc to close without changing anything. Modeled after opencode's
/// own "Select model" popup.
fn render_model_picker(
    f: &mut Frame,
    area: Rect,
    current_model: &str,
    entries: &[PickerEntry],
    selected: usize,
) -> Rect {
    let width = area.width.saturating_sub(6).clamp(30, 70);
    let height = (entries.len() as u16 + 4).clamp(6, area.height.saturating_sub(4));
    let popup = centered_rect(width, height, area);

    f.render_widget(Clear, popup);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style())
        .style(Style::default().bg(theme::PANEL))
        .title(Line::from(vec![
            Span::styled(" select model ", theme::green().add_modifier(Modifier::BOLD)),
        ]))
        .title_bottom(Line::from(
            Span::styled(" ↑/↓ navigate · enter select · esc dismiss ", theme::faint()),
        ).alignment(Alignment::Center));
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let items: Vec<ListItem> = entries
        .iter()
        .map(|entry| match entry {
            PickerEntry::Header(name) => ListItem::new(Line::from(Span::styled(
                name.to_uppercase(),
                theme::green().add_modifier(Modifier::BOLD),
            ))),
            PickerEntry::Model { model, .. } => {
                let marker = if model.id == current_model { "✓ " } else { "  " };
                ListItem::new(Line::from(vec![
                    Span::styled(marker, theme::green().add_modifier(Modifier::BOLD)),
                    Span::styled(model.name.clone(), theme::text()),
                    Span::raw("  "),
                    Span::styled(model.id.clone(), theme::dim()),
                ]))
            }
        })
        .collect();
    let list = List::new(items).highlight_style(
        Style::default().bg(theme::PANEL2).fg(theme::VIOLET).add_modifier(Modifier::BOLD),
    );
    let mut list_state = ListState::default();
    list_state.select(Some(selected.min(entries.len().saturating_sub(1))));
    f.render_stateful_widget(list, inner, &mut list_state);

    inner
}

/// The `/provider` popup: providers grouped into Local / Free / Paid headers
/// with a status dot (green = ready, amber = needs a key) and a hint that
/// selecting a key-less provider opens the paste prompt.
fn render_provider_picker(
    f: &mut Frame,
    area: Rect,
    current_provider: &str,
    current_model: &str,
    entries: &[ProviderEntry],
    selected: usize,
) -> Rect {
    let width = area.width.saturating_sub(6).clamp(36, 76);
    let height = (entries.len() as u16 + 4).clamp(8, area.height.saturating_sub(4));
    let popup = centered_rect(width, height, area);

    f.render_widget(Clear, popup);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style())
        .style(Style::default().bg(theme::PANEL))
        .title(Line::from(vec![
            Span::styled(" select provider ", theme::green().add_modifier(Modifier::BOLD)),
        ]))
        .title_bottom(Line::from(
            Span::styled(
                " ↑/↓ navigate · enter select (or paste key) · esc dismiss ",
                theme::faint(),
            ),
        )
        .alignment(Alignment::Center));
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let items: Vec<ListItem> = entries
        .iter()
        .map(|entry| match entry {
            ProviderEntry::Header(name) => ListItem::new(Line::from(Span::styled(
                format!(" {} ", name.to_uppercase()),
                theme::green().add_modifier(Modifier::BOLD),
            ))),
            ProviderEntry::Provider {
                name,
                kind,
                model,
                ready,
            } => {
                let is_current = name == current_provider;
                let dot = if *ready { theme::green() } else { theme::gold() };
                let dot_label = if *ready { "●" } else { "◌" };
                let status = if is_current { " (current)" } else { "" };
                let key_note = if *ready {
                    String::new()
                } else {
                    "  needs key".to_string()
                };
                ListItem::new(Line::from(vec![
                    Span::styled(dot_label, dot.add_modifier(Modifier::BOLD)),
                    Span::raw("  "),
                    Span::styled(name.clone(), theme::text().add_modifier(Modifier::BOLD)),
                    Span::styled(status, theme::green()),
                    Span::raw("  "),
                    Span::styled(kind.clone(), theme::dim()),
                    Span::styled(format!(" / {model}"), theme::faint()),
                    Span::styled(key_note, theme::gold()),
                ]))
            }
            ProviderEntry::Model { model, free, .. } => {
                let is_current = model.id == current_model;
                let tag = if is_current {
                    "✓ ".to_string()
                } else {
                    "   ".to_string()
                };
                let tier = if *free {
                    Span::styled("free", theme::green().add_modifier(Modifier::BOLD))
                } else {
                    Span::styled("paid", theme::gold().add_modifier(Modifier::BOLD))
                };
                ListItem::new(Line::from(vec![
                    Span::raw("   "),
                    Span::styled(tag, theme::green().add_modifier(Modifier::BOLD)),
                    Span::styled(model.name.clone(), theme::text()),
                    Span::raw("  "),
                    tier,
                    Span::styled(format!("  ·  {}", model.id), theme::faint()),
                ]))
            }
        })
        .collect();
    let list = List::new(items).highlight_style(
        Style::default().bg(theme::PANEL2).fg(theme::VIOLET).add_modifier(Modifier::BOLD),
    );
    let mut list_state = ListState::default();
    list_state.select(Some(selected.min(entries.len().saturating_sub(1))));
    f.render_stateful_widget(list, inner, &mut list_state);

    inner
}

/// The top bar is 3 rows and the bottom hint/session line is 1, per the HTML.
const TOPBAR_H: u16 = 3;
/// Right-hand sidebar width (the HTML's 300px TODO panel ≈ 44 columns).
const SIDE_W: u16 = 44;

/// Linearly interpolate between two RGB colors (used for the TODO progress
/// bar's violet → mode-accent gradient).
fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    fn rgb(c: Color) -> (u8, u8, u8) {
        match c {
            Color::Rgb(r, g, b) => (r, g, b),
            _ => (120, 120, 120),
        }
    }
    let (ar, ag, ab) = rgb(a);
    let (br, bg, bb) = rgb(b);
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color::Rgb(mix(ar, br), mix(ag, bg), mix(ab, bb))
}

/// Top bar — the HTML `.topbar`: logo + version on the left, the mode
/// segmented-pill control, and the provider button (with status dot) tucked
/// in the top-right corner. Returns the provider button's rect for mouse hit
/// testing.
fn render_topbar(f: &mut Frame, area: Rect, state: &AppState, config: &Config) -> Option<Rect> {
    let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1)])
        .split(area);

    // Row 0: logo (left) + provider button (right). The wordmark carries the
    // same rainbow/pulse sweep as the splash logo; the ⚡ bolt stays violet.
    let t = state.started.elapsed().as_millis();
    let wordmark = super::decor::animated_wordmark("ZEUS", t);
    let mut logo_spans = vec![
        Span::styled("⚡ ", theme::violet().add_modifier(Modifier::BOLD)),
    ];
    logo_spans.extend(wordmark);
    logo_spans.push(Span::styled("  v1.0", theme::faint()));
    let logo = Line::from(logo_spans);
    let provider_text = format!(
        "● {}  {}  ▾",
        state.provider, state.model
    );
    let provider_w = provider_text.chars().count() as u16;
    let top = Layout::horizontal([Constraint::Min(0), Constraint::Length(provider_w + 2)]).split(rows[0]);
    f.render_widget(Paragraph::new(logo), top[0]);

    let provider_line = Line::from(vec![
        Span::styled("● ", provider_status_style(config, &state.provider)),
        Span::styled(state.provider.clone(), theme::dim()),
        Span::styled("  ", theme::faint()),
        Span::styled(state.model.clone(), theme::text().add_modifier(Modifier::BOLD)),
        Span::styled("  ▾", theme::faint()),
    ]);
    f.render_widget(Paragraph::new(provider_line), top[1]);
    let provider_btn = top[1];

    // Row 1: mode segmented pills + a status readout on the right.
    render_mode_pills(f, rows[1], state.agent_mode);

    // Row 2: hairline separator (HTML `border-bottom` of the topbar).
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "─".repeat(area.width as usize),
            Style::default().fg(theme::BORDER),
        ))),
        rows[2],
    );

    Some(provider_btn)
}

/// The HTML `.modes` segmented control — PLAN/BUILD/AUTO with the active
/// segment filled in its mode accent and the others dimmed.
fn render_mode_pills(f: &mut Frame, area: Rect, mode: AgentMode) {
    let pills = [
        ("PLAN", AgentMode::Plan),
        ("BUILD", AgentMode::Build),
        ("AUTO", AgentMode::Auto),
    ];
    let mut spans = vec![Span::styled("  [ ", theme::faint())];
    for (i, (name, m)) in pills.iter().enumerate() {
        if *m == mode {
            spans.push(Span::styled(
                format!(" {} ", name),
                Style::default()
                    .fg(theme::VOID)
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
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// The HTML `.side` panel: TODOs header w/ live count, the progress bar
/// (fills as tasks are completed), the checklist, and the session footer.
/// Returns the todo-list rect for mouse-click row mapping.
fn render_side(f: &mut Frame, area: Rect, state: &AppState) -> Rect {
    // Fill the panel background.
    f.render_widget(
        Paragraph::new("").style(Style::default().bg(theme::PANEL)),
        area,
    );

    let done = state.todos.iter().filter(|t| t.done).count();
    let total = state.todos.len();
    let accent = mode_accent(state.agent_mode);

    let rows = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(4),
    ])
    .split(area);

    // Header.
    let head = Line::from(vec![
        Span::styled(
            "TODOs",
            Style::default().fg(theme::DIM).add_modifier(Modifier::BOLD),
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

    // Footer: Session / Tokens / Cost / Branch.
    render_side_foot(f, rows[3], state);

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
            spans.push(Span::styled("█", Style::default().fg(lerp_color(theme::VIOLET, accent, t))));
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
        lines.push(Line::from(vec![
            Span::styled("  no tasks yet — ask Zeus to fix something", theme::faint()),
        ]));
        f.render_widget(Paragraph::new(lines), area);
        return;
    }
    for item in &state.todos {
        let box_mark = if item.done { "✓" } else { " " };
        let box_style = if item.done {
            Style::default()
                .fg(theme::VOID)
                .bg(theme::GREEN)
                .add_modifier(Modifier::BOLD)
        } else if item.active {
            Style::default().fg(accent).add_modifier(Modifier::BOLD)
        } else {
            theme::faint()
        };
        let label_style = if item.done {
            Style::default().fg(theme::FAINT).add_modifier(Modifier::CROSSED_OUT)
        } else if item.active {
            Style::default().fg(accent)
        } else {
            theme::text()
        };
        lines.push(Line::from(vec![
            Span::styled("[", theme::faint()),
            Span::styled(box_mark, box_style),
            Span::styled("]", theme::faint()),
            Span::raw(" "),
            Span::styled(item.text.clone(), label_style),
        ]));
    }
    f.render_widget(Paragraph::new(lines), area);
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
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(area);
    let vals: [(String, String); 4] = [
        ("Session".into(), session),
        ("Tokens".into(), "0 / 200k".into()),
        ("Cost".into(), "$0.00".into()),
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

/// Filter trimmed drop_searched dropdown entries into (original_index,
/// row). Header/footer rows are excluded.
/// Filter trimmed `drop_search` entries into a flat row list (header rows
/// included so the group names render, but never selectable). Header rows
/// are dropped entirely while a search query is active, since a flat match
/// list needs no group labels.
fn drop_filtered(state: &AppState) -> Vec<ProviderEntry> {
    let Some(dd) = &state.dropdown else { return Vec::new() };
    let q = state.drop_search.to_lowercase();
    dd.entries
        .iter().filter(|&e| match e {
            ProviderEntry::Header(_) => q.is_empty(),
            ProviderEntry::Provider { name, model, .. } => {
                q.is_empty() || name.to_lowercase().contains(&q) || model.to_lowercase().contains(&q)
            }
            ProviderEntry::Model { provider, model, .. } => {
                q.is_empty()
                    || model.name.to_lowercase().contains(&q)
                    || model.id.to_lowercase().contains(&q)
                    || provider.to_lowercase().contains(&q)
            }
        }).cloned()
        .collect()
}

/// The top-right provider dropdown — the HTML `.provider-panel`. Shows a
/// search bar and the provider/model rows while browsing; swaps to an inline
/// API-key entry when a key-less provider/model is chosen.
fn render_dropdown(f: &mut Frame, area: Rect, state: &AppState) -> Option<Rect> {
    let dd = state.dropdown.as_ref()?;
    let keying = dd.keying.as_ref();
    let w = area.width.min(56);
    let list_h = if keying.is_some() {
        3
    } else {
        let n = drop_filtered(state).len().min(16) as u16;
        1 + n + 1
    };
    let h = (list_h + 2).min(area.height.saturating_sub(TOPBAR_H));
    let x = area.x + area.width.saturating_sub(w);
    let y = area.y + TOPBAR_H;
    let rect = Rect { x, y, width: w, height: h };

    f.render_widget(Clear, rect);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::VIOLET_DIM))
        .style(Style::default().bg(theme::ELEVATED));
    f.render_widget(block.clone(), rect);
    let inner = block.inner(rect);

    if let Some(k) = keying {
        let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1)])
            .split(inner);
        let prompt = Line::from(vec![
            Span::styled("🔑 API key for ", theme::gold().add_modifier(Modifier::BOLD)),
            Span::styled(k.provider.clone(), theme::text().add_modifier(Modifier::BOLD)),
        ]);
        f.render_widget(Paragraph::new(prompt), rows[0]);
        let key_line = if k.key.is_empty() {
            Line::from(vec![Span::styled(
                "  paste the key…",
                placeholder_style(),
            )])
        } else {
            let mut masked: String = String::new();
            let visible = k
                .key
                .chars()
                .last()
                .map(|c| c.to_string())
                .unwrap_or_else(|| " ".into());
            masked.push_str(&"•".repeat(k.key.chars().count().saturating_sub(1)));
            masked.push_str(&visible);
            Line::from(vec![
                Span::styled("  ", theme::faint()),
                Span::styled(masked, theme::green()),
            ])
        };
        f.render_widget(Paragraph::new(key_line), rows[1]);
        f.render_widget(
            Paragraph::new(Line::from(vec![Span::styled(
                " enter save · esc back · backspace edit ",
                theme::faint(),
            )])),
            rows[2],
        );
    } else {
        let rows = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(inner);
        let search_line = if state.drop_search.is_empty() {
            Line::from(vec![
                Span::styled("▸", theme::violet()),
                Span::raw("  "),
                Span::styled("Search providers or models…", placeholder_style()),
            ])
        } else {
            Line::from(vec![
                Span::styled("▸ ", theme::violet()),
                Span::styled(state.drop_search.clone(), theme::text()),
            ])
        };
        f.render_widget(Paragraph::new(search_line), rows[0]);

        let filtered = drop_filtered(state);
        if !state.drop_search.is_empty() && filtered.is_empty() {
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("  no matches", theme::faint()),
                ])),
                rows[1],
            );
            return Some(rect);
        }
        let items: Vec<ListItem> = filtered
            .iter()
            .map(|e| match e {
                ProviderEntry::Header(name) => ListItem::new(Line::from(vec![
                    Span::styled(
                        format!(" {} ", name.to_uppercase()),
                        theme::green().add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  "),
                ])),
                ProviderEntry::Provider { name, kind: _, model, ready } => {
                    let dot = if *ready { "●" } else { "◌" };
                    let dot_style = if *ready { theme::green() } else { theme::gold() };
                    let need = if *ready { "" } else { " key" };
                    ListItem::new(Line::from(vec![
                        Span::styled(dot, dot_style.add_modifier(Modifier::BOLD)),
                        Span::raw(" "),
                        Span::styled(name.clone(), theme::text().add_modifier(Modifier::BOLD)),
                        Span::raw(" "),
                        Span::styled(model.clone(), theme::faint()),
                        Span::styled(need, theme::gold()),
                    ]))
                }
                ProviderEntry::Model { provider, model, free, .. } => {
                    let tier = if *free { "free" } else { "paid" };
                    let tier_style = if *free { theme::green() } else { theme::gold() };
                    ListItem::new(Line::from(vec![
                        Span::raw("  · "),
                        Span::styled(model.name.clone(), theme::text()),
                        Span::raw(" "),
                        Span::styled(tier, tier_style.add_modifier(Modifier::BOLD)),
                        Span::raw(" "),
                        Span::styled(provider.clone(), theme::faint()),
                    ]))
                }
            })
            .collect();
        let list = List::new(items).highlight_style(
            Style::default()
                .bg(theme::PANEL2)
                .fg(theme::VIOLET)
                .add_modifier(Modifier::BOLD),
        );
        let mut list_state = ListState::default();
        let selectable = filtered.len().saturating_sub(1);
        let sel = (0..=selectable)
            .rev()
            .find(|&i| {
                !matches!(filtered.get(i), Some(ProviderEntry::Header(_)))
            })
            .unwrap_or(0);
        list_state.select(Some(sel.min(selectable)));
        f.render_stateful_widget(list, rows[1], &mut list_state);
    }

    Some(rect)
}

/// The provider dot in the top-right button and pickers: green when the
fn provider_status_ok(config: &Config, provider: &str) -> bool {
    let Some(cfg) = config.providers.get(provider) else {
        return false;
    };
    if matches!(cfg.kind.as_str(), "ollama" | "lmstudio" | "llamacpp")
        || cfg.headers.contains_key("Authorization")
    {
        return true;
    }
    match &cfg.api_key_env {
        Some(var) => std::env::var(var).map(|k| !k.is_empty()).unwrap_or(false),
        None => true,
    }
}

fn provider_status_style(config: &Config, provider: &str) -> Style {
    if provider_status_ok(config, provider) {
        theme::green()
    } else {
        theme::gold()
    }
}

fn render(f: &mut Frame, state: &mut AppState, config: &Config) {
    let area = f.area();

    // Fill the whole frame with the void background first.
    f.render_widget(Block::default().style(Style::default().bg(theme::VOID)), area);

    let rows = Layout::vertical([
        Constraint::Length(TOPBAR_H),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .split(area);

    if let Some(btn) = render_topbar(f, rows[0], state, config) {
        state.provider_btn_area = Some(btn);
    }

    let has_side = area.width >= 100;
    let main = Layout::horizontal([
        Constraint::Min(0),
        Constraint::Length(if has_side { SIDE_W } else { 0 }),
    ])
    .split(rows[1]);
    state.todo_area = if has_side { Some(main[1]) } else { None };

    let matches = state.command_matches();
    render_chat_column(f, main[0], state, &matches);

    if has_side {
        let list_area = render_side(f, main[1], state);
        state.todo_area = Some(list_area);
    }

    // Top-right provider dropdown popover (drawn above the main content).
    if let Some(rect) = render_dropdown(f, rows[1], state) {
        if let Some(dd) = state.dropdown.as_mut() {
            dd.area = Some(rect);
        }
    } else if let Some(dd) = state.dropdown.as_mut() {
        dd.area = None;
    }

    let picker_area = if let Mode::ModelPicker { entries, selected } = &state.mode {
        Some(render_model_picker(f, area, &state.model, entries, *selected))
    } else {
        None
    };
    state.model_picker_area = picker_area;

    let provider_picker_area =
        if let Mode::ProviderPicker { entries, selected } = &state.mode {
            Some(render_provider_picker(
                f,
                area,
                &state.provider,
                &state.model,
                entries,
                *selected,
            ))
        } else {
            None
        };
    state.provider_picker_area = provider_picker_area;
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
        let approver = move |req: &PermissionRequest| -> ApprovalDecision {
            if yes {
                return ApprovalDecision::Approved;
            }
            let (reply_tx, reply_rx) = std::sync::mpsc::channel();
            let _ = tx_approval.send(UiEvent::Approval(ApprovalRequestMsg {
                request: req.clone(),
                reply: reply_tx,
            }));
            reply_rx.recv().unwrap_or(ApprovalDecision::Denied)
        };
        let result = agent.run_turn(&message, on_event, approver).await;
        (agent, result)
    });
    *turn_handle = Some(handle);
}

/// `/plan <goal>`: same plumbing as `start_turn`, but drives
/// `Agent::plan_turn` — research read-only, persist `.agent/tasks.json`,
/// and return without executing anything. Plan mode stays on afterwards so
/// the next turns remain observational until the user switches to Auto.
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
        let approver = move |req: &PermissionRequest| -> ApprovalDecision {
            if yes {
                return ApprovalDecision::Approved;
            }
            let (reply_tx, reply_rx) = std::sync::mpsc::channel();
            let _ = tx_approval.send(UiEvent::Approval(ApprovalRequestMsg {
                request: req.clone(),
                reply: reply_tx,
            }));
            reply_rx.recv().unwrap_or(ApprovalDecision::Denied)
        };
        let result = agent.plan_turn(&message, on_event, approver).await;
        (agent, result)
    });
    *turn_handle = Some(handle);
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
    if let Mode::Approval(_) = &state.mode {
        let decision = match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => Some(ApprovalDecision::Approved),
            KeyCode::Char('s') | KeyCode::Char('S') => Some(ApprovalDecision::ApprovedForSession),
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => Some(ApprovalDecision::Denied),
            _ => None,
        };
        if let Some(decision) = decision {
            if let Mode::Approval(pending) = std::mem::replace(&mut state.mode, Mode::Chat) {
                let _ = pending.reply.send(decision);
            }
        }
        return Ok(());
    }

    // ---- Top-right provider dropdown (search / key entry) ----
    if state.dropdown.is_some() {
        // Inline API-key entry mode.
        if let Some(k) = state
            .dropdown
            .as_ref()
            .and_then(|d| d.keying.as_ref())
            .map(|k| (k.provider.clone(), k.key.clone()))
        {
            let (provider, keybuf) = k;
            let mut keyval = keybuf;
            match key.code {
                KeyCode::Enter => {
                    state.dropdown = None;
                    persist_key_and_switch(&provider, &keyval, config, agent_slot, state);
                }
                KeyCode::Esc => {
                    if let Some(d) = state.dropdown.as_mut() {
                        d.keying = None;
                    }
                }
                KeyCode::Backspace => {
                    keyval.pop();
                    if let Some(d) = state.dropdown.as_mut() {
                        d.keying.as_mut().unwrap().key = keyval;
                    }
                }
                KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    keyval.push(c);
                    if let Some(d) = state.dropdown.as_mut() {
                        d.keying.as_mut().unwrap().key = keyval;
                    }
                }
                _ => {}
            }
            return Ok(());
        }

        // List mode — search + navigate. Group headers are part of the
        // filtered list for rendering, but navigation and selection skip them.
        let filtered = drop_filtered(state);
        match key.code {
            KeyCode::Esc => state.dropdown = None,
            KeyCode::Up => {
                if let Some(d) = state.dropdown.as_mut() {
                    d.selected = dropdown_prev_selectable(&filtered, d.selected);
                }
            }
            KeyCode::Down => {
                if let Some(d) = state.dropdown.as_mut() {
                    d.selected = dropdown_next_selectable(&filtered, d.selected);
                }
            }
            KeyCode::Tab | KeyCode::Enter => {
                let sel = state
                    .dropdown
                    .as_ref()
                    .map(|d| d.selected)
                    .unwrap_or(0);
                if let Some(entry) = filtered.get(sel) {
                    let picked = entry.clone();
                    dropdown_apply(state, agent_slot, config, &picked);
                }
            }
            KeyCode::Backspace => {
                state.drop_search.pop();
                if let Some(d) = state.dropdown.as_mut() {
                    d.selected = 0;
                }
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                state.drop_search.push(c);
                if let Some(d) = state.dropdown.as_mut() {
                    d.selected = 0;
                }
            }
            _ => {}
        }
        return Ok(());
    }

    if let Mode::ModelPicker { .. } = &state.mode {
        match key.code {
            KeyCode::Up => {
                let Mode::ModelPicker { entries, selected } = &mut state.mode else { unreachable!() };
                *selected = picker_move(entries, *selected, -1);
            }
            KeyCode::Down => {
                let Mode::ModelPicker { entries, selected } = &mut state.mode else { unreachable!() };
                *selected = picker_move(entries, *selected, 1);
            }
            KeyCode::Enter => {
                let Mode::ModelPicker { entries, selected } = &state.mode else { unreachable!() };
                let chosen = match entries.get(*selected) {
                    Some(PickerEntry::Model { provider, model }) => {
                        Some((provider.clone(), model.id.clone()))
                    }
                    _ => None,
                };
                match chosen {
                    Some((provider, model_id)) => {
                        apply_picker_choice(provider, model_id, state, agent_slot, config)
                    }
                    None => state.mode = Mode::Chat,
                }
            }
            KeyCode::Esc => {
                state.mode = Mode::Chat;
            }
            _ => {}
        }
        return Ok(());
    }

    if let Mode::ProviderPicker { .. } = &state.mode {
        match key.code {
            KeyCode::Up => {
                let Mode::ProviderPicker { entries, selected } = &mut state.mode else { unreachable!() };
                *selected = provider_picker_move(entries, *selected, -1);
            }
            KeyCode::Down => {
                let Mode::ProviderPicker { entries, selected } = &mut state.mode else { unreachable!() };
                *selected = provider_picker_move(entries, *selected, 1);
            }
            KeyCode::Enter => {
                let Mode::ProviderPicker { entries, selected } = &state.mode else { unreachable!() };
                match entries.get(*selected) {
                    Some(ProviderEntry::Provider { name, ready, .. }) => {
                        let (name, ready) = (name.clone(), *ready);
                        apply_provider_picker_choice(name, ready, config, agent_slot, state);
                    }
                    Some(ProviderEntry::Model { provider, model, .. }) => {
                        let (provider, model_id) = (provider.clone(), model.id.clone());
                        apply_picker_choice(provider, model_id, state, agent_slot, config);
                    }
                    _ => state.mode = Mode::Chat,
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
                let Mode::KeyEntry { provider } = std::mem::replace(&mut state.mode, Mode::Chat)
                    else { unreachable!() };
                let key = std::mem::take(&mut state.input);
                state.cursor = 0;
                let key = key.trim().to_string();
                if key.is_empty() {
                    state.push_error(format!("no key entered for '{provider}' — key not saved"));
                    return Ok(());
                }
                persist_key_and_switch(&provider, &key, config, agent_slot, state);
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

    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        if state.busy {
            if let Some(tx) = cancel_tx.as_ref() {
                let _ = tx.send(true);
            }
        } else {
            state.quit = true;
        }
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
        }
        return Ok(());
    }

    if state.busy {
        return Ok(());
    }

    // While the slash-command dropdown is open, arrow keys move the
    // highlight and Enter/Tab accept the highlighted entry (filling the
    // input, ready for arguments) instead of their normal effect.
    let menu_len = state.command_matches().len();
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
                let selected = state.command_matches()[idx].0.to_string();
                state.input = format!("/{selected} ");
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
        KeyCode::Enter => {
            if state.input.trim().is_empty() {
                return Ok(());
            }
            let raw = std::mem::take(&mut state.input);
            state.cursor = 0;
            let trimmed = raw.trim().to_string();
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
                        let agent = build_agent_repl(config).await?;
                        apply_agent_mode(&agent, state.agent_mode);
                        state.session_id = agent.session_id().to_string();
                        state.model = agent.model().to_string();
                        state.provider = agent.provider_id().to_string();
                        *agent_slot = Some(agent);
                        state.transcript.clear();
                        state.current_reply.clear();
                        state.push_info(format!("cleared — new session={}", state.session_id));
                    }
                    "new" => {
                        let agent = build_agent_repl(config).await?;
                        apply_agent_mode(&agent, state.agent_mode);
                        state.session_id = agent.session_id().to_string();
                        state.model = agent.model().to_string();
                        state.provider = agent.provider_id().to_string();
                        *agent_slot = Some(agent);
                        state.transcript.clear();
                        state.current_reply.clear();
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
                    "model" => {
                        if arg.is_empty() {
                            // No name given — open a picker grouped by
                            // provider (arrow keys/scroll to choose,
                            // Enter/click to apply, Esc to back out) instead
                            // of just printing the current model, so the
                            // user doesn't need to already know an exact
                            // model name to type.
                            let mut groups = list_models_by_provider(config).await;
                            // Current provider's group first (like opencode's
                            // own "Recent" section leading the list) — the
                            // rest stay in the alphabetical order
                            // `list_models_by_provider` already produced.
                            if let Some(pos) = groups.iter().position(|(name, _)| *name == state.provider) {
                                let current = groups.remove(pos);
                                groups.insert(0, current);
                            }
                            if groups.is_empty() {
                                state.push_error(
                                    "no models found on any configured provider (check they're running)",
                                );
                            } else {
                                let mut entries = Vec::new();
                                let mut selected = 0;
                                for (provider_name, models) in groups {
                                    entries.push(PickerEntry::Header(provider_name.clone()));
                                    for model in models {
                                        if model.id == state.model && provider_name == state.provider {
                                            selected = entries.len();
                                        }
                                        entries.push(PickerEntry::Model { provider: provider_name.clone(), model });
                                    }
                                }
                                state.mode = Mode::ModelPicker { entries, selected };
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
                    "agents" => {
                        if arg.eq_ignore_ascii_case("count") {
                            let pools = personas_by_department();
                            let total: usize = pools.iter().map(|(_, list)| list.len()).sum();
                            state.push_info(format!("{total} specialist agents"));
                        } else {
                            let mut text = String::from("Specialist agent pool (grouped by department):");
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
                    "copy" => {
                        let block = state
                            .transcript
                            .iter()
                            .rev()
                            .find(|b| matches!(b.role, Role::Assistant | Role::Tool));
                        match block {
                            Some(b) => match super::clipboard::copy(&b.plain_text()) {
                                Ok(()) => {
                                    state.push_info("copied last block to clipboard");
                                }
                                Err(e) => state.push_error(format!("copy failed: {e}")),
                            },
                            None => state.push_error("nothing to copy yet"),
                        }
                    }
                    _ => {
                        let expanded = expand_slash_command(config, trimmed.clone());
                        if expanded != trimmed {
                            state.push_user(trimmed.clone());
                            start_turn(state, agent_slot, turn_handle, cancel_tx, ui_tx, expanded, yes);
                        } else {
                            state.push_error(format!("unknown command: /{cmd}"));
                        }
                    }
                }
                return Ok(());
            }

            state.push_user(trimmed.clone());
            start_turn(state, agent_slot, turn_handle, cancel_tx, ui_tx, trimmed, yes);
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
            state.command_selected = 0;
        }
        _ => {}
    }
    Ok(())
}

/// Mouse support for the model picker: click a row to select and apply it
/// immediately (matching opencode's own click-to-choose), scroll to move
/// the highlight without applying. Only meaningful while the picker is
/// open — mouse events are otherwise ignored.
async fn handle_mouse(ev: MouseEvent, state: &mut AppState, agent_slot: &mut Option<Agent>, config: &Config) {
    if let MouseEventKind::Down(MouseButton::Left) = ev.kind {
        let col = ev.column;
        let row = ev.row;
        let in_provider = state
            .provider_btn_area
            .map(|a| col >= a.x && col < a.x + a.width && row >= a.y && row < a.y + a.height)
            .unwrap_or(false);
        let in_drop = state
            .dropdown
            .as_ref()
            .and_then(|d| d.area)
            .map(|a| col >= a.x && col < a.x + a.width && row >= a.y && row < a.y + a.height)
            .unwrap_or(false);

        // Dropdown is open: pick a row inside it, close when clicking outside.
        if state.dropdown.is_some() {
            if in_provider {
                state.dropdown = None;
                state.drop_search.clear();
                return;
            }
            if in_drop {
                let area = state.dropdown.as_ref().unwrap().area.unwrap();
                // List rows start just past the border + search row.
                let list_top = area.y + 2;
                let idx = row as isize - list_top as isize;
                let filtered = drop_filtered(state);
                if idx >= 0 && (idx as usize) < filtered.len() {
                    let entry = filtered[idx as usize].clone();
                    dropdown_apply(state, agent_slot, config, &entry);
                }
                return;
            }
            state.dropdown = None;
            state.drop_search.clear();
            return;
        }

        // Provider button: build (from the real config — dynamic, not the
        // hardcoded HTML list) and open the dropdown.
        if in_provider {
            let entries =
                build_dropdown_entries(config, &state.provider, &state.model).await;
            state.dropdown = Some(DropdownState {
                entries,
                selected: 0,
                area: None,
                keying: None,
            });
            state.drop_search.clear();
            return;
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
                        let Mode::ModelPicker { entries, .. } = &state.mode else { unreachable!() };
                        if let Some(PickerEntry::Model { provider, model }) = entries.get(row) {
                            let (provider, model_id) = (provider.clone(), model.id.clone());
                            apply_picker_choice(provider, model_id, state, agent_slot, config);
                        }
                    }
                }
                MouseEventKind::ScrollUp => {
                    let Mode::ModelPicker { entries, selected } = &mut state.mode else { unreachable!() };
                    *selected = picker_move(entries, *selected, -1);
                }
                MouseEventKind::ScrollDown => {
                    let Mode::ModelPicker { entries, selected } = &mut state.mode else { unreachable!() };
                    *selected = picker_move(entries, *selected, 1);
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
                        let Mode::ProviderPicker { entries, .. } = &state.mode else { unreachable!() };
                        match entries.get(row) {
                            Some(ProviderEntry::Provider { name, ready, .. }) => {
                                apply_provider_picker_choice(
                                    name.clone(),
                                    *ready,
                                    config,
                                    agent_slot,
                                    state,
                                );
                            }
                            Some(ProviderEntry::Model { provider, model, .. }) => {
                                apply_picker_choice(
                                    provider.clone(),
                                    model.id.clone(),
                                    state,
                                    agent_slot,
                                    config,
                                );
                            }
                            _ => {}
                        }
                    }
                }
                MouseEventKind::ScrollUp => {
                    let Mode::ProviderPicker { entries, selected } = &mut state.mode else { unreachable!() };
                    *selected = provider_picker_move(entries, *selected, -1);
                }
                MouseEventKind::ScrollDown => {
                    let Mode::ProviderPicker { entries, selected } = &mut state.mode else { unreachable!() };
                    *selected = provider_picker_move(entries, *selected, 1);
                }
                _ => {}
            }
        }
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
    let mut state = AppState::new(&agent, known_commands, dir, config.project_root.is_some());
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

    loop {
        tokio::select! {
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
                        }
                    }
                    Event::Mouse(mouse) => {
                        handle_mouse(mouse, &mut state, &mut agent_slot, config).await;
                    }
                    _ => {}
                }
            }
            Some(ui_event) = ui_rx.recv() => {
                match ui_event {
                    UiEvent::Agent(ev) => state.apply_agent_event(ev),
                    UiEvent::Approval(req) => state.mode = Mode::Approval(req),
                }
            }
            res = async { turn_handle.as_mut().unwrap().await }, if turn_handle.is_some() => {
                turn_handle = None;
                state.busy = false;
                cancel_tx = None;
                match res {
                    Ok((agent, turn_result)) => {
                        agent_slot = Some(agent);
                        state.flush_current_reply();
                        if let Err(e) = turn_result {
                            state.push_error(format!("turn failed: {e:#}"));
                        }
                    }
                    Err(join_err) => {
                        state.push_error(format!("internal error: {join_err}"));
                        let agent = build_agent_repl(config).await?;
                        apply_agent_mode(&agent, state.agent_mode);
                        agent_slot = Some(agent);
                    }
                }
            }
        }

        if state.quit {
            break;
        }
terminal.draw(|f| render(f, &mut state, config))?;
        sync_cursor_visibility(terminal, &state);
    }
    Ok(())
}

/// Current git branch for the side-panel footer. Best-effort — any failure
/// degrades silently to "(no git repo)".
fn build_dir_info(config: &Config) -> DirInfo {
    let path = config
        .project_root
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")));

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
    if state.busy || !matches!(state.mode, Mode::Chat) {
        terminal.hide_cursor().ok();
    }
}

pub async fn run(config: &Config, agent: Agent, yes: bool) -> Result<()> {
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
    let mut terminal = Terminal::new(backend).context("init terminal")?;

    let result = run_app(&mut terminal, config, agent, yes).await;

    disable_raw_mode().ok();
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture,
        DisableBracketedPaste
    )
    .ok();
    terminal.show_cursor().ok();

    result
}
