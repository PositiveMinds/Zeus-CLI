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
    build_agent_repl_with, build_agent_repl_with_session, expand_slash_command,
    git_engine_for_agent, known_slash_commands, list_models_by_provider,
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
use zeus_agent::{personas_by_department, Agent, AgentEvent, SessionStore, SessionSummary, TurnResult};
use zeus_config::{Config, KeysFile};
use zeus_fs::{ApprovalDecision, PermissionRequest};
use zeus_provider::{create_provider, ModelInfo, TokenUsage};

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
    /// from this (splitting on newlines); `lines` exists for future blocks
    /// that need pre-built styled rows (tool-call chips, diff blocks) rather
    /// than plain text.
    text: String,
    /// Pre-built styled rows (empty for plain text blocks).
    lines: Vec<Vec<Span<'static>>>,
    /// Memoized `to_lines()` output. A block's fields never change once
    /// pushed to the transcript, but every keystroke redraws the whole
    /// chat column — without this, a long conversation would re-run
    /// markdown/syntax highlighting on every past message on every single
    /// keypress, and typing gets visibly laggier as history grows. A
    /// `RefCell` rather than a `OnceCell` since `toggle_expanded` needs to
    /// invalidate it — the one field that *can* change post-push.
    rendered: std::cell::RefCell<Option<Vec<Line<'static>>>>,
    /// User-expanded past the `MAX_TOOL_LINES` fold (see `is_foldable`,
    /// `toggle_expanded`). Only meaningful for `Tool`/`ToolError` blocks
    /// whose body is long enough to fold in the first place.
    expanded: std::cell::Cell<bool>,
}

impl Block_ {
    /// Body-line cap before a `Tool`/`ToolError` result folds behind a
    /// "click to expand" note instead of flooding the transcript — a real
    /// changeset diff or a big grep/read result could otherwise push
    /// everything else off screen.
    const MAX_TOOL_LINES: usize = 40;

    fn new(role: Role, text: String) -> Self {
        Self {
            role,
            text,
            lines: Vec::new(),
            rendered: std::cell::RefCell::new(None),
            expanded: std::cell::Cell::new(false),
        }
    }

    /// Whether this block is currently folded and has more to show — the
    /// click-to-expand path in `handle_mouse` checks this before deciding
    /// whether a click expands the message or copies it.
    fn is_foldable(&self) -> bool {
        if self.expanded.get() || !matches!(self.role, Role::Tool | Role::ToolError) {
            return false;
        }
        let body_line_count = self.text.splitn(2, '\n').nth(1).map(|b| b.lines().count()).unwrap_or(0);
        body_line_count > Self::MAX_TOOL_LINES
    }

    /// Reveals the rest of a folded tool result and invalidates the
    /// rendered-lines cache so the next `content_lines()` call re-renders
    /// in full instead of returning the still-truncated cached version.
    fn toggle_expanded(&self) {
        self.expanded.set(!self.expanded.get());
        *self.rendered.borrow_mut() = None;
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

    /// Rendered lines for the transcript, `width` columns wide. User and
    /// Assistant turns render as bordered chat bubbles (`bubble_lines`,
    /// matching `zeus-cli.html`'s `.bubble` rows); everything else (tool
    /// status lines, errors, info) stays plain marker-prefixed text like a
    /// log line, matching the HTML's unbubbled `·`/`✗` status rows.
    fn to_lines(&self, width: u16) -> Vec<Line<'static>> {
        match self.role {
            Role::User | Role::Assistant => self.bubble_lines(width),
            _ => self.marked_lines(),
        }
    }

    fn marked_lines(&self) -> Vec<Line<'static>> {
        let marker_style = self.role.marker_style();
        let marker = self.role.marker();
        let content = self.content_lines();
        content
            .into_iter()
            .enumerate()
            .map(|(i, line)| {
                let mut spans = vec![Span::styled(
                    if i == 0 { marker } else { Self::pad() },
                    if i == 0 { marker_style } else { theme::faint() },
                )];
                spans.extend(line.spans);
                Line::from(spans)
            })
            .collect()
    }

    /// A bordered chat bubble: right-aligned with a trailing "YOU" tag for
    /// the user, left-aligned with a leading "⚡" avatar and a violet
    /// left-accent border for the assistant — the terminal equivalent of
    /// the HTML's `.msg.user`/`.msg.zeus` bubble rows.
    /// A single vertical accent line per side marks whose turn it is — no
    /// full box. A `╭─╮…╰─╯` border on every single message (the original
    /// design, modeled directly off `zeus-cli.html`) added three rows of
    /// pure chrome per exchange; over a long, already code-block-heavy
    /// session that reads as noisy compared to how most terminal coding
    /// agents render turns (a colored marker/indent, not a bordered card).
    /// Alignment + the accent line + (for the user) a trailing "YOU" tag
    /// are enough to mark a turn without the extra weight.
    fn bubble_lines(&self, width: u16) -> Vec<Line<'static>> {
        let avail = (width as usize).max(20);
        let border_style = Style::default().fg(theme::BORDER_SOFT);
        if self.role == Role::User {
            let content_w = ((avail * 7) / 10).clamp(20, 84);
            let wrapped = wrap_preserving_newlines(&self.text, content_w);
            let mut out: Vec<Line<'static>> = wrapped
                .into_iter()
                .map(|line| {
                    Line::from(vec![
                        Span::styled(line, theme::user_text()),
                        Span::styled(" │", border_style),
                    ])
                    .alignment(Alignment::Right)
                })
                .collect();
            out.push(
                Line::from(Span::styled("YOU", theme::dim().add_modifier(Modifier::BOLD)))
                    .alignment(Alignment::Right),
            );
            out
        } else {
            // Every line is wrapped to the available width first —
            // `content_lines()` (markdown/syntax-highlighted spans) is
            // never pre-wrapped to any width on its own, so a long
            // paragraph or a wide highlighted line needs its own pass here
            // rather than sailing straight past the terminal edge.
            let max_inner_w = avail.saturating_sub(2).max(10);
            let content: Vec<Line<'static>> = self
                .content_lines()
                .into_iter()
                .flat_map(|line| {
                    if line_char_width(&line) > max_inner_w {
                        wrap_spans(&line.spans, max_inner_w)
                            .into_iter()
                            .map(Line::from)
                            .collect::<Vec<_>>()
                    } else {
                        vec![line]
                    }
                })
                .collect();
            content
                .into_iter()
                .enumerate()
                .map(|(i, line)| {
                    // The ⚡ avatar stands in for the accent line on row
                    // one; a plain violet bar carries it down the rest of
                    // a multi-line reply so the turn still reads as one
                    // continuous block while scrolling past it.
                    let mut spans = if i == 0 {
                        vec![Span::styled("⚡ ", theme::violet().add_modifier(Modifier::BOLD))]
                    } else {
                        vec![Span::styled("│ ", theme::violet())]
                    };
                    spans.extend(line.spans);
                    Line::from(spans)
                })
                .collect()
        }
    }

    /// Marker-free content lines — memoized, since a long conversation
    /// redraws the whole chat column on every keypress and re-running
    /// markdown/syntax highlighting on every past message each time would
    /// make typing visibly laggier as history grows.
    fn content_lines(&self) -> Vec<Line<'static>> {
        if let Some(cached) = self.rendered.borrow().as_ref() {
            return cached.clone();
        }
        let computed = self.compute_lines();
        *self.rendered.borrow_mut() = Some(computed.clone());
        computed
    }

    fn compute_lines(&self) -> Vec<Line<'static>> {
        let text_style = self.role.text_style();
        if !self.lines.is_empty() {
            return self.lines.iter().cloned().map(Line::from).collect();
        }

        // Assistant replies may embed fenced code blocks — tokenize and color
        // them instead of emitting everything in a single flat span.
        if self.role == Role::Assistant {
            let highlighted = super::highlight::markdown_lines(&self.text, text_style);
            if !highlighted.is_empty() {
                return highlighted.into_iter().map(Line::from).collect();
            }
        }

        // Tool calls were dumped as flat, uncolored, unbounded text — the
        // call header ("toolname {raw json args}" / "toolname (done)")
        // read as an undifferentiated wall of one color, and a real
        // changeset diff in the result could flood the whole transcript
        // with hundreds of lines the same way. The header now gets the
        // tool name bolded and its status/args styled distinctly; the body
        // gets the same +/- diff coloring the approval preview already
        // uses when it looks like one, and any result gets capped so one
        // huge dump can't push everything else off screen (the full text
        // is still there via `/copy` or scrolling back).
        if matches!(self.role, Role::Tool | Role::ToolError) {
            let mut header_and_body = self.text.splitn(2, '\n');
            let header = header_and_body.next().unwrap_or("");
            let body = header_and_body.next().unwrap_or("");

            let mut lines: Vec<Line<'static>> = vec![style_tool_header(header, self.role == Role::ToolError)];
            let mut body_lines: Vec<Line<'static>> = if super::highlight::looks_like_diff(body) {
                super::highlight::diff_lines(body, text_style)
                    .into_iter()
                    .map(Line::from)
                    .collect()
            } else {
                body.lines().map(|l| style_tool_body_line(l, text_style)).collect()
            };
            if body_lines.len() > Self::MAX_TOOL_LINES && !self.expanded.get() {
                let omitted = body_lines.len() - Self::MAX_TOOL_LINES;
                body_lines.truncate(Self::MAX_TOOL_LINES);
                body_lines.push(Line::from(Span::styled(
                    format!("… {omitted} more line(s) — click this message to expand"),
                    theme::faint().add_modifier(Modifier::ITALIC),
                )));
            }
            lines.append(&mut body_lines);
            return lines;
        }

        let mut raw_lines: Vec<&str> = self.text.lines().collect();
        if raw_lines.is_empty() {
            raw_lines.push("");
        }
        raw_lines
            .iter()
            .map(|l| Line::from(Span::styled(l.to_string(), text_style)))
            .collect()
    }
}

/// A tool call's header line: `"toolname {raw args}"` (just started) or
/// `"toolname (done)"`/`"toolname (failed)"` (finished) — a fixed format
/// this module itself writes in `apply_agent_event`, not external input, so
/// splitting on the first space/parenthesized status is safe. Bolds the
/// tool name, colors the status word, and dims+caps raw call arguments so a
/// long inline JSON blob (a big grep pattern, a large file write) doesn't
/// dominate the line the way the unstyled flat dump used to.
fn style_tool_header(header: &str, is_error: bool) -> Line<'static> {
    let (name, rest) = header.split_once(' ').unwrap_or((header, ""));
    let name_style = if is_error {
        theme::red().add_modifier(Modifier::BOLD)
    } else {
        theme::cyan().add_modifier(Modifier::BOLD)
    };
    let mut spans = vec![Span::styled(name.to_string(), name_style)];
    if !rest.is_empty() {
        spans.push(Span::raw(" "));
        match rest.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
            Some(status) => {
                let status_style = if status == "done" {
                    theme::green().add_modifier(Modifier::BOLD)
                } else {
                    theme::red().add_modifier(Modifier::BOLD)
                };
                spans.push(Span::styled(format!("· {status}"), status_style));
            }
            None => {
                const MAX_ARGS_CHARS: usize = 140;
                let count = rest.chars().count();
                let args = if count > MAX_ARGS_CHARS {
                    let head: String = rest.chars().take(MAX_ARGS_CHARS).collect();
                    format!("{head}…")
                } else {
                    rest.to_string()
                };
                spans.push(Span::styled(args, theme::faint()));
            }
        }
    }
    Line::from(spans)
}

/// A tool result body line — plain content keeps the role's normal text
/// style, but the `--- stdout ---`/`--- stderr ---` section dividers
/// (written by the `run` tool) get set apart instead of blending into the
/// output they're labeling.
fn style_tool_body_line(line: &str, text_style: Style) -> Line<'static> {
    let trimmed = line.trim();
    if trimmed.starts_with("--- ") && trimmed.ends_with(" ---") {
        return Line::from(Span::styled(
            line.to_string(),
            theme::faint().add_modifier(Modifier::ITALIC),
        ));
    }
    Line::from(Span::styled(line.to_string(), text_style))
}

/// Total character width of a line's spans (assumes narrow/ambiguous-width
/// glyphs only — the same assumption ratatui's own layout math makes).
fn line_char_width(line: &Line) -> usize {
    line.spans.iter().map(|s| s.content.chars().count()).sum()
}

/// Word-wraps each `\n`-separated paragraph independently to `width`
/// columns, so a deliberate blank line in a pasted message survives.
fn wrap_preserving_newlines(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.split('\n') {
        if para.is_empty() {
            out.push(String::new());
        } else {
            out.extend(wrap_text(para, width));
        }
    }
    out
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
    /// Top-right provider dropdown finished probing configured providers —
    /// see the doc comment on `fetching_providers` for why this is spawned
    /// rather than awaited inline. Carries the raw per-provider model scan
    /// too, so the main loop can cache it (`AppState::model_cache`) for the
    /// next dropdown/picker open.
    DropdownReady(Vec<ProviderEntry>, Vec<(String, Vec<zeus_provider::ModelInfo>)>),
    /// `/model`'s picker finished probing — empty entries means "no models found".
    ModelPickerReady(Vec<PickerEntry>, usize, Vec<(String, Vec<zeus_provider::ModelInfo>)>),
    /// `/provider`'s picker finished probing — empty entries means "no providers".
    ProviderPickerReady(Vec<ProviderEntry>, usize, Vec<(String, Vec<zeus_provider::ModelInfo>)>),
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
#[derive(Clone)]
enum PickerEntry {
    Header(String),
    /// A vendor-family grouping nested under a `Header` — e.g. "Anthropic"
    /// under the "OPENROUTER" provider header, when that provider's catalog
    /// spans more than one recognizable family. Non-selectable, same as
    /// `Header` — every "skip non-Model rows" navigation check already
    /// matches on `Model { .. }` specifically, so this needs no separate
    /// handling there.
    SubHeader(String),
    Model { provider: String, model: ModelInfo },
}

/// One row in the provider picker: a non-selectable group header (paid /
/// free / local), a selectable provider, or a selectable model belonging to
/// that provider. Flat list so a single index drives keyboard and mouse
/// selection.
#[derive(Clone)]
enum ProviderEntry {
    Header(String),
    /// A vendor-family grouping nested under a provider's `Header` — see
    /// `PickerEntry::SubHeader`. Non-selectable, same as `Header`.
    SubHeader(String),
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
        if !matches!(entries[idx as usize], ProviderEntry::Header(_) | ProviderEntry::SubHeader(_)) {
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
        if !matches!(entries.get(idx), Some(ProviderEntry::Header(_) | ProviderEntry::SubHeader(_))) {
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
        if !matches!(entries.get(idx), Some(ProviderEntry::Header(_) | ProviderEntry::SubHeader(_))) {
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
    // A newly-keyed provider can now list models it couldn't before —
    // drop the cached scan so the next picker/dropdown open re-probes
    // instead of showing it as still empty.
    state.model_cache = None;
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
            state.context_window = None;
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
        // A header row isn't a choice — leave the dropdown open rather
        // than silently closing it on a stray Enter.
        ProviderEntry::Header(_) | ProviderEntry::SubHeader(_) => {}
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

/// Infers the underlying model vendor/family from its id — used to further
/// Recognizes well-known public model-family prefixes, for sub-grouping an
/// aggregator provider's (OpenRouter, OpenCode Zen) catalog — which
/// otherwise dumps dozens of unrelated vendors' models under one flat list
/// with no way to tell a Claude model from a Gemini one at a glance.
/// Returns `None` for anything unrecognized rather than inventing a family
/// from the id itself: a provider's small in-house catalog (e.g. six
/// differently-named free models with no shared vendor) must stay one flat
/// list, not fragment into six one-model "families".
fn model_family(id: &str) -> Option<&'static str> {
    let lower = id.to_ascii_lowercase();
    // Aggregators commonly namespace ids as "vendor/model-name".
    let lower = lower.rsplit('/').next().unwrap_or(&lower);
    const FAMILIES: &[(&str, &str)] = &[
        ("claude", "Anthropic"),
        ("chatgpt", "OpenAI"),
        ("gpt", "OpenAI"),
        ("o1", "OpenAI"),
        ("o3", "OpenAI"),
        ("o4", "OpenAI"),
        ("gemini", "Google Gemini"),
        ("grok", "xAI"),
        ("deepseek", "DeepSeek"),
        ("glm", "Zhipu GLM"),
        ("minimax", "MiniMax"),
        ("llama", "Meta Llama"),
        ("qwen", "Qwen"),
        ("mixtral", "Mistral"),
        ("codestral", "Mistral"),
        ("mistral", "Mistral"),
        ("command", "Cohere"),
        ("phi", "Microsoft Phi"),
        ("sonar", "Perplexity"),
        ("nova", "Amazon Nova"),
        ("yi-", "01.AI"),
    ];
    FAMILIES.iter().find(|(prefix, _)| lower.starts_with(prefix)).map(|(_, family)| *family)
}

/// Splits a provider's model catalog into vendor-family sub-groups for
/// display, sorted alphabetically by family name. Only kicks in once at
/// least two *recognized* families are actually present — a single-vendor
/// provider (Anthropic, OpenAI, …), or a small in-house catalog with no
/// recognizable big-lab prefixes at all, returns one `None`-keyed group
/// (meaning "no sub-header, render flat") instead. Models with no
/// recognized family are pooled into that same flat, header-less group —
/// they're never given their own singleton sub-header.
fn group_models_by_family(
    models: &[zeus_provider::ModelInfo],
) -> Vec<(Option<String>, Vec<zeus_provider::ModelInfo>)> {
    let recognized: std::collections::BTreeSet<&str> =
        models.iter().filter_map(|m| model_family(&m.id)).collect();
    if recognized.len() <= 1 {
        return vec![(None, models.to_vec())];
    }
    let mut unrecognized = Vec::new();
    let mut by_family: std::collections::BTreeMap<&str, Vec<zeus_provider::ModelInfo>> =
        std::collections::BTreeMap::new();
    for m in models {
        match model_family(&m.id) {
            Some(family) => by_family.entry(family).or_default().push(m.clone()),
            None => unrecognized.push(m.clone()),
        }
    }
    let mut groups: Vec<(Option<String>, Vec<zeus_provider::ModelInfo>)> = Vec::new();
    if !unrecognized.is_empty() {
        groups.push((None, unrecognized));
    }
    groups.extend(by_family.into_iter().map(|(f, ms)| (Some(f.to_string()), ms)));
    groups
}

/// Build grouped provider-picker entries: one header row per *provider*
/// (Anthropic, OpenRouter, OpenCode Zen, …) — the current provider leads,
/// the rest follow alphabetically, same ordering as the top-right dropdown.
/// Each provider's row carries its kind, default model, and whether it's
/// immediately usable (local kind, stored key, or env key set); its real
/// models (when reachable) are listed underneath, each individually tagged
/// free or paid — a single provider like OpenRouter can offer both. A
/// provider that can't list models (no key, server down) still shows as a
/// switchable row.
fn provider_picker_entries(
    config: &Config,
    current: &str,
    current_model: &str,
    models: &[(String, Vec<zeus_provider::ModelInfo>)],
) -> (Vec<ProviderEntry>, usize) {
    let mut names: Vec<&String> = config.providers.providers.keys().collect();
    names.sort();
    if let Some(pos) = names.iter().position(|n| n.as_str() == current) {
        let cur = names.remove(pos);
        names.insert(0, cur);
    }
    // Pull a few recommended providers (skipping the current one, already
    // up front) just after it, in a fixed order — a lightweight take on
    // other tools' "Popular" section, without a whole separate no-models
    // "connect a provider" screen.
    const POPULAR: [&str; 4] = ["opencodezen", "openrouter", "anthropic", "openai"];
    let mut names = names;
    let mut ordered: Vec<&String> = Vec::with_capacity(names.len());
    if !names.is_empty() {
        ordered.push(names.remove(0));
    }
    for pop in POPULAR {
        if let Some(pos) = names.iter().position(|n| n.as_str() == pop) {
            ordered.push(names.remove(pos));
        }
    }
    ordered.extend(names);
    let names = ordered;

    let mut entries = Vec::new();
    // Defaults to the current provider's row (always index 1: its header
    // goes in first since `current` was moved to the front of `names`
    // above) rather than 0, which is always a header — so Enter works
    // immediately without needing an arrow key press first.
    let mut selected = 1;
    for name in names {
        let Some(cfg) = config.providers.get(name) else { continue };
        let ready = provider_status_ok(config, name);
        entries.push(ProviderEntry::Header(name.clone()));
        if name.as_str() == current && ready {
            selected = entries.len();
        }
        entries.push(ProviderEntry::Provider {
            name: name.to_string(),
            kind: cfg.kind.clone(),
            model: cfg.default_model.clone().unwrap_or_default(),
            ready,
        });
        // Models for this provider, if reachable, right under its row —
        // sub-grouped by vendor family for an aggregator (OpenRouter,
        // OpenCode Zen) whose catalog spans more than one recognizable
        // vendor, so it doesn't dump 30+ unrelated models in one flat list.
        if let Some((_, provider_models)) = models.iter().find(|(n, _)| n == name) {
            for (family, family_models) in group_models_by_family(provider_models) {
                if let Some(family) = family {
                    entries.push(ProviderEntry::SubHeader(family));
                }
                for m in family_models {
                    if m.id == current_model && name.as_str() == current {
                        selected = entries.len();
                    }
                    let free = is_free_model(&m.id);
                    entries.push(ProviderEntry::Model {
                        provider: name.to_string(),
                        model: m,
                        free,
                    });
                }
            }
        }
    }
    (entries, selected)
}

/// Backs `/model`'s no-argument picker: probes every configured provider,
/// leads with "Favorites" then "Recent" sections (each only shown when
/// non-empty, and only for models that actually came back from the scan),
/// puts the current provider's own group first among the rest (like
/// opencode's own picker), and flattens into `PickerEntry` rows with the
/// current model pre-selected. Empty result means "no models found".
fn build_model_picker_entries(
    models: &[(String, Vec<zeus_provider::ModelInfo>)],
    current_provider: &str,
    current_model: &str,
    recent: &[(String, String)],
    favorites: &[(String, String)],
) -> (Vec<PickerEntry>, usize) {
    let find = |provider: &str, model_id: &str| -> Option<zeus_provider::ModelInfo> {
        models
            .iter()
            .find(|(p, _)| p == provider)
            .and_then(|(_, ms)| ms.iter().find(|m| m.id == model_id))
            .cloned()
    };
    let mut entries = Vec::new();
    let mut selected = None;
    let push_section = |entries: &mut Vec<PickerEntry>, selected: &mut Option<usize>, title: &str, pairs: &[(String, String)]| {
        let rows: Vec<(String, zeus_provider::ModelInfo)> = pairs
            .iter()
            .filter_map(|(p, m)| find(p, m).map(|mi| (p.clone(), mi)))
            .collect();
        if rows.is_empty() {
            return;
        }
        entries.push(PickerEntry::Header(title.to_string()));
        for (provider, model) in rows {
            if selected.is_none() && model.id == current_model && provider == current_provider {
                *selected = Some(entries.len());
            }
            entries.push(PickerEntry::Model { provider, model });
        }
    };
    push_section(&mut entries, &mut selected, "Favorites", favorites);
    push_section(&mut entries, &mut selected, "Recent", recent);

    let mut groups = models.to_vec();
    if let Some(pos) = groups.iter().position(|(name, _)| name == current_provider) {
        let current = groups.remove(pos);
        groups.insert(0, current);
    }
    for (provider_name, models) in groups {
        entries.push(PickerEntry::Header(provider_name.clone()));
        // Sub-group by vendor family for an aggregator (OpenRouter, OpenCode
        // Zen) whose catalog spans more than one recognizable vendor — see
        // `group_models_by_family`.
        for (family, family_models) in group_models_by_family(&models) {
            if let Some(family) = family {
                entries.push(PickerEntry::SubHeader(family));
            }
            for model in family_models {
                if selected.is_none() && model.id == current_model && provider_name == current_provider {
                    selected = Some(entries.len());
                }
                entries.push(PickerEntry::Model { provider: provider_name.clone(), model });
            }
        }
    }
    let selected = selected.unwrap_or_else(|| first_selectable_picker(&entries));
    (entries, selected)
}

/// First `Model` row in a `PickerEntry` list — the default selection so
/// Enter works immediately without an arrow-key press first (index 0 is
/// always a `Header`).
fn first_selectable_picker(entries: &[PickerEntry]) -> usize {
    entries.iter().position(|e| matches!(e, PickerEntry::Model { .. })).unwrap_or(0)
}

/// Opens the `/provider` grouped picker popup — shared by the `/provider`
/// command and the "no provider connected yet" nudge on a failed send. A
/// cached scan opens it instantly; otherwise probing every configured
/// provider can take several seconds (each has up to a 3s timeout), so
/// that's spawned rather than awaited here — awaiting inline would freeze
/// the whole render loop for that long.
fn open_provider_picker(state: &mut AppState, config: &Config, ui_tx: &mpsc::UnboundedSender<UiEvent>) {
    if let Some(groups) = state.model_cache.clone() {
        let (entries, selected) = provider_picker_entries(config, &state.provider, &state.model, &groups);
        if entries.is_empty() {
            state.push_error("no providers configured — see config.toml / providers.toml");
        } else {
            state.mode = Mode::ProviderPicker { entries, selected };
        }
    } else {
        state.fetching_providers = true;
        state.push_info("fetching providers…");
        let cfg = config.clone();
        let provider = state.provider.clone();
        let model = state.model.clone();
        let tx = ui_tx.clone();
        tokio::spawn(async move {
            let groups = list_models_by_provider(&cfg).await;
            let (entries, selected) = provider_picker_entries(&cfg, &provider, &model, &groups);
            let _ = tx.send(UiEvent::ProviderPickerReady(entries, selected, groups));
        });
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
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
) {
    let parts: Vec<&str> = arg.split_whitespace().collect();
    match parts.as_slice() {
        // `/provider` with no args — open the grouped picker popup.
        [] => open_provider_picker(state, config, ui_tx),
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
    /// True from `PlanGenerated` until the orchestrated run ends. While a
    /// real plan is driving the checklist, `complete_task`'s generic
    /// tool-name heuristic must stay out of the way — it exists to give
    /// non-orchestrated turns *some* checklist feedback, but during a real
    /// plan it would mark whichever step happens to be first-undone as done
    /// just because *some* unrelated tool call succeeded, stealing credit
    /// from the step actually in progress and leaving the checklist telling
    /// a false story about what's finished.
    plan_active: bool,
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
    /// Rendered rects of the empty-state's three example chips, for mouse
    /// hit-testing — same pattern as `provider_btn_area`.
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

/// Transcript search — an overlay on top of `Mode::Chat`, not a `Mode`
/// variant of its own, the same "extra popup state alongside normal chat"
/// pattern `DropdownState` already uses; a new `Mode` would need updating
/// every exhaustive match on it across this file for one narrow feature.
struct SearchState {
    query: String,
    /// Indices into `AppState::transcript` whose text matches `query`
    /// (case-insensitive substring), oldest first — recomputed on every
    /// keystroke.
    matches: Vec<usize>,
    /// Which entry in `matches` is currently focused, jumped to with Enter.
    current: usize,
}

/// `/sessions` picker — an overlay on `Mode::Chat`, same pattern as
/// `SearchState`: Up/Down/click to pick, Enter/click-to-apply to resume,
/// Esc to close.
struct SessionPickerState {
    entries: Vec<SessionSummary>,
    selected: usize,
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

/// Builds the compact provider/model rows for the top-right dropdown,
/// regrouping configured providers so the current one leads, then the rest in
/// alphabetical order — same scanning approach as the full `/provider` picker.
fn build_dropdown_entries(
    config: &Config,
    current: &str,
    _current_model: &str,
    models: &[(String, Vec<zeus_provider::ModelInfo>)],
) -> Vec<ProviderEntry> {
    let mut names: Vec<&String> = config.providers.providers.keys().collect();
    names.sort();
    if let Some(pos) = names.iter().position(|n| n.as_str() == current) {
        let cur = names.remove(pos);
        names.insert(0, cur);
    }
    let mut entries = Vec::new();
    let mut any = false;
    for name in names {
        let Some(cfg) = config.providers.get(name) else { continue };
        let ready = provider_status_ok(config, name);
        any = true;
        entries.push(ProviderEntry::Header(name.to_string()));
        let provider_models = models
            .iter()
            .find(|(n, _)| n == (name as &String))
            .map(|(_, m)| m.as_slice())
            .unwrap_or(&[]);
        if provider_models.is_empty() {
            // Nothing fetched yet (needs a key, or unreachable) — a single
            // status row instead of repeating the header with no models
            // under it.
            entries.push(ProviderEntry::Provider {
                name: name.to_string(),
                kind: cfg.kind.clone(),
                model: cfg.default_model.clone().unwrap_or_default(),
                ready,
            });
        } else {
            for (family, family_models) in group_models_by_family(provider_models) {
                if let Some(family) = family {
                    entries.push(ProviderEntry::SubHeader(family));
                }
                for m in family_models {
                    let free = is_free_model(&m.id);
                    entries.push(ProviderEntry::Model {
                        provider: name.to_string(),
                        model: m,
                        free,
                    });
                }
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
            plan_active: false,
            drop_search: String::new(),
            todo_area: None,
            provider_btn_area: None,
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
        self.transcript_scroll = if next >= self.transcript_max_scroll { None } else { Some(next) };
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
        self.recent_models.retain(|(p, m)| !(p == provider && m == model));
        self.recent_models.insert(0, (provider.to_string(), model.to_string()));
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
            self.favorite_models.push((provider.to_string(), model.to_string()));
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
            AgentEvent::ToolCallStarted { id, name, arguments } => {
                self.flush_current_reply();
                let path = touched_path(&name, &arguments);
                self.tool_call_meta.insert(id, (std::time::Instant::now(), path));
                self.transcript.push(Block_::new(
                    Role::Tool,
                    format!("{name} {arguments}"),
                ));
            }
            AgentEvent::ToolCallFinished { id, name, result, is_error } => {
                self.flush_current_reply();
                let (elapsed, path) = match self.tool_call_meta.remove(&id) {
                    Some((start, path)) => (Some(start.elapsed()), path),
                    None => (None, None),
                };
                let role = if is_error { Role::ToolError } else { Role::Tool };
                let marker = if is_error { "failed" } else { "done" };
                let timing = elapsed.map(|d| format!(", {}", fmt_duration(d))).unwrap_or_default();
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
                    && matches!(name.as_str(), "apply_patch" | "write" | "edit" | "update" | "run" | "patch")
                {
                    self.complete_task(&name);
                }
            }
            AgentEvent::Compacted(c) => {
                self.push_info(format!("(compacted {} earlier message(s))", c.removed_messages));
            }
            AgentEvent::Cancelled => {
                self.push_info("(cancelled)");
                self.plan_active = false;
            }
            AgentEvent::Done => self.flush_current_reply(),
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
                    .map(|s| TodoItem { text: s.description.clone(), done: false, active: false, duration: None })
                    .collect();
            }
            AgentEvent::PlanStepStarted { step } => {
                self.push_info(format!("plan step {} · {}", step.id, step.description));
                self.active_persona = step.persona.clone();
                self.plan_step_started.insert(step.description.clone(), std::time::Instant::now());
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
                let elapsed = self.plan_step_started.remove(&step.description).map(|t| t.elapsed());
                let timing = elapsed.map(|d| format!(" ({})", fmt_duration(d))).unwrap_or_default();
                self.transcript.push(Block_::new(
                    Role::Tool,
                    format!("step {} done{timing} · {}\n{}", step.id, step.description, summary),
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
            AgentEvent::WorkflowStarted { id, description, phases } => {
                let roster = phases
                    .iter()
                    .map(|p| format!("{} [{}]", p.prompt, p.persona))
                    .collect::<Vec<_>>()
                    .join(" → ");
                self.push_info(format!("workflow '{id}' — {description}"));
                self.transcript
                    .push(Block_::new(Role::Info, format!("{roster}")));
            }
            AgentEvent::WorkflowPhaseStarted { name, persona } => {
                self.push_info(format!("▶ {name} [as {persona}]"));
                self.active_persona = Some(persona);
            }
            AgentEvent::WorkflowPhaseDone { name, persona, summary } => {
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
                self.transcript.push(Block_::new(
                    Role::Info,
                    format!("review ({persona})"),
                ));
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
                self.todos.last().map(|x| x.text.clone()).unwrap_or_default()
            };
            if !self.todos.iter().any(|x| x.done && x.text == text) {
                self.todos.push(TodoItem { text, done: true, active: false, duration: None });
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
        let Some(block_idx) = self.search.as_ref().and_then(|s| s.matches.get(s.current).copied()) else {
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

/// Masks a secret (API key) as it's typed/pasted — all but the last
/// character become `•`, so you can confirm you pasted the right thing
/// without the key sitting in plaintext on screen.
fn mask_secret(s: &str) -> String {
    let mut masked = "•".repeat(s.chars().count().saturating_sub(1));
    if let Some(last) = s.chars().last() {
        masked.push(last);
    }
    masked
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

/// One-line pitch + signup URL for the key-entry modal — the "why this
/// provider, where do I get a key" copy a first-time user actually needs.
/// Any provider not listed here (custom entries in `providers.toml`) still
/// gets a sane generic prompt.
fn provider_blurb(name: &str) -> (&'static str, Option<&'static str>) {
    match name {
        "opencodezen" => (
            "OpenCode Zen gives you access to a curated set of coding models — including free tiers — with a single API key.",
            Some("https://opencode.ai/zen"),
        ),
        "openrouter" => (
            "OpenRouter routes a single API key to hundreds of models across many providers, including several free ones.",
            Some("https://openrouter.ai/keys"),
        ),
        "anthropic" => ("Claude models, directly from Anthropic.", Some("https://console.anthropic.com/settings/keys")),
        "openai" => ("GPT models, directly from OpenAI.", Some("https://platform.openai.com/api-keys")),
        "gemini" => ("Gemini models, directly from Google.", Some("https://aistudio.google.com/apikey")),
        "grok" => ("Grok models, directly from xAI.", Some("https://console.x.ai")),
        "deepseek" => ("DeepSeek's own models, directly from DeepSeek.", Some("https://platform.deepseek.com/api_keys")),
        "mistral" => ("Mistral's own models, directly from Mistral AI.", Some("https://console.mistral.ai/api-keys")),
        "groq" => ("Very fast inference on open models, via Groq's LPU hardware.", Some("https://console.groq.com/keys")),
        "together" => ("A wide catalog of open models hosted by Together AI.", Some("https://api.together.ai/settings/api-keys")),
        "fireworks" => ("Fast hosted inference for open models, via Fireworks AI.", Some("https://fireworks.ai/account/api-keys")),
        "perplexity" => ("Perplexity's search-grounded models.", Some("https://www.perplexity.ai/settings/api")),
        "cohere" => ("Cohere's Command model family.", Some("https://dashboard.cohere.com/api-keys")),
        "cerebras" => ("Extremely fast inference on open models, via Cerebras hardware.", Some("https://cloud.cerebras.ai")),
        "deepinfra" => ("A wide catalog of open models hosted by DeepInfra.", Some("https://deepinfra.com/dash/api_keys")),
        "novita" => ("A wide catalog of open models hosted by Novita AI.", Some("https://novita.ai/settings/key-management")),
        _ => ("Paste your API key below to connect this provider.", None),
    }
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct FavoritesFile {
    #[serde(default)]
    favorites: Vec<FavEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct FavEntry {
    provider: String,
    model: String,
}

/// Lives next to `keys.toml` — same directory, same "small user-editable
/// TOML file in the zeus home dir" convention.
fn favorites_path(config: &Config) -> std::path::PathBuf {
    config
        .global
        .keys_toml
        .parent()
        .map(|p| p.join("favorites.toml"))
        .unwrap_or_else(|| std::path::PathBuf::from("favorites.toml"))
}

fn load_favorites(config: &Config) -> Vec<(String, String)> {
    let path = favorites_path(config);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    toml::from_str::<FavoritesFile>(&text)
        .map(|f| f.favorites.into_iter().map(|e| (e.provider, e.model)).collect())
        .unwrap_or_default()
}

fn save_favorites(config: &Config, favorites: &[(String, String)]) {
    let path = favorites_path(config);
    let file = FavoritesFile {
        favorites: favorites
            .iter()
            .map(|(provider, model)| FavEntry { provider: provider.clone(), model: model.clone() })
            .collect(),
    };
    if let Ok(text) = toml::to_string_pretty(&file) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, text);
    }
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
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

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
    /// `--violet`: brand accent (`#a855ff`) — the fallback `accent()` uses
    /// when `settings.accent_color` isn't set.
    pub const VIOLET: Color = Color::Rgb(0xa8, 0x55, 0xff);
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

    /// zeus-empty-state.html `--ink` page background (`#080b13`) — the
    /// full-screen splash's own palette, distinct from `zeus-cli.html`'s.
    pub const INK: Color = Color::Rgb(0x08, 0x0b, 0x13);
    /// `--muted`: empty-state secondary text (`#8290ad`).
    pub const MUTED: Color = Color::Rgb(0x82, 0x90, 0xad);
    /// `--faint` (empty-state): tertiary text (`#566180`).
    pub const EMPTY_FAINT: Color = Color::Rgb(0x56, 0x61, 0x80);
    /// `--teal`: empty-state status dot + composer accent (`#57efce`).
    pub const TEAL: Color = Color::Rgb(0x57, 0xef, 0xce);
    /// `--cyan` (empty-state): send-button "ready" color, the far end of the
    /// HTML's `linear-gradient(120deg, var(--teal), var(--cyan))` (`#58c0ff`).
    pub const EMPTY_CYAN: Color = Color::Rgb(0x58, 0xc0, 0xff);
    /// Wordmark gradient stops — near-white → `--gold-soft` → `--gold`,
    /// matching the HTML's `linear-gradient(100deg, #eef3ff 0%, #f6d98a 52%, #f2c661 100%)`.
    pub const WORDMARK_START: Color = Color::Rgb(0xee, 0xf3, 0xff);
    pub const WORDMARK_MID: Color = Color::Rgb(0xf6, 0xd9, 0x8a);
    pub const WORDMARK_END: Color = Color::Rgb(0xf2, 0xc6, 0x61);

    // Packed as 0x00RRGGBB; the top byte doubles as an "unset" flag (real
    // RGB values never set it) since `AtomicU32` has no niche for `None` the
    // way `Option<Color>` would — a plain atomic (rather than a `OnceLock`)
    // is what lets `/settings accent` change this again at runtime instead
    // of only ever being set once at startup.
    const ACCENT_UNSET: u32 = u32::MAX;
    static ACCENT: AtomicU32 = AtomicU32::new(ACCENT_UNSET);
    static REDUCED_MOTION: AtomicBool = AtomicBool::new(false);
    static NOTIFY_ON_COMPLETION: AtomicBool = AtomicBool::new(true);

    /// Seeds the runtime-adjustable display settings from `Config` once,
    /// before the first frame — called from `tui::run`. `/settings`
    /// afterwards updates these same atomics directly, so a change applies
    /// immediately without a restart, the same way switching model/provider
    /// already does.
    pub fn init_runtime(accent_hex: Option<&str>, reduced_motion: bool, notify_on_completion: bool) {
        if let Some(color) = accent_hex.and_then(parse_hex_color) {
            set_accent(color);
        }
        REDUCED_MOTION.store(reduced_motion, Ordering::Relaxed);
        NOTIFY_ON_COMPLETION.store(notify_on_completion, Ordering::Relaxed);
    }

    fn parse_hex_color(s: &str) -> Option<Color> {
        let s = s.trim().trim_start_matches('#');
        if s.len() != 6 {
            return None;
        }
        let r = u8::from_str_radix(&s[0..2], 16).ok()?;
        let g = u8::from_str_radix(&s[2..4], 16).ok()?;
        let b = u8::from_str_radix(&s[4..6], 16).ok()?;
        Some(Color::Rgb(r, g, b))
    }

    /// Sets the brand accent color right now, applied on the very next
    /// frame. `/settings accent <#hex>` behind this; `parse_hex_color` gates
    /// the string form so a bad hex string never reaches here.
    pub fn set_accent(color: Color) {
        if let Color::Rgb(r, g, b) = color {
            let packed = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
            ACCENT.store(packed, Ordering::Relaxed);
        }
    }

    /// Clears the accent override, reverting to the built-in violet.
    pub fn reset_accent() {
        ACCENT.store(ACCENT_UNSET, Ordering::Relaxed);
    }

    /// Parses a `#rrggbb` string and applies it as the accent if valid;
    /// returns whether it parsed — lets `/settings accent <hex>` reject a
    /// bad string with a clear error instead of silently no-op'ing.
    pub fn set_accent_hex(hex: &str) -> bool {
        match parse_hex_color(hex) {
            Some(color) => {
                set_accent(color);
                true
            }
            None => false,
        }
    }

    /// Brand accent — the configured `accent_color` when set, else the
    /// built-in violet. Used wherever the UI needs the "current brand
    /// color" as a bare `Color` rather than a `Style`.
    pub fn accent() -> Color {
        let packed = ACCENT.load(Ordering::Relaxed);
        if packed == ACCENT_UNSET {
            VIOLET
        } else {
            Color::Rgb(((packed >> 16) & 0xff) as u8, ((packed >> 8) & 0xff) as u8, (packed & 0xff) as u8)
        }
    }

    pub fn reduced_motion() -> bool {
        REDUCED_MOTION.load(Ordering::Relaxed)
    }

    pub fn set_reduced_motion(v: bool) {
        REDUCED_MOTION.store(v, Ordering::Relaxed);
    }

    pub fn notify_on_completion() -> bool {
        NOTIFY_ON_COMPLETION.load(Ordering::Relaxed)
    }

    pub fn set_notify_on_completion(v: bool) {
        NOTIFY_ON_COMPLETION.store(v, Ordering::Relaxed);
    }

    /// A muted (60% brightness) variant of `accent()` for subtle borders.
    pub fn accent_dim() -> Color {
        match accent() {
            Color::Rgb(r, g, b) => Color::Rgb(
                (r as f32 * 0.6) as u8,
                (g as f32 * 0.6) as u8,
                (b as f32 * 0.6) as u8,
            ),
            other => other,
        }
    }

    pub fn muted() -> Style {
        Style::default().fg(MUTED)
    }
    pub fn empty_faint() -> Style {
        Style::default().fg(EMPTY_FAINT)
    }
    pub fn teal() -> Style {
        Style::default().fg(TEAL)
    }

    pub fn violet() -> Style {
        Style::default().fg(accent())
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
/// rows: `.pc` command labels colored `--mode-color`, `.pd` descriptions
/// dim, `.active` row tinted with the mode color).
fn render_menu(f: &mut Frame, area: Rect, matches: &[(&str, &str)], selected: usize, accent: Color) -> Rect {
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
                Span::styled(format!("/{name:<name_width$}"), Style::default().fg(accent).add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled(desc.to_string(), placeholder_style()),
            ]))
        })
        .collect();
    let list = List::new(items).highlight_style(
        Style::default().bg(theme::PANEL2).fg(accent).add_modifier(Modifier::BOLD),
    );
    let mut list_state = ListState::default();
    list_state.select(Some(selected.min(matches.len().saturating_sub(1))));
    f.render_stateful_widget(list, inner, &mut list_state);
    inner
}

/// The pinned input box: a mode-colored accent bar down the left edge, an
/// elevated flat panel (no full box border), the placeholder/typed text on
/// its own line, and mode/model/provider status on the line beneath it —
/// no `›` caret, no SEND button; Enter is the only way to send. The bar
/// color and status line repaint whenever the mode switches.
fn render_input_box(f: &mut Frame, area: Rect, state: &AppState, input_text_h: u16) {
    let accent = mode_accent(state.agent_mode);
    let focused = !state.busy && matches!(state.mode, Mode::Chat);
    let bar = if focused { accent } else { theme::BORDER };
    // A single thick accent bar down the left edge on a flat elevated panel
    // — no full box border, no `›` caret — rather than the earlier
    // all-sides bordered composer.
    let block = Block::default()
        .borders(Borders::LEFT)
        .border_type(BorderType::Thick)
        .border_style(Style::default().fg(bar))
        .style(Style::default().bg(theme::PANEL2));
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

    if matches!(state.mode, Mode::Approval(_)) {
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

    let input_line = if state.busy {
        Line::from(Span::styled(
            format!("{} zeus is working…", spinner_glyph(state)),
            placeholder_style(),
        ))
    } else if state.input.is_empty() {
        Line::from(vec![
            Span::styled("Ask anything… ", placeholder_style()),
            Span::styled("\"Fix a TODO in the codebase\"", placeholder_style()),
        ])
    } else {
        Line::from(Span::raw(state.input.clone()))
    };
    f.render_widget(Paragraph::new(input_line).wrap(Wrap { trim: false }), text_row);

    // Mode · model provider — no send button, no caret; Enter is the only
    // way to send, same as it always was, just no longer advertised with a
    // dedicated button now that the box has no border to anchor one to.
    let status = Line::from(vec![
        Span::styled(state.agent_mode.label(), Style::default().fg(accent).add_modifier(Modifier::BOLD)),
        Span::styled(" · ", theme::faint()),
        Span::styled(state.model.clone(), theme::text().add_modifier(Modifier::BOLD)),
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
        let wrapped_before = wrap_text(&typed_before, inner.width.max(1) as usize);
        let last_row = input_text_h.saturating_sub(1);
        let row = (wrapped_before.len() as u16).saturating_sub(1).min(last_row);
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
        Span::styled(format!(" {label}"), theme::faint()),
    ]
}

/// Mode/model/provider now lives inside the input box itself (its second
/// line), so this row is just the key-binding legend underneath it.
fn render_hints(f: &mut Frame, area: Rect, _state: &AppState) {
    let mut spans = hint_pair("tab", "agents");
    spans.push(Span::raw("    "));
    spans.extend(hint_pair("/", "commands"));
    spans.push(Span::raw("    "));
    spans.extend(hint_pair("click msg", "copy"));
    spans.push(Span::raw("    "));
    spans.extend(hint_pair("ctrl+f", "find"));
    spans.push(Span::raw("    "));
    spans.extend(hint_pair("esc", "close"));
    f.render_widget(Paragraph::new(Line::from(spans)), area);
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
    let mut lines: Vec<Line<'static>> = Vec::new();
    for block in &state.transcript {
        lines.extend(block.to_lines(width));
        lines.push(Line::from(""));
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
        format!("{}  ·  {}/{}", search.query, search.current + 1, search.matches.len())
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
        .style(Style::default().bg(theme::PANEL))
        .title(Line::from(Span::styled(
            " find ",
            Style::default().fg(theme::accent()).add_modifier(Modifier::BOLD),
        )))
        .title_bottom(
            Line::from(Span::styled(" enter next · esc close ", theme::faint())).alignment(Alignment::Center),
        );
    let inner = block.inner(bar_area);
    f.render_widget(block, bar_area);
    f.render_widget(Paragraph::new(Line::from(Span::styled(label, theme::text()))), inner);
}

/// The `/sessions` picker: saved conversations, arrow keys or click to
/// navigate, Enter/click to resume, Esc to close — modeled after
/// `render_model_picker`'s popup chrome. Returns the list's rect for mouse
/// hit-testing.
fn render_session_picker(f: &mut Frame, area: Rect, picker: &SessionPickerState) -> Rect {
    let width = area.width.saturating_sub(6).clamp(40, 96);
    let height = (picker.entries.len() as u16 + 4)
        .min(PICKER_MAX_H)
        .clamp(8, area.height.saturating_sub(4).max(8));
    let popup = centered_rect(width, height, area);

    f.render_widget(Clear, popup);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style())
        .style(Style::default().bg(theme::PANEL))
        .title(Line::from(vec![Span::styled(
            " resume session ",
            theme::green().add_modifier(Modifier::BOLD),
        )]))
        .title(
            Line::from(Span::styled(format!("{} ", picker.entries.len()), theme::faint()))
                .alignment(Alignment::Right),
        )
        .title_bottom(
            Line::from(Span::styled(" ↑/↓ navigate · enter resume · esc dismiss ", theme::faint()))
                .alignment(Alignment::Center),
        );
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let inner_w = inner.width as usize;
    let items: Vec<ListItem> = picker
        .entries
        .iter()
        .map(|s| {
            let preview = if s.last_user.is_empty() {
                "(no user message yet)".to_string()
            } else {
                s.last_user.clone()
            };
            let head = format!("{}  {} msg  ", s.id, s.message_count);
            let room = inner_w.saturating_sub(char_count(&head));
            let preview: String = preview.chars().take(room).collect();
            ListItem::new(Line::from(vec![
                Span::styled(s.id.clone(), theme::text().add_modifier(Modifier::BOLD)),
                Span::styled(format!("  {} msg  ", s.message_count), theme::dim()),
                Span::styled(preview, theme::faint()),
            ]))
        })
        .collect();
    let list = List::new(items).highlight_style(
        Style::default().bg(theme::PANEL2).fg(theme::accent()).add_modifier(Modifier::BOLD),
    );
    let mut list_state = ListState::default();
    list_state.select(Some(picker.selected.min(picker.entries.len().saturating_sub(1))));
    f.render_stateful_widget(list, inner, &mut list_state);

    inner
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
async fn resume_session(session_id: String, config: &Config, agent_slot: &mut Option<Agent>, state: &mut AppState) {
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
            state.push_info(format!("resumed session={session_id} — continuing from its saved context"));
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
    let menu_h = menu_height(&state.command_matches());
    // The text row grows with the input — a long message used to be
    // capped at a single fixed row and silently clip everything past the
    // first wrapped line, with no indication anything was cut off. Capped
    // at 6 rows so a very long paste can't eat the whole transcript; a
    // second, always-1-row line underneath carries mode/model/provider
    // status normally (a prompt/preview line for the TUI-only
    // `Approval`/`KeyEntry` states) — no top/bottom border to add for
    // (the box only has a left accent bar, not a full box border).
    let composer_inner_w = area.width.saturating_sub(3).max(10) as usize;
    let input_text_h = wrap_text(&state.input, composer_inner_w).len().clamp(1, 6) as u16;
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

    state.command_menu_area = if menu_h > 0 {
        let matches = state.command_matches();
        Some(render_menu(f, rows[1], &matches, state.command_selected, mode_accent(state.agent_mode)))
    } else {
        None
    };
    render_input_box(f, rows[2], state, input_text_h);
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

/// The `Mode::KeyEntry` screen — a centered "API key" modal with a short
/// pitch for the provider, a link to get a key, a masked input field, and
/// an `enter submit` footer. Returns the input field's inner rect so the
/// caller can place the terminal cursor in it.
fn render_key_entry_modal(f: &mut Frame, area: Rect, provider: &str, input: &str) -> Rect {
    let (blurb, url) = provider_blurb(provider);
    let width = area.width.saturating_sub(10).clamp(40, 76);
    let blurb_lines = textwrap_len(blurb, width as usize - 4);
    // title, blank, blurb lines, blank, url line (if any), blank, input box (3), blank, footer
    let height = (3 + blurb_lines + if url.is_some() { 2 } else { 0 } + 5)
        .clamp(9, area.height.saturating_sub(4).max(9) as usize) as u16;
    let popup = centered_rect(width, height, area);

    f.render_widget(Clear, popup);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style())
        .style(Style::default().bg(theme::PANEL))
        .title(Line::from(vec![Span::styled(
            " API key ",
            theme::green().add_modifier(Modifier::BOLD),
        )]))
        .title(Line::from(Span::styled(" esc ", theme::faint())).alignment(Alignment::Right))
        .title_bottom(
            Line::from(Span::styled(" enter submit ", theme::faint())).alignment(Alignment::Center),
        );
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let mut rows: Vec<Constraint> = vec![Constraint::Length(1)];
    for _ in 0..blurb_lines {
        rows.push(Constraint::Length(1));
    }
    if url.is_some() {
        rows.push(Constraint::Length(1));
        rows.push(Constraint::Length(1));
    }
    rows.push(Constraint::Length(1));
    rows.push(Constraint::Length(3));
    let rows = Layout::vertical(rows).split(inner);

    let mut r = 0usize;
    f.render_widget(
        Paragraph::new(format!("Paste your {provider} API key below.")).style(theme::dim()),
        rows[r],
    );
    r += 1;
    for line in wrap_text(blurb, width as usize - 4) {
        f.render_widget(Paragraph::new(line).style(theme::text()), rows[r]);
        r += 1;
    }
    if let Some(url) = url {
        r += 1; // blank spacer row
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("Go to ", theme::dim()),
                Span::styled(url, theme::cyan().add_modifier(Modifier::UNDERLINED)),
                Span::styled(" to get a key", theme::dim()),
            ])),
            rows[r],
        );
        r += 1;
    }
    r += 1; // blank spacer row before the input box

    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::TEAL));
    let input_inner = input_block.inner(rows[r]);
    f.render_widget(input_block, rows[r]);
    let key_line = if input.is_empty() {
        Line::from(Span::styled("paste or type your key…", placeholder_style()))
    } else {
        Line::from(Span::styled(mask_secret(input), theme::green()))
    };
    f.render_widget(Paragraph::new(key_line), input_inner);

    input_inner
}

/// A pending tool-permission ask, as a centered modal with the *actual*
/// diff/content preview (up to what `zeus-fs` computed — ~40 changed
/// lines) instead of a single clipped line squeezed into the pinned input
/// bar — a high-stakes "can I do this?" moment deserves to actually be
/// legible, not a truncated one-liner you have to trust blind.
fn render_approval_modal(f: &mut Frame, area: Rect, pending: &ApprovalRequestMsg) {
    let req = &pending.request;
    let preview = req.preview.as_deref().unwrap_or("");
    let width = area.width.saturating_sub(8).clamp(50, 120);
    let preview_w = width.saturating_sub(4) as usize;
    let preview_lines: Vec<Line> = if super::highlight::looks_like_diff(preview) {
        super::highlight::diff_lines(preview, placeholder_style())
            .into_iter()
            .map(Line::from)
            .collect()
    } else if !preview.is_empty() {
        preview
            .lines()
            .flat_map(|l| wrap_text(l, preview_w))
            .map(|l| Line::from(Span::styled(l, placeholder_style())))
            .collect()
    } else {
        Vec::new()
    };

    let max_preview_h = area.height.saturating_sub(10).max(3) as usize;
    let preview_h = preview_lines.len().min(max_preview_h);
    let truncated = preview_lines.len() > preview_h;
    let height = (5 + preview_h + usize::from(truncated))
        .clamp(6, area.height.saturating_sub(4).max(6) as usize) as u16;
    let popup = centered_rect(width, height, area);

    f.render_widget(Clear, popup);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::GOLD))
        .style(Style::default().bg(theme::PANEL))
        .title(Line::from(vec![Span::styled(
            " permission needed ",
            theme::gold().add_modifier(Modifier::BOLD),
        )]))
        .title_bottom(
            Line::from(Span::styled(
                " y approve · s approve for session · n/esc deny ",
                theme::faint(),
            ))
            .alignment(Alignment::Center),
        );
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Min(0)]).split(inner);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("Allow {}?", req.description),
            theme::text().add_modifier(Modifier::BOLD),
        ))),
        rows[0],
    );

    let mut shown: Vec<Line> = preview_lines.into_iter().take(preview_h).collect();
    if truncated {
        shown.push(Line::from(Span::styled(
            "… preview truncated, scroll not yet supported here",
            theme::faint(),
        )));
    }
    f.render_widget(Paragraph::new(shown).wrap(Wrap { trim: false }), rows[2]);
}

/// Greedy word-wrap into lines no wider than `width` columns.
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let width = width.max(10);
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let candidate_len = if current.is_empty() { word.len() } else { current.len() + 1 + word.len() };
        if candidate_len > width && !current.is_empty() {
            lines.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn textwrap_len(text: &str, width: usize) -> usize {
    wrap_text(text, width).len()
}

/// Word-wraps a line of *styled* spans to `width` columns, preserving each
/// character's original style — the span-aware equivalent of `wrap_text`.
/// Assistant bubbles carry markdown/syntax-highlighted spans that
/// `wrap_text` can't touch (it only understands plain strings), so a long
/// reply or a wide highlighted line needs this instead to stay inside its
/// box rather than overflowing past the right border.
fn wrap_spans(spans: &[Span<'static>], width: usize) -> Vec<Vec<Span<'static>>> {
    let width = width.max(10);
    let chars: Vec<(char, Style)> = spans
        .iter()
        .flat_map(|s| {
            let style = s.style;
            s.content.chars().map(move |c| (c, style)).collect::<Vec<_>>()
        })
        .collect();

    // Split into words on plain spaces, same convention `wrap_text` uses
    // (consecutive/leading/trailing whitespace collapses to single spaces
    // between words — acceptable for prose; code fences are typically
    // short enough per-line not to need wrapping in practice, and when
    // they do, this still keeps them on-screen instead of overflowing).
    let mut words: Vec<&[(char, Style)]> = Vec::new();
    let mut start = 0;
    for (i, (c, _)) in chars.iter().enumerate() {
        if *c == ' ' {
            if i > start {
                words.push(&chars[start..i]);
            }
            start = i + 1;
        }
    }
    if start < chars.len() {
        words.push(&chars[start..]);
    }

    let mut lines: Vec<Vec<(char, Style)>> = Vec::new();
    let mut current: Vec<(char, Style)> = Vec::new();
    for word in words {
        if current.len() > 0 && current.len() + 1 + word.len() > width {
            lines.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            // Match the joining space's style to the word before it, so a
            // highlighted/background-colored run (a code span, a chip
            // pill) doesn't show a stray default-styled gap.
            let space_style = current.last().map(|(_, s)| *s).unwrap_or_default();
            current.push((' ', space_style));
        }
        if word.len() > width {
            // A single "word" longer than the whole width (a long
            // identifier/URL/string) — hard-break it instead of
            // overflowing anyway.
            for chunk in word.chunks(width) {
                if !current.is_empty() {
                    lines.push(std::mem::take(&mut current));
                }
                current.extend_from_slice(chunk);
            }
        } else {
            current.extend_from_slice(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(Vec::new());
    }

    // Re-encode each line's (char, style) run back into merged spans.
    lines
        .into_iter()
        .map(|line| {
            let mut out: Vec<Span<'static>> = Vec::new();
            let mut buf = String::new();
            let mut buf_style = Style::default();
            for (c, style) in line {
                if buf.is_empty() {
                    buf_style = style;
                } else if style != buf_style {
                    out.push(Span::styled(std::mem::take(&mut buf), buf_style));
                    buf_style = style;
                }
                buf.push(c);
            }
            if !buf.is_empty() {
                out.push(Span::styled(buf, buf_style));
            }
            out
        })
        .collect()
}

/// Cap on the model/provider picker popups' height — without this a big
/// catalog (an aggregator provider alone can list 50+ models) sizes the
/// popup to fit every row at once and the modal swallows nearly the whole
/// screen. Ratatui's stateful `List` already scrolls to keep the selected
/// row in view, so capping the height just means the list scrolls instead
/// of the popup growing without bound — same navigation, smaller footprint.
const PICKER_MAX_H: u16 = 20;

/// A centered modal listing the current provider's available models —
/// arrow keys or a mouse click/scroll to navigate, Enter or a click to
/// select, Esc to close without changing anything. Modeled after opencode's
/// own "Select model" popup.
fn render_model_picker(
    f: &mut Frame,
    area: Rect,
    current_provider: &str,
    current_model: &str,
    entries: &[PickerEntry],
    selected: usize,
    search: &str,
    favorites: &[(String, String)],
) -> Rect {
    let width = area.width.saturating_sub(6).clamp(36, 78);
    let filtered = model_picker_filtered(entries, search);
    let height = (filtered.len() as u16 + 6)
        .min(PICKER_MAX_H)
        .clamp(9, area.height.saturating_sub(4).max(9));
    let popup = centered_rect(width, height, area);

    f.render_widget(Clear, popup);
    let corner = Line::from(vec![
        Span::styled(format!("{} ", filtered.len()), theme::faint()),
        Span::styled("esc ", theme::faint()),
    ])
    .alignment(Alignment::Right);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style())
        .style(Style::default().bg(theme::PANEL))
        .title(Line::from(vec![Span::styled(
            " select model ",
            theme::green().add_modifier(Modifier::BOLD),
        )]))
        .title(corner)
        .title_bottom(
            Line::from(Span::styled(
                " ↑/↓ navigate · enter select · ctrl+f favorite · ctrl+a connect provider · esc dismiss ",
                theme::faint(),
            ))
            .alignment(Alignment::Center),
        );
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Min(0)]).split(inner);
    let search_line = if search.is_empty() {
        Line::from(vec![
            Span::styled("▸ ", theme::violet()),
            Span::styled("Search providers or models…", placeholder_style()),
        ])
    } else {
        Line::from(vec![
            Span::styled("▸ ", theme::violet()),
            Span::styled(search.to_string(), theme::text()),
        ])
    };
    f.render_widget(Paragraph::new(search_line), rows[0]);

    let list_area = rows[2];
    let inner_w = list_area.width as usize;

    let items: Vec<ListItem> = filtered
        .iter()
        .map(|entry| match entry {
            PickerEntry::Header(name) => ListItem::new(Line::from(Span::styled(
                name.to_uppercase(),
                theme::green().add_modifier(Modifier::BOLD),
            ))),
            PickerEntry::SubHeader(family) => ListItem::new(Line::from(vec![
                Span::styled("  ", theme::faint()),
                Span::styled(family.clone(), theme::dim().add_modifier(Modifier::ITALIC)),
            ])),
            PickerEntry::Model { provider, model } => {
                let is_current = model.id == current_model && provider == current_provider;
                let is_fav = favorites.iter().any(|(p, m)| p == provider && m == &model.id);
                let marker = if is_current { "● " } else { "○ " };
                let marker_style = if is_current {
                    theme::green().add_modifier(Modifier::BOLD)
                } else {
                    theme::faint()
                };
                let star = if is_fav { "★ " } else { "" };
                let provider_suffix = format!(" {provider}");
                let free = is_free_model(&model.id);
                // Context window + price, when known — the same data
                // already fetched for `ModelInfo`/used for the sidebar's
                // cost estimate, just not previously shown in the picker
                // itself (you'd otherwise have to select a model blind to
                // find out its size or cost tier).
                let mut meta_parts: Vec<String> = Vec::new();
                if let Some(window) = model.context_window {
                    meta_parts.push(format_token_count(window));
                }
                if free {
                    meta_parts.push("Free".to_string());
                } else if let Some((prompt_rate, completion_rate)) = cost_per_million_tokens(provider, &model.id) {
                    meta_parts.push(format!("${}/${}", fmt_price(prompt_rate), fmt_price(completion_rate)));
                }
                let tag = meta_parts.join("  ·  ");
                let left_w = char_count(marker) + char_count(star) + char_count(&model.name) + char_count(&provider_suffix);
                let pad_w = inner_w.saturating_sub(left_w + char_count(&tag)).max(1);
                let mut spans = vec![
                    Span::styled(marker, marker_style),
                    Span::styled(star, theme::gold()),
                    Span::styled(model.name.clone(), theme::text().add_modifier(Modifier::BOLD)),
                    Span::styled(provider_suffix, theme::dim()),
                ];
                if !tag.is_empty() {
                    spans.push(Span::raw(" ".repeat(pad_w)));
                    let tag_style = if free { theme::green().add_modifier(Modifier::BOLD) } else { theme::faint() };
                    spans.push(Span::styled(tag, tag_style));
                }
                ListItem::new(Line::from(spans))
            }
        })
        .collect();
    let list = List::new(items).highlight_style(
        Style::default().bg(theme::PANEL2).fg(theme::accent()).add_modifier(Modifier::BOLD),
    );
    let mut list_state = ListState::default();
    list_state.select(Some(selected.min(filtered.len().saturating_sub(1))));
    f.render_stateful_widget(list, list_area, &mut list_state);

    list_area
}

/// Walks backward from `idx` to see whether the entry at `idx` sits inside a
/// vendor-family `SubHeader` group (an aggregator provider's catalog spanning
/// more than one recognizable vendor) as opposed to a flat per-provider model
/// list. Stops at the nearest `SubHeader` (grouped) or `Header`/`Provider`
/// row (not grouped) — a `Model` row just keeps scanning past its siblings.
fn family_context_for(entries: &[ProviderEntry], idx: usize) -> Option<&str> {
    let mut i = idx;
    while i > 0 {
        i -= 1;
        match &entries[i] {
            ProviderEntry::SubHeader(family) => return Some(family.as_str()),
            ProviderEntry::Header(_) | ProviderEntry::Provider { .. } => return None,
            ProviderEntry::Model { .. } => {}
        }
    }
    None
}

/// The `/provider` popup: providers grouped into Local / Free / Paid headers
/// with a status dot (green = ready, amber = needs a key) and a hint that
/// selecting a key-less provider opens the paste prompt.
fn render_provider_picker(
    f: &mut Frame,
    area: Rect,
    config: &Config,
    current_provider: &str,
    current_model: &str,
    entries: &[ProviderEntry],
    selected: usize,
) -> Rect {
    let width = area.width.saturating_sub(6).clamp(36, 76);
    let height = (entries.len() as u16 + 4)
        .min(PICKER_MAX_H)
        .clamp(8, area.height.saturating_sub(4).max(8));
    let popup = centered_rect(width, height, area);

    // A model grouped under a vendor `SubHeader` (an aggregator's catalog,
    // e.g. OpenCode Zen serving Anthropic/DeepSeek/Gemini models) is reached
    // through the aggregator's own key, not a separate one — so if that
    // aggregator is already configured, picking one of its models needs no
    // extra key and the footer says so instead of implying one might be
    // asked for.
    let selectable_model_count = entries
        .iter()
        .filter(|e| matches!(e, ProviderEntry::Model { .. }))
        .count();
    let aggregator_note = match entries.get(selected) {
        Some(ProviderEntry::Model { provider, .. })
            if family_context_for(entries, selected).is_some()
                && provider_status_ok(config, provider) =>
        {
            Some(provider.clone())
        }
        _ => None,
    };

    f.render_widget(Clear, popup);
    let corner = Line::from(vec![
        Span::styled(format!("{selectable_model_count} "), theme::faint()),
        Span::styled("esc ", theme::faint()),
    ])
    .alignment(Alignment::Right);
    let footer = match &aggregator_note {
        Some(provider) => format!(
            " ↑/↓ navigate · enter select · billed via your {provider} key, no separate key needed · esc dismiss "
        ),
        None => " ↑/↓ navigate · enter select (or paste key) · esc dismiss ".to_string(),
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style())
        .style(Style::default().bg(theme::PANEL))
        .title(Line::from(vec![
            Span::styled(" select provider ", theme::green().add_modifier(Modifier::BOLD)),
        ]))
        .title(corner)
        .title_bottom(
            Line::from(Span::styled(footer, theme::faint())).alignment(Alignment::Center),
        );
    let inner = block.inner(popup);
    f.render_widget(block, popup);

    let items: Vec<ListItem> = entries
        .iter()
        .map(|entry| match entry {
            ProviderEntry::Header(name) => ListItem::new(Line::from(Span::styled(
                format!(" {} ", name.to_uppercase()),
                theme::green().add_modifier(Modifier::BOLD),
            ))),
            ProviderEntry::SubHeader(family) => ListItem::new(Line::from(vec![
                Span::styled("   ", theme::faint()),
                Span::styled(family.clone(), theme::dim().add_modifier(Modifier::ITALIC)),
            ])),
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
        Style::default().bg(theme::PANEL2).fg(theme::accent()).add_modifier(Modifier::BOLD),
    );
    let mut list_state = ListState::default();
    list_state.select(Some(selected.min(entries.len().saturating_sub(1))));
    f.render_stateful_widget(list, inner, &mut list_state);

    inner
}

/// The top bar is a single flex row (logo, mode pills, spacer, provider
/// button all inline) plus its `border-bottom` hairline — 2 rows total,
/// matching the HTML's `.topbar` exactly rather than stacking modes below
/// the logo on their own row.
const TOPBAR_H: u16 = 2;
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

/// A clean, static left-to-right 3-stop color gradient across `text`'s
/// characters (near-white → gold-soft → gold at the 0/52/100% marks) — the
/// empty-state's "ZEUS" wordmark, matching the HTML's smooth text-fill
/// gradient instead of a blocky animated banner. Bold throughout, since a
/// wordmark this prominent reads better solid than thin.
fn gradient_wordmark(text: &str) -> Vec<Span<'static>> {
    let len = text.chars().count().max(1) as f32;
    text.chars()
        .enumerate()
        .map(|(i, c)| {
            let t = i as f32 / (len - 1.0).max(1.0);
            let color = if t < 0.52 {
                lerp_color(theme::WORDMARK_START, theme::WORDMARK_MID, t / 0.52)
            } else {
                lerp_color(theme::WORDMARK_MID, theme::WORDMARK_END, (t - 0.52) / 0.48)
            };
            Span::styled(c.to_string(), Style::default().fg(color).add_modifier(Modifier::BOLD))
        })
        .collect()
}

/// Top bar — the HTML `.topbar`: a single flex row with the logo, the mode
/// segmented-pill control, a flexible spacer, and the provider button (with
/// status dot) all inline, followed by the `border-bottom` hairline.
/// Returns the provider button's rect for mouse hit testing.
fn render_topbar(f: &mut Frame, area: Rect, state: &AppState, config: &Config) -> Option<Rect> {
    let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).split(area);

    // The wordmark carries the same rainbow/pulse sweep as the splash logo;
    // the ⚡ bolt stays violet, like the HTML's drop-shadowed SVG. Settling
    // `reduced_motion = true` in settings.toml swaps it for plain static
    // text instead — a redraw every tick isn't welcome on every link
    // (slow SSH), and some people just don't want a pulsing CLI.
    let mut logo_spans = vec![Span::styled("⚡ ", theme::violet().add_modifier(Modifier::BOLD))];
    if theme::reduced_motion() {
        logo_spans.push(Span::styled("ZEUS", theme::text().add_modifier(Modifier::BOLD)));
    } else {
        let t = state.started.elapsed().as_millis();
        logo_spans.extend(super::decor::animated_wordmark("ZEUS", t));
    }
    logo_spans.push(Span::styled("  v1.0", theme::faint()));
    let logo = Line::from(logo_spans);

    let pills = mode_pills_line(state.agent_mode);

    // Bracketed like the mode pills to hint at the HTML's bordered pill
    // button, since a single topbar row leaves no room for an actual border.
    // Filled vs. hollow, not just green-vs-gold, matching the same ●/◌
    // convention the provider picker/dropdown already use — a bare color
    // dot is the one status signal in the topbar with no shape backup for
    // colorblind users to fall back on.
    let ready_dot = if provider_status_ok(config, &state.provider) { "● " } else { "◌ " };
    let provider_line = Line::from(vec![
        Span::styled("[ ", theme::faint()),
        Span::styled(ready_dot, provider_status_style(config, &state.provider)),
        Span::styled(state.provider.clone(), theme::dim()),
        Span::styled("  ", theme::faint()),
        Span::styled(state.model.clone(), theme::text().add_modifier(Modifier::BOLD)),
        Span::styled("  ▾", theme::faint()),
        Span::styled(" ]", theme::faint()),
    ]);

    // Which specialist is driving the current Auto-mode plan step /
    // `/workflow` phase — previously only visible as a scrolling info line
    // in the transcript, easy to lose once a long run scrolls past it.
    let persona_line = state.active_persona.as_ref().map(|p| {
        Line::from(vec![
            Span::styled(" ▸ ", theme::faint()),
            Span::styled(p.clone(), theme::gold().add_modifier(Modifier::ITALIC)),
        ])
    });
    let persona_w = persona_line.as_ref().map(|l| l.width() as u16).unwrap_or(0);

    // logo | gap | mode pills | persona chip | flexible spacer | provider
    // button — the HTML's `display:flex; gap:18px` row with
    // `.topbar-spacer{flex:1}`.
    let cols = Layout::horizontal([
        Constraint::Length(logo.width() as u16),
        Constraint::Length(3),
        Constraint::Length(pills.width() as u16),
        Constraint::Length(persona_w),
        Constraint::Min(0),
        Constraint::Length(provider_line.width() as u16 + 2),
    ])
    .split(rows[0]);
    f.render_widget(Paragraph::new(logo), cols[0]);
    f.render_widget(Paragraph::new(pills), cols[2]);
    if let Some(persona_line) = persona_line {
        f.render_widget(Paragraph::new(persona_line), cols[3]);
    }
    f.render_widget(Paragraph::new(provider_line), cols[5]);
    let provider_btn = cols[5];

    // Hairline separator — the HTML `border-bottom` of the topbar.
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "─".repeat(area.width as usize),
            Style::default().fg(theme::BORDER),
        ))),
        rows[1],
    );

    Some(provider_btn)
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
    Line::from(spans)
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
            spans.push(Span::styled("█", Style::default().fg(lerp_color(theme::accent(), accent, t))));
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
        let mut spans = vec![
            Span::styled("[", theme::faint()),
            Span::styled(box_mark, box_style),
            Span::styled("]", theme::faint()),
            Span::raw(" "),
            Span::styled(item.text.clone(), label_style),
        ];
        if let Some(d) = item.duration {
            spans.push(Span::styled(format!("  {}", fmt_duration(d)), theme::faint()));
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
        Style::default().fg(theme::DIM).add_modifier(Modifier::BOLD),
    ))];
    let max_rows = (area.height as usize).saturating_sub(1);
    for path in state.files_touched.iter().rev().take(max_rows) {
        lines.push(Line::from(Span::styled(format!("· {path}"), theme::dim())));
    }
    f.render_widget(Paragraph::new(lines), area);
}

/// Short human-readable duration for a completed tool call/plan step —
/// "480ms" under a second, "2.3s" at or above.
fn fmt_duration(d: std::time::Duration) -> String {
    if d.as_secs() >= 1 {
        format!("{:.1}s", d.as_secs_f32())
    } else {
        format!("{}ms", d.as_millis())
    }
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

/// Abbreviates a token count the way the reference UI does ("42.1k"
/// instead of "42123") — anything under 1000 prints as-is.
fn format_token_count(n: u32) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

/// Best-effort public $/1M-token list prices for a handful of common
/// models, `(prompt, completion)`. Deliberately small and provider/model
/// prefix-matched rather than exhaustive: a wrong guessed price is worse
/// than no price at all, so anything not recognized here falls back to
/// "no estimate" (see `estimate_cost`) instead of a fabricated number.
/// Rates drift over time — treat this as approximate, not a bill.
fn cost_per_million_tokens(provider: &str, model_id: &str) -> Option<(f64, f64)> {
    let id = model_id.to_ascii_lowercase();
    Some(match provider {
        "anthropic" if id.contains("haiku") => (0.80, 4.00),
        "anthropic" if id.contains("opus") => (15.00, 75.00),
        "anthropic" => (3.00, 15.00), // sonnet family default
        "openai" if id.contains("gpt-4o-mini") || id.contains("gpt-5-nano") => (0.15, 0.60),
        "openai" if id.contains("mini") || id.contains("nano") => (0.15, 0.60),
        "openai" => (2.50, 10.00),
        "deepseek" => (0.27, 1.10),
        "gemini" if id.contains("flash") => (0.075, 0.30),
        "gemini" => (1.25, 5.00),
        _ => return None,
    })
}

/// Formats a $/1M-token rate compactly for the model picker — whole
/// numbers print bare ("3"), fractional ones keep two decimals ("0.15").
fn fmt_price(v: f64) -> String {
    if (v - v.round()).abs() < 0.001 {
        format!("{v:.0}")
    } else {
        format!("{v:.2}")
    }
}

/// `None` means "we don't have pricing data for this provider/model" —
/// callers should render that as "no estimate available", never as $0.00
/// (which would misleadingly imply the usage was actually free).
fn estimate_cost(provider: &str, model_id: &str, usage: &TokenUsage) -> Option<f64> {
    let (prompt_rate, completion_rate) = cost_per_million_tokens(provider, model_id)?;
    Some(
        (usage.prompt_tokens as f64 / 1_000_000.0) * prompt_rate
            + (usage.completion_tokens as f64 / 1_000_000.0) * completion_rate,
    )
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
            ProviderEntry::Header(_) | ProviderEntry::SubHeader(_) => q.is_empty(),
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

/// Filter the `/model` picker's entries by `state.model_picker_search`
/// (matches on model name/id or provider name) — same convention as
/// `drop_filtered`: headers only survive with an empty query, since a
/// flat match list needs no group labels.
fn model_picker_filtered(entries: &[PickerEntry], search: &str) -> Vec<PickerEntry> {
    let q = search.to_lowercase();
    entries
        .iter()
        .filter(|e| match e {
            PickerEntry::Header(_) | PickerEntry::SubHeader(_) => q.is_empty(),
            PickerEntry::Model { provider, model } => {
                q.is_empty()
                    || model.name.to_lowercase().contains(&q)
                    || model.id.to_lowercase().contains(&q)
                    || provider.to_lowercase().contains(&q)
            }
        })
        .cloned()
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
        .border_style(Style::default().fg(theme::accent_dim()))
        .style(Style::default().bg(theme::ELEVATED));
    f.render_widget(block.clone(), rect);
    let inner = block.inner(rect);

    if let Some(k) = keying {
        let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1)])
            .split(inner);
        let prompt = Line::from(vec![
            Span::styled("key: API key for ", theme::gold().add_modifier(Modifier::BOLD)),
            Span::styled(k.provider.clone(), theme::text().add_modifier(Modifier::BOLD)),
        ]);
        f.render_widget(Paragraph::new(prompt), rows[0]);
        let key_line = if k.key.is_empty() {
            Line::from(vec![Span::styled(
                "  paste the key…",
                placeholder_style(),
            )])
        } else {
            Line::from(vec![
                Span::styled("  ", theme::faint()),
                Span::styled(mask_secret(&k.key), theme::green()),
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
        if filtered.is_empty() && state.fetching_providers {
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(
                        format!("  {} fetching providers…", spinner_glyph(state)),
                        theme::dim(),
                    ),
                ])),
                rows[1],
            );
            return Some(rect);
        }
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
                ProviderEntry::SubHeader(family) => ListItem::new(Line::from(vec![
                    Span::styled("   ", theme::faint()),
                    Span::styled(family.clone(), theme::dim().add_modifier(Modifier::ITALIC)),
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
                .fg(theme::accent())
                .add_modifier(Modifier::BOLD),
        );
        let mut list_state = ListState::default();
        let selectable = filtered.len().saturating_sub(1);
        let sel = (0..=selectable)
            .rev()
            .find(|&i| {
                !matches!(filtered.get(i), Some(ProviderEntry::Header(_) | ProviderEntry::SubHeader(_)))
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

/// Example prompts shown as clickable chips — verbatim from
/// `zeus-empty-state.html`'s `#chips` row.
const EXAMPLE_CHIPS: [&str; 3] = [
    "Scaffold a new API",
    "Explain this codebase",
    "Write tests for a file",
];

/// Centers a `width`-wide, `height`-tall rect horizontally in `area` at a
/// specific `y` row (unlike `centered_rect`, which also centers vertically).
fn centered_row(area: Rect, y: u16, height: u16, width: u16) -> Rect {
    let width = width.min(area.width);
    Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y,
        width,
        height: height.min(area.height.saturating_sub(y.saturating_sub(area.y))),
    }
}

/// Paints an opaque `--ink` rect so the network canvas underneath doesn't
/// show through the empty-state's text blocks — the terminal stand-in for
/// the HTML's soft radial `.veil` (hard-edged here, since ratatui can't
/// alpha-blend text over canvas cells).
fn opaque(f: &mut Frame, area: Rect) {
    // `Block`'s style only recolors existing cells — it doesn't blank their
    // glyphs, so stale content underneath would otherwise show through
    // tinted rather than covered. `Clear` actually resets the cells first.
    f.render_widget(Clear, area);
    f.render_widget(Block::default().style(Style::default().bg(theme::INK)), area);
}

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
    f.render_widget(Block::default().style(Style::default().bg(theme::INK)), area);

    // ---- Centered content stack ----
    const STATUS_H: u16 = 1;
    const BANNER_H: u16 = 1;
    const QUESTION_H: u16 = 1;
    const COMPOSER_H: u16 = 3;
    const CHIPS_H: u16 = 1;
    const HINT_H: u16 = 1;
    const GAP: u16 = 1;
    let total_h = STATUS_H + GAP + BANNER_H + GAP + QUESTION_H + GAP + COMPOSER_H + GAP + CHIPS_H + GAP + HINT_H;
    let mut y = area.y + (area.height.saturating_sub(total_h)) / 2;

    // Status eyebrow: pulsing dot (mirrors the CSS `ping` keyframe) +
    // readiness text. Reflects whether the *current* provider actually has
    // a key/is reachable — a first-run session with nothing configured yet
    // used to always claim "READY" here regardless, so the example chips
    // below invited a task that would only fail with an error after Enter,
    // instead of pointing straight at `/model`/`/provider` up front.
    let ready = provider_status_ok(config, &state.provider);
    let pulse = 0.5 + 0.5 * (t_ms * 0.0024).sin();
    let dot_color = if ready { theme::TEAL } else { theme::GOLD };
    let dot_style = if pulse > 0.6 {
        Style::default().fg(dot_color).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(dot_color)
    };
    let label = if ready {
        "  Z E U S   R E A D Y"
    } else {
        "  N O   P R O V I D E R  ·  / M O D E L   T O   C O N N E C T"
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
        Paragraph::new(Line::from(Span::styled(question, theme::muted()))).alignment(Alignment::Center),
        question_area,
    );
    y += QUESTION_H + GAP;

    // Composer: rounded box, teal `›` glyph, live input, a send glyph that
    // lights up once there's something to send.
    let composer_w = (area.width.saturating_sub(8)).clamp(24, 64);
    let composer_area = centered_row(area, y, COMPOSER_H, composer_w);
    opaque(f, composer_area);
    let ready = !state.input.trim().is_empty();
    let composer_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::TEAL))
        .style(Style::default().bg(theme::INK));
    let composer_inner = composer_block.inner(composer_area);
    f.render_widget(composer_block, composer_area);
    let cols = Layout::horizontal([Constraint::Length(2), Constraint::Min(0), Constraint::Length(3)]).split(composer_inner);
    f.render_widget(Paragraph::new(Span::styled("›", theme::teal().add_modifier(Modifier::BOLD))), cols[0]);
    let input_line = if state.input.is_empty() {
        Line::from(Span::styled("Describe a task, or paste a file path…", theme::empty_faint()))
    } else {
        Line::from(Span::styled(state.input.clone(), theme::text()))
    };
    f.render_widget(Paragraph::new(input_line), cols[1]);
    let send_style = if ready {
        // The lit-up half of the HTML's teal→cyan send-button gradient.
        Style::default().fg(theme::EMPTY_CYAN).add_modifier(Modifier::BOLD)
    } else {
        theme::empty_faint()
    };
    f.render_widget(Paragraph::new(Span::styled("➜", send_style)).alignment(Alignment::Right), cols[2]);
    if matches!(state.mode, Mode::Chat) && !state.busy {
        let cursor_col = char_count(&state.input.chars().take(state.cursor).collect::<String>()) as u16;
        f.set_cursor_position((cols[1].x + cursor_col, cols[1].y));
    }
    y += COMPOSER_H + GAP;

    // Slash-command palette, when typing "/…", floats just below the composer.
    let matches = state.command_matches();
    let menu_h = menu_height(&matches);
    state.command_menu_area = if menu_h > 0 {
        let menu_area = centered_row(area, y, menu_h, composer_w);
        Some(render_menu(f, menu_area, &matches, state.command_selected, theme::TEAL))
    } else {
        None
    };

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
        areas.push(Rect { x: col, y, width: w, height: 1 });
        col += w;
        if i < EXAMPLE_CHIPS.len() - 1 {
            spans.push(Span::styled(gap, theme::faint()));
            col += gap.chars().count() as u16;
        }
    }
    f.render_widget(Paragraph::new(Line::from(spans)).alignment(Alignment::Center), chips_area);
    state.chip_areas = areas;
    y += CHIPS_H + GAP;

    // Hint row.
    let hint = "Enter to start  ·  Esc to clear";
    let hint_area = centered_row(area, y, HINT_H, hint.chars().count() as u16 + 2);
    opaque(f, hint_area);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(hint, theme::empty_faint()))).alignment(Alignment::Center),
        hint_area,
    );
}

fn render(f: &mut Frame, state: &mut AppState, config: &Config) {
    let area = f.area();

    if state.showing_empty_state() {
        render_empty_state(f, area, state, config);
        return;
    }
    state.chip_areas.clear();

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
        let divider = vec![Line::from(Span::styled("│", Style::default().fg(theme::BORDER))); main[2].height as usize];
        f.render_widget(Paragraph::new(divider), main[2]);
        let list_area = render_side(f, main[3], state);
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

    let provider_picker_area =
        if let Mode::ProviderPicker { entries, selected } = &state.mode {
            Some(render_provider_picker(
                f,
                area,
                config,
                &state.provider,
                &state.model,
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
        let cursor_col = char_count(&state.input.chars().take(state.cursor).collect::<String>()) as u16;
        f.set_cursor_position((input_rect.x + cursor_col.min(input_rect.width.saturating_sub(1)), input_rect.y));
    }

    if let Mode::Approval(pending) = &state.mode {
        render_approval_modal(f, area, pending);
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
        let result = agent.review_turn(on_event, approver).await;
        (agent, result)
    });
    *turn_handle = Some(handle);
}

/// `/suggest`: read-only next-feature recommendations grounded in the repo.
fn start_suggest_turn(
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
        let result = agent.suggest_turn(on_event, approver).await;
        (agent, result)
    });
    *turn_handle = Some(handle);
}

/// `/workflow <name> <goal>`: run a declarative multi-specialist pipeline.
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

    // ---- Session-resume picker ----
    if state.session_picker.is_some() {
        match key.code {
            KeyCode::Esc => state.session_picker = None,
            KeyCode::Up => {
                if let Some(picker) = state.session_picker.as_mut() {
                    picker.selected = picker.selected.saturating_sub(1);
                }
            }
            KeyCode::Down => {
                if let Some(picker) = state.session_picker.as_mut() {
                    if picker.selected + 1 < picker.entries.len() {
                        picker.selected += 1;
                    }
                }
            }
            KeyCode::Enter => {
                if let Some(picker) = state.session_picker.take() {
                    if let Some(entry) = picker.entries.get(picker.selected) {
                        resume_session(entry.id.clone(), config, agent_slot, state).await;
                    }
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
    // ctrl+f opens it — only in plain chat, so it doesn't steal the
    // model picker's own ctrl+f (toggle favorite) or the dropdown's.
    if matches!(state.mode, Mode::Chat)
        && state.dropdown.is_none()
        && key.code == KeyCode::Char('f')
        && key.modifiers.contains(KeyModifiers::CONTROL)
    {
        state.search = Some(SearchState { query: String::new(), matches: Vec::new(), current: 0 });
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
        // ctrl+a jumps straight to the provider picker ("connect a
        // provider") — checked before the plain Char branch below since
        // both match on `KeyCode::Char`.
        if key.code == KeyCode::Char('a') && key.modifiers.contains(KeyModifiers::CONTROL) {
            open_provider_picker(state, config, ui_tx);
            return Ok(());
        }
        if key.code == KeyCode::Char('f') && key.modifiers.contains(KeyModifiers::CONTROL) {
            let Mode::ModelPicker { entries, selected } = &state.mode else { unreachable!() };
            let filtered = model_picker_filtered(entries, &state.model_picker_search);
            if let Some(PickerEntry::Model { provider, model }) = filtered.get(*selected) {
                let (provider, model_id) = (provider.clone(), model.id.clone());
                state.toggle_favorite_model(&provider, &model_id, config);
            }
            return Ok(());
        }
        match key.code {
            KeyCode::Up => {
                let Mode::ModelPicker { entries, selected } = &mut state.mode else { unreachable!() };
                let filtered = model_picker_filtered(entries, &state.model_picker_search);
                *selected = picker_move(&filtered, *selected, -1);
            }
            KeyCode::Down => {
                let Mode::ModelPicker { entries, selected } = &mut state.mode else { unreachable!() };
                let filtered = model_picker_filtered(entries, &state.model_picker_search);
                *selected = picker_move(&filtered, *selected, 1);
            }
            KeyCode::Enter => {
                let Mode::ModelPicker { entries, selected } = &state.mode else { unreachable!() };
                let filtered = model_picker_filtered(entries, &state.model_picker_search);
                let chosen = match filtered.get(*selected) {
                    Some(PickerEntry::Model { provider, model }) => {
                        Some((provider.clone(), model.id.clone()))
                    }
                    // A header row isn't a choice — leave the picker open
                    // rather than silently closing it on a stray Enter.
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
                let Mode::ModelPicker { entries, selected } = &mut state.mode else { unreachable!() };
                let filtered = model_picker_filtered(entries, &state.model_picker_search);
                *selected = first_selectable_picker(&filtered);
            }
            KeyCode::Char(c) => {
                state.model_picker_search.push(c);
                let Mode::ModelPicker { entries, selected } = &mut state.mode else { unreachable!() };
                let filtered = model_picker_filtered(entries, &state.model_picker_search);
                *selected = first_selectable_picker(&filtered);
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
                        apply_model_choice_or_key_entry(provider, model_id, config, agent_slot, state);
                    }
                    // A header row isn't a choice — leave the picker open
                    // rather than silently closing it on a stray Enter.
                    Some(ProviderEntry::Header(_)) | Some(ProviderEntry::SubHeader(_)) | None => {}
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

    // A keybinding for the same thing `/copy` does — reaching for a
    // slash command just to copy the last reply is a lot of typing for
    // something you'd want mid-conversation without breaking flow.
    if key.code == KeyCode::Char('y') && key.modifiers.contains(KeyModifiers::CONTROL) {
        copy_last_response(state);
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
                        let agent =
                            build_agent_repl_with(config, Some(state.provider.clone()), Some(state.model.clone()))
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
                        let agent =
                            build_agent_repl_with(config, Some(state.provider.clone()), Some(state.model.clone()))
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
                                Ok(out) if out.stdout.trim().is_empty() => state.push_info("(no changes)"),
                                Ok(out) => state.push_info(out.stdout),
                                Err(e) => state.push_error(format!("diff failed: {e}")),
                            }
                        }
                    }
                    "undo" => {
                        if let Some(agent) = agent_slot.as_ref() {
                            let ws = agent.workspace();
                            let turn_id = ws.files.turn_id.clone();
                            let snaps = ws.files.checkpoints.load_snapshots(&turn_id).unwrap_or_default();
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
                                    let (entries, selected) =
                                        build_model_picker_entries(&groups, &provider, &model, &recent, &favorites);
                                    let _ = tx.send(UiEvent::ModelPickerReady(entries, selected, groups));
                                });
                            }
                        } else if let Some(agent) = agent_slot.as_mut() {
                            agent.set_model(arg.to_string());
                            state.model = arg.to_string();
                            state.push_info(format!("switched to model: {arg}"));
                        }
                    }
                    "provider" => {
                        handle_provider_tui(arg, config, agent_slot, state, ui_tx).await;
                    }
                    "session" => state.push_info(format!("session={}", state.session_id)),
                    "sessions" => {
                        let store = SessionStore::new(config.global.sessions.clone());
                        match store.summaries() {
                            Ok(entries) if entries.is_empty() => state.push_info(
                                "no saved sessions yet — send a message to create one",
                            ),
                            Ok(entries) => {
                                state.session_picker = Some(SessionPickerState { entries, selected: 0 });
                            }
                            Err(e) => state.push_error(format!("couldn't list sessions: {e:#}")),
                        }
                    }
                    "understand" => {
                        if arg.is_empty() {
                            state.push_error(
                                "usage: /understand <topic> — e.g. /understand authentication".to_string(),
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
                        start_suggest_turn(state, agent_slot, turn_handle, cancel_tx, ui_tx, yes);
                    }
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
                                "usage: /bg <goal> — run an orchestrated plan in the background".to_string(),
                            );
                        } else if matches!(rest, "list" | "output" | "stop") {
                            state.push_info(
                                "manage background tasks with the `zeus bg` subcommand: zeus bg list · zeus bg output <id> · zeus bg stop <id>".to_string(),
                            );
                        } else {
                            let (goal, workflow) = match rest.rsplit_once("@@workflow:") {
                                Some((g, name)) => (g.trim(), Some(name.trim())),
                                None => (rest, None),
                            };
                            match crate::spawn_bg_orchestrate(config, goal, workflow) {
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
                    "copy" => copy_last_response(state),
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

            // Catch the common "never set up a key" case before it burns a
            // turn on a network call that's guaranteed to fail with a wall
            // of red API-error text — nudge straight to the picker instead.
            if !provider_status_ok(config, &state.provider) {
                state.push_error(format!(
                    "'{}' isn't connected yet — opening the provider picker to set it up",
                    state.provider
                ));
                open_provider_picker(state, config, ui_tx);
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
            let step = state.transcript_area.map(|a| a.height.saturating_sub(2)).unwrap_or(10).max(1);
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
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
) {
    if let MouseEventKind::Down(MouseButton::Left) = ev.kind {
        let col = ev.column;
        let row = ev.row;

        // Empty-state example chips: click one to fill the composer.
        if let Some(idx) = state.chip_areas.iter().position(|a| {
            col >= a.x && col < a.x + a.width && row >= a.y && row < a.y + a.height
        }) {
            if let Some(label) = EXAMPLE_CHIPS.get(idx) {
                state.input = label.to_string();
                state.cursor = char_count(&state.input);
            }
            return;
        }

        // Slash-command palette: click a row to accept it, same as
        // pressing Enter/Tab on the highlighted entry — fills the input
        // ready for arguments rather than sending immediately.
        if let Some(area) = state.command_menu_area {
            if col >= area.x && col < area.x + area.width && row >= area.y && row < area.y + area.height {
                let idx = (row - area.y) as usize;
                let name = state.command_matches().get(idx).map(|(n, _)| n.to_string());
                if let Some(name) = name {
                    state.input = format!("/{name} ");
                    state.cursor = char_count(&state.input);
                    state.command_selected = 0;
                }
                return;
            }
        }

        // Session-resume picker: click a row to resume it immediately —
        // same click-to-choose interaction as the model picker.
        if let Some(area) = state.session_picker_area {
            if col >= area.x && col < area.x + area.width && row >= area.y && row < area.y + area.height {
                let idx = (row - area.y) as usize;
                let session_id = state
                    .session_picker
                    .as_ref()
                    .and_then(|p| p.entries.get(idx))
                    .map(|e| e.id.clone());
                if let Some(id) = session_id {
                    state.session_picker = None;
                    resume_session(id, config, agent_slot, state).await;
                }
                return;
            }
        }

        // Transcript: click any message to copy its full text to the
        // clipboard — previously `/copy`/ctrl+y could only ever grab the
        // single most recent reply, with no way to reach back further.
        if matches!(state.mode, Mode::Chat) {
            if let Some(area) = state.transcript_area {
                if col >= area.x && col < area.x + area.width && row >= area.y && row < area.y + area.height {
                    let clicked_row = (row - area.y) + state.transcript_applied_scroll;
                    let hit = state
                        .transcript_block_rows
                        .iter()
                        .position(|&(start, end)| clicked_row >= start && clicked_row < end);
                    if let Some(idx) = hit {
                        if let Some(block) = state.transcript.get(idx) {
                            // A folded tool result expands on its first
                            // click (revealing what `MAX_TOOL_LINES`
                            // hid) — copying it can wait for a second
                            // click once there's something worth copying
                            // to see, rather than silently doing both
                            // and burying the "it expanded" feedback.
                            if block.is_foldable() {
                                block.toggle_expanded();
                                return;
                            }
                            let text = block.plain_text();
                            match super::clipboard::copy(&text) {
                                Ok(()) => state.push_info(format!(
                                    "copied {} char(s) to clipboard",
                                    text.chars().count()
                                )),
                                Err(e) => state.push_error(format!("copy failed: {e}")),
                            }
                        }
                        return;
                    }
                }
            }
        }

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

        // Provider button: a cached model scan builds the dropdown
        // instantly; otherwise it opens empty (showing a "fetching…" row)
        // and probes providers in the background — probing every
        // configured provider can take several seconds, and awaiting it
        // inline here would freeze the whole render loop for that long.
        if in_provider {
            if let Some(groups) = state.model_cache.clone() {
                let entries = build_dropdown_entries(config, &state.provider, &state.model, &groups);
                let selected = dropdown_next_selectable(&entries, 0);
                state.dropdown = Some(DropdownState { entries, selected, area: None, keying: None });
                state.drop_search.clear();
                return;
            }
            state.dropdown = Some(DropdownState {
                entries: Vec::new(),
                selected: 0,
                area: None,
                keying: None,
            });
            state.drop_search.clear();
            state.fetching_providers = true;
            let cfg = config.clone();
            let provider = state.provider.clone();
            let model = state.model.clone();
            let tx = ui_tx.clone();
            tokio::spawn(async move {
                let groups = list_models_by_provider(&cfg).await;
                let entries = build_dropdown_entries(&cfg, &provider, &model, &groups);
                let _ = tx.send(UiEvent::DropdownReady(entries, groups));
            });
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

    // Slash-command palette: scroll to move the highlight, without needing
    // to be over the palette itself — it's the only thing visible to
    // scroll while it's open.
    if !state.busy {
        let menu_len = state.command_matches().len();
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
                        let Mode::ModelPicker { entries, .. } = &state.mode else { unreachable!() };
                        let filtered = model_picker_filtered(entries, &state.model_picker_search);
                        if let Some(PickerEntry::Model { provider, model }) = filtered.get(row) {
                            let (provider, model_id) = (provider.clone(), model.id.clone());
                            apply_model_choice_or_key_entry(provider, model_id, config, agent_slot, state);
                        }
                    }
                }
                MouseEventKind::ScrollUp => {
                    let Mode::ModelPicker { entries, selected } = &mut state.mode else { unreachable!() };
                    let filtered = model_picker_filtered(entries, &state.model_picker_search);
                    *selected = picker_move(&filtered, *selected, -1);
                }
                MouseEventKind::ScrollDown => {
                    let Mode::ModelPicker { entries, selected } = &mut state.mode else { unreachable!() };
                    let filtered = model_picker_filtered(entries, &state.model_picker_search);
                    *selected = picker_move(&filtered, *selected, 1);
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
                                apply_model_choice_or_key_entry(
                                    provider.clone(),
                                    model.id.clone(),
                                    config,
                                    agent_slot,
                                    state,
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
    let mut state = AppState::new(&agent, known_commands, dir, config.project_root.is_some(), config);
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
                        } else if let Some(k) = state.dropdown.as_mut().and_then(|d| d.keying.as_mut()) {
                            // The inline dropdown key-entry field has its own
                            // buffer (`KeyingState::key`), separate from
                            // `state.input` — pasting an API key here is the
                            // common case (keys are long random tokens), so
                            // this needs the same handling as the full-screen
                            // `Mode::KeyEntry` modal above, just targeting a
                            // different buffer.
                            k.key.push_str(&text);
                        } else if state.dropdown.is_none() && matches!(state.mode, Mode::Chat) {
                            // Plain composer paste (a message, or slash-command
                            // text — both live in `state.input`). Pickers and
                            // the dropdown search use their own buffers and
                            // don't want raw text dumped into `state.input`.
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
                        handle_mouse(mouse, &mut state, &mut agent_slot, config, &ui_tx).await;
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
                    UiEvent::Approval(req) => state.mode = Mode::Approval(req),
                    UiEvent::DropdownReady(entries, groups) => {
                        state.fetching_providers = false;
                        state.model_cache = Some(groups);
                        // Discarded if the user already closed the dropdown
                        // before the probe finished.
                        if let Some(dd) = state.dropdown.as_mut() {
                            dd.selected = dropdown_next_selectable(&entries, 0);
                            dd.entries = entries;
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
                    UiEvent::ProviderPickerReady(entries, selected, groups) => {
                        state.fetching_providers = false;
                        state.model_cache = Some(groups);
                        if matches!(state.mode, Mode::Chat) {
                            if entries.is_empty() {
                                state.push_error("no providers configured — see config.toml / providers.toml");
                            } else {
                                state.mode = Mode::ProviderPicker { entries, selected };
                            }
                        }
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
                            }
                            Err(e) => state.push_error(format!("turn failed: {e:#}")),
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
    if state.busy || !matches!(state.mode, Mode::Chat | Mode::KeyEntry { .. }) {
        terminal.hide_cursor().ok();
    }
}

pub async fn run(config: &Config, agent: Agent, yes: bool) -> Result<()> {
    theme::init_runtime(
        config.settings.accent_color.as_deref(),
        config.settings.reduced_motion,
        config.settings.notify_on_completion,
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
