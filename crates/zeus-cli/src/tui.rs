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
    build_agent, describe_providers, expand_slash_command, known_slash_commands,
    list_models_by_provider, persist_default_provider, print_repl_help_lines,
};
use anyhow::{Context, Result};
use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
    MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
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
    fn prefix(&self) -> &'static str {
        match self {
            Role::User => "❯ ",
            Role::Assistant => "● ",
            Role::Tool => "◆ ",
            Role::ToolError => "✗ ",
            Role::Info => "· ",
            Role::Error => "✗ ",
        }
    }

    fn style(&self) -> Style {
        match self {
            Role::User => theme::amber(),
            Role::Assistant => theme::green(),
            Role::Tool => theme::blue(),
            Role::ToolError => theme::red(),
            Role::Info => theme::muted(),
            Role::Error => theme::red(),
        }
    }
}

struct Block_ {
    role: Role,
    text: String,
}

impl Block_ {
    fn to_lines(&self) -> Vec<Line<'static>> {
        let style = self.role.style();
        let prefix = self.role.prefix();
        let mut lines: Vec<Line<'static>> = self
            .text
            .lines()
            .enumerate()
            .map(|(i, l)| {
                let text = if i == 0 {
                    format!("{prefix}{l}")
                } else {
                    format!("  {l}")
                };
                Line::from(Span::styled(text, style))
            })
            .collect();
        if lines.is_empty() {
            lines.push(Line::from(Span::styled(prefix.to_string(), style)));
        }
        lines
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

/// The `/provider` slash command inside the TUI: list all configured
/// providers (with live key/local status), switch the active one, or set a
/// cloud key for the session. Mirrors the plain-REPL handler, but pushes
/// messages into the transcript instead of printing to stdout.
fn handle_provider_tui(
    arg: &str,
    config: &Config,
    agent_slot: &mut Option<Agent>,
    state: &mut AppState,
) {
    let parts: Vec<&str> = arg.split_whitespace().collect();
    match parts.as_slice() {
        [] => {
            state.push_info(format!(
                "current: {} / {}",
                state.provider, state.model
            ));
            state.push_info("configured providers:");
            for line in describe_providers(config) {
                state.push_info(line);
            }
            state.push_info("/provider <name> to switch · /provider key <name> <KEY> to set a key");
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
}

/// Project-root facts shown in the SuperCode-style "directory" header panel.
pub struct DirInfo {
    pub workspace: String,
    pub path: String,
    pub git_branch: Option<String>,
    pub file_count: usize,
}

impl AppState {
    fn new(agent: &Agent, known_commands: Vec<(String, String)>, dir: DirInfo) -> Self {
        Self {
            transcript: Vec::new(),
            current_reply: String::new(),
            input: String::new(),
            cursor: 0,
            busy: false,
            quit: false,
            mode: Mode::Chat,
            agent_mode: AgentMode::Build,
            model: agent.model().to_string(),
            provider: agent.provider_id().to_string(),
            session_id: agent.session_id().to_string(),
            known_commands,
            dir,
            command_selected: 0,
            model_picker_area: None,
        }
    }

    fn flush_current_reply(&mut self) {
        if !self.current_reply.is_empty() {
            let text = std::mem::take(&mut self.current_reply);
            self.transcript.push(Block_ { role: Role::Assistant, text });
        }
    }

    fn apply_agent_event(&mut self, ev: AgentEvent) {
        match ev {
            AgentEvent::TextDelta(t) => self.current_reply.push_str(&t),
            AgentEvent::ToolCallStarted { name, arguments, .. } => {
                self.flush_current_reply();
                self.transcript.push(Block_ {
                    role: Role::Tool,
                    text: format!("{name} {arguments}"),
                });
            }
            AgentEvent::ToolCallFinished { name, result, is_error, .. } => {
                self.flush_current_reply();
                let role = if is_error { Role::ToolError } else { Role::Tool };
                let marker = if is_error { "failed" } else { "done" };
                self.transcript.push(Block_ {
                    role,
                    text: format!("{name} ({marker})\n{result}"),
                });
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
                self.transcript.push(Block_ {
                    role: Role::Tool,
                    text: format!("review ({persona})\n{report}"),
                });
            }
            AgentEvent::PlanStepDone { step, summary } => {
                self.transcript.push(Block_ {
                    role: Role::Tool,
                    text: format!("step {} done · {}\n{}", step.id, step.description, summary),
                });
            }
            AgentEvent::OrchestrationDone { summary } => {
                self.flush_current_reply();
                self.transcript.push(Block_ { role: Role::Assistant, text: summary });
            }
        }
    }

    fn push_user(&mut self, text: String) {
        self.transcript.push(Block_ { role: Role::User, text });
    }

    fn push_info(&mut self, text: impl Into<String>) {
        self.transcript.push(Block_ { role: Role::Info, text: text.into() });
    }

    fn push_error(&mut self, text: impl Into<String>) {
        self.transcript.push(Block_ { role: Role::Error, text: text.into() });
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
fn ai_gradient_color(t: f32) -> Color {
    const STOPS: [(f32, (u8, u8, u8)); 5] = [
        (0.0, (255, 125, 40)),
        (0.25, (235, 70, 150)),
        (0.5, (175, 75, 225)),
        (0.75, (75, 125, 240)),
        (1.0, (55, 220, 225)),
    ];
    let t = t.clamp(0.0, 1.0);
    let mut lo = STOPS[0];
    let mut hi = STOPS[STOPS.len() - 1];
    for pair in STOPS.windows(2) {
        if t >= pair[0].0 && t <= pair[1].0 {
            lo = pair[0];
            hi = pair[1];
            break;
        }
    }
    let span = (hi.0 - lo.0).max(f32::EPSILON);
    let local_t = ((t - lo.0) / span).clamp(0.0, 1.0);
    let mix = |a: u8, b: u8| -> u8 { (a as f32 + (b as f32 - a as f32) * local_t).round() as u8 };
    Color::Rgb(mix(lo.1 .0, hi.1 .0), mix(lo.1 .1, hi.1 .1), mix(lo.1 .2, hi.1 .2))
}

fn zeus_logo_lines() -> Vec<Line<'static>> {
    const LETTERS: [[&str; 6]; 4] = [
        ["██████", ".....█", "....█.", "..█...", ".█....", "██████"], // Z
        ["██████", "█.....", "█████.", "█.....", "█.....", "██████"], // E
        ["█....█", "█....█", "█....█", "█....█", "█....█", "██████"], // U
        ["██████", "█.....", "██████", ".....█", ".....█", "██████"], // S
    ];
    // One combined row per letter-row, letters joined with a single-column
    // gap, so the gradient sweeps smoothly across the whole word instead of
    // jumping in four flat per-letter blocks.
    let combined_rows: Vec<String> = (0..6)
        .map(|row| {
            LETTERS
                .iter()
                .map(|letter| letter[row])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    let width = combined_rows[0].chars().count().max(1);

    combined_rows
        .iter()
        .map(|row| {
            let spans: Vec<Span<'static>> = row
                .chars()
                .enumerate()
                .map(|(col, ch)| {
                    let display = if ch == '.' { ' ' } else { ch };
                    let t = col as f32 / (width - 1).max(1) as f32;
                    Span::styled(
                        display.to_string(),
                        Style::default().fg(ai_gradient_color(t)).add_modifier(Modifier::BOLD),
                    )
                })
                .collect();
            Line::from(spans)
        })
        .collect()
}

fn placeholder_style() -> Style {
    theme::faint()
}

/// SuperCode-theme terminal palette (mirrors `supercode ui/*.html`).
mod theme {
    use ratatui::style::{Color, Style};

    /// Dark, near-black green-tinted page background (`#0a0c0a`).
    pub const BG: Color = Color::Rgb(0x0a, 0x0c, 0x0a);
    /// Slightly lighter panel background (`#14161a`).
    pub const PANEL: Color = Color::Rgb(0x14, 0x16, 0x1a);
    /// Bordered panel surface / slab (`#1c1f26`).
    pub const SURFACE: Color = Color::Rgb(0x1c, 0x1f, 0x26);
    /// Neon-green brand/accent (`#22e88f`).
    pub const GREEN: Color = Color::Rgb(0x22, 0xe8, 0x8f);
    /// Amber highlight / user (`#e8a83b`).
    pub const AMBER: Color = Color::Rgb(0xe8, 0xa8, 0x3b);
    /// Sky-blue path / link accent (`#7fb1e0`).
    pub const BLUE: Color = Color::Rgb(0x7f, 0xb1, 0xe0);
    /// Red for errors/destructive (`#e2564f`).
    pub const RED: Color = Color::Rgb(0xe2, 0x56, 0x4f);
    /// Primary readable text (`#d6d9de`).
    pub const TEXT: Color = Color::Rgb(0xd6, 0xd9, 0xde);
    /// Muted secondary text (`#8a8f98`).
    pub const MUTED: Color = Color::Rgb(0x8a, 0x8f, 0x98);
    /// Faint tertiary text (`#5c6068`).
    pub const FAINT: Color = Color::Rgb(0x5c, 0x60, 0x68);

    pub fn green() -> Style {
        Style::default().fg(GREEN)
    }
    pub fn amber() -> Style {
        Style::default().fg(AMBER)
    }
    pub fn blue() -> Style {
        Style::default().fg(BLUE)
    }
    pub fn red() -> Style {
        Style::default().fg(RED)
    }
    pub fn text() -> Style {
        Style::default().fg(TEXT)
    }
    pub fn muted() -> Style {
        Style::default().fg(MUTED)
    }
    pub fn faint() -> Style {
        Style::default().fg(FAINT)
    }
}

fn status_style() -> Style {
    theme::muted()
}

fn hint_key_style() -> Style {
    Style::default().fg(theme::BG).bg(theme::GREEN)
}

fn border_style() -> Style {
    theme::green()
}

fn menu_height(matches: &[(&str, &str)]) -> u16 {
    if matches.is_empty() {
        0
    } else {
        matches.len().min(8) as u16 + 2
    }
}

/// Slash-command dropdown — a name column (padded to align) plus a dim
/// description column, with a full-width highlight bar on the selected row.
/// Modeled after opencode's own `/`-menu.
fn render_menu(f: &mut Frame, area: Rect, matches: &[(&str, &str)], selected: usize) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::SURFACE));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let name_width = matches.iter().map(|(n, _)| n.len()).max().unwrap_or(0).max(8);
    let items: Vec<ListItem> = matches
        .iter()
        .map(|(name, desc)| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("/{name:<name_width$}"), theme::green().add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::styled(desc.to_string(), placeholder_style()),
            ]))
        })
        .collect();
    let list = List::new(items).highlight_style(
        Style::default().fg(theme::AMBER).add_modifier(Modifier::BOLD),
    );
    let mut list_state = ListState::default();
    list_state.select(Some(selected.min(matches.len().saturating_sub(1))));
    f.render_stateful_widget(list, inner, &mut list_state);
}

/// Renders the pinned input box — SuperCode-style: a `[chat] >` prompt, a
/// filled mode chip, and a model · provider · session status line below.
fn render_input_box(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style())
        .style(Style::default().bg(theme::PANEL));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).split(inner);

    if let Mode::Approval(pending) = &state.mode {
        let line = Line::from(vec![
            Span::styled("[chat] > ", theme::amber().add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("Allow {}?", pending.request.description),
                theme::text().add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("[y] approve · [n] deny · [s] session", theme::faint()),
        ]);
        f.render_widget(Paragraph::new(line), rows[0]);
        let preview = pending
            .request
            .preview
            .as_deref()
            .unwrap_or("")
            .lines()
            .next()
            .unwrap_or("");
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(preview.to_string(), placeholder_style()))),
            rows[1],
        );
        return;
    }

    let prompt_style = if state.busy { theme::faint() } else { theme::green() };
    let input_line = if state.busy {
        Line::from(vec![
            Span::styled("[chat] ", prompt_style),
            Span::styled("> ", prompt_style),
            Span::styled("zeus is working…", placeholder_style()),
        ])
    } else if state.input.is_empty() {
        Line::from(vec![
            Span::styled("[chat] ", theme::green()),
            Span::styled("> ", theme::green()),
            Span::styled("Ask anything... \"Fix broken tests\"", placeholder_style()),
        ])
    } else {
        Line::from(vec![
            Span::styled("[chat] ", theme::green()),
            Span::styled("> ", theme::green()),
            Span::raw(state.input.clone()),
        ])
    };
    f.render_widget(Paragraph::new(input_line), rows[0]);

    let mode_style = match state.agent_mode {
        AgentMode::Build => Style::default().fg(theme::BG).bg(theme::GREEN),
        AgentMode::Plan => Style::default().fg(theme::BG).bg(theme::AMBER),
        AgentMode::Auto => Style::default().fg(theme::BG).bg(theme::BLUE),
    };
    let status = Line::from(vec![
        Span::styled(format!(" {} ", state.agent_mode.label()), mode_style.add_modifier(Modifier::BOLD)),
        Span::styled(" · ", status_style()),
        Span::styled(state.model.clone(), theme::text()),
        Span::styled(" · ", status_style()),
        Span::styled(state.provider.clone(), theme::muted()),
        Span::styled(format!(" · session={}", state.session_id), theme::faint()),
    ]);
    f.render_widget(Paragraph::new(status), rows[1]);

    if !state.busy && matches!(state.mode, Mode::Chat) {
        // Cursor sits just past the `[chat] > ` prompt.
        let base = "[chat] > ".chars().count() as u16;
        let cursor_col = base + char_count(&state.input.chars().take(state.cursor).collect::<String>()) as u16;
        f.set_cursor_position((inner.x + cursor_col, inner.y));
    }
}

fn render_hints(f: &mut Frame, area: Rect) {
    let line = Line::from(vec![
        Span::styled(" enter ", hint_key_style()),
        Span::raw(" send   "),
        Span::styled(" tab ", hint_key_style()),
        Span::raw(" plan/build   "),
        Span::styled(" / ", hint_key_style()),
        Span::raw(" commands   "),
        Span::styled(" ctrl+c ", hint_key_style()),
        Span::raw(" cancel / quit"),
    ])
    .alignment(Alignment::Center);
    f.render_widget(Paragraph::new(line), area);
}

fn transcript_text(state: &AppState) -> Text<'static> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    for block in &state.transcript {
        lines.extend(block.to_lines());
        lines.push(Line::from(""));
    }
    if !state.current_reply.is_empty() {
        let streaming = Block_ {
            role: Role::Assistant,
            text: state.current_reply.clone(),
        };
        lines.extend(streaming.to_lines());
    } else if state.busy {
        lines.push(Line::from(Span::styled("● thinking…", placeholder_style())));
    }
    Text::from(lines)
}

fn render_splash(f: &mut Frame, area: Rect, state: &AppState, matches: &[(&str, &str)]) {
    let logo = zeus_logo_lines();
    let logo_h = logo.len() as u16;
    let dir_h: u16 = 5; // directory info panel
    let box_h: u16 = 4;
    let menu_h = menu_height(matches);
    let hint_h: u16 = 1;
    let gap1: u16 = 2;
    let gap2: u16 = 1;
    let gap3: u16 = 2;
    let total = dir_h + gap1 + logo_h + gap2 + menu_h + box_h + gap3 + hint_h;
    let top_pad = area.height.saturating_sub(total) / 2;

    let rows = Layout::vertical([
        Constraint::Length(top_pad),
        Constraint::Length(dir_h),
        Constraint::Length(gap1),
        Constraint::Length(logo_h),
        Constraint::Length(gap2),
        Constraint::Length(menu_h),
        Constraint::Length(box_h),
        Constraint::Length(gap3),
        Constraint::Length(hint_h),
        Constraint::Min(0),
    ])
    .split(area);

    let box_w = area.width.saturating_sub(4).min(76).max(30);
    let cols = Layout::horizontal([Constraint::Min(0), Constraint::Length(box_w), Constraint::Min(0)]);

    render_dir_panel(f, cols.split(rows[1])[1], state);
    f.render_widget(Paragraph::new(logo).alignment(Alignment::Center), rows[3]);

    if menu_h > 0 {
        render_menu(f, cols.split(rows[5])[1], matches, state.command_selected);
    }
    render_input_box(f, cols.split(rows[6])[1], state);
    render_hints(f, rows[8]);
}

/// SuperCode-style bordered "directory" panel: workspace, path, git branch
/// (if any), and file count — mirroring the `supercode_cli_chat_screen`
/// mockup's workspace/path/git/files box.
fn render_dir_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let d = &state.dir;
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::SURFACE))
        .title(Line::from(
            Span::styled(" directory ", theme::green()),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines = vec![
        Line::from(vec![
            Span::styled("workspace", theme::green().add_modifier(Modifier::BOLD)),
            Span::styled(" · ", theme::faint()),
            Span::styled(d.workspace.clone(), theme::text()),
        ]),
        Line::from(vec![
            Span::styled("path", theme::green().add_modifier(Modifier::BOLD)),
            Span::styled(" · ", theme::faint()),
            Span::styled(d.path.clone(), theme::blue()),
        ]),
        Line::from(vec![
            Span::styled("files", theme::green().add_modifier(Modifier::BOLD)),
            Span::styled(" · ", theme::faint()),
            Span::styled(d.file_count.to_string(), theme::text()),
        ]),
    ];
    if let Some(branch) = &d.git_branch {
        lines.push(Line::from(vec![
            Span::styled("git", theme::green().add_modifier(Modifier::BOLD)),
            Span::styled(" · ", theme::faint()),
            Span::styled(branch.clone(), theme::amber()),
            Span::styled(" ●", theme::green()),
        ]));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn render_chat(f: &mut Frame, area: Rect, state: &AppState, matches: &[(&str, &str)]) {
    let menu_h = menu_height(matches);
    let rows = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(menu_h),
        Constraint::Length(4),
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

    // The input box (and its dropdown menu) stay a fixed, centered width —
    // same as the splash view and the same idea as opencode's own chatbox —
    // rather than stretching edge-to-edge just because the transcript above
    // it uses the full terminal width.
    let box_w = area.width.saturating_sub(4).min(76).max(30);
    let boxed = Layout::horizontal([Constraint::Min(0), Constraint::Length(box_w), Constraint::Min(0)]);

    if menu_h > 0 {
        let menu_cols = boxed.split(rows[1]);
        render_menu(f, menu_cols[1], matches, state.command_selected);
    }
    let input_cols = boxed.split(rows[2]);
    render_input_box(f, input_cols[1], state);
    render_hints(f, rows[3]);
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
    let width = area.width.saturating_sub(6).min(70).max(30);
    let height = (entries.len() as u16 + 4).min(area.height.saturating_sub(4)).max(6);
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
                    Span::styled(model.id.clone(), theme::muted()),
                ]))
            }
        })
        .collect();
    let list = List::new(items).highlight_style(
        Style::default().bg(theme::SURFACE).fg(theme::GREEN).add_modifier(Modifier::BOLD),
    );
    let mut list_state = ListState::default();
    list_state.select(Some(selected.min(entries.len().saturating_sub(1))));
    f.render_stateful_widget(list, inner, &mut list_state);

    inner
}

fn render(f: &mut Frame, state: &mut AppState, config: &Config) {
    let area = f.area();

    // Frame the whole app with a SuperCode-style header bar on top and a
    // status bar along the bottom, like the HTML mockups' breadcrumb header
    // and the `[chat] · model · tab mode` footer.
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .split(area);
    render_header_bar(f, rows[0], state, config);

    let matches = state.command_matches();
    if state.transcript.is_empty() && state.current_reply.is_empty() {
        render_splash(f, rows[1], state, &matches);
    } else {
        render_chat(f, rows[1], state, &matches);
    }

    render_status_bar(f, rows[2], state);

    let picker_area = if let Mode::ModelPicker { entries, selected } = &state.mode {
        Some(render_model_picker(f, area, &state.model, entries, *selected))
    } else {
        None
    };
    state.model_picker_area = picker_area;
}

/// SuperCode-style breadcrumb header: `supercode · chat · <model>` on the
/// left with a green left-rule, status (`ready`/`busy · type to chat`) on
/// the right, separated by a horizontal rule.
fn render_header_bar(f: &mut Frame, area: Rect, state: &AppState, config: &Config) {
    let status_right = if state.busy {
        "busy · esc to interrupt"
    } else {
        "ready · type to chat"
    };
    let parts = Layout::horizontal([Constraint::Min(0), Constraint::Length(status_right.len() as u16 + 4)])
        .split(area);
    let left = Line::from(vec![
        Span::styled("zeus", theme::green().add_modifier(Modifier::BOLD)),
        Span::styled(" · ", theme::faint()),
        Span::styled("chat", theme::text()),
        Span::styled(" · ", theme::faint()),
        Span::styled(provider_dot(config, &state.provider), provider_status_style(config, &state.provider)),
        Span::styled(state.provider.clone(), theme::text().add_modifier(Modifier::BOLD)),
        Span::styled(" / ", theme::faint()),
        Span::styled(state.model.clone(), theme::text()),
        Span::styled(" · ", theme::faint()),
        Span::styled(format!("session={}", state.session_id), theme::muted()),
    ]);
    f.render_widget(Paragraph::new(left), parts[0]);

    let right = Line::from(vec![
        Span::styled(
            if state.busy { "busy" } else { "ready" },
            if state.busy { theme::amber() } else { theme::green() },
        ),
        Span::styled(" · ", theme::faint()),
        Span::styled(
            if state.busy { "esc to interrupt" } else { "type to chat" },
            theme::muted(),
        ),
    ]);
    f.render_widget(Paragraph::new(right.alignment(Alignment::Right)), parts[1]);

    let dash = Line::from(Span::styled(
        "─".repeat(area.width as usize),
        Style::default().fg(theme::SURFACE),
    ));
    f.render_widget(Paragraph::new(dash), Rect {
        x: area.x,
        y: area.y,
        width: area.width,
        height: 1,
    });
}

/// A leading status glyph for a provider in the header: green dot when it
/// actually has a key (cloud) or is a local kind, amber dot when a cloud
/// provider is missing its key.
fn provider_dot(config: &Config, provider: &str) -> &'static str {
    if provider_status_ok(config, provider) {
        "●"
    } else {
        "●"
    }
}

fn provider_status_ok(config: &Config, provider: &str) -> bool {
    let Some(cfg) = config.providers.get(provider) else {
        return false;
    };
    if matches!(cfg.kind.as_str(), "ollama" | "lmstudio" | "llamacpp") {
        true
    } else if cfg.headers.contains_key("Authorization") {
        true
    } else if let Some(var) = &cfg.api_key_env {
        std::env::var(var).map(|k| !k.is_empty()).unwrap_or(false)
    } else {
        true
    }
}

fn provider_status_style(config: &Config, provider: &str) -> Style {
    if provider_status_ok(config, provider) {
        theme::green()
    } else {
        theme::amber()
    }
}

/// SuperCode-style bottom status bar: mode chip + git branch + connected,
/// and a version on the right — the `[chat] · glm-5.2 · tab mode` footer.
fn render_status_bar(f: &mut Frame, area: Rect, state: &AppState) {
    let git = state
        .dir
        .git_branch
        .clone()
        .unwrap_or_else(|| "(no git repo)".to_string());
    let line = Line::from(vec![
        Span::styled(format!(" {} ", state.agent_mode.label()), {
            let bg = match state.agent_mode {
                AgentMode::Build => theme::GREEN,
                AgentMode::Plan => theme::AMBER,
                AgentMode::Auto => theme::BLUE,
            };
            Style::default().fg(theme::BG).bg(bg).add_modifier(Modifier::BOLD)
        }),
        Span::styled(" · ", theme::faint()),
        Span::styled(state.model.clone(), theme::muted()),
        Span::styled(" · ", theme::faint()),
        Span::styled("⚡ ", theme::green()),
        Span::styled(format!("{git} · {} files", state.dir.file_count), theme::muted()),
        Span::raw("  "),
        Span::styled("zeus", theme::green()),
    ]);
    f.render_widget(Paragraph::new(line), area);
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
                        let agent = build_agent(config, None, None, None).await?;
                        apply_agent_mode(&agent, state.agent_mode);
                        state.session_id = agent.session_id().to_string();
                        state.model = agent.model().to_string();
                        state.provider = agent.provider_id().to_string();
                        *agent_slot = Some(agent);
                        state.transcript.clear();
                        state.current_reply.clear();
                        state.push_info(format!("cleared — new session={}", state.session_id));
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
                        handle_provider_tui(arg, config, agent_slot, state);
                    }
                    "session" => state.push_info(format!("session={}", state.session_id)),
                    "agents" => {
                        if arg.to_ascii_lowercase() == "count" {
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
fn handle_mouse(ev: MouseEvent, state: &mut AppState, agent_slot: &mut Option<Agent>, config: &Config) {
    if !matches!(state.mode, Mode::ModelPicker { .. }) {
        return;
    }
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

async fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    config: &Config,
    agent: Agent,
    yes: bool,
) -> Result<()> {
    let known_commands = known_slash_commands(config);
    let dir = build_dir_info(config);
    let mut state = AppState::new(&agent, known_commands, dir);
    let mut agent_slot = Some(agent);
    let mut turn_handle: Option<TurnJoin> = None;
    let mut cancel_tx: Option<watch::Sender<bool>> = None;

    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<UiEvent>();
    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<Event>();
    std::thread::spawn(move || loop {
        match crossterm::event::read() {
            Ok(ev) => {
                if input_tx.send(ev).is_err() {
                    break;
                }
            }
            Err(_) => break,
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
                    Event::Mouse(mouse) => handle_mouse(mouse, &mut state, &mut agent_slot, config),
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
                        let agent = build_agent(config, None, None, None).await?;
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

/// Gather project-root facts for the SuperCode-style "directory" header
/// panel: workspace folder name, root path, current git branch (if any), and
/// a rough file count. Best-effort — any piece that fails degrades silently.
fn build_dir_info(config: &Config) -> DirInfo {
    let path = config
        .project_root
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")));
    let workspace = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());

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

    // Rough file count, capped so a huge tree doesn't stall startup.
    let file_count = walk_file_count(&path, 0, 50_000);

    DirInfo {
        workspace,
        path: path.display().to_string(),
        git_branch,
        file_count,
    }
}

/// Depth-limited recursive file count (skips hidden + vendor dirs). Stops
/// early once `cap` is reached so startup never crawls on a giant tree.
fn walk_file_count(dir: &std::path::Path, depth: usize, max: usize) -> usize {
    if depth > 6 || max == 0 {
        return 0;
    }
    let mut count = 0usize;
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if depth == 0 && matches!(name.as_str(), ".git" | "node_modules" | "target" | ".agent" | ".zeus")
        {
            continue;
        }
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_dir() {
            count += walk_file_count(&entry.path(), depth + 1, max.saturating_sub(count));
        } else if ft.is_file() {
            count += 1;
        }
        if count >= max {
            return count;
        }
    }
    count
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
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture).context("enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("init terminal")?;

    let result = run_app(&mut terminal, config, agent, yes).await;

    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture).ok();
    terminal.show_cursor().ok();

    result
}
