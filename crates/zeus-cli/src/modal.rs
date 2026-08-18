//! The modal/popup family for the TUI: the shared chrome (dimmed backdrop,
//! rounded border, opaque fill, selection bar) plus the six popups that use
//! it — the model/provider/session pickers, the API-key entry screen, the
//! permission-approval modal, and the two-pane diff viewer. Extracted out of
//! `tui.rs` so the popup layer is testable and can evolve independently of
//! the render loop / key-handling monolith.

use super::pickers::{is_free_model, model_picker_filtered, PickerEntry, ProviderEntry};
use super::theme;
use super::tui_text::{
    centered_rect, char_count, cost_per_million_tokens, fmt_price, format_token_count, mask_secret,
    textwrap_len, wrap_text,
};
use super::FileEntry;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap,
};
use ratatui::Frame;
use zeus_agent::SessionSummary;
use zeus_fs::{ApprovalDecision, PermissionRequest};

/// A pending permission ask, bridged out of the spawned turn task. The reply
/// is a oneshot so a modal ask can be answered at most once; dropping the
/// sender (turn cancelled, app closing) resolves the waiting approver as a
/// deny.
pub(crate) struct ApprovalRequestMsg {
    pub(crate) request: PermissionRequest,
    pub(crate) reply: tokio::sync::oneshot::Sender<ApprovalDecision>,
}

/// `SearchState`: Up/Down/click to pick, Enter/click-to-apply to resume,
/// Esc to close.
pub(crate) struct SessionPickerState {
    pub(crate) entries: Vec<SessionSummary>,
    pub(crate) selected: usize,
    /// Type-to-filter query — same convention as the model picker's own
    /// search, matched against a session's id and last-message preview.
    pub(crate) search: String,
}

/// Sessions matching `picker.search` (id or last-user-message substring,
/// case-insensitive) — everything when the query is empty.
pub(crate) fn session_picker_filtered(picker: &SessionPickerState) -> Vec<&SessionSummary> {
    let q = picker.search.to_lowercase();
    picker
        .entries
        .iter()
        .filter(|s| {
            q.is_empty()
                || s.id.to_lowercase().contains(&q)
                || s.last_user.to_lowercase().contains(&q)
        })
        .collect()
}

/// Any provider not listed here (custom entries in `providers.toml`) still
/// gets a sane generic prompt.
pub(crate) fn provider_blurb(name: &str) -> (&'static str, Option<&'static str>) {
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

/// Warm-to-cool "AI" gradient (orange → pink → purple → blue → cyan),
/// matching the sweep across Antigravity CLI's logo — `t` in `0.0..=1.0`.
pub(crate) fn placeholder_style() -> Style {
    theme::faint()
}

/// Violet border used by the centered picker popups.
pub(crate) fn border_style() -> Style {
    theme::violet()
}

/// Solid full-width highlight bar for the selected list row (warm
/// `modal_selected_bg` fill, near-black text, bold) — shared by all four
/// picker/menu lists so selection looks identical everywhere.
fn selected_style() -> Style {
    Style::default()
        .bg(theme::modal_selected_bg())
        .fg(theme::void())
        .add_modifier(Modifier::BOLD)
}

/// Shared chrome for every centered popup: clear the area, then draw a
/// rounded bordered box with the opaque modal fill and optional title /
/// right-aligned corner / centered bottom hint. Returns the inner rect the
/// caller lays content out in. `border` defaults to the violet picker
/// chrome; a popup can override it (the approval modal keeps its gold).
struct Popup {
    area: Rect,
    title: Option<Line<'static>>,
    corner: Option<Line<'static>>,
    bottom: Option<Line<'static>>,
    border: Style,
}

impl Popup {
    fn new(area: Rect) -> Self {
        Self {
            area,
            title: None,
            corner: None,
            bottom: None,
            border: border_style(),
        }
    }

    fn title(mut self, t: Line<'static>) -> Self {
        self.title = Some(t);
        self
    }

    fn corner(mut self, c: Line<'static>) -> Self {
        self.corner = Some(c);
        self
    }

    fn bottom(mut self, b: Line<'static>) -> Self {
        self.bottom = Some(b);
        self
    }

    fn border(mut self, s: Style) -> Self {
        self.border = s;
        self
    }

    fn render(self, f: &mut Frame) -> Rect {
        f.render_widget(Clear, self.area);
        let mut block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(self.border)
            .style(Style::default().bg(theme::modal_bg()));
        if let Some(t) = self.title {
            block = block.title(t);
        }
        if let Some(c) = self.corner {
            block = block.title(c);
        }
        if let Some(b) = self.bottom {
            block = block.title_bottom(b);
        }
        let inner = block.inner(self.area);
        f.render_widget(block, self.area);
        inner
    }
}

/// Slash-command dropdown — command names in violet bold, descriptions dim,
/// highlight bar on the selected row (mirrors the HTML `.palette-item`
/// rows: `.pc` command labels colored `--mode-color`, `.pd` descriptions
/// dim, `.active` row tinted with the mode color).
pub(crate) fn render_menu(
    f: &mut Frame,
    area: Rect,
    matches: &[(String, String)],
    selected: usize,
    accent: Color,
) -> Rect {
    // Backdrop-dim, opaque fill, violet chrome, solid selection bar — the
    // same modal treatment as the picker family, since this is a real modal
    // (opened by typing "/" or ctrl+p) rather than a passive dropdown, even
    // though it was originally styled as the latter.
    dim_backdrop(f, f.area());
    let inner = Popup::new(area).render(f);

    let name_width = matches
        .iter()
        .map(|(n, _)| n.len())
        .max()
        .unwrap_or(0)
        .max(8);
    let items: Vec<ListItem> = matches
        .iter()
        .map(|(name, desc)| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("/{name:<name_width$}"),
                    Style::default().fg(accent).add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled(desc.to_string(), placeholder_style()),
            ]))
        })
        .collect();
    let list = List::new(items).highlight_style(selected_style());
    let mut list_state = ListState::default();
    list_state.select(Some(selected.min(matches.len().saturating_sub(1))));
    f.render_stateful_widget(list, inner, &mut list_state);
    inner
}

pub(crate) fn render_session_picker(
    f: &mut Frame,
    area: Rect,
    picker: &SessionPickerState,
) -> Rect {
    let width = area.width.saturating_sub(6).clamp(40, 96);
    let filtered = session_picker_filtered(picker);
    let height = (filtered.len() as u16 + 5)
        .min(PICKER_MAX_H)
        .clamp(9, area.height.saturating_sub(4).max(9));
    let popup = centered_rect(width, height, area);

    let inner = Popup::new(popup)
        .title(Line::from(vec![Span::styled(
            " Resume session ",
            theme::text().add_modifier(Modifier::BOLD),
        )]))
        .corner(Line::from(vec![
            Span::styled(format!("{} ", filtered.len()), theme::faint()),
            Span::styled("esc ", theme::dim()),
        ]))
        .bottom(
            Line::from(Span::styled(
                " ↑/↓ navigate · enter resume · esc dismiss ",
                theme::dim(),
            ))
            .alignment(Alignment::Center),
        )
        .render(f);

    let rows = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(inner);
    let search_line = if picker.search.is_empty() {
        Line::from(vec![
            Span::styled("▸ ", theme::violet()),
            Span::styled("Search sessions…", placeholder_style()),
        ])
    } else {
        Line::from(vec![
            Span::styled("▸ ", theme::violet()),
            Span::styled(picker.search.clone(), theme::text()),
        ])
    };
    f.render_widget(Paragraph::new(search_line), rows[0]);

    let list_area = rows[1];
    let inner_w = list_area.width as usize;
    let items: Vec<ListItem> = filtered
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
    let list = List::new(items).highlight_style(selected_style());
    let mut list_state = ListState::default();
    list_state.select(Some(picker.selected.min(filtered.len().saturating_sub(1))));
    f.render_stateful_widget(list, list_area, &mut list_state);

    list_area
}

/// The ctrl+o filesystem picker — a directory browser that inserts picked
/// files into the composer as quoted paths. Directories render with a
/// trailing `/` and bold; Enter descends or inserts, ←/Backspace goes up,
/// ctrl+h toggles hidden files. Files already staged in `.agent/uploads/`
/// carry a `✓`; file sizes appear right-aligned.
pub(crate) fn render_file_picker(
    f: &mut Frame,
    area: Rect,
    cwd: &std::path::Path,
    entries: &[FileEntry],
    selected: usize,
    show_hidden: bool,
) -> Rect {
    let width = area.width.saturating_sub(6).clamp(48, 96);
    let height = (entries.len() as u16 + 6)
        .min(PICKER_MAX_H)
        .clamp(10, area.height.saturating_sub(4).max(10));
    let popup = centered_rect(width, height, area);

    let inner = Popup::new(popup)
        .title(Line::from(vec![
            Span::styled(" Open file ", theme::text().add_modifier(Modifier::BOLD)),
            Span::styled(cwd.display().to_string(), theme::faint()),
        ]))
        .corner(Line::from(vec![
            Span::styled(format!("{} entries ", entries.len()), theme::faint()),
            Span::styled("esc ", theme::dim()),
        ]))
        .bottom(
            Line::from(Span::styled(
                " ↑/↓ move · enter open dir / insert file · ← back · ctrl+h hidden · esc close ",
                theme::dim(),
            ))
            .alignment(Alignment::Center),
        )
        .render(f);

    let rows = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(inner);
    let head = Line::from(vec![
        Span::styled("▸ ", theme::violet()),
        Span::styled(
            format!(
                "picked files are inserted into the composer as quoted paths{}",
                if show_hidden {
                    "  ·  hidden shown"
                } else {
                    ""
                },
            ),
            theme::dim(),
        ),
    ]);
    f.render_widget(Paragraph::new(head), rows[0]);

    let list_area = rows[1];
    let name_width = (list_area.width as usize).saturating_sub(10).max(6);
    let items: Vec<ListItem> = entries
        .iter()
        .map(|e| {
            let marker = if e.staged { "✓" } else { " " };
            let marker_style = if e.staged {
                theme::green()
            } else {
                theme::dim()
            };
            let name = if e.is_dir {
                format!("{}/", e.name)
            } else {
                e.name.clone()
            };
            let truncated: String = name.chars().take(name_width).collect();
            let trunc_len = truncated.chars().count();
            let name_style = if e.is_dir {
                theme::text().add_modifier(Modifier::BOLD)
            } else if e.hidden {
                theme::faint()
            } else {
                theme::text()
            };
            let size_str = if e.is_dir {
                "dir".to_string()
            } else {
                crate::human_size(e.size)
            };
            ListItem::new(Line::from(vec![
                Span::styled(marker, marker_style),
                Span::styled(truncated, name_style),
                Span::styled(
                    " ".repeat(name_width.saturating_sub(trunc_len)),
                    theme::faint(),
                ),
                Span::styled(size_str, theme::faint()),
            ]))
        })
        .collect();
    let list = List::new(items).highlight_style(selected_style());
    let mut list_state = ListState::default();
    list_state.select(Some(selected.min(entries.len().saturating_sub(1))));
    f.render_stateful_widget(list, list_area, &mut list_state);

    list_area
}

pub(crate) fn dim_backdrop(f: &mut Frame, area: Rect) {
    fn dim(c: Color) -> Color {
        match c {
            Color::Rgb(r, g, b) => Color::Rgb(
                (r as f32 * 0.45) as u8,
                (g as f32 * 0.45) as u8,
                (b as f32 * 0.45) as u8,
            ),
            other => other,
        }
    }
    let buf = f.buffer_mut();
    for y in area.y..area.y.saturating_add(area.height) {
        for x in area.x..area.x.saturating_add(area.width) {
            if let Some(cell) = buf.cell_mut((x, y)) {
                cell.fg = dim(cell.fg);
                cell.bg = dim(cell.bg);
            }
        }
    }
}

/// The `Mode::KeyEntry` screen — a centered "API key" modal with a short
/// pitch for the provider, a link to get a key, a masked input field, and
/// an `enter submit` footer. Returns the input field's inner rect so the
/// caller can place the terminal cursor in it.
pub(crate) fn render_key_entry_modal(
    f: &mut Frame,
    area: Rect,
    provider: &str,
    input: &str,
) -> Rect {
    let (blurb, url) = provider_blurb(provider);
    let width = area.width.saturating_sub(10).clamp(40, 76);
    let blurb_lines = textwrap_len(blurb, width as usize - 4);
    // A pasted key is one long unbroken token — masked or not, it's the
    // same length as typed, so a long one (some provider tokens run past
    // 100 chars) needs the same wrap-and-grow treatment as any other input
    // box instead of running off the edge of a fixed single-line field.
    // Sized against the *real* input inner width (`width - 4`: 2 for the
    // modal's own border, 2 for the input box's) — the previous `width - 6`
    // guess wrapped keys at a narrower width than the box actually draws
    // with, so the modal was sized up to a line too tall for keys whose
    // length landed between the two widths.
    let key_display_len = if input.is_empty() {
        0
    } else {
        char_count(input)
    };
    let input_box_inner_w = (width as usize).saturating_sub(4).max(10);
    let input_text_h = wrap_text(&"x".repeat(key_display_len), input_box_inner_w)
        .len()
        .clamp(1, 5) as u16;
    // title, blank, blurb lines, blank, url line (if any), blank, input box (border×2 + text), blank, footer
    let height = (3 + blurb_lines + if url.is_some() { 2 } else { 0 } + 4 + input_text_h as usize)
        .clamp(9, area.height.saturating_sub(4).max(9) as usize) as u16;
    let popup = centered_rect(width, height, area);

    let inner = Popup::new(popup)
        .title(Line::from(vec![Span::styled(
            " API key ",
            theme::text().add_modifier(Modifier::BOLD),
        )]))
        .corner(Line::from(Span::styled(" esc ", theme::dim())))
        .bottom(
            Line::from(Span::styled(" enter submit ", theme::dim())).alignment(Alignment::Center),
        )
        .render(f);

    let mut rows: Vec<Constraint> = vec![Constraint::Length(1)];
    for _ in 0..blurb_lines {
        rows.push(Constraint::Length(1));
    }
    if url.is_some() {
        // `Min(0)` rather than `Length(1)` — a blank spacer, so on a short
        // terminal it's the first thing the layout solver sacrifices down
        // to zero height instead of squeezing the input box itself (which
        // stays `Length`-fixed below). On a tall enough terminal it simply
        // absorbs the little slack `height`'s own clamp already builds in.
        rows.push(Constraint::Min(0));
        rows.push(Constraint::Length(1));
    }
    rows.push(Constraint::Min(0));
    rows.push(Constraint::Length(input_text_h + 2));
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

    // Violet border (matching the outer modal's own chrome) and plain text
    // color — this used to be a one-off teal border with green typed text,
    // an unexplained divergence from every other input field in the app.
    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style());
    let input_inner = input_block.inner(rows[r]);
    f.render_widget(input_block, rows[r]);
    if input.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "paste or type your key…",
                placeholder_style(),
            ))),
            input_inner,
        );
    } else {
        f.render_widget(
            Paragraph::new(mask_secret(input))
                .style(theme::text())
                .wrap(Wrap { trim: false }),
            input_inner,
        );
    }

    input_inner
}

/// A pending tool-permission ask, as a centered modal with the *actual*
/// diff/content preview (up to what `zeus-fs` computed — ~40 changed
/// lines) instead of a single clipped line squeezed into the pinned input
/// bar — a high-stakes "can I do this?" moment deserves to actually be
/// legible, not a truncated one-liner you have to trust blind.
pub(crate) fn render_approval_modal(
    f: &mut Frame,
    area: Rect,
    pending: &ApprovalRequestMsg,
    scroll: usize,
) {
    let req = &pending.request;
    let preview = req.preview.as_deref().unwrap_or("");
    let width = area.width.saturating_sub(8).clamp(50, 120);
    let preview_w = width.saturating_sub(4) as usize;
    let preview_lines: Vec<Line> = if crate::highlight::looks_like_diff(preview) {
        crate::highlight::diff_lines(preview, placeholder_style(), preview_w)
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

    let inner = Popup::new(popup)
        // Gold border/title stays — a permission ask is deliberately
        // distinct from a neutral picker — but the fill and title casing
        // still match the rest of the modal family for consistency.
        .border(Style::default().fg(theme::GOLD))
        .title(Line::from(vec![Span::styled(
            " Permission needed ",
            theme::gold().add_modifier(Modifier::BOLD),
        )]))
        .bottom(
            Line::from(Span::styled(
                " y approve · s for session · n/esc deny · ↑/↓ scroll ",
                theme::dim(),
            ))
            .alignment(Alignment::Center),
        )
        .render(f);

    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .split(inner);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("Allow {}?", req.description),
            theme::text().add_modifier(Modifier::BOLD),
        ))),
        rows[0],
    );

    // Clamp the counter the key handler nudged to the real line count, then
    // show a scrolled window of the preview (↑/↓, pgup/pgdn) instead of only
    // ever showing the top of it.
    let max_scroll = preview_lines.len().saturating_sub(preview_h);
    let scroll = scroll.min(max_scroll);
    let end = scroll + preview_h;
    let mut shown: Vec<Line> = preview_lines[scroll..end].to_vec();
    if truncated {
        let remaining = preview_lines.len() - end;
        let hint = if remaining > 0 {
            format!("… {remaining} more line(s) below — ↑/↓ (pgup/pgdn) scroll")
        } else {
            "end of preview — ↑/↓ (pgup/pgdn) scroll".to_string()
        };
        shown.push(Line::from(Span::styled(hint, theme::faint())));
    }
    f.render_widget(Paragraph::new(shown).wrap(Wrap { trim: false }), rows[2]);
}

/// The `/diff` command's two-pane side-by-side view: a centered modal that
/// lays each diff row out as `old │ new` cells, with the `-`/`+` markers
/// stripped and a red/green tint telling removed from added. Header rows
/// (file metadata and `@@` hunks) span the full width in dim/teal. The
/// render clamps `scroll` to the real row count, so the key handler just
/// nudges it; esc returns to Chat (handled in the key path, not here).
pub(crate) fn render_diff_modal(
    f: &mut Frame,
    area: Rect,
    rows: &[crate::highlight::DiffRow],
    scroll: usize,
) {
    let width = area.width.saturating_sub(8).clamp(70, 180);
    let max_h = area.height.saturating_sub(6);
    // `max_h` is already the popup's full target height; the `clamp(10, _)`
    // floor is only reachable when the terminal is so short it would panic
    // (clamp requires min <= max), so just take the space available — same
    // result on any normal terminal, no crash at the 15-row boundary.
    let height = max_h.max(1) as usize;
    let popup = centered_rect(width, height as u16, area);

    let inner = Popup::new(popup)
        .border(Style::default().fg(theme::border()))
        .title(Line::from(vec![Span::styled(
            " Diff ",
            theme::teal().add_modifier(Modifier::BOLD),
        )]))
        .bottom(
            Line::from(Span::styled(
                " esc close · ↑/↓ (pgup/pgdn) scroll ",
                theme::dim(),
            ))
            .alignment(Alignment::Center),
        )
        .render(f);

    let col_w = (inner.width.saturating_sub(3) / 2).max(10) as usize;
    let max_scroll = rows.len().saturating_sub(inner.height as usize);
    let scroll = scroll.min(max_scroll);
    let end = (scroll + inner.height as usize).min(rows.len());
    let mut shown: Vec<Line> = Vec::new();
    for row in &rows[scroll..end] {
        let line = match row {
            crate::highlight::DiffRow::Header(h) => Line::from(Span::styled(
                h.to_string(),
                if h.starts_with("@@") {
                    theme::teal()
                } else {
                    theme::dim()
                },
            )),
            crate::highlight::DiffRow::Pair(old, new) => {
                // Removed/added rows carry a `-`/`+` glyph prefix on top of
                // their red/green tint, so the direction of change reads
                // even without color (Light theme, color-blind users,
                // or a monochrome terminal). Context rows stay bare.
                let old_txt = if old.is_some() && new.is_none() {
                    format!("- {}", old.as_deref().unwrap_or(""))
                } else {
                    old.as_deref().unwrap_or("").to_string()
                };
                let new_txt = if new.is_some() && old.is_none() {
                    format!("+ {}", new.as_deref().unwrap_or(""))
                } else {
                    new.as_deref().unwrap_or("").to_string()
                };
                let old_pad = old_txt.chars().count().min(col_w);
                let old_spans = if old.is_some() && new.is_some() {
                    vec![Span::styled(old_txt.to_string(), theme::text())]
                } else if old.is_some() {
                    vec![Span::styled(old_txt.to_string(), theme::red())]
                } else {
                    vec![Span::styled("", theme::dim())]
                };
                let new_spans = if new.is_some() && old.is_some() {
                    vec![Span::styled(new_txt.to_string(), theme::text())]
                } else if new.is_some() {
                    vec![Span::styled(new_txt.to_string(), theme::green())]
                } else {
                    vec![Span::styled("", theme::dim())]
                };
                let mut spans = old_spans;
                let pad = " ".repeat(col_w.saturating_sub(old_pad));
                spans.push(Span::styled(pad, theme::dim()));
                spans.push(Span::styled("│", theme::border_soft()));
                spans.push(Span::styled(" ", theme::dim()));
                spans.extend(new_spans);
                Line::from(spans)
            }
        };
        shown.push(line);
    }
    f.render_widget(Paragraph::new(shown), inner);
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
#[allow(clippy::too_many_arguments)] // single call-site UI modal; grouping has no reuse
pub(crate) fn render_model_picker(
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

    let corner = Line::from(vec![
        Span::styled(format!("{} ", filtered.len()), theme::faint()),
        Span::styled("esc ", theme::dim()),
    ])
    .alignment(Alignment::Right);
    let inner = Popup::new(popup)
        .title(Line::from(vec![Span::styled(
            " Select model ",
            theme::text().add_modifier(Modifier::BOLD),
        )]))
        .corner(corner)
        .bottom(
            Line::from(Span::styled(
                " ↑/↓ navigate · enter select · ctrl+f favorite · ctrl+a connect provider · esc dismiss ",
                theme::dim(),
            ))
            .alignment(Alignment::Center),
        )
        .render(f);

    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .split(inner);
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
                theme::violet().add_modifier(Modifier::BOLD),
            ))),
            PickerEntry::SubHeader(family) => ListItem::new(Line::from(vec![
                Span::styled("  ", theme::faint()),
                Span::styled(family.clone(), theme::dim().add_modifier(Modifier::ITALIC)),
            ])),
            PickerEntry::Model { provider, model } => {
                let is_current = model.id == current_model && provider == current_provider;
                let is_fav = favorites
                    .iter()
                    .any(|(p, m)| p == provider && m == &model.id);
                let marker = if is_current { "● " } else { "  " };
                let marker_style = theme::gold().add_modifier(Modifier::BOLD);
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
                } else if let Some((prompt_rate, completion_rate)) =
                    cost_per_million_tokens(provider, &model.id)
                {
                    meta_parts.push(format!(
                        "${}/${}",
                        fmt_price(prompt_rate),
                        fmt_price(completion_rate)
                    ));
                }
                let tag = meta_parts.join("  ·  ");
                let left_w = char_count(marker)
                    + char_count(star)
                    + char_count(&model.name)
                    + char_count(&provider_suffix);
                let pad_w = inner_w.saturating_sub(left_w + char_count(&tag)).max(1);
                let mut spans = vec![
                    Span::styled(marker, marker_style),
                    Span::styled(star, theme::gold()),
                    Span::styled(
                        model.name.clone(),
                        theme::text().add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(provider_suffix, theme::dim()),
                ];
                if !tag.is_empty() {
                    spans.push(Span::raw(" ".repeat(pad_w)));
                    let tag_style = if free {
                        theme::gold().add_modifier(Modifier::BOLD)
                    } else {
                        theme::faint()
                    };
                    spans.push(Span::styled(tag, tag_style));
                }
                ListItem::new(Line::from(spans))
            }
        })
        .collect();
    // A solid full-width warm highlight bar for the selected row, matching
    // the reference product's own selection style, instead of a subtle
    // panel-tint + colored-text highlight.
    let list = List::new(items).highlight_style(selected_style());
    let mut list_state = ListState::default();
    list_state.select(Some(selected.min(filtered.len().saturating_sub(1))));
    f.render_stateful_widget(list, list_area, &mut list_state);

    list_area
}

/// The `/provider` popup: providers grouped into Local / Free / Paid headers
/// with a status dot (green = ready, amber = needs a key) and a hint that
/// selecting a key-less provider opens the paste prompt.
pub(crate) fn render_provider_picker(
    f: &mut Frame,
    area: Rect,
    current_provider: &str,
    entries: &[ProviderEntry],
    selected: usize,
) -> Rect {
    let width = area.width.saturating_sub(6).clamp(36, 76);
    let height = (entries.len() as u16 + 4)
        .min(PICKER_MAX_H)
        .clamp(8, area.height.saturating_sub(4).max(8));
    let popup = centered_rect(width, height, area);

    let provider_count = entries
        .iter()
        .filter(|e| matches!(e, ProviderEntry::Provider { .. }))
        .count();

    let corner = Line::from(vec![
        Span::styled(format!("{provider_count} "), theme::faint()),
        Span::styled("esc ", theme::dim()),
    ])
    .alignment(Alignment::Right);
    // Picking a provider that still needs a key doesn't dead-end here — it
    // opens the key-paste prompt, then automatically opens the model picker
    // for that provider once the key is saved (see `persist_key_and_switch`).
    let footer = " ↑/↓ navigate · enter select (asks for a key if needed) · ctrl+k set/update key · esc dismiss ";
    let inner = Popup::new(popup)
        .title(Line::from(vec![Span::styled(
            " Select provider ",
            theme::text().add_modifier(Modifier::BOLD),
        )]))
        .corner(corner)
        .bottom(Line::from(Span::styled(footer, theme::dim())).alignment(Alignment::Center))
        .render(f);

    let items: Vec<ListItem> = entries
        .iter()
        .map(|entry| match entry {
            ProviderEntry::Header(name) => ListItem::new(Line::from(Span::styled(
                format!(" {} ", name.to_uppercase()),
                theme::violet().add_modifier(Modifier::BOLD),
            ))),
            ProviderEntry::Provider { name, kind, ready } => {
                let is_current = name == current_provider;
                // A connected provider gets a plain checkmark gutter,
                // matching the reference product's own treatment — no
                // separate "(current)"/"needs key" text; readiness alone
                // tells the story, and `apply_provider_picker_choice`
                // routes an unready pick straight to the key prompt anyway.
                let gutter = if *ready {
                    Span::styled("✓", theme::green().add_modifier(Modifier::BOLD))
                } else {
                    Span::styled(" ", theme::faint())
                };
                let name_style = if is_current {
                    theme::text().add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
                } else {
                    theme::text().add_modifier(Modifier::BOLD)
                };
                ListItem::new(Line::from(vec![
                    gutter,
                    Span::raw("  "),
                    Span::styled(name.clone(), name_style),
                    Span::raw("  "),
                    Span::styled(kind.clone(), theme::dim()),
                ]))
            }
        })
        .collect();
    let list = List::new(items).highlight_style(selected_style());
    let mut list_state = ListState::default();
    list_state.select(Some(selected.min(entries.len().saturating_sub(1))));
    f.render_stateful_widget(list, inner, &mut list_state);

    inner
}
