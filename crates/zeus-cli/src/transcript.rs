//! Transcript message rendering: role-marked, bubbled, or card lines for
//! the chat column. Extracted from `tui.rs` so the transcript row logic can
//! be tested and evolved without the ~5.5k-line TUI monolith.

use super::theme;
use super::tui_text::{wrap_preserving_newlines, wrap_spans};
use ratatui::layout::Alignment;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

pub(crate) enum Role {
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

pub(crate) struct Block_ {
    pub(crate) role: Role,
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

    /// Reading-width cap for assistant replies and tool-result cards on a
    /// wide/ultrawide terminal. Without it, prose and tool output stretch
    /// to the full chat-column width — hundreds of columns on an ultrawide
    /// monitor — both a real readability regression and an asymmetry with
    /// the user's own messages, which already cap at 84 (see `bubble_lines`).
    const MAX_CONTENT_W: usize = 100;

    pub(crate) fn new(role: Role, text: String) -> Self {
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
    pub(crate) fn is_foldable(&self) -> bool {
        if self.expanded.get() || !matches!(self.role, Role::Tool | Role::ToolError) {
            return false;
        }
        let body_line_count = self
            .text
            .split_once('\n')
            .map(|x| x.1)
            .map(|b| b.lines().count())
            .unwrap_or(0);
        body_line_count > Self::MAX_TOOL_LINES
    }

    /// Reveals the rest of a folded tool result and invalidates the
    /// rendered-lines cache so the next `content_lines()` call re-renders
    /// in full instead of returning the still-truncated cached version.
    pub(crate) fn toggle_expanded(&self) {
        self.expanded.set(!self.expanded.get());
        *self.rendered.borrow_mut() = None;
    }

    /// Plain text of this block: the stored `text` when present, otherwise the
    /// concatenated content of its styled spans (e.g. a copied diff).
    pub(crate) fn plain_text(&self) -> String {
        if !self.text.is_empty() {
            return self.text.clone();
        }
        self.lines
            .iter()
            .map(|spans| spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Indent applied to continuation lines so they clear the avatar marker.
    fn pad() -> &'static str {
        "    "
    }

    /// Rendered lines for the transcript, `width` columns wide. User and
    /// Assistant turns render as bordered chat bubbles (`bubble_lines`,
    /// matching `zeus-cli.html`'s `.bubble` rows); a tool call that actually
    /// produced output gets a card (`tool_card_lines`) instead of blending
    /// into the log — everything else (a bare tool-call-started line, plain
    /// status/error/info lines) stays plain marker-prefixed text like a log
    /// line, matching the HTML's unbubbled `·`/`✗` status rows. This mirrors
    /// a two-tier "inline vs card" split a reference product uses: cheap
    /// tool activity reads as a log line, a tool call with a real result
    /// (a diff, a file read, command output) reads as a distinct block.
    pub(crate) fn to_lines(&self, width: u16) -> Vec<Line<'static>> {
        match self.role {
            Role::User | Role::Assistant => self.bubble_lines(width),
            Role::Tool | Role::ToolError if self.has_output_body() => self.tool_card_lines(width),
            _ => self.marked_lines(),
        }
    }

    /// Whether this Tool/ToolError block's `text` has a body past its
    /// header line (see `style_tool_header`'s header/body split) — a bare
    /// `ToolCallStarted` line, or a `ToolCallFinished` whose result was
    /// truly empty, has none.
    fn has_output_body(&self) -> bool {
        self.text
            .split_once('\n')
            .map(|x| x.1)
            .is_some_and(|body| !body.trim().is_empty())
    }

    /// A tool call that produced real output gets a left-accent-bar card —
    /// the same visual language the composer already uses (`Borders::LEFT` +
    /// a filled panel background) — instead of blending into the plain log
    /// lines around it.
    ///
    /// Built by hand (spans with an explicit `bg` filled out to `width`)
    /// rather than an actual `ratatui::widgets::Block` because the whole
    /// transcript is one flattened `Text` rendered by a single `Paragraph`,
    /// the same reason `bubble_lines` draws its own border characters
    /// instead of nesting a real bordered widget.
    fn tool_card_lines(&self, width: u16) -> Vec<Line<'static>> {
        let is_error = self.role == Role::ToolError;
        let accent = if is_error {
            theme::red()
        } else {
            theme::cyan()
        };
        let bg = theme::CARD_BG;
        let avail = (width as usize)
            .saturating_sub(2)
            .clamp(20, Self::MAX_CONTENT_W);

        let pad_row = || {
            Line::from(vec![
                Span::styled("▎", accent.bg(bg)),
                Span::styled(" ".repeat(avail + 1), Style::default().bg(bg)),
            ])
        };
        let content = self.content_lines();
        let mut out = Vec::with_capacity(content.len() + 2);
        out.push(pad_row());
        for line in content {
            let mut spans = vec![Span::styled("▎ ", accent.bg(bg))];
            let mut used = 2usize;
            for s in line.spans {
                if used > avail {
                    break;
                }
                // Preserve a span's own background (e.g. a diff +/- line's
                // tint) instead of flattening it into the card's panel bg.
                let style = if s.style.bg.is_some() {
                    s.style
                } else {
                    s.style.bg(bg)
                };
                let remaining = avail + 1 - used;
                let content_len = s.content.chars().count();
                if content_len > remaining {
                    // Diff lines are pre-padded to a fixed width (see
                    // `DIFF_DEFAULT_WIDTH`) independent of the real live
                    // width, since that padding is baked in at memoization
                    // time before this function ever sees a terminal size.
                    // Clip here — the one place that knows the real width —
                    // so an over-wide span can't escape the card and get
                    // re-wrapped by the transcript's own `Paragraph::wrap`,
                    // which would otherwise leave a near-empty tinted
                    // "ghost" row trailing every diff line on any terminal
                    // narrower than that fixed pad width.
                    let clipped: String = s.content.chars().take(remaining).collect();
                    used += remaining;
                    spans.push(Span::styled(clipped, style));
                    break;
                }
                used += content_len;
                spans.push(Span::styled(s.content, style));
            }
            if used < avail + 1 {
                spans.push(Span::styled(
                    " ".repeat(avail + 1 - used),
                    Style::default().bg(bg),
                ));
            }
            out.push(Line::from(spans));
        }
        out.push(pad_row());
        out
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
                Line::from(Span::styled(
                    "YOU",
                    theme::dim().add_modifier(Modifier::BOLD),
                ))
                .alignment(Alignment::Right),
            );
            out
        } else {
            // Every line is wrapped to the available width first —
            // `content_lines()` (markdown/syntax-highlighted spans) is
            // never pre-wrapped to any width on its own, so a long
            // paragraph or a wide highlighted line needs its own pass here
            // rather than sailing straight past the terminal edge.
            let max_inner_w = avail.saturating_sub(2).clamp(10, Self::MAX_CONTENT_W);
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
                        vec![Span::styled(
                            "⚡ ",
                            theme::violet().add_modifier(Modifier::BOLD),
                        )]
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
            let highlighted = crate::highlight::markdown_lines(&self.text, text_style);
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

            let mut lines: Vec<Line<'static>> =
                vec![style_tool_header(header, self.role == Role::ToolError)];
            let mut body_lines: Vec<Line<'static>> = if crate::highlight::looks_like_diff(body) {
                crate::highlight::diff_lines(body, text_style, crate::highlight::DIFF_DEFAULT_WIDTH)
                    .into_iter()
                    .map(Line::from)
                    .collect()
            } else {
                body.lines()
                    .map(|l| style_tool_body_line(l, text_style))
                    .collect()
            };
            if body_lines.len() > Self::MAX_TOOL_LINES && !self.expanded.get() {
                let omitted = body_lines.len() - Self::MAX_TOOL_LINES;
                body_lines.truncate(Self::MAX_TOOL_LINES);
                body_lines.push(Line::from(Span::styled(
                    format!("… {omitted} more line(s) — click this message to expand"),
                    // `dim()` not `faint()` — this is an actionable
                    // instruction, not decoration, and `faint()` measures
                    // well under WCAG AA contrast against the surrounding
                    // card background.
                    theme::dim().add_modifier(Modifier::ITALIC),
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
/// `"toolname (done)"`/`"toolname (done, 297ms)"`/`"toolname (failed)"`
/// (finished, with a timing suffix appended whenever `apply_agent_event`
/// knows how long the call took) — a fixed format this module itself
/// writes, not external input, so splitting on the first space/
/// parenthesized status is safe. Bolds the tool name, colors the status
/// word, and dims+caps raw call arguments so a long inline JSON blob (a
/// big grep pattern, a large file write) doesn't dominate the line the way
/// the unstyled flat dump used to.
pub(crate) fn style_tool_header(header: &str, is_error: bool) -> Line<'static> {
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
                // Trust the block's actual role for color rather than
                // re-parsing the status word: it now carries a variable
                // ", <N>ms" timing suffix ("done, 297ms"), and an exact
                // `== "done"` match would silently miscolor every timed
                // success red, as if every tool call had failed.
                let status_style = if is_error {
                    theme::red().add_modifier(Modifier::BOLD)
                } else {
                    theme::green().add_modifier(Modifier::BOLD)
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
pub(crate) fn style_tool_body_line(line: &str, text_style: Style) -> Line<'static> {
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
pub(crate) fn line_char_width(line: &Line) -> usize {
    line.spans.iter().map(|s| s.content.chars().count()).sum()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Flatten a block's rendered lines into their plain text, the way a
    /// transcript row reads after styling.
    fn flat(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect()
    }

    #[test]
    fn tool_block_folds_long_output_and_expands() {
        let body = "read_file src/main.rs\n".to_string() + &"payload line\n".repeat(50);
        let block = Block_::new(Role::Tool, body);
        assert!(block.is_foldable());
        let folded = flat(&block.to_lines(120));
        assert!(folded.contains("more line(s)"));
        // The 50-body-line dump folds down to the 40-line cap.
        assert_eq!(
            folded.matches("payload line").count(),
            Block_::MAX_TOOL_LINES
        );

        block.toggle_expanded();
        assert!(!block.is_foldable());
        let expanded = flat(&block.to_lines(120));
        assert!(!expanded.contains("more line(s)"));
        // Expanded reveals every line, no truncation.
        assert_eq!(expanded.matches("payload line").count(), 50);
    }

    #[test]
    fn short_tool_output_does_not_fold() {
        let block = Block_::new(Role::Tool, "bash echo hi\nok".to_string());
        assert!(!block.is_foldable());
    }

    #[test]
    fn user_bubble_renders_marker_and_text() {
        let block = Block_::new(Role::User, "hello world".to_string());
        let rendered = flat(&block.to_lines(80));
        assert!(rendered.contains("YOU"));
        assert!(rendered.contains("hello world"));
    }
}
