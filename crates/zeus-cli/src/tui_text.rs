//! Pure text/geometry/formatting helpers for the TUI — no AppState, no
//! Agent, no rendering/event-loop state. Kept separate from `tui.rs` so the
//! interactive half stays small and these stay trivially unit-testable.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, Clear};
use ratatui::Frame;
use zeus_provider::TokenUsage;

use super::theme;
/// Word-wraps each `\n`-separated paragraph independently to `width`
/// columns, so a deliberate blank line in a pasted message survives.
pub(crate) fn wrap_preserving_newlines(text: &str, width: usize) -> Vec<String> {
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

pub(crate) fn char_count(s: &str) -> usize {
    s.chars().count()
}

/// Masks a secret (API key) as it's typed/pasted — all but the last
/// character become `•`, so you can confirm you pasted the right thing
/// without the key sitting in plaintext on screen.
pub(crate) fn mask_secret(s: &str) -> String {
    let mut masked = "•".repeat(s.chars().count().saturating_sub(1));
    if let Some(last) = s.chars().last() {
        masked.push(last);
    }
    masked
}

pub(crate) fn menu_height(matches: &[(&str, &str)]) -> u16 {
    if matches.is_empty() {
        0
    } else {
        matches.len().min(8) as u16 + 2
    }
}

/// Centers a `width`x`height` box within `area` (clamped to fit).
pub(crate) fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    }
}

/// Greedy word-wrap into lines no wider than `width` columns.
pub(crate) fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let width = width.max(10);
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let mut remaining = word;
        loop {
            let word_len = char_count(remaining);
            let candidate_len = if current.is_empty() {
                word_len
            } else {
                char_count(&current) + 1 + word_len
            };
            if candidate_len <= width {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(remaining);
                break;
            }
            if !current.is_empty() {
                lines.push(std::mem::take(&mut current));
                continue;
            }
            // `remaining` alone is still longer than `width` — a pasted API
            // key, URL, or path with no whitespace to break on. Ratatui's
            // real `Paragraph` wrap hard-breaks an overlong word at the
            // column boundary rather than letting it run past the edge, so
            // this has to match that or every caller sizing a box / placing
            // a cursor from this function's line count disagrees with what
            // actually gets drawn.
            let chunk: String = remaining.chars().take(width).collect();
            let chunk_bytes = chunk.len();
            lines.push(chunk);
            remaining = &remaining[chunk_bytes..];
            if remaining.is_empty() {
                break;
            }
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

pub(crate) fn textwrap_len(text: &str, width: usize) -> usize {
    wrap_text(text, width).len()
}

/// Word-wraps a line of *styled* spans to `width` columns, preserving each
/// character's original style — the span-aware equivalent of `wrap_text`.
/// Assistant bubbles carry markdown/syntax-highlighted spans that
/// `wrap_text` can't touch (it only understands plain strings), so a long
/// reply or a wide highlighted line needs this instead to stay inside its
/// box rather than overflowing past the right border.
pub(crate) fn wrap_spans(spans: &[Span<'static>], width: usize) -> Vec<Vec<Span<'static>>> {
    let width = width.max(10);
    let chars: Vec<(char, Style)> = spans
        .iter()
        .flat_map(|s| {
            let style = s.style;
            s.content
                .chars()
                .map(move |c| (c, style))
                .collect::<Vec<_>>()
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
        if !current.is_empty() && current.len() + 1 + word.len() > width {
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

/// Linearly interpolate between two RGB colors (used for the TODO progress
/// bar's violet → mode-accent gradient).
pub(crate) fn lerp_color(a: Color, b: Color, t: f32) -> Color {
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
pub(crate) fn gradient_wordmark(text: &str) -> Vec<Span<'static>> {
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
            Span::styled(
                c.to_string(),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            )
        })
        .collect()
}

/// "480ms" under a second, "2.3s" at or above.
pub(crate) fn fmt_duration(d: std::time::Duration) -> String {
    if d.as_secs() >= 1 {
        format!("{:.1}s", d.as_secs_f32())
    } else {
        format!("{}ms", d.as_millis())
    }
}

/// Abbreviates a token count the way the reference UI does ("42.1k"
/// instead of "42123") — anything under 1000 prints as-is.
pub(crate) fn format_token_count(n: u32) -> String {
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
pub(crate) fn cost_per_million_tokens(provider: &str, model_id: &str) -> Option<(f64, f64)> {
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
pub(crate) fn fmt_price(v: f64) -> String {
    if (v - v.round()).abs() < 0.001 {
        format!("{v:.0}")
    } else {
        format!("{v:.2}")
    }
}

/// `None` means "we don't have pricing data for this provider/model" —
/// callers should render that as "no estimate available", never as $0.00
/// (which would misleadingly imply the usage was actually free).
pub(crate) fn estimate_cost(provider: &str, model_id: &str, usage: &TokenUsage) -> Option<f64> {
    let (prompt_rate, completion_rate) = cost_per_million_tokens(provider, model_id)?;
    Some(
        (usage.prompt_tokens as f64 / 1_000_000.0) * prompt_rate
            + (usage.completion_tokens as f64 / 1_000_000.0) * completion_rate,
    )
}

/// Centers a `width`-wide, `height`-tall rect horizontally in `area` at a
/// specific `y` row (unlike `centered_rect`, which also centers vertically).
pub(crate) fn centered_row(area: Rect, y: u16, height: u16, width: u16) -> Rect {
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
pub(crate) fn opaque(f: &mut Frame, area: Rect) {
    // `Block`'s style only recolors existing cells — it doesn't blank their
    // glyphs, so stale content underneath would otherwise show through
    // tinted rather than covered. `Clear` actually resets the cells first.
    f.render_widget(Clear, area);
    f.render_widget(
        Block::default().style(Style::default().bg(theme::INK)),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_text_fits_on_one_line() {
        assert_eq!(
            wrap_text("hello world", 20),
            vec!["hello world".to_string()]
        );
    }

    #[test]
    fn wrap_text_empty_input_yields_one_empty_line() {
        assert_eq!(wrap_text("", 20), vec![String::new()]);
    }

    #[test]
    fn wrap_text_breaks_on_whitespace_within_width() {
        let lines = wrap_text("the quick brown fox jumps over the lazy dog", 12);
        for line in &lines {
            assert!(char_count(line) <= 12, "line {line:?} exceeds width 12");
        }
        // Rejoining with single spaces must reproduce the original words —
        // wrapping shouldn't drop or duplicate any text.
        assert_eq!(
            lines.join(" "),
            "the quick brown fox jumps over the lazy dog"
        );
    }

    /// Regression test for a real bug: a single unbroken token longer than
    /// `width` (a pasted API key, URL, long path) used to come back as one
    /// oversized line, while ratatui's own `Paragraph` wrap hard-breaks it —
    /// height/cursor-position math computed from this function's line count
    /// then disagreed with what was actually drawn (tui.rs, `wrap_text`).
    #[test]
    fn wrap_text_hard_breaks_a_long_unbroken_token() {
        let token = "a".repeat(26);
        let lines = wrap_text(&token, 10);
        assert_eq!(lines, vec!["a".repeat(10), "a".repeat(10), "a".repeat(6)]);
        for line in &lines {
            assert!(char_count(line) <= 10);
        }
    }

    #[test]
    fn wrap_text_hard_break_is_utf8_safe() {
        // Multi-byte characters (each 3 bytes in UTF-8) — a byte-index split
        // here would panic or corrupt the string; `wrap_text` must only
        // ever split on char boundaries.
        let token = "日".repeat(15);
        let lines = wrap_text(&token, 10);
        for line in &lines {
            assert!(char_count(line) <= 10);
        }
        assert_eq!(lines.iter().map(|l| char_count(l)).sum::<usize>(), 15);
    }

    #[test]
    fn wrap_text_mixes_short_words_and_a_long_token() {
        let url = "x".repeat(60);
        let text = format!("hi {url} bye");
        let lines = wrap_text(&text, 20);
        for line in &lines {
            assert!(char_count(line) <= 20, "line {line:?} exceeds width 20");
        }
        // Every original character still shows up somewhere.
        let rejoined: String = lines.concat();
        assert!(rejoined.contains("hi"));
        assert!(rejoined.contains("bye"));
        assert_eq!(rejoined.chars().filter(|&c| c == 'x').count(), 60);
    }

    #[test]
    fn wrap_text_clamps_width_to_a_minimum_of_ten() {
        // A caller-requested width under 10 (e.g. a tiny terminal) doesn't
        // shrink further — `width.max(10)` is a floor, not a suggestion.
        let lines = wrap_text("abcdefghijklmnop", 2);
        assert_eq!(lines, vec!["abcdefghij".to_string(), "klmnop".to_string()]);
    }

    #[test]
    fn wrap_preserving_newlines_keeps_blank_lines() {
        assert_eq!(
            wrap_preserving_newlines("hello\n\nworld", 40),
            vec!["hello".to_string(), String::new(), "world".to_string()]
        );
    }

    #[test]
    fn mask_secret_keeps_only_the_last_character() {
        assert_eq!(mask_secret(""), "");
        assert_eq!(mask_secret("a"), "a");
        assert_eq!(mask_secret("abcd"), "•••d");
    }

    #[test]
    fn centered_row_centers_horizontally_at_a_fixed_y() {
        let area = Rect {
            x: 0,
            y: 0,
            width: 100,
            height: 50,
        };
        let r = centered_row(area, 10, 5, 20);
        assert_eq!(
            r,
            Rect {
                x: 40,
                y: 10,
                width: 20,
                height: 5
            }
        );
    }

    #[test]
    fn centered_row_clamps_width_to_the_area() {
        let area = Rect {
            x: 0,
            y: 0,
            width: 100,
            height: 50,
        };
        let r = centered_row(area, 0, 5, 150);
        assert_eq!(r.width, 100);
        assert_eq!(r.x, 0);
    }

    #[test]
    fn centered_row_collapses_height_when_y_is_past_the_area() {
        // The exact mechanism that protects a short terminal from an
        // overflowing empty-state screen: a row positioned past the visible
        // area shrinks to zero height instead of panicking or drawing
        // out-of-bounds.
        let area = Rect {
            x: 0,
            y: 0,
            width: 100,
            height: 10,
        };
        let r = centered_row(area, 20, 5, 20);
        assert_eq!(r.height, 0);
    }

    #[test]
    fn centered_rect_centers_both_axes() {
        let area = Rect {
            x: 0,
            y: 0,
            width: 100,
            height: 50,
        };
        let r = centered_rect(30, 10, area);
        assert_eq!(
            r,
            Rect {
                x: 35,
                y: 20,
                width: 30,
                height: 10
            }
        );
    }

    #[test]
    fn centered_rect_clamps_to_the_area_when_oversized() {
        let area = Rect {
            x: 0,
            y: 0,
            width: 40,
            height: 20,
        };
        let r = centered_rect(1000, 1000, area);
        assert_eq!(
            r,
            Rect {
                x: 0,
                y: 0,
                width: 40,
                height: 20
            }
        );
    }

    #[test]
    fn menu_height_is_zero_when_empty() {
        assert_eq!(menu_height(&[]), 0);
    }

    #[test]
    fn menu_height_grows_with_match_count_up_to_a_cap() {
        let three = [("a", ""), ("b", ""), ("c", "")];
        assert_eq!(menu_height(&three), 5);

        let twenty: Vec<(&str, &str)> = (0..20).map(|_| ("cmd", "desc")).collect();
        // Capped at 8 visible rows + 2 (border) regardless of match count —
        // a large command list scrolls instead of the popup growing without
        // bound.
        assert_eq!(menu_height(&twenty), 10);
    }
}
