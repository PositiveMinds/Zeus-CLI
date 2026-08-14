use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use std::sync::OnceLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Color as SC, FontStyle as SF, Theme, ThemeSet};
use syntect::parsing::{SyntaxReference, SyntaxSet};
use syntect::util::LinesWithEndings;

/// Code-block background, matching the demo bubble's dark `pre` backing.
const CODE_BG: Color = Color::Rgb(0x0c, 0x0f, 0x16);
/// Inline `` `code` `` foreground — warm amber, kept distinct from the
/// cyan used for bullets/ordered-list markers/diff hunks so inline code
/// reads as its own thing rather than reusing an unrelated marker color.
const INLINE_CODE_FG: Color = Color::Rgb(0xe0, 0xaf, 0x68);

/// Fenced-code syntax theme — magenta/purple keywords, warm tan strings,
/// muted green comments, soft-green numbers, cyan function names, on the
/// same dark background the rest of a code block uses. Picked to match a
/// specific reference screenshot rather than any bundled syntect theme
/// (the closest built-in, base16-ocean, doesn't have the purple keyword —
/// hence a small hand-built `.tmTheme` instead of `ThemeSet::load_defaults`).
const CODE_THEME_TMTHEME: &str = r##"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>name</key>
	<string>zeus-dark</string>
	<key>settings</key>
	<array>
		<dict>
			<key>settings</key>
			<dict>
				<key>background</key>
				<string>#0C0F16</string>
				<key>foreground</key>
				<string>#D4D4D4</string>
			</dict>
		</dict>
		<dict>
			<key>scope</key>
			<string>comment</string>
			<key>settings</key>
			<dict>
				<key>foreground</key>
				<string>#6A7280</string>
				<key>fontStyle</key>
				<string>italic</string>
			</dict>
		</dict>
		<dict>
			<key>scope</key>
			<string>keyword, keyword.control, storage.type, storage.modifier, keyword.operator.word</string>
			<key>settings</key>
			<dict>
				<key>foreground</key>
				<string>#C586C0</string>
				<key>fontStyle</key>
				<string>italic</string>
			</dict>
		</dict>
		<dict>
			<key>scope</key>
			<string>string</string>
			<key>settings</key>
			<dict>
				<key>foreground</key>
				<string>#CE9178</string>
			</dict>
		</dict>
		<dict>
			<key>scope</key>
			<string>constant.numeric, constant.language</string>
			<key>settings</key>
			<dict>
				<key>foreground</key>
				<string>#B5CEA8</string>
			</dict>
		</dict>
		<dict>
			<key>scope</key>
			<string>entity.name.function, support.function, meta.function-call</string>
			<key>settings</key>
			<dict>
				<key>foreground</key>
				<string>#58C0FF</string>
			</dict>
		</dict>
		<dict>
			<key>scope</key>
			<string>entity.name.class, entity.name.type, support.type, support.class</string>
			<key>settings</key>
			<dict>
				<key>foreground</key>
				<string>#4EC9B0</string>
			</dict>
		</dict>
		<dict>
			<key>scope</key>
			<string>variable.parameter</string>
			<key>settings</key>
			<dict>
				<key>foreground</key>
				<string>#9CDCFE</string>
			</dict>
		</dict>
	</array>
</dict>
</plist>
"##;

static SYNTAX_SET: OnceLock<SyntaxSet> = OnceLock::new();
static CODE_THEME: OnceLock<Theme> = OnceLock::new();

fn syntax_set() -> &'static SyntaxSet {
    SYNTAX_SET.get_or_init(SyntaxSet::load_defaults_newlines)
}

fn theme() -> &'static Theme {
    CODE_THEME.get_or_init(|| {
        match ThemeSet::load_from_reader(&mut std::io::Cursor::new(CODE_THEME_TMTHEME.as_bytes())) {
            Ok(t) => t,
            // A broken embedded .tmTheme (a plist typo, a bad hex color —
            // exactly the class of mistake this hand-authored string has
            // already had once) would otherwise panic the whole app the
            // first time any fenced code block renders, rather than at
            // startup where it'd be caught immediately. Degrade to the
            // bundled default instead of taking the app down over syntax
            // highlighting colors.
            Err(e) => {
                eprintln!("warning: embedded zeus-dark .tmTheme failed to parse ({e}), falling back to default theme");
                ThemeSet::load_defaults()
                    .themes
                    .remove("base16-ocean.dark")
                    .expect("syntect's bundled base16-ocean.dark theme exists")
            }
        }
    })
}

fn rat_color(c: SC) -> Color {
    Color::Rgb(c.r, c.g, c.b)
}

/// Best-effort language-token resolution for fenced code blocks. Syntect's
/// bundled `load_defaults_newlines()` package is the old Sublime Text
/// default set — it's missing several languages fences commonly name
/// (TypeScript/TSX/JSX, TOML, Dockerfile, Vue/Svelte...). Rather than
/// silently falling back to zero highlighting for those, try the raw
/// token/extension first, then a small table of close substitutes, before
/// giving up on plain text.
fn resolve_syntax(lang: Option<&str>) -> &'static SyntaxReference {
    let set = syntax_set();
    let raw = match lang.map(|l| l.trim().to_lowercase()) {
        Some(r) if !r.is_empty() => r,
        _ => return set.find_syntax_plain_text(),
    };
    if let Some(s) = set
        .find_syntax_by_token(&raw)
        .or_else(|| set.find_syntax_by_extension(&raw))
    {
        return s;
    }
    for alias in lang_aliases(&raw) {
        if let Some(s) = set
            .find_syntax_by_token(alias)
            .or_else(|| set.find_syntax_by_extension(alias))
        {
            return s;
        }
    }
    set.find_syntax_plain_text()
}

/// Close substitutes to try, in order, when a fence's language token isn't
/// directly known to the bundled syntax set — not a full alias system, just
/// the common cases an AI coding agent's own fences actually use.
fn lang_aliases(token: &str) -> &'static [&'static str] {
    match token {
        "ts" | "typescript" | "tsx" | "jsx" | "mjs" | "cjs" => &["js", "javascript"],
        "yml" => &["yaml"],
        "sh" | "shell" | "zsh" | "console" => &["bash", "sh"],
        "c++" | "cplusplus" | "cxx" => &["cpp"],
        "cs" | "c#" => &["csharp", "cs"],
        "objc" | "objectivec" => &["objective-c"],
        "toml" | "cfg" | "dotenv" | "env" => &["ini"],
        "vue" | "svelte" => &["html"],
        "md" => &["markdown"],
        "dockerfile" | "docker" => &["bash"],
        _ => &[],
    }
}

fn rock_style(style: &syntect::highlighting::Style) -> Style {
    let mut s = if style.background.a == 0 {
        Style::default().bg(CODE_BG)
    } else {
        Style::default().bg(rat_color(style.background))
    };
    s = s.fg(rat_color(style.foreground));
    if style.font_style.contains(SF::BOLD) {
        s = s.add_modifier(Modifier::BOLD);
    }
    if style.font_style.contains(SF::ITALIC) {
        s = s.add_modifier(Modifier::ITALIC);
    }
    if style.font_style.contains(SF::UNDERLINE) {
        s = s.add_modifier(Modifier::UNDERLINED);
    }
    s
}

/// Split a body of text into per-line styled spans. Fenced ```code blocks are
/// tokenized with syntect and colored using the loaded theme; everything else
/// uses `plain_style` unchanged.
pub fn markdown_lines(text: &str, plain_style: Style) -> Vec<Vec<Span<'static>>> {
    let mut out: Vec<Vec<Span<'static>>> = Vec::new();
    let mut plain: Vec<String> = Vec::new();
    let mut code: Vec<String> = Vec::new();
    let mut lang: Option<String> = None;
    let mut in_fence = false;

    let flush_plain = |out: &mut Vec<Vec<Span<'static>>>, plain: &Vec<String>| {
        for l in plain {
            out.push(style_markdown_line(l, plain_style));
        }
    };
    let flush_code =
        |out: &mut Vec<Vec<Span<'static>>>, code: &Vec<String>, lang: &Option<String>| {
            let syntax = resolve_syntax(lang.as_deref());
            let mut h = HighlightLines::new(syntax, theme());
            for line_w in LinesWithEndings::from(&code.join("\n")) {
                let line = line_w.trim_end_matches('\n');
                let ranges = match h.highlight_line(line, syntax_set()) {
                    Ok(r) => r,
                    Err(_) => {
                        out.push(vec![Span::styled(line.to_string(), plain_style)]);
                        continue;
                    }
                };
                let spans: Vec<Span<'static>> = ranges
                    .iter()
                    .map(|(style, text)| Span::styled(text.to_string(), rock_style(style)))
                    .collect();
                out.push(spans);
            }
        };

    for line in text.split('\n') {
        let trimmed = line.trim_start();
        if in_fence && trimmed.starts_with("```") {
            flush_code(&mut out, &code, &lang);
            code = Vec::new();
            lang = None;
            in_fence = false;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("```") {
            flush_plain(&mut out, &plain);
            plain.clear();
            lang = Some(rest.trim().to_string());
            code = Vec::new();
            in_fence = true;
            continue;
        }
        if in_fence {
            code.push(line.to_string());
        } else {
            plain.push(line.to_string());
        }
    }
    flush_plain(&mut out, &plain);
    if in_fence {
        flush_code(&mut out, &code, &lang);
    }
    out
}

/// Lightweight styling for a single non-fenced markdown line: a colored
/// marker for headers/list items/blockquotes, plus inline `**bold**` and
/// `` `code` `` spans within it. Not a full markdown parser (tables and
/// nested structure aren't attempted) — just the handful of constructs
/// that show up constantly in LLM prose and otherwise render as raw
/// asterisks/backticks/hashes with no distinction at all.
fn style_markdown_line(line: &str, base: Style) -> Vec<Span<'static>> {
    let trimmed = line.trim_start();
    let indent_len = line.len() - trimmed.len();
    let mut spans = Vec::new();
    if indent_len > 0 {
        spans.push(Span::styled(line[..indent_len].to_string(), base));
    }

    if let Some(rest) = trimmed
        .strip_prefix("### ")
        .or_else(|| trimmed.strip_prefix("## "))
        .or_else(|| trimmed.strip_prefix("# "))
    {
        spans.extend(style_inline(rest, base.add_modifier(Modifier::BOLD)));
        return spans;
    }
    if let Some(rest) = trimmed.strip_prefix("> ") {
        spans.push(Span::styled("▍ ", Style::default().fg(META_FG)));
        spans.extend(style_inline(rest, base.add_modifier(Modifier::ITALIC)));
        return spans;
    }
    if let Some(rest) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
    {
        spans.push(Span::styled(
            "• ",
            Style::default().fg(HUNK_FG).add_modifier(Modifier::BOLD),
        ));
        spans.extend(style_inline(rest, base));
        return spans;
    }
    if let Some((marker, rest)) = split_ordered_marker(trimmed) {
        spans.push(Span::styled(
            format!("{marker} "),
            Style::default().fg(HUNK_FG).add_modifier(Modifier::BOLD),
        ));
        spans.extend(style_inline(rest, base));
        return spans;
    }
    spans.extend(style_inline(trimmed, base));
    spans
}

/// Splits a `"1. rest"`-style ordered-list marker off the front of a line.
fn split_ordered_marker(s: &str) -> Option<(&str, &str)> {
    let dot = s.find(". ")?;
    let num = &s[..dot];
    if !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()) {
        Some((&s[..=dot], &s[dot + 2..]))
    } else {
        None
    }
}

/// Splits `**bold**`, `` `code` ``, and `~~strikethrough~~` out of a line's
/// remaining text into separately styled spans; everything else keeps
/// `base`. Picks whichever delimiter opens earliest so the three compose
/// correctly regardless of order.
fn style_inline(text: &str, base: Style) -> Vec<Span<'static>> {
    #[derive(Clone, Copy)]
    enum Kind {
        Bold,
        Code,
        Strike,
    }
    let mut spans = Vec::new();
    let mut rest = text;
    loop {
        let candidates = [
            (rest.find("**"), Kind::Bold),
            (rest.find('`'), Kind::Code),
            (rest.find("~~"), Kind::Strike),
        ];
        let Some((pos, kind)) = candidates
            .into_iter()
            .filter_map(|(p, k)| p.map(|pv| (pv, k)))
            .min_by_key(|(pv, _)| *pv)
        else {
            if !rest.is_empty() {
                spans.push(Span::styled(rest.to_string(), base));
            }
            break;
        };
        let (delim, delim_len, style_for): (&str, usize, fn(&str, Style) -> Style) = match kind {
            Kind::Bold => ("**", 2, |_, base| base.add_modifier(Modifier::BOLD)),
            Kind::Code => ("`", 1, |_, _| {
                Style::default().fg(INLINE_CODE_FG).bg(CODE_BG)
            }),
            Kind::Strike => ("~~", 2, |_, base| base.add_modifier(Modifier::CROSSED_OUT)),
        };
        match rest[pos + delim_len..].find(delim) {
            Some(end_rel) => {
                if pos > 0 {
                    spans.push(Span::styled(rest[..pos].to_string(), base));
                }
                spans.push(Span::styled(
                    rest[pos + delim_len..pos + delim_len + end_rel].to_string(),
                    style_for(delim, base),
                ));
                rest = &rest[pos + delim_len + end_rel + delim_len..];
            }
            None => {
                // Unmatched delimiter — no closing marker, so treat the
                // remainder as plain text rather than eating it.
                spans.push(Span::styled(rest.to_string(), base));
                break;
            }
        }
    }
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }
    spans
}

// ---------------------------------------------------------------------------
// Diff rendering
// ---------------------------------------------------------------------------

// Colors + whole-line tint strength picked to match a specific reference
// screenshot: a fairly visible dark-teal bar behind added lines and a
// dark-maroon bar behind removed ones (not the barely-there tint this used
// to be), with off-white body text and a brighter sign color for the
// leading `+`/`-` itself.
const ADD_FG: Color = Color::Rgb(0xd4, 0xf5, 0xe8);
const DEL_FG: Color = Color::Rgb(0xf5, 0xd9, 0xe0);
const HUNK_FG: Color = Color::Rgb(0x56, 0xb6, 0xc2);
const META_FG: Color = Color::Rgb(0x8f, 0x9b, 0xb0);
// Brightened from the original #0d2b28/#331224 — those sat at almost
// exactly `tui::theme::CARD_BG`'s luminance (contrast ~1.05:1), so a diff
// tint inside a tool-result card was separated from the card background by
// hue alone. Foreground text stays comfortably legible on the brighter
// versions (still 10:1+); the leading sign color's contrast was the actual
// limit on how far these could move (see `ADD_SIGN_FG`/`DEL_SIGN_FG`).
const ADD_BG: Color = Color::Rgb(0x16, 0x3d, 0x37);
const DEL_BG: Color = Color::Rgb(0x50, 0x1d, 0x38);
const HUNK_BG: Color = Color::Rgb(0x0a, 0x18, 0x1d);
// Brighter than `ADD_FG`/`DEL_FG` — just the leading `+`/`-` marker gets
// this, so it reads as a distinct "sign" against the softer body-text
// color, the way a gutter marker does in an editor's diff view.
const ADD_SIGN_FG: Color = Color::Rgb(0x4d, 0xe3, 0xb0);
const DEL_SIGN_FG: Color = Color::Rgb(0xf0, 0x6a, 0x9a);

/// Heuristic: is this text a unified diff we should pass to `diff_lines`?
pub fn looks_like_diff(text: &str) -> bool {
    text.lines().take(40).any(|l| {
        let t = l.trim_start();
        t.starts_with("diff --git")
            || t.starts_with("index ")
            || t.starts_with("@@ -")
            || t.starts_with("+++ ") && l.starts_with('+')
            || t.starts_with("--- ") && (l.starts_with('-') || l.starts_with("--- "))
    })
}

/// A changed line's background tint should read as a solid bar across the
/// available width (like an editor's diff gutter), not just hug the text —
/// this pads short lines with trailing spaces so the `bg` color fills the
/// row. `width` is a best-effort target: callers that already know the live
/// render width pass it exactly, callers that don't (e.g. `compute_lines`,
/// memoized independent of terminal size) pass a fixed reasonable default.
fn pad_to(line: &str, width: usize) -> String {
    let len = line.chars().count();
    if len >= width {
        line.to_string()
    } else {
        format!("{line}{}", " ".repeat(width - len))
    }
}

/// A changed line's leading `+`/`-` gets a brighter "sign" color than the
/// rest of the line, both sharing the same background tint — reads as a
/// distinct gutter marker rather than one flat-colored run of text.
fn diff_body_spans(
    line: &str,
    sign_fg: Color,
    body_fg: Color,
    bg: Color,
    width: usize,
) -> Vec<Span<'static>> {
    let mut chars = line.chars();
    let sign = chars.next().unwrap_or(' ');
    let rest: String = chars.collect();
    let padded_rest = pad_to(&format!("{sign}{rest}"), width);
    let rest_with_pad = padded_rest.chars().skip(1).collect::<String>();
    vec![
        Span::styled(
            sign.to_string(),
            Style::default()
                .fg(sign_fg)
                .bg(bg)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(rest_with_pad, Style::default().fg(body_fg).bg(bg)),
    ]
}

/// Fixed fallback width for callers that can't supply the live render width
/// (diff bodies rendered from `compute_lines`, which is memoized independent
/// of terminal size) — wide enough to fully tint the vast majority of real
/// terminal widths; harmless overshoot on anything wider just means the tint
/// doesn't quite reach the far edge.
pub const DIFF_DEFAULT_WIDTH: usize = 120;

/// Colorize a unified diff into styled spans for the TUI. Lines are colored by
/// their leading marker: `+` green, `-` red, `@@` hunks cyan, file metadata
/// (diff/---/+++/index/new file) dim, unchanged lines keep `plain_style`.
/// `width` sets how far the changed-line background tint extends.
pub fn diff_lines(text: &str, plain_style: Style, width: usize) -> Vec<Vec<Span<'static>>> {
    text.lines()
        .map(|l| {
            let tag = l.chars().next().unwrap_or(' ');
            let spans = match tag {
                'd' if l.starts_with("diff ") => {
                    vec![Span::styled(l.to_string(), Style::default().fg(META_FG))]
                }
                '-' if l.starts_with("--- ") => {
                    vec![Span::styled(l.to_string(), Style::default().fg(META_FG))]
                }
                '+' if l.starts_with("+++ ") => {
                    vec![Span::styled(l.to_string(), Style::default().fg(META_FG))]
                }
                'i' if l.starts_with("index ") => {
                    vec![Span::styled(l.to_string(), Style::default().fg(META_FG))]
                }
                'n' if l.starts_with("new file ") => {
                    vec![Span::styled(l.to_string(), Style::default().fg(META_FG))]
                }
                '#' if l.starts_with("Binary files ") => {
                    vec![Span::styled(l.to_string(), Style::default().fg(META_FG))]
                }
                '+' => diff_body_spans(l, ADD_SIGN_FG, ADD_FG, ADD_BG, width),
                '-' => diff_body_spans(l, DEL_SIGN_FG, DEL_FG, DEL_BG, width),
                '@' => vec![Span::styled(
                    pad_to(l, width),
                    Style::default().fg(HUNK_FG).bg(HUNK_BG),
                )],
                _ => vec![Span::styled(l.to_string(), plain_style)],
            };
            spans
        })
        .collect()
}

/// One row of a side-by-side diff: a full-width `Header` (file metadata or
/// `@@` hunk line) or a `Pair` of old/new column cells. A `Pair` with both
/// sides set is an unchanged context line; a removed-only row shows on the
/// left, an added-only row on the right.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffRow {
    Header(String),
    Pair(Option<String>, Option<String>),
}

/// Parse a unified diff into rows for the two-pane `/diff` view. Removed
/// (`-`) and added (`+`) lines that appear adjacent are paired 1:1 in order
/// (FIFO) so a changed line reads as `old → new` across the two columns
/// rather than two stacked rows. Unchanged lines fill both cells.
pub fn side_by_side_rows(text: &str) -> Vec<DiffRow> {
    let mut rows = Vec::new();
    let mut removed: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    for l in text.lines() {
        let tag = l.chars().next().unwrap_or(' ');
        if l.starts_with("diff ")
            || l.starts_with("index ")
            || l.starts_with("--- ")
            || l.starts_with("+++ ")
            || l.starts_with("new file ")
            || l.starts_with("Binary files ")
            || tag == '@'
        {
            for old in removed.drain(..) {
                rows.push(DiffRow::Pair(Some(old), None));
            }
            rows.push(DiffRow::Header(l.to_string()));
        } else if tag == '-' {
            removed.push_back(l[1..].to_string());
        } else if tag == '+' {
            match removed.pop_front() {
                Some(old) => rows.push(DiffRow::Pair(Some(old), Some(l[1..].to_string()))),
                None => rows.push(DiffRow::Pair(None, Some(l[1..].to_string()))),
            }
        } else {
            for old in removed.drain(..) {
                rows.push(DiffRow::Pair(Some(old), None));
            }
            rows.push(DiffRow::Pair(Some(l.to_string()), Some(l.to_string())));
        }
    }
    for old in removed.drain(..) {
        rows.push(DiffRow::Pair(Some(old), None));
    }
    rows
}

/// ANSI-escape a unified diff for the plain REPL / non-TUI output. Honors the
/// same fancy-output gate as the rest of the REPL styling (`styled`): when
/// stdout is piped/redirected or `NO_COLOR` is set, the diff is returned
/// plain so no escape codes leak into scripts or log files.
pub fn ansi_diff(text: &str) -> String {
    if !crate::ui::supports_fancy_output() {
        return text.to_string();
    }
    let mut out = String::new();
    for l in text.lines() {
        let tag = l.chars().next().unwrap_or(' ');
        match tag {
            '+' => out.push_str(&format!("\x1b[38;2;90;175;138m{l}\x1b[0m\n")),
            '-' => out.push_str(&format!("\x1b[38;2;224;97;113m{l}\x1b[0m\n")),
            '@' => out.push_str(&format!("\x1b[38;2;86;182;194m{l}\x1b[0m\n")),
            _ => out.push_str(&format!("{l}\n")),
        }
    }
    if text.ends_with('\n') {
        let _ = out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Style;

    #[test]
    fn plain_text_untouched() {
        let lines = markdown_lines("hello world", Style::default());
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0][0].content, "hello world");
    }

    #[test]
    fn fenced_rust_block() {
        let text = "```rust\nfn main() {}\n```\nafter";
        let lines = markdown_lines(text, Style::default());
        let flat: String = lines
            .iter()
            .flat_map(|l| l.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("");
        assert!(flat.contains("fn main() {}"));
        assert!(flat.contains("after"));
    }

    #[test]
    fn unlabeled_fence() {
        let lines = markdown_lines("```\nx\n```", Style::default());
        let flat: String = lines
            .iter()
            .flat_map(|l| l.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("");
        assert!(flat.contains('x'));
    }

    #[test]
    fn side_by_side_pairs_adjacent_removed_added() {
        let text = "--- a/f\n+++ b/f\n@@ -1,3 +1,3 @@\n old\n-changed\n+changed!\nsame\n";
        let rows = side_by_side_rows(text);
        assert!(rows[0] == DiffRow::Header("--- a/f".to_string()));
        assert!(rows[1] == DiffRow::Header("+++ b/f".to_string()));
        assert!(matches!(&rows[2], DiffRow::Header(h) if h.starts_with("@@")));
        assert!(rows[3] == DiffRow::Pair(Some(" old".into()), Some(" old".into())));
        assert!(rows[4] == DiffRow::Pair(Some("changed".into()), Some("changed!".into())));
        assert!(rows[5] == DiffRow::Pair(Some("same".into()), Some("same".into())));
    }

    #[test]
    fn side_by_side_unmatched_removed_and_added_stand_alone() {
        let rows = side_by_side_rows("@@ -1 +1 @@\n-old\n+new\n");
        assert!(rows[1] == DiffRow::Pair(Some("old".into()), Some("new".into())));
    }

    #[test]
    fn side_by_side_removed_only_rows_have_empty_right_cell() {
        let rows = side_by_side_rows("@@ -1 +1 @@\n-dead\n");
        assert!(rows[1] == DiffRow::Pair(Some("dead".into()), None));
    }
}
