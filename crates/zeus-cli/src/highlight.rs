use std::sync::OnceLock;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
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

static SYNTAX_SET: OnceLock<SyntaxSet> = OnceLock::new();
static THEME_SET: OnceLock<ThemeSet> = OnceLock::new();

fn syntax_set() -> &'static SyntaxSet {
    SYNTAX_SET.get_or_init(SyntaxSet::load_defaults_newlines)
}

fn theme() -> &'static Theme {
    THEME_SET
        .get_or_init(ThemeSet::load_defaults)
        .themes
        .get("base16-ocean.dark")
        .expect("default syntect theme present")
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

    let flush_plain =
        |out: &mut Vec<Vec<Span<'static>>>, plain: &Vec<String>| {
            for l in plain {
                out.push(style_markdown_line(l, plain_style));
            }
        };
    let flush_code = |out: &mut Vec<Vec<Span<'static>>>,
                      code: &Vec<String>,
                      lang: &Option<String>| {
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
        if trimmed.starts_with("```") {
            flush_plain(&mut out, &plain);
            plain.clear();
            lang = Some(trimmed[3..].trim().to_string());
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
    if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
        spans.push(Span::styled("• ", Style::default().fg(HUNK_FG).add_modifier(Modifier::BOLD)));
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
            Kind::Code => ("`", 1, |_, _| Style::default().fg(INLINE_CODE_FG).bg(CODE_BG)),
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
}

// ---------------------------------------------------------------------------
// Diff rendering
// ---------------------------------------------------------------------------

const ADD_FG: Color = Color::Rgb(0x5a, 0xaf, 0x8a);
const DEL_FG: Color = Color::Rgb(0xe0, 0x61, 0x71);
const HUNK_FG: Color = Color::Rgb(0x56, 0xb6, 0xc2);
const META_FG: Color = Color::Rgb(0x8f, 0x9b, 0xb0);
const ADD_BG: Color = Color::Rgb(0x0c, 0x12, 0x0e);
const DEL_BG: Color = Color::Rgb(0x16, 0x0e, 0x10);
const HUNK_BG: Color = Color::Rgb(0x0a, 0x18, 0x1d);

/// Heuristic: is this text a unified diff we should pass to `diff_lines`?
pub fn looks_like_diff(text: &str) -> bool {
    text.lines()
        .take(40)
        .any(|l| {
            let t = l.trim_start();
            t.starts_with("diff --git")
                || t.starts_with("index ")
                || t.starts_with("@@ -")
                || t.starts_with("+++ ") && l.starts_with('+')
                || t.starts_with("--- ") && (l.starts_with('-') || l.starts_with("--- "))
        })
}

/// Colorize a unified diff into styled spans for the TUI. Lines are colored by
/// their leading marker: `+` green, `-` red, `@@` hunks cyan, file metadata
/// (diff/---/+++/index/new file) dim, unchanged lines keep `plain_style`.
pub fn diff_lines(text: &str, plain_style: Style) -> Vec<Vec<Span<'static>>> {
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
                '+' => vec![Span::styled(l.to_string(), Style::default().fg(ADD_FG).bg(ADD_BG))],
                '-' => vec![Span::styled(l.to_string(), Style::default().fg(DEL_FG).bg(DEL_BG))],
                '@' => vec![Span::styled(l.to_string(), Style::default().fg(HUNK_FG).bg(HUNK_BG))],
                _ => vec![Span::styled(l.to_string(), plain_style)],
            };
            spans
        })
        .collect()
}

/// ANSI-escape a unified diff for the plain REPL / non-TUI output.
pub fn ansi_diff(text: &str) -> String {
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