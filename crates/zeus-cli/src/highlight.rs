use std::sync::OnceLock;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Color as SC, FontStyle as SF, Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

/// Code-block background, matching the demo bubble's dark `pre` backing.
const CODE_BG: Color = Color::Rgb(0x0c, 0x0f, 0x16);

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
                out.push(vec![Span::styled(l.clone(), plain_style)]);
            }
        };
    let flush_code = |out: &mut Vec<Vec<Span<'static>>>,
                      code: &Vec<String>,
                      lang: &Option<String>| {
        let syntax = lang
            .as_deref()
            .and_then(|l| syntax_set().find_syntax_by_token(&l.to_lowercase()))
            .unwrap_or_else(|| syntax_set().find_syntax_plain_text());
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