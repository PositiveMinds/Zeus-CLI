//! Terminal presentation for the non-interactive/piped REPL fallback: a
//! clean colorized chat interface. Skipped entirely when stdout isn't a real
//! terminal (piped or redirected — this matters a lot here, since
//! testing/scripting against `zeus` routinely pipes its stdout through
//! `grep`/`cat`/log files) or when `NO_COLOR` is set. Never assume a TTY.
//!
//! The interactive-terminal experience itself lives in `tui.rs` (a full
//! ratatui interface), which doesn't use these helpers — they're anstyle/
//! plain-ANSI, not ratatui styles.

use anstyle::{AnsiColor, Reset, Style};
use std::io::IsTerminal;

/// Pure decision behind `supports_fancy_output`, split out so the TTY/NO_COLOR
/// logic is unit-testable without a real terminal.
fn should_style(is_terminal: bool, no_color: bool) -> bool {
    is_terminal && !no_color
}

pub(crate) fn supports_fancy_output() -> bool {
    should_style(
        std::io::stdout().is_terminal(),
        std::env::var_os("NO_COLOR").is_some(),
    )
}

pub fn prompt_style() -> Style {
    Style::new()
        .fg_color(Some(AnsiColor::BrightGreen.into()))
        .bold()
}

pub fn assistant_marker_style() -> Style {
    Style::new().fg_color(Some(AnsiColor::Green.into())).bold()
}

pub fn tool_style() -> Style {
    Style::new().fg_color(Some(AnsiColor::Cyan.into()))
}

pub fn error_style() -> Style {
    Style::new().fg_color(Some(AnsiColor::Red.into()))
}

pub fn warn_style() -> Style {
    Style::new().fg_color(Some(AnsiColor::Yellow.into()))
}

pub fn dim_style() -> Style {
    Style::new().dimmed()
}

/// Wrap `text` in `style`'s ANSI codes, or return it unchanged when fancy
/// output isn't supported (piped output, `NO_COLOR`, non-TTY).
pub fn styled(style: Style, text: &str) -> String {
    if supports_fancy_output() {
        format!("{}{text}{}", style.render(), Reset.render())
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_style_requires_terminal_and_no_no_color() {
        assert!(should_style(true, false));
        assert!(!should_style(false, false));
        assert!(!should_style(true, true));
        assert!(!should_style(false, true));
    }

    #[test]
    fn styled_returns_plain_when_stdout_is_not_a_terminal() {
        // Under the test harness stdout is captured (never a TTY), so the
        // fancy gate must be off and `styled` must pass the text through
        // untouched — the same condition a piped/scripted session hits.
        assert_eq!(styled(dim_style(), "hello"), "hello");
        assert_eq!(styled(error_style(), "boom"), "boom");
    }
}
