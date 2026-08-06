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

fn supports_fancy_output() -> bool {
    std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
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
