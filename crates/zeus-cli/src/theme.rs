//! Terminal color palette and runtime-adjustable display settings for the
//! interactive TUI — the zeus-cli.html palette, variables taken 1:1 from the
//! reference page's `:root` block (`--void` … `--red`) plus the mode accent
//! colors, so the TUI reproduces the HTML mockup exactly. Extracted from the
//! inline `mod theme` in `tui.rs`; referenced by `tui.rs` and `tui_text.rs`.

use ratatui::style::{Color, Style};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// `--void`: page background (`#07080d`).
pub const VOID: Color = Color::Rgb(0x07, 0x08, 0x0d);
/// `--panel`: panel background (`#0d1017`).
pub const PANEL: Color = Color::Rgb(0x0d, 0x10, 0x17);
/// `--panel-2`: slightly lighter panel (`#11141c`).
pub const PANEL2: Color = Color::Rgb(0x11, 0x14, 0x1c);
/// Solid near-black background for every full-screen modal (model,
/// provider, key-entry, session picker, approval, command palette) —
/// `PANEL` (`#0d1017`) sits so close to `VOID` (`#07080d`) that those
/// modals read as translucent, with the transcript text behind them
/// still legible through the low-contrast fill. These are meant to be
/// fully opaque overlays (matching the reference product's own solid
/// black popup), so they get a
/// distinctly darker, higher-contrast fill instead of reusing `PANEL`.
pub const MODAL_BG: Color = Color::Rgb(0x00, 0x00, 0x00);
/// Full-width selected-row highlight for those same popups — a solid
/// warm bar instead of `PANEL2` + colored text, matching the reference
/// product's own selection style.
pub const MODAL_SELECTED_BG: Color = Color::Rgb(0xd9, 0x8a, 0x4f);
/// Tool-call card background (`Block_::tool_card_lines`) — deliberately
/// much lighter than `PANEL2`/`ELEVATED`, which read as barely-there
/// against the transcript's `VOID` background once real syntax-
/// highlighted or diff-tinted text sits on top of them (the same
/// too-subtle-to-read-as-a-surface mistake `MODAL_BG` already fixed
/// for the popups, just needing the opposite direction here — a card
/// embedded in a dark transcript needs to read as clearly *elevated*,
/// not clearly *opaque*).
pub const CARD_BG: Color = Color::Rgb(0x1c, 0x22, 0x30);
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
        Color::Rgb(
            ((packed >> 16) & 0xff) as u8,
            ((packed >> 8) & 0xff) as u8,
            (packed & 0xff) as u8,
        )
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
