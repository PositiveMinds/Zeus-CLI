//! Terminal color palette and runtime-adjustable display settings for the
//! interactive TUI — the zeus-cli.html palette, variables taken 1:1 from the
//! reference page's `:root` block (`--void` … `--red`) plus the mode accent
//! colors, so the TUI reproduces the HTML mockup exactly. Extracted from the
//! inline `mod theme` in `tui.rs`; referenced by `tui.rs` and `tui_text.rs`.
//!
//! Three built-in presets (Dark, Light, HighContrast) swap the *surface*
//! palette — backgrounds, borders, text tiers, selection, wordmark gradient —
//! as one unit via `/theme`; the brand accent colors (violet/cyan/gold/
//! magenta/green/red) stay fixed so mode and status read identically on any
//! theme. `ThemeKind::Dark` is the reference HTML palette.

use ratatui::style::{Color, Style};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};

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
/// Transcript text-selection highlight — the same warm bar the popup rows
/// use, so "selected" reads consistently across the app whether you're
/// picking a model or copying messages with ctrl+y.
pub const SELECTED_BG: Color = Color::Rgb(0xd9, 0x8a, 0x4f);
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
/// Wordmark gradient stops — near-white → `--gold-soft` → `--gold`,
/// matching the HTML's `linear-gradient(100deg, #eef3ff 0%, #f6d98a 52%, #f2c661 100%)`.
pub const WORDMARK_START: Color = Color::Rgb(0xee, 0xf3, 0xff);
pub const WORDMARK_MID: Color = Color::Rgb(0xf6, 0xd9, 0x8a);
pub const WORDMARK_END: Color = Color::Rgb(0xf2, 0xc6, 0x61);

/// The complete switchable surface palette for one theme preset. Brand
/// accents (violet/cyan/gold/magenta/green/red) are deliberately *not*
/// members — they stay constant across themes so mode/status color
/// semantics don't drift.
#[derive(Clone, Copy)]
pub struct Palette {
    pub void: Color,
    pub panel: Color,
    pub panel2: Color,
    pub ink: Color,
    pub modal_bg: Color,
    pub modal_selected_bg: Color,
    pub selected_bg: Color,
    pub card_bg: Color,
    pub border: Color,
    pub border_soft: Color,
    pub text: Color,
    pub dim: Color,
    pub faint: Color,
    pub muted: Color,
    pub empty_faint: Color,
    pub user_text: Color,
    pub teal: Color,
    pub wordmark_start: Color,
    pub wordmark_mid: Color,
    pub wordmark_end: Color,
}

/// The default Dark preset — the reference HTML palette, i.e. the
/// original constants above unchanged.
const DARK: Palette = Palette {
    void: VOID,
    panel: PANEL,
    panel2: PANEL2,
    ink: INK,
    modal_bg: MODAL_BG,
    modal_selected_bg: MODAL_SELECTED_BG,
    selected_bg: SELECTED_BG,
    card_bg: CARD_BG,
    border: BORDER,
    border_soft: BORDER_SOFT,
    text: TEXT,
    dim: DIM,
    faint: FAINT,
    muted: MUTED,
    empty_faint: EMPTY_FAINT,
    user_text: Color::Rgb(0xc7, 0xca, 0xdb),
    teal: TEAL,
    wordmark_start: WORDMARK_START,
    wordmark_mid: WORDMARK_MID,
    wordmark_end: WORDMARK_END,
};

/// A light-on-dark-ink inverted preset for bright terminals / daylight
/// work — the same design tokens, remapped so backgrounds are near-white,
/// text near-black, and the accent-leaning surfaces (selection, wordmark,
/// empty-state teal/cyan) get darker variants that stay legible on white.
const LIGHT: Palette = Palette {
    void: Color::Rgb(0xf6, 0xf7, 0xfb),
    panel: Color::Rgb(0xed, 0xef, 0xf5),
    panel2: Color::Rgb(0xe5, 0xe7, 0xf0),
    ink: Color::Rgb(0xfb, 0xfc, 0xfe),
    modal_bg: Color::Rgb(0xff, 0xff, 0xff),
    modal_selected_bg: Color::Rgb(0xc8, 0x74, 0x3a),
    selected_bg: Color::Rgb(0xc8, 0x74, 0x3a),
    card_bg: Color::Rgb(0xe4, 0xe7, 0xf0),
    border: Color::Rgb(0xc5, 0xca, 0xd8),
    border_soft: Color::Rgb(0xd7, 0xdb, 0xe6),
    text: Color::Rgb(0x1c, 0x20, 0x30),
    dim: Color::Rgb(0x5a, 0x61, 0x73),
    faint: Color::Rgb(0x8a, 0x90, 0xa3),
    muted: Color::Rgb(0x4c, 0x56, 0x70),
    empty_faint: Color::Rgb(0x6a, 0x73, 0x90),
    user_text: Color::Rgb(0x2a, 0x2f, 0x3f),
    teal: Color::Rgb(0x0e, 0x8f, 0x7c),
    wordmark_start: Color::Rgb(0x14, 0x18, 0x29),
    wordmark_mid: Color::Rgb(0xa8, 0x6a, 0x1f),
    wordmark_end: Color::Rgb(0x8a, 0x56, 0x18),
};

/// Maximum-contrast preset for low-vision / sunlight — pure black
/// backgrounds, pure white primary text, and the strongest-separation
/// versions of every secondary tier (dim/faint/border all pushed well
/// above the ~3:1 floor rather than the ~2:1 the Dark palette allows).
const HIGH_CONTRAST: Palette = Palette {
    void: Color::Rgb(0x00, 0x00, 0x00),
    panel: Color::Rgb(0x0a, 0x0a, 0x0a),
    panel2: Color::Rgb(0x15, 0x15, 0x15),
    ink: Color::Rgb(0x00, 0x00, 0x00),
    modal_bg: Color::Rgb(0x00, 0x00, 0x00),
    modal_selected_bg: Color::Rgb(0xff, 0xcc, 0x66),
    selected_bg: Color::Rgb(0xff, 0xcc, 0x66),
    card_bg: Color::Rgb(0x20, 0x20, 0x20),
    border: Color::Rgb(0xd0, 0xd0, 0xd0),
    border_soft: Color::Rgb(0x9a, 0x9a, 0x9a),
    text: Color::Rgb(0xff, 0xff, 0xff),
    dim: Color::Rgb(0xe0, 0xe0, 0xe0),
    faint: Color::Rgb(0xb0, 0xb0, 0xb0),
    muted: Color::Rgb(0xcf, 0xcf, 0xcf),
    empty_faint: Color::Rgb(0xa0, 0xa0, 0xa0),
    user_text: Color::Rgb(0xff, 0xff, 0xff),
    teal: Color::Rgb(0x00, 0xff, 0xbe),
    wordmark_start: Color::Rgb(0xff, 0xff, 0xff),
    wordmark_mid: Color::Rgb(0xff, 0xd7, 0x5e),
    wordmark_end: Color::Rgb(0xff, 0xb0, 0x20),
};

const PALETTES: [Palette; 3] = [DARK, LIGHT, HIGH_CONTRAST];

/// The active theme preset. Indexed into `PALETTES`; a plain atomic
/// (rather than a `OnceLock`) is what lets `/theme` change it at runtime
/// again, exactly like `ACCENT`.
static THEME_INDEX: AtomicU8 = AtomicU8::new(0);

/// The available theme presets, in `/theme` cycle order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeKind {
    Dark,
    Light,
    HighContrast,
}

impl ThemeKind {
    pub const ALL: [ThemeKind; 3] = [ThemeKind::Dark, ThemeKind::Light, ThemeKind::HighContrast];

    pub fn label(self) -> &'static str {
        match self {
            ThemeKind::Dark => "dark",
            ThemeKind::Light => "light",
            ThemeKind::HighContrast => "high-contrast",
        }
    }

    pub fn from_label(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|k| k.label() == s)
    }
}

/// The palette backing the current theme.
fn palette() -> &'static Palette {
    &PALETTES[THEME_INDEX.load(Ordering::Relaxed) as usize]
}

/// Switches the active theme preset immediately — applied on the very
/// next frame. `/theme <name>` behind this.
pub fn set_theme(kind: ThemeKind) {
    THEME_INDEX.store(kind as u8, Ordering::Relaxed);
}

/// The currently active theme preset.
pub fn current_theme() -> ThemeKind {
    ThemeKind::ALL[THEME_INDEX.load(Ordering::Relaxed) as usize]
}

pub fn void() -> Color {
    palette().void
}
pub fn panel() -> Color {
    palette().panel
}
pub fn panel2() -> Color {
    palette().panel2
}
pub fn ink() -> Color {
    palette().ink
}
pub fn modal_bg() -> Color {
    palette().modal_bg
}
pub fn modal_selected_bg() -> Color {
    palette().modal_selected_bg
}
pub fn selected_bg() -> Color {
    palette().selected_bg
}
pub fn card_bg() -> Color {
    palette().card_bg
}
pub fn border() -> Color {
    palette().border
}
pub fn border_soft() -> Color {
    palette().border_soft
}
pub fn text_color() -> Color {
    palette().text
}
pub fn dim_color() -> Color {
    palette().dim
}
pub fn faint_color() -> Color {
    palette().faint
}
pub fn muted_color() -> Color {
    palette().muted
}
pub fn empty_faint_color() -> Color {
    palette().empty_faint
}
pub fn user_text_color() -> Color {
    palette().user_text
}
pub fn teal_color() -> Color {
    palette().teal
}
pub fn wordmark_start() -> Color {
    palette().wordmark_start
}
pub fn wordmark_mid() -> Color {
    palette().wordmark_mid
}
pub fn wordmark_end() -> Color {
    palette().wordmark_end
}

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
pub fn init_runtime(
    accent_hex: Option<&str>,
    reduced_motion: bool,
    notify_on_completion: bool,
    theme: Option<&str>,
) {
    if let Some(color) = accent_hex.and_then(parse_hex_color) {
        set_accent(color);
    }
    REDUCED_MOTION.store(reduced_motion, Ordering::Relaxed);
    NOTIFY_ON_COMPLETION.store(notify_on_completion, Ordering::Relaxed);
    if let Some(kind) = theme.and_then(ThemeKind::from_label) {
        set_theme(kind);
    }
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
    Style::default().fg(muted_color())
}
pub fn empty_faint() -> Style {
    Style::default().fg(empty_faint_color())
}
pub fn teal() -> Style {
    Style::default().fg(teal_color())
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
    Style::default().fg(text_color())
}
pub fn dim() -> Style {
    Style::default().fg(dim_color())
}
pub fn faint() -> Style {
    Style::default().fg(faint_color())
}
/// User bubble text — the `user_text` palette entry (`#c7cadb` on Dark).
pub fn user_text() -> Style {
    Style::default().fg(user_text_color())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theme_kind_labels_roundtrip() {
        for kind in ThemeKind::ALL {
            assert_eq!(ThemeKind::from_label(kind.label()), Some(kind));
        }
        assert_eq!(ThemeKind::from_label("nope"), None);
        assert_eq!(ThemeKind::from_label(""), None);
    }

    #[test]
    fn set_theme_switches_palette_accessors() {
        set_theme(ThemeKind::Dark);
        assert_eq!(void(), VOID);
        assert_eq!(text_color(), TEXT);
        assert_eq!(user_text_color(), Color::Rgb(0xc7, 0xca, 0xdb));

        set_theme(ThemeKind::HighContrast);
        assert_eq!(void(), Color::Rgb(0x00, 0x00, 0x00));
        assert_eq!(text_color(), Color::Rgb(0xff, 0xff, 0xff));
        assert_eq!(selected_bg(), Color::Rgb(0xff, 0xcc, 0x66));

        set_theme(ThemeKind::Light);
        assert_eq!(void(), Color::Rgb(0xf6, 0xf7, 0xfb));
        assert_eq!(text_color(), Color::Rgb(0x1c, 0x20, 0x30));

        // Restore the default so other tests see a clean slate.
        set_theme(ThemeKind::Dark);
        assert_eq!(current_theme(), ThemeKind::Dark);
    }

    #[test]
    fn dark_palette_matches_reference_constants() {
        set_theme(ThemeKind::Dark);
        assert_eq!(panel(), PANEL);
        assert_eq!(panel2(), PANEL2);
        assert_eq!(ink(), INK);
        assert_eq!(modal_bg(), MODAL_BG);
        assert_eq!(modal_selected_bg(), MODAL_SELECTED_BG);
        assert_eq!(selected_bg(), SELECTED_BG);
        assert_eq!(card_bg(), CARD_BG);
        assert_eq!(border(), BORDER);
        assert_eq!(border_soft(), BORDER_SOFT);
        assert_eq!(dim_color(), DIM);
        assert_eq!(faint_color(), FAINT);
        assert_eq!(muted_color(), MUTED);
        assert_eq!(empty_faint_color(), EMPTY_FAINT);
        assert_eq!(teal_color(), TEAL);
        assert_eq!(wordmark_start(), WORDMARK_START);
        assert_eq!(wordmark_mid(), WORDMARK_MID);
        assert_eq!(wordmark_end(), WORDMARK_END);
    }
}
