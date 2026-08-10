//! Decorative animation: a rainbow/pulse-swept wordmark. Pure-ratatui, no
//! extra deps — hue steps are computed from HSV→RGB per character, and the
//! overall brightness pulses on a slow sine so the wordmark looks alive
//! without flashing. Drives the topbar's "ZEUS" wordmark.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;

/// Convert an HSV triple (hue 0-360, s/v 0-1) to an RGB `Color`.
fn hsv(h: f32, s: f32, v: f32) -> Color {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0).floor() as i32 % 6 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let to = |v: f32| ((v + m) * 255.0).round() as u8;
    Color::Rgb(to(r), to(g), to(b))
}

/// Hue for a character at `t` milliseconds into the animation. Each column
/// advances ~12° so the whole wordmark carries a single moving rainbow sweep;
/// `pulse` (0..1) dims the whole thing on a slow sine for the pulse effect.
fn char_hue(t_ms: u128, col: usize) -> f32 {
    ((t_ms as f32 / 12.0) + (col as f32 * 12.0)) % 360.0
}

fn pulse_brightness(t_ms: u128) -> f32 {
    // 0.55..1.0 on a ~2.2s period: steady enough to read, alive enough to see.
    let s = ((t_ms as f32 / 1100.0) * std::f32::consts::TAU).sin();
    0.55 + 0.45 * ((s + 1.0) / 2.0)
}

/// A single character styled with its rainbow hue at the given animation time.
fn rainbow_char(ch: char, t_ms: u128, col: usize) -> Span<'static> {
    let hue = char_hue(t_ms, col);
    let v = pulse_brightness(t_ms);
    let mut style = Style::default().fg(hsv(hue, 0.75, v));
    // Peak brightness rows stand a touch bolder so the pulse is legible.
    if v > 0.88 {
        style = style.add_modifier(Modifier::BOLD);
    }
    Span::styled(ch.to_string(), style)
}

/// Rainbow+pulse styled wordmark for the topbar ("ZEUS" after the ⚡).
pub fn animated_wordmark(text: &str, t_ms: u128) -> Vec<Span<'static>> {
    let mut col = 0;
    text.chars()
        .map(|c| {
            let s = rainbow_char(c, t_ms, col);
            col += 1;
            s
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hue_always_in_bounds() {
        for ms in [0u128, 100, 3333, 55_555] {
            for col in 0..40 {
                let hue = char_hue(ms, col);
                assert!((0.0..360.0).contains(&hue));
            }
        }
    }

    #[test]
    fn brightness_stays_readable() {
        for ms in (0..10_000).step_by(137) {
            let v = pulse_brightness(ms);
            assert!(v > 0.5 && v <= 1.0, "v={v}");
        }
    }

}
