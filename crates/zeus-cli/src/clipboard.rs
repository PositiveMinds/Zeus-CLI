//! Clipboard helpers: copy text/diffs to the system clipboard.

/// Copy `text` to the system clipboard. Returns a short human-readable error
/// when the platform clipboard is unavailable (e.g. headless SSH sessions).
pub fn copy(text: &str) -> Result<(), String> {
    let mut cb = arboard::Clipboard::new().map_err(|e| format!("clipboard unavailable: {e}"))?;
    cb.set_text(text.to_string())
        .map_err(|e| format!("copy failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_returns_result() {
        // Can't assert on clipboard success (headless), but it must not panic
        // and must return a Result.
        let _ = copy("test");
    }
}