//! Minimal line-based diff for approval previews.
//!
//! Not a general-purpose diffing library — just enough to show a user what
//! will actually change before they approve a write/edit/delete, per the
//! blueprint's "show exactly what will change" requirement. Output is capped
//! so a huge file doesn't flood an interactive prompt.

const MAX_INPUT_LINES: usize = 500;
const MAX_PREVIEW_LINES: usize = 60;
/// Number of unchanged context lines shown before/after each change hunk.
const CONTEXT_LINES: usize = 2;

/// Render a capped, line-based diff between `old` and `new` content.
/// Shows context lines around changes and original line numbers.
pub fn preview_diff(old: &str, new: &str) -> String {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    if old_lines.len() > MAX_INPUT_LINES || new_lines.len() > MAX_INPUT_LINES {
        return format!(
            "(diff omitted — {} → {} lines, too large to preview)",
            old_lines.len(),
            new_lines.len()
        );
    }

    let hunks = diff_with_line_numbers(&old_lines, &new_lines);
    if hunks.is_empty() {
        return "(no line-level changes)".to_string();
    }

    let mut out = String::new();
    let mut total_lines = 0usize;
    for hunk in &hunks {
        for line in &hunk.lines {
            if total_lines >= MAX_PREVIEW_LINES {
                let remaining = hunks.iter().map(|h| h.lines.len()).sum::<usize>() - total_lines;
                if remaining > 0 {
                    out.push_str(&format!("… ({remaining} more changed line(s) not shown)\n"));
                }
                return out.trim_end().to_string();
            }
            out.push_str(line);
            out.push('\n');
            total_lines += 1;
        }
    }
    out.trim_end().to_string()
}

/// Render a concise edit-range preview for the `edit` tool: shows which line
/// range changed, with a few context lines. More compact than full diff for
/// targeted string replacements.
pub fn edit_range_preview(old: &str, new: &str, old_string: &str) -> String {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    // Find the line range of the first match in old_lines.
    let match_start = old_lines
        .iter()
        .position(|line| line.contains(old_string))
        .unwrap_or(0);
    let match_lines = old_string.lines().count().max(1);
    let match_end = (match_start + match_lines).min(old_lines.len());

    // Show context around the change.
    let ctx_start = match_start.saturating_sub(CONTEXT_LINES);
    let ctx_end = (match_end + CONTEXT_LINES).min(old_lines.len());

    let mut out = String::new();
    out.push_str(&format!("  lines {}–{}:\n", ctx_start + 1, ctx_end));

    for (i, line) in old_lines.iter().enumerate().take(ctx_end).skip(ctx_start) {
        if i >= match_start && i < match_end {
            out.push_str(&format!("  {:>4} - {}\n", i + 1, line));
        } else {
            out.push_str(&format!("  {:>4}   {}\n", i + 1, line));
        }
    }
    out.push_str("  →\n");

    // Show the new lines that replace the match.
    let end = (match_start + match_lines).min(new_lines.len());
    for (i, line) in new_lines.iter().enumerate().take(end).skip(match_start) {
        out.push_str(&format!("  {:>4} + {}\n", i + 1, line));
    }

    out.trim_end().to_string()
}

/// A hunk of diff output with line numbers.
struct Hunk {
    lines: Vec<String>,
}

/// LCS-based line diff that produces hunks with line numbers and context.
fn diff_with_line_numbers(old: &[&str], new: &[&str]) -> Vec<Hunk> {
    let n = old.len();
    let m = new.len();

    // Build LCS table.
    let mut lcs = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if old[i] == new[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }

    // Extract changed line positions (old_idx, new_idx, is_deletion).
    let mut changes: Vec<(Option<usize>, Option<usize>, bool)> = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if old[i] == new[j] {
            i += 1;
            j += 1;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            changes.push((Some(i), None, true));
            i += 1;
        } else {
            changes.push((None, Some(j), false));
            j += 1;
        }
    }
    while i < n {
        changes.push((Some(i), None, true));
        i += 1;
    }
    while j < m {
        changes.push((None, Some(j), false));
        j += 1;
    }
    if changes.is_empty() {
        return Vec::new();
    }

    // Group changes into hunks with context.
    let mut hunks: Vec<Hunk> = Vec::new();
    let mut current_lines: Vec<String> = Vec::new();
    let mut last_old_end: usize = 0;
    let mut _last_new_end: usize = 0;

    for (old_idx, new_idx, is_del) in &changes {
        let old_line = old_idx.map(|i| i + 1).unwrap_or(0);
        let new_line = new_idx.map(|j| j + 1).unwrap_or(0);
        let pos = old_line.max(new_line);

        // Add context before this change if there's a gap.
        let context_start = pos.saturating_sub(CONTEXT_LINES + 1);
        if pos > last_old_end + 1 && !current_lines.is_empty() {
            // End current hunk, start new one with context.
            hunks.push(Hunk {
                lines: current_lines.clone(),
            });
            current_lines.clear();
            // Add leading context to new hunk.
            for ci in context_start..pos.saturating_sub(1).min(n) {
                if ci < old.len() {
                    current_lines.push(format!("  {:>4}   {}", ci + 1, old[ci]));
                }
            }
        } else if current_lines.is_empty() {
            // First change in a new hunk — add leading context.
            for ci in context_start..pos.saturating_sub(1).min(n) {
                if ci < old.len() {
                    current_lines.push(format!("  {:>4}   {}", ci + 1, old[ci]));
                }
            }
        }

        if *is_del {
            if let Some(idx) = old_idx {
                current_lines.push(format!("- {:>4}   {}", idx + 1, old[*idx]));
                last_old_end = idx + 1;
            }
        } else if let Some(idx) = new_idx {
            current_lines.push(format!("+ {:>4}   {}", idx + 1, new[*idx]));
            _last_new_end = idx + 1;
        }
    }

    // Add trailing context.
    let trailing_start = last_old_end;
    let trailing_end = (trailing_start + CONTEXT_LINES).min(n);
    for (ci, line) in old
        .iter()
        .enumerate()
        .take(trailing_end)
        .skip(trailing_start)
    {
        current_lines.push(format!("  {:>4}   {}", ci + 1, line));
    }

    if !current_lines.is_empty() {
        hunks.push(Hunk {
            lines: current_lines,
        });
    }

    hunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shows_added_and_removed_lines() {
        let d = preview_diff("a\nb\nc", "a\nx\nc");
        assert!(d.contains("-"));
        assert!(d.contains("+"));
        assert!(d.contains("b") || d.contains("x"));
    }

    #[test]
    fn create_from_empty_shows_all_additions() {
        let d = preview_diff("", "one\ntwo");
        assert!(d.contains("+"));
        assert!(d.contains("one") || d.contains("two"));
    }

    #[test]
    fn delete_to_empty_shows_all_removals() {
        let d = preview_diff("one\ntwo", "");
        assert!(d.contains("-"));
        assert!(d.contains("one") || d.contains("two"));
    }

    #[test]
    fn identical_content_has_no_changes() {
        assert_eq!(preview_diff("same", "same"), "(no line-level changes)");
    }

    #[test]
    fn caps_huge_input() {
        let big = "x\n".repeat(MAX_INPUT_LINES + 1);
        let d = preview_diff(&big, "y");
        assert!(d.starts_with("(diff omitted"));
    }

    #[test]
    fn caps_preview_length() {
        let old: String = (0..MAX_PREVIEW_LINES + 10)
            .map(|i| format!("old{i}\n"))
            .collect();
        let new: String = (0..MAX_PREVIEW_LINES + 10)
            .map(|i| format!("new{i}\n"))
            .collect();
        let d = preview_diff(&old, &new);
        assert!(d.contains("more changed line(s) not shown"));
    }

    #[test]
    fn context_lines_surround_changes() {
        let old = "line1\nline2\nline3\nline4\nline5\nline6\nline7";
        let new = "line1\nline2\nCHANGED\nline4\nline5\nline6\nline7";
        let d = preview_diff(old, new);
        // Should show context lines around the change
        assert!(d.contains("line2"));
        assert!(d.contains("CHANGED"));
        assert!(d.contains("line4"));
    }

    #[test]
    fn edit_range_preview_shows_match_location() {
        let old = "alpha\nbeta\ngamma\ndelta";
        let new = "alpha\nBETA\ngamma\ndelta";
        let d = edit_range_preview(old, new, "beta");
        assert!(d.contains("beta") || d.contains("BETA"));
    }
}
