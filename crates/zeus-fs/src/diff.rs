//! Minimal line-based diff for approval previews.
//!
//! Not a general-purpose diffing library — just enough to show a user what
//! will actually change before they approve a write/edit/delete, per the
//! blueprint's "show exactly what will change" requirement. Output is capped
//! so a huge file doesn't flood an interactive prompt.

const MAX_INPUT_LINES: usize = 500;
const MAX_PREVIEW_LINES: usize = 40;

/// Render a capped, line-based diff between `old` and `new` content.
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

    let changes = line_diff(&old_lines, &new_lines);
    if changes.is_empty() {
        return "(no line-level changes)".to_string();
    }
    if changes.len() > MAX_PREVIEW_LINES {
        let shown = changes[..MAX_PREVIEW_LINES].join("\n");
        format!(
            "{shown}\n… ({} more changed line(s) not shown)",
            changes.len() - MAX_PREVIEW_LINES
        )
    } else {
        changes.join("\n")
    }
}

/// Classic LCS-based line diff, returning `+`/`-` prefixed changed lines only
/// (unchanged context lines are omitted to keep previews short).
fn line_diff(old: &[&str], new: &[&str]) -> Vec<String> {
    let n = old.len();
    let m = new.len();
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

    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if old[i] == new[j] {
            i += 1;
            j += 1;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            out.push(format!("- {}", old[i]));
            i += 1;
        } else {
            out.push(format!("+ {}", new[j]));
            j += 1;
        }
    }
    while i < n {
        out.push(format!("- {}", old[i]));
        i += 1;
    }
    while j < m {
        out.push(format!("+ {}", new[j]));
        j += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shows_added_and_removed_lines() {
        let d = preview_diff("a\nb\nc", "a\nx\nc");
        assert_eq!(d, "- b\n+ x");
    }

    #[test]
    fn create_from_empty_shows_all_additions() {
        let d = preview_diff("", "one\ntwo");
        assert_eq!(d, "+ one\n+ two");
    }

    #[test]
    fn delete_to_empty_shows_all_removals() {
        let d = preview_diff("one\ntwo", "");
        assert_eq!(d, "- one\n- two");
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
}
