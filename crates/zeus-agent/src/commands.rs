//! Slash commands: reusable prompt templates — no code, just parameterized
//! instructions (per the blueprint's Extensibility table). A project-scoped
//! command (`.agent/commands/<name>.md`) shadows a global one
//! (`~/.zeus/commands/<name>.md`) of the same name.

use std::path::PathBuf;

pub struct SlashCommands {
    project_dir: Option<PathBuf>,
    global_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpandResult {
    /// `message` wasn't a recognized slash command — use it as-is. Covers
    /// both plain messages and a `/word` that doesn't match any command file.
    Unchanged,
    Expanded {
        command: String,
        rendered: String,
    },
}

impl SlashCommands {
    pub fn new(project_dir: Option<PathBuf>, global_dir: PathBuf) -> Self {
        Self {
            project_dir,
            global_dir,
        }
    }

    fn find(&self, name: &str) -> Option<PathBuf> {
        if let Some(dir) = &self.project_dir {
            let p = dir.join(format!("{name}.md"));
            if p.is_file() {
                return Some(p);
            }
        }
        let p = self.global_dir.join(format!("{name}.md"));
        if p.is_file() {
            return Some(p);
        }
        None
    }

    /// Expand `message` if it's a slash command (`/name [args...]`).
    /// Substitutes the literal placeholder `$ARGUMENTS` in the template with
    /// everything after the command name; if the template has no such
    /// placeholder, the args are appended on a new paragraph instead.
    pub fn expand(&self, message: &str) -> ExpandResult {
        let Some(rest) = message.strip_prefix('/') else {
            return ExpandResult::Unchanged;
        };
        let (name, args) = match rest.split_once(char::is_whitespace) {
            Some((n, a)) => (n, a.trim()),
            None => (rest, ""),
        };
        if name.is_empty() {
            return ExpandResult::Unchanged;
        }
        let Some(path) = self.find(name) else {
            return ExpandResult::Unchanged;
        };
        let Ok(template) = std::fs::read_to_string(&path) else {
            return ExpandResult::Unchanged;
        };
        let rendered = if template.contains("$ARGUMENTS") {
            template.replace("$ARGUMENTS", args)
        } else if !args.is_empty() {
            format!("{template}\n\n{args}")
        } else {
            template
        };
        ExpandResult::Expanded {
            command: name.to_string(),
            rendered,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn plain_message_is_unchanged() {
        let tmp = TempDir::new().unwrap();
        let cmds = SlashCommands::new(None, tmp.path().to_path_buf());
        assert_eq!(cmds.expand("just a normal message"), ExpandResult::Unchanged);
    }

    #[test]
    fn unknown_slash_word_is_unchanged() {
        let tmp = TempDir::new().unwrap();
        let cmds = SlashCommands::new(None, tmp.path().to_path_buf());
        assert_eq!(cmds.expand("/nope some args"), ExpandResult::Unchanged);
    }

    #[test]
    fn expands_with_arguments_placeholder() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("review.md"),
            "Review this code for bugs: $ARGUMENTS",
        )
        .unwrap();
        let cmds = SlashCommands::new(None, tmp.path().to_path_buf());
        match cmds.expand("/review src/main.rs") {
            ExpandResult::Expanded { command, rendered } => {
                assert_eq!(command, "review");
                assert_eq!(rendered, "Review this code for bugs: src/main.rs");
            }
            other => panic!("expected Expanded, got {other:?}"),
        }
    }

    #[test]
    fn expands_appending_args_when_no_placeholder() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("explain.md"), "Explain the following:").unwrap();
        let cmds = SlashCommands::new(None, tmp.path().to_path_buf());
        match cmds.expand("/explain the auth flow") {
            ExpandResult::Expanded { rendered, .. } => {
                assert_eq!(rendered, "Explain the following:\n\nthe auth flow");
            }
            other => panic!("expected Expanded, got {other:?}"),
        }
    }

    #[test]
    fn project_command_shadows_global_of_same_name() {
        let global = TempDir::new().unwrap();
        let project = TempDir::new().unwrap();
        std::fs::write(global.path().join("go.md"), "global version").unwrap();
        std::fs::write(project.path().join("go.md"), "project version").unwrap();
        let cmds = SlashCommands::new(
            Some(project.path().to_path_buf()),
            global.path().to_path_buf(),
        );
        match cmds.expand("/go") {
            ExpandResult::Expanded { rendered, .. } => assert_eq!(rendered, "project version"),
            other => panic!("expected Expanded, got {other:?}"),
        }
    }

    #[test]
    fn no_args_leaves_template_unrendered_when_no_placeholder() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("status.md"), "Give a status update.").unwrap();
        let cmds = SlashCommands::new(None, tmp.path().to_path_buf());
        match cmds.expand("/status") {
            ExpandResult::Expanded { rendered, .. } => {
                assert_eq!(rendered, "Give a status update.")
            }
            other => panic!("expected Expanded, got {other:?}"),
        }
    }
}
