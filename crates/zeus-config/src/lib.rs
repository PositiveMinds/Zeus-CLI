//! Layered configuration for zeus.
//!
//! Resolution order (highest precedence last):
//! global `settings.toml` → project `settings.toml` → project `settings.local.toml`
//!
//! Filesystem is the source of truth; no application database.

mod paths;
mod providers;
mod settings;
mod error;
mod keys;

pub use error::{ConfigError, Result};
pub use keys::KeysFile;
pub use paths::{ensure_global_dirs, global_home, project_agent_dir, GlobalPaths, ProjectPaths};
pub use providers::{ProviderConfig, ProvidersFile};
pub use settings::{
    set_accent_color, set_notify_on_completion, set_reduced_motion, AgentSettings,
    LlamaCppSettings, LocalModelEntry, McpServerConfig, PermissionDefault, PermissionRule,
    PermissionState, SettingsLayer, SettingsStack,
};

use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Fully resolved runtime configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub global: GlobalPaths,
    pub project: Option<ProjectPaths>,
    pub settings: AgentSettings,
    pub providers: ProvidersFile,
    /// Absolute path to the project root if detected.
    pub project_root: Option<PathBuf>,
}

impl Config {
    /// Load config for an optional project root.
    ///
    /// If `project_root` is `None`, only global config is used.
    pub fn load(project_root: Option<&Path>) -> Result<Self> {
        let global = GlobalPaths::discover()?;
        ensure_global_dirs(&global)?;

        let mut providers = ProvidersFile::load(&global.providers_toml)?;
        let keys = KeysFile::load(&global.keys_toml)?;
        providers.inject_keys(&keys);

        let mut stack = SettingsStack::new();
        stack.push_file(SettingsLayer::Global, &global.settings_toml)?;

        let project = project_root.map(ProjectPaths::from_root);
        if let Some(ref proj) = project {
            stack.push_file(SettingsLayer::Project, &proj.settings_toml)?;
            stack.push_file(SettingsLayer::Local, &proj.settings_local_toml)?;
        }

        let settings = stack.resolve();
        debug!(
            layers = ?stack.loaded_layers(),
            "resolved agent settings"
        );

        Ok(Self {
            global,
            project,
            settings,
            providers,
            project_root: project_root.map(|p| p.to_path_buf()),
        })
    }

    /// Load using the current working directory as the project root candidate.
    pub fn load_from_cwd() -> Result<Self> {
        let cwd = std::env::current_dir().map_err(ConfigError::Io)?;
        let root = find_project_root(&cwd);
        if let Some(ref r) = root {
            info!(project_root = %r.display(), "detected project root");
        }
        Self::load(root.as_deref())
    }

    /// Write default global config files if they do not already exist.
    pub fn init_global() -> Result<GlobalPaths> {
        let global = GlobalPaths::discover()?;
        ensure_global_dirs(&global)?;
        AgentSettings::write_defaults_if_missing(&global.settings_toml)?;
        ProvidersFile::write_defaults_if_missing(&global.providers_toml)?;
        // Keep config.toml as a small identity/home marker.
        if !global.config_toml.exists() {
            let body = r#"# zeus global config
# See settings.toml for permissions and behavior.
# See providers.toml for model providers.

[agent]
name = "zeus"
version = "0.1.0"
"#;
            std::fs::write(&global.config_toml, body).map_err(ConfigError::Io)?;
        }
        Ok(global)
    }

    /// Initialize `.agent/` in the given project root with default files.
    pub fn init_project(project_root: &Path) -> Result<ProjectPaths> {
        let paths = ProjectPaths::from_root(project_root);
        std::fs::create_dir_all(&paths.root).map_err(ConfigError::Io)?;
        std::fs::create_dir_all(&paths.hooks).map_err(ConfigError::Io)?;
        std::fs::create_dir_all(&paths.commands).map_err(ConfigError::Io)?;
        std::fs::create_dir_all(&paths.checkpoints).map_err(ConfigError::Io)?;

        AgentSettings::write_project_defaults_if_missing(&paths.settings_toml)?;
        if !paths.memory_md.exists() {
            std::fs::write(
                &paths.memory_md,
                "# Project memory\n\n<!-- Durable facts only. Ephemeral task state goes in tasks.json. -->\n",
            )
            .map_err(ConfigError::Io)?;
        }
        if !paths.tasks_json.exists() {
            std::fs::write(&paths.tasks_json, "[]\n").map_err(ConfigError::Io)?;
        }
        if !paths.index_json.exists() {
            std::fs::write(&paths.index_json, "{}\n").map_err(ConfigError::Io)?;
        }
        if !paths.instructions_md.exists() {
            std::fs::write(
                &paths.instructions_md,
                "# Project Rules\n\n<!-- Project-specific conventions and constraints. The agent reads this \
before any task. Fill in what applies to this repo; delete the rest. -->\n\n- Add new code in the project's \
existing language(s) and structure.\n- Never modify generated files.\n- Run the project's test suite after backend \
changes.\n- Use the project's existing error types and patterns.\n",
            )
            .map_err(ConfigError::Io)?;
        }
        // settings.local.toml is intentionally not created by default (gitignored personal overrides).
        Ok(paths)
    }
}

/// Walk up from `start` looking for `.agent/` or `.git/` to identify the
/// project root. Falls back to `start` itself if neither is found anywhere
/// above it — treating "wherever you ran zeus from" as an ad-hoc project
/// root, rather than refusing to work at all. This is safe: every settings
/// file lookup already tolerates a missing `.agent/` (falls back to
/// global/builtin defaults), and file operations are still contained to
/// this root exactly as if `.agent/` existed — nothing here loosens path
/// containment, it only decides *which* directory that containment applies
/// to when no explicit marker is present.
pub fn find_project_root(start: &Path) -> Option<PathBuf> {
    let mut current = Some(start);
    while let Some(dir) = current {
        if dir.join(".agent").is_dir() || dir.join(".git").is_dir() {
            return Some(dir.to_path_buf());
        }
        current = dir.parent();
    }
    Some(start.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn find_project_root_detects_git() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir(tmp.path().join(".git")).unwrap();
        let nested = tmp.path().join("src").join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        let root = find_project_root(&nested).unwrap();
        assert_eq!(root, tmp.path());
    }

    #[test]
    fn find_project_root_falls_back_to_start_when_nothing_found() {
        // No .git/.agent under this fresh tempdir — should return the
        // starting directory itself rather than None, so `zeus` works in a
        // plain, uninitialized directory instead of refusing to run.
        //
        // One caveat: on Windows the OS temp dir lives inside the user's
        // home (e.g. C:\Users\<name>\AppData\Local\Temp), so walking up may
        // cross a real home `.agent`/`.git`. In that case the lookup *should*
        // resolve to that ancestor project, not the temp dir — so we only
        // assert the strict fallback when no marker naturally sits between
        // the temp dir and the filesystem root.
        let tmp = TempDir::new().unwrap();
        let isolated = tmp.path().join("no_markers_here");
        std::fs::create_dir_all(&isolated).unwrap();

        let ancestor_marker = (0..64)
            .scan(Some(isolated.clone()), |acc, _| {
                let current = acc.clone();
                *acc = current.as_ref().and_then(|c| c.parent().map(Path::to_path_buf));
                current
            })
            .any(|p| p.join(".agent").is_dir() || p.join(".git").is_dir());

        let found = find_project_root(&isolated);
        if ancestor_marker {
            // A real project/home root exists above the temp dir; trust the
            // walk-up over the naive fallback rather than asserting it.
            assert!(found.is_some());
        } else {
            assert_eq!(found, Some(isolated));
        }
    }

    #[test]
    fn load_global_only_when_no_project() {
        let tmp = TempDir::new().unwrap();
        let home = tmp.path().join("home");
        std::env::set_var("ZEUS_HOME", &home);
        let cfg = Config::load(None).unwrap();
        assert!(cfg.project.is_none());
        assert!(cfg.global.root.exists());
        std::env::remove_var("ZEUS_HOME");
    }
}
