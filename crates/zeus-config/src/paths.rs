//! Global and project filesystem layout.
//!
//! ```text
//! ~/.zeus/
//! ├── config.toml
//! ├── providers.toml
//! ├── settings.toml
//! ├── settings.local.toml
//! ├── sessions/
//! ├── memory/
//! ├── cache/
//! ├── logs/
//! ├── plugins/
//! ├── commands/
//! └── prompts/
//!
//! my-project/.agent/
//! ├── config.toml
//! ├── settings.toml
//! ├── settings.local.toml
//! ├── memory.md
//! ├── tasks.json
//! ├── index.json
//! ├── hooks/
//! ├── commands/
//! └── checkpoints/
//! ```

use crate::error::{ConfigError, Result};
use std::path::{Path, PathBuf};

/// Override via `ZEUS_HOME` (useful for tests and portable installs).
pub fn global_home() -> Result<PathBuf> {
    if let Ok(override_home) = std::env::var("ZEUS_HOME") {
        return Ok(PathBuf::from(override_home));
    }
    let home = dirs::home_dir().ok_or(ConfigError::NoHomeDir)?;
    Ok(home.join(".zeus"))
}

/// Paths under `~/.zeus/`.
#[derive(Debug, Clone)]
pub struct GlobalPaths {
    pub root: PathBuf,
    pub config_toml: PathBuf,
    pub providers_toml: PathBuf,
    pub settings_toml: PathBuf,
    pub settings_local_toml: PathBuf,
    pub sessions: PathBuf,
    pub memory: PathBuf,
    pub cache: PathBuf,
    pub logs: PathBuf,
    pub plugins: PathBuf,
    pub commands: PathBuf,
    pub prompts: PathBuf,
    /// Custom specialist-agent personas (`~/.zeus/personas/*.toml`), merged
    /// into the built-in roster at load time.
    pub personas: PathBuf,
    /// Downloaded model files (GGUF, etc.) land here — scanned by
    /// `zeus models --local` so downloaded-but-not-yet-served models are
    /// still discoverable without any server running.
    pub models: PathBuf,
}

impl GlobalPaths {
    pub fn discover() -> Result<Self> {
        Ok(Self::from_root(global_home()?))
    }

    pub fn from_root(root: PathBuf) -> Self {
        Self {
            config_toml: root.join("config.toml"),
            providers_toml: root.join("providers.toml"),
            settings_toml: root.join("settings.toml"),
            settings_local_toml: root.join("settings.local.toml"),
            sessions: root.join("sessions"),
            memory: root.join("memory"),
            cache: root.join("cache"),
            logs: root.join("logs"),
            plugins: root.join("plugins"),
            commands: root.join("commands"),
            prompts: root.join("prompts"),
            personas: root.join("personas"),
            models: root.join("models"),
            root,
        }
    }
}

/// Paths under `<project>/.agent/`.
#[derive(Debug, Clone)]
pub struct ProjectPaths {
    pub root: PathBuf,
    pub config_toml: PathBuf,
    pub settings_toml: PathBuf,
    pub settings_local_toml: PathBuf,
    pub memory_md: PathBuf,
    pub tasks_json: PathBuf,
    pub index_json: PathBuf,
    pub hooks: PathBuf,
    pub commands: PathBuf,
    pub checkpoints: PathBuf,
}

impl ProjectPaths {
    pub fn from_root(project_root: &Path) -> Self {
        let root = project_root.join(".agent");
        Self {
            config_toml: root.join("config.toml"),
            settings_toml: root.join("settings.toml"),
            settings_local_toml: root.join("settings.local.toml"),
            memory_md: root.join("memory.md"),
            tasks_json: root.join("tasks.json"),
            index_json: root.join("index.json"),
            hooks: root.join("hooks"),
            commands: root.join("commands"),
            checkpoints: root.join("checkpoints"),
            root,
        }
    }
}

/// Ensure global directory tree exists.
pub fn ensure_global_dirs(paths: &GlobalPaths) -> Result<()> {
    for dir in [
        &paths.root,
        &paths.sessions,
        &paths.memory,
        &paths.cache,
        &paths.logs,
        &paths.plugins,
        &paths.commands,
        &paths.prompts,
        &paths.personas,
        &paths.models,
    ] {
        std::fs::create_dir_all(dir).map_err(ConfigError::Io)?;
    }
    Ok(())
}

/// Project `.agent/` directory helper.
pub fn project_agent_dir(project_root: &Path) -> PathBuf {
    project_root.join(".agent")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn ensure_global_dirs_creates_tree() {
        let tmp = TempDir::new().unwrap();
        let paths = GlobalPaths::from_root(tmp.path().join("agent-home"));
        ensure_global_dirs(&paths).unwrap();
        assert!(paths.logs.is_dir());
        assert!(paths.sessions.is_dir());
        assert!(paths.plugins.is_dir());
        assert!(paths.models.is_dir());
    }

    #[test]
    fn zeus_home_override() {
        let tmp = TempDir::new().unwrap();
        std::env::set_var("ZEUS_HOME", tmp.path());
        let home = global_home().unwrap();
        assert_eq!(home, tmp.path());
        std::env::remove_var("ZEUS_HOME");
    }
}
