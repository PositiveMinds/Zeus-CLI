//! Layered configuration for zeus.
//!
//! Resolution order (highest precedence last):
//! global `settings.toml` → project `settings.toml` → project `settings.local.toml`
//!
//! Filesystem is the source of truth; no application database.

mod error;
mod keys;
mod paths;
mod providers;
mod settings;

pub use error::{ConfigError, Result};
pub use keys::KeysFile;
pub use paths::{ensure_global_dirs, global_home, project_agent_dir, GlobalPaths, ProjectPaths};
pub use providers::{ProviderConfig, ProvidersFile};
pub use settings::{
    set_accent_color, set_notify_on_completion, set_reduced_motion, set_theme, AgentSettings,
    LlamaCppSettings, LocalModelEntry, McpServerConfig, PermissionDefault, PermissionRule,
    PermissionSettings, PermissionState, SettingsLayer, SettingsStack,
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

/// Resolve the project root a zeus session is scoped to.
///
/// Priority:
///  - A `.git`/`.agent` marker *directly at* `start` — the session is scoped
///    to exactly the real current directory.
///  - A git work tree containing `start` — `git rev-parse --show-toplevel`
///    names the authoritative project root. It cannot be a "false broader
///    environment", because the current directory genuinely lives inside that
///    repo.
///  - Otherwise `start` itself, treated as an ad-hoc project root rather than
///    refusing to work at all. This is safe: every settings file lookup
///    already tolerates a missing `.agent/` (falls back to global/builtin
///    defaults), and file operations are still contained to this root exactly
///    as if `.agent/` existed — nothing here loosens path containment, it only
///    decides *which* directory that containment applies to when no explicit
///    marker is present.
///
/// The resolution never silently widens into a broader environment on behalf
/// of a narrower starting directory:
///  - The user's home directory (or any ancestor of it, e.g. a drive root) is
///    never adopted as the root — even when git reports it as the toplevel of
///    a dotfiles repo. `$HOME` is user-land, so such a root would sweep in
///    every unrelated project living there.
///  - An ancestor's `.agent/` marker is *not* followed. `.agent/` gets
///    created unconditionally the moment any turn actually needs repo context
///    (see `zeus-agent`'s `repo_context`/`project::persist`) — so one earlier
///    bare `zeus` run directly in `$HOME` (or in some parent folder such as
///    Desktop) plants a marker there permanently, and honoring it from below
///    would widen every subfolder's project root up to that poisoned parent,
///    mixing in whatever unrelated projects happen to live there. Running
///    `zeus` directly in such a directory is still honored as-is (its own
///    marker is accepted); this only stops something *underneath* it from
///    climbing up into it.
///  - Ancestor `.git/` markers *are* honored — a real repo, which zeus itself
///    never plants — but the climb still stops before the home boundary.
pub fn find_project_root(start: &Path) -> Option<PathBuf> {
    if start.join(".agent").is_dir() || start.join(".git").is_dir() {
        return Some(start.to_path_buf());
    }
    let home = dirs::home_dir();
    let home_canon = home.as_ref().and_then(|h| h.canonicalize().ok());

    // Inside a git work tree, git knows the authoritative root — unless it
    // resolves to home (or above), which is never a legitimate project.
    if let Some(toplevel) = git_toplevel(start) {
        if !is_home_or_ancestor(&toplevel, home.as_deref(), home_canon.as_deref()) {
            return Some(toplevel);
        }
    }

    let mut current = start.parent();
    while let Some(dir) = current {
        if is_home_or_ancestor(dir, home.as_deref(), home_canon.as_deref()) {
            break;
        }
        if dir.join(".git").is_dir() {
            return Some(dir.to_path_buf());
        }
        current = dir.parent();
    }
    Some(start.to_path_buf())
}

/// Ask git for the authoritative project root (work-tree toplevel) of
/// `start`. `None` when `start` isn't inside a git work tree or git isn't
/// available — the caller then falls back to scoping on `start` itself.
fn git_toplevel(start: &Path) -> Option<PathBuf> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(start)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(PathBuf::from(trimmed))
}

/// True when `p` is the user's home directory or any ancestor of it (e.g. a
/// drive root). Compared canonically so case/separator differences (git on
/// Windows emits `/`) can't hide the boundary. A root that contains `$HOME`
/// can only be a "false broader environment" that would sweep in every
/// unrelated project under it.
fn is_home_or_ancestor(p: &Path, home: Option<&Path>, home_canon: Option<&Path>) -> bool {
    if let Some(home) = home {
        if p == home {
            return true;
        }
    }
    let Some(home_canon) = home_canon else {
        return false;
    };
    match p.canonicalize() {
        Ok(canon) => home_canon.starts_with(canon),
        Err(_) => false,
    }
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
        // On Windows the OS temp dir lives inside the user's home (e.g.
        // C:\Users\<name>\AppData\Local\Temp) — this used to mean the walk
        // could cross a real home `.git`/`.agent` and adopt the home
        // directory as the root instead of `isolated`. That was the actual
        // bug (a `zeus` session in some unrelated empty folder anywhere
        // under `$HOME` silently widening to the user's entire home tree,
        // sweeping in every other project sitting there) — the walk no
        // longer crosses into `$HOME` on behalf of a narrower starting
        // directory, so the strict fallback now holds unconditionally.
        let tmp = TempDir::new().unwrap();
        let isolated = tmp.path().join("no_markers_here");
        std::fs::create_dir_all(&isolated).unwrap();

        assert_eq!(find_project_root(&isolated), Some(isolated));
    }

    #[test]
    fn find_project_root_does_not_widen_to_home_even_if_home_has_a_marker() {
        // Regression for the home-root-widening bug: a `.git`/`.agent`
        // sitting at `$HOME` (e.g. planted by an earlier bare `zeus` run
        // directly there, or a dotfiles repo) must never get adopted as the
        // project root for some unrelated, narrower directory underneath
        // it — the scope stays on the real current directory.
        let Some(home) = dirs::home_dir() else {
            return; // no home dir resolvable in this environment — skip
        };
        let probe = home.join(format!("zeus-root-widening-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&probe);
        std::fs::create_dir_all(&probe).unwrap();

        let found = find_project_root(&probe);

        let _ = std::fs::remove_dir_all(&probe);
        assert_eq!(
            found,
            Some(probe),
            "a marker at $HOME must never widen an unrelated subfolder's project root to $HOME"
        );
    }

    #[test]
    fn find_project_root_ignores_ancestor_agent_marker() {
        // Regression for the poison-marker vector behind the original bug: an
        // `.agent/` at an *ancestor* (planted there by an earlier bare `zeus`
        // run directly in that folder) must never get adopted as the project
        // root for a narrower starting directory. Scope stays on the real
        // current directory — ancestor `.agent` markers are not followed at
        // all, only one directly at `start`.
        let tmp = TempDir::new().unwrap();
        let poisoned = tmp.path().join("planted_here");
        std::fs::create_dir_all(poisoned.join(".agent")).unwrap();
        let start = poisoned.join("zeus_test").join("nested");
        std::fs::create_dir_all(&start).unwrap();

        assert_eq!(
            find_project_root(&start),
            Some(start),
            "an ancestor `.agent` marker must not widen the project root"
        );
    }

    #[test]
    fn find_project_root_prefers_current_dir_marker() {
        // A marker directly at the current directory wins over any outer repo
        // or marker: the session is scoped to exactly where it is running.
        let tmp = TempDir::new().unwrap();
        let inner = tmp.path().join("inner");
        std::fs::create_dir_all(inner.join(".agent")).unwrap();
        std::fs::create_dir_all(tmp.path().join(".git")).unwrap();

        assert_eq!(find_project_root(&inner), Some(inner));
        assert_eq!(
            find_project_root(tmp.path()),
            Some(tmp.path().to_path_buf())
        );
    }

    #[test]
    fn find_project_root_uses_git_toplevel_from_nested_dir() {
        // Inside a git work tree the authoritative root comes from git (the
        // `.git/HEAD` file is all it needs); if git isn't available the
        // `.git`-dir walk still resolves to the same root.
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join(".git")).unwrap();
        std::fs::write(tmp.path().join(".git/HEAD"), "ref: refs/heads/master\n").unwrap();
        let nested = tmp.path().join("a/b/c");
        std::fs::create_dir_all(&nested).unwrap();

        assert_eq!(find_project_root(&nested), Some(tmp.path().to_path_buf()));
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
