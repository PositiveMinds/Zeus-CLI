//! Agent settings with layered merge (global → project → local).

use crate::error::{ConfigError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Which layer a settings file belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SettingsLayer {
    /// Built-in safe defaults (always present; lowest precedence).
    Builtin,
    /// `~/.zeus/settings.toml`
    Global,
    /// `<project>/.agent/settings.toml` (checked in, shared)
    Project,
    /// `<project>/.agent/settings.local.toml` (gitignored, personal)
    Local,
}

impl SettingsLayer {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Builtin => "builtin",
            Self::Global => "global",
            Self::Project => "project",
            Self::Local => "local",
        }
    }
}

/// Three-state permission model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PermissionState {
    Allow,
    #[default]
    Ask,
    Deny,
}

/// Default permission for a tool name when no more specific rule matches.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PermissionDefault {
    pub tool: String,
    pub state: PermissionState,
}

/// Scoped permission rule (tool + optional path/command pattern).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PermissionRule {
    pub tool: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub command: Option<String>,
    pub state: PermissionState,
}

/// Fully merged agent settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AgentSettings {
    #[serde(default)]
    pub model: ModelSettings,
    #[serde(default)]
    pub permissions: PermissionSettings,
    #[serde(default)]
    pub context: ContextSettings,
    #[serde(default)]
    pub logging: LoggingSettings,
    /// Known project roots for cross-project search (Phase 2+).
    #[serde(default)]
    pub project_roots: Vec<String>,
    /// External MCP tool servers to connect to (Phase 4+).
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
    /// Extra directories to scan for local model files (GGUF/safetensors),
    /// beyond `~/.zeus/models/` and the best-effort LM Studio/Hugging-Face
    /// cache defaults — for anything in a nonstandard location.
    #[serde(default)]
    pub extra_model_dirs: Vec<String>,
    /// llama.cpp local serving engine: binary resolution, port, and an
    /// auto-downloadable catalog of GGUF models.
    #[serde(default)]
    pub llamacpp: LlamaCppSettings,
    /// Disables the TUI's animated wordmark/spinner in favor of static
    /// text — for terminals/links where redrawing every 100ms is
    /// unwelcome (slow SSH, a strong no-motion preference), or just a
    /// quieter look. Off by default.
    #[serde(default)]
    pub reduced_motion: bool,
    /// TUI brand accent color as a `#rrggbb` hex string, overriding the
    /// default violet used for borders, the progress bar, and highlighted
    /// rows. `None` (the default) keeps the built-in violet.
    #[serde(default)]
    pub accent_color: Option<String>,
    /// Rings the terminal bell when a turn finishes — the cue that lets you
    /// tab away during a long/Auto-mode run instead of watching it. On by
    /// default; most terminals map BEL to a visual flash unless the user
    /// has configured an audible one.
    #[serde(default = "default_true")]
    pub notify_on_completion: bool,
    /// TUI color theme preset as a name (`dark`, `light`, `high-contrast`),
    /// persisted from `/theme`. `None` (the default) keeps `dark`.
    #[serde(default)]
    pub theme: Option<String>,
    /// Cap on how many read-only orchestration steps run concurrently (the
    /// "max parallel read steps" bound on the headless read-only batch).
    /// `None` (the default) keeps the CLI's built-in bound of 2.
    #[serde(default)]
    pub max_parallel_read_steps: Option<usize>,
}

/// One configured external MCP tool server, spawned over stdio.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpServerConfig {
    /// Unique name; used as the tool-name prefix (`mcp__<name>__<tool>`) and
    /// as the merge key across settings layers.
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelSettings {
    /// Default provider id (matches a key in providers.toml).
    #[serde(default = "default_provider")]
    pub provider: String,
    /// Default model name for that provider.
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

fn default_provider() -> String {
    "ollama".into()
}

fn default_model() -> String {
    "llama3.2".into()
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model: default_model(),
            temperature: None,
            max_tokens: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlamaCppSettings {
    /// Path to the `llama-server` binary. When unset, zeus searches PATH and,
    /// if still not found, downloads a llama.cpp release build into
    /// `~/.zeus/bin/` on first use.
    #[serde(default)]
    pub binary: Option<String>,
    /// Port the spawned llama-server listens on.
    #[serde(default = "default_llamacpp_port")]
    pub port: u16,
    #[serde(default)]
    pub threads: Option<u32>,
    #[serde(default)]
    pub ctx: Option<u32>,
    /// Auto-downloadable GGUF models: `name` is what `--model`/the picker
    /// refer to; `repo` + `file` point at a Hugging Face GGUF download. When
    /// the file is absent it's pulled into `~/.zeus/models/` on demand.
    #[serde(default)]
    pub models: Vec<LocalModelEntry>,
}

impl Default for LlamaCppSettings {
    fn default() -> Self {
        Self {
            binary: None,
            port: default_llamacpp_port(),
            threads: None,
            ctx: None,
            models: Vec::new(),
        }
    }
}

fn default_llamacpp_port() -> u16 {
    8080
}

/// One auto-downloadable local model for the llama.cpp engine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LocalModelEntry {
    /// Short name used to select the model (e.g. `llama3.2`).
    pub name: String,
    /// Hugging Face repo id, e.g. `bartowski/Llama-3.2-3B-Instruct-GGUF`.
    pub repo: String,
    /// File within the repo, e.g. `Llama-3.2-3B-Instruct-Q4_K_M.gguf`.
    pub file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PermissionSettings {
    /// Per-tool defaults when no rule matches.
    #[serde(default)]
    pub defaults: Vec<PermissionDefault>,
    /// Specific rules (path/command scoped). Higher precedence within a layer.
    #[serde(default)]
    pub rules: Vec<PermissionRule>,
    /// Session-only auto-approve escape hatch flag name (never persisted unless explicit).
    #[serde(default)]
    pub allow_session_auto_approve: bool,
}

impl PermissionSettings {
    /// Built-in safe defaults: destructive-by-default denied/ask.
    pub fn builtin_safe() -> Self {
        Self {
            defaults: vec![
                PermissionDefault {
                    tool: "read".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "search".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "write".into(),
                    state: PermissionState::Ask,
                },
                PermissionDefault {
                    tool: "edit".into(),
                    state: PermissionState::Ask,
                },
                PermissionDefault {
                    tool: "delete".into(),
                    state: PermissionState::Ask,
                },
                PermissionDefault {
                    tool: "bash".into(),
                    state: PermissionState::Ask,
                },
                // Git — read-only tier: allow.
                PermissionDefault {
                    tool: "git_status".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "git_diff".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "git_blame".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "git_log".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "git_show".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "git_branch_list".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "git_remote_list".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "git_tag_list".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "git_stash_list".into(),
                    state: PermissionState::Allow,
                },
                // Git — reversible write tier: allow (commit always previews the diff regardless).
                PermissionDefault {
                    tool: "git_add".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "git_commit".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "git_stash".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "git_branch".into(),
                    state: PermissionState::Allow,
                },
                PermissionDefault {
                    tool: "git_tag".into(),
                    state: PermissionState::Allow,
                },
                // Git — working-tree-changing / network / history-rewriting: ask.
                PermissionDefault {
                    tool: "git_branch_delete".into(),
                    state: PermissionState::Ask,
                },
                PermissionDefault {
                    tool: "git_checkout".into(),
                    state: PermissionState::Ask,
                },
                PermissionDefault {
                    tool: "git_fetch".into(),
                    state: PermissionState::Ask,
                },
                PermissionDefault {
                    tool: "git_pull".into(),
                    state: PermissionState::Ask,
                },
                PermissionDefault {
                    tool: "git_push".into(),
                    state: PermissionState::Ask,
                },
                PermissionDefault {
                    tool: "git_reset".into(),
                    state: PermissionState::Ask,
                },
                PermissionDefault {
                    tool: "git_revert".into(),
                    state: PermissionState::Ask,
                },
                PermissionDefault {
                    tool: "git_cherry_pick".into(),
                    state: PermissionState::Ask,
                },
                PermissionDefault {
                    tool: "git_rebase".into(),
                    state: PermissionState::Ask,
                },
                PermissionDefault {
                    tool: "git_merge".into(),
                    state: PermissionState::Ask,
                },
            ],
            rules: vec![
                PermissionRule {
                    tool: "read".into(),
                    path: Some("**/.env".into()),
                    command: None,
                    state: PermissionState::Deny,
                },
                PermissionRule {
                    tool: "read".into(),
                    path: Some("**/*credential*".into()),
                    command: None,
                    state: PermissionState::Deny,
                },
                PermissionRule {
                    tool: "bash".into(),
                    path: None,
                    command: Some("rm -rf*".into()),
                    state: PermissionState::Deny,
                },
                PermissionRule {
                    tool: "bash".into(),
                    path: None,
                    command: Some("git push --force*".into()),
                    state: PermissionState::Deny,
                },
                PermissionRule {
                    tool: "bash".into(),
                    path: None,
                    command: Some("git reset --hard*".into()),
                    state: PermissionState::Deny,
                },
                // The structured git_push/git_reset tools are a different
                // tool name from "bash", so the two rules above (scoped to
                // tool="bash") don't cover them — re-declare the same
                // force-push/hard-reset denial here rather than let it
                // silently fall through to a plain "ask".
                PermissionRule {
                    tool: "git_push".into(),
                    path: None,
                    command: Some("--force*".into()),
                    state: PermissionState::Deny,
                },
                PermissionRule {
                    tool: "git_reset".into(),
                    path: None,
                    command: Some("--hard*".into()),
                    state: PermissionState::Deny,
                },
            ],
            allow_session_auto_approve: false,
        }
    }
}

impl Default for PermissionSettings {
    fn default() -> Self {
        Self::builtin_safe()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContextSettings {
    /// Soft threshold (0.0–1.0) of model window before summarization triggers.
    #[serde(default = "default_compact_threshold")]
    pub compact_threshold: f32,
    /// Recent turns to keep verbatim when compacting.
    #[serde(default = "default_keep_recent_turns")]
    pub keep_recent_turns: u32,
}

fn default_compact_threshold() -> f32 {
    0.8
}

fn default_keep_recent_turns() -> u32 {
    6
}

impl Default for ContextSettings {
    fn default() -> Self {
        Self {
            compact_threshold: default_compact_threshold(),
            keep_recent_turns: default_keep_recent_turns(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoggingSettings {
    #[serde(default = "default_log_level")]
    pub level: String,
    /// Also write JSON lines to the global logs directory.
    #[serde(default = "default_true")]
    pub file: bool,
}

fn default_log_level() -> String {
    "info".into()
}

fn default_true() -> bool {
    true
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            file: true,
        }
    }
}

/// Partial settings as stored on disk (all fields optional for merge).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PartialSettings {
    #[serde(default)]
    model: Option<PartialModel>,
    #[serde(default)]
    permissions: Option<PartialPermissions>,
    #[serde(default)]
    context: Option<PartialContext>,
    #[serde(default)]
    logging: Option<PartialLogging>,
    #[serde(default)]
    project_roots: Option<Vec<String>>,
    #[serde(default)]
    mcp_servers: Option<Vec<McpServerConfig>>,
    #[serde(default)]
    extra_model_dirs: Option<Vec<String>>,
    #[serde(default)]
    llamacpp: Option<LlamaCppSettings>,
    #[serde(default)]
    reduced_motion: Option<bool>,
    #[serde(default)]
    accent_color: Option<String>,
    #[serde(default)]
    notify_on_completion: Option<bool>,
    #[serde(default)]
    theme: Option<String>,
    #[serde(default)]
    max_parallel_read_steps: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PartialModel {
    provider: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PartialPermissions {
    defaults: Option<Vec<PermissionDefault>>,
    rules: Option<Vec<PermissionRule>>,
    allow_session_auto_approve: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PartialContext {
    compact_threshold: Option<f32>,
    keep_recent_turns: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PartialLogging {
    level: Option<String>,
    file: Option<bool>,
}

/// Stack of settings layers used for resolution.
#[derive(Debug, Default)]
pub struct SettingsStack {
    layers: Vec<(SettingsLayer, PartialSettings)>,
}

impl SettingsStack {
    pub fn new() -> Self {
        let mut s = Self::default();
        // Encode builtin defaults as a synthetic partial for merge simplicity.
        s.layers.push((
            SettingsLayer::Builtin,
            PartialSettings {
                model: Some(PartialModel {
                    provider: Some(default_provider()),
                    model: Some(default_model()),
                    temperature: None,
                    max_tokens: None,
                }),
                permissions: Some(PartialPermissions {
                    defaults: Some(PermissionSettings::builtin_safe().defaults),
                    rules: Some(PermissionSettings::builtin_safe().rules),
                    allow_session_auto_approve: Some(false),
                }),
                context: Some(PartialContext {
                    compact_threshold: Some(default_compact_threshold()),
                    keep_recent_turns: Some(default_keep_recent_turns()),
                }),
                logging: Some(PartialLogging {
                    level: Some(default_log_level()),
                    file: Some(true),
                }),
                project_roots: Some(Vec::new()),
                mcp_servers: Some(Vec::new()),
                extra_model_dirs: Some(Vec::new()),
                llamacpp: Some(LlamaCppSettings::default()),
                reduced_motion: Some(false),
                accent_color: None,
                notify_on_completion: Some(true),
                theme: None,
                max_parallel_read_steps: None,
            },
        ));
        s
    }

    /// Load a settings file if it exists; missing files are skipped (not an error).
    pub fn push_file(&mut self, layer: SettingsLayer, path: &Path) -> Result<()> {
        if !path.exists() {
            return Ok(());
        }
        let text = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
        let partial: PartialSettings =
            toml::from_str(&text).map_err(|source| ConfigError::TomlParse {
                path: path.to_path_buf(),
                source: Box::new(source),
            })?;
        self.layers.push((layer, partial));
        Ok(())
    }

    pub fn loaded_layers(&self) -> Vec<&'static str> {
        self.layers.iter().map(|(l, _)| l.as_str()).collect()
    }

    /// Merge all layers; later layers win field-by-field.
    pub fn resolve(&self) -> AgentSettings {
        let mut out = AgentSettings {
            model: ModelSettings {
                provider: String::new(),
                model: String::new(),
                temperature: None,
                max_tokens: None,
            },
            permissions: PermissionSettings {
                defaults: Vec::new(),
                rules: Vec::new(),
                allow_session_auto_approve: false,
            },
            context: ContextSettings {
                compact_threshold: default_compact_threshold(),
                keep_recent_turns: default_keep_recent_turns(),
            },
            logging: LoggingSettings {
                level: default_log_level(),
                file: true,
            },
            project_roots: Vec::new(),
            mcp_servers: Vec::new(),
            extra_model_dirs: Vec::new(),
            llamacpp: LlamaCppSettings::default(),
            reduced_motion: false,
            accent_color: None,
            notify_on_completion: true,
            theme: None,
            max_parallel_read_steps: None,
        };

        for (_layer, partial) in &self.layers {
            if let Some(m) = &partial.model {
                if let Some(p) = &m.provider {
                    out.model.provider = p.clone();
                }
                if let Some(model) = &m.model {
                    out.model.model = model.clone();
                }
                if m.temperature.is_some() {
                    out.model.temperature = m.temperature;
                }
                if m.max_tokens.is_some() {
                    out.model.max_tokens = m.max_tokens;
                }
            }
            if let Some(p) = &partial.permissions {
                if let Some(d) = &p.defaults {
                    // Later layer replaces defaults map by tool name, then keeps order.
                    let mut map: HashMap<String, PermissionState> = out
                        .permissions
                        .defaults
                        .iter()
                        .map(|x| (x.tool.clone(), x.state))
                        .collect();
                    for item in d {
                        map.insert(item.tool.clone(), item.state);
                    }
                    out.permissions.defaults = map
                        .into_iter()
                        .map(|(tool, state)| PermissionDefault { tool, state })
                        .collect();
                    out.permissions.defaults.sort_by(|a, b| a.tool.cmp(&b.tool));
                }
                if let Some(rules) = &p.rules {
                    // Append rules; later layers can override by matching tool+path+command.
                    for rule in rules {
                        if let Some(existing) = out.permissions.rules.iter_mut().find(|r| {
                            r.tool == rule.tool && r.path == rule.path && r.command == rule.command
                        }) {
                            *existing = rule.clone();
                        } else {
                            out.permissions.rules.push(rule.clone());
                        }
                    }
                }
                if let Some(flag) = p.allow_session_auto_approve {
                    out.permissions.allow_session_auto_approve = flag;
                }
            }
            if let Some(c) = &partial.context {
                if let Some(t) = c.compact_threshold {
                    out.context.compact_threshold = t;
                }
                if let Some(k) = c.keep_recent_turns {
                    out.context.keep_recent_turns = k;
                }
            }
            if let Some(l) = &partial.logging {
                if let Some(level) = &l.level {
                    out.logging.level = level.clone();
                }
                if let Some(file) = l.file {
                    out.logging.file = file;
                }
            }
            if let Some(rm) = partial.reduced_motion {
                out.reduced_motion = rm;
            }
            if let Some(color) = &partial.accent_color {
                out.accent_color = Some(color.clone());
            }
            if let Some(v) = partial.notify_on_completion {
                out.notify_on_completion = v;
            }
            if let Some(theme) = &partial.theme {
                out.theme = Some(theme.clone());
            }
            if let Some(v) = partial.max_parallel_read_steps {
                out.max_parallel_read_steps = Some(v);
            }
            if let Some(roots) = &partial.project_roots {
                // Later layers replace the list entirely when present.
                out.project_roots = roots.clone();
            }
            if let Some(servers) = &partial.mcp_servers {
                // Append/override by name, same pattern as permission rules.
                for server in servers {
                    if let Some(existing) =
                        out.mcp_servers.iter_mut().find(|s| s.name == server.name)
                    {
                        *existing = server.clone();
                    } else {
                        out.mcp_servers.push(server.clone());
                    }
                }
            }
            if let Some(dirs) = &partial.extra_model_dirs {
                // Later layers replace the list entirely, same as project_roots.
                out.extra_model_dirs = dirs.clone();
            }
            if let Some(lcpp) = &partial.llamacpp {
                out.llamacpp = lcpp.clone();
            }
        }

        // Ensure model defaults if somehow empty.
        if out.model.provider.is_empty() {
            out.model.provider = default_provider();
        }
        if out.model.model.is_empty() {
            out.model.model = default_model();
        }

        out
    }
}

impl AgentSettings {
    pub fn write_defaults_if_missing(path: &Path) -> Result<()> {
        if path.exists() {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(ConfigError::Io)?;
        }
        let body = r#"# Global zeus settings
# Overridden by project .agent/settings.toml and settings.local.toml

[model]
provider = "ollama"
model = "llama3.2"

[logging]
level = "info"
file = true

[context]
compact_threshold = 0.8
keep_recent_turns = 6

# project_roots = ["/path/to/other/project"]
"#;
        std::fs::write(path, body).map_err(ConfigError::Io)?;
        Ok(())
    }

    pub fn write_project_defaults_if_missing(path: &Path) -> Result<()> {
        if path.exists() {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(ConfigError::Io)?;
        }
        let body = r#"# Project settings (checked into git — shared with the team)
# Personal overrides go in settings.local.toml (gitignored)

# [model]
# provider = "ollama"
# model = "llama3.2"

# Example: allow bash for common build tools in this project
# [[permissions.defaults]]
# tool = "bash"
# state = "ask"
"#;
        std::fs::write(path, body).map_err(ConfigError::Io)?;
        Ok(())
    }
}

fn load_partial(path: &Path) -> Result<PartialSettings> {
    if !path.exists() {
        return Ok(PartialSettings::default());
    }
    let text = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
    toml::from_str(&text).map_err(|source| ConfigError::TomlParse {
        path: path.to_path_buf(),
        source: Box::new(source),
    })
}

fn save_partial(path: &Path, partial: &PartialSettings) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(ConfigError::Io)?;
    }
    let text = toml::to_string_pretty(partial).map_err(ConfigError::TomlSerialize)?;
    std::fs::write(path, text).map_err(ConfigError::Io)
}

/// Reads the on-disk partial settings at `path` (or starts from an empty
/// one if the file doesn't exist yet), flips one field, and rewrites it —
/// backs the in-app `/settings` command so users never have to hand-edit
/// `settings.toml`. `path` is normally `Config::global.settings_toml`.
pub fn set_reduced_motion(path: &Path, value: bool) -> Result<()> {
    let mut partial = load_partial(path)?;
    partial.reduced_motion = Some(value);
    save_partial(path, &partial)
}

/// Same, for `accent_color` (`None` clears the override back to the
/// built-in violet).
pub fn set_accent_color(path: &Path, value: Option<String>) -> Result<()> {
    let mut partial = load_partial(path)?;
    partial.accent_color = value;
    save_partial(path, &partial)
}

/// Same, for `notify_on_completion`.
pub fn set_notify_on_completion(path: &Path, value: bool) -> Result<()> {
    let mut partial = load_partial(path)?;
    partial.notify_on_completion = Some(value);
    save_partial(path, &partial)
}

/// Same, for the TUI theme preset (`None` clears back to `dark`).
pub fn set_theme(path: &Path, value: Option<String>) -> Result<()> {
    let mut partial = load_partial(path)?;
    partial.theme = value;
    save_partial(path, &partial)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn local_overrides_project_and_global() {
        let tmp = TempDir::new().unwrap();
        let global = tmp.path().join("global.toml");
        let project = tmp.path().join("project.toml");
        let local = tmp.path().join("local.toml");

        std::fs::write(
            &global,
            r#"
[model]
provider = "ollama"
model = "from-global"
[logging]
level = "warn"
"#,
        )
        .unwrap();
        std::fs::write(
            &project,
            r#"
[model]
model = "from-project"
"#,
        )
        .unwrap();
        std::fs::write(
            &local,
            r#"
[model]
model = "from-local"
provider = "openai"
"#,
        )
        .unwrap();

        let mut stack = SettingsStack::new();
        stack.push_file(SettingsLayer::Global, &global).unwrap();
        stack.push_file(SettingsLayer::Project, &project).unwrap();
        stack.push_file(SettingsLayer::Local, &local).unwrap();
        let resolved = stack.resolve();

        assert_eq!(resolved.model.model, "from-local");
        assert_eq!(resolved.model.provider, "openai");
        assert_eq!(resolved.logging.level, "warn");
    }

    #[test]
    fn missing_files_are_ok() {
        let mut stack = SettingsStack::new();
        stack
            .push_file(
                SettingsLayer::Global,
                Path::new("/nonexistent/settings.toml"),
            )
            .unwrap();
        let resolved = stack.resolve();
        assert_eq!(resolved.model.provider, "ollama");
    }

    #[test]
    fn builtin_denies_destructive_bash() {
        let settings = AgentSettings::default();
        let deny_rm =
            settings.permissions.rules.iter().any(|r| {
                r.command.as_deref() == Some("rm -rf*") && r.state == PermissionState::Deny
            });
        assert!(deny_rm);
    }
}
