//! Private API-key store (`~/.zeus/keys.toml`).
//!
//! Keys the user sets in-app with `/provider key <name> <KEY>` live here so
//! they survive restarts. It's a user-owned file alongside `providers.toml`
//! — kept out of any repo. At load time these keys are injected into each
//! provider's config as an embedded `Authorization: Bearer <key>` header
//! (the registry already prefers an embedded header over the env var), so
//! a stored key "just works" without any extra wiring.

use crate::error::{ConfigError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Key store: provider name → secret key.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KeysFile {
    /// Map of provider name to its stored API key.
    #[serde(default)]
    pub keys: HashMap<String, String>,
}

impl KeysFile {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
        toml::from_str(&text).map_err(|source| ConfigError::TomlParse {
            path: path.to_path_buf(),
            source: Box::new(source),
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(ConfigError::Io)?;
        }
        let text = toml::to_string_pretty(self).map_err(ConfigError::TomlSerialize)?;
        std::fs::write(path, text).map_err(ConfigError::Io)?;
        restrict_to_owner(path)
    }

    pub fn get(&self, name: &str) -> Option<&String> {
        self.keys.get(name)
    }
}

/// Restricts a just-written secrets file to owner-read/write only. On Unix
/// this is the difference between a key file honoring the process umask
/// (often group/world-readable) and matching how `ssh`/`aws`/`gh` protect
/// their own credential files. A no-op on Windows, where the per-user
/// profile directory this file lives under is already ACL'd to the owner by
/// default. Best-effort: a permission-set failure shouldn't undo an
/// otherwise-successful save.
#[cfg(unix)]
fn restrict_to_owner(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let perms = std::fs::Permissions::from_mode(0o600);
    let _ = std::fs::set_permissions(path, perms);
    Ok(())
}

#[cfg(not(unix))]
fn restrict_to_owner(_path: &Path) -> Result<()> {
    Ok(())
}