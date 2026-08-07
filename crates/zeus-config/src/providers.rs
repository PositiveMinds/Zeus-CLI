//! Provider configuration (`providers.toml`).

use crate::error::{ConfigError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Entire providers file.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvidersFile {
    /// Named provider entries.
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
}

/// One model provider endpoint configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderConfig {
    /// Provider kind: ollama | lmstudio | llamacpp | openai | anthropic | gemini | grok | openrouter
    pub kind: String,
    /// Base URL (required for local and OpenAI-compatible endpoints).
    #[serde(default)]
    pub base_url: Option<String>,
    /// Environment variable name holding the API key (never store the key itself).
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Optional default model for this provider.
    #[serde(default)]
    pub default_model: Option<String>,
    /// Extra headers as key → value or env-var reference.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Whether this provider supports embeddings.
    #[serde(default)]
    pub embeddings: bool,
    /// Whether prompt caching is available on this provider.
    #[serde(default)]
    pub prompt_cache: bool,
}

impl ProvidersFile {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::builtin_defaults());
        }
        let text = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
        let mut file: ProvidersFile =
            toml::from_str(&text).map_err(|source| ConfigError::TomlParse {
                path: path.to_path_buf(),
                source,
            })?;
        // Merge builtin defaults for any missing well-known providers.
        let defaults = Self::builtin_defaults();
        for (name, cfg) in defaults.providers {
            file.providers.entry(name).or_insert(cfg);
        }
        Ok(file)
    }

    pub fn get(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.get(name)
    }

    pub fn write_defaults_if_missing(path: &Path) -> Result<()> {
        if path.exists() {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(ConfigError::Io)?;
        }
        let body = r#"# Model providers
# API keys are never stored here — reference env var names only.

[providers.ollama]
kind = "ollama"
base_url = "http://127.0.0.1:11434"
default_model = "llama3.2"
embeddings = true
prompt_cache = false

[providers.openai]
kind = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
default_model = "gpt-4o-mini"
embeddings = true
prompt_cache = true

[providers.anthropic]
kind = "anthropic"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"
default_model = "claude-sonnet-4-20250514"
embeddings = false
prompt_cache = true

[providers.gemini]
kind = "gemini"
base_url = "https://generativelanguage.googleapis.com"
api_key_env = "GEMINI_API_KEY"
default_model = "gemini-2.0-flash"
embeddings = true
prompt_cache = false

[providers.grok]
kind = "grok"
base_url = "https://api.x.ai/v1"
api_key_env = "XAI_API_KEY"
default_model = "grok-3"
embeddings = false
prompt_cache = false

[providers.openrouter]
kind = "openrouter"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
default_model = "openrouter/auto"
embeddings = false
prompt_cache = false

[providers.opencodezen]
kind = "opencodezen"
base_url = "https://opencode.ai/zen/v1"
api_key_env = "OPENCODE_API_KEY"
default_model = "glm-4.6-free"
embeddings = false
prompt_cache = true

[providers.lmstudio]
kind = "lmstudio"
base_url = "http://127.0.0.1:1234/v1"
default_model = "local-model"
embeddings = true
prompt_cache = true

[providers.llamacpp]
kind = "llamacpp"
base_url = "http://127.0.0.1:8080/v1"
default_model = "local-model"
embeddings = true
prompt_cache = true
"#;
        std::fs::write(path, body).map_err(ConfigError::Io)?;
        Ok(())
    }

    pub fn builtin_defaults() -> Self {
        let mut providers = HashMap::new();
        providers.insert(
            "ollama".into(),
            ProviderConfig {
                kind: "ollama".into(),
                base_url: Some("http://127.0.0.1:11434".into()),
                api_key_env: None,
                default_model: Some("llama3.2".into()),
                headers: HashMap::new(),
                embeddings: true,
                prompt_cache: false,
            },
        );
        providers.insert(
            "openai".into(),
            ProviderConfig {
                kind: "openai".into(),
                base_url: Some("https://api.openai.com/v1".into()),
                api_key_env: Some("OPENAI_API_KEY".into()),
                default_model: Some("gpt-4o-mini".into()),
                headers: HashMap::new(),
                embeddings: true,
                prompt_cache: true,
            },
        );
        providers.insert(
            "opencodezen".into(),
            ProviderConfig {
                kind: "opencodezen".into(),
                base_url: Some("https://opencode.ai/zen/v1".into()),
                api_key_env: Some("OPENCODE_API_KEY".into()),
                default_model: Some("glm-4.6-free".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: true,
            },
        );
        Self { providers }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn load_missing_returns_defaults() {
        let file = ProvidersFile::load(Path::new("/no/such/providers.toml")).unwrap();
        assert!(file.get("ollama").is_some());
        assert!(file.get("openai").is_some());
    }

    #[test]
    fn write_and_load_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("providers.toml");
        ProvidersFile::write_defaults_if_missing(&path).unwrap();
        let file = ProvidersFile::load(&path).unwrap();
        assert_eq!(file.get("openai").unwrap().kind, "openai");
        assert_eq!(
            file.get("anthropic").unwrap().api_key_env.as_deref(),
            Some("ANTHROPIC_API_KEY")
        );
    }
}
