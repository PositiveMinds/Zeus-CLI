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
                source: Box::new(source),
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

    /// Inject stored keys as embedded `Authorization: Bearer ...` headers so
    /// a key persisted in `keys.toml` is used without any extra wiring.
    pub fn inject_keys(&mut self, keys: &crate::KeysFile) {
        for (name, key) in &keys.keys {
            if let Some(cfg) = self.providers.get_mut(name) {
                cfg.headers
                    .insert("Authorization".to_string(), format!("Bearer {key}"));
            }
        }
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
default_model = "deepseek-v4-flash-free"
embeddings = false
prompt_cache = true

[providers.deepseek]
kind = "deepseek"
base_url = "https://api.deepseek.com/v1"
api_key_env = "DEEPSEEK_API_KEY"
default_model = "deepseek-chat"
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

[providers.mistral]
kind = "mistral"
base_url = "https://api.mistral.ai/v1"
api_key_env = "MISTRAL_API_KEY"
default_model = "mistral-large-latest"
embeddings = false
prompt_cache = false

[providers.groq]
kind = "groq"
base_url = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"
default_model = "llama-3.3-70b-versatile"
embeddings = false
prompt_cache = false

[providers.together]
kind = "together"
base_url = "https://api.together.xyz/v1"
api_key_env = "TOGETHER_API_KEY"
default_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
embeddings = false
prompt_cache = false

[providers.fireworks]
kind = "fireworks"
base_url = "https://api.fireworks.ai/inference/v1"
api_key_env = "FIREWORKS_API_KEY"
default_model = "accounts/fireworks/models/llama-v3p3-70b-instruct"
embeddings = false
prompt_cache = false

[providers.perplexity]
kind = "perplexity"
base_url = "https://api.perplexity.ai"
api_key_env = "PERPLEXITY_API_KEY"
default_model = "sonar"
embeddings = false
prompt_cache = false

[providers.cohere]
kind = "cohere"
base_url = "https://api.cohere.ai/compatibility/v1"
api_key_env = "COHERE_API_KEY"
default_model = "command-r-plus"
embeddings = false
prompt_cache = false

[providers.cerebras]
kind = "cerebras"
base_url = "https://api.cerebras.ai/v1"
api_key_env = "CEREBRAS_API_KEY"
default_model = "llama-3.3-70b"
embeddings = false
prompt_cache = false

[providers.deepinfra]
kind = "deepinfra"
base_url = "https://api.deepinfra.com/v1/openai"
api_key_env = "DEEPINFRA_API_KEY"
default_model = "meta-llama/Llama-3.3-70B-Instruct"
embeddings = false
prompt_cache = false

[providers.novita]
kind = "novita"
base_url = "https://api.novita.ai/v3/openai"
api_key_env = "NOVITA_API_KEY"
default_model = "meta-llama/llama-3.3-70b-instruct"
embeddings = false
prompt_cache = false
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
                default_model: Some("deepseek-v4-flash-free".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: true,
            },
        );
        providers.insert(
            "deepseek".into(),
            ProviderConfig {
                kind: "deepseek".into(),
                base_url: Some("https://api.deepseek.com/v1".into()),
                api_key_env: Some("DEEPSEEK_API_KEY".into()),
                default_model: Some("deepseek-chat".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: true,
            },
        );
        providers.insert(
            "anthropic".into(),
            ProviderConfig {
                kind: "anthropic".into(),
                base_url: Some("https://api.anthropic.com".into()),
                api_key_env: Some("ANTHROPIC_API_KEY".into()),
                default_model: Some("claude-sonnet-4-20250514".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: true,
            },
        );
        providers.insert(
            "gemini".into(),
            ProviderConfig {
                kind: "gemini".into(),
                base_url: Some("https://generativelanguage.googleapis.com/v1beta/openai".into()),
                api_key_env: Some("GEMINI_API_KEY".into()),
                default_model: Some("gemini-2.0-flash".into()),
                headers: HashMap::new(),
                embeddings: true,
                prompt_cache: false,
            },
        );
        providers.insert(
            "grok".into(),
            ProviderConfig {
                kind: "grok".into(),
                base_url: Some("https://api.x.ai/v1".into()),
                api_key_env: Some("XAI_API_KEY".into()),
                default_model: Some("grok-3".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: false,
            },
        );
        providers.insert(
            "openrouter".into(),
            ProviderConfig {
                kind: "openrouter".into(),
                base_url: Some("https://openrouter.ai/api/v1".into()),
                api_key_env: Some("OPENROUTER_API_KEY".into()),
                default_model: Some("openrouter/auto".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: false,
            },
        );
        providers.insert(
            "lmstudio".into(),
            ProviderConfig {
                kind: "lmstudio".into(),
                base_url: Some("http://127.0.0.1:1234/v1".into()),
                api_key_env: None,
                default_model: Some("local-model".into()),
                headers: HashMap::new(),
                embeddings: true,
                prompt_cache: true,
            },
        );
        providers.insert(
            "llamacpp".into(),
            ProviderConfig {
                kind: "llamacpp".into(),
                base_url: Some("http://127.0.0.1:8080/v1".into()),
                api_key_env: None,
                default_model: Some("local-model".into()),
                headers: HashMap::new(),
                embeddings: true,
                prompt_cache: true,
            },
        );
        providers.insert(
            "mistral".into(),
            ProviderConfig {
                kind: "mistral".into(),
                base_url: Some("https://api.mistral.ai/v1".into()),
                api_key_env: Some("MISTRAL_API_KEY".into()),
                default_model: Some("mistral-large-latest".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: false,
            },
        );
        providers.insert(
            "groq".into(),
            ProviderConfig {
                kind: "groq".into(),
                base_url: Some("https://api.groq.com/openai/v1".into()),
                api_key_env: Some("GROQ_API_KEY".into()),
                default_model: Some("llama-3.3-70b-versatile".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: false,
            },
        );
        providers.insert(
            "together".into(),
            ProviderConfig {
                kind: "together".into(),
                base_url: Some("https://api.together.xyz/v1".into()),
                api_key_env: Some("TOGETHER_API_KEY".into()),
                default_model: Some("meta-llama/Llama-3.3-70B-Instruct-Turbo".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: false,
            },
        );
        providers.insert(
            "fireworks".into(),
            ProviderConfig {
                kind: "fireworks".into(),
                base_url: Some("https://api.fireworks.ai/inference/v1".into()),
                api_key_env: Some("FIREWORKS_API_KEY".into()),
                default_model: Some("accounts/fireworks/models/llama-v3p3-70b-instruct".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: false,
            },
        );
        providers.insert(
            "perplexity".into(),
            ProviderConfig {
                kind: "perplexity".into(),
                base_url: Some("https://api.perplexity.ai".into()),
                api_key_env: Some("PERPLEXITY_API_KEY".into()),
                default_model: Some("sonar".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: false,
            },
        );
        providers.insert(
            "cohere".into(),
            ProviderConfig {
                kind: "cohere".into(),
                base_url: Some("https://api.cohere.ai/compatibility/v1".into()),
                api_key_env: Some("COHERE_API_KEY".into()),
                default_model: Some("command-r-plus".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: false,
            },
        );
        providers.insert(
            "cerebras".into(),
            ProviderConfig {
                kind: "cerebras".into(),
                base_url: Some("https://api.cerebras.ai/v1".into()),
                api_key_env: Some("CEREBRAS_API_KEY".into()),
                default_model: Some("llama-3.3-70b".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: false,
            },
        );
        providers.insert(
            "deepinfra".into(),
            ProviderConfig {
                kind: "deepinfra".into(),
                base_url: Some("https://api.deepinfra.com/v1/openai".into()),
                api_key_env: Some("DEEPINFRA_API_KEY".into()),
                default_model: Some("meta-llama/Llama-3.3-70B-Instruct".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: false,
            },
        );
        providers.insert(
            "novita".into(),
            ProviderConfig {
                kind: "novita".into(),
                base_url: Some("https://api.novita.ai/v3/openai".into()),
                api_key_env: Some("NOVITA_API_KEY".into()),
                default_model: Some("meta-llama/llama-3.3-70b-instruct".into()),
                headers: HashMap::new(),
                embeddings: false,
                prompt_cache: false,
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
