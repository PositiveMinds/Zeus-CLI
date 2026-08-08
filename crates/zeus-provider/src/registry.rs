//! Build a concrete provider from config.

use crate::anthropic::AnthropicProvider;
use crate::error::{ProviderError, Result};
use crate::ollama::OllamaProvider;
use crate::openai_compat::OpenAiCompatProvider;
use crate::ModelProvider;
use zeus_config::{ProviderConfig, ProvidersFile};
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use tracing::info;

/// Shared handle to a live provider.
pub type ProviderHandle = Arc<dyn ModelProvider>;

fn embedded_key(cfg: &ProviderConfig) -> Option<String> {
    // A key may be embedded directly via headers["Authorization"] = "Bearer ..."
    cfg.headers
        .get("Authorization")
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|k| k.to_string())
}

/// Resolve the bearer key: embedded header first, then `api_key_env` env var.
pub fn resolve_api_key(cfg: &ProviderConfig, _name: &str) -> Result<Option<String>> {
    if let Some(k) = embedded_key(cfg) {
        return Ok(Some(k));
    }
    let Some(var) = cfg.api_key_env.as_deref() else {
        return Ok(None);
    };
    match env::var(var) {
        Ok(k) if !k.is_empty() => Ok(Some(k)),
        Ok(_) => Ok(None),
        Err(env::VarError::NotPresent) => Err(ProviderError::MissingApiKey(var.to_string())),
        Err(env::VarError::NotUnicode(_)) => Err(ProviderError::MissingApiKey(var.to_string())),
    }
}

/// Resolve extra headers, translating any `${ENV_VAR}` / `$ENV_VAR` references.
fn resolve_headers(cfg: &ProviderConfig) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for (k, v) in &cfg.headers {
        if k.eq_ignore_ascii_case("authorization") {
            continue; // handled via api_key
        }
        let resolved = if let Some(rest) = v.strip_prefix('$') {
            env::var(rest).unwrap_or_default()
        } else {
            v.clone()
        };
        out.insert(k.clone(), resolved);
    }
    out
}

/// Create a provider by name from the providers file.
pub fn create_provider(name: &str, file: &ProvidersFile) -> Result<ProviderHandle> {
    let cfg = file
        .get(name)
        .ok_or_else(|| ProviderError::NotFound(name.to_string()))?;
    create_from_config(name, cfg)
}

pub fn create_from_config(name: &str, cfg: &ProviderConfig) -> Result<ProviderHandle> {
    match cfg.kind.as_str() {
        "ollama" => {
            let base_url = cfg
                .base_url
                .clone()
                .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());
            info!(provider = name, %base_url, "using ollama provider");
            Ok(Arc::new(OllamaProvider::new(name, base_url)))
        }
        "lmstudio" => {
            let base_url = cfg
                .base_url
                .clone()
                .unwrap_or_else(|| "http://127.0.0.1:1234/v1".to_string());
            info!(provider = name, %base_url, "using lmstudio provider");
            Ok(Arc::new(OpenAiCompatProvider::new(name, base_url)))
        }
        "llamacpp" => {
            let base_url = cfg
                .base_url
                .clone()
                .unwrap_or_else(|| "http://127.0.0.1:8080/v1".to_string());
            info!(provider = name, %base_url, "using llama.cpp provider");
            Ok(Arc::new(OpenAiCompatProvider::new(name, base_url)))
        }
        // Hosted OpenAI-compatible endpoints. OpenCode Zen and OpenRouter are
        // gateways; openai/grok are first-party; gemini exposes an
        // OpenAI-compatible route under `/v1beta/openai`.
        "openai" => cloud_openai_compat(name, cfg, "https://api.openai.com/v1"),
        "deepseek" => cloud_openai_compat(name, cfg, "https://api.deepseek.com/v1"),
        "grok" => cloud_openai_compat(name, cfg, "https://api.x.ai/v1"),
        "openrouter" => cloud_openai_compat(name, cfg, "https://openrouter.ai/api/v1"),
        "opencodezen" => cloud_openai_compat(name, cfg, "https://opencode.ai/zen/v1"),
        "gemini" => cloud_openai_compat(
            name,
            cfg,
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
        "anthropic" => {
            let base_url = cfg
                .base_url
                .clone()
                .unwrap_or_else(|| "https://api.anthropic.com".to_string());
            let key = resolve_api_key(cfg, name)?;
            let provider = match key {
                Some(k) => AnthropicProvider::new(name, &base_url).with_api_key(k),
                None => AnthropicProvider::new(name, &base_url),
            };
            info!(provider = name, %base_url, "using anthropic provider");
            Ok(Arc::new(provider))
        }
        other => Err(ProviderError::UnsupportedKind(other.to_string())),
    }
}

fn cloud_openai_compat(
    name: &str,
    cfg: &ProviderConfig,
    default_base: &str,
) -> Result<ProviderHandle> {
    let base_url = cfg
        .base_url
        .clone()
        .unwrap_or_else(|| default_base.to_string());
    let key = resolve_api_key(cfg, name)?;
    let mut p = OpenAiCompatProvider::new(name, base_url);
    if let Some(k) = key {
        p = p.with_api_key(k);
    }
    p = p.with_headers(resolve_headers(cfg));
    info!(provider = name, "using openai-compatible cloud provider");
    Ok(Arc::new(p))
}

/// Create a provider from its config.
pub fn create_default(provider_name: &str, file: &ProvidersFile) -> Result<ProviderHandle> {
    match create_provider(provider_name, file) {
        Ok(p) => Ok(p),
        // A provider that fails to construct (e.g. missing API key) surfaces as
        // a real error. No silent swap.
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeus_config::ProvidersFile;

    #[test]
    fn mock_kind_is_rejected() {
        let file = ProvidersFile::builtin_defaults();
        match create_provider("mock", &file) {
            Ok(_) => panic!("expected mock provider to be unsupported"),
            Err(err) => assert!(matches!(err, ProviderError::NotFound(_))),
        }
    }

    #[test]
    fn unknown_provider_errors() {
        let file = ProvidersFile::builtin_defaults();
        match create_provider("nope", &file) {
            Ok(_) => panic!("expected NotFound"),
            Err(err) => assert!(matches!(err, ProviderError::NotFound(_))),
        }
    }

    #[test]
    fn cloud_kind_without_key_errors_instead_of_falling_back() {
        let file = ProvidersFile::builtin_defaults();
        // "openai" is a real OpenAI-compatible client now; without a key it
        // must surface a MissingApiKey error — not silently fall back to a
        // fake provider, which would hide a misconfigured remote provider.
        match create_provider("openai", &file) {
            Ok(_) => panic!("expected MissingApiKey for unconfigured openai"),
            Err(err) => assert!(matches!(err, ProviderError::MissingApiKey(_))),
        }
    }

    #[test]
    fn openai_constructs_with_embedded_key() {
        let mut file = ProvidersFile::builtin_defaults();
        if let Some(cfg) = file.providers.get_mut("openai") {
            cfg.api_key_env = None;
            cfg.headers
                .insert("Authorization".into(), "Bearer sk-test-key".into());
        }
        let p = create_provider("openai", &file).unwrap();
        assert_eq!(p.id(), "openai");
    }

    #[test]
    fn ollama_constructs_directly_without_falling_back() {
        let file = ProvidersFile::builtin_defaults();
        let p = create_default("ollama", &file).unwrap();
        assert_eq!(p.id(), "ollama");
    }

    #[test]
    fn opencodezen_is_a_registered_provider_kind() {
        let mut file = ProvidersFile::builtin_defaults();
        // Point at a guaranteed-absent env var so the "no key configured"
        // expectation doesn't depend on whatever key vars the machine running
        // the test happens to have set (e.g. a real OPENCODE_API_KEY).
        if let Some(cfg) = file.providers.get_mut("opencodezen") {
            cfg.api_key_env = Some("ZEUS_TEST_UNSET_OPENCODEZEN_KEY".into());
        }
        match create_provider("opencodezen", &file) {
            Ok(_) => panic!("expected MissingApiKey for unconfigured opencodezen"),
            Err(err) => assert!(matches!(err, ProviderError::MissingApiKey(_))),
        }
    }
}
