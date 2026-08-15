//! Static fallback model catalogs for cloud providers whose live `/models`
//! probe can't be reached (missing API key, network down, timeout). The
//! model picker then still shows *every* configured cloud provider grouped,
//! with as many models as possible, instead of silently dropping providers
//! that happened to fail the probe. Picking a model from a key-less provider
//! opens the key-entry screen, so these rows are always actionable.
//!
//! Local providers (`ollama`/`lmstudio`/`llamacpp`) are intentionally absent:
//! their catalogs are whatever is downloaded on this machine, and the live
//! probe is the only honest source for them.

use zeus_provider::ModelInfo;

fn catalog(ids: &[&str]) -> Vec<ModelInfo> {
    ids.iter()
        .map(|id| ModelInfo {
            id: id.to_string(),
            name: id.to_string(),
            context_window: None,
        })
        .collect()
}

/// Curated model list for `provider`, or `None` when the provider is a
/// local one (probe-only) or unknown. Mirrors the naming the providers'
/// own APIs use today; a live probe result always wins over this.
pub(crate) fn known_models(provider: &str) -> Option<Vec<ModelInfo>> {
    Some(catalog(match provider {
        "openai" => OPENAI,
        "anthropic" => ANTHROPIC,
        "gemini" => GEMINI,
        "grok" => GROK,
        "deepseek" => DEEPSEEK,
        "mistral" => MISTRAL,
        "groq" => GROQ,
        "together" => TOGETHER,
        "fireworks" => FIREWORKS,
        "openrouter" => OPENROUTER,
        "opencodezen" => OPENCODEZEN,
        _ => return None,
    }))
}

const OPENAI: &[&str] = &[
    "gpt-5.6",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.3",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o3",
    "o3-mini",
    "o4-mini",
    "o1",
    "gpt-4.5-preview",
    "chatgpt-4o-latest",
];

const ANTHROPIC: &[&str] = &[
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-opus-4",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-sonnet-4",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5",
    "claude-3-7-sonnet",
    "claude-3-5-sonnet",
    "claude-3-5-haiku",
];

const GEMINI: &[&str] = &[
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
];

const GROK: &[&str] = &[
    "grok-4.6",
    "grok-4.5",
    "grok-4",
    "grok-3",
    "grok-3-fast",
    "grok-3-mini",
];

const DEEPSEEK: &[&str] = &[
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-reasoner",
    "deepseek-chat",
];

const MISTRAL: &[&str] = &[
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
    "codestral-latest",
    "ministral-3b",
];

const GROQ: &[&str] = &[
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.2-90b-text-preview",
    "deepseek-r1-distill-llama-70b",
    "qwen-2.5-coder-32b",
];

const TOGETHER: &[&str] = &[
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Llama-3.1-405B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
];

const FIREWORKS: &[&str] = &[
    "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "accounts/fireworks/models/llama-v3p1-405b-instruct",
    "accounts/fireworks/models/deepseek-v3",
    "accounts/fireworks/models/deepseek-r1",
    "accounts/fireworks/models/qwen2p5-coder-32b-instruct",
];

const OPENROUTER: &[&str] = &[
    "openrouter/auto",
    "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4",
    "openai/gpt-5",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
];

/// Mirrors the live opencodezen catalog so the recommended provider still
/// shows its full list when the probe is unreachable.
const OPENCODEZEN: &[&str] = &[
    "claude-fable-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-sonnet-4",
    "claude-haiku-4-5",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-pro",
    "gemini-3-flash",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.3-codex-spark",
    "gpt-5.3-codex",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.1",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5",
    "gpt-5-codex",
    "gpt-5-nano",
    "grok-build-0.1",
    "grok-4.6",
    "grok-4.5",
    "muse-spark-1.2",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "glm-5.2",
    "glm-5.1",
    "glm-5",
    "minimax-m3",
    "minimax-m2.7",
    "minimax-m2.5",
    "kimi-k3",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "kimi-k2.5",
    "qwen3.6-plus",
    "qwen3.5-plus",
    "big-pickle",
    "deepseek-v4-flash-free",
    "mimo-v2.5-free",
    "hy3-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
    "laguna-s-2.1-free",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_cloud_providers_have_fallbacks() {
        for p in [
            "openai",
            "anthropic",
            "gemini",
            "grok",
            "deepseek",
            "mistral",
            "groq",
            "together",
            "fireworks",
            "openrouter",
            "opencodezen",
        ] {
            let m = known_models(p).unwrap_or_else(|| panic!("no catalog for {p}"));
            assert!(!m.is_empty(), "empty catalog for {p}");
        }
    }

    #[test]
    fn local_providers_have_no_fallback() {
        for p in ["ollama", "lmstudio", "llamacpp", "unknown-provider"] {
            assert!(known_models(p).is_none(), "unexpected catalog for {p}");
        }
    }

    #[test]
    fn model_ids_are_unique_per_catalog() {
        for p in [
            "openai",
            "anthropic",
            "gemini",
            "grok",
            "deepseek",
            "mistral",
            "groq",
            "together",
            "fireworks",
            "openrouter",
            "opencodezen",
        ] {
            let m = known_models(p).unwrap();
            let ids: Vec<_> = m.iter().map(|x| x.id.as_str()).collect();
            let mut uniq = ids.clone();
            uniq.sort();
            uniq.dedup();
            assert_eq!(ids.len(), uniq.len(), "duplicate model ids for {p}");
        }
    }

    #[test]
    fn openrouter_catalog_uses_api_model_ids() {
        // OpenRouter model IDs are bare (`deepseek/deepseek-chat`,
        // `anthropic/claude-opus-4`) — the only `openrouter/…` prefixed ID
        // the API accepts is the auto-routing `openrouter/auto`. Anything
        // else with that prefix gets a 400 from the live endpoint.
        for m in known_models("openrouter").unwrap() {
            if m.id == "openrouter/auto" {
                continue;
            }
            assert!(
                !m.id.starts_with("openrouter/"),
                "invalid OpenRouter model id: {}",
                m.id
            );
            assert!(
                m.id.contains('/'),
                "OpenRouter model ids carry an org/owner prefix: {}",
                m.id
            );
        }
    }
}
