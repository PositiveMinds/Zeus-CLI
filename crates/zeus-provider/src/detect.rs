//! Local-provider auto-detection: a quick reachability probe so the CLI can
//! notice "the configured default (e.g. Ollama) isn't actually running" and
//! either switch to another local server that *is* running, or fail with a
//! clear message — instead of hanging or failing with an opaque connection
//! error deep inside the first real chat call.

use std::time::Duration;
use zeus_config::{ProviderConfig, ProvidersFile};

/// Local kinds this module knows how to probe, in priority order — first
/// reachable one wins when auto-detecting an alternative.
const LOCAL_KIND_PRIORITY: &[&str] = &["ollama", "lmstudio", "llamacpp"];

/// The lightweight endpoint to hit for a quick "is anything listening and
/// answering HTTP here" check — not a full chat request. `None` for kinds
/// this module doesn't know how to probe (cloud providers); callers
/// should treat that as "assume reachable, not our job to check."
fn local_probe_url(cfg: &ProviderConfig) -> Option<String> {
    let base = cfg.base_url.as_deref()?.trim_end_matches('/');
    match cfg.kind.as_str() {
        "ollama" => Some(format!("{base}/api/tags")),
        "lmstudio" | "llamacpp" => Some(format!("{base}/models")),
        _ => None,
    }
}

/// True if an HTTP request to `url` completes at all (any status code)
/// within `timeout` — a connection refused/timeout means false. Deliberately
/// lenient on status code: even a 404 proves *something* HTTP-shaped is
/// listening, which is all this heuristic claims.
pub async fn is_reachable(url: &str, timeout: Duration) -> bool {
    let client = reqwest::Client::new();
    matches!(
        tokio::time::timeout(timeout, client.get(url).send()).await,
        Ok(Ok(_))
    )
}

/// Reachability check for a specific configured provider. Kinds this module
/// doesn't probe (cloud APIs) are assumed reachable — checking them is
/// out of scope here; their own request will surface a real error if not.
pub async fn is_provider_reachable(cfg: &ProviderConfig, timeout: Duration) -> bool {
    match local_probe_url(cfg) {
        Some(url) => is_reachable(&url, timeout).await,
        None => true,
    }
}

/// Probe every configured local-kind provider in priority order
/// (ollama > lmstudio > llamacpp) and return the name of the first one
/// that's actually reachable right now. `None` if none are.
pub async fn detect_local_provider(providers: &ProvidersFile) -> Option<String> {
    for kind in LOCAL_KIND_PRIORITY {
        for (name, cfg) in &providers.providers {
            if cfg.kind == *kind && is_provider_reachable(cfg, Duration::from_millis(800)).await {
                return Some(name.clone());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn cfg(kind: &str, base_url: Option<&str>) -> ProviderConfig {
        ProviderConfig {
            kind: kind.to_string(),
            base_url: base_url.map(String::from),
            api_key_env: None,
            default_model: None,
            headers: HashMap::new(),
            embeddings: false,
            prompt_cache: false,
        }
    }

    #[test]
    fn probe_url_matches_expected_endpoint_per_kind() {
        assert_eq!(
            local_probe_url(&cfg("ollama", Some("http://127.0.0.1:11434"))),
            Some("http://127.0.0.1:11434/api/tags".to_string())
        );
        assert_eq!(
            local_probe_url(&cfg("lmstudio", Some("http://127.0.0.1:1234/v1"))),
            Some("http://127.0.0.1:1234/v1/models".to_string())
        );
        assert_eq!(
            local_probe_url(&cfg("llamacpp", Some("http://127.0.0.1:8080/v1/"))),
            Some("http://127.0.0.1:8080/v1/models".to_string())
        );
        assert_eq!(local_probe_url(&cfg("anthropic", Some("https://api.anthropic.com"))), None);
        assert_eq!(local_probe_url(&cfg("openai", Some("https://api.openai.com/v1"))), None);
    }

    #[tokio::test]
    async fn unreachable_port_is_not_reachable() {
        // Port 1 is a reserved/unlikely-bound port; a short timeout keeps
        // this test fast even if something unexpected answers.
        let reachable = is_reachable("http://127.0.0.1:1/nope", Duration::from_millis(300)).await;
        assert!(!reachable);
    }

    #[tokio::test]
    async fn detect_returns_none_when_nothing_configured_is_reachable() {
        let mut providers = HashMap::new();
        providers.insert(
            "ollama".to_string(),
            cfg("ollama", Some("http://127.0.0.1:1")),
        );
        let file = ProvidersFile { providers };
        assert_eq!(detect_local_provider(&file).await, None);
    }

    #[tokio::test]
    async fn non_local_kind_is_assumed_reachable_without_probing() {
        assert!(is_provider_reachable(&cfg("anthropic", Some("https://ai.io")), Duration::from_millis(100)).await);
        assert!(
            is_provider_reachable(
                &cfg("openai", Some("https://api.openai.com/v1")),
                Duration::from_millis(100)
            )
            .await
        );
    }
}
