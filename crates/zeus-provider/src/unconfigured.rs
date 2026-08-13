//! Placeholder provider used when no provider is configured or has an API
//! key at startup. It lets the interactive UI launch in a "setup" state so the
//! user can configure providers/keys in-app (`/provider`, `/provider key`)
//! instead of being locked out at the very first command. Any actual request
//! (chat, stream, embeddings, tokens) surfaces a clear, actionable error.

use crate::{error::ProviderError, types::*, ChatStream};
use async_trait::async_trait;

/// Info to show in the UI so the error message is actionable.
pub struct UnconfiguredProvider {
    /// The provider/model the user tried to use, if known.
    pub requested: Option<String>,
}

impl UnconfiguredProvider {
    fn error(&self) -> ProviderError {
        let what = self
            .requested
            .as_deref()
            .unwrap_or("this provider is not configured");
        ProviderError::Api(format!(
            "no provider is ready. Run `/provider` to see your providers, `/provider <name>` to \
             switch, and `/provider key <name> <KEY>` to set a key ({what})."
        ))
    }
}

#[async_trait]
impl super::ModelProvider for UnconfiguredProvider {
    fn id(&self) -> &str {
        "unconfigured"
    }

    async fn chat(&self, _request: ChatRequest) -> crate::Result<ChatResponse> {
        Err(self.error())
    }

    async fn stream(&self, _request: ChatRequest) -> crate::Result<ChatStream> {
        Err(self.error())
    }

    async fn list_models(&self) -> crate::Result<Vec<ModelInfo>> {
        Ok(vec![])
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> crate::Result<EmbeddingResponse> {
        Err(self.error())
    }

    async fn count_tokens(&self, _request: TokenCountRequest) -> crate::Result<TokenCountResponse> {
        Err(self.error())
    }

    fn supports_prompt_cache(&self) -> bool {
        false
    }
}
