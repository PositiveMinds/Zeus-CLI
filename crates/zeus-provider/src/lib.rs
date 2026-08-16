//! Provider-agnostic LLM interface.
//!
//! ```rust,ignore
//! trait ModelProvider {
//!     async fn chat();
//!     async fn stream();          // must support cancellation mid-stream
//!     async fn list_models();
//!     async fn embeddings();
//!     async fn count_tokens();     // for context-budget accounting
//!     fn supports_prompt_cache() -> bool;
//! }
//! ```

mod anthropic;
mod detect;
mod download;
mod error;
mod heuristics;
mod llamacpp;
mod local_models;
mod ollama;
mod openai_compat;
mod registry;
mod types;
mod unconfigured;

pub use anthropic::AnthropicProvider;
pub use detect::{detect_local_provider, is_provider_reachable, is_reachable};
pub use download::{download_asset, download_hf_file};
pub use error::{ProviderError, Result};
pub use llamacpp::{
    ensure_model_file, ensure_server_binary, find_on_path, locate_server_binary,
    resolve_local_model, serve, spawn_server_and_wait, ServerInfo, DEFAULT_MODEL_CATALOG,
};
pub use local_models::{import_model_file, scan_local_models, scan_system_models, LocalModelFile};
pub use ollama::OllamaProvider;
pub use openai_compat::OpenAiCompatProvider;
pub use registry::{create_default, create_provider, ProviderHandle};
pub use types::*;
pub use unconfigured::UnconfiguredProvider;

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

/// Stream of incremental chat events (tokens / tool calls / end).
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>;

/// Core model provider abstraction. All backends implement this trait.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Provider id (e.g. "ollama", "openai", "anthropic").
    fn id(&self) -> &str;

    /// Non-streaming chat completion.
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse>;

    /// Streaming chat; must honor `request.cancel` / drop cancellation.
    async fn stream(&self, request: ChatRequest) -> Result<ChatStream>;

    /// List models available from this provider.
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;

    /// Embedding vectors for the given texts.
    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse>;

    /// Count tokens for context-budget accounting.
    /// Approximate when the backend has no exact counter.
    async fn count_tokens(&self, request: TokenCountRequest) -> Result<TokenCountResponse>;

    /// Whether the provider can cache stable system/tool blocks.
    fn supports_prompt_cache(&self) -> bool;
}
