//! Ollama HTTP provider — talks to a local `ollama serve` instance
//! (default `http://127.0.0.1:11434`).

use crate::error::{ProviderError, Result};
use crate::heuristics::estimate_messages;
use crate::types::*;
use crate::{ChatStream, ModelProvider};
use async_trait::async_trait;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::ReceiverStream;

pub struct OllamaProvider {
    id: String,
    base_url: String,
    client: reqwest::Client,
}

impl OllamaProvider {
    pub fn new(id: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
        }
    }

    fn chat_url(&self) -> String {
        format!("{}/api/chat", self.base_url)
    }
    fn tags_url(&self) -> String {
        format!("{}/api/tags", self.base_url)
    }
    fn embed_url(&self) -> String {
        format!("{}/api/embed", self.base_url)
    }
    fn pull_url(&self) -> String {
        format!("{}/api/pull", self.base_url)
    }

    /// Pull a model via Ollama's own `/api/pull` (what `ollama pull <model>`
    /// uses under the hood) — streams NDJSON progress lines, calling
    /// `on_progress` with each raw status string (e.g. "pulling manifest",
    /// "downloading sha256:...", "verifying sha256:...", "success"). Once
    /// pulled, the model is Ollama's own to manage — it shows up in
    /// `list_models()` automatically, no separate "auto-detect" step needed
    /// for this path (unlike a raw file downloaded straight from
    /// Hugging Face, which needs the local-model-file scan instead).
    pub async fn pull(&self, model: &str, mut on_progress: impl FnMut(&str)) -> Result<()> {
        #[derive(serde::Deserialize)]
        struct PullStatus {
            #[serde(default)]
            status: String,
            #[serde(default)]
            error: Option<String>,
        }

        let body = serde_json::json!({ "name": model, "stream": true });
        let resp = self
            .client
            .post(self.pull_url())
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_err)?;
        let resp = error_for_status(resp).await?;

        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(map_reqwest_err)?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer.drain(..=pos);
                if line.is_empty() {
                    continue;
                }
                let status: PullStatus = serde_json::from_str(&line)
                    .map_err(|e| ProviderError::Api(format!("bad pull status line: {e}")))?;
                if let Some(err) = status.error {
                    return Err(ProviderError::Api(err));
                }
                on_progress(&status.status);
            }
        }
        Ok(())
    }
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn to_ollama_message(m: &Message) -> Value {
    let mut obj = json!({
        "role": role_str(m.role),
        "content": m.content,
    });
    if !m.tool_calls.is_empty() {
        let calls: Vec<Value> = m
            .tool_calls
            .iter()
            .map(|c| {
                let args: Value = serde_json::from_str(&c.arguments).unwrap_or(json!({}));
                json!({ "function": { "name": c.name, "arguments": args } })
            })
            .collect();
        obj["tool_calls"] = json!(calls);
    }
    if !m.images.is_empty() {
        obj["images"] = json!(m
            .images
            .iter()
            .map(|img| img.data_base64.clone())
            .collect::<Vec<_>>());
    }
    obj
}

fn to_ollama_tools(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            })
        })
        .collect()
}

fn build_chat_body(request: &ChatRequest, stream: bool) -> Value {
    let mut body = json!({
        "model": request.model,
        "messages": request.messages.iter().map(to_ollama_message).collect::<Vec<_>>(),
        "stream": stream,
    });
    if !request.tools.is_empty() {
        body["tools"] = json!(to_ollama_tools(&request.tools));
    }
    let mut options = serde_json::Map::new();
    if let Some(t) = request.temperature {
        options.insert("temperature".into(), json!(t));
    }
    if let Some(m) = request.max_tokens {
        options.insert("num_predict".into(), json!(m));
    }
    if !options.is_empty() {
        body["options"] = Value::Object(options);
    }
    body
}

#[derive(Debug, Deserialize)]
struct OllamaToolCallFn {
    name: String,
    arguments: Value,
}
#[derive(Debug, Deserialize)]
struct OllamaToolCall {
    function: OllamaToolCallFn,
}
#[derive(Debug, Deserialize, Default)]
struct OllamaMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Vec<OllamaToolCall>,
}
#[derive(Debug, Deserialize)]
struct OllamaChatChunk {
    #[serde(default)]
    message: Option<OllamaMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    error: Option<String>,
}

fn map_reqwest_err(e: reqwest::Error) -> ProviderError {
    ProviderError::Transport(e.to_string())
}

async fn error_for_status(resp: reqwest::Response) -> Result<reqwest::Response> {
    if resp.status().is_success() {
        return Ok(resp);
    }
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    Err(ProviderError::Api(format!("ollama {status}: {text}")))
}

fn tool_calls_from(msg: &OllamaMessage, id_offset: usize) -> Vec<ToolCall> {
    msg.tool_calls
        .iter()
        .enumerate()
        .map(|(i, c)| ToolCall {
            id: format!("call-{}", id_offset + i),
            name: c.function.name.clone(),
            arguments: c.function.arguments.to_string(),
        })
        .collect()
}

#[async_trait]
impl ModelProvider for OllamaProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        if request.is_cancelled() {
            return Err(ProviderError::Cancelled);
        }
        let model = request.model.clone();
        let body = build_chat_body(&request, false);
        let resp = self
            .client
            .post(self.chat_url())
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_err)?;
        let resp = error_for_status(resp).await?;
        let chunk: OllamaChatChunk = resp.json().await.map_err(map_reqwest_err)?;
        if let Some(err) = chunk.error {
            return Err(ProviderError::Api(err));
        }
        let ollama_msg = chunk.message.unwrap_or_default();
        let tool_calls = tool_calls_from(&ollama_msg, 0);
        let finish_reason = if tool_calls.is_empty() {
            FinishReason::Stop
        } else {
            FinishReason::ToolCalls
        };
        let mut message = Message::assistant(ollama_msg.content);
        message.tool_calls = tool_calls;

        Ok(ChatResponse {
            message,
            usage: TokenUsage::new(
                chunk.prompt_eval_count.unwrap_or(0),
                chunk.eval_count.unwrap_or(0),
            ),
            finish_reason,
            model,
        })
    }

    async fn stream(&self, request: ChatRequest) -> Result<ChatStream> {
        if request.is_cancelled() {
            return Err(ProviderError::Cancelled);
        }
        let cancel = request.cancel.clone();
        let body = build_chat_body(&request, true);
        let resp = self
            .client
            .post(self.chat_url())
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_err)?;
        let resp = error_for_status(resp).await?;

        // Ollama streams NDJSON (one JSON object per line); network chunk
        // boundaries don't align with line boundaries, so a spawned task
        // owns the byte stream, buffers partial lines, and pushes parsed
        // events into a channel — simpler to reason about than a hand-rolled
        // `stream::unfold` state machine over the raw byte stream.
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamEvent>>(32);

        tokio::spawn(async move {
            let mut byte_stream = resp.bytes_stream();
            let mut buffer = String::new();
            let mut saw_tool_call = false;
            let mut next_call_id = 0usize;

            loop {
                if cancel.as_ref().map(|rx| *rx.borrow()).unwrap_or(false) {
                    let _ = tx
                        .send(Ok(StreamEvent::Done {
                            finish_reason: FinishReason::Cancelled,
                            usage: TokenUsage::default(),
                        }))
                        .await;
                    return;
                }

                let chunk = match byte_stream.next().await {
                    Some(Ok(bytes)) => bytes,
                    Some(Err(e)) => {
                        let _ = tx.send(Err(map_reqwest_err(e))).await;
                        return;
                    }
                    None => {
                        // Stream closed without an explicit `done` line —
                        // treat as a clean stop rather than hanging forever.
                        let _ = tx
                            .send(Ok(StreamEvent::Done {
                                finish_reason: FinishReason::Stop,
                                usage: TokenUsage::default(),
                            }))
                            .await;
                        return;
                    }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer.drain(..=pos);
                    if line.is_empty() {
                        continue;
                    }
                    let parsed: std::result::Result<OllamaChatChunk, _> =
                        serde_json::from_str(&line);
                    let parsed = match parsed {
                        Ok(c) => c,
                        Err(e) => {
                            let _ = tx
                                .send(Err(ProviderError::Api(format!("bad ollama chunk: {e}"))))
                                .await;
                            return;
                        }
                    };
                    if let Some(err) = parsed.error {
                        let _ = tx.send(Err(ProviderError::Api(err))).await;
                        return;
                    }
                    if let Some(msg) = &parsed.message {
                        if !msg.content.is_empty()
                            && tx
                                .send(Ok(StreamEvent::TextDelta {
                                    text: msg.content.clone(),
                                }))
                                .await
                                .is_err()
                        {
                            return;
                        }
                        for call in tool_calls_from(msg, next_call_id) {
                            saw_tool_call = true;
                            if tx
                                .send(Ok(StreamEvent::ToolCallDelta {
                                    id: call.id,
                                    name: Some(call.name),
                                    arguments_delta: call.arguments,
                                }))
                                .await
                                .is_err()
                            {
                                return;
                            }
                        }
                        next_call_id += msg.tool_calls.len();
                    }
                    if parsed.done {
                        let _ = tx
                            .send(Ok(StreamEvent::Done {
                                finish_reason: if saw_tool_call {
                                    FinishReason::ToolCalls
                                } else {
                                    FinishReason::Stop
                                },
                                usage: TokenUsage::new(
                                    parsed.prompt_eval_count.unwrap_or(0),
                                    parsed.eval_count.unwrap_or(0),
                                ),
                            }))
                            .await;
                        return;
                    }
                }
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        #[derive(Deserialize)]
        struct Details {
            #[serde(default)]
            context_length: Option<u32>,
        }
        #[derive(Deserialize)]
        struct Tag {
            name: String,
            #[serde(default)]
            details: Option<Details>,
        }
        #[derive(Deserialize)]
        struct TagsResponse {
            #[serde(default)]
            models: Vec<Tag>,
        }

        let resp = self
            .client
            .get(self.tags_url())
            .send()
            .await
            .map_err(map_reqwest_err)?;
        let resp = error_for_status(resp).await?;
        let parsed: TagsResponse = resp.json().await.map_err(map_reqwest_err)?;
        Ok(parsed
            .models
            .into_iter()
            .map(|t| ModelInfo {
                id: t.name.clone(),
                name: t.name,
                context_window: t.details.and_then(|d| d.context_length),
            })
            .collect())
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        #[derive(Deserialize)]
        struct EmbedResponse {
            #[serde(default)]
            embeddings: Vec<Vec<f32>>,
        }
        let body = json!({ "model": request.model, "input": request.input });
        let resp = self
            .client
            .post(self.embed_url())
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_err)?;
        let resp = error_for_status(resp).await?;
        let parsed: EmbedResponse = resp.json().await.map_err(map_reqwest_err)?;
        let tokens = estimate_messages(
            &request
                .input
                .iter()
                .map(|s| Message::user(s.clone()))
                .collect::<Vec<_>>(),
            &[],
        );
        Ok(EmbeddingResponse {
            vectors: parsed.embeddings,
            usage: TokenUsage::new(tokens, 0),
        })
    }

    async fn count_tokens(&self, request: TokenCountRequest) -> Result<TokenCountResponse> {
        // Ollama has no free-standing tokenize endpoint across all server
        // versions; fall back to the same approximate heuristic as other
        // provider, honestly marked `approximate: true`.
        Ok(TokenCountResponse {
            tokens: estimate_messages(&request.messages, &request.tools),
            approximate: true,
        })
    }

    fn supports_prompt_cache(&self) -> bool {
        false
    }
}
