//! Native Anthropic Messages API client (`/v1/messages`).
//!
//! Claude's wire format differs from OpenAI chat-completions: it uses
//! content *blocks* (`text`, `tool_use`, `tool_result`), a separate top-level
//! `system` field, and a distinct SSE event stream (`content_block_delta`,
//! `message_delta`, …). An OpenAI-compatible shim would risk mangling the
//! tool-call block structure, so Claude gets its own transport here.

use crate::error::{ProviderError, Result};
use crate::heuristics::estimate_messages;
use crate::types::*;
use crate::ChatStream;
use crate::ModelProvider;
use async_trait::async_trait;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct AnthropicProvider {
    id: String,
    base_url: String,
    client: reqwest::Client,
    api_key: Option<String>,
    extra_headers: HashMap<String, String>,
}

impl AnthropicProvider {
    pub fn new(id: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
            api_key: None,
            extra_headers: HashMap::new(),
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_headers<I: IntoIterator<Item = (String, String)>>(mut self, headers: I) -> Self {
        self.extra_headers.extend(headers);
        self
    }

    fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut map = reqwest::header::HeaderMap::new();
        for (k, v) in &self.extra_headers {
            if let Ok(name) = reqwest::header::HeaderName::from_bytes(k.as_bytes()) {
                if let Ok(value) = v.parse() {
                    map.insert(name, value);
                }
            }
        }
        if let Some(key) = &self.api_key {
            map.insert(
                "x-api-key",
                key.parse()
                    .unwrap_or_else(|_| reqwest::header::HeaderValue::from_static("")),
            );
        }
        map.insert(
            "anthropic-version",
            reqwest::header::HeaderValue::from_static("2023-06-01"),
        );
        map
    }
}

/// Build Anthropic `messages` (list of role + content blocks), collapsing
/// consecutive same-role turns and grouping tool results into the ambient
/// user message.
fn to_anthropic_messages(messages: &[Message]) -> Vec<Value> {
    let mut out: Vec<(String, Vec<Value>)> = Vec::new();
    for m in messages {
        match m.role {
            Role::System => continue, // handled via top-level `system`
            Role::User | Role::Tool => {
                let mut blocks: Vec<Value> = if m.role == Role::Tool {
                    vec![json!({
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                        "content": m.content
                    })]
                } else {
                    vec![json!({ "type": "text", "text": m.content })]
                };
                for img in &m.images {
                    blocks.push(json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.mime_type,
                            "data": img.data_base64,
                        }
                    }));
                }
                if matches!(out.last(), Some((role, _)) if role == "user") {
                    out.last_mut().unwrap().1.extend(blocks);
                } else {
                    out.push(("user".to_string(), blocks));
                }
            }
            Role::Assistant => {
                let mut blocks = vec![json!({ "type": "text", "text": m.content })];
                for tc in &m.tool_calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": parse_tool_input(&tc.arguments),
                    }));
                }
                if matches!(out.last(), Some((role, _)) if role == "assistant") {
                    out.last_mut().unwrap().1.extend(blocks);
                } else {
                    out.push(("assistant".to_string(), blocks));
                }
            }
        }
    }
    out.into_iter()
        .map(|(role, blocks)| json!({ "role": role, "content": blocks }))
        .collect()
}

/// Anthropic keeps JSON args as objects; parse our string form.
fn parse_tool_input(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or(Value::Null)
}

fn system_text(messages: &[Message]) -> String {
    messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.content.clone())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn to_anthropic_tools(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            })
        })
        .collect()
}

fn build_body(request: &ChatRequest, stream: bool) -> Value {
    let mut body = json!({
        "model": request.model,
        "max_tokens": request.max_tokens.unwrap_or(8192),
        "messages": to_anthropic_messages(&request.messages),
        "stream": stream,
    });
    let system = system_text(&request.messages);
    if !system.is_empty() {
        body["system"] = json!(system);
    }
    if !request.tools.is_empty() {
        body["tools"] = json!(to_anthropic_tools(&request.tools));
    }
    if let Some(t) = request.temperature {
        body["temperature"] = json!(t);
    }
    body
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
    Err(ProviderError::Http {
        status: status.as_u16(),
        message: text,
    })
}

// --- Non-streaming response shape ---

#[derive(Debug, Deserialize)]
struct AnBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<Value>,
}
#[derive(Debug, Deserialize)]
struct AnUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}
#[derive(Debug, Deserialize)]
struct AnMessage {
    #[serde(default)]
    content: Vec<AnBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<AnUsage>,
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        if request.is_cancelled() {
            return Err(ProviderError::Cancelled);
        }
        let model = request.model.clone();
        let body = build_body(&request, false);
        let resp = crate::retry::with_retry(|| {
            let body = body.clone();
            async move {
                let resp = self
                    .client
                    .post(self.messages_url())
                    .headers(self.headers())
                    .json(&body)
                    .send()
                    .await
                    .map_err(map_reqwest_err)?;
                error_for_status(resp).await
            }
        })
        .await?;
        let parsed: AnMessage = resp.json().await.map_err(map_reqwest_err)?;

        let mut text = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for block in parsed.content {
            match block.kind.as_str() {
                "text" => {
                    if let Some(t) = block.text {
                        text.push_str(&t);
                    }
                }
                "tool_use" => {
                    tool_calls.push(ToolCall {
                        id: block.id.unwrap_or_default(),
                        name: block.name.unwrap_or_default(),
                        arguments: block.input.map(|v| v.to_string()).unwrap_or_default(),
                        extra_content: None,
                    });
                }
                _ => {}
            }
        }
        let mut message = Message::assistant(text);
        message.tool_calls = tool_calls;
        let usage = parsed
            .usage
            .map(|u| TokenUsage::new(u.input_tokens, u.output_tokens))
            .unwrap_or_default();
        let finish_reason = match parsed.stop_reason.as_deref() {
            Some("tool_use") => FinishReason::ToolCalls,
            Some("max_tokens") => FinishReason::Length,
            _ if !message.tool_calls.is_empty() => FinishReason::ToolCalls,
            _ => FinishReason::Stop,
        };
        Ok(ChatResponse {
            message,
            usage,
            finish_reason,
            model,
        })
    }

    async fn stream(&self, request: ChatRequest) -> Result<ChatStream> {
        if request.is_cancelled() {
            return Err(ProviderError::Cancelled);
        }
        let cancel = request.cancel.clone();
        let body = build_body(&request, true);
        let resp = crate::retry::with_retry(|| {
            let body = body.clone();
            async move {
                let resp = self
                    .client
                    .post(self.messages_url())
                    .headers(self.headers())
                    .json(&body)
                    .send()
                    .await
                    .map_err(map_reqwest_err)?;
                error_for_status(resp).await
            }
        })
        .await?;

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamEvent>>(32);

        tokio::spawn(async move {
            let mut byte_stream = resp.bytes_stream();
            let mut buffer = String::new();
            let mut saw_tool_call = false;
            // Anthropic streams keep an incrementing block index and a partial
            // tool input; track current text / tool-args buffers.
            let mut text_buf = String::new();
            let mut tool_buf: HashMap<u64, (String, String, String)> = HashMap::new(); // index -> (id, name, args)
            let mut last_usage = TokenUsage::default();
            let mut stop_reason: Option<String> = None;

            loop {
                if cancel.as_ref().map(|rx| *rx.borrow()).unwrap_or(false) {
                    let _ = tx
                        .send(Ok(StreamEvent::Done {
                            finish_reason: FinishReason::Cancelled,
                            usage: last_usage,
                        }))
                        .await;
                    return;
                }

                let chunk = match byte_stream.next().await {
                    Some(Ok(b)) => b,
                    Some(Err(e)) => {
                        let _ = tx.send(Err(map_reqwest_err(e))).await;
                        return;
                    }
                    None => {
                        let _ = tx
                            .send(Ok(StreamEvent::Done {
                                finish_reason: map_anthropic_reason(
                                    stop_reason.as_deref(),
                                    saw_tool_call,
                                ),
                                usage: last_usage,
                            }))
                            .await;
                        return;
                    }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer.drain(..=pos);
                    let Some(data) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let data = data.trim();
                    if data.is_empty() {
                        continue;
                    }
                    if data.starts_with("event:") {
                        // Not part of the payload; the event type is inside `data`.
                        continue;
                    }
                    let parsed: Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(_) => continue, // skip non-JSON control lines
                    };
                    handle_anthropic_event(
                        &parsed,
                        &tx,
                        &mut text_buf,
                        &mut tool_buf,
                        &mut saw_tool_call,
                        &mut last_usage,
                        &mut stop_reason,
                    )
                    .await;
                }
            }
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        // Anthropic doesn't offer a public /models listing analogous to
        // OpenAI; report the handful of current families statically.
        Ok(SUPPORTED_ANTHROPIC_MODELS
            .iter()
            .map(|name| ModelInfo {
                id: (*name).to_string(),
                name: (*name).to_string(),
                context_window: Some(200_000),
            })
            .collect())
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        Err(ProviderError::EmbeddingsUnsupported)
    }

    async fn count_tokens(&self, request: TokenCountRequest) -> Result<TokenCountResponse> {
        Ok(TokenCountResponse {
            tokens: estimate_messages(&request.messages, &request.tools),
            approximate: true,
        })
    }

    fn supports_prompt_cache(&self) -> bool {
        // Anthropic has explicit prompt caching; we keep it conservative and
        // report true so the agent may rely on caching for the system block.
        true
    }
}

const SUPPORTED_ANTHROPIC_MODELS: &[&str] = &[
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-7-sonnet-latest",
    "claude-sonnet-4",
    "claude-sonnet-4-20250514",
    "claude-opus-4",
    "claude-opus-4-20250514",
    "claude-opus-4-1",
];

fn map_anthropic_reason(reason: Option<&str>, had_tool_calls: bool) -> FinishReason {
    match reason {
        Some("tool_use") => FinishReason::ToolCalls,
        Some("max_tokens") => FinishReason::Length,
        _ if had_tool_calls => FinishReason::ToolCalls,
        _ => FinishReason::Stop,
    }
}

/// Apply one Anthropic SSE event (the `data:` payload) to the stream state.
async fn handle_anthropic_event(
    parsed: &Value,
    tx: &tokio::sync::mpsc::Sender<Result<StreamEvent>>,
    text_buf: &mut String,
    tool_buf: &mut HashMap<u64, (String, String, String)>,
    saw_tool_call: &mut bool,
    last_usage: &mut TokenUsage,
    stop_reason: &mut Option<String>,
) {
    let event_type = parsed["type"].as_str().unwrap_or("");
    match event_type {
        "content_block_start" => {
            let idx = parsed["index"].as_u64().unwrap_or(0);
            let block = &parsed["content_block"];
            if block["type"] == "tool_use" {
                *saw_tool_call = true;
                let id = block["id"].as_str().unwrap_or("").to_string();
                let name = block["name"].as_str().unwrap_or("").to_string();
                tool_buf.insert(idx, (id, name, String::new()));
            }
        }
        "content_block_delta" => {
            let idx = parsed["index"].as_u64().unwrap_or(0);
            let delta = &parsed["delta"];
            match delta["type"].as_str() {
                Some("text_delta") => {
                    let text = delta["text"].as_str().unwrap_or("");
                    text_buf.push_str(text);
                    let _ = tx
                        .send(Ok(StreamEvent::TextDelta {
                            text: text.to_string(),
                        }))
                        .await;
                }
                Some("input_json_delta") => {
                    let partial = delta["partial_json"].as_str().unwrap_or("");
                    if let Some((_, _, args)) = tool_buf.get_mut(&idx) {
                        args.push_str(partial);
                    } else {
                        // A tool delta arrived without a matching start block;
                        // synthesize one so downstream can assemble arguments.
                        tool_buf
                            .entry(idx)
                            .or_insert_with(|| {
                                *saw_tool_call = true;
                                ("tool_unknown".to_string(), String::new(), String::new())
                            })
                            .2
                            .push_str(partial);
                    }
                }
                _ => {}
            }
        }
        "content_block_stop" => {
            let idx = parsed["index"].as_u64().unwrap_or(0);
            if let Some((id, name, args)) = tool_buf.remove(&idx) {
                let _ = tx
                    .send(Ok(StreamEvent::ToolCallDelta {
                        id,
                        name: Some(name),
                        arguments_delta: args,
                        extra_content: None,
                    }))
                    .await;
            }
        }
        "message_delta" => {
            if let Some(usage) = parsed["usage"].as_object() {
                if let Some(out) = usage["output_tokens"].as_u64() {
                    last_usage.completion_tokens = out as u32;
                    last_usage.total_tokens = last_usage.prompt_tokens + out as u32;
                }
            }
            if let Some(reason) = parsed["delta"]["stop_reason"].as_str() {
                *stop_reason = Some(reason.to_string());
            }
        }
        "message_start" => {
            if let Some(usage) = parsed["message"]["usage"].as_object() {
                if let Some(input) = usage["input_tokens"].as_u64() {
                    last_usage.prompt_tokens = input as u32;
                    last_usage.total_tokens = input as u32;
                }
            }
        }
        _ => {}
    }
}
