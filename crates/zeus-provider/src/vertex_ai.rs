//! Google Cloud Vertex AI provider — uses the Vertex AI endpoint with
//! GCP service account authentication (OAuth2 tokens).
//!
//! Vertex AI requires:
//! - Project ID and location in the URL
//! - OAuth2 access token from service account credentials
//! - Different endpoint structure than the Gemini API

use crate::error::{ProviderError, Result};
use crate::heuristics::estimate_messages;
use crate::types::*;
use crate::{ChatStream, ModelProvider};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static CALL_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_call_id() -> String {
    format!("call_{}", CALL_COUNTER.fetch_add(1, Ordering::Relaxed))
}

pub struct VertexAiProvider {
    id: String,
    project_id: String,
    location: String,
    client: reqwest::Client,
    /// OAuth2 access token (obtained from service account)
    access_token: Option<String>,
    headers: HashMap<String, String>,
}

impl VertexAiProvider {
    pub fn new(
        id: impl Into<String>,
        project_id: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            project_id: project_id.into(),
            location: location.into(),
            client: reqwest::Client::new(),
            access_token: None,
            headers: HashMap::new(),
        }
    }

    pub fn with_access_token(mut self, token: impl Into<String>) -> Self {
        self.access_token = Some(token.into());
        self
    }

    pub fn with_headers<I: IntoIterator<Item = (String, String)>>(mut self, headers: I) -> Self {
        self.headers.extend(headers);
        self
    }

    fn headers(&self) -> HashMap<String, String> {
        let mut headers = self.headers.clone();
        if let Some(token) = &self.access_token {
            headers.insert("Authorization".into(), format!("Bearer {token}"));
        }
        headers
    }

    fn reqwest_headers(&self) -> reqwest::header::HeaderMap {
        let mut map = reqwest::header::HeaderMap::new();
        for (k, v) in self.headers() {
            if let Ok(name) = reqwest::header::HeaderName::from_bytes(k.as_bytes()) {
                if let Ok(value) = v.parse() {
                    map.insert(name, value);
                }
            }
        }
        map
    }

    fn base_url(&self) -> String {
        format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google",
            self.location, self.project_id, self.location
        )
    }

    fn chat_url(&self, model: &str) -> String {
        format!("{}/models/{}:generateContent", self.base_url(), model)
    }

    fn chat_stream_url(&self, model: &str) -> String {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url(),
            model
        )
    }

    fn models_url(&self) -> String {
        format!("{}/models", self.base_url())
    }
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "model",
        Role::Tool => "function",
    }
}

fn to_vertex_message(m: &Message) -> Value {
    let role = role_str(m.role);
    let mut parts: Vec<Value> = Vec::new();

    if !m.content.is_empty() {
        parts.push(json!({ "text": m.content }));
    }

    for img in &m.images {
        parts.push(json!({
            "inlineData": {
                "mimeType": img.mime_type,
                "data": img.data_base64
            }
        }));
    }

    if !m.tool_calls.is_empty() {
        let function_calls: Vec<Value> = m
            .tool_calls
            .iter()
            .map(|tc| {
                json!({
                    "functionCall": {
                        "name": tc.name,
                        "args": serde_json::from_str::<Value>(&tc.arguments).unwrap_or(json!({}))
                    }
                })
            })
            .collect();
        parts.extend(function_calls);
    }

    if let Some(tid) = &m.tool_call_id {
        parts.push(json!({
            "functionResponse": {
                "name": tid,
                "response": {
                    "content": m.content
                }
            }
        }));
    }

    json!({
        "role": role,
        "parts": parts
    })
}

#[async_trait]
impl ModelProvider for VertexAiProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let contents: Vec<Value> = request.messages.iter().map(to_vertex_message).collect();

        let mut body = json!({
            "contents": contents,
        });

        let mut generation_config = json!({});
        if let Some(t) = request.temperature {
            generation_config["temperature"] = json!(t);
        }
        if let Some(max) = request.max_tokens {
            generation_config["maxOutputTokens"] = json!(max);
        }
        if generation_config.as_object().is_some_and(|o| !o.is_empty()) {
            body["generationConfig"] = generation_config;
        }

        if !request.tools.is_empty() {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "functionDeclarations": [{
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }]
                    })
                })
                .collect();
            body["tools"] = json!(tools);
        }

        let url = self.chat_url(&request.model);
        let resp = self
            .client
            .post(&url)
            .headers(self.reqwest_headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Http {
                status: 0,
                message: e.to_string(),
            })?;

        let status = resp.status().as_u16();
        if status != 200 {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Http {
                status,
                message: text,
            });
        }

        let v: Value = resp.json().await.map_err(|e| ProviderError::Http {
            status: 200,
            message: e.to_string(),
        })?;

        let candidate = v["candidates"]
            .as_array()
            .and_then(|c| c.first())
            .ok_or_else(|| ProviderError::Http {
                status: 200,
                message: "no candidates in response".to_string(),
            })?;

        let content = &candidate["content"];
        let parts = content["parts"]
            .as_array()
            .ok_or_else(|| ProviderError::Http {
                status: 200,
                message: "no parts in response".to_string(),
            })?;

        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        for part in parts {
            if let Some(text) = part["text"].as_str() {
                text_content.push_str(text);
            }
            if let Some(function_call) = part.get("functionCall") {
                let name = function_call["name"].as_str().unwrap_or("").to_string();
                let args = function_call["args"]
                    .as_object()
                    .map(|o| serde_json::to_string(o).unwrap_or_default())
                    .unwrap_or_default();
                tool_calls.push(ToolCall {
                    id: next_call_id(),
                    name,
                    arguments: args,
                    extra_content: None,
                });
            }
        }

        let usage_metadata = &v["usageMetadata"];
        let usage = TokenUsage {
            prompt_tokens: usage_metadata["promptTokenCount"].as_u64().unwrap_or(0) as u32,
            completion_tokens: usage_metadata["candidatesTokenCount"].as_u64().unwrap_or(0) as u32,
            total_tokens: usage_metadata["totalTokenCount"].as_u64().unwrap_or(0) as u32,
        };

        let finish_reason = match candidate["finishReason"].as_str() {
            Some("STOP") => FinishReason::Stop,
            Some("MAX_TOKENS") => FinishReason::Length,
            Some("SAFETY") => FinishReason::Stop,
            _ => FinishReason::Stop,
        };

        Ok(ChatResponse {
            message: Message {
                role: Role::Assistant,
                content: text_content,
                tool_calls,
                tool_call_id: None,
                images: Vec::new(),
            },
            finish_reason,
            usage,
            model: request.model,
        })
    }

    async fn stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let contents: Vec<Value> = request.messages.iter().map(to_vertex_message).collect();

        let mut body = json!({
            "contents": contents,
        });

        let mut generation_config = json!({});
        if let Some(t) = request.temperature {
            generation_config["temperature"] = json!(t);
        }
        if let Some(max) = request.max_tokens {
            generation_config["maxOutputTokens"] = json!(max);
        }
        if generation_config.as_object().is_some_and(|o| !o.is_empty()) {
            body["generationConfig"] = generation_config;
        }

        if !request.tools.is_empty() {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "functionDeclarations": [{
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }]
                    })
                })
                .collect();
            body["tools"] = json!(tools);
        }

        let url = self.chat_stream_url(&request.model);
        let resp = self
            .client
            .post(&url)
            .headers(self.reqwest_headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Http {
                status: 0,
                message: e.to_string(),
            })?;

        let status = resp.status().as_u16();
        if status != 200 {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Http {
                status,
                message: text,
            });
        }

        let stream = resp.bytes_stream().filter_map(|chunk| async move {
            let bytes = chunk.ok()?;
            let text = String::from_utf8_lossy(&bytes);
            let mut events = Vec::new();

            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() || !line.starts_with("data: ") {
                    continue;
                }
                let data = &line[6..];
                if let Ok(v) = serde_json::from_str::<Value>(data) {
                    if let Some(candidates) = v["candidates"].as_array() {
                        if let Some(candidate) = candidates.first() {
                            if let Some(parts) = candidate["content"]["parts"].as_array() {
                                for part in parts {
                                    if let Some(text) = part["text"].as_str() {
                                        if !text.is_empty() {
                                            events.push(Ok(StreamEvent::TextDelta {
                                                text: text.to_string(),
                                            }));
                                        }
                                    }
                                    if let Some(function_call) = part.get("functionCall") {
                                        let name = function_call["name"]
                                            .as_str()
                                            .unwrap_or("")
                                            .to_string();
                                        let args = function_call["args"]
                                            .as_object()
                                            .map(|o| serde_json::to_string(o).unwrap_or_default())
                                            .unwrap_or_default();
                                        events.push(Ok(StreamEvent::ToolCallDelta {
                                            id: next_call_id(),
                                            name: Some(name),
                                            arguments_delta: args,
                                            extra_content: None,
                                        }));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            futures::stream::iter(events).next().await
        });

        Ok(Box::pin(stream))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = self.models_url();
        let resp = self
            .client
            .get(&url)
            .headers(self.reqwest_headers())
            .send()
            .await
            .map_err(|e| ProviderError::Http {
                status: 0,
                message: e.to_string(),
            })?;

        let status = resp.status().as_u16();
        if status != 200 {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Http {
                status,
                message: text,
            });
        }

        let v: Value = resp.json().await.map_err(|e| ProviderError::Http {
            status: 200,
            message: e.to_string(),
        })?;

        let models = v["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let id = m["name"].as_str()?.split('/').next_back()?.to_string();
                        let display_name = m["displayName"].as_str().unwrap_or(&id).to_string();
                        Some(ModelInfo {
                            id,
                            name: display_name,
                            context_window: None,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        Err(ProviderError::Http {
            status: 501,
            message: "Vertex AI embeddings not implemented".to_string(),
        })
    }

    async fn count_tokens(&self, request: TokenCountRequest) -> Result<TokenCountResponse> {
        let approx = estimate_messages(&request.messages, &request.tools);
        Ok(TokenCountResponse {
            tokens: approx,
            approximate: true,
        })
    }

    fn supports_prompt_cache(&self) -> bool {
        false
    }
}
