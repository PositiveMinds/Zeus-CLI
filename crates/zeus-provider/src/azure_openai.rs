//! Azure OpenAI provider — uses the Azure-specific API version parameter
//! and deployment-based URL structure instead of the standard OpenAI endpoint.
//!
//! Azure OpenAI requires:
//! - `api-version` query parameter on all requests
//! - Deployment-based URLs: `{base}/deployments/{deployment}/chat/completions`
//! - `api-key` header instead of `Authorization: Bearer`

use crate::error::{ProviderError, Result};
use crate::heuristics::estimate_messages;
use crate::types::*;
use crate::{ChatStream, ModelProvider};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct AzureOpenAiProvider {
    id: String,
    base_url: String,
    api_version: String,
    deployment: String,
    client: reqwest::Client,
    api_key: Option<String>,
    headers: HashMap<String, String>,
}

impl AzureOpenAiProvider {
    pub fn new(id: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_version: "2024-02-15-preview".to_string(),
            deployment: "gpt-4o".to_string(),
            client: reqwest::Client::new(),
            api_key: None,
            headers: HashMap::new(),
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    pub fn with_deployment(mut self, deployment: impl Into<String>) -> Self {
        self.deployment = deployment.into();
        self
    }

    pub fn with_headers<I: IntoIterator<Item = (String, String)>>(mut self, headers: I) -> Self {
        self.headers.extend(headers);
        self
    }

    fn headers(&self) -> HashMap<String, String> {
        let mut headers = self.headers.clone();
        if let Some(key) = &self.api_key {
            headers.insert("api-key".into(), key.clone());
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

    fn chat_url(&self) -> String {
        format!(
            "{}/deployments/{}/chat/completions?api-version={}",
            self.base_url, self.deployment, self.api_version
        )
    }

    fn models_url(&self) -> String {
        format!("{}/models?api-version={}", self.base_url, self.api_version)
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

fn to_azure_message(m: &Message) -> Value {
    let mut obj = json!({ "role": role_str(m.role) });
    if m.images.is_empty() {
        obj["content"] = json!(m.content);
    } else {
        let mut parts: Vec<Value> = vec![json!({ "type": "text", "text": m.content })];
        for img in &m.images {
            parts.push(json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", img.mime_type, img.data_base64)
                }
            }));
        }
        obj["content"] = json!(parts);
    }
    if !m.tool_calls.is_empty() {
        let calls: Vec<Value> = m
            .tool_calls
            .iter()
            .map(|tc| {
                json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments
                    }
                })
            })
            .collect();
        obj["tool_calls"] = json!(calls);
    }
    if let Some(tid) = &m.tool_call_id {
        obj["tool_call_id"] = json!(tid);
    }
    obj
}

#[async_trait]
impl ModelProvider for AzureOpenAiProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let mut body = json!({
            "model": request.model,
            "messages": request.messages.iter().map(to_azure_message).collect::<Vec<_>>(),
        });
        if let Some(t) = request.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(max) = request.max_tokens {
            body["max_tokens"] = json!(max);
        }
        if !request.tools.is_empty() {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            body["tools"] = json!(tools);
        }

        let url = self.chat_url();
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

        let choice = v["choices"]
            .as_array()
            .and_then(|c| c.first())
            .ok_or_else(|| ProviderError::Http {
                status: 200,
                message: "no choices in response".to_string(),
            })?;

        let message = &choice["message"];
        let content = message["content"].as_str().unwrap_or("").to_string();
        let tool_calls = message["tool_calls"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|tc| {
                        let id = tc["id"].as_str()?.to_string();
                        let name = tc["function"]["name"].as_str()?.to_string();
                        let arguments = tc["function"]["arguments"]
                            .as_str()
                            .unwrap_or("{}")
                            .to_string();
                        Some(ToolCall {
                            id,
                            name,
                            arguments,
                            extra_content: None,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let usage = TokenUsage {
            prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
            total_tokens: v["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
        };

        let finish_reason = match choice["finish_reason"].as_str() {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::Length,
            Some("tool_calls") => FinishReason::ToolCalls,
            _ => FinishReason::Stop,
        };

        Ok(ChatResponse {
            message: Message {
                role: Role::Assistant,
                content,
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
        let mut body = json!({
            "model": request.model,
            "messages": request.messages.iter().map(to_azure_message).collect::<Vec<_>>(),
            "stream": true,
        });
        if let Some(t) = request.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(max) = request.max_tokens {
            body["max_tokens"] = json!(max);
        }
        if !request.tools.is_empty() {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            body["tools"] = json!(tools);
        }

        let url = self.chat_url();
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
                if data == "[DONE]" {
                    events.push(Ok(StreamEvent::Done {
                        finish_reason: FinishReason::Stop,
                        usage: TokenUsage::default(),
                    }));
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<Value>(data) {
                    if let Some(delta) = v["choices"][0]["delta"].as_object() {
                        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                            if !content.is_empty() {
                                events.push(Ok(StreamEvent::TextDelta {
                                    text: content.to_string(),
                                }));
                            }
                        }
                        if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array())
                        {
                            for tc in tool_calls {
                                let name = tc["function"]["name"].as_str().map(|s| s.to_string());
                                let args =
                                    tc["function"]["arguments"].as_str().map(|s| s.to_string());
                                events.push(Ok(StreamEvent::ToolCallDelta {
                                    id: tc["id"].as_str().unwrap_or("").to_string(),
                                    name,
                                    arguments_delta: args.unwrap_or_default(),
                                    extra_content: None,
                                }));
                            }
                        }
                    }
                    if let Some(usage) = v["usage"].as_object() {
                        let prompt = usage["prompt_tokens"].as_u64().unwrap_or(0) as u32;
                        let completion = usage["completion_tokens"].as_u64().unwrap_or(0) as u32;
                        if prompt > 0 || completion > 0 {
                            events.push(Ok(StreamEvent::Done {
                                finish_reason: FinishReason::Stop,
                                usage: TokenUsage {
                                    prompt_tokens: prompt,
                                    completion_tokens: completion,
                                    total_tokens: prompt + completion,
                                },
                            }));
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

        let models = v["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let id = m["id"].as_str()?.to_string();
                        let name = m["id"].as_str()?.to_string();
                        Some(ModelInfo {
                            id,
                            name,
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
            message: "Azure OpenAI embeddings not implemented".to_string(),
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
