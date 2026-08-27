//! AWS Bedrock provider — uses the Bedrock Converse API with
//! AWS Signature Version 4 (SigV4) request signing.
//!
//! Bedrock requires:
//! - AWS access key + secret key (from env or IAM role)
//! - SigV4 signing on every request
//! - Region-specific endpoint: `bedrock-runtime.{region}.amazonaws.com`
//! - Model ARN format: `anthropic.claude-3-5-sonnet-20241022-v2:0`

use crate::error::{ProviderError, Result};
use crate::heuristics::estimate_messages;
use crate::types::*;
use crate::{ChatStream, ModelProvider};
use async_trait::async_trait;
use futures::StreamExt;
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

pub struct BedrockProvider {
    id: String,
    region: String,
    client: Client,
    access_key: Option<String>,
    secret_key: Option<String>,
    session_token: Option<String>,
    headers: HashMap<String, String>,
}

impl BedrockProvider {
    pub fn new(id: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            region: region.into(),
            client: Client::new(),
            access_key: None,
            secret_key: None,
            session_token: None,
            headers: HashMap::new(),
        }
    }

    pub fn with_access_key(mut self, key: impl Into<String>) -> Self {
        self.access_key = Some(key.into());
        self
    }

    pub fn with_secret_key(mut self, key: impl Into<String>) -> Self {
        self.secret_key = Some(key.into());
        self
    }

    pub fn with_session_token(mut self, token: impl Into<String>) -> Self {
        self.session_token = Some(token.into());
        self
    }

    pub fn with_headers<I: IntoIterator<Item = (String, String)>>(mut self, headers: I) -> Self {
        self.headers.extend(headers);
        self
    }

    fn endpoint(&self) -> String {
        format!("https://bedrock-runtime.{}.amazonaws.com", self.region)
    }

    fn model_id(&self, model: &str) -> String {
        if model.contains('.') || model.starts_with("arn:") {
            model.to_string()
        } else {
            format!("anthropic.{model}")
        }
    }

    fn converse_url(&self, model: &str) -> String {
        let model_id = self.model_id(model);
        format!("{}/model/{}/invoke", self.endpoint(), model_id)
    }

    fn converse_stream_url(&self, model: &str) -> String {
        let model_id = self.model_id(model);
        format!(
            "{}/model/{}/invoke-with-response-stream",
            self.endpoint(),
            model_id
        )
    }

    /// Sign a request using AWS Signature Version 4.
    fn sign_request(
        &self,
        method: &str,
        url: &str,
        headers: &mut HashMap<String, String>,
        body: &[u8],
    ) -> Result<()> {
        let access_key = self
            .access_key
            .as_deref()
            .ok_or_else(|| ProviderError::MissingApiKey("AWS_ACCESS_KEY_ID".to_string()))?;
        let secret_key = self
            .secret_key
            .as_deref()
            .ok_or_else(|| ProviderError::MissingApiKey("AWS_SECRET_ACCESS_KEY".to_string()))?;

        let now =
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| ProviderError::Http {
                    status: 0,
                    message: e.to_string(),
                })?;
        let timestamp = now.as_secs();
        let datestamp = format!("{}", timestamp / 86400);
        let amz_date = format!("{}T000000Z", datestamp);

        let parsed = url::Url::parse(url).map_err(|e| ProviderError::Http {
            status: 0,
            message: e.to_string(),
        })?;
        let host = parsed.host_str().unwrap_or("").to_string();
        let path = parsed.path();

        let payload_hash = {
            let mut hasher = Sha256::new();
            hasher.update(body);
            format!("{:x}", hasher.finalize())
        };

        headers.insert("host".to_string(), host.clone());
        headers.insert("x-amz-date".to_string(), amz_date.clone());
        headers.insert("x-amz-content-sha256".to_string(), payload_hash.clone());
        if let Some(token) = &self.session_token {
            headers.insert("x-amz-security-token".to_string(), token.clone());
        }

        let mut sorted_headers: Vec<_> = headers.iter().collect();
        sorted_headers.sort_by_key(|(k, _)| k.to_lowercase());

        let canonical_headers: String = sorted_headers
            .iter()
            .map(|(k, v)| format!("{}:{}\n", k.to_lowercase(), v.trim()))
            .collect();

        let signed_headers: String = sorted_headers
            .iter()
            .map(|(k, _)| k.to_lowercase())
            .collect::<Vec<_>>()
            .join(";");

        let canonical_querystring = "";

        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            method, path, canonical_querystring, canonical_headers, signed_headers, payload_hash
        );

        let algorithm = "AWS4-HMAC-SHA256";
        let credential_scope = format!("{}/{}/aws4_request", datestamp, self.region);
        let canonical_request_hash = {
            let mut hasher = Sha256::new();
            hasher.update(canonical_request.as_bytes());
            format!("{:x}", hasher.finalize())
        };
        let string_to_sign = format!(
            "{}\n{}\n{}\n{}",
            algorithm, amz_date, credential_scope, canonical_request_hash
        );

        let k_date = {
            let mut mac = HmacSha256::new_from_slice(format!("AWS4{}", secret_key).as_bytes())
                .map_err(|e| ProviderError::Http {
                    status: 0,
                    message: e.to_string(),
                })?;
            mac.update(datestamp.as_bytes());
            mac.finalize().into_bytes()
        };
        let k_region = {
            let mut mac = HmacSha256::new_from_slice(&k_date).map_err(|e| ProviderError::Http {
                status: 0,
                message: e.to_string(),
            })?;
            mac.update(self.region.as_bytes());
            mac.finalize().into_bytes()
        };
        let k_service = {
            let mut mac =
                HmacSha256::new_from_slice(&k_region).map_err(|e| ProviderError::Http {
                    status: 0,
                    message: e.to_string(),
                })?;
            mac.update(b"bedrock");
            mac.finalize().into_bytes()
        };
        let k_signing = {
            let mut mac =
                HmacSha256::new_from_slice(&k_service).map_err(|e| ProviderError::Http {
                    status: 0,
                    message: e.to_string(),
                })?;
            mac.update(b"aws4_request");
            mac.finalize().into_bytes()
        };

        let signature = {
            let mut mac =
                HmacSha256::new_from_slice(&k_signing).map_err(|e| ProviderError::Http {
                    status: 0,
                    message: e.to_string(),
                })?;
            mac.update(string_to_sign.as_bytes());
            format!("{:x}", mac.finalize().into_bytes())
        };

        let authorization = format!(
            "{} Credential={}/{}, SignedHeaders={}, Signature={}",
            algorithm, access_key, credential_scope, signed_headers, signature
        );
        headers.insert("authorization".to_string(), authorization);

        Ok(())
    }

    fn reqwest_headers(&self, extra: &HashMap<String, String>) -> reqwest::header::HeaderMap {
        let mut map = reqwest::header::HeaderMap::new();
        for (k, v) in extra {
            if let Ok(name) = reqwest::header::HeaderName::from_bytes(k.as_bytes()) {
                if let Ok(value) = v.parse() {
                    map.insert(name, value);
                }
            }
        }
        map
    }
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "user",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "user",
    }
}

fn to_bedrock_message(m: &Message) -> Value {
    let mut content_parts: Vec<Value> = Vec::new();

    if !m.content.is_empty() {
        content_parts.push(json!({ "text": m.content }));
    }

    for img in &m.images {
        content_parts.push(json!({
            "image": {
                "format": img.mime_type.split('/').next_back().unwrap_or("jpeg"),
                "source": {
                    "bytes": img.data_base64
                }
            }
        }));
    }

    json!({
        "role": role_str(m.role),
        "content": content_parts
    })
}

#[async_trait]
impl ModelProvider for BedrockProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let system_msgs: Vec<&Message> = request
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .collect();
        let conv_msgs: Vec<&Message> = request
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .collect();

        let mut body = json!({
            "messages": conv_msgs.iter().map(|m| to_bedrock_message(m)).collect::<Vec<_>>(),
        });

        if !system_msgs.is_empty() {
            let system_text: String = system_msgs
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            body["system"] = json!(system_text);
        }

        let mut inference_config = json!({});
        if let Some(t) = request.temperature {
            inference_config["temperature"] = json!(t);
        }
        if let Some(max) = request.max_tokens {
            inference_config["maxTokens"] = json!(max);
        }
        if inference_config.as_object().is_some_and(|o| !o.is_empty()) {
            body["inferenceConfig"] = inference_config;
        }

        if !request.tools.is_empty() {
            let tool_defs: Vec<Value> = request
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "toolSpec": {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": {
                                "json": t.parameters
                            }
                        }
                    })
                })
                .collect();
            body["tools"] = json!(tool_defs);
        }

        let body_bytes = serde_json::to_vec(&body).map_err(|e| ProviderError::Http {
            status: 0,
            message: e.to_string(),
        })?;
        let url = self.converse_url(&request.model);

        let mut headers = HashMap::new();
        headers.insert("content-type".to_string(), "application/json".to_string());
        self.sign_request("POST", &url, &mut headers, &body_bytes)?;

        let resp = self
            .client
            .post(&url)
            .headers(self.reqwest_headers(&headers))
            .body(body_bytes)
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

        let output = &v["output"];
        let message = &output["message"];
        let content_parts = message["content"]
            .as_array()
            .ok_or_else(|| ProviderError::Http {
                status: 200,
                message: "no content in response".to_string(),
            })?;

        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        for part in content_parts {
            if let Some(text) = part["text"].as_str() {
                text_content.push_str(text);
            }
            if let Some(tool_use) = part.get("toolUse") {
                let name = tool_use["name"].as_str().unwrap_or("").to_string();
                let input = tool_use["input"]
                    .as_object()
                    .map(|o| serde_json::to_string(o).unwrap_or_default())
                    .unwrap_or_default();
                let tool_id = tool_use["toolUseId"].as_str().unwrap_or("").to_string();
                tool_calls.push(ToolCall {
                    id: tool_id,
                    name,
                    arguments: input,
                    extra_content: None,
                });
            }
        }

        let usage = TokenUsage {
            prompt_tokens: v["usage"]["inputTokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: v["usage"]["outputTokens"].as_u64().unwrap_or(0) as u32,
            total_tokens: v["usage"]["totalTokens"].as_u64().unwrap_or(0) as u32,
        };

        let finish_reason = match output["stopReason"].as_str() {
            Some("end_turn") => FinishReason::Stop,
            Some("max_tokens") => FinishReason::Length,
            Some("tool_use") => FinishReason::ToolCalls,
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
        let system_msgs: Vec<&Message> = request
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .collect();
        let conv_msgs: Vec<&Message> = request
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .collect();

        let mut body = json!({
            "messages": conv_msgs.iter().map(|m| to_bedrock_message(m)).collect::<Vec<_>>(),
        });

        if !system_msgs.is_empty() {
            let system_text: String = system_msgs
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            body["system"] = json!(system_text);
        }

        let mut inference_config = json!({});
        if let Some(t) = request.temperature {
            inference_config["temperature"] = json!(t);
        }
        if let Some(max) = request.max_tokens {
            inference_config["maxTokens"] = json!(max);
        }
        if inference_config.as_object().is_some_and(|o| !o.is_empty()) {
            body["inferenceConfig"] = inference_config;
        }

        if !request.tools.is_empty() {
            let tool_defs: Vec<Value> = request
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "toolSpec": {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": {
                                "json": t.parameters
                            }
                        }
                    })
                })
                .collect();
            body["tools"] = json!(tool_defs);
        }

        let body_bytes = serde_json::to_vec(&body).map_err(|e| ProviderError::Http {
            status: 0,
            message: e.to_string(),
        })?;
        let url = self.converse_stream_url(&request.model);

        let mut headers = HashMap::new();
        headers.insert("content-type".to_string(), "application/json".to_string());
        self.sign_request("POST", &url, &mut headers, &body_bytes)?;

        let resp = self
            .client
            .post(&url)
            .headers(self.reqwest_headers(&headers))
            .body(body_bytes)
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
                if line.is_empty() || !line.starts_with("bytes:") {
                    continue;
                }
                let data = &line[6..];
                if let Ok(v) = serde_json::from_str::<Value>(data) {
                    match v["header"]["event"].as_str() {
                        Some("contentBlockDelta") => {
                            if let Some(text_delta) = v["chunk"]["bytes"]
                                .as_str()
                                .and_then(|b| serde_json::from_str::<Value>(b).ok())
                                .and_then(|v| v["delta"]["text"].as_str().map(|s| s.to_string()))
                            {
                                if !text_delta.is_empty() {
                                    events.push(Ok(StreamEvent::TextDelta { text: text_delta }));
                                }
                            }
                        }
                        Some("messageStop") => {
                            let stop_reason = v["chunk"]["bytes"]
                                .as_str()
                                .and_then(|b| serde_json::from_str::<Value>(b).ok())
                                .and_then(|v| v["stopReason"].as_str().map(|s| s.to_string()));

                            let finish_reason = match stop_reason.as_deref() {
                                Some("end_turn") => FinishReason::Stop,
                                Some("max_tokens") => FinishReason::Length,
                                Some("tool_use") => FinishReason::ToolCalls,
                                _ => FinishReason::Stop,
                            };
                            events.push(Ok(StreamEvent::Done {
                                finish_reason,
                                usage: TokenUsage::default(),
                            }));
                        }
                        _ => {}
                    }
                }
            }

            futures::stream::iter(events).next().await
        });

        Ok(Box::pin(stream))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![
            ModelInfo {
                id: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
                name: "Claude 3.5 Sonnet v2".to_string(),
                context_window: Some(200_000),
            },
            ModelInfo {
                id: "anthropic.claude-3-5-haiku-20241022-v2:0".to_string(),
                name: "Claude 3.5 Haiku v2".to_string(),
                context_window: Some(200_000),
            },
            ModelInfo {
                id: "anthropic.claude-3-opus-20240229-v1:0".to_string(),
                name: "Claude 3 Opus".to_string(),
                context_window: Some(200_000),
            },
            ModelInfo {
                id: "anthropic.claude-3-sonnet-20240229-v1:0".to_string(),
                name: "Claude 3 Sonnet".to_string(),
                context_window: Some(200_000),
            },
            ModelInfo {
                id: "anthropic.claude-3-haiku-20240307-v1:0".to_string(),
                name: "Claude 3 Haiku".to_string(),
                context_window: Some(200_000),
            },
            ModelInfo {
                id: "meta.llama3-70b-instruct-v1:0".to_string(),
                name: "Llama 3 70B".to_string(),
                context_window: Some(8_192),
            },
            ModelInfo {
                id: "meta.llama3-8b-instruct-v1:0".to_string(),
                name: "Llama 3 8B".to_string(),
                context_window: Some(8_192),
            },
            ModelInfo {
                id: "amazon.titan-text-express-v1".to_string(),
                name: "Titan Text Express".to_string(),
                context_window: Some(8_192),
            },
            ModelInfo {
                id: "ai21.j2-ultra-v1".to_string(),
                name: "Jurassic-2 Ultra".to_string(),
                context_window: Some(8_192),
            },
        ])
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        Err(ProviderError::Http {
            status: 501,
            message: "Bedrock embeddings not implemented".to_string(),
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
