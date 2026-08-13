//! Shared client for OpenAI-Chat-Completions-compatible API endpoints — every
//! provider that speaks the `/v1/chat/completions` + SSE dialect: LM Studio
//! and `llama-server` locally, and the OpenAI-compatible hosted routes of
//! openai, grok (x.ai), openrouter, opencode zen, and gemini. Only the
//! configured `base_url`, auth header, and extra headers differ.

use crate::error::{ProviderError, Result};
use crate::heuristics::estimate_messages;
use crate::types::*;
use crate::{ChatStream, ModelProvider};
use async_trait::async_trait;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct OpenAiCompatProvider {
    id: String,
    base_url: String,
    client: reqwest::Client,
    /// Optional `Authorization: Bearer <key>`.
    api_key: Option<String>,
    /// Extra headers (resolved from env-var references by the registry).
    headers: HashMap<String, String>,
}

impl OpenAiCompatProvider {
    pub fn new(id: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
            api_key: None,
            headers: HashMap::new(),
        }
    }

    /// Attach a bearer token (used as `Authorization: Bearer <key>`).
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Attach extra headers (e.g. `x-api-key`, `HTTP-Referer` for OpenRouter).
    pub fn with_headers<I: IntoIterator<Item = (String, String)>>(mut self, headers: I) -> Self {
        self.headers.extend(headers);
        self
    }

    fn headers(&self) -> HashMap<String, String> {
        let mut headers = self.headers.clone();
        if let Some(key) = &self.api_key {
            headers.insert("Authorization".into(), format!("Bearer {key}"));
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
        format!("{}/chat/completions", self.base_url)
    }
    fn models_url(&self) -> String {
        format!("{}/models", self.base_url)
    }
    fn embeddings_url(&self) -> String {
        format!("{}/embeddings", self.base_url)
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

fn to_openai_message(m: &Message) -> Value {
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
    if let Some(id) = &m.tool_call_id {
        obj["tool_call_id"] = json!(id);
    }
    if !m.tool_calls.is_empty() {
        obj["tool_calls"] = json!(m
            .tool_calls
            .iter()
            .map(|c| json!({
                "id": c.id,
                "type": "function",
                "function": { "name": c.name, "arguments": c.arguments },
            }))
            .collect::<Vec<_>>());
    }
    obj
}

fn to_openai_tools(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": { "name": t.name, "description": t.description, "parameters": t.parameters }
            })
        })
        .collect()
}

fn build_body(request: &ChatRequest, stream: bool) -> Value {
    let mut body = json!({
        "model": request.model,
        "messages": request.messages.iter().map(to_openai_message).collect::<Vec<_>>(),
        "stream": stream,
    });
    if !request.tools.is_empty() {
        body["tools"] = json!(to_openai_tools(&request.tools));
    }
    if let Some(t) = request.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(m) = request.max_tokens {
        body["max_tokens"] = json!(m);
    }
    body
}

#[derive(Debug, Deserialize)]
struct OaFunctionCall {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}
#[derive(Debug, Deserialize)]
struct OaToolCall {
    #[serde(default)]
    id: String,
    function: OaFunctionCall,
}
#[derive(Debug, Deserialize, Default)]
struct OaMessage {
    #[serde(default)]
    content: Option<String>,
    // `Option<Vec<_>>` rather than `Vec<_>` — some OpenAI-compatible
    // providers (observed from deepseek/opencodezen) send an explicit
    // `"tool_calls": null` instead of omitting the field or sending `[]`.
    // `#[serde(default)]` alone only covers a *missing* key; an explicit
    // `null` still fails a bare `Vec<_>` deserializer ("invalid type: null,
    // expected a sequence"), whereas `Option`'s deserializer treats `null`
    // as `None`.
    #[serde(default)]
    tool_calls: Option<Vec<OaToolCall>>,
}
#[derive(Debug, Deserialize)]
struct OaUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}
#[derive(Debug, Deserialize)]
struct OaChoice {
    #[serde(default)]
    message: Option<OaMessage>,
    #[serde(default)]
    finish_reason: Option<String>,
}
#[derive(Debug, Deserialize)]
struct OaChatResponse {
    #[serde(default)]
    choices: Vec<OaChoice>,
    #[serde(default)]
    usage: Option<OaUsage>,
}

fn map_finish_reason(reason: Option<&str>, had_tool_calls: bool) -> FinishReason {
    match reason {
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("length") => FinishReason::Length,
        _ if had_tool_calls => FinishReason::ToolCalls,
        _ => FinishReason::Stop,
    }
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
    Err(ProviderError::Api(format!("{status}: {text}")))
}

// --- Streaming delta shapes (a strict subset of the full chunk; unknown
// fields are ignored rather than rejected, for forward compatibility) ---

#[derive(Debug, Deserialize)]
struct OaToolCallDelta {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OaFunctionDelta>,
}
#[derive(Debug, Deserialize, Default)]
struct OaFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}
#[derive(Debug, Deserialize, Default)]
struct OaDelta {
    #[serde(default)]
    content: Option<String>,
    // See the comment on `OaMessage::tool_calls` — same explicit-`null` issue.
    #[serde(default)]
    tool_calls: Option<Vec<OaToolCallDelta>>,
}
#[derive(Debug, Deserialize)]
struct OaStreamChoice {
    #[serde(default)]
    delta: Option<OaDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}
#[derive(Debug, Deserialize)]
struct OaStreamChunk {
    #[serde(default)]
    choices: Vec<OaStreamChoice>,
    #[serde(default)]
    usage: Option<OaUsage>,
}

#[async_trait]
impl ModelProvider for OpenAiCompatProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        if request.is_cancelled() {
            return Err(ProviderError::Cancelled);
        }
        let model = request.model.clone();
        let body = build_body(&request, false);
        let resp = self
            .client
            .post(self.chat_url())
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_err)?;
        let resp = error_for_status(resp).await?;
        let parsed: OaChatResponse = resp.json().await.map_err(map_reqwest_err)?;
        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::Api("no choices in response".into()))?;
        let oa_msg = choice.message.unwrap_or_default();
        let tool_calls: Vec<ToolCall> = oa_msg
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|c| ToolCall {
                id: c.id,
                name: c.function.name,
                arguments: c.function.arguments,
            })
            .collect();
        let finish_reason =
            map_finish_reason(choice.finish_reason.as_deref(), !tool_calls.is_empty());
        let mut message = Message::assistant(oa_msg.content.unwrap_or_default());
        message.tool_calls = tool_calls;
        let usage = parsed
            .usage
            .map(|u| TokenUsage::new(u.prompt_tokens, u.completion_tokens))
            .unwrap_or_default();

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
        let resp = self
            .client
            .post(self.chat_url())
            .headers(self.reqwest_headers())
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_err)?;
        let resp = error_for_status(resp).await?;

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamEvent>>(32);

        tokio::spawn(async move {
            let mut byte_stream = resp.bytes_stream();
            let mut buffer = String::new();
            // Maps a tool-call's stream `index` to the id first assigned to
            // it, so later deltas (which may omit `id`) route to the same
            // logical call.
            let mut call_ids: HashMap<usize, String> = HashMap::new();
            let mut saw_tool_call = false;
            let mut last_usage = TokenUsage::default();
            let mut last_finish_reason: Option<String> = None;

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
                                finish_reason: map_finish_reason(
                                    last_finish_reason.as_deref(),
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
                    // SSE framing: only "data: ..." lines carry payload;
                    // everything else (blank lines, "event: ..." fields) is
                    // ignored. "[DONE]" is the SSE end-of-stream sentinel.
                    let Some(data) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let data = data.trim();
                    if data.is_empty() {
                        continue;
                    }
                    if data == "[DONE]" {
                        let _ = tx
                            .send(Ok(StreamEvent::Done {
                                finish_reason: map_finish_reason(
                                    last_finish_reason.as_deref(),
                                    saw_tool_call,
                                ),
                                usage: last_usage,
                            }))
                            .await;
                        return;
                    }

                    let parsed: std::result::Result<OaStreamChunk, _> = serde_json::from_str(data);
                    let parsed = match parsed {
                        Ok(c) => c,
                        Err(e) => {
                            let _ = tx
                                .send(Err(ProviderError::Api(format!("bad stream chunk: {e}"))))
                                .await;
                            return;
                        }
                    };
                    if let Some(u) = parsed.usage {
                        last_usage = TokenUsage::new(u.prompt_tokens, u.completion_tokens);
                    }
                    for choice in parsed.choices {
                        if choice.finish_reason.is_some() {
                            last_finish_reason = choice.finish_reason.clone();
                        }
                        let Some(delta) = choice.delta else { continue };
                        if let Some(text) = delta.content {
                            if !text.is_empty()
                                && tx.send(Ok(StreamEvent::TextDelta { text })).await.is_err()
                            {
                                return;
                            }
                        }
                        for tc in delta.tool_calls.unwrap_or_default() {
                            saw_tool_call = true;
                            let id = tc
                                .id
                                .or_else(|| call_ids.get(&tc.index).cloned())
                                .unwrap_or_else(|| format!("call-{}", tc.index));
                            call_ids.entry(tc.index).or_insert_with(|| id.clone());
                            let name = tc.function.as_ref().and_then(|f| f.name.clone());
                            let arguments_delta =
                                tc.function.and_then(|f| f.arguments).unwrap_or_default();
                            if tx
                                .send(Ok(StreamEvent::ToolCallDelta {
                                    id,
                                    name,
                                    arguments_delta,
                                }))
                                .await
                                .is_err()
                            {
                                return;
                            }
                        }
                    }
                }
            }
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        #[derive(Deserialize)]
        struct OaModel {
            id: String,
        }
        #[derive(Deserialize)]
        struct OaModelsResponse {
            #[serde(default)]
            data: Vec<OaModel>,
        }
        let resp = self
            .client
            .get(self.models_url())
            .headers(self.reqwest_headers())
            .send()
            .await
            .map_err(map_reqwest_err)?;
        let resp = error_for_status(resp).await?;
        let parsed: OaModelsResponse = resp.json().await.map_err(map_reqwest_err)?;
        Ok(parsed
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id.clone(),
                name: m.id,
                // The OpenAI-compatible `/v1/models` shape doesn't carry a
                // context-window field; neither LM Studio nor llama-server
                // include one here (unlike Ollama's /api/tags).
                context_window: None,
            })
            .collect())
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        #[derive(Deserialize)]
        struct OaEmbedding {
            embedding: Vec<f32>,
        }
        #[derive(Deserialize)]
        struct OaEmbeddingsResponse {
            #[serde(default)]
            data: Vec<OaEmbedding>,
            #[serde(default)]
            usage: Option<OaUsage>,
        }
        let body = json!({ "model": request.model, "input": request.input });
        let resp = self
            .client
            .post(self.embeddings_url())
            .headers(self.reqwest_headers())
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_err)?;
        let resp = error_for_status(resp).await?;
        let parsed: OaEmbeddingsResponse = resp.json().await.map_err(map_reqwest_err)?;
        let usage = parsed
            .usage
            .map(|u| TokenUsage::new(u.prompt_tokens, 0))
            .unwrap_or_else(|| TokenUsage::new(estimate_messages(&[], &[]), 0));
        Ok(EmbeddingResponse {
            vectors: parsed.data.into_iter().map(|d| d.embedding).collect(),
            usage,
        })
    }

    async fn count_tokens(&self, request: TokenCountRequest) -> Result<TokenCountResponse> {
        // Neither LM Studio nor llama-server expose a free-standing tokenize
        // endpoint we can rely on being present across versions; fall back
        // to the same approximate heuristic as the Ollama provider
        // honestly marked `approximate: true`.
        Ok(TokenCountResponse {
            tokens: estimate_messages(&request.messages, &request.tools),
            approximate: true,
        })
    }

    fn supports_prompt_cache(&self) -> bool {
        // Neither exposes an explicit cache-control API, but both LM Studio
        // and llama-server automatically reuse the KV cache for a repeated
        // prompt prefix (llama.cpp's server-side prompt caching), so a
        // stable system prompt/tool block genuinely is cheaper on
        // subsequent calls — just implicitly, not via a declared parameter.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Child, Command};
    use std::time::Duration;

    /// A tiny real HTTP server (Python) implementing enough of the OpenAI
    /// `/v1/models`, `/v1/chat/completions` (streaming + non-streaming), and
    /// `/v1/embeddings` dialect to exercise the real client/transport code
    /// end to end — this is what both LM Studio and llama-server speak, so
    /// passing against this proves the wire format is right, not just that
    /// the JSON shapes compile.
    fn server_script() -> &'static str {
        r#"
import sys, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _read_json(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        return json.loads(body) if body else {}

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(200, {"data": [{"id": "test-model"}]})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            body = self._read_json()
            model = body.get("model", "")
            if body.get("stream"):
                self._stream_chat(model)
            else:
                self._chat(model)
        elif self.path == "/v1/embeddings":
            body = self._read_json()
            inputs = body.get("input", [])
            self._send_json(200, {
                "data": [{"embedding": [0.1, 0.2, 0.3], "index": i} for i in range(len(inputs))],
                "usage": {"prompt_tokens": 5, "completion_tokens": 0},
            })
        else:
            self._send_json(404, {"error": "not found"})

    def _chat(self, model):
        if model == "tool-model":
            resp = {"choices": [{
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{"id": "call_1", "type": "function",
                                            "function": {"name": "shout", "arguments": "{\"text\":\"hi\"}"}}]},
                "finish_reason": "tool_calls",
            }], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        elif model == "null-tool-calls-model":
            # Some OpenAI-compatible providers (observed from deepseek /
            # opencodezen) send an explicit `"tool_calls": null` instead of
            # omitting the field — regression coverage for that shape.
            resp = {"choices": [{
                "message": {"role": "assistant", "content": "no tools here", "tool_calls": None},
                "finish_reason": "stop",
            }], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        else:
            resp = {"choices": [{"message": {"role": "assistant", "content": "hello from test server"},
                                  "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        self._send_json(200, resp)

    def _stream_chat(self, model):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if model == "tool-model":
            chunks = [
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "shout", "arguments": ""}}]}, "finish_reason": None}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"text\""}}]}, "finish_reason": None}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ":\"hi\"}"}}]}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            ]
        elif model == "null-tool-calls-model":
            chunks = [
                {"choices": [{"delta": {"content": "hi", "tool_calls": None}, "finish_reason": None}]},
                {"choices": [{"delta": {"tool_calls": None}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            ]
        else:
            chunks = [
                {"choices": [{"delta": {"content": "hello "}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "world"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            ]
        for c in chunks:
            self.wfile.write(("data: " + json.dumps(c) + "\n\n").encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
server.serve_forever()
"#
    }

    fn python_cmd() -> &'static str {
        if cfg!(windows) {
            "python"
        } else {
            "python3"
        }
    }

    struct TestServer {
        child: Child,
        base_url: String,
    }

    impl TestServer {
        fn start(port: u16) -> Self {
            let tmp = std::env::temp_dir().join(format!("zeus_oa_test_server_{port}.py"));
            std::fs::write(&tmp, server_script()).unwrap();
            let child = Command::new(python_cmd())
                .arg(&tmp)
                .arg(port.to_string())
                .spawn()
                .expect("failed to spawn test server");
            // Poll for readiness rather than a fixed sleep.
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            loop {
                if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
                    break;
                }
                if std::time::Instant::now() >= deadline {
                    panic!("test server on port {port} did not become ready in time");
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Self {
                child,
                base_url: format!("http://127.0.0.1:{port}/v1"),
            }
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    #[tokio::test]
    async fn list_models_works() {
        let server = TestServer::start(18091);
        let provider = OpenAiCompatProvider::new("test", &server.base_url);
        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "test-model");
    }

    #[tokio::test]
    async fn chat_returns_plain_text() {
        let server = TestServer::start(18092);
        let provider = OpenAiCompatProvider::new("test", &server.base_url);
        let resp = provider
            .chat(ChatRequest::new("text-model", vec![Message::user("hi")]))
            .await
            .unwrap();
        assert_eq!(resp.message.content, "hello from test server");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        assert_eq!(resp.usage.prompt_tokens, 10);
    }

    #[tokio::test]
    async fn chat_returns_tool_call() {
        let server = TestServer::start(18093);
        let provider = OpenAiCompatProvider::new("test", &server.base_url);
        let resp = provider
            .chat(ChatRequest::new("tool-model", vec![Message::user("hi")]))
            .await
            .unwrap();
        assert_eq!(resp.finish_reason, FinishReason::ToolCalls);
        assert_eq!(resp.message.tool_calls.len(), 1);
        assert_eq!(resp.message.tool_calls[0].name, "shout");
        assert_eq!(resp.message.tool_calls[0].arguments, r#"{"text":"hi"}"#);
    }

    #[tokio::test]
    async fn stream_yields_text_deltas() {
        let server = TestServer::start(18094);
        let provider = OpenAiCompatProvider::new("test", &server.base_url);
        let mut stream = provider
            .stream(ChatRequest::new("text-model", vec![Message::user("hi")]))
            .await
            .unwrap();
        let mut text = String::new();
        let mut finish = None;
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::TextDelta { text: t } => text.push_str(&t),
                StreamEvent::Done { finish_reason, .. } => finish = Some(finish_reason),
                _ => {}
            }
        }
        assert_eq!(text, "hello world");
        assert_eq!(finish, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn stream_accumulates_tool_call_across_chunks() {
        let server = TestServer::start(18095);
        let provider = OpenAiCompatProvider::new("test", &server.base_url);
        let mut stream = provider
            .stream(ChatRequest::new("tool-model", vec![Message::user("hi")]))
            .await
            .unwrap();
        let mut name = None;
        let mut arguments = String::new();
        let mut finish = None;
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::ToolCallDelta {
                    name: n,
                    arguments_delta,
                    ..
                } => {
                    if n.is_some() {
                        name = n;
                    }
                    arguments.push_str(&arguments_delta);
                }
                StreamEvent::Done { finish_reason, .. } => finish = Some(finish_reason),
                _ => {}
            }
        }
        assert_eq!(name, Some("shout".to_string()));
        assert_eq!(arguments, r#"{"text":"hi"}"#);
        assert_eq!(finish, Some(FinishReason::ToolCalls));
    }

    /// Regression test for a provider (observed from deepseek/opencodezen)
    /// sending an explicit `"tool_calls": null` in the message body rather
    /// than omitting the field — used to fail deserialization with "invalid
    /// type: null, expected a sequence".
    #[tokio::test]
    async fn chat_tolerates_explicit_null_tool_calls() {
        let server = TestServer::start(18098);
        let provider = OpenAiCompatProvider::new("test", &server.base_url);
        let resp = provider
            .chat(ChatRequest::new(
                "null-tool-calls-model",
                vec![Message::user("hi")],
            ))
            .await
            .unwrap();
        assert_eq!(resp.message.content, "no tools here");
        assert!(resp.message.tool_calls.is_empty());
        assert_eq!(resp.finish_reason, FinishReason::Stop);
    }

    /// Same regression, but for a streamed delta's `"tool_calls": null`.
    #[tokio::test]
    async fn stream_tolerates_explicit_null_tool_calls() {
        let server = TestServer::start(18099);
        let provider = OpenAiCompatProvider::new("test", &server.base_url);
        let mut stream = provider
            .stream(ChatRequest::new(
                "null-tool-calls-model",
                vec![Message::user("hi")],
            ))
            .await
            .unwrap();
        let mut text = String::new();
        let mut finish = None;
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::TextDelta { text: t } => text.push_str(&t),
                StreamEvent::Done { finish_reason, .. } => finish = Some(finish_reason),
                _ => {}
            }
        }
        assert_eq!(text, "hi");
        assert_eq!(finish, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn embeddings_work() {
        let server = TestServer::start(18096);
        let provider = OpenAiCompatProvider::new("test", &server.base_url);
        let resp = provider
            .embeddings(EmbeddingRequest {
                model: "test-model".into(),
                input: vec!["a".into(), "b".into()],
            })
            .await
            .unwrap();
        assert_eq!(resp.vectors.len(), 2);
        assert_eq!(resp.vectors[0], vec![0.1, 0.2, 0.3]);
    }
}
