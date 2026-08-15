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
    Err(ProviderError::Http {
        status: status.as_u16(),
        message: format!("ollama {status}: {text}"),
    })
}

fn tool_calls_from(msg: &OllamaMessage, id_offset: usize) -> Vec<ToolCall> {
    msg.tool_calls
        .iter()
        .enumerate()
        .map(|(i, c)| ToolCall {
            id: format!("call-{}", id_offset + i),
            name: c.function.name.clone(),
            arguments: c.function.arguments.to_string(),
            extra_content: None,
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
                                    extra_content: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use std::process::{Child, Command};
    use std::time::Duration;

    /// A tiny real HTTP server (Python) implementing enough of Ollama's
    /// `/api/tags`, `/api/chat` (streaming + non-streaming NDJSON), `/api/embed`,
    /// and `/api/pull` dialect to exercise the real client end to end.
    fn server_script() -> &'static str {
        r#"
import sys, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT_FILE = sys.argv[1]

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
        if self.path == "/api/tags":
            self._send_json(200, {"models": [
                {"name": "test-model", "details": {"context_length": 8192}},
                {"name": "other-model"},
            ]})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/chat":
            body = self._read_json()
            model = body.get("model", "")
            if body.get("stream"):
                self._stream_chat(model)
            else:
                self._chat(model)
        elif self.path == "/api/embed":
            body = self._read_json()
            inputs = body.get("input", [])
            self._send_json(200, {"embeddings": [[0.1, 0.2, 0.3] for _ in inputs]})
        elif self.path == "/api/pull":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for line in ["pulling manifest", "verifying sha256:abc", "success"]:
                self.wfile.write(json.dumps({"status": line}).encode("utf-8") + b"\n")
            self.wfile.flush()
        else:
            self._send_json(404, {"error": "not found"})

    def _chat(self, model):
        if model == "missing-model":
            self._send_json(404, {"error": "model not found"})
        elif model == "tool-model":
            resp = {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "shout", "arguments": {"text": "hi"}}}],
                },
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
        else:
            resp = {
                "message": {"role": "assistant", "content": "hello from ollama"},
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
        self._send_json(200, resp)

    def _stream_chat(self, model):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        chunks = [
            {"message": {"role": "assistant", "content": "hello "}, "done": False},
            {"message": {"role": "assistant", "content": "world"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True,
             "prompt_eval_count": 10, "eval_count": 5},
        ]
        for c in chunks:
            self.wfile.write(json.dumps(c).encode("utf-8") + b"\n")
        self.wfile.flush()

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
with open(PORT_FILE, "w") as f:
    f.write(str(server.server_address[1]))
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
        _dir: tempfile::TempDir,
    }

    impl TestServer {
        /// Spawn the python test server on an OS-assigned ephemeral port (never a
        /// hardcoded one — a fixed port could already be held by a leaked server
        /// from an earlier crashed run, and the old readiness probe would then
        /// happily talk to a stale process). The child binds port 0 and writes the
        /// actual port to a file inside a fresh temp dir; we poll for that file,
        /// which can only ever be produced by *this* server. The child is killed
        /// even if the readiness wait fails, so no process leaks on panic.
        fn start() -> Self {
        let dir = tempfile::TempDir::new().expect("create test server temp dir");
        let script_path = dir.path().join("server.py");
        let port_file = dir.path().join("port");
        std::fs::write(&script_path, server_script()).unwrap();
        let mut child = Command::new(python_cmd())
            .arg(&script_path)
            .arg(&port_file)
            .spawn()
            .expect("failed to spawn test server");
        // If anything below panics, still kill the child before unwinding.
        let mut guard = KillOnDrop(&mut child, true);
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let port = loop {
            if let Ok(contents) = std::fs::read_to_string(&port_file) {
                if let Ok(port) = contents.trim().parse::<u16>() {
                    break port;
                }
            }
            if std::time::Instant::now() >= deadline {
                panic!("test server did not report its port in time");
            }
            std::thread::sleep(Duration::from_millis(25));
        };
        guard.disarm();
        drop(guard);
        Self {
            child,
            base_url: format!("http://127.0.0.1:{port}"),
            _dir: dir,
        }
    }
    }

    /// Kills the spawned child when it goes out of scope (used as a panic
    /// guard while `start` is still waiting on the port file).
    struct KillOnDrop<'a>(&'a mut Child, bool);

    impl KillOnDrop<'_> {
        fn disarm(&mut self) {
            self.1 = false;
        }
    }

    impl Drop for KillOnDrop<'_> {
        fn drop(&mut self) {
            if self.1 {
                let _ = self.0.kill();
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
        let server = TestServer::start();
        let provider = OllamaProvider::new("test", &server.base_url);
        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "test-model");
        assert_eq!(models[0].context_window, Some(8192));
    }

    #[tokio::test]
    async fn chat_returns_plain_text() {
        let server = TestServer::start();
        let provider = OllamaProvider::new("test", &server.base_url);
        let resp = provider
            .chat(ChatRequest::new("text-model", vec![Message::user("hi")]))
            .await
            .unwrap();
        assert_eq!(resp.message.content, "hello from ollama");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 5);
    }

    #[tokio::test]
    async fn chat_returns_tool_call() {
        let server = TestServer::start();
        let provider = OllamaProvider::new("test", &server.base_url);
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
        let server = TestServer::start();
        let provider = OllamaProvider::new("test", &server.base_url);
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
    async fn embeddings_work() {
        let server = TestServer::start();
        let provider = OllamaProvider::new("test", &server.base_url);
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

    #[tokio::test]
    async fn pull_streams_progress_lines() {
        let server = TestServer::start();
        let provider = OllamaProvider::new("test", &server.base_url);
        let mut seen = Vec::new();
        provider
            .pull("my-model", |s| seen.push(s.to_string()))
            .await
            .unwrap();
        assert_eq!(
            seen,
            vec!["pulling manifest", "verifying sha256:abc", "success"]
        );
    }

    #[tokio::test]
    async fn error_for_status_exposes_http_status() {
        let server = TestServer::start();
        let provider = OllamaProvider::new("test", &server.base_url);
        // POST to an unknown path → 404, surfaced as ProviderError::Http.
        let resp = provider
            .chat(ChatRequest::new("missing-model", vec![Message::user("hi")]))
            .await
            .unwrap_err();
        assert_eq!(resp.http_status(), Some(404));
    }
}
