//! Deterministic mock provider for tests and offline development.

use crate::error::{ProviderError, Result};
use crate::heuristics::{estimate_messages, estimate_tokens};
use crate::types::*;
use crate::{ChatStream, ModelProvider};
use async_trait::async_trait;
use futures::stream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::debug;

/// In-memory provider that echoes / scripted replies.
pub struct MockProvider {
    id: String,
    /// Fixed reply body; if empty, echoes the last user message.
    reply: Mutex<String>,
    /// Artificial stream delay per chunk (ms).
    chunk_delay_ms: u64,
    call_counter: AtomicU64,
    /// When set, the *next* stream/chat call requests this tool call instead
    /// of a normal text reply, then is consumed (so the following call falls
    /// back to a normal reply — letting a scripted tool-call/result cycle
    /// terminate). Test/dev helper only.
    scripted_tool_call: Mutex<Option<(String, String, String)>>,
    /// When true, `scripted_tool_call` is never consumed — every call
    /// requests the same tool call again. Test helper for simulating a model
    /// that never converges to a final answer (e.g. to exercise the
    /// max-tool-iterations path).
    repeat_tool_call: bool,
}

impl MockProvider {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            reply: Mutex::new(String::new()),
            chunk_delay_ms: 0,
            call_counter: AtomicU64::new(0),
            scripted_tool_call: Mutex::new(None),
            repeat_tool_call: false,
        }
    }

    pub fn with_reply(self, reply: impl Into<String>) -> Self {
        // set via blocking lock in constructor path only
        *self.reply.try_lock().expect("mock reply lock") = reply.into();
        self
    }

    pub fn with_chunk_delay_ms(mut self, ms: u64) -> Self {
        self.chunk_delay_ms = ms;
        self
    }

    /// Script the next stream/chat call to request this single tool call.
    pub fn with_tool_call(
        self,
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        *self.scripted_tool_call.try_lock().expect("mock tool_call lock") =
            Some((id.into(), name.into(), arguments.into()));
        self
    }

    /// Like `with_tool_call`, but every subsequent call requests the same
    /// tool call again instead of the script being consumed after one use —
    /// simulates a model that never converges to a final answer.
    pub fn with_repeating_tool_call(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        *self.scripted_tool_call.try_lock().expect("mock tool_call lock") =
            Some((id.into(), name.into(), arguments.into()));
        self.repeat_tool_call = true;
        self
    }

    async fn build_reply_text(&self, request: &ChatRequest) -> String {
        let custom = self.reply.lock().await.clone();
        if !custom.is_empty() {
            return custom;
        }
        let last_user = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .unwrap_or("");
        format!("mock-echo: {last_user}")
    }
}

#[async_trait]
impl ModelProvider for MockProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        if request.is_cancelled() {
            return Err(ProviderError::Cancelled);
        }
        let n = self.call_counter.fetch_add(1, Ordering::SeqCst);
        debug!(call = n, model = %request.model, "mock chat");

        let prompt = estimate_messages(&request.messages, &request.tools);

        let scripted = if self.repeat_tool_call {
            self.scripted_tool_call.lock().await.clone()
        } else {
            self.scripted_tool_call.lock().await.take()
        };
        if let Some((id, name, arguments)) = scripted {
            let mut message = Message::assistant("");
            message.tool_calls = vec![ToolCall { id, name, arguments }];
            return Ok(ChatResponse {
                message,
                usage: TokenUsage::new(prompt, 0),
                finish_reason: FinishReason::ToolCalls,
                model: request.model,
            });
        }

        let text = self.build_reply_text(&request).await;
        let completion = estimate_tokens(&text);

        Ok(ChatResponse {
            message: Message::assistant(text),
            usage: TokenUsage::new(prompt, completion),
            finish_reason: FinishReason::Stop,
            model: request.model,
        })
    }

    async fn stream(&self, request: ChatRequest) -> Result<ChatStream> {
        if request.is_cancelled() {
            return Err(ProviderError::Cancelled);
        }

        let scripted = if self.repeat_tool_call {
            self.scripted_tool_call.lock().await.clone()
        } else {
            self.scripted_tool_call.lock().await.take()
        };
        if let Some((id, name, arguments)) = scripted {
            let prompt = estimate_messages(&request.messages, &request.tools);
            let s = stream::iter(vec![
                Ok(StreamEvent::ToolCallDelta {
                    id,
                    name: Some(name),
                    arguments_delta: arguments,
                }),
                Ok(StreamEvent::Done {
                    finish_reason: FinishReason::ToolCalls,
                    usage: TokenUsage::new(prompt, 0),
                }),
            ]);
            return Ok(Box::pin(s));
        }

        let text = self.build_reply_text(&request).await;
        let prompt = estimate_messages(&request.messages, &request.tools);
        let completion = estimate_tokens(&text);
        let delay = self.chunk_delay_ms;
        let cancel = request.cancel.clone();

        // Chunk by words for realistic streaming.
        let words: Vec<String> = text.split_whitespace().map(|w| w.to_string()).collect();
        let word_count = words.len().max(1);

        let s = stream::unfold(
            (words, 0usize, false),
            move |(words, idx, done)| {
                let cancel = cancel.clone();
                async move {
                    if done {
                        return None;
                    }
                    if cancel.as_ref().map(|rx| *rx.borrow()).unwrap_or(false) {
                        return Some((
                            Ok(StreamEvent::Done {
                                finish_reason: FinishReason::Cancelled,
                                usage: TokenUsage::new(prompt, 0),
                            }),
                            (words, idx, true),
                        ));
                    }
                    if delay > 0 {
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                    }
                    if idx < words.len() {
                        let piece = if idx + 1 < words.len() {
                            format!("{} ", words[idx])
                        } else {
                            words[idx].clone()
                        };
                        return Some((
                            Ok(StreamEvent::TextDelta { text: piece }),
                            (words, idx + 1, false),
                        ));
                    }
                    Some((
                        Ok(StreamEvent::Done {
                            finish_reason: FinishReason::Stop,
                            usage: TokenUsage::new(prompt, completion),
                        }),
                        (words, idx, true),
                    ))
                }
            },
        );

        // Ensure empty reply still yields Done.
        let _ = word_count;
        Ok(Box::pin(s))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![
            ModelInfo {
                id: "mock-model".into(),
                name: "Mock Model".into(),
                context_window: Some(128_000),
            },
            ModelInfo {
                id: "mock-small".into(),
                name: "Mock Small".into(),
                context_window: Some(8_192),
            },
        ])
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let dim = 8;
        let vectors = request
            .input
            .iter()
            .map(|text| {
                let mut v = vec![0.0f32; dim];
                for (i, b) in text.bytes().enumerate() {
                    v[i % dim] += (b as f32) / 255.0;
                }
                // L2 normalize lightly
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                for x in &mut v {
                    *x /= norm;
                }
                v
            })
            .collect();
        let tokens: u32 = request.input.iter().map(|t| estimate_tokens(t)).sum();
        Ok(EmbeddingResponse {
            vectors,
            usage: TokenUsage::new(tokens, 0),
        })
    }

    async fn count_tokens(&self, request: TokenCountRequest) -> Result<TokenCountResponse> {
        Ok(TokenCountResponse {
            tokens: estimate_messages(&request.messages, &request.tools),
            approximate: true,
        })
    }

    fn supports_prompt_cache(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use tokio::sync::watch;

    #[tokio::test]
    async fn chat_echoes_user() {
        let p = MockProvider::new("mock");
        let resp = p
            .chat(ChatRequest::new(
                "mock-model",
                vec![Message::user("hello world")],
            ))
            .await
            .unwrap();
        assert!(resp.message.content.contains("hello world"));
        assert!(resp.usage.total_tokens > 0);
    }

    #[tokio::test]
    async fn stream_yields_deltas_then_done() {
        let p = MockProvider::new("mock").with_reply("one two three");
        let mut stream = p
            .stream(ChatRequest::new("mock-model", vec![Message::user("x")]))
            .await
            .unwrap();
        let mut text = String::new();
        let mut done = false;
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::TextDelta { text: t } => text.push_str(&t),
                StreamEvent::Done { finish_reason, .. } => {
                    assert_eq!(finish_reason, FinishReason::Stop);
                    done = true;
                }
                _ => {}
            }
        }
        assert!(done);
        assert!(text.contains("one"));
    }

    #[tokio::test]
    async fn stream_honors_cancel() {
        let p = MockProvider::new("mock")
            .with_reply("alpha beta gamma delta epsilon")
            .with_chunk_delay_ms(5);
        let (tx, rx) = watch::channel(false);
        let mut req = ChatRequest::new("mock-model", vec![Message::user("x")]);
        req.cancel = Some(rx);
        let mut stream = p.stream(req).await.unwrap();
        // Cancel after first event opportunity
        let _ = stream.next().await;
        let _ = tx.send(true);
        let mut saw_cancel = false;
        while let Some(ev) = stream.next().await {
            if let Ok(StreamEvent::Done { finish_reason, .. }) = ev {
                if finish_reason == FinishReason::Cancelled {
                    saw_cancel = true;
                }
            }
        }
        assert!(saw_cancel);
    }

    #[tokio::test]
    async fn embeddings_and_token_count() {
        let p = MockProvider::new("mock");
        let emb = p
            .embeddings(EmbeddingRequest {
                model: "mock-model".into(),
                input: vec!["a".into(), "b".into()],
            })
            .await
            .unwrap();
        assert_eq!(emb.vectors.len(), 2);
        assert_eq!(emb.vectors[0].len(), 8);

        let tc = p
            .count_tokens(TokenCountRequest {
                model: "mock-model".into(),
                messages: vec![Message::user("count these tokens please")],
                tools: vec![],
            })
            .await
            .unwrap();
        assert!(tc.tokens > 0);
        assert!(tc.approximate);
    }

    #[tokio::test]
    async fn list_models() {
        let p = MockProvider::new("mock");
        let models = p.list_models().await.unwrap();
        assert!(!models.is_empty());
    }
}
