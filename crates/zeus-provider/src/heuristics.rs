//! Shared approximate token-counting heuristic. Used by any backend that
//! doesn't expose a real tokenizer/count endpoint (the mock provider, and
//! local servers like Ollama that don't have a free tokenize call).

use crate::types::{Message, ToolSpec};

/// Simple whitespace / punctuation token estimator (~4 chars per token).
pub fn estimate_tokens(text: &str) -> u32 {
    if text.is_empty() {
        return 0;
    }
    let words = text.split_whitespace().count() as u32;
    let by_chars = (text.len() as u32).div_ceil(4);
    words.max(by_chars / 2).max(1)
}

pub fn estimate_messages(messages: &[Message], tools: &[ToolSpec]) -> u32 {
    let msg_tokens: u32 = messages
        .iter()
        .map(|m| estimate_tokens(&m.content) + 4) // role overhead
        .sum();
    let tool_tokens: u32 = tools
        .iter()
        .map(|t| {
            estimate_tokens(&t.name)
                + estimate_tokens(&t.description)
                + estimate_tokens(&t.parameters.to_string())
                + 8
        })
        .sum();
    msg_tokens + tool_tokens
}
