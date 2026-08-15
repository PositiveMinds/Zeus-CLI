//! Context window management: token-budget tracking and compaction boundary
//! selection. This module is deliberately pure/sync — it decides *where* to
//! split the conversation for compaction; the actual summarization call
//! (which needs the provider) lives in `Agent::maybe_compact`.

use zeus_provider::{Message, Role};

/// Tracks a model's context window and the soft threshold at which
/// compaction should trigger.
#[derive(Debug, Clone)]
pub struct ContextManager {
    pub window: u32,
    pub compact_threshold: f32,
    pub keep_recent_turns: u32,
}

/// Outcome of a compaction pass, for logging/observability.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactResult {
    pub compacted: bool,
    pub removed_messages: usize,
}

impl ContextManager {
    pub fn new(window: u32, compact_threshold: f32, keep_recent_turns: u32) -> Self {
        Self {
            window,
            compact_threshold: compact_threshold.clamp(0.1, 0.99),
            keep_recent_turns,
        }
    }

    /// True once `current_tokens` crosses the soft threshold of the window.
    pub fn should_compact(&self, current_tokens: u32) -> bool {
        self.window > 0 && (current_tokens as f32) >= (self.window as f32 * self.compact_threshold)
    }

    /// Approximate message count to keep verbatim, derived from
    /// `keep_recent_turns` (~2 messages per user/assistant turn; tool
    /// messages riding along get pulled in for free by the pairing-safety
    /// rule in `compaction_boundary`).
    pub fn keep_recent_messages(&self) -> usize {
        (self.keep_recent_turns as usize) * 2
    }

    /// Index into `messages` marking the start of the "keep verbatim" tail;
    /// everything before this index is a candidate for summarization.
    ///
    /// Invariants preserved:
    /// - Leading system message(s) are never summarized away.
    /// - The boundary never splits an assistant tool-call message from its
    ///   tool result message(s) — if it would, the boundary is pushed
    ///   earlier (grown into the "keep" side) until the pair is intact.
    pub fn compaction_boundary(&self, messages: &[Message], keep_recent: usize) -> usize {
        if messages.len() <= keep_recent {
            return 0;
        }
        let mut first_non_system = 0;
        while first_non_system < messages.len() && messages[first_non_system].role == Role::System {
            first_non_system += 1;
        }

        let mut boundary = (messages.len() - keep_recent).max(first_non_system);

        // If the first kept message is a tool result, its owning assistant
        // tool-call message is earlier (on the summarize side) — walk the
        // boundary back until it's no longer sitting on a Tool message.
        while boundary > first_non_system && messages[boundary].role == Role::Tool {
            boundary -= 1;
        }
        boundary.min(messages.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(s: &str) -> Message {
        Message::user(s)
    }
    fn assistant(s: &str) -> Message {
        Message::assistant(s)
    }
    fn tool_call_msg(id: &str) -> Message {
        let mut m = Message::assistant("");
        m.tool_calls = vec![zeus_provider::ToolCall {
            id: id.into(),
            name: "read".into(),
            arguments: "{}".into(),
            extra_content: None,
        }];
        m
    }

    #[test]
    fn should_compact_respects_threshold() {
        let ctx = ContextManager::new(1000, 0.8, 6);
        assert!(!ctx.should_compact(700));
        assert!(ctx.should_compact(800));
        assert!(ctx.should_compact(999));
    }

    #[test]
    fn zero_window_never_compacts() {
        let ctx = ContextManager::new(0, 0.8, 6);
        assert!(!ctx.should_compact(1_000_000));
    }

    #[test]
    fn boundary_keeps_all_when_short() {
        let ctx = ContextManager::new(1000, 0.8, 6);
        let messages = vec![user("a"), assistant("b")];
        assert_eq!(ctx.compaction_boundary(&messages, 6), 0);
    }

    #[test]
    fn boundary_never_summarizes_leading_system() {
        let ctx = ContextManager::new(1000, 0.8, 1);
        let messages = vec![
            Message::system("sys"),
            user("1"),
            assistant("2"),
            user("3"),
            assistant("4"),
        ];
        let boundary = ctx.compaction_boundary(&messages, 2);
        assert!(boundary >= 1, "must not summarize the system message");
    }

    #[test]
    fn boundary_never_splits_tool_call_pair() {
        let ctx = ContextManager::new(1000, 0.8, 1);
        let messages = vec![
            user("1"),
            assistant("2"),
            tool_call_msg("call-1"),
            Message::tool_result("call-1", "result"),
            user("3"),
        ];
        // Ask to keep only the last 2 messages — naive math (len - keep)
        // would land the boundary right on the tool-result message,
        // splitting it from the assistant call that requested it.
        let boundary = ctx.compaction_boundary(&messages, 2);
        assert_ne!(messages[boundary].role, Role::Tool);
        // The tool-call message must be on the "keep" side too (index 2).
        assert!(boundary <= 2);
    }
}
