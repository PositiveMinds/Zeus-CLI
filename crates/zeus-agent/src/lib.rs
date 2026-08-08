//! Phase 3 — Execution: agent loop, context management, tools, terminal.
//!
//! Core cycle:
//! 1. Append user message
//! 2. Compact context if near budget
//! 3. Stream provider response
//! 4. Execute tool calls (permission-gated), append results
//! 5. Repeat until final answer or cancel
//! 6. Checkpoint conversation at turn boundary

mod agent;
mod background;
mod commands;
mod context;
mod error;
mod hooks;
mod mcp;
mod personas;
mod plans;
mod plugin;
mod session;
mod terminal;
mod tools;

pub use agent::{Agent, AgentEvent, AgentOptions, ContextUsage, PlanStep, TurnResult};
pub use background::{BackgroundTask, BackgroundTaskRegistry, TaskStatus};
pub use commands::{ExpandResult, SlashCommands};
pub use context::{ContextManager, CompactResult};
pub use error::{AgentError, Result};
pub use hooks::{HookRunner, PreToolUseOutcome};
pub use mcp::{McpClient, McpTool};
pub use personas::{
    load_custom_personas, persona_by_id, personas_by_department, recommend_persona,
    recommend_reviewer, Persona, ALL_PERSONAS,
};
pub use plans::{TaskPlan, TaskStep};
pub use plugin::{load_all as load_all_plugins, LoadedPlugin, PluginCallResult};
pub use session::{new_session_id, ConversationState, SessionStore, TranscriptEntry};
pub use terminal::{
    CommandHistory, CommandProfile, CommandRecord, Sandbox, TerminalOptions, TerminalRunner,
    TerminalOutput,
};
pub use tools::{builtin_tool_specs, platform_tool_specs, ToolManager, ToolResult};
