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
mod analyze;
mod background;
mod commands;
mod context;
mod docread;
mod error;
mod hooks;
mod mcp;
mod personas;
mod plans;
mod plugin;
mod project;
mod session;
mod skills;
mod terminal;
mod tools;
mod workflows;

pub use agent::{Agent, AgentEvent, AgentOptions, ContextUsage, PlanStep, StepResult, TurnResult};
pub use analyze::{analyze_repo, GitReport, ProbeHit, ProbeReport, RepoFile, RepoFingerprint};
pub use background::{BackgroundTask, BackgroundTaskRegistry, TaskStatus};
pub use commands::{ExpandResult, SlashCommands};
pub use context::{CompactResult, ContextManager};
pub use docread::{extract as extract_document, Document};
pub use error::{AgentError, Result};
pub use hooks::{HookRunner, PreToolUseOutcome};
pub use mcp::{McpClient, McpTool};
pub use personas::{
    load_custom_personas, persona_by_id, personas_by_department, recommend_persona,
    recommend_reviewer, Persona, ALL_PERSONAS,
};
pub use plans::{StepMetrics, TaskPlan, TaskStep};
pub use plugin::{load_all as load_all_plugins, LoadedPlugin, PluginCallResult};
pub use project::load_or_analyze;
pub use session::{
    new_session_id, ConversationState, SessionStore, SessionSummary, TranscriptEntry,
};
pub use skills::{discover_in_dir, parse_skill, skill_resources, Skill, SkillArg, SkillTier};
pub use terminal::{
    CommandHistory, CommandProfile, CommandRecord, Sandbox, TerminalOptions, TerminalOutput,
    TerminalRunner,
};
pub use tools::{builtin_tool_specs, platform_tool_specs, ToolManager, ToolResult};
pub use workflows::{
    discover_all as discover_workflows, parse_workflow, Workflow, WorkflowPhaseDef,
};
