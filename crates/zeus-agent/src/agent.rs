//! The Agent Loop: message history ⇄ tool calls ⇄ tool results, cancellable.
//!
//! Cycle (matches the blueprint's "The Agent Loop" section):
//! 1. Append user message.
//! 2. Compact context if near the model's window (see `context`).
//! 3. Stream the provider's response.
//! 4. Resolve any tool calls through the Tool Manager (permission-gated).
//! 5. Repeat until a final answer with no pending tool calls, or cancelled.
//! 6. Persist conversation state at the turn boundary.

use crate::context::{CompactResult, ContextManager};
use crate::error::{AgentError, Result};
use crate::personas::{
    persona_by_id, personas_by_department, recommend_persona, recommend_reviewer, Persona,
};
use crate::plans::TaskPlan;
use crate::session::{ConversationState, SessionStore};
use crate::tools::{ToolManager, ToolResult};
use futures::StreamExt;
use std::collections::HashMap;
use tokio::sync::watch;
use tracing::{debug, warn};
use zeus_fs::{ApprovalDecision, PermissionRequest};
use zeus_provider::{
    ChatRequest, FinishReason, Message, ModelProvider, Role, StreamEvent, TokenCountRequest,
    TokenUsage, ToolCall, ToolSpec,
};

#[derive(Debug, Clone)]
pub struct AgentOptions {
    pub model: String,
    /// Safety valve against a runaway tool-call loop.
    pub max_tool_iterations: usize,
    pub temperature: Option<f32>,
    /// Caps how many tokens a single reply may generate — bounds worst-case
    /// latency (a model that rambles on with no natural stop point,
    /// especially slow CPU-bound local inference, otherwise generates for as
    /// long as its context window allows). `None` leaves it uncapped.
    pub max_tokens: Option<u32>,
    /// How many independent *read-only* plan steps may run concurrently.
    /// File-mutating steps always run sequentially (they touch the shared
    /// workspace, and we don't want concurrent editors stepping on each
    /// other). `1` disables parallelism entirely — the prior sequential
    /// behaviour. Amounts to "bounded safe parallelism".
    pub max_parallel_read_steps: usize,
    /// Where to persist the structured plan (`.agent/tasks.json`). When
    /// `None`, Auto/plan runs skip persistence but the approval gate still
    /// applies.
    pub tasks_file: Option<std::path::PathBuf>,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            model: "llama3.2".into(),
            max_tool_iterations: 8,
            temperature: None,
            max_tokens: None,
            max_parallel_read_steps: 2,
            tasks_file: None,
        }
    }
}

/// Streamed events for a single turn, surfaced to whatever UI drives the agent.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    TextDelta(String),
    ToolCallStarted {
        id: String,
        name: String,
        arguments: String,
    },
    ToolCallFinished {
        id: String,
        name: String,
        result: String,
        is_error: bool,
    },
    Compacted(CompactResult),
    Cancelled,
    Done,
    /// The model called `todowrite` — it owns and replaces its whole
    /// checklist on every call (not an incremental patch), so the UI should
    /// just overwrite whatever it was showing rather than merge.
    TodosUpdated {
        todos: Vec<TodoStatus>,
    },
    /// Orchestrated `/plan` runs: the planning pass produced an ordered
    /// list of subtasks to execute.
    PlanGenerated {
        steps: Vec<PlanStep>,
    },
    /// A subtask from the plan is about to be executed as its own turn.
    PlanStepStarted {
        step: PlanStep,
    },
    /// A subtask finished and produced a final answer.
    PlanStepDone {
        step: PlanStep,
        summary: String,
    },
    /// A review pass over completed plan work ran; `persona` is the reviewer
    /// id that drove it and `report` is its findings.
    PlanReviewed {
        persona: String,
        report: String,
    },
    /// The lead-reviewer gate rejected the completed work: the plan ran, but
    /// the reviewer's findings mean it isn't DONE. `report` is the reviewer
    /// verdict the user declined to accept.
    OrchestrationRevision {
        report: String,
    },
    /// The user declined an individual recommended step; it is skipped while
    /// the rest of the plan continues.
    PlanStepDeclined {
        step: PlanStep,
    },
    /// All subtasks completed; `summary` is the combined result.
    OrchestrationDone {
        summary: String,
    },
    /// A `/workflow` run started: `id` is the workflow name and `description`
    /// its one-liner, `phases` the ordered specialist pipeline it will follow.
    WorkflowStarted {
        id: String,
        description: String,
        phases: Vec<crate::workflows::WorkflowPhaseDef>,
    },
    /// One phase of a `/workflow` run is about to execute as its own turn.
    WorkflowPhaseStarted {
        name: String,
        persona: String,
    },
    /// A `/workflow` phase finished; `summary` is its final answer.
    WorkflowPhaseDone {
        name: String,
        persona: String,
        summary: String,
    },
    /// Every phase of a `/workflow` run completed; `summary` is the stitched
    /// result.
    WorkflowDone {
        summary: String,
    },
    /// Repository understanding (computed once per session): `stack` is the
    /// deterministic project banner, `relevance` is this request's existing-
    /// code matches. Shown to the user so they see what zeus found before it
    /// writes anything.
    RepoAnalyzed {
        stack: String,
        relevance: String,
    },
    RepoRelevanceUpdated {
        relevance: String,
    },
    /// `/orient` finished and wrote the generated docs into `.agent/`.
    OrientationSaved {
        docs: crate::project::WrittenDocs,
    },
    /// A standalone `/review` pass over the current uncommitted diff finished;
    /// `persona` is the reviewer id that drove it and `report` its findings.
    ReviewUncommitted {
        persona: String,
        report: String,
    },
    /// A `/suggest` pass produced next-feature recommendations grounded in the
    /// project's current implementation; `report` is the ranked list.
    FeaturesSuggested {
        report: String,
    },
}

/// One row of the model-owned checklist the `todowrite` tool writes —
/// mirrors that tool's own JSON shape (`content`/`status`) rather than a
/// richer type, since it's parsed straight out of the tool call's raw
/// arguments after dispatch.
#[derive(Debug, Clone)]
pub struct TodoStatus {
    pub content: String,
    pub status: String,
}

/// One subtask in an orchestrated `/plan` run.
#[derive(Debug, Clone)]
pub struct PlanStep {
    pub id: usize,
    pub description: String,
    /// Why this approach (what the planner recommends and its trade-offs), so
    /// the user can make an informed accept/decline per step.
    pub rationale: String,
    /// Optional specialist-agent id (from `MVP_PERSONAS`) to steer this step;
    /// `None` means run it with the generic coding agent.
    pub persona: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TurnResult {
    pub final_text: String,
    pub tool_calls: usize,
    pub cancelled: bool,
    /// Summed across every model round trip this turn made (a turn can
    /// call the provider more than once — one per tool-call iteration —
    /// so this is the total for the whole turn, not just the last call).
    pub usage: TokenUsage,
}

/// Snapshot of context usage for a `/context`-style status line.
#[derive(Debug, Clone)]
pub struct ContextUsage {
    pub tokens: u32,
    pub approximate: bool,
    pub window: u32,
    pub message_count: usize,
}

/// Ties together the provider, tools, context management, and session
/// persistence into the runnable per-turn loop.
pub struct Agent {
    provider: std::sync::Arc<dyn ModelProvider>,
    tools: ToolManager,
    context: ContextManager,
    sessions: SessionStore,
    state: ConversationState,
    options: AgentOptions,
    cancel_tx: watch::Sender<bool>,
    cancel_rx: watch::Receiver<bool>,
    /// Master switch for *automatic* per-turn compaction (the threshold
    /// check before each turn). `compact_now`/`/compact` ignores this and
    /// always runs. Default on.
    auto_compact: std::sync::atomic::AtomicBool,
    /// Auto mode: each turn is first planned (read-only) and then executed
    /// step-by-step through the orchestrator, instead of just being executed.
    /// Off (Build) by default; enabled by selecting Auto in the mode switch.
    auto_mode: std::sync::atomic::AtomicBool,
    /// Repository understanding, computed once per session and reused. Lets
    /// the agent ground every request in what already exists (reuse > rewrite).
    repo: Option<crate::analyze::RepoFingerprint>,
}

impl Agent {
    pub fn new(
        provider: std::sync::Arc<dyn ModelProvider>,
        tools: ToolManager,
        context: ContextManager,
        sessions: SessionStore,
        state: ConversationState,
        options: AgentOptions,
    ) -> Self {
        let (cancel_tx, cancel_rx) = watch::channel(false);
        Self {
            provider,
            tools,
            context,
            sessions,
            state,
            options,
            cancel_tx,
            cancel_rx,
            auto_compact: std::sync::atomic::AtomicBool::new(true),
            auto_mode: std::sync::atomic::AtomicBool::new(false),
            repo: None,
        }
    }

    /// Handle for an external caller (UI, signal handler) to cancel the
    /// in-flight turn — aborts both the provider stream and any running tool
    /// call (the `bash` tool checks the same flag via its own cancel token;
    /// callers wanting that to propagate should share one `AtomicBool`/watch
    /// pair between this and the `ToolManager`'s terminal cancel token).
    pub fn cancel_handle(&self) -> watch::Sender<bool> {
        self.cancel_tx.clone()
    }

    pub fn messages(&self) -> &[Message] {
        &self.state.messages
    }

    pub fn session_id(&self) -> &str {
        &self.state.session_id
    }

    pub fn model(&self) -> &str {
        &self.options.model
    }

    pub fn provider_id(&self) -> &str {
        self.provider.id()
    }

    /// The bound project workspace (files, checkpoints, git root) — lets the
    /// UI layer reach checkpoint/git operations (`/undo`, `/diff`) without
    /// re-deriving them from `Config`.
    pub fn workspace(&self) -> &zeus_fs::Workspace {
        self.tools.workspace()
    }

    /// Models the current provider actually has available — backs a
    /// `/model` picker UI (list + select) rather than requiring the user to
    /// already know an exact model name to type.
    pub async fn list_models(&self) -> Result<Vec<zeus_provider::ModelInfo>> {
        self.provider
            .list_models()
            .await
            .map_err(AgentError::Provider)
    }

    /// Plan mode: read-only research/proposal, no mutating tool calls —
    /// enforced centrally in the `ToolManager`, not per-tool, so switching
    /// modes can't be bypassed by a tool configured Allow in settings.
    pub fn set_plan_mode(&self, enabled: bool) {
        self.tools.set_plan_mode(enabled);
    }

    pub fn plan_mode(&self) -> bool {
        self.tools.plan_mode()
    }

    /// Auto mode: each request is first planned then executed step-by-step
    /// (delegating `run_turn` through `orchestrate`), rather than handled in
    /// a single pass. Selectable alongside Build (direct) and Plan (read-only)
    /// in the mode switch.
    pub fn set_auto_mode(&self, enabled: bool) {
        self.auto_mode
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn auto_mode(&self) -> bool {
        self.auto_mode.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Toggle automatic per-turn compaction (`/autocompact on|off`). When
    /// disabled, the context is left to grow until the user runs `/compact` —
    /// useful on long, chatty sessions where you'd rather not lose earlier
    /// detail. Explicit `/compact` always works regardless of this.
    pub fn set_auto_compact(&self, enabled: bool) {
        self.auto_compact
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn auto_compact(&self) -> bool {
        self.auto_compact.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Switch which model future turns use, keeping the same conversation
    /// history and session — same provider only (switching providers mid-
    /// session would mean rebuilding the whole tool/workspace stack, out of
    /// scope for a `/model` switch).
    pub fn set_model(&mut self, model: impl Into<String>) {
        self.options.model = model.into();
    }

    /// Swap which provider future turns use — the tool/workspace/session
    /// stack is untouched, so conversation history carries over exactly
    /// like a same-provider `/model` switch does. Callers should also call
    /// `set_model` with a model that actually exists on the new provider.
    pub fn set_provider(&mut self, provider: std::sync::Arc<dyn ModelProvider>) {
        self.provider = provider;
    }

    /// Current context usage for a `/context`-style status line: actual
    /// token count against the model's window, via the same
    /// `count_tokens` call the automatic compaction check uses.
    pub async fn context_usage(&self) -> Result<ContextUsage> {
        let count = self
            .provider
            .count_tokens(TokenCountRequest {
                model: self.options.model.clone(),
                messages: self.state.messages.clone(),
                tools: self.tools.all_tool_specs(),
            })
            .await
            .map_err(AgentError::Provider)?;
        Ok(ContextUsage {
            tokens: count.tokens,
            approximate: count.approximate,
            window: self.context.window,
            message_count: self.state.messages.len(),
        })
    }

    /// Run one full turn: user message in, tool-call cycle, final answer out.
    pub async fn run_turn<E, A>(
        &mut self,
        user_message: &str,
        mut on_event: E,
        approver: A,
    ) -> Result<TurnResult>
    where
        E: FnMut(AgentEvent),
        A: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        // Fresh cancellation scope for this turn.
        let _ = self.cancel_tx.send(false);
        let report = self.repo_context(user_message, &mut on_event);
        let mut content = if report.is_empty() {
            // Small talk / meta (see `request_likely_needs_context`) — no
            // repo banner, project rules, memory, or "verify with grep"
            // nudge wrapped around it; none of that helps answer "hello"
            // and all of it costs real tokens on every single turn
            // otherwise, not just the ones that actually touch the repo.
            user_message.to_string()
        } else {
            format!("{report}\n\n---\nUser: {user_message}")
        };
        // Auto mode used to mean "plan the request into steps up front,
        // execute each as its own turn" (`orchestrate`) — dropped in favor
        // of the same continuous loop every other mode uses, driven by a
        // prompt nudge instead of a separate code path. A plan drawn up
        // before any work starts goes stale the moment step 1 reveals
        // something step 3 assumed wasn't true; a model tracking its own
        // checklist (`todowrite`) as it actually learns things doesn't
        // have that failure mode. `delegate` covers the other half of what
        // the old planner did (assigning a specialist to a step) without
        // requiring the assignment to be decided before execution begins.
        // `zeus agent --auto`/`/bg orchestrate` still get the full
        // reviewable upfront plan via `orchestrate()` directly, for
        // whoever explicitly wants that ceremony.
        if self.auto_mode() {
            content.push_str(
                "\n\n---\nAuto mode: work through this fully autonomously. Break multi-step \
                 work down with `todowrite` and keep it current as you go; reach for `delegate` \
                 when part of the work clearly calls for a specific specialist's expertise \
                 (see its tool description for the roster). Keep going until the request is \
                 genuinely done rather than stopping to check in early.",
            );
        }
        self.state.messages.push(Message::user(content));

        if let Some(result) = self.maybe_compact().await? {
            on_event(AgentEvent::Compacted(result));
        }

        self.drive_turn(on_event, approver).await
    }

    /// Cheap, deterministic check for whether a request is plausibly about the
    /// codebase at all. Greetings/small talk/meta questions get sent to the
    /// model as-is, with no repo banner, project rules, memory context, or
    /// "verify with grep/glob" nudge attached — none of that helps answer
    /// "hello", it costs real tokens on *every* turn regardless of relevance
    /// (`repo_context` used to run unconditionally), and the "verify with grep"
    /// instruction in particular risked nudging the model toward an unneeded
    /// exploratory tool call on a message that was never asking for one.
    ///
    /// Errs toward `true` (include full context) whenever it isn't confident —
    /// a false "needs context" costs a few hundred tokens; a false "doesn't
    /// need it" risks answering a real coding question blind. Only an exact
    /// (case-insensitive, punctuation-trimmed) match against a short list of
    /// unambiguous small talk short-circuits it; anything longer or unrecognized
    /// falls through to the normal full-context path.
    fn request_likely_needs_context(request: &str) -> bool {
        let trimmed = request.trim();
        if trimmed.chars().count() > 60 {
            return true;
        }
        let lower = trimmed.to_ascii_lowercase();
        let lower = lower.trim_end_matches(|c: char| c.is_ascii_punctuation());
        const SMALL_TALK: &[&str] = &[
            "hi",
            "hello",
            "hey",
            "yo",
            "sup",
            "hiya",
            "howdy",
            "thanks",
            "thank you",
            "thx",
            "ty",
            "cheers",
            "ok",
            "okay",
            "cool",
            "nice",
            "great",
            "awesome",
            "got it",
            "sounds good",
            "np",
            "how are you",
            "how's it going",
            "who are you",
            "what are you",
            "what can you do",
            "good morning",
            "good afternoon",
            "good evening",
            "good night",
            "bye",
            "goodbye",
            "see you",
            "later",
            "yes",
            "no",
            "sure",
            "yep",
            "nope",
        ];
        !SMALL_TALK.contains(&lower)
    }

    /// Compute (once) and root the conversation in the repository: cache the
    /// deterministic fingerprint and return the "repository understanding"
    /// block for this request (stack banner + existing related code). The
    /// fingerprint is also handed to the tool layer (`understand_repo`).
    fn repo_context(&mut self, request: &str, on_event: &mut dyn FnMut(AgentEvent)) -> String {
        let root = self.tools.project_root();
        if self.repo.is_none() {
            let fp = crate::project::load_or_analyze(&root);
            let stack = fp.banner_lines().join("\n");
            self.tools.set_repo(Some(fp.clone()));
            self.repo = Some(fp);
            on_event(AgentEvent::RepoAnalyzed {
                stack,
                relevance: String::new(),
            });
        }
        if !Self::request_likely_needs_context(request) {
            return String::new();
        }
        let fp = self.repo.as_ref().expect("repo seeded above");
        let probe = fp.probe(request);
        let relevance = if probe.hits.is_empty() {
            String::new()
        } else {
            probe.render()
        };
        let banner = fp.banner_lines().join("\n");
        if !relevance.is_empty() {
            on_event(AgentEvent::RepoRelevanceUpdated {
                relevance: relevance.clone(),
            });
        }
        let mut out = format!("Repository understanding:\n{banner}");
        let rules = crate::project::project_rules_context(&root);
        if !rules.is_empty() {
            out.push_str("\n\n");
            out.push_str(&rules);
        }
        let memory = crate::project::memory_context(&root, request);
        if !memory.is_empty() {
            out.push_str("\n\n");
            out.push_str(&memory);
        }
        if relevance.is_empty() {
            out.push_str(
                "\n\nNo obviously-relevant existing modules matched this request by name. \
                 Verify with grep/glob before writing new files; if nothing exists, build from scratch.",
            );
        } else {
            out.push_str("\n\n");
            out.push_str(&relevance);
        }
        out
    }

    /// On-demand semantic repository pass (`/understand <topic>`): under
    /// read-only Plan mode, drive the model to read the relevant files and
    /// report what already exists, is partial/stubbed, and what's missing —
    /// grounded in the deterministic fingerprint + probe above.
    pub async fn understand_topic<E, A>(
        &mut self,
        topic: &str,
        mut on_event: E,
        mut approver: A,
    ) -> Result<TurnResult>
    where
        E: FnMut(AgentEvent),
        A: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let _ = self.cancel_tx.send(false);
        let report = self.repo_context(topic, &mut on_event);
        let prompt = format!(
            "Repository intelligence scan for: `{topic}`.\n\
             Read the relevant existing files below (plus any you discover with grep/glob/code_symbols) \
             and produce a terse, precise report answering:\n\
             - What already EXISTS for this subject (components, modules, schemas, tables, routes, \
               middlewares, config, helpers) — exact file paths.\n\
             - What is PARTIAL or stubbed (file exists but unimplemented).\n\
             - What is MISSING entirely.\n\
             End with a 3-5 bullet summary and the top relevant file paths.\n\
             Do NOT modify any files — read-only.\n\n{report}",
            report = report
        );
        self.state.messages.push(Message::user(prompt));

        if let Some(result) = self.maybe_compact().await? {
            on_event(AgentEvent::Compacted(result));
        }

        let was_plan = self.plan_mode();
        self.set_plan_mode(true);
        let turn = self.drive_turn(&mut on_event, &mut approver).await;
        self.set_plan_mode(was_plan);
        turn
    }

    /// Repository orientation (`/orient`): a read-only pass that reads the
    /// core modules and produces two docs — `.agent/architecture.md` and
    /// `.agent/conventions.md` — marking the model's sight of the project in
    /// the persistent map. Never modifies project code.
    pub async fn orient_turn<E, A>(
        &mut self,
        mut on_event: E,
        mut approver: A,
    ) -> Result<TurnResult>
    where
        E: FnMut(AgentEvent),
        A: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let _ = self.cancel_tx.send(false);
        let report = self.repo_context("orientation architecture conventions", &mut on_event);
        let prompt = format!(
            "Project orientation pass. Read this repo understanding below, then explore the entry \
             points and core modules with read/grep/glob/code_symbols (sparingly — prefer the \
             understanding plus a handful of reads) and produce TWO documents, wrapped in fenced \
             markers, in your FINAL ANSWER, in this exact order:\n\n\
             [ARCH]\n# Architecture\n- stack and framework overview\n- entry points and startup order\n\
             - module/layer map with file paths\n- how components talk (routing, DI, events, data flow)\n\
             - key tables/schemas or data stores\n[/ARCH]\n\n\
             [CONV]\n## Conventions\n- languages, tooling, formatting\n- naming and project structure rules\n\
             - error handling and testing patterns\n- git/workflow conventions\n- anti-patterns / gotchas\n\
             [/CONV]\n\n\
             Keep each doc under ~80 lines: terse bullet style. This is read-only — do NOT modify files.\n\n\
             {report}",
            report = report
        );
        self.state.messages.push(Message::user(prompt));

        if let Some(result) = self.maybe_compact().await? {
            on_event(AgentEvent::Compacted(result));
        }

        let was_plan = self.plan_mode();
        self.set_plan_mode(true);
        let turn = self.drive_turn(&mut on_event, &mut approver).await;
        self.set_plan_mode(was_plan);
        let turn = turn?;

        let (arch, conv) = split_orientation_docs(&turn.final_text);
        let root = self.tools.project_root();
        let mut written = crate::project::WrittenDocs::default();
        if let Some(body) = arch {
            written.architecture =
                crate::project::write_generated_doc(&root, crate::project::ARCHITECTURE_DOC, &body);
        }
        if let Some(body) = conv {
            written.conventions =
                crate::project::write_generated_doc(&root, crate::project::CONVENTIONS_DOC, &body);
        }
        on_event(AgentEvent::OrientationSaved { docs: written });

        Ok(turn)
    }

    /// Standalone review of the current uncommitted changes (`/review`): a
    /// read-only pass that gathers the working-tree diff, matches a
    /// `reviewer: true` persona to it, and drives the model to report
    /// concrete findings. Never modifies any files — plan mode is forced for
    /// the whole turn. Emits `ReviewUncommitted` with the report.
    pub async fn review_turn<E, A>(
        &mut self,
        mut on_event: E,
        mut approver: A,
    ) -> Result<TurnResult>
    where
        E: FnMut(AgentEvent),
        A: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let _ = self.cancel_tx.send(false);
        let report = self.repo_context("review uncommitted changes", &mut on_event);
        let persona = recommend_reviewer("uncommitted changes review");
        let hunter = persona
            .map(|p| prepend_persona_prompt(&mut self.state, p.id))
            .unwrap_or(false);

        let prompt = format!(
            "Review the uncommitted working-tree changes in this repository.\n\
             Start by running `git_status` and `git_diff` (both unstaged and staged) \
             to see exactly what changed, then read the changed files as needed.\n\
             Report concrete findings:\n\
             - correctness bugs and regressions\n\
             - security or safety concerns\n\
             - missing tests or broken assumptions\n\
             - naming, structure, and style issues\n\
             End with a one-line verdict (APPROVE / REVISE) and the top 3 issues, if any.\n\
             Review only — do NOT modify any files.\n\n{report}",
            report = report
        );
        self.state.messages.push(Message::user(prompt));

        if let Some(result) = self.maybe_compact().await? {
            on_event(AgentEvent::Compacted(result));
        }

        let was_plan = self.plan_mode();
        self.set_plan_mode(true);
        let turn = self.drive_turn(&mut on_event, &mut approver).await;
        self.set_plan_mode(was_plan);
        let turn = turn?;

        if hunter {
            self.state.messages.remove(0);
        }
        let persona_id = persona
            .map(|p| p.id.to_string())
            .unwrap_or_else(|| "reviewer".into());
        on_event(AgentEvent::ReviewUncommitted {
            persona: persona_id,
            report: turn.final_text.clone(),
        });

        Ok(turn)
    }

    /// Next-feature recommendations (`/suggest`): a read-only pass that reads
    /// the repository understanding and proposes the most relevant features to
    /// implement next, ranked by fit and effort. Always grounded in what
    /// already exists (the persistent map + probe) so suggestions stay
    /// project-specific rather than generic. Emits `FeaturesSuggested`.
    pub async fn suggest_turn<E, A>(
        &mut self,
        mut on_event: E,
        mut approver: A,
    ) -> Result<TurnResult>
    where
        E: FnMut(AgentEvent),
        A: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let _ = self.cancel_tx.send(false);
        let report = self.repo_context("next features to implement", &mut on_event);
        let prompt = format!(
            "Recommend the 3-5 most valuable features to build NEXT in this project, \
             grounded strictly in what already EXISTS here.\n\n\
             For each feature give:\n\
             - name and the real file/module it would extend or add\n\
             - why it is the natural next step given the current implementation\n\
             - what depends on it or blocks it\n\
             - rough effort size (S/M/L)\n\
             Rank by impact-to-effort. Skip anything the repo already has. Do NOT modify files.\n\n{report}",
            report = report
        );
        self.state.messages.push(Message::user(prompt));

        if let Some(result) = self.maybe_compact().await? {
            on_event(AgentEvent::Compacted(result));
        }

        let was_plan = self.plan_mode();
        self.set_plan_mode(true);
        let turn = self.drive_turn(&mut on_event, &mut approver).await;
        self.set_plan_mode(was_plan);
        let turn = turn?;

        on_event(AgentEvent::FeaturesSuggested {
            report: turn.final_text.clone(),
        });

        Ok(turn)
    }

    /// Plan mode (v1) entry point: research the goal read-only and produce a
    /// structured, *persisted* plan — without executing anything. Unlike an
    /// auto-mode run, this never touches the workspace: plan mode is forced
    /// on for the duration, so every tool call the research pass makes is
    /// read-only. The plan is written to `options.tasks_file`
    /// (`.agent/tasks.json`) with `approved: false`, leaving the actual
    /// execution to a later Auto-mode run (which gates on user approval).
    pub async fn plan_turn<E, A>(
        &mut self,
        goal: &str,
        mut on_event: E,
        mut approver: A,
    ) -> Result<TurnResult>
    where
        E: FnMut(AgentEvent),
        A: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let _ = self.cancel_tx.send(false);
        let report = self.repo_context(goal, &mut on_event);
        self.state
            .messages
            .push(Message::user(format!("{report}\n\n---\nPlan goal: {goal}")));

        if let Some(result) = self.maybe_compact().await? {
            on_event(AgentEvent::Compacted(result));
        }

        let (steps, plan_usage) = self.plan_task(goal).await?;
        on_event(AgentEvent::PlanGenerated {
            steps: steps.clone(),
        });

        // Research pass: read-only by force, so the plan is grounded in real
        // files rather than guesses. Runs as the goal's recommended
        // specialist — the plan "lead" — so the approach write-up comes from
        // the roster member who best matches the work. The model's final
        // answer is the approach write-up (persisted as the plan's `notes`).
        let was_plan = self.plan_mode();
        self.set_plan_mode(true);
        let persona_injected = recommend_persona(goal)
            .map(|p| prepend_persona_prompt(&mut self.state, p.id))
            .unwrap_or(false);
        let turn = self.drive_turn(&mut on_event, &mut approver).await;
        if persona_injected {
            self.state.messages.remove(0);
        }
        self.set_plan_mode(was_plan);
        let mut turn = turn?;
        add_usage(&mut turn.usage, &plan_usage);

        self.write_task_plan(&TaskPlan::from_steps(goal, &steps, &turn.final_text, false))?;

        Ok(turn)
    }

    /// Persist the current structured plan to `.agent/tasks.json` (if a
    /// `tasks_file` was configured). Silently skipped otherwise.
    fn write_task_plan(&self, plan: &TaskPlan) -> Result<()> {
        if let Some(path) = &self.options.tasks_file {
            plan.write(path)?;
        }
        Ok(())
    }

    /// Orchestrated `/plan` run: ask a planning-only call (no tools) to
    /// break the goal into an ordered list of subtasks, then execute each
    /// subtask as its own full tool-using turn, carrying forward a summary
    /// of what earlier steps did. Sequential by design — steps like "run the
    /// tests" after "edit the file" depend on order, and concurrent tool-
    /// using agents against the same working directory race file writes.
    ///
    /// Events are surfaced per stage: `PlanGenerated` once, then
    /// `PlanStepStarted`/agent events/`PlanStepDone` per subtask, then
    /// `OrchestrationDone` with the combined summary.
    pub async fn orchestrate<E, A>(
        &mut self,
        goal: &str,
        mut on_event: E,
        mut approver: A,
    ) -> Result<(String, TokenUsage)>
    where
        E: FnMut(AgentEvent),
        A: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let _ = self.cancel_tx.send(false);
        let mut total_usage = TokenUsage::default();

        let (steps, usage) = self.plan_task(goal).await?;
        add_usage(&mut total_usage, &usage);
        on_event(AgentEvent::PlanGenerated {
            steps: steps.clone(),
        });

        // Persist the drafted plan up front, then hold at the review gate:
        // nothing below executes until the user approves. This is the
        // "review-before-execute" step — Auto mode used to plan-then-run
        // with no checkpoint, so a misframed goal would sail straight into
        // file edits.
        let mut plan = TaskPlan::from_steps(goal, &steps, "", false);
        // Snapshot any previously persisted plan BEFORE overwriting it, so a
        // re-planned run can diff the new step list against the old one rather
        // than making the user re-read everything.
        let prior_plan = match &self.options.tasks_file {
            Some(path) => TaskPlan::read(path)?,
            None => None,
        };
        self.write_task_plan(&plan)?;

        let mut preview = self.render_plan_preview(&steps);
        if let (Some(path), Some(prior)) = (&self.options.tasks_file, prior_plan) {
            if let Some(diff) = plan.diff_vs(&prior) {
                preview.push_str(&format!("\n--- plan changed vs {}\n{diff}", path.display()));
            }
        }

        let approved = matches!(
            approver(&PermissionRequest {
                tool: "plan_execute".into(),
                path: self.options.tasks_file.clone(),
                command: None,
                description: format!(
                    "execute the reviewed plan ({} step(s)) for: {goal}",
                    steps.len()
                ),
                preview: Some(preview),
                overwrites: false,
            }),
            ApprovalDecision::Approved | ApprovalDecision::ApprovedForSession
        );
        if !approved {
            let summary = format!(
                "plan drafted ({n} step(s)) but you declined to execute — nothing changed. \
                 Review {path} (and/or fine-tune the goal) then rerun to approve.",
                n = steps.len(),
                path = self
                    .options
                    .tasks_file
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "the plan".to_string())
            );
            on_event(AgentEvent::OrchestrationDone {
                summary: summary.clone(),
            });
            return Ok((summary, total_usage));
        }
        plan.approved = true;
        self.write_task_plan(&plan)?;

        // Per-step acceptance: each recommended approach can be accepted or
        // declined independently. Declined steps are skipped (and reported),
        // the accepted ones run in order. `ApprovedForSession` auto-accepts
        // every remaining step without re-prompting.
        let mut accepted: Vec<PlanStep> = Vec::with_capacity(steps.len());
        let mut declined: Vec<PlanStep> = Vec::new();
        let mut auto_accept = false;
        for step in &steps {
            if auto_accept {
                accepted.push(step.clone());
                continue;
            }
            let decision = approver(&PermissionRequest {
                tool: "plan_step".into(),
                path: self.options.tasks_file.clone(),
                command: None,
                description: format!("accept recommended step {}: {}", step.id, step.description),
                preview: Some(Self::step_preview(step)),
                overwrites: false,
            });
            match decision {
                ApprovalDecision::Approved | ApprovalDecision::ApprovedForSession => {
                    if decision == ApprovalDecision::ApprovedForSession {
                        auto_accept = true;
                    }
                    accepted.push(step.clone());
                }
                ApprovalDecision::Denied => {
                    declined.push(step.clone());
                    on_event(AgentEvent::PlanStepDeclined { step: step.clone() });
                }
            }
        }
        let steps: Vec<PlanStep> = accepted;

        let mut summaries: Vec<String> = Vec::new();
        let mut prior_content = String::new();

        // Safe, bounded parallelism: consecutive *read-only* steps (personas
        // that only inspect) may run as independent headless provider calls —
        // they never mutate the shared workspace or conversation, so there's
        // no edit race. File-mutating steps stay on the sequential `drive_turn`
        // loop below. `max_parallel_read_steps` caps how many run at once;
        // `1` reproduces the old fully-sequential behaviour.
        let parallel = self.options.max_parallel_read_steps.max(1);
        let mut idx = 0usize;
        let steps_slice: Vec<PlanStep> = steps;
        while idx < steps_slice.len() {
            if *self.cancel_rx.borrow() {
                let summary = self.cancel_orchestration(goal, &summaries, &mut on_event)?;
                return Ok((summary, total_usage));
            }
            // Sweep forward over the run of consecutive read-only steps.
            let mut run_end = idx;
            while run_end < steps_slice.len() && is_read_only_step(&steps_slice[run_end]) {
                run_end += 1;
            }
            let read_run = if run_end > idx && parallel > 1 {
                (idx, run_end)
            } else {
                // Not a run (or parallelism off) — fall through to sequential.
                (0, 0)
            };

            if read_run.1 == 0 {
                // Sequential step (mutating, or parallelism disabled).
                let step = steps_slice[idx].clone();
                on_event(AgentEvent::PlanStepStarted { step: step.clone() });
                let step_prompt = orchestration_step_prompt(goal, &summaries, &step);
                let persona_injected = if let Some(id) = &step.persona {
                    prepend_persona_prompt(&mut self.state, id)
                } else {
                    false
                };
                self.state.messages.push(Message::user(step_prompt));
                let result = self.drive_turn(&mut on_event, &mut approver).await?;
                if persona_injected {
                    self.state.messages.remove(0);
                }
                add_usage(&mut total_usage, &result.usage);
                if result.cancelled {
                    let summary = self.cancel_orchestration(goal, &summaries, &mut on_event)?;
                    return Ok((summary, total_usage));
                }
                prior_content = result.final_text.clone();
                summaries.push(step_summary(&step, &result.final_text));
                let done_id = step.id;
                on_event(AgentEvent::PlanStepDone {
                    step,
                    summary: result.final_text,
                });
                plan.mark_done(done_id);
                self.write_task_plan(&plan)?;
                idx += 1;
            } else {
                // Run the read-only batch concurrently (bounded). All steps in
                // the run share the same completed-so-far snapshot and only
                // read files mentally, so their model calls are independent.
                let (start, end) = (read_run.0, read_run.1);
                let opt_model = self.options.model.clone();
                let opt_temperature = self.options.temperature;
                let opt_max_tokens = self.options.max_tokens;
                let provider = self.provider.clone();
                let cancel = self.cancel_rx.clone();
                let base_snapshot = summaries.join("\n");

                let futures = steps_slice[start..end]
                    .iter()
                    .map(|step| {
                        let step = step.clone();
                        let opt_model = opt_model.clone();
                        let provider = provider.clone();
                        let cancel = cancel.clone();
                        let base_snapshot = base_snapshot.clone();
                        async move {
                            run_headless_step(
                                provider,
                                HeadlessSpec {
                                    model: opt_model,
                                    temperature: opt_temperature,
                                    max_tokens: opt_max_tokens,
                                },
                                &step,
                                goal,
                                &base_snapshot,
                                cancel,
                            )
                            .await
                        }
                    })
                    .collect::<Vec<_>>();
                let results = futures::future::join_all(futures).await;

                let mut batch_cancelled = false;
                for (step, res) in steps_slice[start..end].iter().zip(results) {
                    on_event(AgentEvent::PlanStepStarted { step: step.clone() });
                    match res {
                        Ok((final_text, cancelled, step_usage)) => {
                            add_usage(&mut total_usage, &step_usage);
                            if cancelled {
                                // Later steps in this same `join_all` batch
                                // may have "succeeded" only because they
                                // raced ahead before the cancel signal
                                // landed — treat the whole batch as cut
                                // short rather than reporting some of it
                                // done, since there's no meaningful order
                                // among concurrent steps to say which ones
                                // "really" finished first.
                                batch_cancelled = true;
                                break;
                            }
                            prior_content = final_text.clone();
                            summaries.push(step_summary(step, &final_text));
                            plan.mark_done(step.id);
                            on_event(AgentEvent::PlanStepDone {
                                step: step.clone(),
                                summary: final_text,
                            });
                        }
                        Err(e) => {
                            summaries.push(step_summary(step, &format!("(step failed: {e})")));
                            plan.mark_done(step.id);
                            on_event(AgentEvent::PlanStepDone {
                                step: step.clone(),
                                summary: format!("(step failed: {e})"),
                            });
                        }
                    }
                }
                if batch_cancelled {
                    let summary = self.cancel_orchestration(goal, &summaries, &mut on_event)?;
                    return Ok((summary, total_usage));
                }
                self.write_task_plan(&plan)?;
                idx = end;
            }
        }

        let final_summary = if summaries.len() > 1 {
            format!(
                "Completed {} steps for: {goal}\n{}",
                summaries.len(),
                summaries.join("\n")
            )
        } else {
            prior_content
        };

        // Review the completed work: pick a matching reviewer persona and
        // give it one read-only turn over the combined result. Reviewers are
        // constrained to non-mutating tools, so this can't edit files.
        let (review_report, review_usage, review_cancelled) = self
            .reviewer_pass(goal, &final_summary, &mut on_event, &mut approver)
            .await?;
        add_usage(&mut total_usage, &review_usage);
        if review_cancelled {
            let summary = self.cancel_orchestration(goal, &summaries, &mut on_event)?;
            return Ok((summary, total_usage));
        }

        // Lead-reviewer gate: the work ran, but it isn't DONE until the
        // human accepts the review. When a reviewer produced findings we
        // hold for an explicit accept; auto-approve (`yes`) mode sails
        // through. A rejected review emits `OrchestrationRevision` instead
        // of completing the plan.
        if let Some(report) = review_report {
            let decision = approver(&PermissionRequest {
                tool: "review_accept".into(),
                path: self.options.tasks_file.clone(),
                command: None,
                description: format!("lead reviewer of '{goal}' is done; accept the work?"),
                preview: Some(report.clone()),
                overwrites: false,
            });
            if !matches!(
                decision,
                ApprovalDecision::Approved | ApprovalDecision::ApprovedForSession
            ) {
                let summary =
                    format!("Work for '{goal}' was NOT accepted by the lead reviewer.\n\n{report}");
                on_event(AgentEvent::OrchestrationRevision {
                    report: summary.clone(),
                });
                return Ok((summary, total_usage));
            }
            let summary = self.finish_orchestration(
                format!("{final_summary}\n\nReview:\n{report}"),
                &plan,
                &mut on_event,
            )?;
            return Ok((summary, total_usage));
        }
        let summary = self.finish_orchestration(final_summary, &plan, &mut on_event)?;
        Ok((summary, total_usage))
    }

    /// Cancelled mid-orchestration — reports actual partial progress instead
    /// of letting the loop run to completion and `finish_orchestration` mark
    /// every step done regardless of how far execution actually got. Unlike
    /// `finish_orchestration`, this never touches the persisted plan's
    /// `done` flags, since "cancelled" isn't "done".
    fn cancel_orchestration<E>(
        &mut self,
        goal: &str,
        summaries: &[String],
        on_event: &mut E,
    ) -> Result<String>
    where
        E: FnMut(AgentEvent),
    {
        let summary = if summaries.is_empty() {
            format!("orchestration for '{goal}' cancelled before any step completed.")
        } else {
            format!(
                "orchestration for '{goal}' cancelled after {} of the planned step(s):\n{}",
                summaries.len(),
                summaries.join("\n")
            )
        };
        on_event(AgentEvent::Cancelled);
        self.persist()?;
        Ok(summary)
    }

    /// Stamp the completed plan: every step done, notes = the final
    /// orchestrated summary (including any reviewer report), then emit
    /// `OrchestrationDone`.
    fn finish_orchestration<E>(
        &mut self,
        summary: String,
        plan: &TaskPlan,
        on_event: &mut E,
    ) -> Result<String>
    where
        E: FnMut(AgentEvent),
    {
        let mut plan = plan.clone();
        plan.mark_all_done();
        plan.notes = summary.clone();
        self.write_task_plan(&plan)?;
        on_event(AgentEvent::OrchestrationDone {
            summary: summary.clone(),
        });
        Ok(summary)
    }

    /// Run a declarative multi-specialist workflow: each phase is one full
    /// tool-using turn driven by that phase's persona. Phases run in order
    /// (a pipeline — later phases can rely on earlier ones' work), each
    /// scoped to the overall goal plus a running snapshot of what earlier
    /// phases did. A `gate` phase holds for explicit approval before it runs;
    /// a `read_only` phase forces plan mode regardless of agent mode.
    ///
    /// Unlike `/plan`, the pipeline is fixed by the workflow file rather than
    /// planned by the model — the "assembly line" form of the workforce.
    pub async fn run_workflow<E, A>(
        &mut self,
        goal: &str,
        workflow: &crate::workflows::Workflow,
        mut on_event: E,
        mut approver: A,
    ) -> Result<String>
    where
        E: FnMut(AgentEvent),
        A: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let _ = self.cancel_tx.send(false);

        on_event(AgentEvent::WorkflowStarted {
            id: workflow.id.clone(),
            description: workflow.description.clone(),
            phases: workflow.phases.clone(),
        });

        let mut summaries: Vec<String> = Vec::new();
        for (i, phase) in workflow.phases.iter().enumerate() {
            if *self.cancel_rx.borrow() {
                break;
            }
            on_event(AgentEvent::WorkflowPhaseStarted {
                name: phase.prompt.clone(),
                persona: phase.persona.clone(),
            });
            if phase.gate {
                let decision = approver(&PermissionRequest {
                    tool: "workflow_phase".into(),
                    path: self.options.tasks_file.clone(),
                    command: None,
                    description: format!(
                        "workflow '{}' phase {} — {} (as {}): run it?",
                        workflow.id,
                        i + 1,
                        phase.prompt,
                        phase.persona
                    ),
                    preview: None,
                    overwrites: false,
                });
                if !matches!(
                    decision,
                    ApprovalDecision::Approved | ApprovalDecision::ApprovedForSession
                ) {
                    return Ok(format!(
                        "Workflow '{}' stopped at phase {} ('{}') — you declined to run it.",
                        workflow.id,
                        i + 1,
                        phase.prompt
                    ));
                }
            }

            let prompt = format!(
                "Overall goal: {goal}\n{}\nYour subtask: {}",
                if summaries.is_empty() {
                    String::new()
                } else {
                    format!("Completed so far:\n{}\n", summaries.join("\n"))
                },
                phase.prompt
            );
            let persona_injected = prepend_persona_prompt(&mut self.state, &phase.persona);
            let was_plan = self.plan_mode();
            if phase.read_only {
                self.set_plan_mode(true);
            }
            self.state.messages.push(Message::user(prompt));
            let result = self.drive_turn(&mut on_event, &mut approver).await;
            self.set_plan_mode(was_plan);
            if persona_injected {
                self.state.messages.remove(0);
            }
            let result = result?;
            summaries.push(format!("[{}] {}", phase.persona, result.final_text));
            on_event(AgentEvent::WorkflowPhaseDone {
                name: phase.prompt.clone(),
                persona: phase.persona.clone(),
                summary: result.final_text,
            });
        }

        let final_summary = if summaries.is_empty() {
            format!("Workflow '{}' produced no output for: {goal}", workflow.id)
        } else {
            format!(
                "Workflow '{}' completed for: {goal}\n{}",
                workflow.id,
                summaries.join("\n")
            )
        };
        on_event(AgentEvent::WorkflowDone {
            summary: final_summary.clone(),
        });
        Ok(final_summary)
    }

    /// Compact, human-readable preview of the planned steps, shown in the
    /// approval prompt before anything executes.
    fn render_plan_preview(&self, steps: &[PlanStep]) -> String {
        let mut out = String::from("Proposed plan:\n");
        for step in steps {
            out.push_str(&format!(
                "  {}. {}{}\n",
                step.id,
                step.description,
                step.persona
                    .as_ref()
                    .map(|p| format!("  [{p}]"))
                    .unwrap_or_default()
            ));
            if !step.rationale.is_empty() {
                out.push_str(&format!("     why: {}\n", step.rationale));
            }
        }
        out
    }

    /// One-line preview of a single planned step, including the rationale for
    /// choosing that approach. Used for the per-step accept/deny gate.
    fn step_preview(step: &PlanStep) -> String {
        let mut s = format!("{}. {}", step.id, step.description);
        if let Some(p) = step.persona.as_deref() {
            s.push_str(&format!("  [{p}]"));
        }
        if !step.rationale.is_empty() {
            s.push_str(&format!("\n   why: {}", step.rationale));
        }
        s
    }

    /// One read-only review pass over completed `work`, driven by a
    /// `reviewer: true` persona matched to the goal. Emits a `PlanReviewed`
    /// event with the report. Returns `(report, usage, was_cancelled)` —
    /// `report` is `None` when no reviewer is available *or* the pass was
    /// cancelled (an empty `final_text` is ambiguous between the two, so
    /// callers needing to tell them apart check `was_cancelled` instead of
    /// inferring it from `report.is_none()`).
    async fn reviewer_pass<E, A>(
        &mut self,
        goal: &str,
        work: &str,
        on_event: &mut E,
        approver: &mut A,
    ) -> Result<(Option<String>, TokenUsage, bool)>
    where
        E: FnMut(AgentEvent),
        A: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let Some(persona) = recommend_reviewer(goal) else {
            return Ok((None, TokenUsage::default(), false));
        };
        let injected = prepend_persona_prompt(&mut self.state, persona.id);
        let review_prompt = format!(
            "Review the work produced for this goal, then report concrete findings and \
             any required fixes.\n\nGoal: {goal}\n\nWork produced:\n{work}\n\n\
             Review only — do not edit files. End with a concise verdict."
        );
        self.state.messages.push(Message::user(review_prompt));
        let result = self.drive_turn(&mut *on_event, &mut *approver).await?;
        if injected {
            self.state.messages.remove(0);
        }
        if result.cancelled {
            return Ok((None, result.usage, true));
        }
        if result.final_text.is_empty() {
            return Ok((None, result.usage, false));
        }
        on_event(AgentEvent::PlanReviewed {
            persona: persona.id.to_string(),
            report: result.final_text.clone(),
        });
        Ok((Some(result.final_text), result.usage, false))
    }

    /// Planning pass for an orchestrated run: a tool-free call asking the
    /// model to break the goal into 2-6 ordered subtasks, returned as a JSON
    /// array of strings so it can be parsed deterministically rather than
    /// scraped from prose. Falls back to a single step (the whole goal) if
    /// the response isn't parseable.
    async fn plan_task(&mut self, goal: &str) -> Result<(Vec<PlanStep>, TokenUsage)> {
        let response = self
            .provider
            .chat(ChatRequest {
                model: self.options.model.clone(),
                messages: vec![
                    Message::system(
                        "You are a planning agent. Break the user's goal into 2-6 concrete, \
                         ordered subtasks that a coding agent with file and shell access can \
                         execute one at a time. For each subtask give a short `description` of \
                         the action and a short `rationale` explaining why this approach and \
                         its trade-offs (1-2 sentences). Respond with ONLY a JSON array of \
                         objects, no prose, no markdown fences. Example: \
                         [{\"description\": \"Read package.json\", \"rationale\": \"Confirms the \
                         dependency list before editing.\"}]",
                    ),
                    Message::user(goal),
                ],
                tools: Vec::new(),
                temperature: None,
                max_tokens: Some(1024),
                cancel: Some(self.cancel_rx.clone()),
            })
            .await
            .map_err(AgentError::Provider)?;
        let usage = response.usage.clone();

        let text = response.message.content;
        // Tolerate both the object form `[{description, rationale}]` and the
        // older plain-string form `["read x"]`.
        let parsed = serde_json::from_str::<Vec<serde_json::Value>>(text.trim())
            .ok()
            .filter(|v| !v.is_empty());
        let fallback = || {
            vec![PlanStep {
                id: 1,
                description: goal.to_string(),
                rationale: String::new(),
                persona: recommend_persona(goal).map(|p| p.id.to_string()),
            }]
        };
        let steps = match parsed {
            Some(items) => {
                let steps: Vec<PlanStep> = items
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, v)| {
                        let description = v
                            .get("description")
                            .and_then(|d| d.as_str())
                            .map(str::to_string)
                            .or_else(|| v.as_str().map(str::to_string));
                        let description = description?;
                        let rationale = v
                            .get("rationale")
                            .and_then(|r| r.as_str())
                            .unwrap_or_default()
                            .to_string();
                        Some(PlanStep {
                            id: i + 1,
                            description: description.clone(),
                            rationale,
                            persona: recommend_persona(&description).map(|p| p.id.to_string()),
                        })
                    })
                    .collect();
                if steps.is_empty() {
                    fallback()
                } else {
                    steps
                }
            }
            None => fallback(),
        };
        Ok((steps, usage))
    }

    /// `delegate` tool: lets the model consult a specialist mid-turn for
    /// expert input, instead of every specialist assignment requiring a
    /// plan drawn up before execution starts (see `orchestrate`'s doc
    /// comment for how the two coexist — this doesn't replace it, it's
    /// what a plain `run_turn`/Auto-mode continuous loop reaches for when
    /// it recognizes work matching a specialist's domain).
    ///
    /// Deliberately bounded and read-only: the specialist gets its own
    /// short-lived message list (not the primary conversation) and only
    /// read-only tools (`read`/`grep`/`glob`/etc, regardless of what that
    /// persona's own tool allow-list would otherwise permit in a planned
    /// step) so it can ground its answer in the real codebase without
    /// being able to mutate anything itself — only the primary agent that
    /// called `delegate` writes/edits/runs, acting on the recommendation.
    /// Returns `(result, usage, was_cancelled)` — `was_cancelled` tells the
    /// caller to treat the *whole* turn as cancelled rather than feeding
    /// `result` back to the model as a normal tool result, so a cancel that
    /// lands mid-consultation gets the same clean "(cancelled)" outcome as
    /// every other cancellation point in `drive_turn` instead of either a
    /// hard turn-abort or a fabricated-looking successful answer.
    async fn run_delegate(&self, arguments: &str) -> Result<(ToolResult, TokenUsage, bool)> {
        #[derive(serde::Deserialize)]
        struct DelegateArgs {
            persona: String,
            task: String,
        }
        let args: DelegateArgs = match serde_json::from_str(arguments) {
            Ok(a) => a,
            Err(e) => {
                return Ok((
                    ToolResult::err(format!("bad delegate arguments: {e}")),
                    TokenUsage::default(),
                    false,
                ))
            }
        };
        let Some(persona) = persona_by_id(&args.persona) else {
            let roster: Vec<&str> = personas_by_department()
                .into_iter()
                .flat_map(|(_, ps)| ps.into_iter().map(|p| p.id))
                .collect();
            return Ok((
                ToolResult::err(format!(
                    "unknown specialist id '{}' — see /agents for the roster: {}",
                    args.persona,
                    roster.join(", ")
                )),
                TokenUsage::default(),
                false,
            ));
        };

        let mut messages = vec![
            Message::system(persona.system_prompt()),
            Message::user(args.task),
        ];
        let mut usage = TokenUsage::default();
        const MAX_ITERATIONS: usize = 5;
        for _ in 0..MAX_ITERATIONS {
            if *self.cancel_rx.borrow() {
                return Ok((ToolResult::ok(String::new()), usage, true));
            }
            let request = ChatRequest {
                model: self.options.model.clone(),
                messages: messages.clone(),
                tools: self.tools.read_only_tool_specs(),
                temperature: self.options.temperature,
                max_tokens: self.options.max_tokens,
                cancel: Some(self.cancel_rx.clone()),
            };
            let mut stream = match self.provider.stream(request).await {
                Ok(s) => s,
                Err(zeus_provider::ProviderError::Cancelled) => {
                    return Ok((ToolResult::ok(String::new()), usage, true));
                }
                Err(e) => return Err(AgentError::Provider(e)),
            };
            let mut text = String::new();
            let mut calls: HashMap<String, (Option<String>, String)> = HashMap::new();
            let mut call_order: Vec<String> = Vec::new();
            let mut finish = FinishReason::Stop;
            while let Some(ev) = stream.next().await {
                match ev.map_err(AgentError::Provider)? {
                    StreamEvent::TextDelta { text: t } => text.push_str(&t),
                    StreamEvent::ToolCallDelta {
                        id,
                        name,
                        arguments_delta,
                    } => {
                        let entry = calls.entry(id.clone()).or_insert_with(|| {
                            call_order.push(id.clone());
                            (None, String::new())
                        });
                        if let Some(n) = name {
                            entry.0 = Some(n);
                        }
                        entry.1.push_str(&arguments_delta);
                    }
                    StreamEvent::Done {
                        finish_reason,
                        usage: u,
                    } => {
                        usage.prompt_tokens += u.prompt_tokens;
                        usage.completion_tokens += u.completion_tokens;
                        usage.total_tokens += u.total_tokens;
                        finish = finish_reason;
                    }
                }
            }
            if finish == FinishReason::Cancelled {
                return Ok((ToolResult::ok(text), usage, true));
            }
            if calls.is_empty() {
                let content = format!("[{} ({})]\n{}", persona.role, persona.id, text);
                return Ok((ToolResult::ok(content), usage, false));
            }
            let tool_calls: Vec<ToolCall> = call_order
                .iter()
                .filter_map(|id| {
                    let (name, arguments) = calls.get(id)?;
                    Some(ToolCall {
                        id: id.clone(),
                        name: name.clone().unwrap_or_default(),
                        arguments: arguments.clone(),
                    })
                })
                .collect();
            let mut assistant_msg = Message::assistant(text);
            assistant_msg.tool_calls = tool_calls.clone();
            messages.push(assistant_msg);
            for call in &tool_calls {
                // Every tool exposed here is already filtered to
                // read-only via `read_only_tool_specs`, so there's
                // nothing an "ask" response would meaningfully gate —
                // auto-approve rather than surface a second, confusing
                // permission prompt for a sub-consultation the user
                // didn't directly initiate.
                let result = self.tools.dispatch_with_approver(
                    &call.name,
                    &call.arguments,
                    |_: &PermissionRequest| ApprovalDecision::Approved,
                )?;
                messages.push(Message::tool_result(call.id.clone(), result.content));
            }
        }
        Ok((
            ToolResult::ok(format!(
                "({} consultation hit its step limit without a final answer — try a narrower task)",
                persona.id
            )),
            usage,
            false,
        ))
    }

    /// The tool-calling loop proper: stream the provider, execute any tool
    /// calls (permission-gated via `approver`), feed results back, repeat
    /// until a plain-text final answer or the iteration budget runs out.
    /// Both `run_turn` and `orchestrate` reuse this, so a step in a plan
    /// runs through exactly the same loop as a standalone turn.
    async fn drive_turn<E, A>(&mut self, mut on_event: E, mut approver: A) -> Result<TurnResult>
    where
        E: FnMut(AgentEvent),
        A: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let mut total_tool_calls = 0usize;
        let mut total_usage = TokenUsage::default();

        for _ in 0..self.options.max_tool_iterations {
            if *self.cancel_rx.borrow() {
                on_event(AgentEvent::Cancelled);
                self.persist()?;
                self.tools
                    .hooks()
                    .run_on_stop(&self.state.session_id, "cancelled before model response");
                return Ok(TurnResult {
                    final_text: String::new(),
                    tool_calls: total_tool_calls,
                    cancelled: true,
                    usage: total_usage,
                });
            }

            // Plan mode was advertising the *entire* tool list — every
            // mutating tool included (write/edit/delete/bash/every git
            // write op/...) — even though `ToolManager::dispatch_inner`
            // rejects every one of them in Plan mode anyway. That's pure
            // dead weight on every single Plan-mode request: more than
            // half of Zeus's 60+ built-in tools are mutating, so Plan mode
            // was carrying the same tool-list size/complexity as Build for
            // zero actual capability. Confirmed in practice this alone was
            // enough to make a smaller/free model fail to converge — so
            // Plan mode now only ever sees the tools it could actually use.
            let mut tools = if self.plan_mode() {
                self.tools.read_only_tool_specs()
            } else {
                self.tools.all_tool_specs()
            };
            // `delegate` isn't in `ToolManager` (it needs `self.provider`
            // to run a nested consultation, which the tool dispatcher has
            // no access to) — appended here instead, and intercepted below
            // before reaching `dispatch_with_approver`. Auto-mode only:
            // it exists to replace the old planner's specialist-assignment
            // for the continuous loop, so Build/Plan turns (which never
            // used that) don't need the added size/complexity.
            if self.auto_mode() {
                tools.push(delegate_tool_spec());
            }
            let request = ChatRequest {
                model: self.options.model.clone(),
                messages: self.state.messages.clone(),
                tools,
                temperature: self.options.temperature,
                max_tokens: self.options.max_tokens,
                cancel: Some(self.cancel_rx.clone()),
            };

            let mut stream = self
                .provider
                .stream(request)
                .await
                .map_err(AgentError::Provider)?;

            let mut text = String::new();
            let mut calls: HashMap<String, (Option<String>, String)> = HashMap::new();
            let mut call_order: Vec<String> = Vec::new();
            let mut finish = FinishReason::Stop;

            while let Some(ev) = stream.next().await {
                match ev.map_err(AgentError::Provider)? {
                    StreamEvent::TextDelta { text: t } => {
                        text.push_str(&t);
                        on_event(AgentEvent::TextDelta(t));
                    }
                    StreamEvent::ToolCallDelta {
                        id,
                        name,
                        arguments_delta,
                    } => {
                        let entry = calls.entry(id.clone()).or_insert_with(|| {
                            call_order.push(id.clone());
                            (None, String::new())
                        });
                        if let Some(n) = name {
                            entry.0 = Some(n);
                        }
                        entry.1.push_str(&arguments_delta);
                    }
                    StreamEvent::Done {
                        finish_reason,
                        usage,
                    } => {
                        total_usage.prompt_tokens += usage.prompt_tokens;
                        total_usage.completion_tokens += usage.completion_tokens;
                        total_usage.total_tokens += usage.total_tokens;
                        finish = finish_reason;
                    }
                }
            }

            if finish == FinishReason::Cancelled {
                on_event(AgentEvent::Cancelled);
                self.persist()?;
                self.tools
                    .hooks()
                    .run_on_stop(&self.state.session_id, "cancelled mid-stream");
                return Ok(TurnResult {
                    final_text: text,
                    tool_calls: total_tool_calls,
                    cancelled: true,
                    usage: total_usage,
                });
            }

            if calls.is_empty() {
                // Small local models occasionally emit a bare, meaningless
                // reply (empty, or just stray JSON punctuation like "{}") —
                // observed with a large tool list confusing a 3B model into
                // an aborted function-call-shaped attempt that never became
                // an actual tool call. Rather than silently showing that
                // raw junk with no explanation, append a note so the user
                // knows what happened and what to try instead.
                let trimmed = text.trim();
                let is_degenerate = trimmed.is_empty()
                    || (!trimmed.is_empty()
                        && trimmed
                            .chars()
                            .all(|c| c.is_whitespace() || matches!(c, '{' | '}' | '[' | ']')));
                let mut final_text = text;
                if is_degenerate {
                    let note = "\n\n(that came back empty/malformed instead of a real answer — \
                        small local models sometimes struggle with a large tool list. Try \
                        rephrasing, or switch models with /model.)";
                    on_event(AgentEvent::TextDelta(note.to_string()));
                    final_text.push_str(note);
                }
                self.state
                    .messages
                    .push(Message::assistant(final_text.clone()));
                self.persist()?;
                on_event(AgentEvent::Done);
                self.tools.hooks().run_on_stop(
                    &self.state.session_id,
                    &format!("turn finished: {total_tool_calls} tool call(s)"),
                );
                return Ok(TurnResult {
                    final_text,
                    tool_calls: total_tool_calls,
                    cancelled: false,
                    usage: total_usage,
                });
            }

            // Assistant message carrying the requested tool calls, then one
            // tool-result message per call — appended immediately after, so
            // the pairing invariant `ContextManager` relies on always holds.
            let tool_calls: Vec<ToolCall> = call_order
                .iter()
                .map(|id| {
                    let (name, arguments) = calls.get(id).unwrap();
                    ToolCall {
                        id: id.clone(),
                        name: name.clone().unwrap_or_default(),
                        arguments: arguments.clone(),
                    }
                })
                .collect();
            let mut assistant_msg = Message::assistant(text);
            assistant_msg.tool_calls = tool_calls.clone();
            self.state.messages.push(assistant_msg);

            for call in &tool_calls {
                if *self.cancel_rx.borrow() {
                    on_event(AgentEvent::Cancelled);
                    self.persist()?;
                    self.tools
                        .hooks()
                        .run_on_stop(&self.state.session_id, "cancelled mid-tool-call");
                    return Ok(TurnResult {
                        final_text: String::new(),
                        tool_calls: total_tool_calls,
                        cancelled: true,
                        usage: total_usage,
                    });
                }
                on_event(AgentEvent::ToolCallStarted {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                });
                let result = if call.name == "delegate" {
                    let (result, delegate_usage, was_cancelled) =
                        self.run_delegate(&call.arguments).await?;
                    total_usage.prompt_tokens += delegate_usage.prompt_tokens;
                    total_usage.completion_tokens += delegate_usage.completion_tokens;
                    total_usage.total_tokens += delegate_usage.total_tokens;
                    if was_cancelled {
                        on_event(AgentEvent::Cancelled);
                        self.persist()?;
                        self.tools
                            .hooks()
                            .run_on_stop(&self.state.session_id, "cancelled mid-stream");
                        return Ok(TurnResult {
                            final_text: String::new(),
                            tool_calls: total_tool_calls,
                            cancelled: true,
                            usage: total_usage,
                        });
                    }
                    result
                } else {
                    self.tools
                        .dispatch_with_approver(&call.name, &call.arguments, &mut approver)?
                };
                total_tool_calls += 1;
                on_event(AgentEvent::ToolCallFinished {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    result: result.content.clone(),
                    is_error: result.is_error,
                });
                if call.name == "todowrite" && !result.is_error {
                    if let Some(todos) = parse_todowrite_args(&call.arguments) {
                        on_event(AgentEvent::TodosUpdated { todos });
                    }
                }
                self.state
                    .messages
                    .push(Message::tool_result(call.id.clone(), result.content));
                if !result.images.is_empty() {
                    // Give the model the image bytes on a fresh user message —
                    // every provider mapper (anthropic/openai/ollama) handles
                    // image parts on User messages uniformly.
                    self.state.messages.push(Message::user_with_images(
                        format!(
                            "The tool '{}' produced image content you must inspect visually (attached below).",
                            call.name
                        ),
                        result.images,
                    ));
                }
            }

            self.persist()?;
        }

        // Not a system failure — the model (often a small/local one) just
        // didn't converge to a final answer within the iteration budget,
        // e.g. by repeating the same tool call. This used to be a hard
        // `Err`, which crashed the entire REPL session on a single
        // unlucky turn instead of just failing that turn — fixed to behave
        // like a normal (if apologetic) reply instead, through the same
        // TextDelta/Done event path a real answer would use, so no caller
        // needs special-case handling.
        let fallback_text = format!(
            "(no final answer after {} tool call(s) across {} iterations — the model may be stuck, \
             e.g. repeating the same tool call. Try rephrasing, or breaking the request into \
             smaller steps.)",
            total_tool_calls, self.options.max_tool_iterations
        );
        on_event(AgentEvent::TextDelta(fallback_text.clone()));
        self.state
            .messages
            .push(Message::assistant(fallback_text.clone()));
        self.persist()?;
        on_event(AgentEvent::Done);
        self.tools
            .hooks()
            .run_on_stop(&self.state.session_id, "max tool iterations exceeded");
        Ok(TurnResult {
            final_text: fallback_text,
            tool_calls: total_tool_calls,
            cancelled: false,
            usage: total_usage,
        })
    }

    /// Compact the conversation if it's near the model's context window.
    /// Returns `None` when no compaction was needed.
    async fn maybe_compact(&mut self) -> Result<Option<CompactResult>> {
        self.maybe_compact_inner(false).await
    }

    /// Force a compaction pass right now, bypassing the usual threshold
    /// check — for a user-initiated `/compact` rather than the automatic
    /// per-turn check. Still respects `keep_recent_messages`/the
    /// tool-call-pairing invariant; only the "is it even near the window"
    /// gate is skipped.
    pub async fn compact_now(&mut self) -> Result<CompactResult> {
        Ok(self
            .maybe_compact_inner(true)
            .await?
            .unwrap_or(CompactResult {
                compacted: false,
                removed_messages: 0,
            }))
    }

    async fn maybe_compact_inner(&mut self, force: bool) -> Result<Option<CompactResult>> {
        let count = self
            .provider
            .count_tokens(TokenCountRequest {
                model: self.options.model.clone(),
                messages: self.state.messages.clone(),
                tools: self.tools.all_tool_specs(),
            })
            .await
            .map_err(AgentError::Provider)?;

        if !force && !self.auto_compact() {
            return Ok(None);
        }

        if !force && !self.context.should_compact(count.tokens) {
            return Ok(None);
        }

        let keep_recent = self.context.keep_recent_messages();
        let boundary = self
            .context
            .compaction_boundary(&self.state.messages, keep_recent);
        if boundary == 0 {
            return Ok(None);
        }

        let to_summarize = self.state.messages[..boundary].to_vec();
        let transcript: String = to_summarize
            .iter()
            .map(|m| format!("[{:?}] {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");
        let summary_prompt = format!(
            "Summarize the following earlier conversation concisely, preserving \
             important facts, decisions, and file paths mentioned. This summary \
             will replace the original messages in the model's context.\n\n{transcript}"
        );

        let summary_text = match self
            .provider
            .chat(ChatRequest::new(
                self.options.model.clone(),
                vec![Message::user(summary_prompt)],
            ))
            .await
        {
            Ok(resp) => resp.message.content,
            Err(e) => {
                warn!(?e, "compaction summary call failed; using placeholder");
                format!("[compacted {boundary} earlier message(s); summary unavailable]")
            }
        };

        let mut new_messages = Vec::new();
        let mut i = 0;
        while i < self.state.messages.len() && self.state.messages[i].role == Role::System {
            new_messages.push(self.state.messages[i].clone());
            i += 1;
        }
        new_messages.push(Message::system(format!(
            "[Earlier conversation summary]\n\
             The earlier turns were removed to keep the context small. The original \
             tool outputs they were based on are GONE — treat this summary as a claim, \
             NOT as evidence. Re-run the relevant tool (read/grep/search/glob) before \
             asserting any file content, path, line number, or symbol that is not \
             explicitly quoted here, and expect this summary to be incomplete or wrong.\n\n\
             {summary_text}"
        )));
        new_messages.extend_from_slice(&self.state.messages[boundary..]);

        debug!(removed = boundary, "compacted conversation context");
        self.state.messages = new_messages;

        Ok(Some(CompactResult {
            compacted: true,
            removed_messages: boundary,
        }))
    }

    fn persist(&self) -> Result<()> {
        self.sessions.save(&self.state)
    }
}

/// Prepend a specialist-agent system prompt to a conversation for one step.
/// Returns true if the persona was resolved and prepended (the caller then
/// removes index 0 after the turn). Unknown ids are a no-op so a stale plan
/// can't break execution.
fn prepend_persona_prompt(state: &mut super::session::ConversationState, id: &str) -> bool {
    let Some(persona) = persona_by_id(id) else {
        return false;
    };
    state
        .messages
        .insert(0, Message::system(persona.system_prompt()));
    true
}

/// Whether a planned step is safe to run as a headless concurrent turn —
/// i.e. its persona only inspects the workspace (no write/edit/bash tools)
/// and never mutates shared state. Steps with no persona are treated as
/// potentially mutating and stay sequential.
fn is_read_only_step(step: &PlanStep) -> bool {
    step.persona
        .as_deref()
        .and_then(persona_by_id)
        .map(Persona::read_only)
        .unwrap_or(false)
}

/// Build the per-step prompt for an orchestrated run: the overall goal plus
/// (optionally) a snapshot of what earlier steps completed.
fn orchestration_step_prompt(goal: &str, summaries: &[String], step: &PlanStep) -> String {
    format!(
        "Overall goal: {goal}\n{}Now do this specific subtask: {}",
        if summaries.is_empty() {
            String::new()
        } else {
            format!("Completed so far:\n{}\n", summaries.join("\n"))
        },
        step.description
    )
}

/// One line for the final combined summary.
fn step_summary(step: &PlanStep, text: &str) -> String {
    format!(
        "{}. {} — {}",
        step.id,
        step.description,
        text.chars().take(200).collect::<String>()
    )
}

/// Run one planned step as a *headless* provider call (no tools, no shared
/// conversation mutation) and return its text. Used for concurrent read-only
/// steps: unconnected cloned provider/cancel handle in, finished text out,
/// nothing about `self` shared or mutated.
/// Model knobs shared by headless plan steps.
fn add_usage(total: &mut TokenUsage, delta: &TokenUsage) {
    total.prompt_tokens += delta.prompt_tokens;
    total.completion_tokens += delta.completion_tokens;
    total.total_tokens += delta.total_tokens;
}

struct HeadlessSpec {
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

/// Returns `(text, was_cancelled, usage)`. A cancellation caught before the
/// stream even starts (the provider's own upfront check) and one caught
/// mid-stream (`FinishReason::Cancelled`) both surface through the same
/// `was_cancelled` flag rather than one being an `Err` and the other an
/// `Ok` — callers only need to check one thing.
async fn run_headless_step(
    provider: std::sync::Arc<dyn ModelProvider>,
    spec: HeadlessSpec,
    step: &PlanStep,
    goal: &str,
    snapshot: &str,
    cancel: tokio::sync::watch::Receiver<bool>,
) -> Result<(String, bool, TokenUsage)> {
    let system = match step.persona.as_deref().and_then(persona_by_id) {
        Some(p) => p.system_prompt(),
        None => "You are a careful, read-only analysis agent.".to_string(),
    };
    let user = format!(
        "Overall goal: {goal}\n{}Now do this specific subtask: {}",
        if snapshot.is_empty() {
            String::new()
        } else {
            format!("Completed so far:\n{}\n", snapshot)
        },
        step.description
    );

    let request = ChatRequest {
        model: spec.model,
        messages: vec![Message::system(system), Message::user(user)],
        tools: Vec::new(),
        temperature: spec.temperature,
        max_tokens: spec.max_tokens.or(Some(512)),
        cancel: Some(cancel),
    };
    let mut stream = match provider.stream(request).await {
        Ok(s) => s,
        Err(zeus_provider::ProviderError::Cancelled) => {
            return Ok((String::new(), true, TokenUsage::default()));
        }
        Err(e) => return Err(AgentError::Provider(e)),
    };
    let mut text = String::new();
    let mut cancelled = false;
    let mut usage = TokenUsage::default();
    while let Some(ev) = stream.next().await {
        match ev.map_err(AgentError::Provider)? {
            StreamEvent::TextDelta { text: t } => text.push_str(&t),
            StreamEvent::ToolCallDelta { .. } => {}
            StreamEvent::Done {
                finish_reason,
                usage: u,
            } => {
                if finish_reason == FinishReason::Cancelled {
                    cancelled = true;
                }
                usage.prompt_tokens += u.prompt_tokens;
                usage.completion_tokens += u.completion_tokens;
                usage.total_tokens += u.total_tokens;
            }
        }
    }
    Ok((text, cancelled, usage))
}

/// The `delegate` tool's spec — built fresh (not a `const`) since the
/// roster of specialist ids in its description comes from `ALL_PERSONAS`
/// at runtime rather than being hand-copied and left to drift out of sync.
fn delegate_tool_spec() -> ToolSpec {
    // Department names only (not all ~40 individual specialist ids) — the
    // full roster used to be embedded here on every single request, which
    // meaningfully bloats an already-large tool list and measurably hurt
    // smaller/free models' tool-calling reliability in practice (they'd
    // return empty/malformed responses or never converge — the exact
    // failure `drive_turn` already has a fallback message for). The model
    // can call `/agents`-equivalent info via the roster departments here
    // and pick a specific id from context/its own knowledge; getting an
    // unknown id back is handled gracefully (`run_delegate` reports it).
    let departments: Vec<&str> = personas_by_department()
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    ToolSpec {
        name: "delegate".to_string(),
        description: format!(
            "Consult a specialist for expert input on part of the current task — use this \
             when a piece of work clearly calls for specific expertise instead of guessing \
             yourself. Read-only: the specialist can inspect the codebase but not write/edit/run \
             anything; it returns a recommendation for you to act on. Not for delegating whole \
             tasks wholesale. Departments: {}. Give a persona id matching the relevant \
             department (e.g. \"security-engineer\", \"backend-engineer\") — an unrecognized id \
             is reported back rather than failing silently.",
            departments.join(", ")
        ),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "persona": {"type": "string", "description": "specialist id, e.g. \"security-engineer\""},
                "task": {"type": "string", "description": "the specific question or subtask for them to address"}
            },
            "required": ["persona", "task"]
        }),
    }
}

/// Parses a `todowrite` call's raw arguments (already schema-validated by
/// `ToolManager::do_todowrite`) back into `TodoStatus` rows for the UI.
/// Returns `None` on anything unexpected rather than erroring the turn —
/// this is a best-effort UI update, not something that should fail the
/// tool call itself (which already succeeded by the time this runs).
fn parse_todowrite_args(arguments: &str) -> Option<Vec<TodoStatus>> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    let todos = value.get("todos")?.as_array()?;
    Some(
        todos
            .iter()
            .filter_map(|t| {
                let content = t.get("content")?.as_str()?.to_string();
                let status = t.get("status")?.as_str()?.to_string();
                Some(TodoStatus { content, status })
            })
            .collect(),
    )
}

fn split_orientation_docs(text: &str) -> (Option<String>, Option<String>) {
    fn extract(text: &str, open: &str, close: &str) -> Option<String> {
        let start = text.find(open)? + open.len();
        let end = text[start..].find(close)?;
        Some(text[start..start + end].trim().to_string())
    }
    (
        extract(text, "[ARCH]", "[/ARCH]"),
        extract(text, "[CONV]", "[/CONV]"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{ConversationState, SessionStore};
    use crate::{BackgroundTaskRegistry, ContextManager, HookRunner, TerminalRunner, ToolManager};
    use std::collections::VecDeque;
    use std::path::Path;
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;
    use zeus_config::{AgentSettings, Config, GlobalPaths, ProvidersFile};
    use zeus_fs::{ApprovalDecision, PermissionRequest, Workspace};
    use zeus_provider::{
        ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, FinishReason, Message,
        ModelInfo, StreamEvent, TokenCountRequest, TokenCountResponse, TokenUsage,
    };

    #[test]
    fn orientation_docs_split() {
        let text = "preamble\n[ARCH]\n# Architecture\nthe map\n[/ARCH]\nand\n[CONV]\n## Conventions\nstyle\n[/CONV]\n";
        let (arch, conv) = split_orientation_docs(text);
        assert_eq!(arch.as_deref(), Some("# Architecture\nthe map"));
        assert_eq!(conv.as_deref(), Some("## Conventions\nstyle"));
    }

    #[test]
    fn orientation_docs_missing_markers() {
        let (arch, conv) = split_orientation_docs("no markers here");
        assert!(arch.is_none());
        assert!(conv.is_none());
    }

    /// One scripted reply for the mock provider. Each `stream()` call pops the
    /// next entry — modelling one agent loop iteration (text-only final answer,
    /// or a round of tool calls to execute before the next iteration).
    #[derive(Debug, Clone)]
    enum MockReply {
        /// A plain text-only assistant reply (finish reason Stop). This is what
        /// a mock "final answer" looks like.
        Text(String),
        /// One or more tool calls the model issues this iteration; the agent
        /// executes them, feeds results back, then streams the next reply.
        ToolCalls(Vec<(String, String, String)>),
    }

    /// Deterministic, in-memory `ModelProvider` for turn-level tests: a script
    /// of recorded replies. Never touches a network or disk model.
    #[derive(Debug, Clone)]
    struct MockProvider {
        script: Arc<Mutex<VecDeque<MockReply>>>,
    }

    impl MockProvider {
        fn new(replies: Vec<MockReply>) -> Self {
            Self {
                script: Arc::new(Mutex::new(VecDeque::from(replies))),
            }
        }

        fn next(&self) -> MockReply {
            self.script
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(MockReply::Text("(mock stream exhausted)".into()))
        }
    }

    #[async_trait::async_trait]
    impl zeus_provider::ModelProvider for MockProvider {
        fn supports_prompt_cache(&self) -> bool {
            false
        }
        fn id(&self) -> &str {
            "mock"
        }

        async fn chat(&self, _request: ChatRequest) -> zeus_provider::Result<ChatResponse> {
            let reply = self.next();
            match reply {
                MockReply::Text(t) => Ok(ChatResponse {
                    message: Message::assistant(t),
                    usage: TokenUsage::new(1, 1),
                    finish_reason: FinishReason::Stop,
                    model: "mock".into(),
                }),
                MockReply::ToolCalls(calls) => {
                    let tool_calls: Vec<zeus_provider::ToolCall> = calls
                        .into_iter()
                        .map(|(id, name, arguments)| zeus_provider::ToolCall {
                            id,
                            name,
                            arguments,
                        })
                        .collect();
                    let mut message = Message::assistant("");
                    message.tool_calls = tool_calls;
                    Ok(ChatResponse {
                        message,
                        usage: TokenUsage::new(1, 1),
                        finish_reason: FinishReason::ToolCalls,
                        model: "mock".into(),
                    })
                }
            }
        }

        async fn stream(
            &self,
            _request: ChatRequest,
        ) -> zeus_provider::Result<zeus_provider::ChatStream> {
            let reply = self.next();
            let mut events: Vec<StreamEvent> = Vec::new();
            match &reply {
                MockReply::Text(t) => {
                    events.push(StreamEvent::TextDelta { text: t.clone() });
                    events.push(StreamEvent::Done {
                        finish_reason: FinishReason::Stop,
                        usage: TokenUsage::new(1, 1),
                    });
                }
                MockReply::ToolCalls(calls) => {
                    for (id, name, args) in calls {
                        events.push(StreamEvent::ToolCallDelta {
                            id: id.clone(),
                            name: Some(name.clone()),
                            arguments_delta: args.clone(),
                        });
                    }
                    events.push(StreamEvent::Done {
                        finish_reason: FinishReason::ToolCalls,
                        usage: TokenUsage::new(1, 1),
                    });
                }
            }
            let stream = futures::stream::iter(events.into_iter().map(Ok));
            Ok(Box::pin(stream))
        }

        async fn list_models(&self) -> zeus_provider::Result<Vec<ModelInfo>> {
            Ok(vec![ModelInfo {
                id: "mock".into(),
                name: "Mock Model".into(),
                context_window: Some(128_000),
            }])
        }

        async fn embeddings(
            &self,
            _request: EmbeddingRequest,
        ) -> zeus_provider::Result<EmbeddingResponse> {
            Ok(EmbeddingResponse {
                vectors: Vec::new(),
                usage: TokenUsage::new(0, 0),
            })
        }

        async fn count_tokens(
            &self,
            _request: TokenCountRequest,
        ) -> zeus_provider::Result<TokenCountResponse> {
            Ok(TokenCountResponse {
                tokens: 1,
                approximate: true,
            })
        }
    }

    fn approve(_: &PermissionRequest) -> ApprovalDecision {
        ApprovalDecision::Approved
    }

    /// Build a ToolManager rooted at a temp project dir (same shape as the
    /// `zeus-fs`/tools test helpers), so `read`/`grep`/`git_status` all run
    /// against an isolated on-disk tree.
    fn tool_manager(root: &Path) -> ToolManager {
        std::fs::create_dir_all(root).unwrap();
        let config = Config {
            global: GlobalPaths::from_root(root.join(".zeus-home")),
            project: None,
            settings: AgentSettings::default(),
            providers: ProvidersFile::default(),
            project_root: Some(root.to_path_buf()),
        };
        let workspace = Workspace::from_config(&config).unwrap();
        let terminal = TerminalRunner::new(root.join(".agent/checkpoints"));
        let background = BackgroundTaskRegistry::new(root.join(".agent/background"));
        let hooks = HookRunner::new(root.join(".agent/hooks"), root.to_path_buf());
        let mut tools = ToolManager::new(
            workspace,
            terminal,
            background,
            hooks,
            Vec::new(), // no MCP
            Vec::new(), // no plugins
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
        );
        tools.set_global_skills_dir(None);
        tools
    }

    fn test_agent(root: &Path, provider: MockProvider) -> Agent {
        Agent::new(
            Arc::new(provider),
            tool_manager(root),
            ContextManager::new(128_000, 0.8, 6),
            SessionStore::new(root.join(".sessions")),
            ConversationState::new("test-session"),
            super::AgentOptions {
                model: "mock".into(),
                max_tool_iterations: 8,
                temperature: None,
                max_tokens: Some(1024),
                max_parallel_read_steps: 2,
                tasks_file: None,
            },
        )
    }

    #[tokio::test]
    async fn run_turn_text_only_finalizes() {
        let tmp = TempDir::new().unwrap();
        let mut agent = test_agent(
            tmp.path(),
            MockProvider::new(vec![MockReply::Text("hello there".into())]),
        );
        let mut events: Vec<AgentEvent> = Vec::new();
        let result = agent
            .run_turn("hi", |ev| events.push(ev), approve)
            .await
            .unwrap();

        assert_eq!(result.final_text, "hello there");
        assert_eq!(result.tool_calls, 0);
        assert!(!result.cancelled);
        assert!(events.iter().any(|e| matches!(e, AgentEvent::Done)));
    }

    #[tokio::test]
    async fn run_turn_read_tool_then_final_answer() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("a.txt"), "content 42").unwrap();
        let mut agent = test_agent(
            root,
            MockProvider::new(vec![
                MockReply::ToolCalls(vec![(
                    "call-read".into(),
                    "read".into(),
                    r#"{"path":"a.txt"}"#.into(),
                )]),
                MockReply::Text("file says content 42".into()),
            ]),
        );
        let mut events: Vec<AgentEvent> = Vec::new();
        let result = agent
            .run_turn("read a.txt", |ev| events.push(ev), approve)
            .await
            .unwrap();

        assert_eq!(result.tool_calls, 1);
        assert_eq!(result.final_text, "file says content 42");
        assert!(!result.cancelled);
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolCallStarted { name, .. } if name == "read")));
        let read_result = events
            .iter()
            .find_map(|e| match e {
                AgentEvent::ToolCallFinished { name, result, .. } if name == "read" => {
                    Some(result.clone())
                }
                _ => None,
            })
            .expect("read tool result event");
        assert!(read_result.contains("content 42"));
    }

    #[tokio::test]
    async fn review_turn_forces_plan_mode_and_emits_report() {
        let tmp = TempDir::new().unwrap();
        let mut agent = test_agent(
            tmp.path(),
            MockProvider::new(vec![MockReply::Text("verdict: APPROVE".into())]),
        );
        let mut events: Vec<AgentEvent> = Vec::new();
        let before = agent.plan_mode();
        let result = agent
            .review_turn(|ev| events.push(ev), approve)
            .await
            .unwrap();

        assert!(!result.cancelled);
        assert_eq!(result.final_text, "verdict: APPROVE");
        // Plan mode was forced on for the read-only review pass, then restored.
        assert!(!before);
        assert_eq!(agent.plan_mode(), before);
        let review = events.iter().find_map(|e| match e {
            AgentEvent::ReviewUncommitted { persona, report } => {
                Some((persona.clone(), report.clone()))
            }
            _ => None,
        });
        let (persona, report) = review.expect("ReviewUncommitted event emitted");
        assert!(!persona.is_empty());
        assert_eq!(report, "verdict: APPROVE");
    }

    #[tokio::test]
    async fn suggest_turn_emits_features() {
        let tmp = TempDir::new().unwrap();
        let mut agent = test_agent(
            tmp.path(),
            MockProvider::new(vec![MockReply::Text("1. auth (M)\n2. observer (S)".into())]),
        );
        let mut events: Vec<AgentEvent> = Vec::new();
        let result = agent
            .suggest_turn(|ev| events.push(ev), approve)
            .await
            .unwrap();

        assert!(!result.cancelled);
        assert_eq!(result.final_text, "1. auth (M)\n2. observer (S)");
        assert!(events.iter().any(
            |e| matches!(e, AgentEvent::FeaturesSuggested { report } if report.contains("auth"))
        ));
    }

    /// Approve everything except the lead-reviewer gate, so we can exercise
    /// the reject path of `orchestrate` end to end.
    fn approve_but_deny_review(req: &PermissionRequest) -> ApprovalDecision {
        if req.tool == "review_accept" {
            ApprovalDecision::Denied
        } else {
            ApprovalDecision::Approved
        }
    }

    /// One-step orchestration script: the planning call is `chat()` and pops
    /// the plan JSON; the single step and the review pass each stream one
    /// plain-text answer.
    fn orchestration_script(plan_json: &str, step_text: &str, review_text: &str) -> Vec<MockReply> {
        vec![
            MockReply::Text(plan_json.to_string()),
            MockReply::Text(step_text.to_string()),
            MockReply::Text(review_text.to_string()),
        ]
    }

    #[tokio::test]
    async fn orchestrate_rejects_work_when_lead_reviewer_denied() {
        let tmp = TempDir::new().unwrap();
        let mut agent = test_agent(
            tmp.path(),
            MockProvider::new(orchestration_script(
                r#"[{"description":"add a test","rationale":"proves behavior"}]"#,
                "step one done",
                "found a bug: the null case is unhandled",
            )),
        );
        let mut events: Vec<AgentEvent> = Vec::new();
        let (summary, _usage) = agent
            .orchestrate("add a test", |ev| events.push(ev), approve_but_deny_review)
            .await
            .unwrap();

        assert!(summary.contains("NOT accepted"), "{summary}");
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::OrchestrationRevision { report } if report.contains("NOT accepted"))));
        assert!(!events
            .iter()
            .any(|e| matches!(e, AgentEvent::OrchestrationDone { .. })));
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::PlanReviewed { persona, .. } if !persona.is_empty())));
    }

    #[tokio::test]
    async fn orchestrate_completes_when_lead_reviewer_accepted() {
        let tmp = TempDir::new().unwrap();
        let mut agent = test_agent(
            tmp.path(),
            MockProvider::new(orchestration_script(
                r#"[{"description":"add a test","rationale":"proves behavior"}]"#,
                "step one done",
                "verdict: LGTM",
            )),
        );
        let mut events: Vec<AgentEvent> = Vec::new();
        let (summary, _usage) = agent
            .orchestrate("add a test", |ev| events.push(ev), approve)
            .await
            .unwrap();

        assert!(summary.contains("Review:"), "{summary}");
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::OrchestrationDone { .. })));
        assert!(!events
            .iter()
            .any(|e| matches!(e, AgentEvent::OrchestrationRevision { .. })));
    }

    #[tokio::test]
    async fn orchestrate_persists_per_step_progress_to_tasks_json() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let mut agent = test_agent(
            root,
            MockProvider::new(orchestration_script(
                r#"[{"description":"first","rationale":"one"},{"description":"second","rationale":"two"},{"description":"third","rationale":"three"}]"#,
                "did the first thing",
                "verdict: LGTM",
            )),
        );
        // With `max_parallel_read_steps = 1` every step runs sequentially, so
        // the persisted file is written once per completed step and the final
        // state reflects the exact run order.
        agent.options.max_parallel_read_steps = 1;
        let tasks_file = root.join(".agent/tasks.json");
        agent.options.tasks_file = Some(tasks_file.clone());

        let (summary, _usage) = agent
            .orchestrate("do the thing", |_ev| {}, approve)
            .await
            .unwrap();
        assert!(summary.contains("3 steps"), "{summary}");

        let plan = crate::plans::TaskPlan::read(&tasks_file)
            .unwrap()
            .expect("tasks.json written");
        assert!(plan.approved);
        assert_eq!(plan.completed(), 3, "all steps done at finish");
        assert!(plan.steps.iter().all(|s| s.done));
    }

    #[tokio::test]
    async fn orchestrate_diffs_against_a_prior_plan_in_the_approval_preview() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let tasks_file = root.join(".agent/tasks.json");
        let script = orchestration_script(
            r#"[{"description":"new approach","rationale":"revised"},{"description":"second","rationale":"two"}]"#,
            "done",
            "verdict: LGTM",
        );

        // A plan from a prior run already exists on disk.
        let prior = crate::plans::TaskPlan::from_steps(
            "do the thing",
            &[crate::agent::PlanStep {
                id: 1,
                description: "old approach".into(),
                rationale: "first".into(),
                persona: None,
            }],
            "",
            false,
        );
        prior.write(&tasks_file).unwrap();

        let mut agent = test_agent(root, MockProvider::new(script));
        agent.options.tasks_file = Some(tasks_file.clone());
        let previews: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let previews2 = previews.clone();
        let (summary, _usage) = agent
            .orchestrate(
                "do the thing",
                |_ev| {},
                move |req: &PermissionRequest| {
                    if let Some(p) = &req.preview {
                        previews2.lock().unwrap().push(p.clone());
                    }
                    ApprovalDecision::Approved
                },
            )
            .await
            .unwrap();
        assert!(summary.contains("2 steps"), "{summary}");

        // The plan_execute gate's preview mentions the change from the old
        // step list and carries the new step text.
        let joined = previews.lock().unwrap().join("\n");
        assert!(joined.contains("plan changed vs"), "{joined}");
        assert!(joined.contains("new approach"), "{joined}");
        assert!(joined.contains("old approach"), "{joined}");
    }

    #[tokio::test]
    async fn plan_turn_dispatches_roster_personas_to_steps() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("db.rs"), "// migration helper").unwrap();
        let mut agent = test_agent(
            root,
            MockProvider::new(vec![
                // plan_task() uses chat(): pop the structured step list.
                MockReply::Text(
                    r#"[{"description":"write a database migration","rationale":"adds schema"},{"description":"run the tests","rationale":"verify the change"}]"#
                        .to_string(),
                ),
                // The research pass (stream) then answers.
                MockReply::Text("Add the migration, then run tests. No files touched.".to_string()),
            ]),
        );
        let mut events: Vec<AgentEvent> = Vec::new();
        agent
            .plan_turn("add a database migration", |ev| events.push(ev), approve)
            .await
            .unwrap();

        let steps = events
            .iter()
            .find_map(|e| match e {
                AgentEvent::PlanGenerated { steps } => Some(steps.clone()),
                _ => None,
            })
            .expect("PlanGenerated event emitted");
        assert_eq!(steps.len(), 2);
        // Each step is dispatched to the specialist best matching it:
        // "database migration" → database-engineer; "run the tests" → a QA
        // reviewer-capable persona (or, failing a keyword match, it stays
        // None — the roster is advisory, never blocking).
        assert_eq!(steps[0].persona.as_deref(), Some("database-engineer"));
    }

    #[tokio::test]
    async fn run_workflow_steps_through_persona_phases() {
        let tmp = TempDir::new().unwrap();
        let wf = crate::workflows::Workflow {
            id: "ship".into(),
            description: "A two-phase pipeline".into(),
            phases: vec![
                crate::workflows::WorkflowPhaseDef {
                    persona: "backend-engineer".into(),
                    prompt: "Implement the change".into(),
                    gate: false,
                    read_only: false,
                },
                crate::workflows::WorkflowPhaseDef {
                    persona: "qa-engineer".into(),
                    prompt: "Verify the change".into(),
                    gate: false,
                    read_only: false,
                },
            ],
            origin: tmp.path().join("ship.toml"),
        };
        let mut agent = test_agent(
            tmp.path(),
            MockProvider::new(vec![
                MockReply::Text("implemented".to_string()),
                MockReply::Text("tests pass".to_string()),
            ]),
        );
        let mut events: Vec<AgentEvent> = Vec::new();
        let summary = agent
            .run_workflow("ship a change", &wf, |ev| events.push(ev), approve)
            .await
            .unwrap();

        assert!(summary.contains("implemented"), "{summary}");
        assert!(summary.contains("tests pass"), "{summary}");
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::WorkflowStarted { id, .. } if id == "ship")));
        let started: Vec<&AgentEvent> = events
            .iter()
            .filter(|e| matches!(e, AgentEvent::WorkflowPhaseStarted { .. }))
            .collect();
        assert_eq!(started.len(), 2);
        assert!(events.iter().any(|e| matches!(
            e,
            AgentEvent::WorkflowPhaseDone { persona, summary, .. }
                if persona == "qa-engineer" && summary == "tests pass"
        )));
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::WorkflowDone { .. })));
    }
}
