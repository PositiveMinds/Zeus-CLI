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
use crate::personas::{persona_by_id, recommend_persona, recommend_reviewer, Persona};
use crate::session::{ConversationState, SessionStore};
use crate::tools::ToolManager;
use futures::StreamExt;
use std::collections::HashMap;
use tokio::sync::watch;
use tracing::{debug, warn};
use zeus_fs::{ApprovalDecision, PermissionRequest};
use zeus_provider::{
    ChatRequest, FinishReason, Message, ModelProvider, Role, StreamEvent, TokenCountRequest,
    ToolCall,
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
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            model: "mock-model".into(),
            max_tool_iterations: 8,
            temperature: None,
            max_tokens: None,
            max_parallel_read_steps: 2,
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
    /// Orchestrated `/plan` runs: the planning pass produced an ordered
    /// list of subtasks to execute.
    PlanGenerated { steps: Vec<PlanStep> },
    /// A subtask from the plan is about to be executed as its own turn.
    PlanStepStarted { step: PlanStep },
    /// A subtask finished and produced a final answer.
    PlanStepDone { step: PlanStep, summary: String },
    /// A review pass over completed plan work ran; `persona` is the reviewer
    /// id that drove it and `report` is its findings.
    PlanReviewed { persona: String, report: String },
    /// All subtasks completed; `summary` is the combined result.
    OrchestrationDone { summary: String },
}

/// One subtask in an orchestrated `/plan` run.
#[derive(Debug, Clone)]
pub struct PlanStep {
    pub id: usize,
    pub description: String,
    /// Optional specialist-agent id (from `MVP_PERSONAS`) to steer this step;
    /// `None` means run it with the generic coding agent.
    pub persona: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TurnResult {
    pub final_text: String,
    pub tool_calls: usize,
    pub cancelled: bool,
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

    /// Models the current provider actually has available — backs a
    /// `/model` picker UI (list + select) rather than requiring the user to
    /// already know an exact model name to type.
    pub async fn list_models(&self) -> Result<Vec<zeus_provider::ModelInfo>> {
        self.provider.list_models().await.map_err(AgentError::Provider)
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
        self.auto_mode.store(enabled, std::sync::atomic::Ordering::Relaxed);
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
        self.state.messages.push(Message::user(user_message));

        if let Some(result) = self.maybe_compact().await? {
            on_event(AgentEvent::Compacted(result));
        }

        // Auto mode: plan the request, then execute each planned step.
        if self.auto_mode() {
            let summary = self.orchestrate(user_message, on_event, approver).await?;
            return Ok(TurnResult {
                final_text: summary,
                ..TurnResult::default()
            });
        }

        self.drive_turn(on_event, approver).await
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
    ) -> Result<String>
    where
        E: FnMut(AgentEvent),
        A: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let _ = self.cancel_tx.send(false);

        let steps = self.plan_task(goal).await?;
        on_event(AgentEvent::PlanGenerated { steps: steps.clone() });

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
                prior_content = result.final_text.clone();
                summaries.push(step_summary(&step, &result.final_text));
                on_event(AgentEvent::PlanStepDone {
                    step,
                    summary: result.final_text,
                });
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
                                opt_model,
                                opt_temperature,
                                opt_max_tokens,
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

                for (step, res) in steps_slice[start..end].iter().zip(results) {
                    on_event(AgentEvent::PlanStepStarted { step: step.clone() });
                    match res {
                        Ok(final_text) => {
                            prior_content = final_text.clone();
                            summaries.push(step_summary(step, &final_text));
                            on_event(AgentEvent::PlanStepDone {
                                step: step.clone(),
                                summary: final_text,
                            });
                        }
                        Err(e) => {
                            summaries.push(step_summary(step, &format!("(step failed: {e})")));
                            on_event(AgentEvent::PlanStepDone {
                                step: step.clone(),
                                summary: format!("(step failed: {e})"),
                            });
                        }
                    }
                }
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
        let review_report = self.reviewer_pass(goal, &final_summary, &mut on_event, &mut approver).await?;

        let final_summary = if let Some(report) = review_report {
            format!("{final_summary}\n\nReview:\n{report}")
        } else {
            final_summary
        };
        on_event(AgentEvent::OrchestrationDone {
            summary: final_summary.clone(),
        });
        Ok(final_summary)
    }

    /// One read-only review pass over completed `work`, driven by a
    /// `reviewer: true` persona matched to the goal. Emits a `PlanReviewed`
    /// event with the report. Returns the report text, or `None` when no
    /// reviewer is available.
    async fn reviewer_pass<E, A>(
        &mut self,
        goal: &str,
        work: &str,
        on_event: &mut E,
        approver: &mut A,
    ) -> Result<Option<String>>
    where
        E: FnMut(AgentEvent),
        A: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let Some(persona) = recommend_reviewer(goal) else {
            return Ok(None);
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
        if result.final_text.is_empty() {
            return Ok(None);
        }
        on_event(AgentEvent::PlanReviewed {
            persona: persona.id.to_string(),
            report: result.final_text.clone(),
        });
        Ok(Some(result.final_text))
    }

    /// Planning pass for an orchestrated run: a tool-free call asking the
    /// model to break the goal into 2-6 ordered subtasks, returned as a JSON
    /// array of strings so it can be parsed deterministically rather than
    /// scraped from prose. Falls back to a single step (the whole goal) if
    /// the response isn't parseable.
    async fn plan_task(&mut self, goal: &str) -> Result<Vec<PlanStep>> {
        let response = self
            .provider
            .chat(ChatRequest {
                model: self.options.model.clone(),
                messages: vec![
                    Message::system(
                        "You are a planning agent. Break the user's goal into 2-6 concrete, \
                         ordered subtasks that a coding agent with file and shell access can \
                         execute one at a time. Respond with ONLY a JSON array of strings, \
                         no prose, no markdown fences. Example: \
                         [\"Read package.json\", \"Add the missing dependency\"]",
                    ),
                    Message::user(goal),
                ],
                tools: Vec::new(),
                temperature: None,
                max_tokens: Some(512),
                cancel: Some(self.cancel_rx.clone()),
            })
            .await
            .map_err(AgentError::Provider)?;

        let text = response.message.content;
        let parsed = serde_json::from_str::<Vec<String>>(text.trim())
            .ok()
            .filter(|v| !v.is_empty());
        match parsed {
            Some(steps) => Ok(steps
                .into_iter()
                .enumerate()
                .map(|(i, description)| PlanStep {
                    id: i + 1,
                    description: description.clone(),
                    persona: recommend_persona(&description).map(|p| p.id.to_string()),
                })
                .collect()),
            None => Ok(vec![PlanStep {
                id: 1,
                description: goal.to_string(),
                persona: recommend_persona(goal).map(|p| p.id.to_string()),
            }]),
        }
    }

    /// The tool-calling loop proper: stream the provider, execute any tool
    /// calls (permission-gated via `approver`), feed results back, repeat
    /// until a plain-text final answer or the iteration budget runs out.
    /// Both `run_turn` and `orchestrate` reuse this, so a step in a plan
    /// runs through exactly the same loop as a standalone turn.
    async fn drive_turn<E, A>(
        &mut self,
        mut on_event: E,
        mut approver: A,
    ) -> Result<TurnResult>
    where
        E: FnMut(AgentEvent),
        A: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let mut total_tool_calls = 0usize;

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
                });
            }

            let request = ChatRequest {
                model: self.options.model.clone(),
                messages: self.state.messages.clone(),
                tools: self.tools.all_tool_specs(),
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
                    StreamEvent::Done { finish_reason, .. } => {
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
                        && trimmed.chars().all(|c| c.is_whitespace() || matches!(c, '{' | '}' | '[' | ']')));
                let mut final_text = text;
                if is_degenerate {
                    let note = "\n\n(that came back empty/malformed instead of a real answer — \
                        small local models sometimes struggle with a large tool list. Try \
                        rephrasing, or switch models with /model.)";
                    on_event(AgentEvent::TextDelta(note.to_string()));
                    final_text.push_str(note);
                }
                self.state.messages.push(Message::assistant(final_text.clone()));
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
                    });
                }
                on_event(AgentEvent::ToolCallStarted {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                });
                let result =
                    self.tools
                        .dispatch_with_approver(&call.name, &call.arguments, &mut approver)?;
                total_tool_calls += 1;
                on_event(AgentEvent::ToolCallFinished {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    result: result.content.clone(),
                    is_error: result.is_error,
                });
                self.state
                    .messages
                    .push(Message::tool_result(call.id.clone(), result.content));
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
        self.state.messages.push(Message::assistant(fallback_text.clone()));
        self.persist()?;
        on_event(AgentEvent::Done);
        self.tools
            .hooks()
            .run_on_stop(&self.state.session_id, "max tool iterations exceeded");
        Ok(TurnResult {
            final_text: fallback_text,
            tool_calls: total_tool_calls,
            cancelled: false,
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
            .chat(ChatRequest::new(self.options.model.clone(), vec![Message::user(summary_prompt)]))
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
            "[Earlier conversation summary]\n{summary_text}"
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal::TerminalRunner;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use tempfile::TempDir;
    use zeus_config::{AgentSettings, Config, GlobalPaths, ProvidersFile};
    use zeus_fs::Workspace;
    use zeus_provider::MockProvider;

    fn approve(_: &PermissionRequest) -> ApprovalDecision {
        ApprovalDecision::Approved
    }

    fn make_agent(root: &std::path::Path, provider: MockProvider) -> Agent {
        make_agent_with_options(root, provider, AgentOptions::default())
    }

    fn make_agent_with_options(
        root: &std::path::Path,
        provider: MockProvider,
        options: AgentOptions,
    ) -> Agent {
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
        let background = crate::background::BackgroundTaskRegistry::new(root.join(".agent/background"));
        let hooks = crate::hooks::HookRunner::new(root.join(".agent/hooks"), root.to_path_buf());
        let tools = ToolManager::new(
            workspace,
            terminal,
            background,
            hooks,
            Vec::new(),
            Vec::new(),
            Arc::new(AtomicBool::new(false)),
        );
        let context = ContextManager::new(128_000, 0.8, 6);
        let sessions = SessionStore::new(root.join(".agent/sessions"));
        let state = ConversationState::new("test-session");
        Agent::new(Arc::new(provider), tools, context, sessions, state, options)
    }

    #[tokio::test]
    async fn orchestrator_plans_then_runs_each_step() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        // The planning `chat` call returns a JSON array of two subtasks,
        // which `plan_task` parses; each step then runs through `drive_turn`
        // (streaming) and gets the same reply text echoed back.
        let provider = MockProvider::new("mock")
            .with_reply(r#"["inspect config", "run the build"]"#);
        let mut agent = make_agent(&root, provider);

        let mut events = Vec::new();
        let summary = agent
            .orchestrate("ship the feature", |ev| events.push(ev), approve)
            .await
            .unwrap();

        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::PlanGenerated { steps } if steps.len() == 2)));
        assert!(events
            .iter()
            .filter(|e| matches!(e, AgentEvent::PlanStepStarted { .. }))
            .count()
            == 2);
        assert!(events
            .iter()
            .filter(|e| matches!(e, AgentEvent::PlanStepDone { .. }))
            .count()
            == 2);
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::OrchestrationDone { .. })));
        // Two steps -> the combined multi-line summary is returned.
        assert!(summary.contains("Completed 2 steps"));
    }

    #[tokio::test]
    async fn orchestrator_falls_back_to_single_step_when_plan_unparseable() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        // A non-JSON plan reply makes the planner fall back to a single step.
        let provider = MockProvider::new("mock").with_reply("just do the whole thing");
        let mut agent = make_agent(&root, provider);

        let mut events = Vec::new();
        let _ = agent
            .orchestrate("do everything", |ev| events.push(ev), approve)
            .await
            .unwrap();

        let plan = events
            .iter()
            .find_map(|e| match e {
                AgentEvent::PlanGenerated { steps } => Some(steps.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].description, "do everything");
    }

    #[tokio::test]
    async fn orchestrator_runs_a_reviewer_pass_after_steps() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        // Every provider call (planning, each step, review) returns this same
        // reply, so the whole pipeline runs without needing a real model.
        let provider = MockProvider::new("mock")
            .with_reply(r#"["inspect config", "run the build"]"#);
        let mut agent = make_agent(&root, provider);

        let mut events = Vec::new();
        let summary = agent
            .orchestrate("ship the feature", |ev| events.push(ev), approve)
            .await
            .unwrap();

        // A reviewer persona should have been dispatched and reported back.
        let reviews: Vec<&String> = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::PlanReviewed { report, .. } => Some(report),
                _ => None,
            })
            .collect();
        assert_eq!(reviews.len(), 1, "expected exactly one review pass");
        assert!(summary.contains("Review:"));
    }

    #[tokio::test]
    async fn read_only_steps_run_concurrently_via_headless_turns() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        // Planning returns two research-y subtasks, each mapping to a
        // read-only persona (literature review / statistician) so both run
        // through the headless parallel path. The single shared reply is
        // echoed by every provider call, including both parallel turns.
        let provider = MockProvider::new("mock")
            .with_reply(r#"["review the literature", "run a statistical analysis"]"#);
        let mut agent = make_agent_with_options(
            &root,
            provider,
            AgentOptions {
                max_parallel_read_steps: 2,
                ..AgentOptions::default()
            },
        );

        let mut events = Vec::new();
        let summary = agent
            .orchestrate("produce a research report", |ev| events.push(ev), approve)
            .await
            .unwrap();

        // Both read-only steps should have started and finished.
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, AgentEvent::PlanStepStarted { .. }))
                .count(),
            2
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, AgentEvent::PlanStepDone { .. }))
                .count(),
            2
        );
        assert!(summary.contains("Completed 2 steps"));
    }

    #[tokio::test]
    async fn plain_text_turn_returns_final_answer_with_no_tool_calls() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let mut agent = make_agent(&root, MockProvider::new("mock").with_reply("hello there"));

        let mut events = Vec::new();
        let result = agent
            .run_turn("hi", |ev| events.push(format!("{ev:?}")), approve)
            .await
            .unwrap();

        assert_eq!(result.tool_calls, 0);
        assert!(!result.cancelled);
        assert!(result.final_text.contains("hello there"));
        assert_eq!(agent.messages().len(), 2); // user + assistant
    }

    #[tokio::test]
    async fn degenerate_bare_json_reply_gets_an_explanatory_note() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let mut agent = make_agent(&root, MockProvider::new("mock").with_reply("{}"));

        let result = agent.run_turn("hello", |_| {}, approve).await.unwrap();

        assert!(!result.cancelled);
        assert!(result.final_text.contains("{}"));
        assert!(result.final_text.contains("small local models"));
    }

    #[tokio::test]
    async fn normal_reply_is_untouched_even_if_it_contains_braces() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let mut agent =
            make_agent(&root, MockProvider::new("mock").with_reply("here's an object: {\"a\":1}"));

        let result = agent.run_turn("hello", |_| {}, approve).await.unwrap();

        assert_eq!(result.final_text, "here's an object: {\"a\":1}");
    }

    #[tokio::test]
    async fn tool_call_is_dispatched_and_result_fed_back() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        // First call requests a `write`; MockProvider consumes the script
        // and the *next* call (after the tool result is appended) falls back
        // to its normal echo reply, letting the loop terminate.
        let provider = MockProvider::new("mock").with_tool_call(
            "call-1",
            "write",
            r#"{"path":"out.txt","content":"from the agent"}"#,
        );
        let mut agent = make_agent(&root, provider);

        let mut tool_finished = None;
        let result = agent
            .run_turn(
                "please write a file",
                |ev| {
                    if let AgentEvent::ToolCallFinished { name, is_error, .. } = &ev {
                        tool_finished = Some((name.clone(), *is_error));
                    }
                },
                approve,
            )
            .await
            .unwrap();

        assert_eq!(result.tool_calls, 1);
        assert!(!result.cancelled);
        assert_eq!(tool_finished, Some(("write".to_string(), false)));
        assert_eq!(
            std::fs::read_to_string(root.join("out.txt")).unwrap(),
            "from the agent"
        );
        // user, assistant(tool_call), tool(result), assistant(final text)
        assert_eq!(agent.messages().len(), 4);
    }

    #[tokio::test]
    async fn max_iterations_exhausted_returns_ok_with_apology_not_err() {
        // Regression test: a model that never stops calling tools (observed
        // in practice with a small local model repeating git_status for
        // unrelated input) used to make `run_turn` return a hard `Err` once
        // `max_tool_iterations` was hit, which propagated all the way up and
        // crashed the whole interactive REPL over a single turn. It should
        // instead behave like a normal (if apologetic) reply.
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let provider =
            MockProvider::new("mock").with_repeating_tool_call("call-1", "git_status", "{}");
        let mut agent = make_agent_with_options(
            &root,
            provider,
            AgentOptions {
                model: "mock-model".into(),
                max_tool_iterations: 2,
                ..AgentOptions::default()
            },
        );

        let mut events = Vec::new();
        let result = agent
            .run_turn("hello", |ev| events.push(format!("{ev:?}")), approve)
            .await
            .expect("must not error out just because the model never converged");

        assert!(!result.cancelled);
        assert_eq!(result.tool_calls, 2);
        assert!(!result.final_text.is_empty());
        // The apology text must have actually reached the caller's event
        // sink (as a normal TextDelta+Done), not just the returned struct —
        // that's what lets the REPL render it exactly like a real reply.
        assert!(events.iter().any(|e| e.contains(&result.final_text)));
        assert!(events.iter().any(|e| e == "Done"));
    }

    #[tokio::test]
    async fn cancelled_turn_is_reported_and_persisted() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let mut agent = make_agent(
            &root,
            MockProvider::new("mock")
                .with_reply("alpha beta gamma delta epsilon")
                .with_chunk_delay_ms(20),
        );
        let cancel = agent.cancel_handle();

        let run = async {
            agent
                .run_turn("hi", |_| {}, approve)
                .await
        };
        tokio::pin!(run);
        // Cancel shortly after starting so the stream sees it mid-flight.
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
            let _ = cancel.send(true);
        });
        let result = run.await.unwrap();
        assert!(result.cancelled);
    }

    #[tokio::test]
    async fn compaction_triggers_when_over_threshold() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        // Tiny window + low threshold so a couple of turns push it over.
        let mut agent = make_agent(&root, MockProvider::new("mock").with_reply("ok"));
        agent.context = ContextManager::new(50, 0.1, 1);

        for i in 0..5 {
            let _ = agent
                .run_turn(&format!("message number {i} with some extra padding text"), |_| {}, approve)
                .await
                .unwrap();
        }
        // After several turns over a tiny window, a synthetic summary system
        // message should have been inserted.
        assert!(agent
            .messages()
            .iter()
            .any(|m| m.role == Role::System && m.content.contains("Earlier conversation summary")));
    }

    #[tokio::test]
    async fn model_can_be_switched_mid_session() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let mut agent = make_agent(&root, MockProvider::new("mock").with_reply("ok"));
        assert_eq!(agent.model(), "mock-model");
        agent.set_model("a-different-model");
        assert_eq!(agent.model(), "a-different-model");

        // A subsequent turn should actually use the switched model — proven
        // by it still working end to end, not just that the getter changed.
        let result = agent.run_turn("hi", |_| {}, approve).await.unwrap();
        assert!(!result.cancelled);
    }

    #[tokio::test]
    async fn context_usage_reflects_added_messages() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let mut agent = make_agent(&root, MockProvider::new("mock").with_reply("ok"));
        let before = agent.context_usage().await.unwrap();
        assert_eq!(before.message_count, 0);
        assert_eq!(before.window, 128_000);

        agent.run_turn("hello there", |_| {}, approve).await.unwrap();
        let after = agent.context_usage().await.unwrap();
        assert_eq!(after.message_count, 2); // user + assistant
        assert!(after.tokens > 0);
    }

    #[tokio::test]
    async fn compact_now_forces_compaction_even_under_threshold() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("proj");
        let mut agent = make_agent(&root, MockProvider::new("mock").with_reply("ok"));
        // Plenty of room under the default 128k window — the automatic
        // per-turn check would never trigger compaction here. A small
        // keep_recent so there's still something *to* compact once forced
        // (compaction is a no-op if there aren't more messages than
        // keep_recent, force or not — force only bypasses the threshold
        // check, not that boundary condition).
        agent.context = ContextManager::new(128_000, 0.8, 1);
        agent.run_turn("message one", |_| {}, approve).await.unwrap();
        agent.run_turn("message two", |_| {}, approve).await.unwrap();
        let before_count = agent.messages().len();
        assert!(before_count > 2);

        let result = agent.compact_now().await.unwrap();
        assert!(result.compacted);
        assert!(agent
            .messages()
            .iter()
            .any(|m| m.role == Role::System && m.content.contains("Earlier conversation summary")));
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
    state.messages.insert(0, Message::system(persona.system_prompt()));
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
async fn run_headless_step(
    provider: std::sync::Arc<dyn ModelProvider>,
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    step: &PlanStep,
    goal: &str,
    snapshot: &str,
    cancel: tokio::sync::watch::Receiver<bool>,
) -> Result<String> {
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
        model,
        messages: vec![Message::system(system), Message::user(user)],
        tools: Vec::new(),
        temperature,
        max_tokens: max_tokens.or(Some(512)),
        cancel: Some(cancel),
    };
    let mut stream = provider.stream(request).await.map_err(AgentError::Provider)?;
    let mut text = String::new();
    while let Some(ev) = stream.next().await {
        match ev.map_err(AgentError::Provider)? {
            StreamEvent::TextDelta { text: t } => text.push_str(&t),
            StreamEvent::ToolCallDelta { .. } => {}
            StreamEvent::Done { .. } => {}
        }
    }
    Ok(text)
}
