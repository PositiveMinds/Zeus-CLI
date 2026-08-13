//! Persisted plan artifacts — the structured plan behind Plan mode, saved to
//! `<project>/.agent/tasks.json`.
//!
//! The blueprint calls for a structured, reviewable artifact that survives
//! the session: `TaskPlan` is written at plan time (`approved: false`),
//! flipped to `true` when the user approves execution, and each step's `done`
//! flag is flipped as the orchestrator completes it — so a human watching
//! `.agent/tasks.json` mid-run sees live per-step progress. Nothing in zeus
//! reads the file back for its own decisions yet; it exists as the durable
//! record a human reviews before and during execution.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::agent::PlanStep;
use crate::error::{AgentError, Result};

/// One planned subtask.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStep {
    pub id: usize,
    pub description: String,
    /// Optional specialist-agent id (from `MVP_PERSONAS`) to steer this step.
    pub persona: Option<String>,
    /// Filled by the orchestrator once the step has been executed.
    #[serde(default)]
    pub done: bool,
}

/// The persisted plan document (`.agent/tasks.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPlan {
    pub goal: String,
    /// Whether the user approved the plan for execution. Written `false` by
    /// `plan_turn`/`orchestrate` and only set to `true` at the approval gate.
    #[serde(default)]
    pub approved: bool,
    #[serde(default)]
    pub steps: Vec<TaskStep>,
    /// The model's research write-up of the approach; swapped for the final
    /// orchestration summary once the steps have run.
    #[serde(default)]
    pub notes: String,
}

impl TaskPlan {
    /// Build a plan from the orchestrator's structured `PlanStep`s, all
    /// pending.
    pub fn from_steps(goal: &str, steps: &[PlanStep], notes: &str, approved: bool) -> Self {
        Self {
            goal: goal.to_string(),
            approved,
            steps: steps
                .iter()
                .map(|s| TaskStep {
                    id: s.id,
                    description: s.description.clone(),
                    persona: s.persona.clone(),
                    done: false,
                })
                .collect(),
            notes: notes.to_string(),
        }
    }

    /// Load a previously persisted plan. `Ok(None)` when the file doesn't
    /// exist yet.
    pub fn read(path: &Path) -> Result<Option<TaskPlan>> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(AgentError::Io(e)),
        };
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| AgentError::Io(io_err(e)))
    }

    /// Pretty-print the plan to `path`, creating parent dirs as needed.
    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(AgentError::Io)?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| AgentError::Io(io_err(e)))?;
        std::fs::write(path, format!("{json}\n")).map_err(AgentError::Io)?;
        Ok(())
    }

    /// Mark every step complete (used once the orchestrator has run them all).
    pub fn mark_all_done(&mut self) {
        for step in &mut self.steps {
            step.done = true;
        }
    }

    /// Mark the step with the given id complete. Unknown ids are ignored —
    /// declined steps never run, so they stay pending.
    pub fn mark_done(&mut self, id: usize) {
        for step in &mut self.steps {
            if step.id == id {
                step.done = true;
            }
        }
    }

    /// How many steps have been completed.
    pub fn completed(&self) -> usize {
        self.steps.iter().filter(|s| s.done).count()
    }

    /// A stable, line-oriented rendering of the plan for diffing: one line per
    /// step, prefixed by its state (`[x]` done / `[ ]` pending), so a plan
    /// diff reads as "what the steps now say".
    pub fn render_lines(&self) -> String {
        let mut out = format!("goal: {}\n", self.goal);
        out.push_str(&format!(
            "approved: {}\n",
            if self.approved { "yes" } else { "no" }
        ));
        for step in &self.steps {
            let state = if step.done { "[x]" } else { "[ ]" };
            out.push_str(&format!(
                "{state} {}. {}{}\n",
                step.id,
                step.description,
                step.persona
                    .as_ref()
                    .map(|p| format!("  [{p}]"))
                    .unwrap_or_default()
            ));
        }
        out
    }

    /// Human-readable diff between this plan and a previously persisted one
    /// (e.g. the plan on disk from a prior `/plan` run). Shows only what
    /// changed in the step list; `None` when `other` is absent.
    pub fn diff_vs(&self, other: &TaskPlan) -> Option<String> {
        let diff = zeus_fs::preview_diff(&other.render_lines(), &self.render_lines());
        if diff == "(no line-level changes)" {
            return None;
        }
        Some(diff)
    }
}

fn io_err(e: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_plan_round_trips_via_disk() {
        let steps = vec![
            PlanStep {
                id: 1,
                description: "read the manifest".into(),
                rationale: "need to see what's declared".into(),
                persona: Some("qa".into()),
            },
            PlanStep {
                id: 2,
                description: "add the missing dep".into(),
                rationale: "build fails without it".into(),
                persona: None,
            },
        ];
        let plan = TaskPlan::from_steps("ship the feature", &steps, "research notes", false);
        let dir = std::env::temp_dir().join(format!("zeus-plan-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tasks.json");

        plan.write(&path).unwrap();
        let loaded = TaskPlan::read(&path).unwrap().unwrap();
        assert_eq!(loaded.goal, "ship the feature");
        assert!(!loaded.approved);
        assert_eq!(loaded.steps.len(), 2);
        assert_eq!(loaded.steps[0].description, "read the manifest");
        assert_eq!(loaded.steps[1].persona, None);
        assert_eq!(loaded.completed(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_missing_plan_is_none() {
        let missing = std::path::Path::new("/nonexistent/zeus/plan/tasks.json");
        assert!(TaskPlan::read(missing).unwrap().is_none());
    }

    #[test]
    fn mark_all_done_sets_every_step() {
        let steps = vec![
            PlanStep {
                id: 1,
                description: "a".into(),
                rationale: "one".into(),
                persona: None,
            },
            PlanStep {
                id: 2,
                description: "b".into(),
                rationale: "two".into(),
                persona: None,
            },
        ];
        let mut plan = TaskPlan::from_steps("g", &steps, "", false);
        assert_eq!(plan.completed(), 0);
        plan.mark_all_done();
        assert_eq!(plan.completed(), 2);
        assert!(plan.steps.iter().all(|s| s.done));
    }

    #[test]
    fn mark_done_flips_only_the_named_step() {
        let steps = vec![
            PlanStep {
                id: 1,
                description: "a".into(),
                rationale: "one".into(),
                persona: None,
            },
            PlanStep {
                id: 2,
                description: "b".into(),
                rationale: "two".into(),
                persona: None,
            },
            PlanStep {
                id: 3,
                description: "c".into(),
                rationale: "three".into(),
                persona: None,
            },
        ];
        let mut plan = TaskPlan::from_steps("g", &steps, "", false);
        plan.mark_done(2);
        assert_eq!(plan.completed(), 1);
        assert!(!plan.steps[0].done);
        assert!(plan.steps[1].done);
        assert!(!plan.steps[2].done);
        // unknown id is a no-op
        plan.mark_done(99);
        assert_eq!(plan.completed(), 1);
    }

    #[test]
    fn render_lines_is_stable_and_state_aware() {
        let steps = vec![
            PlanStep {
                id: 1,
                description: "read the manifest".into(),
                rationale: "need to see what's declared".into(),
                persona: Some("qa".into()),
            },
            PlanStep {
                id: 2,
                description: "add the missing dep".into(),
                rationale: "build fails without it".into(),
                persona: None,
            },
        ];
        let mut plan = TaskPlan::from_steps("ship it", &steps, "", false);
        let before = plan.render_lines();
        assert!(before.contains("goal: ship it"));
        assert!(before.contains("[ ] 1. read the manifest  [qa]"));
        assert!(before.contains("approved: no"));

        plan.mark_done(1);
        let after = plan.render_lines();
        assert!(after.contains("[x] 1. read the manifest  [qa]"));
        assert!(before != after);
    }

    #[test]
    fn diff_vs_reports_only_changed_steps() {
        let mk = |goal: &str, descs: &[&str]| {
            let steps: Vec<PlanStep> = descs
                .iter()
                .enumerate()
                .map(|(i, d)| PlanStep {
                    id: i + 1,
                    description: d.to_string(),
                    rationale: "r".into(),
                    persona: None,
                })
                .collect();
            TaskPlan::from_steps(goal, &steps, "", false)
        };
        // Same plan → no diff.
        let old = mk("g", &["a", "b"]);
        let new = mk("g", &["a", "b"]);
        assert!(new.diff_vs(&old).is_none());

        // One step changed → diff mentions it.
        let new2 = mk("g", &["a", "B CHANGED"]);
        let diff = new2.diff_vs(&old).expect("diff present");
        assert!(diff.contains("B CHANGED"), "{diff}");
    }
}
