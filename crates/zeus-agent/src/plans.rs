//! Persisted plan artifacts — the structured plan behind Plan mode, saved to
//! `<project>/.agent/tasks.json`.
//!
//! The blueprint calls for a structured, reviewable artifact that survives
//! the session: `TaskPlan` is written at plan time (`approved: false`),
//! flipped to `true` when the user approves execution, and updated again when
//! every step completes. Nothing in zeus reads it back yet — it exists as the
//! durable record a human reviews before hitting "execute".

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

    /// How many steps have been completed.
    pub fn completed(&self) -> usize {
        self.steps.iter().filter(|s| s.done).count()
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
}
