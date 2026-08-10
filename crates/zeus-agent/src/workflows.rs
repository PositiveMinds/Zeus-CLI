//! Declarative multi-specialist workflows.
//!
//! A workflow is a named pipeline of ordered specialist phases — the
//! "workforce" differentiator. Each phase names a persona (from the roster, or
//! a custom persona shadowing one) plus the task that specialist should
//! perform in that stage. `/workflow <name> <goal>` runs a workflow from the
//! start, gate to finish, exactly like an orchestrated auto-mode `/plan` run:
//! each phase is one full tool-using turn driven by that phase's persona.
//!
//! Formats served:
//!   - `<project>/.agent/workflows/<name>.toml`   (project, highest priority)
//!   - `~/.zeus/workflows/<name>.toml`             (user/global)
//!
//! A phase can enable `gate` to force an explicit approve before it runs, or
//! `read_only` to constrain it to non-mutating tools regardless of agent mode.
//! Unknown keys are tolerated so a workflow authored for a newer zeus still
//! loads.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing::debug;

/// One phase in a workflow: a specialist role plus the task for its turn.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkflowPhaseDef {
    /// Roster persona id to drive this phase (e.g. `backend-engineer`).
    pub persona: String,
    /// What this worker should do in its turn (the subtask text).
    pub prompt: String,
    /// Require explicit user approval before this phase runs.
    #[serde(default)]
    pub gate: bool,
    /// Force plan mode (read-only tools only) for this phase.
    #[serde(default)]
    pub read_only: bool,
}

/// TOML mirror of a whole workflow file.
#[derive(Debug, Clone, Deserialize)]
struct WorkflowFile {
    name: Option<String>,
    description: Option<String>,
    #[serde(default, rename = "phase")]
    phases: Vec<WorkflowPhaseDef>,
}

/// A loaded, named workflow ready to run.
#[derive(Debug, Clone)]
pub struct Workflow {
    /// Lowercase id (from `name:` or the file stem).
    pub id: String,
    pub description: String,
    pub phases: Vec<WorkflowPhaseDef>,
    /// Directory the TOML came from (project or global), used only for debug
    /// output.
    #[allow(dead_code)]
    pub origin: PathBuf,
}

/// Parse a single workflow TOML file. `origin` is used for the display name
/// fallback. Returns `None` when unreadable or unparseable.
pub fn parse_workflow(path: &Path) -> Option<Workflow> {
    let raw = std::fs::read_to_string(path).ok()?;
    let file: WorkflowFile = match toml::from_str(&raw) {
        Ok(f) => f,
        Err(e) => {
            debug!(path = %path.display(), "workflow TOML is invalid: {e}");
            return None;
        }
    };
    let fallback = path.file_stem().and_then(|s| s.to_str()).unwrap_or("workflow");
    let id = file
        .name
        .unwrap_or_else(|| fallback.to_string())
        .trim()
        .to_lowercase();
    if id.is_empty() || id.contains([' ', '\\']) || id.contains('/') {
        return None;
    }
    Some(Workflow {
        id,
        description: file.description.unwrap_or_default().trim().to_string(),
        phases: file.phases,
        origin: path.to_path_buf(),
    })
}

/// Scan one directory for `*.toml` workflow files, parsing each best-effort.
pub fn discover(dir: &Path) -> Vec<Workflow> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("toml") {
            if let Some(w) = parse_workflow(&path) {
                out.push(w);
            }
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// Every workflow visible to a project: project `.agent/workflows/*.toml`
/// first (shadowing same-id globals), then global `~/.zeus/workflows/*.toml`.
pub fn discover_all(project_dir: Option<&Path>, global_dir: &Path) -> Vec<Workflow> {
    let mut by_id: HashMap<String, Workflow> = HashMap::new();
    for w in discover(&global_dir.join("workflows")) {
        by_id.entry(w.id.clone()).or_insert(w);
    }
    if let Some(project) = project_dir {
        for w in discover(&project.join(".agent").join("workflows")) {
            by_id.insert(w.id.clone(), w);
        }
    }
    let mut all: Vec<Workflow> = by_id.into_values().collect();
    all.sort_by(|a, b| a.id.cmp(&b.id));
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn parses_a_workflow_with_phases() {
        let tmp = TempDir::new().unwrap();
        let p = write(
            tmp.path(),
            "build.toml",
            r#"
name = "build-backend"
description = "Design, implement, then verify a backend feature"

[[phase]]
persona = "architect"
prompt = "Design the schema"
read_only = true

[[phase]]
persona = "backend-engineer"
prompt = "Implement the schema"
gate = true
"#,
        );
        let w = parse_workflow(&p).unwrap();
        assert_eq!(w.id, "build-backend");
        assert_eq!(w.description, "Design, implement, then verify a backend feature");
        assert_eq!(w.phases.len(), 2);
        assert_eq!(w.phases[0].persona, "architect");
        assert_eq!(w.phases[0].read_only, true);
        assert_eq!(w.phases[0].gate, false);
        assert_eq!(w.phases[1].gate, true);
    }

    #[test]
    fn id_falls_back_to_file_stem() {
        let tmp = TempDir::new().unwrap();
        let p = write(tmp.path(), "release.toml", r#"[[phase]]
persona = "qa-engineer"
prompt = "run the tests""#);
        let w = parse_workflow(&p).unwrap();
        assert_eq!(w.id, "release");
    }

    #[test]
    fn project_workflows_shadow_globals() {
        let tmp = TempDir::new().unwrap();
        let global = tmp.path().join(".zeus");
        let project = tmp.path().join("proj");
        write(
            &global.join("workflows"),
            "ship.toml",
            r#"name = "ship"
description = "global version"
[[phase]]
persona = "qa-engineer"
prompt = "gated global""#,
        );
        write(
            &project.join(".agent/workflows"),
            "ship.toml",
            r#"name = "ship"
description = "project version"
[[phase]]
persona = "backend-engineer"
prompt = "project phase""#,
        );
        let all = discover_all(Some(&project), &global);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].description, "project version");
    }

    #[test]
    fn malformed_workflow_is_skipped() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("workflows");
        write(&dir, "bad.toml", "name = [broken");
        write(&dir, "good.toml", r#"name = "ok"
[[phase]]
persona = "backend"
prompt = "hello""#);
        let found = discover(&dir);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "ok");
    }
}