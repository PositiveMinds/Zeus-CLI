//! Workspace helper: bind project root, settings, gates, engines.

use crate::checkpoint::CheckpointStore;
use crate::ops::FileEngine;
use crate::permission::PermissionGate;
use crate::search::SearchEngine;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use zeus_config::{AgentSettings, Config};

/// Fully wired project workspace for Phase 2 tools.
pub struct Workspace {
    pub project_root: PathBuf,
    pub settings: AgentSettings,
    pub files: FileEngine,
    pub search: SearchEngine,
}

impl Workspace {
    /// Build from loaded config. Requires a project root.
    pub fn from_config(config: &Config) -> Result<Self, String> {
        let project_root = config.project_root.clone().ok_or_else(|| {
            "no project root detected; run inside a project or pass --project-root".to_string()
        })?;

        let checkpoints_dir = config
            .project
            .as_ref()
            .map(|p| p.checkpoints.clone())
            .unwrap_or_else(|| project_root.join(".agent/checkpoints"));

        let turn_id = new_turn_id();
        let store = CheckpointStore::new(checkpoints_dir);
        let _ = store.begin_turn(&turn_id);

        let gate = PermissionGate::new(config.settings.clone(), project_root.clone());
        let gate_search = PermissionGate::new(config.settings.clone(), project_root.clone());

        let extra: Vec<PathBuf> = config
            .settings
            .project_roots
            .iter()
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
            .collect();

        let files = FileEngine::new(project_root.clone(), gate, store, turn_id);
        let search = SearchEngine::new(project_root.clone(), gate_search, extra);

        Ok(Self {
            project_root,
            settings: config.settings.clone(),
            files,
            search,
        })
    }
}

fn new_turn_id() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("turn-{secs}")
}
