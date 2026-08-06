//! Session persistence: conversation state saved under `~/.zeus/sessions/`,
//! independent of the filesystem checkpoint mechanism in zeus-fs (that one
//! restores *files*; this one restores the *conversation*).

use crate::error::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use zeus_provider::Message;

/// One turn's worth of transcript, for history browsing (not required for
/// resume — `ConversationState` alone is enough to continue a session).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEntry {
    pub turn_id: String,
    pub timestamp: String,
    pub messages: Vec<Message>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConversationState {
    pub session_id: String,
    pub messages: Vec<Message>,
}

impl ConversationState {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            messages: Vec::new(),
        }
    }
}

/// Loads/saves `ConversationState` as `<sessions_dir>/<id>.json`.
pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    fn path(&self, session_id: &str) -> PathBuf {
        self.dir.join(format!("{session_id}.json"))
    }

    pub fn load(&self, session_id: &str) -> Result<ConversationState> {
        let path = self.path(session_id);
        if !path.exists() {
            return Ok(ConversationState::new(session_id));
        }
        let text = std::fs::read_to_string(&path)?;
        serde_json::from_str(&text).map_err(|e| AgentError::Session(e.to_string()))
    }

    pub fn save(&self, state: &ConversationState) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let text = serde_json::to_string_pretty(state)
            .map_err(|e| AgentError::Session(e.to_string()))?;
        std::fs::write(self.path(&state.session_id), text)?;
        Ok(())
    }

    pub fn list_session_ids(&self) -> Result<Vec<String>> {
        if !self.dir.exists() {
            return Ok(Vec::new());
        }
        let mut ids = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            if let Some(stem) = entry.path().file_stem() {
                if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
                    ids.push(stem.to_string_lossy().into_owned());
                }
            }
        }
        ids.sort();
        Ok(ids)
    }
}

/// A fresh, timestamp-derived session id — not `Date.now()`-style
/// nondeterminism-sensitive since this is a real running CLI, not a replayed
/// script.
pub fn new_session_id() -> String {
    format!("session-{}", chrono::Utc::now().format("%Y%m%d%H%M%S%3f"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use zeus_provider::Message;

    #[test]
    fn load_missing_session_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf());
        let state = store.load("does-not-exist").unwrap();
        assert!(state.messages.is_empty());
    }

    #[test]
    fn save_then_load_roundtrips() {
        let tmp = TempDir::new().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf());
        let mut state = ConversationState::new("s1");
        state.messages.push(Message::user("hello"));
        state.messages.push(Message::assistant("hi there"));
        store.save(&state).unwrap();

        let loaded = store.load("s1").unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.messages[0].content, "hello");
    }

    #[test]
    fn list_session_ids_finds_saved_sessions() {
        let tmp = TempDir::new().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf());
        store.save(&ConversationState::new("a")).unwrap();
        store.save(&ConversationState::new("b")).unwrap();
        let mut ids = store.list_session_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
    }
}
