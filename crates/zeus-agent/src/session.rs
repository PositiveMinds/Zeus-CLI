//! Session persistence: conversation state saved under `~/.zeus/sessions/`,
//! independent of the filesystem checkpoint mechanism in zeus-fs (that one
//! restores *files*; this one restores the *conversation*).

use crate::error::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use zeus_provider::{Message, Role};

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
    /// Unix millis of the last save; used for session-recency sorting.
    #[serde(default)]
    pub last_activity: i64,
}

impl ConversationState {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            messages: Vec::new(),
            last_activity: 0,
        }
    }
}

/// A browsable, one-line description of a saved session.
#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub id: String,
    pub message_count: usize,
    /// The last user message text, truncated for a one-line preview.
    pub last_user: String,
    /// Unix seconds of the file's last write (recency for resumes).
    pub modified: i64,
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
        match serde_json::from_str(&text) {
            Ok(state) => Ok(state),
            // A torn/corrupt file (crash mid-write, manual edit) must not
            // brick the session: fall back to a fresh state for the id
            // rather than hard-failing the launch.
            Err(_) => Ok(ConversationState::new(session_id)),
        }
    }

    pub fn save(&self, state: &ConversationState) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let mut state = state.clone();
        state.last_activity = unix_millis();
        let text =
            serde_json::to_string_pretty(&state).map_err(|e| AgentError::Session(e.to_string()))?;
        let target = self.path(&state.session_id);
        // Crash-safe: write to a temp file in the same directory, then rename
        // over the target. A crash mid-write leaves the previous file intact,
        // and the rename is atomic on the same filesystem.
        let tmp = target.with_extension("json.tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &target)?;
        Ok(())
    }

    /// `(id, unix-seconds-of-last-write)` for every saved session, most
    /// recent first. Cheap: only stat + read the id, not the full state.
    pub fn list_session_ids(&self) -> Result<Vec<String>> {
        Ok(self.summaries()?.into_iter().map(|s| s.id).collect())
    }

    /// One-line summary of every saved session, most recently used first.
    pub fn summaries(&self) -> Result<Vec<SessionSummary>> {
        if !self.dir.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_string);
            let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
                continue;
            };
            if ext.as_deref() != Some("json") {
                continue;
            }
            let modified = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            let mut message_count = 0;
            let mut last_user = String::new();
            let mut activity = modified;
            if let Ok(text) = std::fs::read_to_string(entry.path()) {
                if let Ok(state) = serde_json::from_str::<ConversationState>(&text) {
                    message_count = state.messages.len();
                    if state.last_activity > 0 {
                        activity = state.last_activity;
                    }
                    last_user = state
                        .messages
                        .iter()
                        .rev()
                        .find(|m| m.role == Role::User)
                        .map(|m| m.content.clone())
                        .unwrap_or_default();
                }
            }
            let mut last_user: Vec<char> = last_user.chars().take(90).collect();
            if last_user.len() == 90 {
                last_user.extend("…".chars());
            }
            let last_user: String = last_user.into_iter().collect();
            out.push(SessionSummary {
                id: stem,
                message_count,
                last_user,
                modified: activity,
            });
        }
        out.sort_by(|a, b| b.modified.cmp(&a.modified).then_with(|| a.id.cmp(&b.id)));
        Ok(out)
    }

    /// The most recently used saved session, if any.
    pub fn most_recent(&self) -> Result<Option<String>> {
        Ok(self.summaries()?.into_iter().next().map(|s| s.id))
    }
}

/// A fresh, timestamp-derived session id — not `Date.now()`-style
/// nondeterminism-sensitive since this is a real running CLI, not a replayed
/// script.
pub fn new_session_id() -> String {
    format!("session-{}", chrono::Utc::now().format("%Y%m%d%H%M%S%3f"))
}

/// Unix milliseconds, for session-recency stamps.
fn unix_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
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

    #[test]
    fn save_is_atomic_and_leaves_no_temp_file() {
        let tmp = TempDir::new().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf());
        let mut state = ConversationState::new("atomic");
        state.messages.push(Message::user("persist me"));
        store.save(&state).unwrap();

        // Only the target `.json` remains — the temp file is renamed away.
        let names: Vec<String> = std::fs::read_dir(tmp.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec!["atomic.json".to_string()],
            "no .json.tmp leftover"
        );
        assert_eq!(store.load("atomic").unwrap().messages.len(), 1);
    }

    #[test]
    fn load_corrupt_session_falls_back_to_empty() {
        let tmp = TempDir::new().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf());
        std::fs::write(tmp.path().join("torn.json"), "{not valid json").unwrap();
        let state = store.load("torn").unwrap();
        assert_eq!(state.session_id, "torn");
        assert!(
            state.messages.is_empty(),
            "corrupt state loads as a fresh one"
        );
    }

    #[test]
    fn summaries_report_last_user_message_and_recency() {
        let tmp = TempDir::new().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf());
        let mut state = ConversationState::new("first");
        state.messages.push(Message::system("sys"));
        state.messages.push(Message::user("hello world"));
        state.messages.push(Message::assistant("hi"));
        store.save(&state).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut recent = ConversationState::new("second");
        recent.messages.push(Message::user("another session"));
        store.save(&recent).unwrap();

        let summaries = store.summaries().unwrap();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].id, "second", "most recent first");
        assert!(summaries[0].last_user.contains("another"));
        assert_eq!(summaries[1].id, "first");
        assert_eq!(summaries[1].message_count, 3);
        assert!(summaries[1].last_user.contains("hello world"));

        assert_eq!(store.most_recent().unwrap().as_deref(), Some("second"));
    }

    #[test]
    fn summaries_empty_dir() {
        let tmp = TempDir::new().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf());
        assert!(store.summaries().unwrap().is_empty());
        assert_eq!(store.most_recent().unwrap(), None);
    }
}
