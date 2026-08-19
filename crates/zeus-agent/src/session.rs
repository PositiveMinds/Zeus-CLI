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
    /// Optional human label shown in listings/pickers instead of the opaque
    /// id. Kept in the state file (not a sidecar) so it travels with the
    /// session, but `set_label` writes the file directly without touching
    /// `last_activity`, so labeling never bumps a session to "most recent".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl ConversationState {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            messages: Vec::new(),
            last_activity: 0,
            label: None,
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
    /// Unix *millis* of the session's last activity (from `last_activity`
    /// when present, else the file's mtime), used for recency sorting and
    /// prune cutoffs. Millis, not seconds, so two sessions saved close
    /// together keep a stable relative order.
    pub modified: i64,
    /// Human label set via `zeus sessions label`, if any.
    pub label: Option<String>,
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
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            let mut message_count = 0;
            let mut last_user = String::new();
            let mut activity = modified;
            let mut label = None;
            if let Ok(text) = std::fs::read_to_string(entry.path()) {
                if let Ok(state) = serde_json::from_str::<ConversationState>(&text) {
                    message_count = state.messages.len();
                    // `last_activity` is already millis; a stale stamp of 0
                    // falls back to the file mtime above.
                    if state.last_activity > 0 {
                        activity = state.last_activity;
                    }
                    label = state.label;
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
                label,
            });
        }
        out.sort_by(|a, b| b.modified.cmp(&a.modified).then_with(|| a.id.cmp(&b.id)));
        Ok(out)
    }

    /// The most recently used saved session, if any.
    pub fn most_recent(&self) -> Result<Option<String>> {
        Ok(self.summaries()?.into_iter().next().map(|s| s.id))
    }

    /// Whether a session file exists for this id.
    pub fn exists(&self, session_id: &str) -> bool {
        self.path(session_id).exists()
    }

    /// Delete a saved session. Returns whether a file was actually removed
    /// (false when the id doesn't exist). Backs `zeus sessions rm <id>`.
    pub fn remove(&self, session_id: &str) -> Result<bool> {
        let path = self.path(session_id);
        if path.exists() {
            std::fs::remove_file(&path)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete sessions whose last activity is older than `days` days,
    /// returning the removed ids. Backs `zeus sessions prune --older-than`.
    pub fn prune_older_than(&self, days: u64) -> Result<Vec<String>> {
        let cutoff = unix_millis() - days as i64 * 86_400 * 1000;
        let mut removed = Vec::new();
        for s in self.summaries()? {
            if s.modified < cutoff {
                self.remove(&s.id)?;
                removed.push(s.id);
            }
        }
        Ok(removed)
    }

    /// Set (or clear, with `None`) a session's human label, without bumping
    /// its recency stamp. Returns whether the session exists.
    pub fn set_label(&self, session_id: &str, label: Option<String>) -> Result<bool> {
        let path = self.path(session_id);
        if !path.exists() {
            return Ok(false);
        }
        let mut state = self.load(session_id)?;
        state.label = label;
        let text =
            serde_json::to_string_pretty(&state).map_err(|e| AgentError::Session(e.to_string()))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &path)?;
        Ok(true)
    }
}

/// A fresh, timestamp-derived session id — not `Date.now()`-style
/// nondeterminism-sensitive since this is a real running CLI, not a replayed
/// script.
pub fn new_session_id() -> String {
    format!("session-{}", chrono::Utc::now().format("%Y%m%d%H%M%S%3f"))
}

/// Unix milliseconds, for session-recency stamps and prune cutoffs.
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

    #[test]
    fn remove_deletes_only_that_session() {
        let tmp = TempDir::new().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf());
        store.save(&ConversationState::new("a")).unwrap();
        store.save(&ConversationState::new("b")).unwrap();

        assert!(store.remove("a").unwrap());
        assert!(!store.remove("a").unwrap(), "second remove is a no-op");

        let mut ids = store.list_session_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec!["b".to_string()]);
    }

    #[test]
    fn prune_removes_old_sessions_but_keeps_recent() {
        let tmp = TempDir::new().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf());

        let mut old = ConversationState::new("old");
        old.messages.push(Message::user("ancient"));
        store.save(&old).unwrap();

        // Backdate the old session's activity stamp to 40 days ago.
        let old_path = tmp.path().join("old.json");
        let mut on_disk: ConversationState =
            serde_json::from_str(&std::fs::read_to_string(&old_path).unwrap()).unwrap();
        on_disk.last_activity = unix_millis() - 40 * 86_400 * 1000;
        std::fs::write(&old_path, serde_json::to_string_pretty(&on_disk).unwrap()).unwrap();

        store.save(&ConversationState::new("recent")).unwrap();

        let removed = store.prune_older_than(30).unwrap();
        assert_eq!(removed, vec!["old".to_string()]);

        let mut ids = store.list_session_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec!["recent".to_string()]);
    }

    #[test]
    fn set_label_persists_without_bumping_recency() {
        let tmp = TempDir::new().unwrap();
        let store = SessionStore::new(tmp.path().to_path_buf());

        let mut state = ConversationState::new("labelled");
        state.messages.push(Message::user("hi"));
        store.save(&state).unwrap();
        let original_activity = store.load("labelled").unwrap().last_activity;

        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(store
            .set_label("labelled", Some("my label".into()))
            .unwrap());
        assert!(!store.set_label("missing", Some("x".into())).unwrap());

        let loaded = store.load("labelled").unwrap();
        assert_eq!(loaded.label.as_deref(), Some("my label"));
        assert_eq!(
            loaded.last_activity, original_activity,
            "labeling must not change recency"
        );

        let summary = store.summaries().unwrap().remove(0);
        assert_eq!(summary.label.as_deref(), Some("my label"));

        store.set_label("labelled", None).unwrap();
        assert_eq!(store.load("labelled").unwrap().label, None);
    }
}
