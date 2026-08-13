//! Permission Gate: allow / ask / deny for tools, paths, and command patterns.

use crate::error::{FsError, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tracing::{debug, info};
use zeus_config::{AgentSettings, PermissionState};

/// What the agent wants to do.
#[derive(Debug, Clone, Default)]
pub struct PermissionRequest {
    pub tool: String,
    pub path: Option<PathBuf>,
    pub command: Option<String>,
    /// Human-readable description shown on "ask".
    pub description: String,
    /// Actual diff/content preview shown on "ask" — spec requires seeing
    /// exactly what will change, not just a one-line description.
    pub preview: Option<String>,
    /// True if this op would overwrite an existing destination (copy/rename) —
    /// drives the tailored allow/ask default for those tools.
    pub overwrites: bool,
}

/// Resolution from the gate before any user interaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedPermission {
    Allow,
    Ask { reason: String },
    Deny { reason: String },
}

/// Outcome after optional user approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    Approved,
    Denied,
    /// Approve all similar asks for the rest of this process (session-only).
    ApprovedForSession,
}

/// Runtime context for session-scoped auto-approve.
#[derive(Debug, Default)]
pub struct PermissionContext {
    /// Tools auto-approved for this session only (never persisted).
    session_auto_approve: Mutex<HashSet<String>>,
    /// When true, any Ask becomes Allow for the session after one ApprovedForSession.
    session_wide_auto: Mutex<bool>,
}

impl PermissionContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear_session(&self) {
        self.session_auto_approve.lock().unwrap().clear();
        *self.session_wide_auto.lock().unwrap() = false;
    }
}

/// Evaluates permission rules. Does not perform I/O side effects itself.
pub struct PermissionGate {
    settings: AgentSettings,
    project_root: PathBuf,
    ctx: PermissionContext,
    /// Precompiled deny path globs for sensitive reads etc.
    deny_path_globs: GlobSet,
}

impl PermissionGate {
    pub fn new(settings: AgentSettings, project_root: PathBuf) -> Self {
        let mut builder = GlobSetBuilder::new();
        for rule in &settings.permissions.rules {
            if rule.state == PermissionState::Deny {
                if let Some(pat) = &rule.path {
                    if let Ok(g) = Glob::new(pat) {
                        let _ = builder.add(g);
                    }
                }
            }
        }
        let deny_path_globs = builder.build().unwrap_or_else(|_| GlobSet::empty());
        Self {
            settings,
            project_root,
            ctx: PermissionContext::new(),
            deny_path_globs,
        }
    }

    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    pub fn context(&self) -> &PermissionContext {
        &self.ctx
    }

    /// Pure evaluation against rules (no interactive prompt).
    pub fn resolve(&self, req: &PermissionRequest) -> ResolvedPermission {
        // Session-wide auto-approve (escape hatch; never persisted).
        if *self.ctx.session_wide_auto.lock().unwrap() {
            return ResolvedPermission::Allow;
        }
        if self
            .ctx
            .session_auto_approve
            .lock()
            .unwrap()
            .contains(&req.tool)
        {
            return ResolvedPermission::Allow;
        }

        // Delete is always ask unless a narrow explicit allow rule matches.
        if req.tool == "delete" {
            if let Some(state) = self.match_rule(req) {
                return match state {
                    PermissionState::Allow => ResolvedPermission::Allow,
                    PermissionState::Deny => ResolvedPermission::Deny {
                        reason: format!("delete denied by rule: {}", req.description),
                    },
                    PermissionState::Ask => ResolvedPermission::Ask {
                        reason: format!("delete requires approval: {}", req.description),
                    },
                };
            }
            return ResolvedPermission::Ask {
                reason: format!(
                    "delete always requires approval (no silent-allow tier): {}",
                    req.description
                ),
            };
        }

        // Specific rules first (path / command).
        if let Some(state) = self.match_rule(req) {
            return state_to_resolved(state, req);
        }

        // Sensitive path deny via precompiled globs for read.
        if req.tool == "read" {
            if let Some(path) = &req.path {
                let rel = path
                    .strip_prefix(&self.project_root)
                    .unwrap_or(path)
                    .to_string_lossy();
                if self.deny_path_globs.is_match(rel.as_ref())
                    || self.deny_path_globs.is_match(path)
                {
                    return ResolvedPermission::Deny {
                        reason: format!("sensitive path denied: {rel}"),
                    };
                }
            }
        }

        // Tool default (config-provided; exact tool name still overrides the
        // tailored logic below).
        if let Some(def) = self
            .settings
            .permissions
            .defaults
            .iter()
            .find(|d| d.tool == req.tool)
        {
            return state_to_resolved(def.state, req);
        }

        // Built-in tailored defaults: copy/rename are non-destructive (and
        // fully undoable via checkpoints) as long as they don't clobber an
        // existing destination — path escape is already hard-rejected
        // upstream by path containment, so no separate "outside root" check
        // is needed here.
        if req.tool == "copy" || req.tool == "rename" {
            return if req.overwrites {
                ResolvedPermission::Ask {
                    reason: format!(
                        "{} would overwrite an existing destination: {}",
                        req.tool, req.description
                    ),
                }
            } else {
                ResolvedPermission::Allow
            };
        }

        // Built-in safe default: ask.
        ResolvedPermission::Ask {
            reason: format!(
                "no rule for tool '{}'; asking: {}",
                req.tool, req.description
            ),
        }
    }

    fn match_rule(&self, req: &PermissionRequest) -> Option<PermissionState> {
        // Last matching rule wins (higher-precedence layers already merged into settings).
        let mut matched: Option<PermissionState> = None;
        for rule in &self.settings.permissions.rules {
            if rule.tool != req.tool {
                continue;
            }
            if let Some(cmd_pat) = &rule.command {
                let Some(cmd) = &req.command else {
                    continue;
                };
                if !command_matches(cmd_pat, cmd) {
                    continue;
                }
            }
            if let Some(path_pat) = &rule.path {
                let Some(path) = &req.path else {
                    continue;
                };
                let rel = path
                    .strip_prefix(&self.project_root)
                    .unwrap_or(path)
                    .to_string_lossy();
                if let Ok(g) = Glob::new(path_pat) {
                    if !g.compile_matcher().is_match(rel.as_ref())
                        && !g.compile_matcher().is_match(path)
                    {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            matched = Some(rule.state);
        }
        matched
    }

    /// Enforce: Allow → Ok, Deny → Err, Ask → invoke `approver`.
    pub fn enforce<F>(&self, req: &PermissionRequest, mut approver: F) -> Result<()>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        match self.resolve(req) {
            ResolvedPermission::Allow => {
                debug!(tool = %req.tool, "permission allow");
                Ok(())
            }
            ResolvedPermission::Deny { reason } => {
                info!(tool = %req.tool, %reason, "permission deny");
                Err(FsError::Denied(reason))
            }
            ResolvedPermission::Ask { reason } => {
                info!(tool = %req.tool, %reason, "permission ask");
                match approver(req) {
                    ApprovalDecision::Approved => Ok(()),
                    ApprovalDecision::ApprovedForSession => {
                        self.ctx
                            .session_auto_approve
                            .lock()
                            .unwrap()
                            .insert(req.tool.clone());
                        if self.settings.permissions.allow_session_auto_approve {
                            *self.ctx.session_wide_auto.lock().unwrap() = true;
                        }
                        Ok(())
                    }
                    ApprovalDecision::Denied => {
                        Err(FsError::Denied(format!("user denied: {}", req.description)))
                    }
                }
            }
        }
    }

    /// Non-interactive enforce used in tests / auto mode: Ask becomes error unless session auto.
    pub fn enforce_strict(&self, req: &PermissionRequest) -> Result<()> {
        match self.resolve(req) {
            ResolvedPermission::Allow => Ok(()),
            ResolvedPermission::Deny { reason } => Err(FsError::Denied(reason)),
            ResolvedPermission::Ask { reason } => Err(FsError::NeedsApproval(reason)),
        }
    }
}

fn state_to_resolved(state: PermissionState, req: &PermissionRequest) -> ResolvedPermission {
    match state {
        PermissionState::Allow => ResolvedPermission::Allow,
        PermissionState::Deny => ResolvedPermission::Deny {
            reason: req.description.clone(),
        },
        PermissionState::Ask => ResolvedPermission::Ask {
            reason: req.description.clone(),
        },
    }
}

/// Simple command pattern: `*` is a suffix/prefix wildcard (not full regex).
fn command_matches(pattern: &str, command: &str) -> bool {
    let cmd = command.trim();
    if let Some(prefix) = pattern.strip_suffix('*') {
        return cmd.starts_with(prefix) || cmd.contains(prefix);
    }
    cmd == pattern || cmd.starts_with(&format!("{pattern} "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeus_config::AgentSettings;

    fn gate() -> PermissionGate {
        PermissionGate::new(AgentSettings::default(), PathBuf::from("/proj"))
    }

    #[test]
    fn read_allowed_by_default() {
        let g = gate();
        let r = g.resolve(&PermissionRequest {
            tool: "read".into(),
            path: Some(PathBuf::from("/proj/src/main.rs")),
            command: None,
            description: "read main.rs".into(),
            ..Default::default()
        });
        assert_eq!(r, ResolvedPermission::Allow);
    }

    #[test]
    fn delete_always_asks() {
        let g = gate();
        let r = g.resolve(&PermissionRequest {
            tool: "delete".into(),
            path: Some(PathBuf::from("/proj/tmp.txt")),
            command: None,
            description: "delete tmp.txt".into(),
            ..Default::default()
        });
        assert!(matches!(r, ResolvedPermission::Ask { .. }));
    }

    #[test]
    fn rm_rf_denied() {
        let g = gate();
        let r = g.resolve(&PermissionRequest {
            tool: "bash".into(),
            path: None,
            command: Some("rm -rf /".into()),
            description: "run rm -rf".into(),
            ..Default::default()
        });
        assert!(matches!(r, ResolvedPermission::Deny { .. }));
    }

    #[test]
    fn copy_allowed_when_not_overwriting() {
        let g = gate();
        let r = g.resolve(&PermissionRequest {
            tool: "copy".into(),
            path: Some(PathBuf::from("/proj/b.txt")),
            command: None,
            description: "copy a.txt -> b.txt".into(),
            overwrites: false,
            ..Default::default()
        });
        assert_eq!(r, ResolvedPermission::Allow);
    }

    #[test]
    fn copy_asks_when_overwriting() {
        let g = gate();
        let r = g.resolve(&PermissionRequest {
            tool: "copy".into(),
            path: Some(PathBuf::from("/proj/b.txt")),
            command: None,
            description: "copy a.txt -> b.txt".into(),
            overwrites: true,
            ..Default::default()
        });
        assert!(matches!(r, ResolvedPermission::Ask { .. }));
    }

    #[test]
    fn rename_allowed_when_not_overwriting() {
        let g = gate();
        let r = g.resolve(&PermissionRequest {
            tool: "rename".into(),
            path: Some(PathBuf::from("/proj/a.txt")),
            command: None,
            description: "rename a.txt -> b.txt".into(),
            overwrites: false,
            ..Default::default()
        });
        assert_eq!(r, ResolvedPermission::Allow);
    }

    #[test]
    fn session_auto_approve() {
        let g = gate();
        let req = PermissionRequest {
            tool: "write".into(),
            path: Some(PathBuf::from("/proj/a.txt")),
            command: None,
            description: "write a.txt".into(),
            ..Default::default()
        };
        assert!(matches!(g.resolve(&req), ResolvedPermission::Ask { .. }));
        g.enforce(&req, |_| ApprovalDecision::ApprovedForSession)
            .unwrap();
        assert_eq!(g.resolve(&req), ResolvedPermission::Allow);
    }
}
