//! Persistent project map under `<project>/.agent/`.
//!
//! Holds the deterministic fingerprint (`.agent/project.json`), a split-out
//! dependency list (`.agent/dependencies.json`), generated docs
//! (`.agent/architecture.md`, `.agent/conventions.md`), and long-lived
//! memory notes (`.agent/memory/*.md`). Metadata survives between sessions so
//! the model does not rediscover the project on every run: we rescan only when
//! the tree actually changes, using git as the freshness signal.

use crate::analyze::RepoFingerprint;
use std::path::{Path, PathBuf};

pub const ARCHITECTURE_DOC: &str = "architecture.md";
pub const CONVENTIONS_DOC: &str = "conventions.md";

pub fn agent_dir(root: &Path) -> PathBuf {
    root.join(".agent")
}

fn ensure_agent_dir(root: &Path) -> PathBuf {
    let dir = agent_dir(root);
    if !dir.exists() {
        let _ = std::fs::create_dir_all(&dir);
    }
    dir
}

/// Cheap git signature: `HEAD:count-of-porcelain-lines`. `None` when not a
/// git work tree. Never scans the tree, so reuse is nearly free.
fn git_signature(root: &Path) -> Option<String> {
    let head = git(root, &["rev-parse", "HEAD"])?;
    let porcelain = git(root, &["status", "--porcelain"]).unwrap_or_default();
    let count = porcelain.lines().count();
    Some(format!("{}:{}", head.trim(), count))
}

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    if out.status.success() {
        String::from_utf8(out.stdout).ok()
    } else {
        None
    }
}

/// Read `AGENTS.md` and `.agent/instructions.md` — whichever exist (in that
/// priority). Returns the merged file text (trimmed) or empty string.
pub fn project_instructions(root: &Path) -> String {
    let candidates = [
        vec![root.join("AGENTS.md")],
        vec![agent_dir(root).join("instructions.md")],
    ];
    let mut parts = Vec::new();
    for p in candidates.into_iter().flatten() {
        if p.is_file() {
            let text = read_ok(&p);
            if !text.trim().is_empty() {
                parts.push(text);
            }
        }
    }
    parts.join("\n\n")
}

/// Build the "project rules" context block from the instruction files.
pub fn project_rules_context(root: &Path) -> String {
    let text = project_instructions(root);
    if text.trim().is_empty() {
        String::new()
    } else {
        format!(
            "Project rules ({}/AGENTS.md or .agent/instructions.md):\n{}\n",
            root.display(),
            text.trim()
        )
    }
}

/// Load the persisted fingerprint if it matches current git state; otherwise
/// rescan the project and persist. Only rescans when something actually
/// changed, so repeated sessions (and repeated requests) reuse the map.
pub fn load_or_analyze(root: &Path) -> RepoFingerprint {
    let dir = agent_dir(root);
    let map = dir.join("project.json");
    let sig = git_signature(root);

    if let (Some(sig), Some(existing)) = (&sig, read_eq_sig(&dir)) {
        if *sig == existing && map.exists() {
            if let Ok(fp) = serde_json::from_str::<RepoFingerprint>(&read_ok(&map)) {
                return fp;
            }
        }
    }

    let fp = crate::analyze::analyze_repo(root);
    persist(root, &fp, sig.as_deref());
    fp
}

fn read_ok(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_default()
}

fn read_eq_sig(dir: &Path) -> Option<String> {
    let p = dir.join(".sig");
    let s = read_ok(&p);
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Write (or refresh) the on-disk map: `project.json`, `dependencies.json`,
/// and the freshness signature. Hard-fails silently on IO errors — the map is
/// a convenience, not a correctness dependency.
pub fn persist(root: &Path, fp: &RepoFingerprint, signature: Option<&str>) {
    let dir = ensure_agent_dir(root);
    let json = serde_json::to_string_pretty(fp);
    if let Ok(json) = json {
        let _ = std::fs::write(dir.join("project.json"), json);
    }
    if let Some(sig) = signature {
        let _ = std::fs::write(dir.join(".sig"), sig);
    }
    // Combined dependency list for quick, human-readable inspection.
    let mut prod: Vec<(&str, &str)> = Vec::new();
    let mut dev: Vec<(&str, &str)> = Vec::new();
    for d in &fp.dependencies {
        if d.dev {
            dev.push((&d.name, &d.version));
        } else {
            prod.push((&d.name, &d.version));
        }
    }
    let deps = serde_json::json!({
        "dependencies": prod,
        "devDependencies": dev,
    });
    if let Ok(json) = serde_json::to_string_pretty(&deps) {
        let _ = std::fs::write(dir.join("dependencies.json"), json);
    }
}

/// Generated architecture / conventions docs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WrittenDocs {
    pub architecture: bool,
    pub conventions: bool,
}

/// Write a model-generated doc (`.agent/architecture.md` /
/// `.agent/conventions.md`). Returns whether it was written.
pub fn write_generated_doc(root: &Path, name: &str, content: &str) -> bool {
    if name != ARCHITECTURE_DOC && name != CONVENTIONS_DOC {
        return false;
    }
    if content.trim().is_empty() {
        return false;
    }
    let dir = ensure_agent_dir(root);
    std::fs::write(dir.join(name), content).is_ok()
}

// ---------------------------------------------------------------------------
// Memory notes: `.agent/memory/<name>.md`
// ---------------------------------------------------------------------------
fn memory_dir(root: &Path) -> PathBuf {
    ensure_agent_dir(root).join("memory")
}

pub(crate) fn safe_memory_name(name: &str) -> Option<String> {
    let trimmed = name.trim().to_lowercase();
    if trimmed.is_empty() || trimmed.len() > 64 {
        return None;
    }
    if !trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return None;
    }
    Some(trimmed)
}

/// `(name, first meaningful line)` for every memory note, sorted by name.
pub fn memory_index(root: &Path) -> Vec<(String, String)> {
    let dir = memory_dir(root);
    let Ok(entries) = std::fs::read_dir(&dir).map(|e| e.flatten().collect::<Vec<_>>()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in entries {
        let Some(name) = e.file_name().to_str().map(|s| s.to_string()) else {
            continue;
        };
        if !name.ends_with(".md") {
            continue;
        }
        let stem = name.trim_end_matches(".md").to_string();
        let body = read_ok(&e.path());
        let first = body
            .lines()
            .map(|l| l.trim())
            .find(|l| !l.is_empty() && !l.starts_with('#'))
            .unwrap_or("");
        out.push((stem, first.chars().take(180).collect()));
    }
    out.sort();
    out
}

pub fn memory_read(root: &Path, name: &str) -> Option<String> {
    let name = safe_memory_name(name)?;
    let p = memory_dir(root).join(format!("{name}.md"));
    let body = read_ok(&p);
    if body.is_empty() {
        None
    } else {
        Some(body)
    }
}

/// Select memory snippets relevant to `request`, best-effort semantics: a
/// memory is relevant when any probe term appears in its name or body.
pub fn memory_context(root: &Path, request: &str) -> String {
    let lower = request.to_lowercase();
    let terms: Vec<String> = crate::analyze::subjects_for(request)
        .into_iter()
        .flat_map(|s| s.terms)
        .collect();
    let mut lines = Vec::new();
    for (name, first) in memory_index(root) {
        if terms.iter().any(|t| name.contains(t) || lower.contains(t))
            || (name.len() > 2 && lower.contains(&name))
        {
            lines.push(format!("- {name}: {first}"));
        }
    }
    if lines.is_empty() {
        String::new()
    } else {
        format!("long-term memory:\n{}\n", lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, rel: &str, content: &str) {
        let p = root.join(rel);
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(p, content).unwrap();
    }

    #[test]
    fn persist_and_reload_fingerprint() {
        let tmp = std::env::temp_dir().join(format!("mp-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        write(&tmp, "src/lib.rs", "pub fn main() {}\n");
        write(
            &tmp,
            "Cargo.toml",
            "[package]\nname = \"x\"\n[dependencies]\naxum = \"0.7\"\n",
        );
        let fp = crate::analyze::analyze_repo(&tmp);
        persist(&tmp, &fp, Some("deadbeef:0"));
        let loaded: RepoFingerprint =
            serde_json::from_str(&read_ok(&agent_dir(&tmp).join("project.json"))).unwrap();
        assert_eq!(loaded.languages, fp.languages);
        assert!(!fp.dependencies.is_empty());
        assert!(fp.dependencies.iter().any(|d| d.name == "axum"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn project_rules_context_reads_instructions() {
        let tmp = std::env::temp_dir().join(format!("mp-rules-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);
        assert!(project_rules_context(&tmp).is_empty());
        // 1. AGENTS.md at project root.
        std::fs::write(tmp.join("AGENTS.md"), "- Use AGENTS rules\n").unwrap();
        let ctx = project_rules_context(&tmp);
        assert!(ctx.contains("AGENTS"));
        assert!(ctx.contains("Use AGENTS rules"));
        // 2. .agent/instructions.md wins when AGENTS.md also exists.
        std::fs::write(tmp.join("AGENTS.md"), "- AGENTS\n").unwrap();
        let agent = agent_dir(&tmp);
        let _ = std::fs::create_dir_all(&agent);
        std::fs::write(agent.join("instructions.md"), "- .agent instructions\n").unwrap();
        let ctx = project_rules_context(&tmp);
        assert!(ctx.contains(".agent instructions"));
        // 3. AGENTS.md is picked up when .agent/instructions.md is absent.
        let _ = std::fs::remove_file(agent.join("instructions.md"));
        let ctx = project_rules_context(&tmp);
        assert!(ctx.contains("AGENTS"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn memory_roundtrip_and_index() {
        let tmp = std::env::temp_dir().join(format!("mp-mem-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let dir = memory_dir(&tmp);
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("auth.md"), "use token-based auth, stored in Redis").unwrap();
        assert!(safe_memory_name("bad name").is_none());
        assert!(safe_memory_name("../escape").is_none());
        assert_eq!(safe_memory_name("MY-NOTE").as_deref(), Some("my-note"));
        let idx = memory_index(&tmp);
        assert!(idx.iter().any(|(n, _)| n == "auth"));
        assert_eq!(
            memory_read(&tmp, "auth").unwrap(),
            "use token-based auth, stored in Redis"
        );
        let ctx = memory_context(&tmp, "explain how authentication works");
        assert!(
            ctx.contains("auth"),
            "expected memory in context, got: {ctx}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn load_or_analyze_writes_fingerprint_to_root_agent_dir() {
        // Regression: `load_or_analyze` used to hand `persist` the agent dir
        // (`<root>/.agent`) instead of the project root, and `persist` joins
        // `.agent` itself — producing a nested `<root>/.agent/.agent/` with
        // the fingerprint stranded inside it. The fingerprint must land at
        // `<root>/.agent/project.json`, never nested.
        let tmp = std::env::temp_dir().join(format!("mp-agdir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("src")).unwrap();
        std::fs::write(tmp.join("src/lib.rs"), "pub fn f() {}\n").unwrap();
        std::fs::write(tmp.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();

        let fp = load_or_analyze(&tmp);
        assert!(fp.source_count > 0);
        assert!(
            tmp.join(".agent/project.json").is_file(),
            "fingerprint must be at <root>/.agent/project.json"
        );
        assert!(
            !tmp.join(".agent/.agent").exists(),
            "fingerprint must not be nested under .agent/.agent"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
