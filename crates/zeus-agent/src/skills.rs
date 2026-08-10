//! Reusable skills — the SKILL.md convention (Claude Code / Codex style).
//!
//! A skill is a directory containing a `SKILL.md` file with optional YAML
//! frontmatter (a `---`-delimited header) followed by markdown-format
//! instructions the model can load on demand:
//!
//! ```text
//! skills/
//! ├── web-scraping/
//! │   ├── SKILL.md
//! │   ├── helper.py                <- optional bundled resources
//! │   └── examples/
//! └── svelte-components/
//!     └── SKILL.md
//! ```
//!
//! ```markdown
//! ---
//! name: web-scraping
//! description: Extract structured data from web pages with zeus.
//! version: 1.0.0
//! tags: [web, scraping, parsing]
//! dependencies: [requests]
//! ---
//! …instructions…
//! ```
//!
//! Discovery order (highest tier wins on name collision):
//! 1. project skills — `<project>/.agent/skills/<name>/SKILL.md`
//! 2. user skills — `~/.zeus/skills/<name>/SKILL.md`
//! 3. built-in skills shipped with zeus
//!
//! Skills are surfaced to the model through the `list_skills`/`read_skill`
//! tools so it can pull in expertise on demand instead of us bloating every
//! system prompt.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use tracing::warn;

/// Where a skill lives; higher tiers shadow lower tiers by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SkillTier {
    /// Shipped inside zeus.
    Builtin = 0,
    /// Installed under `~/.zeus/skills/`.
    Global = 1,
    /// Committed under `<project>/.agent/skills/`.
    Project = 2,
}

impl SkillTier {
    pub fn label(self) -> &'static str {
        match self {
            SkillTier::Builtin => "builtin",
            SkillTier::Global => "user",
            SkillTier::Project => "project",
        }
    }
}

/// YAML frontmatter of a `SKILL.md`. Unknown keys are tolerated so metadata
/// can evolve without breaking older parsers.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct Frontmatter {
    name: Option<String>,
    description: Option<String>,
    version: Option<String>,
    tags: Vec<String>,
    #[allow(dead_code)]
    license: Option<String>,
    dependencies: Vec<String>,
    args: Vec<SkillArg>,
    /// Other skills this one composes/assumes are loaded first (dependency
    /// order). Enables a single request to chain several skills.
    depends_on: Vec<String>,
}

/// Declarative argument a skill may accept (`args` in frontmatter).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SkillArg {
    #[allow(dead_code)]
    pub name: Option<String>,
    #[allow(dead_code)]
    pub description: Option<String>,
    #[allow(dead_code)]
    pub required: bool,
    #[allow(dead_code)]
    pub default: Option<String>,
}

/// A validated, discoverable skill.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Lowercase unique id (`name:` frontmatter or directory name).
    pub name: String,
    /// One-line description shown in `list_skills`.
    pub description: String,
    /// Semantic version if declared.
    pub version: Option<String>,
    /// Subject tags for search/grouping.
    pub tags: Vec<String>,
    /// Declared dependencies (e.g. system packages or python libs).
    pub dependencies: Vec<String>,
    /// Other skills this one composes (loaded first). Enables composable
    /// skill chains — e.g. a "build-app" skill that depends on database,
    /// backend, frontend and security skills.
    pub depends_on: Vec<String>,
    /// Declared arguments.
    pub args: Vec<SkillArg>,
    /// Markdown instructions (frontmatter stripped).
    pub instructions: String,
    /// Skill directory (`SKILL.md` + bundled resources).
    pub dir: PathBuf,
    /// Source tier (governs name precedence).
    pub tier: SkillTier,
}

impl Skill {
    /// True when the skill dir contains nothing but `SKILL.md`, i.e. has no
    /// bundled resources worth listing.
    pub fn resources_are_empty(&self) -> bool {
        match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries.flatten().all(|e| {
                let n = e.file_name();
                let n = n.to_string_lossy();
                n == "SKILL.md" || n.starts_with('.')
            }),
            Err(_) => true,
        }
    }
}

/// Read a bundled resource's text content for `read_skill`. Only small text
/// files are inlined; binaries/large files yield `None` (the caller just
/// lists the name).
pub fn read_skill_resource(skill: &Skill, resource: &str) -> Option<String> {
    let name = resource.trim_start_matches("./");
    if name.starts_with('.') || name.contains("..") {
        return None;
    }
    let path = skill.dir.join(name);
    let metadata = std::fs::metadata(&path).ok()?;
    if !metadata.is_file() || metadata.len() > 256 * 1024 {
        return None;
    }
    std::fs::read_to_string(&path).ok()
}

/// Parse a `SKILL.md` into a `Skill`. Returns `None` when unreadable, when
/// the frontmatter is unparseable, or when no usable name can be derived.
pub fn parse_skill(dir: &Path, tier: SkillTier) -> Option<Skill> {
    let markdown_path = dir.join("SKILL.md");
    let raw = std::fs::read_to_string(&markdown_path).ok()?;
    let (meta, instructions) = split_frontmatter(&raw);
    let front: Frontmatter = match serde_yaml_ng::from_str(&meta) {
        Ok(f) => f,
        Err(e) => {
            warn!(
                path = %markdown_path.display(),
                "SKILL.md frontmatter is invalid YAML: {e}"
            );
            return None;
        }
    };
    let name = front
        .name
        .or_else(|| dir.file_name().and_then(|n| n.to_str()).map(String::from))
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    if name.is_empty() || name.contains([' ', '\\']) || name.contains('/') {
        warn!(path = %markdown_path.display(), "skill has an invalid name; skipped");
        return None;
    }
    Some(Skill {
        name,
        description: front.description.unwrap_or_default().trim().to_string(),
        version: front.version,
        tags: front.tags,
        dependencies: front.dependencies,
        depends_on: front.depends_on,
        args: front.args,
        instructions: instructions.trim().to_string(),
        dir: dir.to_path_buf(),
        tier,
    })
}

/// Scan a single tier directory for skill subdirectories (those containing a
/// `SKILL.md`), parsing each best-effort.
pub fn discover_in_dir(dir: &Path, tier: SkillTier) -> Vec<Skill> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && path.join("SKILL.md").is_file() {
            if let Some(skill) = parse_skill(&path, tier) {
                out.push(skill);
            }
        }
    }
    out
}

/// One built-in skill entry: (name, description, depends_on, SKILL.md text).
pub type BuiltinSkillDef = (&'static str, &'static str, &'static [&'static str], &'static str);

/// Skills shipped inside the zeus binary. Real SKILL.md content is embedded
/// verbatim via `include_str!` so the instructions read exactly like files a
/// user would author — and project/user skills of the same name override
/// these at runtime.
pub const BUILTIN_SKILLS: &[BuiltinSkillDef] = &[
    (
        "software-engineering",
        "Orchestrate full-stack builds as a pipeline of specialist skills",
        &["project-orientation"],
        include_str!("../skills/software-engineering/SKILL.md"),
    ),
    (
        "project-orientation",
        "Map an unfamiliar repository before any change: architecture, entry points, tests, conventions",
        &[],
        include_str!("../skills/project-orientation/SKILL.md"),
    ),
    (
        "database",
        "Design schemas, SQL, and migrations; optimize queries and indexes",
        &[],
        include_str!("../skills/database/SKILL.md"),
    ),
    (
        "api",
        "Design and build HTTP APIs: REST, validation, auth, error contracts, docs, tests",
        &["database"],
        include_str!("../skills/api/SKILL.md"),
    ),
    (
        "frontend",
        "Build and refine UIs: components, state, styling, accessibility, responsive layout",
        &[],
        include_str!("../skills/frontend/SKILL.md"),
    ),
    (
        "security",
        "Harden applications: authn/authz, secrets, injection vectors, dependency vulnerabilities",
        &[],
        include_str!("../skills/security/SKILL.md"),
    ),
    (
        "qa-testing",
        "Write and repair tests (unit, integration, E2E); diagnose and de-flake failures",
        &["project-orientation"],
        include_str!("../skills/qa-testing/SKILL.md"),
    ),
    (
        "git-workflows",
        "Safe, legible git: commits, branches, diffs, merges, conflict resolution",
        &[],
        include_str!("../skills/git-workflows/SKILL.md"),
    ),
    (
        "web-research",
        "Structured web research: search, fetch, evaluate sources, evidence-backed report",
        &[],
        include_str!("../skills/web-research/SKILL.md"),
    ),
    (
        "document-reading",
        "Extract and interpret PDF, DOCX, XLSX, PPTX documents and act on their content",
        &[],
        include_str!("../skills/document-reading/SKILL.md"),
    ),
    (
        "ui-design",
        "Match and refine UI from source designs/mockups: extract design tokens and apply them",
        &["frontend"],
        include_str!("../skills/ui-design/SKILL.md"),
    ),
    (
        "documentation",
        "Write and update docs that match reality: README, architecture, API docs, guides",
        &[],
        include_str!("../skills/documentation/SKILL.md"),
    ),
    (
        "build-app",
        "Full-stack builds by composing specialist skills — database, API, frontend, security, tests, docs",
        &[
            "project-orientation",
            "database",
            "api",
            "frontend",
            "security",
            "qa-testing",
            "documentation",
        ],
        include_str!("../skills/build-app/SKILL.md"),
    ),
];

/// Build a `Skill` from its built-in definition (name, description, body).
pub fn builtin_skill(def: &BuiltinSkillDef) -> Skill {
    let (name, description, depends_on, instructions) = *def;
    Skill {
        name: name.to_string(),
        description: description.to_string(),
        version: Some("1.0.0".into()),
        tags: Vec::new(),
        dependencies: Vec::new(),
        depends_on: depends_on.iter().map(|s| s.to_string()).collect(),
        args: Vec::new(),
        instructions: instructions.to_string(),
        dir: PathBuf::new(),
        tier: SkillTier::Builtin,
    }
}

/// Resource files bundled inside a skill directory (everything except
/// `SKILL.md` and dotfiles), so the model knows what it can read.
pub fn skill_resources(skill: &Skill) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&skill.dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name == "SKILL.md" || name.starts_with('.') {
                continue;
            }
            out.push(if entry.path().is_dir() {
                format!("{name}/")
            } else {
                name
            });
        }
    }
    out.sort();
    out
}

/// Split frontmatter (`---`/`+++` fenced) from the markdown body. Returns
/// `(meta_yaml, body)`. When no fence is found, `("", raw)`.
fn split_frontmatter(raw: &str) -> (String, &str) {
    let is_fence = |l: &str| l.trim() == "---" || l.trim() == "+++";
    let lines: Vec<&str> = raw
        .split('\n')
        .map(|l| l.trim_end_matches('\r'))
        .collect();
    let first = match lines.iter().position(|l| is_fence(l.trim())) {
        Some(i) if i <= 2 => i,
        _ => return (String::new(), raw),
    };
    let closing = lines[first + 1..]
        .iter()
        .position(|l| is_fence(l.trim()))
        .map(|i| first + 1 + i);
    let Some(closing) = closing else {
        return (String::new(), raw);
    };
    // Byte offset of the line just after the closing fence.
    let mut body_start = 0usize;
    for (_n, line) in raw.split_inclusive('\n').enumerate().take(closing + 1) {
        body_start += line.len();
    }
    // Reconstruct the meta block (strip fence lines and surrounding blanks).
    let meta = lines[first + 1..closing].join("\n");
    (meta, &raw[body_start.min(raw.len())..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_skill(dir: &Path, name: &str, front: &str, body: &str) -> PathBuf {
        let d = dir.join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("SKILL.md"), format!("{front}\n{body}\n")).unwrap();
        d
    }

    #[test]
    fn parses_frontmatter_and_body() {
        let tmp = TempDir::new().unwrap();
        let d = write_skill(
            tmp.path(),
            "web-scraping",
            "---\nname: web-scraping\ndescription: Extract data from web pages\nversion: 1.2.0\ntags: [web, scraping]\ndependencies: [curl]\n---",
            "# Scraping\nUse curl, then parse with python.",
        );
        let skill = parse_skill(&d, SkillTier::Global).unwrap();
        assert_eq!(skill.name, "web-scraping");
        assert_eq!(skill.description, "Extract data from web pages");
        assert_eq!(skill.version.as_deref(), Some("1.2.0"));
        assert_eq!(skill.tags, vec!["web", "scraping"]);
        assert!(skill.instructions.contains("curl"));
        assert_eq!(skill.tier, SkillTier::Global);
    }

    #[test]
    fn missing_frontmatter_uses_dir_name() {
        let tmp = TempDir::new().unwrap();
        let d = write_skill(tmp.path(), "git-workflows", "", "# Git\nRebase rules.");
        let skill = parse_skill(&d, SkillTier::Project).unwrap();
        assert_eq!(skill.name, "git-workflows");
        assert_eq!(skill.description, "");
        assert_eq!(skill.tier, SkillTier::Project);
    }

    #[test]
    fn invalid_frontmatter_is_rejected() {
        let tmp = TempDir::new().unwrap();
        let d = write_skill(tmp.path(), "broken", "---\nname: [unterminated\n---", "x");
        assert!(parse_skill(&d, SkillTier::Builtin).is_none());
    }

    #[test]
    fn name_with_path_chars_is_rejected() {
        let tmp = TempDir::new().unwrap();
        let d = write_skill(tmp.path(), "has space", "---\nname: has space\n---", "x");
        assert!(parse_skill(&d, SkillTier::Builtin).is_none());
    }

    #[test]
    fn discover_in_dir_collects_only_valid() {
        let tmp = TempDir::new().unwrap();
        write_skill(tmp.path(), "alpha", "---\nname: alpha\ndescription: a\n---", "a");
        write_skill(tmp.path(), "beta", "---\nname: beta\n---", "b");
        std::fs::create_dir_all(tmp.path().join("not-a-skill")).unwrap();
        let found = discover_in_dir(tmp.path(), SkillTier::Builtin);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn project_shadows_global_shadows_builtin() {
        assert!(SkillTier::Project > SkillTier::Global);
        assert!(SkillTier::Global > SkillTier::Builtin);
    }

    #[test]
    fn skill_resources_lists_bundled_files() {
        let tmp = TempDir::new().unwrap();
        let d = write_skill(tmp.path(), "alpha", "---\nname: alpha\n---", "a");
        std::fs::write(d.join("helper.py"), "print(1)").unwrap();
        std::fs::create_dir_all(d.join("examples")).unwrap();
        std::fs::write(d.join(".hidden"), "x").unwrap();
        let skill = parse_skill(&d, SkillTier::Builtin).unwrap();
        let res = skill_resources(&skill);
        assert!(res.contains(&"helper.py".to_string()));
        assert!(res.contains(&"examples/".to_string()));
        assert!(!res.contains(&".hidden".to_string()));
    }

    #[test]
    fn frontmatter_stripped_from_body() {
        let tmp = TempDir::new().unwrap();
        let d = write_skill(
            tmp.path(),
            "alpha",
            "---\nname: alpha\ndescription: a\n---\n",
            "Only this body text should remain.",
        );
        let skill = parse_skill(&d, SkillTier::Builtin).unwrap();
        assert!(!skill.instructions.contains("description: a"));
        assert!(skill.instructions.contains("Only this body text"));
    }
}