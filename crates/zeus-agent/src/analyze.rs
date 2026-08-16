//! Repository understanding: a deterministic "what's in this project?"
//! snapshot plus a per-request probe that flags existing code matching what
//! the user asked for.
//!
//! Layers:
//!  - [`analyze_repo`] → [`RepoFingerprint`]: walk manifests + file tree to
//!    detect language stack, frameworks, package manager, database, entry
//!    points, build/test commands, important dirs, git state. Cheap, no model
//!    calls; cached per session on the `Agent`.
//!  - [`RepoFingerprint::probe`] → [`ProbeReport`]: for a user request, list
//!    the modules/files whose *names* already relate to the subject (auth,
//!    database, payments…). Filename-level signal; semantic confirmation
//!    ("the JWT middleware already exists") is the on-demand agent pass
//!    (`/understand <topic>`) that reads those files and reports.
//!
//! Goal: build on and extend what's already there instead of re-authoring it.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Directory names that never add project signal (never scanned).
const IGNORED_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    ".bzr",
    "node_modules",
    "target",
    "build",
    "dist",
    ".next",
    ".nuxt",
    ".output",
    ".svelte-kit",
    ".cache",
    ".venv",
    "venv",
    "env",
    "envs",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".gradle",
    ".idea",
    ".vscode",
    ".agent",
    "coverage",
    ".parcel-cache",
    ".dart_tool",
    ".yarn",
    ".pnpm-store",
    "vendor",
    "third_party",
    "__snapshots__",
];

/// Files that describe the build (never counted as source or tests).
const CONFIG_NAMES: &[&str] = &[
    "cargo.toml",
    "cargo.lock",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lockb",
    "tsconfig.json",
    "svelte.config.js",
    "vite.config.ts",
    "vite.config.js",
    "vitest.config.ts",
    "next.config.js",
    "tailwind.config.js",
    "tailwind.config.ts",
    "postcss.config.js",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yaml",
    "compose.yml",
    "Makefile",
    "README",
    "README.md",
    "Gemfile",
    "Gemfile.lock",
    "go.mod",
    "go.sum",
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
    "requirements-dev.txt",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "biome.json",
    "eslint.config.js",
    "eslint.config.mjs",
    ".prettierrc",
    "webpack.config.js",
    "rollup.config.js",
    "vite.config.mjs",
    "cmakelists.txt",
    "build.gradle",
    "build.gradle.kts",
    "pom.xml",
    "settings.gradle",
    "build.rs",
    "diesel.toml",
    "alembic.ini",
    "manage.py",
    "lerna.json",
    "turbo.json",
    ".env.example",
    ".env.sample",
    "vercel.json",
    "netlify.toml",
    ".node-version",
    ".nvmrc",
];

fn language_for(ext: &str) -> Option<&'static str> {
    Some(match ext {
        "rs" => "Rust",
        "ts" | "mts" | "cts" => "TypeScript",
        "tsx" => "TypeScript (React)",
        "js" | "mjs" | "cjs" => "JavaScript",
        "jsx" => "JavaScript (React)",
        "py" | "pyi" => "Python",
        "go" => "Go",
        "java" => "Java",
        "kt" | "kts" => "Kotlin",
        "swift" => "Swift",
        "rb" => "Ruby",
        "php" => "PHP",
        "c" => "C",
        "h" => "C header",
        "cpp" | "cc" | "cxx" | "hpp" => "C++",
        "cs" => "C#",
        "sql" => "SQL",
        "html" | "htm" | "vue" | "svelte" | "twig" | "ejs" | "hbs" => "HTML",
        "css" | "scss" | "sass" | "less" => "CSS",
        "dart" => "Dart",
        "scala" => "Scala",
        "ex" | "exs" => "Elixir",
        "clj" | "cljs" => "Clojure",
        _ => return None,
    })
}

/// A file counted by the scan, with classifier flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoFile {
    /// Path relative to the project root.
    pub rel: PathBuf,
    pub is_test: bool,
    pub is_config: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GitReport {
    pub present: bool,
    pub branch: String,
    pub unstaged: usize,
    pub staged: usize,
    pub untracked: usize,
    pub conflicts: usize,
}

impl GitReport {
    fn absent() -> Self {
        Self {
            present: false,
            branch: String::new(),
            unstaged: 0,
            staged: 0,
            untracked: 0,
            conflicts: 0,
        }
    }
    pub fn clean(&self) -> bool {
        !self.present || (self.unstaged + self.staged + self.untracked + self.conflicts) == 0
    }
    pub fn one_line(&self) -> String {
        if !self.present {
            return "not a git repo".to_string();
        }
        let n = self.unstaged + self.staged + self.untracked + self.conflicts;
        let state = if n == 0 {
            "clean".to_string()
        } else {
            format!("{n} uncommitted change(s)")
        };
        if self.branch.is_empty() {
            state
        } else {
            format!("{} {}", self.branch, state)
        }
    }
}

/// One dependency extracted from a manifest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    /// Requirement/semver string as written (best-effort).
    pub version: String,
    /// true for devDependencies / [dev-dependencies].
    pub dev: bool,
}

/// Deterministic snapshot of a project's stack and shape.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepoFingerprint {
    /// All scanned files (for the probe pass).
    pub files: Vec<RepoFile>,
    /// (display language name, file count), descending.
    pub languages: Vec<(String, usize)>,
    /// e.g. "React", "Axum", "Tauri".
    pub frameworks: Vec<String>,
    /// e.g. "pnpm", "cargo", "uv".
    pub package_managers: Vec<String>,
    /// e.g. "PostgreSQL", "SQLite", "Prisma".
    pub databases: Vec<String>,
    pub entry_points: Vec<String>,
    pub build_commands: Vec<String>,
    pub test_commands: Vec<String>,
    pub important_dirs: Vec<String>,
    pub source_count: usize,
    pub test_count: usize,
    pub config_count: usize,
    pub git: GitReport,
    /// Parsed dependency records extracted from manifests.
    pub dependencies: Vec<Dependency>,
}

/// Directory basenames that indicate a project area (used for important dirs).
const IMPORTANT_DIR_NAMES: &[&str] = &[
    "src",
    "lib",
    "app",
    "core",
    "server",
    "backend",
    "frontend",
    "client",
    "api",
    "apis",
    "services",
    "shared",
    "components",
    "pages",
    "views",
    "config",
    "configs",
    "tests",
    "test",
    "e2e",
    "migrations",
    "db",
    "infra",
    "deploy",
    "public",
    "assets",
    "crates",
    "modules",
    "packages",
    "apps",
];

impl RepoFingerprint {
    /// Banner lines (`✓ React + TypeScript`, etc.) for the human.
    pub fn banner_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if !self.languages.is_empty() {
            let langs = self
                .languages
                .iter()
                .map(|(name, n)| format!("{} ({n})", name))
                .take(3)
                .collect::<Vec<_>>()
                .join(" + ");
            lines.push(format!("✓ {langs}"));
        }
        if !self.frameworks.is_empty() {
            lines.push(format!("✓ {}", self.frameworks.join(" + ")));
        }
        if !self.package_managers.is_empty() {
            lines.push(format!("✓ {}", self.package_managers.join(" + ")));
        }
        if !self.databases.is_empty() {
            lines.push(format!("✓ {}", self.databases.join(" + ")));
        }
        if self.source_count > 0 {
            lines.push(format!(
                "✓ {} source files, {} test files",
                self.source_count, self.test_count
            ));
        }
        if !self.test_commands.is_empty() {
            lines.push(format!("✓ tests: {}", self.test_commands.join(" · ")));
        }
        if !self.entry_points.is_empty() {
            lines.push(format!(
                "✓ entry: {}",
                self.entry_points
                    .iter()
                    .take(4)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if self.git.present {
            lines.push(format!("✓ git: {}", self.git.one_line()));
        }
        lines
    }

    /// The full report block: stack banner + request-relevant existing code.
    pub fn render(&self, request: &str) -> String {
        let banner = self.banner_lines().join("\n");
        let mut out = format!("Repository understanding:\n{banner}");
        let probe = self.probe(request);
        if probe.hits.is_empty() {
            out.push_str(
                "\n\nNo obviously-relevant existing modules matched by name. Verify with grep/glob \
                 before writing new files; if nothing exists, build from scratch.",
            );
        } else {
            out.push_str("\n\n");
            out.push_str(&probe.render());
        }
        out
    }

    /// Map a request onto existing files. Filename-level signal only.
    pub fn probe(&self, request: &str) -> ProbeReport {
        probe_files(&self.files, request)
    }
}

/// One matched subject (e.g. `authentication`) with the files matching it.
#[derive(Debug, Clone, Default)]
pub struct ProbeHit {
    pub label: String,
    /// (directory rel path, hit count).
    pub dir_hits: Vec<(String, usize)>,
    pub files: Vec<String>,
}

impl ProbeHit {
    pub fn count(&self) -> usize {
        self.dir_hits.iter().map(|(_, n)| *n).sum()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ProbeReport {
    pub hits: Vec<ProbeHit>,
}

impl ProbeReport {
    pub fn render(&self) -> String {
        if self.hits.is_empty() {
            return String::new();
        }
        let mut out = String::from(
            "related code already in the repo (matched by name — read it before reimplementing):",
        );
        for hit in self.hits.iter().take(8) {
            out.push_str(&format!("\n- {} ({}): ", hit.label, hit.count()));
            let dirs: Vec<String> = hit
                .dir_hits
                .iter()
                .take(3)
                .map(|(d, n)| format!("{d} ({n})"))
                .collect();
            if !dirs.is_empty() {
                out.push_str(&dirs.join(", "));
                out.push_str(" · ");
            }
            out.push_str(
                &hit.files
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }
        out
    }
}

/// Subject rules: a request keyword maps to related aliases used to search
/// filenames, then the whole group is presented under one label.
const TOPIC_RULES: &[(&str, &[&str])] = &[
    (
        "authentication",
        &[
            "auth",
            "login",
            "signin",
            "signup",
            "register",
            "logout",
            "jwt",
            "oauth",
            "session",
            "password",
            "credential",
            "user",
            "account",
            "permission",
            "role",
            "token",
            "mfa",
        ],
    ),
    (
        "database",
        &[
            "db",
            "datastore",
            "sql",
            "schema",
            "table",
            "migration",
            "model",
            "entity",
            "query",
            "orm",
            "postgres",
            "postgresql",
            "mysql",
            "sqlite",
            "mongodb",
            "mongo",
            "redis",
            "prisma",
            "sequelize",
            "typeorm",
            "diesel",
            "sqlx",
            "sea-orm",
        ],
    ),
    (
        "api",
        &[
            "api",
            "rest",
            "endpoint",
            "route",
            "graphql",
            "controller",
            "handler",
            "middleware",
            "webhook",
            "service",
        ],
    ),
    (
        "ui/frontend",
        &[
            "frontend",
            "ui",
            "component",
            "page",
            "view",
            "react",
            "vue",
            "svelte",
            "angular",
            "css",
            "tailwind",
            "style",
            "jsx",
            "tsx",
            "design",
            "dashboard",
        ],
    ),
    (
        "backend/web",
        &[
            "backend", "server", "app", "express", "fastify", "nginx", "deploy",
        ],
    ),
    (
        "tests",
        &[
            "test", "spec", "fixture", "mock", "jest", "pytest", "expect",
        ],
    ),
    (
        "configuration",
        &[
            "config",
            "env",
            "environment",
            "setting",
            "dotenv",
            "setup",
            "yml",
            "toml",
        ],
    ),
    (
        "git",
        &[
            "git", "commit", "branch", "merge", "rebase", "stash", "checkout",
        ],
    ),
    (
        "ci/infra",
        &[
            "ci",
            "pipeline",
            "workflow",
            "terraform",
            "ansible",
            "kubernetes",
            "k8s",
            "docker",
            "container",
            "actions",
        ],
    ),
    (
        "security",
        &[
            "security", "vulner", "sanitize", "encrypt", "hash", "captcha", "csrf", "cors", "ssl",
            "tls", "throttle",
        ],
    ),
    (
        "realtime",
        &["websocket", "socket", "pubsub", "stomp", "socket.io"],
    ),
    (
        "search",
        &["search", "index", "elastic", "meilisearch", "solr", "lunr"],
    ),
    (
        "payments",
        &[
            "payment",
            "stripe",
            "checkout",
            "invoice",
            "billing",
            "subscription",
            "charge",
        ],
    ),
    (
        "data/file",
        &[
            "etl",
            "analytics",
            "streaming",
            "kafka",
            "spark",
            "report",
            "export",
            "import",
            "csv",
        ],
    ),
];

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "or", "the", "to", "in", "on", "of", "for", "with", "at", "from", "by", "is",
    "it", "its", "be", "do", "does", "did", "was", "are", "this", "that", "these", "those", "we",
    "you", "they", "i", "my", "your", "our", "have", "has", "had", "will", "would", "should",
    "could", "please", "make", "creating", "add", "create", "remove", "update", "fix", "write",
    "code", "want", "need", "help", "how", "why", "what", "where", "when", "which", "using",
    "used", "use", "like", "just", "about", "into", "not", "no", "yes", "also", "then", "there",
    "here", "all", "some", "any",
];

pub(crate) struct ProbeSubject {
    pub(crate) label: String,
    pub(crate) terms: Vec<String>,
}

/// Curated topics whose aliases appear in the request, plus significant raw
/// words (each becomes its own subject).
pub(crate) fn subjects_for(request: &str) -> Vec<ProbeSubject> {
    let lower = request.to_lowercase();
    let mut out: Vec<ProbeSubject> = Vec::new();
    for (label, aliases) in TOPIC_RULES {
        if aliases.iter().any(|a| lower.contains(a)) {
            out.push(ProbeSubject {
                label: label.to_string(),
                terms: aliases.iter().map(|a| a.to_string()).collect(),
            });
        }
    }
    for word in lower.split(|c: char| !c.is_alphanumeric()) {
        if word.len() >= 3
            && word.len() <= 24
            && !STOPWORDS.contains(&word)
            && !word.chars().all(|c| c.is_ascii_digit())
            && !out.iter().any(|s| s.label == word)
        {
            out.push(ProbeSubject {
                label: word.to_string(),
                terms: vec![word.to_string()],
            });
        }
    }
    out
}

/// Filename-level match: term hits any path segment by exact/prefix equality
/// (so `auth` hits `authentication/`, `user` hits `users.rs`, not `browser`).
fn segment_hit(rel_lower: &str, term: &str) -> bool {
    rel_lower
        .split(['/', '\\', '.', '_', '-'])
        .any(|seg| !seg.is_empty() && (seg == term || seg.starts_with(term)))
}

fn probe_files(files: &[RepoFile], request: &str) -> ProbeReport {
    let subjects = subjects_for(request);
    let mut hits: Vec<ProbeHit> = Vec::new();

    for subject in subjects {
        let mut matches: Vec<usize> = Vec::new();
        for (idx, f) in files.iter().enumerate() {
            if f.is_config {
                continue;
            }
            let rel_lower = f.rel.to_string_lossy().to_lowercase();
            if subject.terms.iter().any(|t| segment_hit(&rel_lower, t)) {
                matches.push(idx);
            }
        }
        if matches.is_empty() {
            continue;
        }
        if matches.len() > 60 {
            matches.truncate(60);
        }
        let mut dir_counter: BTreeMap<String, usize> = BTreeMap::new();
        let mut sample: Vec<String> = Vec::new();
        for idx in &matches {
            let rel = &files[*idx].rel;
            let s = rel.to_string_lossy();
            let dir = match rel.parent() {
                Some(p) if p.as_os_str().is_empty() => ".".to_string(),
                Some(p) => p.to_string_lossy().into_owned(),
                None => ".".to_string(),
            };
            *dir_counter.entry(dir.clone()).or_default() += 1;
            if sample.len() < 5 {
                sample.push(s.into_owned());
            }
        }
        hits.push(ProbeHit {
            label: subject.label,
            dir_hits: dir_counter.into_iter().collect(),
            files: sample,
        });
    }

    // Rank by hit count; keep one per label.
    hits.sort_by_key(|h| std::cmp::Reverse(h.count()));
    let mut seen = BTreeMap::new();
    let mut uniq: Vec<ProbeHit> = Vec::new();
    for h in hits {
        if !seen.contains_key(&h.label) {
            seen.insert(h.label.clone(), ());
            uniq.push(h);
        }
    }
    ProbeReport { hits: uniq }
}

/// Recursively collect files, skipping ignored dirs and capping depth/amount
/// so a pathological repo can't hang the CLI. Every entry is kept only when it
/// genuinely lives inside `root`; symlinks/junctions are resolved and never
/// followed when they point outside it, so a scan can't leak into unrelated
/// trees sitting elsewhere on disk.
fn collect_all(root: &Path) -> Vec<RepoFile> {
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if out.len() > 50_000 {
            break;
        }
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();

            // Symlink/junction safety: a link (or junction) can point at an
            // unrelated tree anywhere on disk. Resolve it and require the
            // target to stay inside this project before listing or descending
            // into it — otherwise the "scan of the current directory" would
            // silently widen into a false broader environment.
            let symlink_target = std::fs::symlink_metadata(&path)
                .ok()
                .filter(|m| m.file_type().is_symlink())
                .and_then(|_| path.canonicalize().ok());
            if let Some(target) = &symlink_target {
                if !target.starts_with(root) {
                    continue;
                }
            }

            // Containment: never include an entry that resolves outside `root`
            // (a broken/stray prefix can't be a project file).
            let rel = match path.strip_prefix(root) {
                Ok(rel) => rel.to_path_buf(),
                Err(_) => continue,
            };

            let is_dir = match &symlink_target {
                Some(target) => target.is_dir(),
                None => path.is_dir(),
            };
            if is_dir {
                if IGNORED_DIRS.contains(&name.as_str()) {
                    continue;
                }
                if rel.components().count() > 10 {
                    continue;
                }
                stack.push(path);
            } else {
                let is_test = is_test_path(&rel);
                let is_config = is_config_path(&rel);
                out.push(RepoFile {
                    rel,
                    is_test,
                    is_config,
                });
            }
        }
    }
    out
}

fn is_config_path(rel: &Path) -> bool {
    let s = rel.to_string_lossy().to_ascii_lowercase();
    if rel.file_name().is_none() {
        return false;
    }
    let base = rel
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_ascii_lowercase();
    CONFIG_NAMES.contains(&base.as_str())
        || s.starts_with(".github/")
        || s.starts_with(".env")
        || (rel.components().count() <= 3
            && rel.extension().map(|e| e.to_ascii_lowercase()) == Some("toml".into())
            && (s.starts_with("config/") || s.starts_with(".config/")))
        || s.ends_with(".config.json")
}

fn is_test_path(rel: &Path) -> bool {
    let s = rel.to_string_lossy().to_ascii_lowercase();
    let base = rel
        .file_name()
        .map(|b| b.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let dir_part = rel
        .parent()
        .map(|p| p.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    s.contains("/test/")
        || dir_part.starts_with("tests")
        || dir_part.ends_with("/tests")
        || dir_part.ends_with("/test")
        || dir_part.contains("__tests__")
        || s.contains(".test.")
        || s.contains(".spec.")
        || s.contains("_test.rs")
        || base.ends_with("_test.go")
        || base.ends_with("_spec.rb")
        || base == "test"
        || base == "tests"
        || base == "spec"
}

/// Determine the test command for this repo via the `test` tool's detector.
fn infer_test_commands(root: &Path) -> Vec<String> {
    crate::tools::detect_test_command(root)
        .into_iter()
        .collect()
}

/// Build the deterministic fingerprint for a project root.
pub fn analyze_repo(root: &Path) -> RepoFingerprint {
    let mut f = RepoFingerprint::default();
    let files = collect_all(root);
    f.files = files.clone();

    // Language counts + source/test/config totals.
    let mut by_ext: BTreeMap<String, usize> = BTreeMap::new();
    for file in &files {
        if file.is_config {
            f.config_count += 1;
            continue;
        }
        let ext = file
            .rel
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext.is_empty() {
            continue;
        }
        if file.is_test {
            f.test_count += 1;
        } else {
            f.source_count += 1;
        }
        *by_ext.entry(ext).or_default() += 1;
    }
    let mut lang_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (ext, n) in &by_ext {
        if let Some(name) = language_for(ext) {
            *lang_counts.entry(name).or_default() += n;
        }
    }
    let mut langs: Vec<(String, usize)> = lang_counts
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    langs.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
    f.languages = langs;

    // Manifest content (root + common app dirs) for framework/db/manager hunts.
    let manifest_sites: Vec<PathBuf> = {
        let mut sites = vec![root.to_path_buf()];
        for dir in [
            "frontend", "backend", "client", "server", "web", "api", "apps", "packages",
        ] {
            let p = root.join(dir);
            if p.is_dir() {
                sites.push(p);
            }
        }
        sites
    };
    let mut texts: Vec<String> = Vec::new();
    for site in &manifest_sites {
        for name in [
            "Cargo.toml",
            "package.json",
            "pyproject.toml",
            "requirements.txt",
            "go.mod",
            "Gemfile",
            "composer.json",
        ] {
            let p = site.join(name);
            if p.is_file() {
                if let Some(t) = read_small(&p) {
                    texts.push(t);
                }
            }
        }
    }
    let joined = texts.join("\n").to_lowercase();

    detect_frameworks(&joined, &mut f);
    detect_databases(&joined, &files, &mut f);
    detect_managers(&joined, &files, &mut f);

    f.entry_points = detect_entry_points(root);
    f.build_commands = detect_build_commands(&texts, root);
    f.test_commands = infer_test_commands(root);
    f.important_dirs = detect_important_dirs(&files);
    f.git = git_report(root);
    f.dependencies = detect_dependencies(&texts);

    // De-dupe + order.
    for list in [
        &mut f.frameworks,
        &mut f.package_managers,
        &mut f.databases,
        &mut f.entry_points,
        &mut f.build_commands,
        &mut f.important_dirs,
    ] {
        list.sort();
        list.dedup();
    }
    f
}

fn read_small(p: &Path) -> Option<String> {
    let meta = p.metadata().ok()?;
    if meta.len() > 1_048_576 {
        return None;
    }
    std::fs::read_to_string(p).ok()
}

fn contains_word(haystack: &str, needle: &str) -> bool {
    let bytes = haystack.as_bytes();
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        let idx = start + pos;
        let before_ok = idx == 0 || !bytes[idx - 1].is_ascii_alphanumeric();
        let after = idx + needle.len();
        let after_ok = after >= bytes.len() || !bytes[after].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
        start = idx + 1;
    }
    false
}

fn detect_frameworks(joined: &str, f: &mut RepoFingerprint) {
    const MARKS: &[(&str, &str)] = &[
        ("axum", "Axum"),
        ("actix-web", "Actix Web"),
        ("rocket", "Rocket"),
        ("warp", "Warp"),
        ("poem", "Poem"),
        ("tauri", "Tauri"),
        ("react", "React"),
        ("preact", "Preact"),
        ("next", "Next.js"),
        ("nuxt", "Nuxt"),
        ("vue", "Vue"),
        ("svelte", "Svelte"),
        ("@angular/core", "Angular"),
        ("vite", "Vite"),
        ("express", "Express"),
        ("fastify", "Fastify"),
        ("@nestjs/core", "NestJS"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "FastAPI"),
        ("django", "Django"),
        ("flask", "Flask"),
        ("gin-gonic", "Gin"),
        ("gofiber", "Fiber"),
        ("laravel", "Laravel"),
        ("sinatra", "Sinatra"),
        ("springframework", "Spring"),
        ("rails", "Ruby on Rails"),
        ("microsoft.aspnetcore", "ASP.NET Core"),
        ("microsoft.net.sdk.web", "ASP.NET Core"),
        ("flutter", "Flutter"),
        ("astro", "Astro"),
        ("remix", "Remix"),
        ("sveltekit", "SvelteKit"),
        ("solid-js", "Solid"),
        ("ember", "Ember"),
        ("backbone", "Backbone"),
        ("koa", "Koa"),
        ("hapi", "Hapi"),
        ("phoenix", "Phoenix"),
        ("symfony", "Symfony"),
        ("codeigniter", "CodeIgniter"),
        ("hugo", "Hugo"),
        ("jekyll", "Jekyll"),
        ("docusaurus", "Docusaurus"),
        ("egui", "egui"),
        ("iced", "Iced"),
        ("slint", "Slint"),
    ];
    for (needle, label) in MARKS {
        if contains_word(joined, needle) && !f.frameworks.iter().any(|x| x == label) {
            f.frameworks.push(label.to_string());
        }
    }
}

fn detect_databases(joined_manifests: &str, files: &[RepoFile], f: &mut RepoFingerprint) {
    const DATABASES: &[(&str, &str)] = &[
        ("postgres", "PostgreSQL"),
        ("mysql", "MySQL"),
        ("mariadb", "MariaDB"),
        ("sqlite", "SQLite"),
        ("rusqlite", "SQLite"),
        ("mongodb", "MongoDB"),
        ("mongo", "MongoDB"),
        ("redis", "Redis"),
        ("cassandra", "Cassandra"),
    ];
    for (needle, label) in DATABASES {
        if joined_manifests.contains(needle) && !f.databases.iter().any(|d| d == label) {
            f.databases.push(label.to_string());
        }
    }
    const ORMS: &[(&str, &str)] = &[
        ("sqlx", "SQLx"),
        ("diesel", "Diesel"),
        ("sea-orm", "SeaORM"),
        ("prisma", "Prisma"),
        ("typeorm", "TypeORM"),
        ("sequelize", "Sequelize"),
        ("sqlalchemy", "SQLAlchemy"),
        ("alembic", "Alembic"),
        ("gorm", "GORM"),
    ];
    for (needle, label) in ORMS {
        if joined_manifests.contains(needle) && !f.databases.iter().any(|d| d == label) {
            f.databases.push(label.to_string());
        }
    }
    let migration_present = files.iter().any(|s2| {
        let s = s2.rel.to_string_lossy();
        s.contains("/migrations/")
            || s.starts_with("migrations/")
            || s.contains("/prisma/")
            || s.starts_with("prisma/")
            || s.contains("/alembic/")
            || s.starts_with("alembic/")
    });
    if migration_present && f.databases.is_empty() {
        f.databases.push("migrations present".to_string());
    }
}

fn detect_managers(_joined_manifests: &str, files: &[RepoFile], f: &mut RepoFingerprint) {
    let has = |name: &str| {
        files.iter().any(|s2| {
            s2.rel.file_name().is_some() && s2.rel.file_name().unwrap().to_string_lossy() == name
        })
    };
    let mut found: Vec<&str> = Vec::new();
    if has("Cargo.toml") {
        found.push("cargo");
    }
    if has("pnpm-lock.yaml") {
        found.push("pnpm");
    } else if has("yarn.lock") {
        found.push("yarn");
    } else if has("bun.lockb") {
        found.push("bun");
    } else if has("package-lock.json") {
        found.push("npm");
    }
    if has("uv.lock") {
        found.push("uv");
    } else if has("poetry.lock") {
        found.push("poetry");
    }
    if has("go.mod") {
        found.push("go modules");
    }
    if has("Gemfile") {
        found.push("bundler");
    }
    for pm in found {
        if !f.package_managers.iter().any(|x| x == pm) {
            f.package_managers.push(pm.to_string());
        }
    }
    // No lockfile but a package.json → npm is the sane default.
    if f.package_managers.is_empty() && has("package.json") {
        f.package_managers.push("npm".to_string());
    }
}

fn detect_entry_points(root: &Path) -> Vec<String> {
    let candidates = [
        "src/main.rs",
        "src/lib.rs",
        "src/main.ts",
        "src/main.tsx",
        "src/main.js",
        "src/index.ts",
        "src/index.tsx",
        "src/index.js",
        "main.py",
        "app.py",
        "manage.py",
        "run.py",
        "wsgi.py",
        "asgi.py",
        "src/main.py",
        "src/app.py",
        "server.js",
        "server.ts",
        "index.js",
        "index.py",
    ];
    candidates
        .iter()
        .filter(|rel| root.join(rel).is_file())
        .map(|s| s.to_string())
        .collect()
}

fn detect_dependencies(texts: &[String]) -> Vec<Dependency> {
    let mut out: Vec<Dependency> = Vec::new();
    for text in texts {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
            parse_json_manifest(&v, &mut out);
            continue;
        }
        if text.contains("[dependencies]") {
            parse_cargo_manifest(text, &mut out);
            continue;
        }
        if text.trim().starts_with("require") || text.contains("require (") {
            parse_go_manifest(text, &mut out);
            continue;
        }
        parse_requirements(text, &mut out);
    }
    // De-dupe by (name, dev), keeping first occurrence.
    let mut seen = std::collections::HashSet::new();
    out.retain(|d| seen.insert((d.name.clone(), d.dev)));
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn parse_json_manifest(v: &serde_json::Value, out: &mut Vec<Dependency>) {
    for (dev, key) in [(false, "dependencies"), (true, "devDependencies")] {
        if let Some(map) = v.get(key).and_then(|d| d.as_object()) {
            for (name, ver) in map {
                let version = ver.as_str().unwrap_or("").to_string();
                out.push(Dependency {
                    name: name.clone(),
                    version,
                    dev,
                });
            }
        }
    }
}

fn parse_cargo_manifest(text: &str, out: &mut Vec<Dependency>) {
    #[derive(PartialEq)]
    enum Sec {
        Dep,
        Dev,
        Other,
    }
    let mut sec = Sec::Dep;
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with('[') && line.ends_with(']') {
            sec = if line.contains("dev-dependencies") {
                Sec::Dev
            } else if line.contains("dependencies") {
                Sec::Dep
            } else {
                Sec::Other
            };
            continue;
        }
        if sec == Sec::Other {
            continue;
        }
        let Some(eq) = line.find('=') else { continue };
        let name = line[..eq].trim().to_string();
        if name.is_empty() || name.contains(' ') || name.contains('.') {
            continue;
        }
        let rhs = line[eq + 1..].trim();
        let version = rhs
            .split_once('"')
            .and_then(|(_, rest)| rest.split_once('"'))
            .map(|(v, _)| v.to_string())
            .unwrap_or_default();
        out.push(Dependency {
            name,
            version,
            dev: sec == Sec::Dev,
        });
    }
}

fn parse_go_manifest(text: &str, out: &mut Vec<Dependency>) {
    let mut in_require = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with("require") && line.contains("(") {
            in_require = true;
            continue;
        }
        if line == ")" {
            in_require = false;
            continue;
        }
        if !in_require && !line.starts_with("require") {
            continue;
        }
        let mut parts = line.split_whitespace();
        let first = parts.next().unwrap_or("");
        if first == "require" {
            continue;
        }
        if first.is_empty() {
            continue;
        }
        let name = first.rsplit('/').next().unwrap_or(first).to_string();
        let version = parts.next().unwrap_or("").to_string();
        out.push(Dependency {
            name,
            version,
            dev: false,
        });
    }
}

fn parse_requirements(text: &str, out: &mut Vec<Dependency>) {
    for raw in text.lines() {
        let mut line = raw.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        // pyproject wrapper: `dependencies = [ "a==1" ]`
        if let Some(eq) = line.find('=') {
            let is_version_op = line[..eq].contains(['=', '>', '<', '~', '!'])
                || line[eq + 1..]
                    .chars()
                    .next()
                    .map(|c| matches!(c, '=' | '>' | '<' | '~' | '!'))
                    .unwrap_or(false);
            if !is_version_op {
                line = line[eq + 1..].trim();
            }
        }
        // strip bundling brackets and quotes
        while line.starts_with('[') {
            line = &line[1..];
        }
        while line.ends_with(']') {
            line = &line[..line.len() - 1];
        }
        let line = line.trim().trim_matches('"').trim().trim_matches(',');
        if line.is_empty() {
            continue;
        }
        // split at the first version operator (>=, ==, ~=, ...)
        let mut split_at = None;
        for (i, b) in line.bytes().enumerate() {
            if matches!(b, b'=' | b'>' | b'<' | b'~' | b'!') {
                split_at = Some(i);
                break;
            }
        }
        let (name, version) = match split_at {
            Some(i) => (
                &line[..i],
                line[i..]
                    .trim_start_matches(['=', '>', '<', '~', '!'])
                    .trim()
                    .to_string(),
            ),
            None => (line, String::new()),
        };
        let name = match name.find('[') {
            Some(b) => &name[..b],
            None => name,
        };
        let name = name.trim().trim_matches('"').trim_matches(',');
        let has_relation = !version.is_empty() || name.contains('_') || name.contains('-');
        if name.is_empty() || name.starts_with('-') || name.len() < 2 || !has_relation {
            continue;
        }
        out.push(Dependency {
            name: name.to_string(),
            version,
            dev: false,
        });
    }
}

fn detect_build_commands(texts: &[String], root: &Path) -> Vec<String> {
    let mut cmds = Vec::new();
    if root.join("Cargo.toml").is_file() {
        cmds.push("cargo build".to_string());
    }
    if root.join("go.mod").is_file() {
        cmds.push("go build ./...".to_string());
    }
    for t in texts {
        if t.contains("\"scripts\"") && t.contains("build") && t.contains("\"name\"") {
            let runner = if root.join("pnpm-lock.yaml").exists() {
                "pnpm"
            } else if root.join("yarn.lock").exists() {
                "yarn"
            } else {
                "npm"
            };
            cmds.push(format!("{runner} run build"));
            break;
        }
    }
    cmds
}

fn detect_important_dirs(files: &[RepoFile]) -> Vec<String> {
    let mut dirs: BTreeMap<String, usize> = BTreeMap::new();
    for f in files {
        let first = f
            .rel
            .components()
            .next()
            .and_then(|c| c.as_os_str().to_str());
        if let Some(first) = first {
            if IMPORTANT_DIR_NAMES.contains(&first) || first.starts_with('.') {
                *dirs.entry(first.to_string()).or_default() += 1;
            }
        }
    }
    let mut v: Vec<(String, usize)> = dirs.into_iter().collect();
    v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    v.into_iter().take(8).map(|(d, _)| d).collect()
}

fn git_report(root: &Path) -> GitReport {
    let mut g = GitReport::absent();
    if !root.join(".git").exists() {
        return g;
    }
    g.present = true;
    if let Some(branch) = git_capture(root, &["branch", "--show-current"]) {
        g.branch = branch.trim().to_string();
    }
    if let Some(porcelain) = git_capture(root, &["status", "--porcelain"]) {
        for line in porcelain.lines() {
            let b = line.as_bytes();
            if b.len() < 2 {
                continue;
            }
            let (x, y) = (b[0] as char, b[1] as char);
            if x == '?' && y == '?' {
                g.untracked += 1;
            } else if x == 'U' || y == 'U' {
                g.conflicts += 1;
            } else if y != ' ' {
                g.unstaged += 1;
            } else {
                g.staged += 1;
            }
        }
    }
    g
}

fn git_capture(root: &Path, args: &[&str]) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn cargo_project() -> TempDir {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("src/db")).unwrap();
        std::fs::create_dir_all(root.join("tests")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname=\"demo\"\n[dependencies]\naxum=\"0.7\"\nsqlx={features=[\"postgres\"]}\nserde=\"1\"\n",
        )
        .unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(root.join("src/auth.rs"), "pub fn login() {}").unwrap();
        std::fs::write(root.join("src/db/users.rs"), "pub struct User;").unwrap();
        std::fs::write(root.join("tests/auth_test.rs"), "#[test] fn t() {}").unwrap();
        tmp
    }

    #[test]
    fn rust_project_detects_full_stack() {
        let tmp = cargo_project();
        let f = analyze_repo(tmp.path());
        assert!(f.languages.iter().any(|(l, _)| l == "Rust"));
        assert!(f.frameworks.iter().any(|x| x == "Axum"));
        assert!(f.databases.iter().any(|d| d == "PostgreSQL"));
        assert!(f.databases.iter().any(|d| d == "SQLx"));
        assert!(f.test_commands.iter().any(|c| c == "cargo test"));
        assert!(f.entry_points.iter().any(|e| e == "src/main.rs"));
        assert!(f.source_count > 0);
        assert!(f.test_count > 0);
        assert!(f.package_managers.iter().any(|m| m == "cargo"));
    }

    #[test]
    fn node_project_detects_react_ts() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("package.json"),
            r#"{"name":"ui","dependencies":{"react":"18","react-dom":"18"},"devDependencies":{"vite":"5","typescript":"5"},"scripts":{"build":"vite build"}}"#,
        )
        .unwrap();
        std::fs::write(root.join("src/main.tsx"), "export const App = () => null;").unwrap();
        std::fs::write(root.join("src/index.css"), "body{}").unwrap();

        let f = analyze_repo(root);
        assert!(f.languages.iter().any(|(l, _)| l == "TypeScript (React)"));
        assert!(f.frameworks.iter().any(|fr| fr == "React"));
        assert!(f.frameworks.iter().any(|fr| fr == "Vite"));
        assert!(f.package_managers.iter().any(|m| m == "npm"));
        assert!(f.source_count >= 2);
    }

    #[test]
    fn probe_finds_existing_auth_and_avoids_rewrite() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("src/auth")).unwrap();
        std::fs::create_dir_all(root.join("src/db")).unwrap();
        std::fs::write(root.join("src/auth/login.rs"), "// x").unwrap();
        std::fs::write(root.join("src/auth/register.rs"), "// x").unwrap();
        std::fs::write(root.join("src/db/users.rs"), "struct User;").unwrap();
        std::fs::write(root.join("src/browser_util.rs"), "// unrelated").unwrap();

        let f = analyze_repo(root);
        let report = f.probe("add user authentication to the api");
        // "authentication" topic (user → users.rs, auth → auth/) must surface.
        assert!(
            report.hits.iter().any(|h| h.label == "authentication"),
            "hits: {:?}",
            report
                .hits
                .iter()
                .map(|h| h.label.as_str())
                .collect::<Vec<_>>()
        );
        // The unrelated "browser" file must not be swept in by "user".
        let auth = report
            .hits
            .iter()
            .find(|h| h.label == "authentication")
            .unwrap();
        assert!(auth.files.iter().any(|f| f.contains("users.rs")));
        assert!(!auth.files.iter().any(|f| f.contains("browser_util")));
    }

    #[test]
    fn probe_respects_segment_boundaries() {
        assert!(segment_hit("src/api/auth.rs", "auth"));
        assert!(segment_hit("src/db/users.rs", "user"));
        assert!(!segment_hit("src/browser_util.rs", "user"));
        assert!(!segment_hit("src/main.rs", "auth"));
    }

    #[test]
    fn git_report_absent_for_non_repo() {
        let tmp = TempDir::new().unwrap();
        let g = git_report(tmp.path());
        assert!(!g.present);
        assert_eq!(g.one_line(), "not a git repo");
    }

    #[cfg(any(unix, windows))]
    fn try_symlink_dir(target: &Path, link: &Path) -> std::io::Result<()> {
        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_dir(target, link)
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(target, link)
        }
    }

    #[test]
    fn collect_all_does_not_follow_symlinks_outside_root() {
        // Regression: a symlink/junction inside the project pointing at an
        // unrelated tree must not widen the scan — the "scan of the current
        // directory" would otherwise leak in files from a false broader
        // environment on the same disk.
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("project");
        let outside = tmp.path().join("unrelated");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(outside.join("secret.rs"), "// must not leak").unwrap();

        let link = root.join("src").join("escaped");
        if try_symlink_dir(&outside, &link).is_err() {
            // Symlinks unavailable here (e.g. Windows without Developer Mode
            // or an admin-elevated shell) — nothing to verify, skip.
            return;
        }

        let files = collect_all(&root);
        assert!(files.iter().any(|f| f.rel.ends_with("main.rs")));
        assert!(
            !files.iter().any(|f| f.rel.ends_with("secret.rs")),
            "a symlink pointing outside the project must not be followed: {files:?}"
        );
    }

    #[test]
    fn detects_dependencies_from_manifests() {
        let cargo = "[package]\nname=\"x\"\n[dependencies]\nserde = \"1\"\naxum = { version = \"0.7\", features=[\"macros\"] }\n[dev-dependencies]\ntokio = \"1\"\n";
        let npm = r#"{"name":"x","dependencies":{"react":"18"},"devDependencies":{"vite":"^5","rollup":"4"}}"#;
        let req = "fastapi==0.99\nuvicorn[standard]>=0.23\n";
        let deps = detect_dependencies(&[cargo.to_string(), npm.to_string(), req.to_string()]);
        let names: Vec<&str> = deps.iter().map(|d| d.name.as_str()).collect();
        for want in [
            "serde", "axum", "tokio", "react", "vite", "rollup", "fastapi", "uvicorn",
        ] {
            assert!(names.contains(&want), "missing {want} in {names:?}");
        }
        let axum = deps.iter().find(|d| d.name == "axum").unwrap();
        assert_eq!(axum.version, "0.7");
        assert!(!axum.dev);
        assert!(deps.iter().any(|d| d.name == "tokio" && d.dev));
    }
}
