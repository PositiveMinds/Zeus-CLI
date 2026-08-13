//! Language detection: map a project root (via manifest files) or a file
//! (via its extension) to one of zeus's supported languages.

use std::path::Path;

/// A major programming language zeus knows how to work with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    Go,
    TypeScript,
    JavaScript,
    Java,
    Kotlin,
    CSharp,
    Swift,
    /// C and C++ (frequently hard to tell apart from a directory alone).
    Cpp,
    Php,
    Ruby,
    Dart,
    Scala,
    Elixir,
    R,
    Zig,
    Haskell,
    Lua,
    Solidity,
}

impl Language {
    pub const ALL: &'static [Language] = &[
        Language::Rust,
        Language::Python,
        Language::Go,
        Language::TypeScript,
        Language::JavaScript,
        Language::Java,
        Language::Kotlin,
        Language::CSharp,
        Language::Swift,
        Language::Cpp,
        Language::Php,
        Language::Ruby,
        Language::Dart,
        Language::Scala,
        Language::Elixir,
        Language::R,
        Language::Zig,
        Language::Haskell,
        Language::Lua,
        Language::Solidity,
    ];

    /// Parse a language from a CLI-ish name ("rust", "ts", "c#", "c++",
    /// "go"...) — case and separators don't matter, and any registered file
    /// extension is accepted too.
    pub fn from_name(name: &str) -> Option<Language> {
        let key = name.trim().to_ascii_lowercase().replace('_', "-");
        let lang = match key.as_str() {
            "rust" | "rs" => Language::Rust,
            "python" | "py" => Language::Python,
            "go" | "golang" => Language::Go,
            "typescript" | "ts" => Language::TypeScript,
            "tsx" => Language::TypeScript,
            "javascript" | "js" | "node" | "jsx" => Language::JavaScript,
            "java" => Language::Java,
            "kotlin" | "kt" | "kts" => Language::Kotlin,
            "csharp" | "cs" | "c#" => Language::CSharp,
            "swift" => Language::Swift,
            "c" | "c++" | "cpp" | "cxx" | "cc" | "h" | "hpp" => Language::Cpp,
            "php" => Language::Php,
            "ruby" | "rb" => Language::Ruby,
            "dart" => Language::Dart,
            "scala" => Language::Scala,
            "elixir" | "ex" | "exs" => Language::Elixir,
            "r" => Language::R,
            "zig" => Language::Zig,
            "haskell" | "hs" => Language::Haskell,
            "lua" => Language::Lua,
            "solidity" | "sol" => Language::Solidity,
            _ => return None,
        };
        Some(lang)
    }

    /// Language for a file extension, if any of the specs claims it.
    pub fn from_ext(ext: &str) -> Option<Language> {
        let ext = ext.trim_start_matches('.').to_ascii_lowercase();
        Language::ALL
            .iter()
            .copied()
            .find(|l| crate::spec::spec(*l).exts.iter().any(|e| *e == ext))
    }
}

/// Detect the primary language of a project directory. Manifests are checked
/// first (most reliable), then the directory's own contents, then the whole
/// tree falls back to a bounded extension census for projects with no
/// manifest at all.
pub fn detect_project(root: &Path) -> Option<Language> {
    if !root.is_dir() {
        return None;
    }
    let file = |name: &str| root.join(name).is_file();

    // Definitive, language-specific manifests first.
    if file("Cargo.toml") {
        return Some(Language::Rust);
    }
    if file("go.mod") {
        return Some(Language::Go);
    }
    if file("pyproject.toml") || file("setup.py") || file("setup.cfg") || file("requirements.txt") {
        return Some(Language::Python);
    }
    if file("composer.json") {
        return Some(Language::Php);
    }
    if file("Gemfile") || contains_file(root, &["gemspec"], 1) {
        return Some(Language::Ruby);
    }
    if file("Package.swift") {
        return Some(Language::Swift);
    }
    if file("pubspec.yaml") {
        return Some(Language::Dart);
    }
    if file("build.gradle.kts") || file("settings.gradle.kts") {
        return Some(Language::Kotlin);
    }
    if file("build.sbt") {
        return Some(Language::Scala);
    }
    if file("mix.exs") {
        return Some(Language::Elixir);
    }
    if file("DESCRIPTION") {
        return Some(Language::R);
    }
    if file("build.zig") {
        return Some(Language::Zig);
    }
    if file("stack.yaml") || file("package.yaml") || contains_file(root, &["cabal"], 2) {
        return Some(Language::Haskell);
    }
    if file("foundry.toml") || file("hardhat.config.ts") || file("hardhat.config.js") {
        return Some(Language::Solidity);
    }
    if file("Makefile") {
        return Some(Language::Cpp);
    }
    // Gradle could be Java or Kotlin; we already checked Kotlin-specific above.
    if file("pom.xml") || file("build.gradle") || file("settings.gradle") {
        return Some(Language::Java);
    }
    if contains_ext(root, &["csproj", "sln"], 2) {
        return Some(Language::CSharp);
    }

    // package.json lives in both JS and TS projects; disambiguate.
    if file("package.json") {
        let has_ts_config = file("tsconfig.json");
        let has_ts_sources = contains_ext(root, &["ts", "tsx"], 2);
        return Some(if has_ts_config || has_ts_sources {
            Language::TypeScript
        } else {
            Language::JavaScript
        });
    }

    // No manifest: whichever supported extension dominates the tree wins.
    ext_census(root, 3)
}

/// Detect the language of a single source file from its extension.
pub fn detect_source(path: &Path) -> Option<Language> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    Language::from_ext(ext)
}

/// Walk `root` up to `max_depth` levels collecting extensions; the most
/// frequent known-language extension decides the project's language.
fn ext_census(root: &Path, max_depth: usize) -> Option<Language> {
    let mut counts: Vec<(Language, usize)> =
        Language::ALL.iter().copied().map(|l| (l, 0)).collect();
    let mut best: Option<Language> = None;
    let mut best_count = 0usize;

    for (path, depth) in walk(root, max_depth) {
        if depth > 0 && path.is_dir() {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext.is_empty() {
            continue;
        }
        if let Some(lang) = Language::from_ext(ext) {
            for (l, c) in counts.iter_mut() {
                if *l == lang {
                    *c += 1;
                }
            }
        }
    }
    for (l, c) in counts {
        if c > best_count {
            best_count = c;
            best = Some(l);
        }
    }
    best
}

/// Return `(path, depth)` for every entry under `root`, bounded by
/// `max_depth`. Hidden dirs and common build/cache trees are skipped.
pub fn walk(root: &Path, max_depth: usize) -> Vec<(std::path::PathBuf, usize)> {
    use std::collections::VecDeque;

    const SKIP_DIR_NAMES: &[&str] = &[
        ".git",
        ".hg",
        ".svn",
        ".agent",
        "node_modules",
        "target",
        "dist",
        "build",
        ".cargo",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".cache",
        ".idea",
        ".vscode",
        "coverage",
    ];

    let mut out = Vec::new();
    let mut queue: VecDeque<(std::path::PathBuf, usize)> = VecDeque::new();
    queue.push_back((root.to_path_buf(), 0));
    while let Some((path, depth)) = queue.pop_front() {
        if depth > max_depth {
            continue;
        }
        out.push((path.clone(), depth));
        if depth == max_depth {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                let name = child.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.starts_with('.') || SKIP_DIR_NAMES.contains(&name) {
                    continue;
                }
                queue.push_back((child, depth + 1));
            } else {
                out.push((child, depth + 1));
            }
        }
    }
    out
}

fn contains_ext(root: &Path, exts: &[&str], depth: usize) -> bool {
    for (path, _) in walk(root, depth) {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if exts.contains(&ext) {
            return true;
        }
    }
    false
}

/// True if any *file* under `root` (bounded depth) has one of `exts`.
fn contains_file(root: &Path, exts: &[&str], depth: usize) -> bool {
    for (path, _) in walk(root, depth) {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if exts.contains(&ext) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write(root: &std::path::Path, rel: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, "").unwrap();
    }

    #[test]
    fn detects_from_manifest_files() {
        let cases = [
            ("Cargo.toml", Language::Rust),
            ("pyproject.toml", Language::Python),
            ("go.mod", Language::Go),
            ("composer.json", Language::Php),
            ("Package.swift", Language::Swift),
            ("pubspec.yaml", Language::Dart),
            ("build.sbt", Language::Scala),
            ("mix.exs", Language::Elixir),
            ("build.zig", Language::Zig),
            ("Cargo.toml", Language::Rust),
        ];
        for (manifest, expected) in cases {
            let tmp = TempDir::new().unwrap();
            write(tmp.path(), manifest);
            assert_eq!(detect_project(tmp.path()), Some(expected), "for {manifest}");
        }
    }

    #[test]
    fn differentiates_ts_vs_js_via_tsconfig() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "package.json");
        write(tmp.path(), "src/index.ts");
        assert_eq!(detect_project(tmp.path()), Some(Language::TypeScript));

        let tmp2 = TempDir::new().unwrap();
        write(tmp2.path(), "package.json");
        write(tmp2.path(), "src/index.js");
        assert_eq!(detect_project(tmp2.path()), Some(Language::JavaScript));
    }

    #[test]
    fn detects_java_after_explicit_kotlin_checks() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "pom.xml");
        assert_eq!(detect_project(tmp.path()), Some(Language::Java));
    }

    #[test]
    fn ext_census_finds_project_without_manifests() {
        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "main.py");
        write(tmp.path(), "helper.py");
        write(tmp.path(), "util.py");
        assert_eq!(detect_project(tmp.path()), Some(Language::Python));
    }

    #[test]
    fn detects_source_from_extension() {
        assert_eq!(
            detect_source(std::path::Path::new("a/main.rs")),
            Some(Language::Rust)
        );
        assert_eq!(
            detect_source(std::path::Path::new("b.tsx")),
            Some(Language::TypeScript)
        );
        assert_eq!(detect_source(std::path::Path::new("c.xyz")), None);
    }

    #[test]
    fn from_name_accepts_common_aliases() {
        assert_eq!(Language::from_name("rust"), Some(Language::Rust));
        assert_eq!(Language::from_name("python"), Some(Language::Python));
        assert_eq!(Language::from_name("C#"), Some(Language::CSharp));
        assert_eq!(Language::from_name("C++"), Some(Language::Cpp));
        assert_eq!(Language::from_name("go"), Some(Language::Go));
        assert_eq!(Language::from_name("ts"), Some(Language::TypeScript));
        assert!(Language::from_name("brainfuck").is_none());
    }
}
