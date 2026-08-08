//! Per-language dev-specs: display name, file extensions, the manifest files
//! that identify a project, and the standard build / test / lint / format
//! commands.
//!
//! Each command is a `Vec` of argv words (spawn-ready, no shell needed).
//! A format command may contain the `{file}` placeholder, substituted with
//! an absolute path when the caller wants to format one target; commands
//! without `{file}` are whole-project formatters.

use crate::detect::Language;

/// Replacement token in per-file format/lint commands.
pub const FILE_PLACEHOLDER: &str = "{file}";

/// How a format command should be invoked for a single target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatStyle {
    /// The command takes the target file path: `gofmt -w {file}`.
    PerFile,
    /// The whole project is formatted (`cargo fmt`, `dotnet format`).
    Project,
}

/// Everything zeus knows about one language.
pub struct LangSpec {
    pub language: Language,
    /// Human display name, e.g. "Rust", "TypeScript".
    pub display_name: &'static str,
    /// Files usually part of this language.
    pub exts: &'static [&'static str],
    /// Manifest file names used to identify a project root.
    pub markers: &'static [&'static str],
    /// Standard build command (argv).
    pub build: &'static [&'static str],
    /// Standard test command (argv).
    pub test: &'static [&'static str],
    /// Standard lint command (argv).
    pub lint: &'static [&'static str],
    /// Lint in auto-fix mode — repair what the linter can fix on its own
    /// (empty = no supported auto-fixer). Format-style commands with
    /// `{file}` are per-file.
    pub lint_fix: &'static [&'static str],
    /// Standard format command (argv); may contain `{file}`.
    pub format: &'static [&'static str],
    /// Whether `{file}` in `format` means "one file".
    pub format_style: FormatStyle,
}

/// The lookup table. Languages are ordered roughly by popularity.
pub fn all_specs() -> &'static [LangSpec] {
    SPECS
}

/// Fetch the spec for a language (always exists).
pub fn spec(lang: Language) -> &'static LangSpec {
    SPECS
        .iter()
        .find(|s| s.language == lang)
        .expect("spec table covers every Language variant")
}

/// Convenient accessor: the four dev commands as
/// `(build, test, lint, lint_fix, format)` arg lists
/// (empty = no standard command).
pub fn dev_commands(
    lang: Language,
) -> (
    &'static [&'static str],
    &'static [&'static str],
    &'static [&'static str],
    &'static [&'static str],
    &'static [&'static str],
) {
    let s = spec(lang);
    (s.build, s.test, s.lint, s.lint_fix, s.format)
}

const SPECS: &[LangSpec] = &[
    LangSpec {
        language: Language::Rust,
        display_name: "Rust",
        exts: &["rs"],
        markers: &["Cargo.toml"],
        build: &["cargo", "build"],
        test: &["cargo", "test"],
        lint: &["cargo", "clippy", "--all-targets", "--", "-D", "warnings"],
        lint_fix: &["cargo", "clippy", "--fix", "--allow-dirty", "--allow-staged"],
        format: &["cargo", "fmt"],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::Python,
        display_name: "Python",
        exts: &["py"],
        markers: &["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"],
        build: &["python", "-m", "build"],
        test: &["python", "-m", "pytest", "-q"],
        lint: &["ruff", "check", "."],
        lint_fix: &["ruff", "check", "--fix", "."],
        format: &["ruff", "format", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Go,
        display_name: "Go",
        exts: &["go"],
        markers: &["go.mod"],
        build: &["go", "build", "."],
        test: &["go", "test", "./..."],
        lint: &["go", "vet", "./..."],
        lint_fix: &["gofmt", "-w", "."],
        format: &["gofmt", "-w", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::TypeScript,
        display_name: "TypeScript",
        exts: &["ts", "tsx", "mts", "cts"],
        markers: &["package.json", "tsconfig.json"],
        build: &["tsc", "-p", "."],
        test: &["npm", "test"],
        lint: &["eslint", "."],
        lint_fix: &["eslint", "--fix", "."],
        format: &["npx", "prettier", "--write", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::JavaScript,
        display_name: "JavaScript",
        exts: &["js", "jsx", "mjs", "cjs"],
        markers: &["package.json"],
        build: &["npm", "run", "build"],
        test: &["npm", "test"],
        lint: &["eslint", "."],
        lint_fix: &["eslint", "--fix", "."],
        format: &["npx", "prettier", "--write", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Java,
        display_name: "Java",
        exts: &["java"],
        markers: &["pom.xml", "build.gradle", "settings.gradle"],
        build: &["mvn", "-q", "-DskipTests", "compile"],
        test: &["mvn", "-q", "test"],
        lint: &["mvn", "-q", "checkstyle:check"],
        lint_fix: &[],
        format: &["mvn", "spotless:apply"],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::Kotlin,
        display_name: "Kotlin",
        exts: &["kt", "kts"],
        markers: &["build.gradle.kts", "settings.gradle.kts"],
        build: &["./gradlew", "build"],
        test: &["./gradlew", "test"],
        lint: &["ktlint"],
        lint_fix: &["ktlint", "--format"],
        format: &["ktlint", "--format"],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::CSharp,
        display_name: "C#",
        exts: &["cs", "csx"],
        markers: &["*.sln", "*.csproj"],
        build: &["dotnet", "build"],
        test: &["dotnet", "test"],
        lint: &["dotnet", "format", "analyzers"],
        lint_fix: &["dotnet", "format"],
        format: &["dotnet", "format", "whitespace"],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::Swift,
        display_name: "Swift",
        exts: &["swift"],
        markers: &["Package.swift"],
        build: &["swift", "build"],
        test: &["swift", "test"],
        lint: &["swiftlint", "lint"],
        lint_fix: &["swiftlint", "--fix"],
        format: &["swiftformat"],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::Cpp,
        display_name: "C / C++",
        exts: &["c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx"],
        markers: &["CMakeLists.txt", "Makefile", "meson.build"],
        build: &["cmake", "--build", "."],
        test: &["ctest"],
        lint: &["cppcheck", "."],
        lint_fix: &["clang-format", "-i", "{file}"],
        format: &["clang-format", "-i", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Php,
        display_name: "PHP",
        exts: &["php"],
        markers: &["composer.json"],
        build: &["composer", "install"],
        test: &["vendor/bin/phpunit"],
        lint: &["phpcs"],
        lint_fix: &["phpcbf"],
        format: &["php-cs-fixer", "fix", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Ruby,
        display_name: "Ruby",
        exts: &["rb"],
        markers: &["Gemfile", "*.gemspec", "Rakefile"],
        build: &["bundle", "exec", "rake", "build"],
        test: &["bundle", "exec", "rspec"],
        lint: &["rubocop"],
        lint_fix: &["rubocop", "-A"],
        format: &["rubocop", "-A", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Dart,
        display_name: "Dart",
        exts: &["dart"],
        markers: &["pubspec.yaml"],
        build: &["dart", "compile", "exe", "bin/main.dart"],
        test: &["dart", "test"],
        lint: &["dart", "analyze"],
        lint_fix: &["dart", "format", "."],
        format: &["dart", "format", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Scala,
        display_name: "Scala",
        exts: &["scala", "sc"],
        markers: &["build.sbt"],
        build: &["sbt", "compile"],
        test: &["sbt", "test"],
        lint: &["scalafmt", "--check"],
        lint_fix: &["scalafmt"],
        format: &["scalafmt"],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::Elixir,
        display_name: "Elixir",
        exts: &["ex", "exs"],
        markers: &["mix.exs"],
        build: &["mix", "compile"],
        test: &["mix", "test"],
        lint: &["mix", "credo", "strict"],
        lint_fix: &["mix", "format"],
        format: &["mix", "format", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::R,
        display_name: "R",
        exts: &["r", "R"],
        markers: &["DESCRIPTION"],
        build: &["R", "CMD", "build", "."],
        test: &["R", "CMD", "check", "--no-manual"],
        lint: &[],
        lint_fix: &[],
        format: &[],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::Zig,
        display_name: "Zig",
        exts: &["zig"],
        markers: &["build.zig"],
        build: &["zig", "build"],
        test: &["zig", "build", "test"],
        lint: &["zig", "fmt", "--check", "."],
        lint_fix: &["zig", "fmt", "."],
        format: &["zig", "fmt", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Haskell,
        display_name: "Haskell",
        exts: &["hs", "lhs"],
        markers: &["*.cabal", "stack.yaml", "package.yaml"],
        build: &["cabal", "build"],
        test: &["cabal", "test"],
        lint: &["hlint"],
        lint_fix: &["ormolu", "--mode", "inplace"],
        format: &["ormolu", "--mode", "inplace", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Lua,
        display_name: "Lua",
        exts: &["lua"],
        markers: &["*.rockspec"],
        build: &["luac", "-p", "main.lua"],
        test: &["busted"],
        lint: &["luacheck", "."],
        lint_fix: &["stylua", "."],
        format: &["stylua", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Solidity,
        display_name: "Solidity",
        exts: &["sol"],
        markers: &["foundry.toml", "hardhat.config.ts", "hardhat.config.js"],
        build: &["forge", "build"],
        test: &["forge", "test"],
        lint: &["forge", "fmt", "--check"],
        lint_fix: &["forge", "fmt"],
        format: &["forge", "fmt"],
        format_style: FormatStyle::Project,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::Language;

    #[test]
    fn every_language_has_a_spec_and_an_ext() {
        for lang in Language::ALL {
            let s = spec(*lang);
            assert!(!s.exts.is_empty(), "{} needs extensions", s.display_name);
        }
    }

    #[test]
    fn build_and_test_commands_are_always_present() {
        for (lang, s) in all_specs().iter().map(|s| (s.language, s)) {
            assert!(!s.build.is_empty(), "{lang:?} needs a build command");
            assert!(!s.test.is_empty(), "{lang:?} needs a test command");
        }
    }
}