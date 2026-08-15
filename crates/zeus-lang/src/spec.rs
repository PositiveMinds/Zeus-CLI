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

/// Convenient accessor: the five dev commands as
/// `(build, test, lint, lint_fix, format)` arg lists
/// (empty = no standard command).
pub type DevCommands<'a> = (
    &'a [&'static str],
    &'a [&'static str],
    &'a [&'static str],
    &'a [&'static str],
    &'a [&'static str],
);

pub fn dev_commands(lang: Language) -> DevCommands<'static> {
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
        lint_fix: &[
            "cargo",
            "clippy",
            "--fix",
            "--allow-dirty",
            "--allow-staged",
        ],
        format: &["cargo", "fmt"],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::Python,
        display_name: "Python",
        exts: &["py"],
        markers: &[
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
        ],
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
        language: Language::Html,
        display_name: "HTML",
        exts: &["html", "htm", "xhtml"],
        markers: &[],
        build: &[],
        test: &[],
        lint: &["npx", "html-validate", "."],
        lint_fix: &["npx", "html-validate", "--fix", "."],
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
    LangSpec {
        language: Language::Clojure,
        display_name: "Clojure",
        exts: &["clj", "cljs", "cljc", "edn"],
        markers: &["deps.edn", "project.clj", "shadow-cljs.edn"],
        build: &["clojure", "-M", "-e", "(compile 'core)"],
        test: &["clojure", "-M:test"],
        lint: &["clj-kondo", "--lint", "src"],
        lint_fix: &["cljstyle", "fix", "."],
        format: &["cljstyle", "replace", "."],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::Julia,
        display_name: "Julia",
        exts: &["jl"],
        markers: &["Project.toml"],
        build: &["julia", "--project=.", "-e", "using Pkg; Pkg.precompile()"],
        test: &["julia", "--project=.", "-e", "using Pkg; Pkg.test()"],
        lint: &[
            "julia",
            "--project=.",
            "-e",
            "using JuliaFormatter; format(\".\")",
        ],
        lint_fix: &[
            "julia",
            "--project=.",
            "-e",
            "using JuliaFormatter; format(\".\")",
        ],
        format: &[
            "julia",
            "--project=.",
            "-e",
            "using JuliaFormatter; format(\".\")",
        ],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::Perl,
        display_name: "Perl",
        exts: &["pl", "pm"],
        markers: &["cpanfile", "Makefile.PL", "dist.ini"],
        build: &["perl", "-c", "script/main.pl"],
        test: &["prove", "-l", "t"],
        lint: &["perlcritic", "."],
        lint_fix: &["perltidy", "-b", "{file}"],
        format: &["perltidy", "-b", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::OCaml,
        display_name: "OCaml",
        exts: &["ml", "mli"],
        markers: &["dune-project", "*.opam"],
        build: &["dune", "build"],
        test: &["dune", "runtest"],
        lint: &["dune", "build", "@lint"],
        lint_fix: &["ocamlformat", "--inplace", "{file}"],
        format: &["ocamlformat", "--inplace", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Nim,
        display_name: "Nim",
        exts: &["nim"],
        markers: &["*.nimble", "nim.cfg"],
        build: &["nim", "c", "src/main.nim"],
        test: &["nim", "test"],
        lint: &["nim", "check", "."],
        lint_fix: &["nimpretty", "{file}"],
        format: &["nimpretty", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Crystal,
        display_name: "Crystal",
        exts: &["cr"],
        markers: &["shard.yml"],
        build: &["crystal", "build", "src/main.cr"],
        test: &["crystal", "spec"],
        lint: &["ameba"],
        lint_fix: &["ameba", "--fix"],
        format: &["crystal", "tool", "format", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Groovy,
        display_name: "Groovy",
        exts: &["groovy", "gradle"],
        markers: &["build.gradle", "settings.gradle", "Jenkinsfile"],
        build: &["gradle", "build"],
        test: &["gradle", "test"],
        lint: &["gradle", "check"],
        lint_fix: &[],
        format: &["gradle", "spotlessApply"],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::Fortran,
        display_name: "Fortran",
        exts: &["f90", "f95", "f03", "f08", "f"],
        markers: &["CMakeLists.txt", "Makefile"],
        build: &["cmake", "--build", "."],
        test: &["ctest"],
        lint: &["fprettify", "--check"],
        lint_fix: &["fprettify", "{file}"],
        format: &["fprettify", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Shell,
        display_name: "Shell",
        exts: &["sh", "bash", "zsh"],
        markers: &[],
        build: &["bash", "-n", "main.sh"],
        test: &["shellcheck", "**/*.sh"],
        lint: &["shellcheck", "**/*.sh"],
        lint_fix: &["shfmt", "-w", "{file}"],
        format: &["shfmt", "-w", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::PowerShell,
        display_name: "PowerShell",
        exts: &["ps1", "psm1", "psd1"],
        markers: &["*.psd1"],
        build: &["pwsh", "-NoProfile", "-File", "main.ps1"],
        test: &["Pester", "Run"],
        lint: &["Invoke-ScriptAnalyzer", "-Path", "."],
        lint_fix: &[],
        format: &[],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::Erlang,
        display_name: "Erlang",
        exts: &["erl", "hrl"],
        markers: &["rebar.config", "erlang.mk"],
        build: &["rebar3", "compile"],
        test: &["rebar3", "eunit"],
        lint: &["rebar3", "dialyzer"],
        lint_fix: &["erlfmt", "-w", "{file}"],
        format: &["erlfmt", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::FSharp,
        display_name: "F#",
        exts: &["fs", "fsx", "fsi"],
        markers: &["*.fsproj"],
        build: &["dotnet", "build"],
        test: &["dotnet", "test"],
        lint: &["dotnet", "fantomas", "--check"],
        lint_fix: &["dotnet", "fantomas"],
        format: &["dotnet", "fantomas"],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::ObjectiveC,
        display_name: "Objective-C",
        exts: &["m"],
        markers: &["Podfile", "*.xcodeproj"],
        build: &["xcodebuild", "-scheme", "app", "build"],
        test: &["xcodebuild", "-scheme", "app", "test"],
        lint: &["clang", "--analyze", "{file}"],
        lint_fix: &["clang-format", "-i", "{file}"],
        format: &["clang-format", "-i", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::V,
        display_name: "V",
        exts: &["v", "vsh"],
        markers: &["v.mod"],
        build: &["v", "."],
        test: &["v", "test", "."],
        lint: &["v", "vet", "."],
        lint_fix: &["v", "fmt", "-w", "{file}"],
        format: &["v", "fmt", "-w", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Ada,
        display_name: "Ada",
        exts: &["adb", "ads", "ada"],
        markers: &["*.gpr"],
        build: &["gnatmake", "-P", "project.gpr"],
        test: &["gnattest"],
        lint: &["gnatcheck"],
        lint_fix: &["gnatpp"],
        format: &["gnatpp", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Pascal,
        display_name: "Pascal",
        exts: &["pas", "pp", "lpr", "dpr"],
        markers: &["*.lpr", "*.dpr"],
        build: &["fpc", "main.pas"],
        test: &["fpc", "test_main.pas"],
        lint: &[],
        lint_fix: &["ptop", "-l", "0", "{file}"],
        format: &["ptop", "-l", "0", "{file}"],
        format_style: FormatStyle::PerFile,
    },
    LangSpec {
        language: Language::Lisp,
        display_name: "Lisp",
        exts: &["lisp", "lsp", "cl"],
        markers: &["*.asd"],
        build: &["sbcl", "--load", "main.lisp"],
        test: &["sbcl", "--load", "test.lisp"],
        lint: &[],
        lint_fix: &[],
        format: &["lisp-format"],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::Scheme,
        display_name: "Scheme",
        exts: &["scm", "ss", "rkt"],
        markers: &["info.rkt"],
        build: &["racket", "main.rkt"],
        test: &["raco", "test"],
        lint: &["raco", "lint"],
        lint_fix: &["raco", "fmt"],
        format: &["raco", "fmt"],
        format_style: FormatStyle::Project,
    },
    LangSpec {
        language: Language::VisualBasic,
        display_name: "Visual Basic",
        exts: &["vb"],
        markers: &["*.vbproj"],
        build: &["dotnet", "build"],
        test: &["dotnet", "test"],
        lint: &["dotnet", "format", "analyzers"],
        lint_fix: &["dotnet", "format"],
        format: &["dotnet", "format", "whitespace"],
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
            if matches!(lang, Language::Html) {
                // Markup has no compile or test step.
                continue;
            }
            assert!(!s.build.is_empty(), "{lang:?} needs a build command");
            assert!(!s.test.is_empty(), "{lang:?} needs a test command");
        }
    }
}
