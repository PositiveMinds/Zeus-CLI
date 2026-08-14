//! Project scaffolding: `zeus project scaffold <lang> <name>` writes a
//! minimal, buildable skeleton for a supported language.

use crate::detect::Language;

/// Error type for scaffolding.
#[derive(Debug)]
pub enum ScaffoldError {
    NoTemplate(Language),
    Io(std::io::Error),
}

impl std::fmt::Display for ScaffoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScaffoldError::NoTemplate(l) => write!(f, "no scaffold template for {l:?}"),
            ScaffoldError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for ScaffoldError {}

/// Languages that have a scaffold template.
pub fn available_scaffold_languages() -> Vec<Language> {
    Language::ALL
        .iter()
        .copied()
        .filter(|l| !templates(*l).is_empty())
        .collect()
}

/// Write a freshly scaffolded project into `target` (created if needed) and
/// return the absolute paths written. `name` is the project / module name.
pub fn scaffold_project(
    lang: Language,
    name: &str,
    target: &std::path::Path,
) -> Result<Vec<std::path::PathBuf>, ScaffoldError> {
    let templates = templates(lang);
    if templates.is_empty() {
        return Err(ScaffoldError::NoTemplate(lang));
    }
    std::fs::create_dir_all(target).map_err(ScaffoldError::Io)?;
    let mut written = Vec::new();
    for (rel, contents) in templates {
        let path = target.join(render(rel, name));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(ScaffoldError::Io)?;
        }
        std::fs::write(&path, render(contents, name)).map_err(ScaffoldError::Io)?;
        written.push(path);
    }
    Ok(written)
}

/// Replace `{name}`-style placeholders in a template literal.
pub(crate) fn render(template: &str, name: &str) -> String {
    let values = [
        ("{name}", snake_case(name)),
        ("{snake}", snake_case(name)),
        ("{Pascal}", pascal_case(name)),
        ("{kebab}", kebab_case(name)),
        ("{Name}", pascal_case(name)),
    ];
    // Walk char-by-char so `$`-prefixed placeholders (JavaScript template
    // literals like `${name}`) are copied verbatim instead of having their
    // `{name}` swallowed by the naive replace.
    let mut out = String::with_capacity(template.len());
    let chars: Vec<char> = template.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == '$' && i + 1 < chars.len() && chars[i + 1] == '{' {
            let mut j = i + 2;
            while j < chars.len() && chars[j] != '}' {
                j += 1;
            }
            let end = (j + 1).min(chars.len());
            let literal: String = chars[i..end].iter().collect();
            out.push_str(&literal);
            i = end;
            continue;
        }
        let mut matched = false;
        for (token, value) in &values {
            let t: Vec<char> = token.chars().collect();
            if chars[i..].starts_with(&t) {
                out.push_str(value);
                i += t.len();
                matched = true;
                break;
            }
        }
        if !matched {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Lowercase, non-alphanumeric stripped -> `snake_case`.
pub fn snake_case(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .to_ascii_lowercase();
    if cleaned.is_empty() {
        "project".to_string()
    } else {
        cleaned
    }
}

/// PascalCase for class/type names.
pub fn pascal_case(name: &str) -> String {
    let words: Vec<String> = name
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect();
    let joined = words.join("");
    if joined.is_empty() {
        "Project".to_string()
    } else {
        joined
    }
}

pub fn kebab_case(name: &str) -> String {
    let out: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .to_ascii_lowercase();
    out.trim_matches('-').to_string()
}

/// Minimal buildable skeleton per language: `(relative path, template)`.
/// `{name}`/`{snake}`/`{Pascal}`/`{kebab}` placeholders are substituted.
fn templates(lang: Language) -> Vec<(&'static str, &'static str)> {
    match lang {
        Language::Rust => vec![
            (
                "Cargo.toml",
                "[package]\nname = \"{kebab}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n",
            ),
            (
                "src/main.rs",
                "fn main() {\n    println!(\"Hello, {name}!\");\n}\n",
            ),
        ],
        Language::Python => vec![
            (
                "pyproject.toml",
                "[project]\nname = \"{kebab}\"\nversion = \"0.1.0\"\n\n[build-system]\nrequires = [\"setuptools\"]\nbuild-backend = \"setuptools.build_meta\"\n",
            ),
            (
                "main.py",
                "def main() -> None:\n    print(\"Hello, {name}!\")\n\n\nif __name__ == \"__main__\":\n    main()\n",
            ),
        ],
        Language::Go => vec![
            (
                "go.mod",
                "module {kebab}\n\ngo 1.21\n",
            ),
            (
                "main.go",
                "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"Hello, {name}!\")\n}\n",
            ),
        ],
        Language::TypeScript => vec![
            (
                "package.json",
                "{\n  \"name\": \"{kebab}\",\n  \"version\": \"0.1.0\",\n  \"scripts\": {\n    \"build\": \"tsc -p .\",\n    \"test\": \"vitest run\",\n    \"lint\": \"eslint .\"\n  },\n  \"devDependencies\": {\n    \"typescript\": \"^5\",\n    \"@types/node\": \"^20\",\n    \"eslint\": \"^9\",\n    \"vitest\": \"^2\"\n  }\n}\n",
            ),
            (
                "tsconfig.json",
                "{\n  \"compilerOptions\": {\n    \"target\": \"ES2022\",\n    \"module\": \"NodeNext\",\n    \"moduleResolution\": \"NodeNext\",\n    \"strict\": true,\n    \"outDir\": \"dist\"\n  },\n  \"include\": [\"src\"]\n}\n",
            ),
            (
                "src/index.ts",
                "const message: string = \"Hello, {name}!\";\nconsole.log(message);\n",
            ),
        ],
        Language::JavaScript => vec![
            (
                "package.json",
                "{\n  \"name\": \"{kebab}\",\n  \"version\": \"0.1.0\",\n  \"scripts\": {\n    \"build\": \"node src/index.js\",\n    \"test\": \"node --test\"\n  }\n}\n",
            ),
            (
                "src/index.js",
                "const message = \"Hello, {name}!\";\nconsole.log(message);\n",
            ),
        ],
        Language::Java => vec![
            (
                "pom.xml",
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<project xmlns=\"http://maven.apache.org/POM/4.0.0\">\n  <modelVersion>4.0.0</modelVersion>\n  <groupId>com.example</groupId>\n  <artifactId>{kebab}</artifactId>\n  <version>0.1.0</version>\n  <properties>\n    <maven.compiler.source>17</maven.compiler.source>\n    <maven.compiler.target>17</maven.compiler.target>\n  </properties>\n</project>\n",
            ),
            (
                "src/main/java/com/example/Main.java",
                "package com.example;\n\npublic class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, {name}!\");\n    }\n}\n",
            ),
        ],
        Language::Kotlin => vec![
            (
                "build.gradle.kts",
                "plugins {\n    application\n    kotlin(\"jvm\") version \"1.9.0\"\n}\n\nrepositories { mavenCentral() }\n\napplication { mainClass.set(\"MainKt\") }\n",
            ),
            (
                "settings.gradle.kts",
                "rootProject.name = \"{kebab}\"\n",
            ),
            (
                "src/main/kotlin/Main.kt",
                "fun main() {\n    println(\"Hello, {name}!\")\n}\n",
            ),
        ],
        Language::CSharp => vec![
            (
                "{kebab}.csproj",
                "<Project Sdk=\"Microsoft.NET.Sdk\">\n  <PropertyGroup>\n    <OutputType>Exe</OutputType>\n    <TargetFramework>net8.0</TargetFramework>\n    <RootNamespace>{Pascal}</RootNamespace>\n    <ImplicitUsings>enable</ImplicitUsings>\n  </PropertyGroup>\n</Project>\n",
            ),
            (
                "Program.cs",
                "Console.WriteLine(\"Hello, {name}!\");\n",
            ),
        ],
        Language::Swift => vec![
            (
                "Package.swift",
                "// swift-tools-version:5.9\nimport PackageDescription\n\nlet package = Package(\n    name: \"{kebab}\",\n    targets: [\n        .executableTarget(name: \"{Pascal}\", path: \"Sources\")\n    ]\n)\n",
            ),
            (
                "Sources/main.swift",
                "print(\"Hello, {name}!\")\n",
            ),
        ],
        Language::Cpp => vec![
            (
                "CMakeLists.txt",
                "cmake_minimum_required(VERSION 3.16)\nproject({kebab})\n\nadd_executable(main src/main.cpp)\n",
            ),
            (
                "src/main.cpp",
                "#include <iostream>\n\nint main() {\n    std::cout << \"Hello, {name}!\\n\";\n    return 0;\n}\n",
            ),
        ],
        Language::Php => vec![
            (
                "composer.json",
                "{\n  \"name\": \"{kebab}\",\n  \"require\": {}\n}\n",
            ),
            (
                "index.php",
                "<?php\n\necho \"Hello, {name}!\" . PHP_EOL;\n",
            ),
        ],
        Language::Ruby => vec![
            (
                "Gemfile",
                "source \"https://rubygems.org\"\n\nruby \">= 3.0\"\ngem \"rspec\", group: :test\n",
            ),
            (
                "main.rb",
                "def main\n  puts \"Hello, {name}!\"\nend\n\nmain\n",
            ),
        ],
        Language::Dart => vec![
            (
                "pubspec.yaml",
                "name: {snake}\nversion: 0.1.0\nenvironment:\n  sdk: \">=3.0.0 <4.0.0\"\n",
            ),
            (
                "bin/main.dart",
                "void main() {\n  print('Hello, {name}!');\n}\n",
            ),
        ],
        Language::Scala => vec![
            (
                "build.sbt",
                "ThisBuild / scalaVersion := \"3.3.1\"\n\nlazy val root = (project in file(\".\"))\n  .settings(\n    name := \"{kebab}\"\n  )\n",
            ),
            (
                "src/main/scala/Main.scala",
                "@main def main(): Unit =\n  println(\"Hello, {name}!\")\n",
            ),
        ],
        Language::Elixir => vec![
            (
                "mix.exs",
                "defmodule {Pascal}.MixProject do\n  use Mix.Project\n\n  def project do\n    [app: :{name}, version: \"0.1.0\"]\n  end\nend\n",
            ),
            (
                "lib/{name}.ex",
                "defmodule {Pascal} do\n  @moduledoc \"\"\"\n  {name}\n  \"\"\"\n\n  def hello do\n    IO.puts(\"Hello, {name}!\")\n  end\nend\n",
            ),
        ],
        Language::R => vec![
            (
                "DESCRIPTION",
                "Package: {name}\nTitle: A {name} package\nVersion: 0.1.0\nDescription: Hello world.\nLicense: MIT\nEncoding: UTF-8\n",
            ),
            (
                "main.R",
                "# Hello, {name}!\nmain <- function() {\n  cat(\"Hello, {name}!\\n\")\n}\n\nmain()\n",
            ),
        ],
        Language::Zig => vec![
            (
                "build.zig",
                "const std = @import(\"std\");\n\npub fn build(b: *std.Build) void {\n    const exe = b.addExecutable(.{\n        .name = \"{kebab}\",\n        .root_source_file = b.path(\"src/main.zig\"),\n        .target = b.standardTargetOptions({}),\n    });\n    b.installArtifact(exe);\n}\n",
            ),
            (
                "src/main.zig",
                "const std = @import(\"std\");\n\npub fn main() !void {\n    const stdout = std.io.getStdOut().writer();\n    try stdout.print(\"Hello, {name}!\\n\", .{});\n}\n",
            ),
        ],
        Language::Haskell => vec![
            (
                "{kebab}.cabal",
                "cabal-version:      2.4\nname:               {kebab}\nversion:            0.1.0\nbuild-type:         Simple\n\nlibrary\n  hs-source-dirs:   src\n  exposed-modules:  Main\n  build-depends:    base >=4.14 && <5\n  default-language: Haskell2010\n",
            ),
            (
                "src/Main.hs",
                "module Main where\n\nmain :: IO ()\nmain = putStrLn \"Hello, {name}!\"\n",
            ),
        ],
        Language::Lua => vec![
            (
                "main.lua",
                "local function main()\n  print(\"Hello, {name}!\")\nend\n\nmain()\n",
            ),
        ],
        Language::Solidity => vec![
            (
                "foundry.toml",
                "[profile.default]\nsrc = \"contracts\"\nout = \"out\"\nlibs = [\"lib\"]\n",
            ),
            (
                "contracts/{Pascal}.sol",
                "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n\ncontract {Pascal} {\n    function hello() public pure returns (string memory) {\n        return \"Hello, {name}!\";\n    }\n}\n",
            ),
        ],
        Language::Clojure => vec![
            (
                "deps.edn",
                "{:paths [\"src\"]\n :deps {org.clojure/clojure {:mvn/version \"1.11.1\"}}}\n",
            ),
            (
                "src/core.clj",
                "(ns core)\n\n(defn -main []\n  (println \"Hello, {name}!\"))\n\n(-main)\n",
            ),
        ],
        Language::Julia => vec![
            (
                "Project.toml",
                "name = \"{Pascal}\"\nuuid = \"00000000-0000-0000-0000-000000000000\"\nauthors = [\"you\"]\nversion = \"0.1.0\"\n",
            ),
            (
                "main.jl",
                "function main()\n    println(\"Hello, {name}!\")\nend\n\nmain()\n",
            ),
        ],
        Language::Perl => vec![
            (
                "cpanfile",
                "requires 'perl', '5.20';\n",
            ),
            (
                "script/main.pl",
                "#!/usr/bin/env perl\nuse strict;\nuse warnings;\n\nprint \"Hello, {name}!\\n\";\n",
            ),
        ],
        Language::OCaml => vec![
            (
                "dune-project",
                "(lang dune 3.0)\n",
            ),
            (
                "bin/main.ml",
                "let () = print_endline \"Hello, {name}!\"\n",
            ),
            (
                "bin/dune",
                "(executable\n (name main))\n",
            ),
        ],
        Language::Nim => vec![
            (
                "{name}.nimble",
                "version = \"0.1.0\"\nauthor = \"you\"\ndescription = \"{name}\"\nlicense = \"MIT\"\nsrcDir = \"src\"\n",
            ),
            (
                "src/main.nim",
                "echo \"Hello, {name}!\"\n",
            ),
        ],
        Language::Crystal => vec![
            (
                "shard.yml",
                "name: {name}\nversion: 0.1.0\n",
            ),
            (
                "src/main.cr",
                "puts \"Hello, {name}!\"\n",
            ),
        ],
        Language::Groovy => vec![
            (
                "build.gradle",
                "plugins { id 'groovy' }\nrepositories { mavenCentral() }\n\ndependencies { implementation 'org.apache.groovy:groovy:4.0.0' }\n",
            ),
            (
                "src/main/groovy/Main.groovy",
                "class Main {\n    static void main(String[] args) {\n        println 'Hello, {name}!'\n    }\n}\n",
            ),
        ],
        Language::Fortran => vec![
            (
                "CMakeLists.txt",
                "cmake_minimum_required(VERSION 3.16)\nproject({kebab} Fortran)\n\nadd_executable(main src/main.f90)\n",
            ),
            (
                "src/main.f90",
                "program main\n  implicit none\n  print *, 'Hello, {name}!'\nend program main\n",
            ),
        ],
        Language::Shell => vec![
            (
                "main.sh",
                "#!/usr/bin/env bash\nset -euo pipefail\n\necho \"Hello, {name}!\"\n",
            ),
        ],
        Language::PowerShell => vec![
            (
                "main.ps1",
                "#!/usr/bin/env pwsh\nWrite-Host \"Hello, {name}!\"\n",
            ),
        ],
        Language::Erlang => vec![
            (
                "rebar.config",
                "{erl_opts, [debug_info]}.\n",
            ),
            (
                "src/main.erl",
                "-module(main).\n-export([hello/0]).\n\nhello() -> io:format(\"Hello, {name}!~n\").\n",
            ),
        ],
        Language::FSharp => vec![
            (
                "{kebab}.fsproj",
                "<Project Sdk=\"Microsoft.NET.Sdk\">\n  <PropertyGroup>\n    <OutputType>Exe</OutputType>\n    <TargetFramework>net8.0</TargetFramework>\n  </PropertyGroup>\n  <ItemGroup>\n    <Compile Include=\"Program.fs\" />\n  </ItemGroup>\n</Project>\n",
            ),
            (
                "Program.fs",
                "printfn \"Hello, {name}!\"\n",
            ),
        ],
        Language::ObjectiveC => vec![
            (
                "Makefile",
                "CC=clang\nCFLAGS=-fobjc-arc -framework Foundation\n\nall:\n\t$(CC) $(CFLAGS) -o main src/main.m\n",
            ),
            (
                "src/main.m",
                "#import <Foundation/Foundation.h>\n\nint main(void) {\n    @autoreleasepool {\n        NSLog(@\"Hello, {name}!\");\n    }\n    return 0;\n}\n",
            ),
        ],
        Language::V => vec![
            (
                "v.mod",
                "Module {\n\tname: '{kebab}'\n\tversion: '0.1.0'\n}\n",
            ),
            (
                "main.v",
                "fn main() {\n\tprintln('Hello, {name}!')\n}\n",
            ),
        ],
        Language::Ada => vec![
            (
                "project.gpr",
                "project {Pascal} is\n   for Source_Dirs use (\"src\");\n   for Main use (\"main.adb\");\nend {Pascal};\n",
            ),
            (
                "src/main.adb",
                "with Ada.Text_IO;\n\nprocedure Main is\nbegin\n   Ada.Text_IO.Put_Line (\"Hello, {name}!\");\nend Main;\n",
            ),
        ],
        Language::Pascal => vec![
            (
                "main.pas",
                "program {Pascal};\n\nbegin\n  writeln('Hello, {name}!');\nend.\n",
            ),
        ],
        Language::Lisp => vec![
            (
                "main.lisp",
                "(format t \"Hello, {name}!~%\")",
            ),
        ],
        Language::Scheme => vec![
            (
                "main.rkt",
                "#lang racket\n\n(displayln \"Hello, {name}!\")\n",
            ),
        ],
        Language::VisualBasic => vec![
            (
                "{kebab}.vbproj",
                "<Project Sdk=\"Microsoft.NET.Sdk\">\n  <PropertyGroup>\n    <OutputType>Exe</OutputType>\n    <RootNamespace>{Pascal}</RootNamespace>\n    <TargetFramework>net8.0</TargetFramework>\n  </PropertyGroup>\n</Project>\n",
            ),
            (
                "Program.vb",
                "Imports System\n\nModule Program\n    Sub Main()\n        Console.WriteLine(\"Hello, {name}!\")\n    End Sub\nEnd Module\n",
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn scaffolds_a_rust_project() {
        let tmp = TempDir::new().unwrap();
        let target = tmp.path().join("my-app");
        let written = scaffold_project(Language::Rust, "my-app", &target).unwrap();
        assert!(target.join("Cargo.toml").exists());
        assert!(target.join("src/main.rs").exists());
        assert_eq!(written.len(), 2);
        let text = std::fs::read_to_string(target.join("Cargo.toml")).unwrap();
        assert!(text.contains("my-app"), "crate name substituted: {text}");
    }

    #[test]
    fn scaffolds_each_available_language() {
        for lang in available_scaffold_languages() {
            let tmp = TempDir::new().unwrap();
            let target = tmp.path().join("proj");
            let written = scaffold_project(lang, "demo-app", &target).unwrap();
            assert!(!written.is_empty(), "{lang:?} produced no files");
            for p in &written {
                assert!(p.exists(), "{lang:?} wrote missing file {}", p.display());
            }
        }
    }

    #[test]
    fn pascal_and_snake_render() {
        assert_eq!(snake_case("My App 2"), "my_app_2");
        assert_eq!(pascal_case("my-app"), "MyApp");
        assert_eq!(kebab_case("Hello World"), "hello-world");
    }

    #[test]
    fn render_substitutes_tokens_but_not_js_template_literals() {
        let tpl = "export function greeting(name: string): string {\n  return `Hello, ${name}!`\n}\n\nfunction App() {\n  return <h1>{greeting('{name}')}</h1>\n}\n";
        let out = render(tpl, "bp-react");
        // The JS template literal `${name}` must survive verbatim...
        assert!(out.contains("Hello, ${name}!"), "{out}");
        assert!(!out.contains("$bp_react"), "{out}");
        // ...while the bare `{name}` placeholder still resolves.
        assert!(out.contains("greeting('bp_react')"), "{out}");
        // Non-$ placeholders all still work.
        assert_eq!(
            render("{snake} {kebab} {Pascal}", "hello world"),
            "hello_world hello-world HelloWorld"
        );
    }
}
