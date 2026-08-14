//! Framework awareness: detect which web/app framework a project uses
//! (React, Django, Rails, …) and scaffold a minimal recognizable skeleton
//! for it. A framework is layered on top of a base `Language`, which is what
//! drives build/test/lint commands and project detection.

use crate::detect::{walk, Language};
use std::path::Path;

/// A major application framework zeus can detect and scaffold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Framework {
    React,
    Vue,
    Svelte,
    NextJs,
    Angular,
    Express,
    Django,
    Flask,
    Laravel,
    Rails,
    SpringBoot,
    Flutter,
    AspNetCore,
}

/// Everything zeus knows about one framework.
pub struct FrameworkSpec {
    pub framework: Framework,
    /// Human display name, e.g. "React".
    pub display_name: &'static str,
    /// The language the framework is written in — the spec that drives the
    /// dev commands (`zeus project commands`).
    pub base: Language,
    /// Manifest files / dependency keys used to identify the framework.
    pub markers: &'static [&'static str],
}

impl Framework {
    pub const ALL: &'static [Framework] = &[
        Framework::React,
        Framework::Vue,
        Framework::Svelte,
        Framework::NextJs,
        Framework::Angular,
        Framework::Express,
        Framework::Django,
        Framework::Flask,
        Framework::Laravel,
        Framework::Rails,
        Framework::SpringBoot,
        Framework::Flutter,
        Framework::AspNetCore,
    ];

    /// Parse a framework name ("react", "vue", "next.js", "django"...).
    pub fn from_name(name: &str) -> Option<Framework> {
        let key = name.trim().to_ascii_lowercase().replace('_', "-");
        let fw = match key.as_str() {
            "react" | "reactjs" | "react.js" => Framework::React,
            "vue" | "vuejs" | "vue.js" | "nuxt" => Framework::Vue,
            "svelte" | "sveltekit" => Framework::Svelte,
            "next" | "nextjs" | "next.js" => Framework::NextJs,
            "angular" | "angularjs" => Framework::Angular,
            "express" | "expressjs" => Framework::Express,
            "django" => Framework::Django,
            "flask" => Framework::Flask,
            "laravel" => Framework::Laravel,
            "rails" | "rubyonrails" | "ruby-on-rails" => Framework::Rails,
            "spring" | "springboot" | "spring-boot" => Framework::SpringBoot,
            "flutter" => Framework::Flutter,
            "aspnet" | "asp.net" | "aspnetcore" | "dotnet-web" | "netcore" => Framework::AspNetCore,
            _ => return None,
        };
        Some(fw)
    }

    /// Detect the framework of a project directory. Manifest contents are
    /// keyword-matched (package.json dependencies, Gemfile, pom.xml, …).
    pub fn detect_framework(root: &Path) -> Option<Framework> {
        if !root.is_dir() {
            return None;
        }
        let text = |names: &[&str]| -> String {
            let mut acc = String::new();
            for n in names {
                let p = root.join(n);
                if p.is_file() {
                    if let Ok(t) = std::fs::read_to_string(&p) {
                        acc.push_str(&t);
                        acc.push('\n');
                    }
                }
            }
            // Cap a couple of manifests so a giant package.json can't turn
            // detection into a memory risk; frameworks match in the first KBs.
            acc.chars().take(128_000).collect()
        };

        // *.csproj files are not at a fixed path — scan a shallow walk.
        let has_csproj = walk(root, 2)
            .iter()
            .any(|(p, _)| p.extension().is_some_and(|e| e == "csproj"));

        let needles: &[(Framework, &[&str])] = &[
            (Framework::React, &["react"]),
            (Framework::Vue, &["vue"]),
            (Framework::Svelte, &["svelte"]),
            (Framework::NextJs, &["next"]),
            (Framework::Angular, &["@angular/core"]),
            (Framework::Express, &["express"]),
            (Framework::Django, &["django", "manage.py"]),
            (Framework::Flask, &["flask"]),
            (Framework::Laravel, &["laravel/framework", "artisan"]),
            (Framework::Rails, &["rails"]),
            (
                Framework::SpringBoot,
                &["spring-boot-starter", "spring-boot"],
            ),
            (Framework::Flutter, &["flutter"]),
        ];

        let joined = text(&[
            "package.json",
            "composer.json",
            "pubspec.yaml",
            "Gemfile",
            "pom.xml",
            "requirements.txt",
            "pyproject.toml",
            "manage.py",
            "artisan",
        ]);
        for (fw, needles) in needles {
            if needles.iter().any(|n| contains_word(&joined, n)) {
                return Some(*fw);
            }
        }
        // A .NET web project: no stable single manifest name, but a
        // `Sdk="Microsoft.NET.Sdk.Web"` csproj is unambiguous.
        if has_csproj {
            return Some(Framework::AspNetCore);
        }
        None
    }
}

fn contains_word(haystack: &str, needle: &str) -> bool {
    haystack.contains(needle)
}

/// Fetch the spec for a framework (always exists).
pub fn framework_spec(fw: Framework) -> &'static FrameworkSpec {
    FRAMEWORKS
        .iter()
        .find(|s| s.framework == fw)
        .expect("framework table covers every Framework variant")
}

const FRAMEWORKS: &[FrameworkSpec] = &[
    FrameworkSpec {
        framework: Framework::React,
        display_name: "React",
        base: Language::TypeScript,
        markers: &["react"],
    },
    FrameworkSpec {
        framework: Framework::Vue,
        display_name: "Vue",
        base: Language::TypeScript,
        markers: &["vue"],
    },
    FrameworkSpec {
        framework: Framework::Svelte,
        display_name: "Svelte",
        base: Language::TypeScript,
        markers: &["svelte"],
    },
    FrameworkSpec {
        framework: Framework::NextJs,
        display_name: "Next.js",
        base: Language::TypeScript,
        markers: &["next"],
    },
    FrameworkSpec {
        framework: Framework::Angular,
        display_name: "Angular",
        base: Language::TypeScript,
        markers: &["@angular/core"],
    },
    FrameworkSpec {
        framework: Framework::Express,
        display_name: "Express",
        base: Language::JavaScript,
        markers: &["express"],
    },
    FrameworkSpec {
        framework: Framework::Django,
        display_name: "Django",
        base: Language::Python,
        markers: &["django", "manage.py"],
    },
    FrameworkSpec {
        framework: Framework::Flask,
        display_name: "Flask",
        base: Language::Python,
        markers: &["flask"],
    },
    FrameworkSpec {
        framework: Framework::Laravel,
        display_name: "Laravel",
        base: Language::Php,
        markers: &["laravel/framework", "artisan"],
    },
    FrameworkSpec {
        framework: Framework::Rails,
        display_name: "Ruby on Rails",
        base: Language::Ruby,
        markers: &["rails"],
    },
    FrameworkSpec {
        framework: Framework::SpringBoot,
        display_name: "Spring Boot",
        base: Language::Java,
        markers: &["spring-boot"],
    },
    FrameworkSpec {
        framework: Framework::Flutter,
        display_name: "Flutter",
        base: Language::Dart,
        markers: &["flutter"],
    },
    FrameworkSpec {
        framework: Framework::AspNetCore,
        display_name: "ASP.NET Core",
        base: Language::CSharp,
        markers: &["Microsoft.NET.Sdk.Web"],
    },
];

/// Write a freshly scaffolded framework project into `target` (created if
/// needed) and return the absolute paths written. `name` is the project name.
pub fn scaffold_framework(
    fw: Framework,
    name: &str,
    target: &Path,
) -> Result<Vec<std::path::PathBuf>, crate::scaffold::ScaffoldError> {
    use crate::scaffold::{render, ScaffoldError};
    let templates = framework_templates(fw);
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

/// Minimal recognizable skeleton per framework: `(relative path, template)`.
/// `{name}`/`{snake}`/`{Pascal}`/`{kebab}` placeholders are substituted.
fn framework_templates(fw: Framework) -> Vec<(&'static str, &'static str)> {
    match fw {
        Framework::React => vec![
            (
                "package.json",
                "{\n  \"name\": \"{kebab}\",\n  \"private\": true,\n  \"version\": \"0.1.0\",\n  \"type\": \"module\",\n  \"scripts\": {\n    \"dev\": \"vite\",\n    \"build\": \"vite build\",\n    \"preview\": \"vite preview\"\n  },\n  \"dependencies\": {\n    \"react\": \"^18\",\n    \"react-dom\": \"^18\"\n  },\n  \"devDependencies\": {\n    \"@vitejs/plugin-react\": \"^4\",\n    \"vite\": \"^5\"\n  }\n}\n",
            ),
            (
                "index.html",
                "<!doctype html>\n<html lang=\"en\">\n  <head>\n    <meta charset=\"UTF-8\" />\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n    <title>{kebab}</title>\n  </head>\n  <body>\n    <div id=\"root\"></div>\n    <script type=\"module\" src=\"/src/main.jsx\"></script>\n  </body>\n</html>\n",
            ),
            (
                "src/main.jsx",
                "import React from 'react'\nimport { createRoot } from 'react-dom/client'\nimport App from './App.jsx'\n\ncreateRoot(document.getElementById('root')).render(\n  <React.StrictMode>\n    <App />\n  </React.StrictMode>\n)\n",
            ),
            (
                "src/App.jsx",
                "function App() {\n  return <h1>Hello, {name}!</h1>\n}\n\nexport default App\n",
            ),
            (
                "src/App.css",
                "body { font-family: system-ui, sans-serif; }\n",
            ),
        ],
        Framework::Vue => vec![
            (
                "package.json",
                "{\n  \"name\": \"{kebab}\",\n  \"private\": true,\n  \"version\": \"0.1.0\",\n  \"type\": \"module\",\n  \"scripts\": {\n    \"dev\": \"vite\",\n    \"build\": \"vite build\",\n    \"preview\": \"vite preview\"\n  },\n  \"dependencies\": {\n    \"vue\": \"^3\"\n  },\n  \"devDependencies\": {\n    \"@vitejs/plugin-vue\": \"^5\",\n    \"vite\": \"^5\"\n  }\n}\n",
            ),
            (
                "index.html",
                "<!doctype html>\n<html lang=\"en\">\n  <head>\n    <meta charset=\"UTF-8\" />\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n    <title>{kebab}</title>\n  </head>\n  <body>\n    <div id=\"app\"></div>\n    <script type=\"module\" src=\"/src/main.js\"></script>\n  </body>\n</html>\n",
            ),
            (
                "src/main.js",
                "import { createApp } from 'vue'\nimport App from './App.vue'\n\ncreateApp(App).mount('#app')\n",
            ),
            (
                "src/App.vue",
                "<template>\n  <h1>Hello, {name}!</h1>\n</template>\n\n<script setup>\n</script>\n",
            ),
        ],
        Framework::Svelte => vec![
            (
                "package.json",
                "{\n  \"name\": \"{kebab}\",\n  \"private\": true,\n  \"version\": \"0.1.0\",\n  \"type\": \"module\",\n  \"scripts\": {\n    \"dev\": \"vite\",\n    \"build\": \"vite build\",\n    \"preview\": \"vite preview\"\n  },\n  \"devDependencies\": {\n    \"@sveltejs/vite-plugin-svelte\": \"^3\",\n    \"svelte\": \"^4\",\n    \"vite\": \"^5\"\n  }\n}\n",
            ),
            (
                "index.html",
                "<!doctype html>\n<html lang=\"en\">\n  <head>\n    <meta charset=\"UTF-8\" />\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n    <title>{kebab}</title>\n  </head>\n  <body>\n    <div id=\"app\"></div>\n    <script type=\"module\" src=\"/src/main.js\"></script>\n  </body>\n</html>\n",
            ),
            (
                "src/main.js",
                "import App from './App.svelte'\n\nconst app = new App({ target: document.getElementById('app') })\n\nexport default app\n",
            ),
            (
                "src/App.svelte",
                "<script>\n  let name = '{name}'\n</script>\n\n<h1>Hello, {name}!</h1>\n",
            ),
        ],
        Framework::NextJs => vec![
            (
                "package.json",
                "{\n  \"name\": \"{kebab}\",\n  \"private\": true,\n  \"version\": \"0.1.0\",\n  \"scripts\": {\n    \"dev\": \"next dev\",\n    \"build\": \"next build\",\n    \"start\": \"next start\"\n  },\n  \"dependencies\": {\n    \"next\": \"^14\",\n    \"react\": \"^18\",\n    \"react-dom\": \"^18\"\n  },\n  \"devDependencies\": {\n    \"typescript\": \"^5\"\n  }\n}\n",
            ),
            (
                "tsconfig.json",
                "{\n  \"compilerOptions\": {\n    \"lib\": [\"dom\", \"dom.iterable\", \"esnext\"],\n    \"module\": \"esnext\",\n    \"moduleResolution\": \"bundler\",\n    \"jsx\": \"preserve\",\n    \"strict\": true\n  },\n  \"include\": [\"next-env.d.ts\", \"app\"]\n}\n",
            ),
            (
                "app/layout.tsx",
                "export default function RootLayout({ children }: { children: React.ReactNode }) {\n  return (\n    <html lang=\"en\">\n      <body>{children}</body>\n    </html>\n  )\n}\n",
            ),
            (
                "app/page.tsx",
                "export default function Page() {\n  return <h1>Hello, {name}!</h1>\n}\n",
            ),
            (
                "next-env.d.ts",
                "/// <reference types=\"next\" />\n",
            ),
        ],
        Framework::Angular => vec![
            (
                "package.json",
                "{\n  \"name\": \"{kebab}\",\n  \"version\": \"0.1.0\",\n  \"scripts\": {\n    \"start\": \"ng serve\",\n    \"build\": \"ng build\",\n    \"test\": \"ng test\"\n  },\n  \"dependencies\": {\n    \"@angular/core\": \"^17\",\n    \"@angular/platform-browser\": \"^17\"\n  },\n  \"devDependencies\": {\n    \"@angular/cli\": \"^17\",\n    \"typescript\": \"~5.3.0\"\n  }\n}\n",
            ),
            (
                "angular.json",
                "{\n  \"version\": 1,\n  \"projects\": {\n    \"{kebab}\": {\n      \"projectType\": \"application\",\n      \"root\": \"\",\n      \"sourceRoot\": \"src\",\n      \"architect\": {\n        \"build\": {\n          \"builder\": \"@angular-devkit/build-angular:browser\"\n        }\n      }\n    }\n  }\n}\n",
            ),
            (
                "src/main.ts",
                "import { bootstrapApplication } from '@angular/platform-browser'\nimport { AppComponent } from './app/app.component'\n\nbootstrapApplication(AppComponent)\n",
            ),
            (
                "src/app/app.component.ts",
                "import { Component } from '@angular/core'\n\n@Component({\n  selector: 'app-root',\n  templateUrl: './app.component.html'\n})\nexport class AppComponent {\n  title = '{name}'\n}\n",
            ),
            (
                "src/app/app.component.html",
                "<h1>Hello, {{ title }}!</h1>\n",
            ),
        ],
        Framework::Express => vec![
            (
                "package.json",
                "{\n  \"name\": \"{kebab}\",\n  \"version\": \"0.1.0\",\n  \"scripts\": {\n    \"start\": \"node server.js\"\n  },\n  \"dependencies\": {\n    \"express\": \"^4\"\n  }\n}\n",
            ),
            (
                "server.js",
                "const express = require('express')\nconst app = express()\n\napp.get('/', (req, res) => {\n  res.send('Hello, {name}!')\n})\n\napp.listen(3000, () => {\n  console.log('listening on http://localhost:3000')\n})\n",
            ),
        ],
        Framework::Django => vec![
            (
                "manage.py",
                "#!/usr/bin/env python\nimport os\nimport sys\n\nif __name__ == \"__main__\":\n    os.environ.setdefault(\"DJANGO_SETTINGS_MODULE\", \"config.settings\")\n    from django.core.management import execute_from_command_line\n    execute_from_command_line(sys.argv)\n",
            ),
            (
                "requirements.txt",
                "Django>=5,<6\n",
            ),
            (
                "config/settings.py",
                "SECRET_KEY = 'dev-only-change-me'\nDEBUG = True\nALLOWED_HOSTS = []\nINSTALLED_APPS = ['django.contrib.contenttypes', 'app']\nROOT_URLCONF = 'config.urls'\nDATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': 'db.sqlite3'}}\nUSE_TZ = True\n",
            ),
            (
                "config/urls.py",
                "from django.urls import path\nfrom app import views\n\nurlpatterns = [\n    path('', views.index),\n]\n",
            ),
            (
                "app/views.py",
                "from django.http import HttpResponse\n\n\ndef index(request):\n    return HttpResponse(\"Hello, {name}!\")\n",
            ),
            (
                "app/__init__.py",
                "",
            ),
            (
                "config/__init__.py",
                "",
            ),
        ],
        Framework::Flask => vec![
            (
                "app.py",
                "from flask import Flask\n\napp = Flask(__name__)\n\n\n@app.route(\"/\")\ndef index():\n    return \"Hello, {name}!\"\n\n\nif __name__ == \"__main__\":\n    app.run(debug=True)\n",
            ),
            (
                "requirements.txt",
                "Flask>=3,<4\n",
            ),
        ],
        Framework::Laravel => vec![
            (
                "composer.json",
                "{\n  \"name\": \"{kebab}/app\",\n  \"require\": {\n    \"php\": \">=8.2\",\n    \"laravel/framework\": \"^11\"\n  }\n}\n",
            ),
            (
                "artisan",
                "#!/usr/bin/env php\n<?php\nuse Illuminate\\Foundation\\Application;\nrequire __DIR__.'/vendor/autoload.php';\n$app = new Application(__DIR__);\necho \"Hello, {name}!\\n\";\n",
            ),
            (
                "routes/web.php",
                "<?php\n\nuse Illuminate\\Support\\Facades\\Route;\n\nRoute::get('/', fn () => 'Hello, {name}!');\n",
            ),
        ],
        Framework::Rails => vec![
            (
                "Gemfile",
                "source \"https://rubygems.org\"\n\ngem \"rails\", \"~> 7.1\"\ngem \"puma\"\n",
            ),
            (
                "config.ru",
                "require_relative \"config/application\"\n\nrun Rails.application\n",
            ),
            (
                "app/controllers/application_controller.rb",
                "class ApplicationController < ActionController::Base\nend\n",
            ),
            (
                "app/views/layouts/application.html.erb",
                "<!doctype html>\n<html>\n  <body>\n    <h1>Hello, {name}!</h1>\n  </body>\n</html>\n",
            ),
        ],
        Framework::SpringBoot => vec![
            (
                "pom.xml",
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<project xmlns=\"http://maven.apache.org/POM/4.0.0\">\n  <modelVersion>4.0.0</modelVersion>\n  <parent>\n    <groupId>org.springframework.boot</groupId>\n    <artifactId>spring-boot-starter-parent</artifactId>\n    <version>3.2.0</version>\n  </parent>\n  <groupId>com.example</groupId>\n  <artifactId>{kebab}</artifactId>\n  <version>0.1.0</version>\n  <dependencies>\n    <dependency>\n      <groupId>org.springframework.boot</groupId>\n      <artifactId>spring-boot-starter-web</artifactId>\n    </dependency>\n  </dependencies>\n</project>\n",
            ),
            (
                "src/main/java/com/example/Application.java",
                "package com.example;\n\nimport org.springframework.boot.SpringApplication;\nimport org.springframework.boot.autoconfigure.SpringBootApplication;\n\n@SpringBootApplication\npublic class Application {\n    public static void main(String[] args) {\n        SpringApplication.run(Application.class, args);\n    }\n}\n",
            ),
            (
                "src/main/java/com/example/HelloController.java",
                "package com.example;\n\nimport org.springframework.web.bind.annotation.*;\n\n@RestController\npublic class HelloController {\n    @GetMapping(\"/\")\n    public String hello() {\n        return \"Hello, {name}!\";\n    }\n}\n",
            ),
        ],
        Framework::Flutter => vec![
            (
                "pubspec.yaml",
                "name: {snake}\nversion: 0.1.0\nenvironment:\n  sdk: \">=3.0.0 <4.0.0\"\ndependencies:\n  flutter:\n    sdk: flutter\n",
            ),
            (
                "lib/main.dart",
                "import 'package:flutter/material.dart';\n\nvoid main() => runApp(const App());\n\nclass App extends StatelessWidget {\n  const App({super.key});\n\n  @override\n  Widget build(BuildContext context) {\n    return MaterialApp(\n      home: Scaffold(\n        body: Center(child: Text('Hello, {name}!')),\n      ),\n    );\n  }\n}\n",
            ),
        ],
        Framework::AspNetCore => vec![
            (
                "{kebab}.csproj",
                "<Project Sdk=\"Microsoft.NET.Sdk.Web\">\n  <PropertyGroup>\n    <TargetFramework>net8.0</TargetFramework>\n    <Nullable>enable</Nullable>\n  </PropertyGroup>\n</Project>\n",
            ),
            (
                "Program.cs",
                "var builder = WebApplication.CreateBuilder(args);\nvar app = builder.Build();\n\napp.MapGet(\"/\", () => \"Hello, {name}!\");\n\napp.Run();\n",
            ),
            (
                "appsettings.json",
                "{\n  \"Logging\": {\n    \"LogLevel\": {\n      \"Default\": \"Information\"\n    }\n  }\n}\n",
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write(root: &std::path::Path, rel: &str, content: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, content).unwrap();
    }

    #[test]
    fn detects_frameworks_from_manifests() {
        let tmp = TempDir::new().unwrap();
        write(
            tmp.path(),
            "package.json",
            r#"{"dependencies": {"react": "^18"}}"#,
        );
        assert_eq!(
            Framework::detect_framework(tmp.path()),
            Some(Framework::React)
        );

        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "manage.py", "# django\n");
        assert_eq!(
            Framework::detect_framework(tmp.path()),
            Some(Framework::Django)
        );

        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "Gemfile", "gem \"rails\"\n");
        assert_eq!(
            Framework::detect_framework(tmp.path()),
            Some(Framework::Rails)
        );

        let tmp = TempDir::new().unwrap();
        write(tmp.path(), "pubspec.yaml", "dependencies:\n  flutter:\n");
        assert_eq!(
            Framework::detect_framework(tmp.path()),
            Some(Framework::Flutter)
        );
    }

    #[test]
    fn from_name_accepts_aliases() {
        assert_eq!(Framework::from_name("react"), Some(Framework::React));
        assert_eq!(Framework::from_name("next.js"), Some(Framework::NextJs));
        assert_eq!(
            Framework::from_name("ruby-on-rails"),
            Some(Framework::Rails)
        );
        assert_eq!(
            Framework::from_name("springboot"),
            Some(Framework::SpringBoot)
        );
        assert!(Framework::from_name("brainfuck").is_none());
    }

    #[test]
    fn scaffolds_each_framework() {
        for fw in Framework::ALL {
            let tmp = TempDir::new().unwrap();
            let target = tmp.path().join("proj");
            let written =
                crate::scaffold::scaffold_project(framework_spec(*fw).base, "demo-app", &target)
                    .unwrap();
            let _ = written;
            let target2 = tmp.path().join("fw");
            let written = scaffold_framework(*fw, "demo-app", &target2).unwrap();
            assert!(!written.is_empty(), "{fw:?} produced no files");
            for p in &written {
                assert!(p.exists(), "{fw:?} wrote missing file {}", p.display());
            }
        }
    }
}
