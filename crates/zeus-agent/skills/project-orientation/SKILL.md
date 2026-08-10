---
name: project-orientation
description: Map an unfamiliar repository before any change — architecture, framework, entry points, tests, conventions.
version: 1.0.0
tags: [project, discovery, architecture, onboarding]
depends_on: []
---

# Project Orientation

Before touching a repo you have not deeply worked in, build a mental model.
The cost is small; the cost of editing blind is large.

## Steps
1. **Layout**: `glob` top-level dirs/files. Read README(s), any
   `AGENTS.md`/`CONTRIBUTING.md`/`docs/` overview. Identify the build tool
   (manifest: `Cargo.toml`, `package.json`, `pyproject.toml`, `go.mod`,
   `pom.xml`, `*.csproj`).
2. **Entry points**: find `main.rs`/`main.py`/`main.ts`/`index.tsx`,
   `app`/`src` roots, server bootstrap, router/controller registration.
3. **Framework & idioms**: note the web framework, ORM, state library, CSS
   approach, test framework. Check `settings`/`config` modules for env vars.
4. **Code index**: run `code_index` then `code_symbols` for the top modules
   so references resolve quickly during the change.
5. **Tests**: find the test layout and run ONE test to confirm the command and
   expected runtime (`cargo test -p x`, `npm test`, `pytest`).
6. **Conventions**: match existing naming, error handling, logging, module
   boundaries. A change that ignores local idioms reads as foreign and gets
   rejected in review.

## Output
Finish with a one-paragraph summary the user can confirm: stack + layout +
entry points + how to build/test + your chosen conventions. Then proceed.

## When NOT to use
- Tiny one-file edits where you already hold all the context.
- Code you just wrote yourself this session.