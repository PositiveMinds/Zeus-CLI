---
name: documentation
description: Write and update docs that match reality — README, architecture notes, API docs, migration guides.
version: 1.0.0
tags: [docs, readme, technical-writing]
depends_on: []
---

# Documentation

Good docs describe what the code actually does from a reader's perspective,
and they're kept in sync with the code.

## When to write
- User asked (README, comments, API docs, changelog, guide).
- A build defined in `build-app` completed and touched user-visible surface.
- Docs exist but now disagree with the code.

## Principles
- **Match reality**: after any code change, re-read the old doc and fix what
  changed. A doc that lies is worse than no doc.
- **Reader-first**: who reads this and what will they try to do? Lead with
  results, commands, invariants. Avoid praising prose.
- **Show, don't tell**: include cmd from the shell, small examples, a
  table-of-contents for long docs.
- **Consistency**: keep the repo's doc format, heading style, and file
  placement. `README.md`? `docs/`? Ask/infer from what exists.

## Deliverables
- README: what, quickstart (commands that actually work), config/env, usage
  example, testing, license.
- API docs: endpoints, request/response shape, errors, auth — every field.
- Architecture: components + data flow + failure modes (box-and-line, not
  essay).
- Changelog/guides: verifiable date/version — never invent versions.

## Verification
- Every command in the doc must have been actually run and work in this repo.
- "Run this" steps include the exact invocation zeus used.
- Do not document things that don't exist yet, except as clearly-marked
  TODO/stubs.