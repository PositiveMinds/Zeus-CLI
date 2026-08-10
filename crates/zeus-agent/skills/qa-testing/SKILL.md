---
name: qa-testing
description: Write and repair tests — unit, integration, E2E — diagnose failures, isolate flakiness.
version: 1.0.0
tags: [testing, qa, unit, integration, e2e]
depends_on: [project-orientation]
---

# QA & Testing

Make tests a first-class deliverable of any code change, not an afterthought.

## Write tests like a QA engineer
- One behavior per test; descriptive names that read as intent
  (`returns_404_when_resource_missing`).
- **Unit**: pure functions, isolated modules, mocks only at boundaries.
- **Integration**: real DB / HTTP where the codebase's harness supports it.
- **E2E**: happy path through the UI (framework's tooling — do NOT vendor a
  new browser framework).
- **Regression for bug fixes**: first write a test that reproduces the bug,
  watch it fail, fix, watch it pass.

## Follow the house style
- Mirror the existing test layout, naming, and assertion style. Do not mix.
- Reuse existing helpers/fixtures; extend, don't duplicate.

## Repairing failing tests
- Read the failure FIRST (assertion diff / stack). Decide the root cause:
  - Production behavior changed → was the behavior the intent? Then fix test.
  - Test is stale → update the expectation to the new contract.
  - Test is flaky (timing/ordering/typography) → make it deterministic:
    await the actual condition, flush state, avoid sleeps.
- Never disable a test to make CI green unless you leave a tracking issue,
  and say so loudly to the user.

## Verification
- Run the project's test command (ask, `cargo test`, `npm test`, `pytest`,
  `go test`…) and confirm the full suite passes, and that you didn't just
  hide failures.
- Report suite count + pass/fail clearly.