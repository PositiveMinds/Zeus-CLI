# AGENTS.md

Rust workspace (edition 2021 — no let-chains). Windows/PowerShell dev machine, Linux CI.

## Commands

- Build: `cargo build --release -p zeus-cli` → `target/release/zeus.exe`
- Test: `cargo test --workspace` (471 tests; warm run ~3 min, cold compile much slower — set generous timeouts)
- Lint (must be clean): `cargo clippy --workspace --all-targets -- -D warnings`
- Format (must be clean): `cargo fmt --all`
- CI: GitHub Actions — `Test` (ubuntu/windows/macos matrix) + `scaffold-build`. Inspect with `gh run list`, `gh run view <id> --log-failed`.

## Gotchas

- One ignored test in zeus-agent is intentional (portable-pty/Windows tracking, `terminal.rs:1135`). Do not "fix" it.
- Kotlin scaffold (`crates/zeus-lang/src/scaffold.rs`) must keep `kotlin("jvm")` version >= 2.0.0 — the CI runner uses Gradle 9, which removed the Conventions API that KGP 1.9 needs.
- `test.yml` probes sccache with a real compile and degrades to an uncached build during GitHub Actions cache outages. The probe must stay `continue-on-error`/non-fatal; `sccache --show-stats` soft-fails (exits 0) on a dead backend and must not be used for the gate.
- `scripts/verify-scaffolds.sh <zeus-binary>` builds every language scaffold (CI `scaffold-build`). PHP scaffold lints `index.php` (root), not `src/index.php`.
- PowerShell shows `NativeCommandError` styling for native stderr (git, cargo) — cosmetic; check exit codes, not the red text.
- Provider live tests use a python mock server that binds port 0; the port-report deadline is 60s (macOS CI cold-starts python slowly under parallel test threads).
