---
name: security
description: Harden applications — audit authn/authz, secrets, injection vectors, and dependency vulnerabilities.
version: 1.0.0
tags: [security, audit, owasp, secrets]
depends_on: []
---

# Security Hardening

Treat the running app as hostile-input surface. Each pass produces a short
issue list ordered by severity with concrete evidence and fixes.

## Critical, always check
1. **Secrets** — scan for hardcoded passwords, API keys, tokens, connection
   strings (in the changed files; grep the touched tree). Never print them.
   Fail the task if you introduced one.
2. **SQL injection** — grep for string-concatenated SQL; verify parameterized
   queries / ORMs everywhere.
3. **Authn/authz** — every mutating route requires authentication; every
   object access is authorized. Check for endpoints that trust client-supplied
   IDs without ownership checks (IDOR).
4. **XSS** — user content must be escaped at the boundary. Check compiled
   HTML, dangerouslySetInnerHTML (if frontend), error pages.
5. **CSRF** — state-changing endpoints are protected by token/same-site cookie
   policy in web apps.
6. **Command injection** — shell calls built from user input must quote/escape.

## Secondary
- **Dependencies**: `npm audit` / `cargo audit` / `pip-audit` if configured.
- **Headers**: `CORS` not `*` for credentialed endpoints; note missing
  security headers.
- **Rate limiting** on auth endpoints prevention for login loops.
- **File uploads** — extension/content-type validation, no executable dirs.

## Output format
List severities with file:line proof:
- 🔴 Critical — exploitable remotely without auth (or auth bypass)
- 🟠 High — exploitable with auth; low bar
- 🟡 Medium — config/defense-in-depth gaps
- 🟢 Info — hygiene

For each: `path:line` — one-line cause — suggested fix. Always provide a fix.
Never claim a finding you did not verify by reading the code.

## After the pass
Fix what the user asked to fix. Re-run tests. Say what remains open.