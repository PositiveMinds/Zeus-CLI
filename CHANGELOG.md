# Changelog

All notable changes to Zeus are tracked here. `zeus update` prints the notes
for the newest release before asking to install it; the full history lives in
this file. Versions follow `major.minor.patch`; each entry covers the
user-visible changes since the previous release.

## Unreleased

- **Sessions housekeeping**: `zeus sessions remove|rm <id>`, `prune
  --older-than <days>`, and `label <id> <name>` (empty name clears the
  label). The TUI `/sessions` picker shows labels, and `v` on a session
  opens it read-only for browsing (scroll with arrow/paging keys, Esc back
  to the live chat).
- **Whole-word approvals**: permission prompts now read `approve / session /
  cancel` (or accept `yes`/`y`, `session`/`s`) instead of bare y/n.
- **`zeus doctor` live checks**: each configured provider now performs a
  real model-list call (12s timeout) so dead API keys, exhausted credits,
  and down local servers show up in the Providers table instead of as a
  later auth error mid-chat.
- **`zeus update` release notes**: `zeus update`/`--check` prints what's new
  in the pending release fetched from GitHub before installing.
- **`/export` everywhere**: the current conversation can be exported to
  Markdown from the REPL (`/export`) and the TUI, including finished
  sessions via `zeus sessions export <id>`.

## 0.1.9

- Session storage (`zeus sessions list/show/export`), background tasks
  (`zeus bg start/logs/follow/kill`), and `zeus doctor`.