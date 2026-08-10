---
name: git-workflows
description: Safe and legible git — commits, branches, diffs, merges, rebases, conflict resolution, history repair.
version: 1.0.0
tags: [git, workflow, commits]
depends_on: []
---

# Git Workflows

Use git deliberately. The user's history is precious; never rewrite without
their say-so (no force-push, no interactive rebase unless asked).

## Before committing
- `git status` + `git diff` — inspect what changed.
- Stage **intended files only**, never secrets or build artifacts.
- Commit message style: match the repo (use `git log --oneline -10` to peek).
  Common good shape: short imperative subject + one-line audience body.

## Branching
- Feature branches off the current default. Keep them short-lived, small.
- **Never merge into a protected branch** without asking.

## Merge/rebase
- Prefer the repo's existing history style. If rebasing, `git rebase -i` is
  destructive — use non-interactive replay and fix conflicts file-by-file.
- On conflict: read both sides; keep both where semantics merge; ask when
  ambiguous.

## Repair
- `git stash` uncommitted work before risky experiments.
- Undo last commit without losing work: `git reset --soft` (keeps staging) or
  revert. Explain the distinction to the user.
- Detect detached HEAD / lost commits with `git reflog` (read-only).

## Hygiene
- Never commit `.env`, secrets, `node_modules`, build output. Update
  `.gitignore` when you notice gaps.
- Large binary/vendor trees don't belong in a repo.

## After any change
- Run the build + relevant tests before suggesting a commit.
- Only commit, branch, push, or open PRs when the user explicitly asks.