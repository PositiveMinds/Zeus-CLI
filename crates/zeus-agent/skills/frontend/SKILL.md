---
name: frontend
description: Build and refine user interfaces — components, state, styling, accessibility, and responsive layout.
version: 1.0.0
tags: [frontend, react, ui, css, accessibility]
depends_on: []
---

# Frontend Engineering

Build UIs that are correct, accessible, and consistent with the existing
component patterns. Match what's already there.

## Match existing patterns first
- Read an existing screen/component before writing a new one. Reuse its
  folder structure, styling approach (CSS modules, Tailwind, styled) and file
  conventions. Consistency beats cleverness.
- Reuse existing UI primitives (buttons, inputs, cards) instead of duplicating.

## State
- Local state for local concerns; lift only what must be shared. Respect the
  existing state strategy (React hooks / Redux / Zustand / context). Fetch at
  the right boundary and pass data down; handle loading + error both.

## Styling & layout
- Use responsive breakpoints — mobile-first. Test at narrow widths.
- Use spacing/size tokens from the design system rather than arbitrary px.
  Honor existing CSS variables / theme.

## Accessibility
- Real `<button>`/`<a>` semantics; not divs with onclick.
- Labels for all inputs; `aria-` attributes where the markup is non-obvious.
- Keyboard-navigable focus order, visible `:focus` states, colour contrast
  on text and icons.

## Performance
- Avoid re-creating handlers/filters in render hotpaths when it hurts.
- No big libraries for trivial features.
- Lazy-load heavy screens or charts where the app already does so.

## Verification
- Run the app's lint/typecheck.
- Run its test command and add tests if the feature is behavioral.
- If a dev server is running, verify in `browser` and describe what the user
  should see and confirm.