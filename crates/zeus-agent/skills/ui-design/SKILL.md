---
name: ui-design
description: Design and refine UI from source designs and screenshots — colors, spacing, typography, layout parity.
version: 1.0.0
tags: [ui, design, figma, mockup, vision]
depends_on: [frontend]
---

# UI Design from Source

Turn a visual reference (screenshot, Figma export, PNG mockup, web page) into
matching, production-quality UI — or critique and improve an existing screen
without touching backend logic.

## When to use
- "Make it look like <image>".
- "Recreate this dashboard/mobile screen."
- "Match the brand: colors, fonts, spacing."
- Design/improve pass on an existing screen.

## Workflow
1. **Look at the source**: `read_image` (screenshot/mockup) or `web_fetch`
   the reference design. Keep the image/visual in context.
2. **Extract the design tokens** — be explicit and state them in your response:
   - Palette: exact hex for primary/secondary/neutral/background/accent.
   - Type: font family, weights, and sizes for heading/body/caption; line
     heights.
   - Spacing: base unit, gaps, radii, paddings.
   - Layout: structure (sidebar/topbar/cards), grid, breakpoints.
3. **Map to the codebase**: reuse the existing theme/CSS-vars/tokens. Add
   tokens only when the source requires values the app lacks.
4. **Build the screen** in components matching repo conventions. Match the
   reference's proportions and hierarchy — that's what "looks like the
   mockup" means.
5. **Verify by eye**: if a dev server runs, open it in `browser`, describe
   what to confirm, and iterate on spacing/colour until it matches.

## Critique mode
When asked to critique (not build), produce:
- Strengths (concrete).
- Issues ranked: layout/spacing, hierarchy, contrast/accessibility, motion,
  responsiveness.
- Suggested fix per issue with before/after rationale.

## Pitfalls
- Never invent fonts from thin air — read the computed/exported styles.
- Accessibility isn't optional: contrast and focus states still apply.
- Keep backend intact — this skill is UI-only; wire using existing
  data/fetch patterns from `frontend` skill.