---
name: web-research
description: Structured web research — search, fetch, evaluate sources, extract evidence, write a recommendations report.
version: 1.0.0
tags: [research, web, report, evidence]
depends_on: []
---

# Web Research

Produce evidence-backed research, not link salad.

## Workflow
1. **Frame the question**: what decision or fact is being researched? Write it
   as a sentence. Define success criteria (e.g. "recommend one library with a
   maintained 2-year track record").
2. **Search**: use the available search/web_fetch tools across several
   phrasings and sources (official docs, package registries, primary sources,
   comparative posts). Favor primary over secondary.
3. **Fetch & read**: `web_fetch` the promising results. Note dates and
   version numbers — stale advice is a top failure mode. Cap context
   (max_chars) and read the relevant sections.
4. **Evaluate evidence**: for each claim, record source + when published +
   relevance. Mark conflicts explicitly.
5. **Synthesize**: group findings into conclusions, rate confidence, list
   open questions.
6. **Optionally persist**: if the user wants a durable artifact, save a
   markdown report in the project docs area (ask location).

## Report format
```markdown
# <Topic>
Question: <restated question>
## Findings        (claim + source + date)
## Trade-offs      (option A vs B, with evidence)
## Recommendation  (decision + why + confidence)
## Open questions
## Sources         (ordered by reliability)
```

## Discipline
- Never fabricate a citation or URL. If an URL failed, say so.
- A "no" result ("no good library exists") is a valid conclusion.
- Bound total effort; stop at enough evidence to decide.