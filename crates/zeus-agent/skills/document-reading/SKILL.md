---
name: document-reading
description: Extract and interpret real-world documents — PDF, DOCX, XLSX, PPTX — then act on their content.
version: 1.0.0
tags: [documents, pdf, extraction, office]
depends_on: [web-research]
---

# Document Reading

The model reads text; the world ships binary. Bridge that gap with the
`read_document` tool and structure.

## When to use
- The user references a file: report.pdf, notes.docx, data.xlsx, survey.pptx.
- A task says "do X per the attached spec/mockup".

## Workflow
1. **Identify the file** (glob/read_document). Check extension and size.
2. **Extract** with `read_document { "path": "<file>" }` — returns text
   content with the source name.
3. **Interpret**:
   - PDFs: headings, tables of contents, page structure. Don't trust the
     extraction ordering blindly when columns are involved.
   - Spreadsheets (xlsx): list sheet names; read the relevant sheet and its
     header row; summarize column meanings before computing.
   - Word files: paragraphs and tables in order.
4. **Ground the work in it**: excerpt the governing passage into your working
   context before coding. Quote chapter/verse (page/sheet/cell) when you
   reference the doc.
5. **Verification**: after your change, re-read the relevant section and
   confirm the output matches the source doc's intent.

## Non-text content
If the document is principally visual (a design PDF/slides, a mockup),
consider `read_image`/`view_image` to get the visual; use the document text
for the labeled/structure content, the visual for layout/spacing/colour.

## Deliverable
- A summary of the doc you pulled (sections, counts, schemas).
- Your work with each assertion traceable back to a doc excerpt.