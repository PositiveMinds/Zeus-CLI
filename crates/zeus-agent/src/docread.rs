//! Document text extraction — turn office/binaries into readable text for the
//! model. Supports text formats (.md/.txt/.log/.json/...), HTML, PDF, DOCX,
//! XLSX, PPTX.

use std::io::Read;
use std::path::Path;

use quick_xml::Reader;
use quick_xml::events::Event;

/// Result of extracting a document.
#[derive(Debug, Default)]
pub struct Document {
    /// Human label describing what was read, e.g. `pdf (3 pages)`, `xlsx (2
    /// sheets)`.
    pub summary: String,
    /// Extracted text (sheets/pages separated by clear markers).
    pub text: String,
}

/// Extract structured text from a document at `path`, chosen by extension.
pub fn extract(path: &Path, max_chars: usize) -> Result<Document, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if len > 200 * 1024 * 1024 {
        return Err("document exceeds the 200MB read limit".into());
    }

    let mut doc = match ext.as_str() {
        "pdf" => extract_pdf(path)?,
        "docx" => extract_ooxml(path, OfficeKind::Word)?,
        "pptx" => extract_ooxml(path, OfficeKind::Slides)?,
        "xlsx" => extract_ooxml(path, OfficeKind::Sheet)?,
        "html" | "htm" => {
            let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
            Document {
                summary: "html".into(),
                text: super::tools::strip_html_pub(&raw),
            }
        }
        // Plain-text family: md, txt, log, json, csv, toml, yaml, xml, hcl…
        _ => {
            let raw = match std::fs::read_to_string(path) {
                Ok(t) => t,
                Err(_) => {
                    // Latin-1-ish decode so we never hard-fail on odd encodings.
                    let bytes = std::fs::read(path).map_err(|e| format!("read error: {e}"))?;
                    bytes.iter().map(|&b| b as char).collect()
                }
            };
            Document {
                summary: if ext.is_empty() { "text".into() } else { ext.clone() },
                text: raw,
            }
        }
    };

    if doc.text.chars().count() > max_chars {
        let trunc: String = doc.text.chars().take(max_chars).collect();
        doc.text = format!("{trunc}\n… (truncated — pass max_chars to raise)");
    }
    Ok(doc)
}

fn extract_pdf(path: &Path) -> Result<Document, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("pdf read: {e}"))?;
    // pdf-extract's own text layer extraction handles page splitting + layout.
    let text = pdf_extract::extract_text_from_mem(&bytes)
        .map_err(|e| format!("unable to parse PDF: {e}"))?;
    if text.trim().is_empty() {
        return Err(
            "PDF parsed but no extractable text found (scanned/image-only PDF? use read_image)".into(),
        );
    }
    Ok(Document {
        summary: format!("pdf ({})", page_count_text(&text)),
        text: collapse_ws(&text),
    })
}

/// Rough page marker detection for the summary (relies on the ↵ separator
/// pdf-extract inserts between pages).
fn page_count_text(text: &str) -> String {
    let pages = text.split("\n\n").count().max(1);
    if pages == 1 {
        "single page".into()
    } else {
        format!("{pages} pages (approx)")
    }
}

/// Which office format we're unpacking.
enum OfficeKind {
    Word,
    Slides,
    Sheet,
}

fn extract_ooxml(path: &Path, kind: OfficeKind) -> Result<Document, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("not a valid zip/office document: {e}"))?;

    match kind {
        OfficeKind::Word => {
            let mut doc_xml = archive
                .by_name("word/document.xml")
                .map_err(|_| "docx missing word/document.xml".to_string())?;
            let mut raw = String::new();
            doc_xml.read_to_string(&mut raw).map_err(|e| e.to_string())?;
            let paragraphs = count_tag(&raw, "w:p");
            let text = text_of(&raw, &["w:t"], &["w:p"]);
            Ok(Document {
                summary: format!("docx ({paragraphs} paragraphs)"),
                text,
            })
        }
        OfficeKind::Slides => {
            let mut slides: Vec<(u32, String)> = Vec::new();
            for i in 0..archive.len() {
                let name = archive
                    .by_index(i)
                    .map_err(|e| e.to_string())?
                    .name()
                    .to_string();
                if let Some(num) = name
                    .strip_prefix("ppt/slides/slide")
                    .and_then(|rest| rest.strip_suffix(".xml"))
                    .and_then(|n| n.parse::<u32>().ok())
                {
                    slides.push((num, name));
                }
            }
            slides.sort_by_key(|(n, _)| *n);
            if slides.is_empty() {
                return Err("pptx contains no slides".into());
            }
            let slide_count = slides.len();
            let mut out = Vec::new();
            for (num, name) in &slides {
                let mut f = archive.by_name(&name).map_err(|e| e.to_string())?;
                let mut raw = String::new();
                f.read_to_string(&mut raw).map_err(|e| e.to_string())?;
                let txt = text_of(&raw, &["a:t"], &["a:p"]);
                if !txt.trim().is_empty() {
                    out.push(format!("\n--- slide {num} ---\n{txt}"));
                }
            }
            Ok(Document {
                summary: format!("pptx ({slide_count} slides)"),
                text: out.join("\n"),
            })
        }
        OfficeKind::Sheet => {
            let mut sheets: Vec<(u32, String)> = Vec::new();
            for i in 0..archive.len() {
                let name = archive
                    .by_index(i)
                    .map_err(|e| e.to_string())?
                    .name()
                    .to_string();
                if name.starts_with("xl/worksheets/sheet") && name.ends_with(".xml") {
                    let num = name
                        .strip_prefix("xl/worksheets/sheet")
                        .and_then(|n| n.strip_suffix(".xml"))
                        .and_then(|n| n.parse::<u32>().ok())
                        .unwrap_or(i as u32);
                    sheets.push((num, name));
                }
            }
            sheets.sort_by_key(|(n, _)| *n);
            if sheets.is_empty() {
                return Err("xlsx contains no worksheets".into());
            }
            let sheet_count = sheets.len();
            let shared = read_shared_strings(&mut archive)?;
            let mut out = Vec::new();
            for (num, name) in &sheets {
                let mut f = archive.by_name(&name).map_err(|e| e.to_string())?;
                let mut raw = String::new();
                f.read_to_string(&mut raw).map_err(|e| e.to_string())?;
                let grid = sheet_rows(&raw, &shared);
                if !grid.trim().is_empty() {
                    out.push(format!("\n--- sheet {num} ---\n{grid}"));
                }
            }
            Ok(Document {
                summary: format!("xlsx ({sheet_count} sheets)"),
                text: out.join("\n"),
            })
        }
    }
}

fn read_shared_strings(
    archive: &mut zip::ZipArchive<std::fs::File>,
) -> Result<Vec<String>, String> {
    let Ok(mut f) = archive.by_name("xl/sharedStrings.xml") else {
        return Ok(Vec::new());
    };
    let mut raw = String::new();
    f.read_to_string(&mut raw).map_err(|e| e.to_string())?;
    // Each <si> holds one or more <t> runs; join text per <si>.
    let mut strings = Vec::new();
    let mut reader = Reader::from_str(&raw);
    let mut cur = String::new();
    let mut in_si = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                if e.name().as_ref() == b"si" {
                    in_si = true;
                    cur.clear();
                }
            }
            Ok(Event::Text(e)) => {
                if in_si {
                    if let Ok(decoded) = e.unescape() {
                        cur.push_str(&decoded);
                    }
                }
            }
            Ok(Event::End(e)) => {
                if e.name().as_ref() == b"si" {
                    let s = cur.trim();
                    if !s.is_empty() {
                        strings.push(s.to_string());
                    }
                    in_si = false;
                }
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
    }
    Ok(strings)
}

/// Pull text out of the listed `text_tags`, breaking a chunk whenever a
/// `break_on` tag closes (paragraph / no-run / cell boundary).
fn text_of(xml: &str, text_tags: &[&str], break_on: &[&str]) -> String {
    let mut out = String::new();
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut in_text = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                if text_tags.iter().any(|t| *t == name) {
                    in_text = true;
                }
            }
            Ok(Event::Text(e)) => {
                if in_text {
                    if let Ok(d) = e.unescape() {
                        out.push_str(&d);
                    }
                }
            }
            Ok(Event::End(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                if text_tags.iter().any(|t| *t == name) {
                    in_text = false;
                }
                if break_on.iter().any(|b| *b == name) {
                    out.push('\n');
                }
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
    }
    collapse_ws(&out)
}

/// Count `<tag` openings.
fn count_tag(xml: &str, tag: &str) -> usize {
    let needle = format!("<{tag}");
    xml.matches(&needle).count()
}

/// Extract a `<row …>…</row>` grid of `<c>` cells into `row N: a | b`.
fn sheet_rows(xml: &str, shared: &[String]) -> String {
    let mut grid: Vec<Vec<String>> = Vec::new();
    let mut rest = xml;
    loop {
        let Some(open) = rest.find("<row") else { break };
        let after_open = &rest[open..];
        let Some(gt) = after_open.find('>') else { break };
        let chunk_start = open + gt + 1;
        let tail = &rest[chunk_start..];
        let Some(close) = tail.find("</row>") else { break };
        let chunk = &tail[..close];
        let mut row = Vec::new();
        let mut from = 0usize;
        loop {
            let Some(cs) = chunk[from..].find("<c") else { break };
            let cstart = from + cs;
            let ctail = &chunk[cstart..];
            let Some(ce) = ctail.find("</c>") else { break };
            let cell = &ctail[..ce];
            row.push(cell_value(cell, shared));
            from = cstart + ce + 4;
        }
        grid.push(row);
        rest = &rest[chunk_start + close..];
    }

    let mut out = String::new();
    for (ri, row) in grid.iter().enumerate() {
        if row.iter().any(|c| !c.is_empty()) {
            out.push_str(&format!("row {}: {}\n", ri + 1, row.join(" | ")));
        }
    }
    collapse_ws(&out)
    .split('\n')
    .collect::<Vec<_>>()
    .join("\n")
}

/// A single cell: `<v>` (number or shared-string index) else inline `<is><t>`.
fn cell_value(cell: &str, shared: &[String]) -> String {
    const V_OPEN: &str = "<v>";
    const V_CLOSE: &str = "</v>";
    if let Some(start) = cell.find(V_OPEN) {
        let inner = &cell[start + V_OPEN.len()..];
        if let Some(end) = inner.find(V_CLOSE) {
            let v = &inner[..end];
            let v = v.trim();
            if let Ok(idx) = v.parse::<usize>() {
                if let Some(s) = shared.get(idx) {
                    return s.clone();
                }
            }
            return v.to_string();
        }
    }
    let mut inline = text_of(cell, &["t"], &[]);
    if inline.is_empty() {
        inline = cell.trim().to_string();
    }
    inline
}

/// Collapse whitespace and drop empty lines.
fn collapse_ws(s: &str) -> String {
    s.split('\n')
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn text_files_extract_verbatim() {
        let tmp = TempDir::new().unwrap();
        let f = tmp.path().join("notes.md");
        std::fs::write(&f, "# Title\n\nHello world\n").unwrap();
        let doc = extract(&f, 10_000).unwrap();
        assert!(doc.text.contains("Hello world"));
        assert_eq!(doc.summary, "md");
    }

    #[test]
    fn truncation_respects_max_chars() {
        let tmp = TempDir::new().unwrap();
        let f = tmp.path().join("seed.txt");
        std::fs::write(&f, "abcdefghijklmnopqrstuvwxyz").unwrap();
        let doc = extract(&f, 10).unwrap();
        assert!(doc.text.chars().count() <= 10 + 40);
        assert!(doc.text.contains("truncated"));
    }

    #[test]
    fn missing_file_errors() {
        let tmp = TempDir::new().unwrap();
        let f = tmp.path().join("nope.pdf");
        assert!(extract(&f, 100).is_err());
    }

    #[test]
    fn docx_round_trips_text() {
        // Build a minimal docx by hand: [Content_Types].xml, _rels/, word/.
        let tmp = TempDir::new().unwrap();
        let f = tmp.path().join("mini.docx");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&f).unwrap());
        let opts = zip::write::SimpleFileOptions::default();
        zip.add_directory("_rels/", opts).unwrap();
        zip.add_directory("word/", opts).unwrap();
        zip.start_file("word/document.xml", opts).unwrap();
        std::io::Write::write_all(
            &mut zip,
            b"<?xml version=\"1.0\"?><w:document xmlns:w=\"urn\"><w:body><w:p><w:r><w:t>Hello Document</w:t></w:r></w:p></w:body></w:document>",
        )
        .unwrap();
        zip.start_file("[Content_Types].xml", opts).unwrap();
        std::io::Write::write_all(
            &mut zip,
            b"<?xml version=\"1.0\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"/>",
        )
        .unwrap();
        zip.finish().unwrap();

        let doc = extract(&f, 10_000).unwrap();
        assert!(doc.text.contains("Hello Document"), "was: {}", doc.text);
    }
}