//! Hybrid retrieval: BM25-style keyword scoring fused with cosine similarity
//! over learned embeddings.

use crate::{Chunk, RagIndex};
use serde::{Deserialize, Serialize};

/// One search hit with the (merged) relevance score, normalized 0..=1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    pub chunk: Chunk,
    /// Keyword (BM25-ish) component, 0..=1.
    pub keyword: f32,
    /// Cosine component, 0..=1 (0 when no vectors are indexed).
    pub vector: f32,
    /// Combined score; higher is better.
    pub score: f32,
}

impl Hit {
    pub fn new(chunk: Chunk, keyword: f32, vector: f32) -> Self {
        let score = 0.6 * keyword + 0.4 * vector;
        Self {
            chunk,
            keyword,
            vector,
            score,
        }
    }
}

/// Tokenize a blob into lowercased word tokens (splits on non-alphanumeric
/// runs; keeps `_` so `Foo_bar` stays one token).
pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars().flat_map(|c| c.to_lowercase()) {
        if c.is_alphanumeric() || c == '_' {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Cosine similarity between two equal-length vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

impl RagIndex {
    /// Score every chunk against `query`. Keyword uses per-document term
    /// frequency with binary-IDF weighting; the vector component uses cosine
    /// similarity against a query embedding when vectors are indexed.
    /// Returns ranked hits (best first), capped at `k`.
    pub fn search(&self, query: &str, k: usize) -> Vec<Hit> {
        let q_tokens = tokens(query);
        if q_tokens.is_empty() || self.documents.is_empty() {
            return Vec::new();
        }
        let n_docs = self.documents.len() as f32;

        // Binary IDF per query term: rarer terms carry more weight.
        let idf: Vec<f32> = q_tokens
            .iter()
            .map(|qt| {
                let df = self
                    .documents
                    .iter()
                    .filter(|c| tokens(&c.text).iter().any(|t| t == qt))
                    .count();
                (n_docs / (df as f32).max(1.0)).ln() + 1.0
            })
            .collect();

        // Build a pseudo query vector: the term-weighted centroid of the
        // vectors of every chunk that matched any query token. Reasonable
        // without embedding the query itself; exact when vectors are 1-hot.
        let query_vec: Option<Vec<f32>> = self
            .vectors
            .as_ref()
            .filter(|vs| vs.iter().any(|v| !v.is_empty()) && vs[0].len() > 0)
            .map(|vs| {
                let dim = vs[0].len();
                let mut acc = vec![0.0f32; dim];
                let mut wsum = 0.0f32;
                for (i, doc) in self.documents.iter().enumerate() {
                    let hits = q_tokens
                        .iter()
                        .filter(|qt| tokens(&doc.text).iter().any(|t| t == *qt))
                        .count();
                    if hits > 0 && i < vs.len() {
                        for (a, b) in acc.iter_mut().zip(vs[i].iter()) {
                            *a += b * hits as f32;
                        }
                        wsum += hits as f32;
                    }
                }
                if wsum > 0.0 {
                    for a in &mut acc {
                        *a /= wsum;
                    }
                }
                acc
            });

        let mut scored: Vec<(Chunk, f32, f32)> = self
            .documents
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let doc_tokens = tokens(&chunk.text);
                let mut kw = 0.0f32;
                for (qt, qi) in q_tokens.iter().zip(idf.iter()) {
                    let tf = doc_tokens.iter().filter(|t| *t == qt).count() as f32;
                    kw += qi * (tf / (tf + 1.5));
                }
                // Keyword component: BM25 saturates; scale by inverse doc
                // count for a rough 0..=1 normalization.
                let keyword = (kw / n_docs).max(0.0).min(1.0);
                let vector = match (query_vec.as_ref(), self.vectors.as_ref()) {
                    (Some(qv), Some(vs)) if i < vs.len() => cosine(qv, &vs[i]),
                    _ => 0.0,
                };
                (chunk.clone(), keyword, vector)
            })
            .collect();

        scored.sort_by(|a, b| {
            let sa = a.1 * 0.6 + a.2 * 0.4;
            let sb = b.1 * 0.6 + b.2 * 0.4;
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored
            .into_iter()
            .take(k.max(1))
            .map(|(chunk, keyword, vector)| Hit::new(chunk, keyword, vector))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_split_on_case_and_punctuation() {
        assert_eq!(tokens("Foo::bar BAZ-baz"), vec!["foo", "bar", "baz", "baz"]);
    }

    #[test]
    fn keyword_search_ranks_relevant_higher() {
        let mut idx = crate::RagIndex::new(std::path::PathBuf::from("/fake"));
        idx.add_chunk(crate::Chunk::new(
            "lib.rs".into(),
            0,
            "fn add(a, b) { a + b }".into(),
        ));
        idx.add_chunk(crate::Chunk::new(
            "util.rs".into(),
            1,
            "fn subtract(a, b) { a - b }".into(),
        ));
        let hits = idx.search("add", 1);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].chunk.text.contains("add"));
    }

    #[test]
    fn idf_penalizes_common_terms() {
        let mut idx = crate::RagIndex::new(std::path::PathBuf::from("/fake"));
        idx.add_chunk(crate::Chunk::new("a.rs".into(), 0, "fn the_value".into()));
        idx.add_chunk(crate::Chunk::new("b.rs".into(), 1, "fn the_other".into()));
        idx.add_chunk(crate::Chunk::new("c.rs".into(), 2, "fn rare_symbol".into()));
        // Searching a term unique to c.rs should rank c above a.
        let hits = idx.search("rare_symbol", 5);
        assert!(hits[0].chunk.path.to_string_lossy().ends_with("c.rs"));
    }

    #[test]
    fn vector_component_boosts_hits_when_present() {
        let mut idx = crate::RagIndex::new(std::path::PathBuf::from("/fake"));
        idx.add_chunk(crate::Chunk::new("d.rs".into(), 0, "keyword token".into()));
        idx.add_chunk(crate::Chunk::new("e.rs".into(), 1, "other stuff".into()));
        // Vectors manually aligned so token 0 is "keyword".
        let vs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        idx.set_vectors(vs);
        let hits = idx.search("keyword", 5);
        assert_eq!(hits[0].vector, 1.0);
    }
}
