//! Benchmarks for zeus-rag: chunking, indexing, persistence, and search.

use criterion::{criterion_group, criterion_main, Criterion};
use std::fs;
use tempfile::TempDir;
use zeus_rag::{PersistedRagIndex, RagIndex};

/// Generate a synthetic Rust source file with `n` functions.
fn synthetic_rs(n: usize) -> String {
    let mut lines = Vec::with_capacity(n * 4);
    for i in 0..n {
        lines.push(format!("/// Function {i} documentation."));
        lines.push(format!("pub fn func_{i}(x: u32) -> u32 {{ x + {i} }}"));
        lines.push(String::new());
    }
    lines.join("\n")
}

/// Create a temp project with `file_count` source files, each with
/// `funcs_per_file` functions.
fn setup_project(file_count: usize, funcs_per_file: usize) -> TempDir {
    let dir = TempDir::new().unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    let body = synthetic_rs(funcs_per_file);
    for i in 0..file_count {
        fs::write(src.join(format!("module_{i}.rs")), &body).unwrap();
    }
    dir
}

fn bench_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking");

    for (file_count, funcs) in [(10, 20), (50, 50), (200, 100)] {
        let dir = setup_project(file_count, funcs);
        let label = format!("{file_count}files_{funcs}funcs");

        group.bench_function(format!("from_project_{label}"), |b| {
            b.iter(|| RagIndex::from_project(dir.path(), 800, 80));
        });
    }
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");

    for (file_count, funcs) in [(10, 20), (50, 50), (200, 100)] {
        let dir = setup_project(file_count, funcs);
        let index = RagIndex::from_project(dir.path(), 800, 80);
        let label = format!("{file_count}files_{funcs}funcs");

        group.bench_function(format!("keyword_{label}"), |b| {
            b.iter(|| index.search("function", 5));
        });

        group.bench_function(format!("multi_word_{label}"), |b| {
            b.iter(|| index.search("public function documentation", 5));
        });
    }
    group.finish();
}

fn bench_index_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistence");

    for (file_count, funcs) in [(10, 20), (50, 50)] {
        let dir = setup_project(file_count, funcs);
        let label = format!("{file_count}files_{funcs}funcs");

        group.bench_function(format!("save_{label}"), |b| {
            let index = RagIndex::from_project(dir.path(), 800, 80);
            let persisted = PersistedRagIndex::from_index(&index);
            b.iter(|| persisted.save(dir.path()).unwrap());
        });

        group.bench_function(format!("load_{label}"), |b| {
            let index = RagIndex::from_project(dir.path(), 800, 80);
            let persisted = PersistedRagIndex::from_index(&index);
            persisted.save(dir.path()).unwrap();
            b.iter(|| PersistedRagIndex::load(dir.path()).unwrap());
        });

        group.bench_function(format!("roundtrip_{label}"), |b| {
            b.iter(|| {
                let index = RagIndex::from_project(dir.path(), 800, 80);
                let persisted = PersistedRagIndex::from_index(&index);
                persisted.save(dir.path()).unwrap();
                let loaded = PersistedRagIndex::load(dir.path()).unwrap();
                let _ = loaded.into_index();
            });
        });
    }
    group.finish();
}

fn bench_refresh(c: &mut Criterion) {
    let mut group = c.benchmark_group("refresh");

    for (file_count, funcs) in [(10, 20), (50, 50)] {
        let dir = setup_project(file_count, funcs);
        let label = format!("{file_count}files_{funcs}funcs");

        // Build initial index and persist it.
        let index = RagIndex::from_project(dir.path(), 800, 80);
        let persisted = PersistedRagIndex::from_index(&index);
        persisted.save(dir.path()).unwrap();

        // Touch one file to make the index stale.
        let first_file = dir.path().join("src/module_0.rs");
        let mut content = fs::read_to_string(&first_file).unwrap();
        content.push_str("\n// touched\n");
        fs::write(&first_file, &content).unwrap();

        group.bench_function(format!("incremental_{label}"), |b| {
            b.iter(|| {
                let mut p = PersistedRagIndex::load(dir.path()).unwrap();
                p.refresh(800, 80);
            });
        });

        // Force rebuild (full walk).
        group.bench_function(format!("full_rebuild_{label}"), |b| {
            b.iter(|| {
                let mut p = PersistedRagIndex::load(dir.path()).unwrap();
                p.refresh(800, 80);
                // Simulate a full rebuild by re-walking.
                let _ = RagIndex::from_project(dir.path(), 800, 80);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_chunking,
    bench_search,
    bench_index_persistence,
    bench_refresh
);
criterion_main!(benches);
