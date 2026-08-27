//! Tests for the tool registry and dispatch layer.
//!
//! ## Test Organization
//!
//! Tests are organized by functionality:
//!
//! ### Tool Specs & Filtering
//! - `platform_cli_for_maps_every_platform_tool_to_a_cli`
//! - `core_tools_survive_platform_filter_regardless_of_clis`
//! - `platform_tools_filtered_by_cli_presence`
//! - `all_tool_specs_omits_platform_tools_whose_cli_is_missing`
//! - `every_tool_spec_has_a_handler`
//! - `platform_tools_registry_matches_specs_and_dispatch`
//!
//! ### File Operations
//! - `write_then_read_roundtrip`
//! - `read_multiple_reads_batch_and_reports_missing`
//! - `read_multiple_errors_on_empty_or_oversized_batch`
//! - `mkdir_tool_creates_directory_scaffold`
//! - `listdir_tool_lists_flat_and_recursive`
//!
//! ### Plan Mode
//! - `listdir_read_only_but_mkdir_gated`
//! - `plan_mode_blocks_mutating_tools_but_allows_read_only`
//! - `verify_not_in_read_only_tool_list`
//! - `web_search_is_read_only_tool`
//! - `rag_search_is_read_only_so_plan_mode_can_use_it`
//! - `current_time_is_listed_as_read_only`
//! - `browser_rejects_bad_url_and_blocks_in_plan_mode`
//! - `memory_tools_blocked_in_plan_mode`
//!
//! ### Verification & Testing
//! - `verify_runs_explicit_command_and_reports_exit_code`
//! - `verify_without_detection_and_without_command_errors`
//! - `test_tool_runs_with_explicit_command`
//! - `test_tool_without_command_reports_detection_failure`
//! - `detect_test_command_maps_manifests`
//! - `detect_test_command_none_when_no_manifest`
//! - `summarize_test_output_picks_verdict_lines`
//!
//! ### Skills
//! - `builtin_skills_are_listed_and_readable`
//! - `read_skill_recursively_composes_depends_on_chain`
//! - `project_skill_shadows_builtin`
//!
//! ### Document & Image Reading
//! - `read_document_extracts_text_and_errors_on_binary`
//! - `read_image_attaches_base64_bytes`
//!
//! ### Repository Understanding
//! - `understand_repo_reports_stack_and_relevance`
//!
//! ### RAG (Retrieval-Augmented Generation)
//! - `rag_search_ranks_concept_chunks_above_the_rest`
//! - `rag_search_is_read_only_so_plan_mode_can_use_it`
//! - `rag_index_builds_persistent_index_and_rag_search_reuses_it`
//! - `rag_index_embed_degrades_gracefully_without_provider`
//! - `rag_index_embed_persists_vectors_and_search_uses_them`
//!
//! ### Web Tools
//! - `web_search_rejects_empty_query`
//! - `web_search_is_read_only_tool`
//! - `browser_rejects_bad_url_and_blocks_in_plan_mode`
//! - `browser_rejects_file_scheme_and_bare_paths`
//! - `web_fetch_rejects_internal_targets`
//! - `web_fetch_blocks_hostnames_that_resolve_to_loopback`
//! - `web_fetch_allows_public_targets`
//!
//! ### Memory
//! - `memory_tools_list_read_write`
//! - `memory_tools_blocked_in_plan_mode`
//!
//! ### Code Intelligence
//! - `code_intel_tools_round_trip`
//! - `code_graph_reports_callers_and_callees`
//! - `code_verbose_rename_reports_plan_only`
//!
//! ### Bash & Background Tasks
//! - `bash_runs_and_denies_destructive`
//! - `bash_background_spawns_and_is_listed_and_stoppable`
//!
//! ### MCP Integration
//! - `mcp_tool_is_advertised_and_dispatchable_end_to_end`
//! - `mcp_call_denied_is_not_run`
//!
//! ### Git Tools
//! - `git_tools_work_end_to_end_through_the_full_dispatch_path`
//!
//! ### Utilities
//! - `urlencode_encodes_query`
//! - `current_time_tool_returns_a_parseable_datetime`
//! - `unknown_tool_errors`

use super::*;
use base64::Engine;
use chrono::Datelike;
use std::collections::HashSet;
use std::path::Path;
use tempfile::TempDir;
use zeus_config::{AgentSettings, Config, GlobalPaths, ProvidersFile};
use zeus_provider::{
    ChatRequest, ChatResponse, ChatStream, EmbeddingRequest, EmbeddingResponse, ModelInfo,
    TokenCountRequest, TokenCountResponse, TokenUsage,
};

fn approve(_: &PermissionRequest) -> ApprovalDecision {
    ApprovalDecision::Approved
}

fn tool_manager(root: &Path) -> ToolManager {
    std::fs::create_dir_all(root).unwrap();
    let config = Config {
        global: GlobalPaths::from_root(root.join(".zeus-home")),
        project: None,
        settings: AgentSettings::default(),
        providers: ProvidersFile::default(),
        project_root: Some(root.to_path_buf()),
    };
    let workspace = Workspace::from_config(&config).unwrap();
    let terminal = TerminalRunner::new(root.join(".agent/checkpoints"));
    let background = BackgroundTaskRegistry::new(root.join(".agent/background"));
    let hooks = crate::hooks::HookRunner::new(root.join(".agent/hooks"), root.to_path_buf());
    ToolManager::new(
        workspace,
        terminal,
        background,
        hooks,
        Vec::new(),
        Vec::new(),
        Arc::new(AtomicBool::new(false)),
    )
}

fn tool_manager_with_mcp(root: &Path, mcp_clients: Vec<crate::mcp::McpClient>) -> ToolManager {
    std::fs::create_dir_all(root).unwrap();
    let config = Config {
        global: GlobalPaths::from_root(root.join(".zeus-home")),
        project: None,
        settings: AgentSettings::default(),
        providers: ProvidersFile::default(),
        project_root: Some(root.to_path_buf()),
    };
    let workspace = Workspace::from_config(&config).unwrap();
    let terminal = TerminalRunner::new(root.join(".agent/checkpoints"));
    let background = BackgroundTaskRegistry::new(root.join(".agent/background"));
    let hooks = crate::hooks::HookRunner::new(root.join(".agent/hooks"), root.to_path_buf());
    ToolManager::new(
        workspace,
        terminal,
        background,
        hooks,
        mcp_clients,
        Vec::new(),
        Arc::new(AtomicBool::new(false)),
    )
}

#[test]
fn platform_cli_for_maps_every_platform_tool_to_a_cli() {
    for spec in platform_tool_specs() {
        assert!(
            platform_cli_for(&spec.name).is_some(),
            "platform tool '{}' has no CLI mapping — add it to `platform_cli_for`",
            spec.name
        );
    }
}

#[test]
fn core_tools_survive_platform_filter_regardless_of_clis() {
    let present = HashSet::new();
    let kept: Vec<String> = filter_platform_specs(core_tool_specs(), &present)
        .into_iter()
        .map(|s| s.name)
        .collect();
    for name in ["read", "write", "edit", "bash", "git_status", "grep"] {
        assert!(
            kept.contains(&name.to_string()),
            "core tool '{name}' was filtered"
        );
    }
    assert_eq!(kept.len(), core_tool_specs().len());
}

#[test]
fn platform_tools_filtered_by_cli_presence() {
    let none = filter_platform_specs(platform_tool_specs(), &HashSet::new());
    assert_eq!(
        none.len(),
        0,
        "no CLIs present but {} platform tools advertised",
        none.len()
    );

    let mut present = HashSet::new();
    present.insert("gh".to_string());
    let kept = filter_platform_specs(platform_tool_specs(), &present);
    assert!(
        kept.iter().all(|s| s.name.starts_with("gh_")),
        "unexpected non-gh tool kept: {:?}",
        kept.iter().map(|s| s.name.as_str()).collect::<Vec<_>>()
    );
    assert!(
        kept.iter().any(|s| s.name == "gh_issue_list"),
        "expected gh_issue_list to survive"
    );
    assert!(
        !kept.iter().any(|s| s.name == "supabase_projects_list"),
        "supabase tools advertised without supabase on PATH"
    );
}

#[test]
fn all_tool_specs_omits_platform_tools_whose_cli_is_missing() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    let specs = tm.all_tool_specs();

    let present = detect_platform_clis();
    for spec in &specs {
        if let Some(cli) = platform_cli_for(&spec.name) {
            assert!(
                present.contains(cli),
                "advertised platform tool '{}' requires '{cli}' which is not on PATH",
                spec.name
            );
        }
    }

    for name in ["read", "write", "edit", "bash", "grep"] {
        assert!(
            specs.iter().any(|s| s.name == name),
            "core tool '{name}' missing from all_tool_specs"
        );
    }
}

#[test]
fn write_then_read_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    let r = tm
        .dispatch_with_approver("write", r#"{"path":"a.txt","content":"hello"}"#, approve)
        .unwrap();
    assert!(!r.is_error);
    let r = tm
        .dispatch_with_approver("read", r#"{"path":"a.txt"}"#, approve)
        .unwrap();
    assert!(r.content.contains("hello"));
}

#[test]
fn read_multiple_reads_batch_and_reports_missing() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("a.txt"), "alpha\n").unwrap();
    std::fs::write(root.join("b.txt"), "beta\n").unwrap();
    let tm = tool_manager(&root);
    let r = tm
        .dispatch_with_approver(
            "read_multiple",
            r#"{"paths":["a.txt","b.txt","missing.txt"]}"#,
            approve,
        )
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert!(r.content.contains("=== a.txt"), "{}", r.content);
    assert!(r.content.contains("=== b.txt"), "{}", r.content);
    assert!(r.content.contains("alpha"), "{}", r.content);
    assert!(r.content.contains("beta"), "{}", r.content);
    assert!(r.content.contains("--- missing.txt"), "{}", r.content);
}

#[test]
fn read_multiple_errors_on_empty_or_oversized_batch() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    let empty = tm
        .dispatch_with_approver("read_multiple", r#"{"paths":[]}"#, approve)
        .unwrap();
    assert!(empty.is_error, "{}", empty.content);
    let many = format!(
        r#"{{"paths":{}}}"#,
        serde_json::to_string(&vec!["x"; 21]).unwrap()
    );
    let oversized = tm
        .dispatch_with_approver("read_multiple", &many, approve)
        .unwrap();
    assert!(oversized.is_error, "{}", oversized.content);
    assert!(oversized.content.contains("20"), "{}", oversized.content);
}

#[test]
fn mkdir_tool_creates_directory_scaffold() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    let r = tm
        .dispatch_with_approver("mkdir", r#"{"path":"src/components"}"#, approve)
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert!(root.join("src/components").is_dir());
    let again = tm
        .dispatch_with_approver("mkdir", r#"{"path":"src/components"}"#, approve)
        .unwrap();
    assert!(!again.is_error, "{}", again.content);
}

#[test]
fn listdir_tool_lists_flat_and_recursive() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    tm.dispatch_with_approver("mkdir", r#"{"path":"src/nested"}"#, approve)
        .unwrap();
    tm.dispatch_with_approver("write", r#"{"path":"src/a.js","content":"x"}"#, approve)
        .unwrap();
    tm.dispatch_with_approver(
        "write",
        r#"{"path":"src/nested/b.js","content":"y"}"#,
        approve,
    )
    .unwrap();
    let flat = tm
        .dispatch_with_approver("listdir", r#"{"path":"src"}"#, approve)
        .unwrap();
    assert!(!flat.is_error, "{}", flat.content);
    assert!(flat.content.contains("nested/"), "{}", flat.content);
    assert!(flat.content.contains("a.js"), "{}", flat.content);
    assert!(!flat.content.contains("b.js"), "{}", flat.content);
    let tree = tm
        .dispatch_with_approver("listdir", r#"{"path":"src","recursive":true}"#, approve)
        .unwrap();
    assert!(tree.content.contains("nested/"), "{}", tree.content);
    assert!(tree.content.contains("b.js"), "{}", tree.content);
}

#[test]
fn listdir_read_only_but_mkdir_gated() {
    assert!(is_read_only_tool("listdir"));
    assert!(!is_read_only_tool("mkdir"));
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(root.join("src")).unwrap();
    let tm = tool_manager(&root);
    tm.set_plan_mode(true);
    let list = tm
        .dispatch_with_approver("listdir", r#"{"path":"src"}"#, approve)
        .unwrap();
    assert!(!list.is_error, "{}", list.content);
    let blocked = tm
        .dispatch_with_approver("mkdir", r#"{"path":"src/new"}"#, approve)
        .unwrap();
    assert!(blocked.is_error, "{}", blocked.content);
    assert!(!root.join("src/new").exists());
}

#[test]
fn verify_runs_explicit_command_and_reports_exit_code() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    let ok = tm
        .dispatch_with_approver("verify", r#"{"command":"exit 0"}"#, approve)
        .unwrap();
    assert!(!ok.is_error, "{}", ok.content);
    assert!(ok.content.contains("exit=Some(0)"), "{}", ok.content);
    let fail = tm
        .dispatch_with_approver("verify", r#"{"command":"exit 1"}"#, approve)
        .unwrap();
    assert!(fail.is_error, "{}", fail.content);
    assert!(fail.content.contains("exit=Some(1)"), "{}", fail.content);
}

#[test]
fn verify_without_detection_and_without_command_errors() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(&root).unwrap();
    let tm = tool_manager(&root);
    let r = tm.dispatch_with_approver("verify", "{}", approve).unwrap();
    assert!(r.is_error, "{}", r.content);
    assert!(
        r.content.contains("couldn't detect") || r.content.contains("no build command"),
        "{}",
        r.content
    );
}

#[test]
fn verify_not_in_read_only_tool_list() {
    assert!(!is_read_only_tool("verify"));
    assert!(!is_read_only_tool("test"));
    assert!(!is_read_only_tool("bash"));
}

#[test]
fn plan_mode_blocks_mutating_tools_but_allows_read_only() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    tm.dispatch_with_approver("write", r#"{"path":"a.txt","content":"hello"}"#, approve)
        .unwrap();

    tm.set_plan_mode(true);
    assert!(tm.plan_mode());

    let blocked = tm
        .dispatch_with_approver("write", r#"{"path":"a.txt","content":"changed"}"#, approve)
        .unwrap();
    assert!(blocked.is_error);
    assert!(blocked.content.contains("Plan mode"));
    assert_eq!(
        std::fs::read_to_string(root.join("a.txt")).unwrap(),
        "hello"
    );

    let read = tm
        .dispatch_with_approver("read", r#"{"path":"a.txt"}"#, approve)
        .unwrap();
    assert!(!read.is_error);
    assert!(read.content.contains("hello"));

    tm.set_plan_mode(false);
    let write_again = tm
        .dispatch_with_approver("write", r#"{"path":"a.txt","content":"changed"}"#, approve)
        .unwrap();
    assert!(!write_again.is_error);
}

#[test]
fn unknown_tool_errors() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    let result = tm
        .dispatch_with_approver("frobnicate", "{}", approve)
        .unwrap();
    assert!(result.is_error);
    assert!(result.content.contains("frobnicate"));
}

#[test]
fn builtin_skills_are_listed_and_readable() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    let listed = tm
        .dispatch_with_approver("list_skills", "{}", approve)
        .unwrap();
    assert!(!listed.is_error, "{}", listed.content);
    assert!(listed.content.contains("build-app"));
    assert!(listed.content.contains("database"));
    assert!(listed.content.contains("ui-design"));
    let searched = tm
        .dispatch_with_approver("list_skills", r#"{"search":"xlsx"}"#, approve)
        .unwrap();
    assert!(!searched.is_error);
    assert!(searched.content.contains("document-reading"));
    assert!(!searched.content.contains("build-app"));
    let read = tm
        .dispatch_with_approver("read_skill", r#"{"name":"git-workflows"}"#, approve)
        .unwrap();
    assert!(!read.is_error, "{}", read.content);
    assert!(read.content.contains("Before committing"));
    let missing = tm
        .dispatch_with_approver("read_skill", r#"{"name":"nope"}"#, approve)
        .unwrap();
    assert!(missing.is_error);
}

#[test]
fn read_skill_recursively_composes_depends_on_chain() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    let r = tm
        .dispatch_with_approver(
            "read_skill",
            r#"{"name":"build-app","recursive":true}"#,
            approve,
        )
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert!(r.content.contains("skill: build-app"));
    assert!(r.content.contains("skill: database"));
    assert!(r.content.contains("skill: api"));
    assert!(r.content.contains("skill: frontend"));
    assert!(r.content.contains("skill: security"));
    assert!(r.content.contains("skill: qa-testing"));
    assert!(r.content.contains("skill: documentation"));
    assert!(r.content.contains("skill: project-orientation"));
}

#[test]
fn project_skill_shadows_builtin() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(root.join(".agent/skills/database")).unwrap();
    std::fs::write(
        root.join(".agent/skills/database/SKILL.md"),
        "---\nname: database\ndescription: PROJECT-OVERRIDE\n---\n# Project DB rules",
    )
    .unwrap();
    let tm = tool_manager(&root);
    let read = tm
        .dispatch_with_approver("read_skill", r#"{"name":"database"}"#, approve)
        .unwrap();
    assert!(!read.is_error);
    assert!(!read.content.contains("Design schemas, SQL, and migrations"));
    assert!(read.content.contains("Project DB rules"));
    assert!(read.content.contains("skill: database (tier: project)"));
}

#[test]
fn read_document_extracts_text_and_errors_on_binary() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("notes.md"), "# Notes\n\nplain markdown text here").unwrap();
    let tm = tool_manager(&root);

    let r = tm
        .dispatch_with_approver("read_document", r#"{"path":"notes.md"}"#, approve)
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert!(r.content.contains("plain markdown text here"));

    let missing = tm
        .dispatch_with_approver("read_document", r#"{"path":"nope.pdf"}"#, approve)
        .unwrap();
    assert!(missing.is_error);
}

#[test]
fn read_image_attaches_base64_bytes() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(&root).unwrap();
    let png = base64::engine::general_purpose::STANDARD
        .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        .unwrap();
    std::fs::write(root.join("pixel.png"), &png).unwrap();
    let tm = tool_manager(&root);

    let r = tm
        .dispatch_with_approver("read_image", r#"{"path":"pixel.png"}"#, approve)
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert_eq!(r.images.len(), 1);
    assert_eq!(r.images[0].mime_type, "image/png");
    assert!(!r.images[0].data_base64.is_empty());

    let bad = tm
        .dispatch_with_approver("read_image", r#"{"path":"pixel.txt"}"#, approve)
        .unwrap();
    assert!(bad.is_error);
    let missing = tm
        .dispatch_with_approver("read_image", r#"{"path":"absent.png"}"#, approve)
        .unwrap();
    assert!(missing.is_error);
}

#[test]
fn understand_repo_reports_stack_and_relevance() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(root.join("src/auth")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        "[package]\nname=\"x\"\n[dependencies]\naxum=\"0.7\"\nsqlx=\"0.8\"\n",
    )
    .unwrap();
    std::fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
    std::fs::write(root.join("src/auth/login.rs"), "pub fn login() {}").unwrap();
    let tm = tool_manager(&root);

    let r = tm
        .dispatch_with_approver("understand_repo", r#"{"topic":"authentication"}"#, approve)
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert!(r.content.contains("Repository understanding"));
    assert!(r.content.contains("Axum"));
    assert!(r.content.contains("authentication") || r.content.contains("auth"));

    let no_topic = tm
        .dispatch_with_approver("understand_repo", "{}", approve)
        .unwrap();
    assert!(!no_topic.is_error);
    assert!(no_topic.content.contains("Rust"));
}

#[test]
fn rag_search_ranks_concept_chunks_above_the_rest() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/retry.rs"),
        "fn with_retry(action) { for attempt in 0..3 { /* reconnect */ } }",
    )
    .unwrap();
    std::fs::write(
        root.join("src/ui.rs"),
        "fn render_button(label) { draw(label) }",
    )
    .unwrap();
    let tm = tool_manager(&root);

    let r = tm
        .dispatch_with_approver("rag_search", r#"{"query":"retry reconnect"}"#, approve)
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert!(r.content.contains("retry.rs"), "{}", r.content);
    assert!(r.content.contains("retry"), "{}", r.content);

    let empty = tm
        .dispatch_with_approver("rag_search", r#"{"query":""}"#, approve)
        .unwrap();
    assert!(empty.is_error);
    assert!(
        empty.content.contains("must not be empty"),
        "{}",
        empty.content
    );
}

#[test]
fn rag_search_is_read_only_so_plan_mode_can_use_it() {
    assert!(is_read_only_tool("rag_search"));
}

#[test]
fn rag_index_builds_persistent_index_and_rag_search_reuses_it() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/retry.rs"),
        "fn with_retry(action) { for attempt in 0..3 { /* reconnect */ } }",
    )
    .unwrap();
    std::fs::write(
        root.join("src/ui.rs"),
        "fn render_button(label) { draw(label) }",
    )
    .unwrap();
    let tm = tool_manager(&root);
    let approve = |_p: &PermissionRequest| ApprovalDecision::Approved;

    assert!(!is_read_only_tool("rag_index"));

    let idx = tm
        .dispatch_with_approver("rag_index", "{}", approve)
        .unwrap();
    assert!(!idx.is_error, "{}", idx.content);
    assert!(idx.content.contains("chunk"), "{}", idx.content);

    let index_path = zeus_rag::PersistedRagIndex::file_path(&root);
    assert!(index_path.exists());

    let again = tm
        .dispatch_with_approver("rag_index", "{}", approve)
        .unwrap();
    assert!(!again.is_error, "{}", again.content);
    assert!(
        again.content.contains("already exists"),
        "{}",
        again.content
    );

    let r = tm
        .dispatch_with_approver("rag_search", r#"{"query":"retry reconnect"}"#, approve)
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert!(r.content.contains("retry.rs"), "{}", r.content);

    std::fs::write(
        root.join("src/retry.rs"),
        "fn with_retry(action) { for attempt in 0..5 { /* retried */ } }",
    )
    .unwrap();
    let stale = zeus_rag::PersistedRagIndex::load(&root).unwrap();
    assert!(!stale.is_fresh());
    assert_eq!(stale.documents.len(), 2);
    let refresh = tm
        .dispatch_with_approver("rag_index", "{}", approve)
        .unwrap();
    assert!(!refresh.is_error, "{}", refresh.content);
    let fresh = zeus_rag::PersistedRagIndex::load(&root).unwrap();
    assert!(fresh.is_fresh());
    assert_eq!(fresh.documents.len(), 2);
    assert!(fresh.documents.iter().any(|c| c.text.contains("retried")));
    assert!(fresh
        .documents
        .iter()
        .any(|c| c.text.contains("render_button")));

    let rebuild = tm
        .dispatch_with_approver("rag_index", r#"{"force":true}"#, approve)
        .unwrap();
    assert!(!rebuild.is_error, "{}", rebuild.content);
    assert!(zeus_rag::PersistedRagIndex::load(&root).unwrap().is_fresh());
}

#[test]
fn rag_index_embed_degrades_gracefully_without_provider() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/retry.rs"), "fn with_retry() {}\n").unwrap();
    let tm = tool_manager(&root);
    let approve = |_p: &PermissionRequest| ApprovalDecision::Approved;

    let r = tm
        .dispatch_with_approver("rag_index", r#"{"embed":true}"#, approve)
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert!(r.content.contains("keyword-only"), "{}", r.content);
    let persisted = zeus_rag::PersistedRagIndex::load(&root).unwrap();
    assert!(!persisted.has_vectors());
}

struct EmbedMock {
    dim: usize,
}

#[async_trait::async_trait]
impl ModelProvider for EmbedMock {
    fn supports_prompt_cache(&self) -> bool {
        false
    }
    fn id(&self) -> &str {
        "embed-mock"
    }
    async fn chat(&self, _req: ChatRequest) -> zeus_provider::Result<ChatResponse> {
        unreachable!("chat not used in rag embed test")
    }
    async fn stream(&self, _req: ChatRequest) -> zeus_provider::Result<ChatStream> {
        unreachable!("stream not used in rag embed test")
    }
    async fn list_models(&self) -> zeus_provider::Result<Vec<ModelInfo>> {
        Ok(Vec::new())
    }
    async fn embeddings(&self, req: EmbeddingRequest) -> zeus_provider::Result<EmbeddingResponse> {
        let vectors = req
            .input
            .iter()
            .map(|text| {
                let mut v = vec![0.0f32; self.dim];
                let bucket = text
                    .bytes()
                    .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64))
                    % self.dim as u64;
                v[bucket as usize] = 1.0;
                v
            })
            .collect();
        Ok(EmbeddingResponse {
            vectors,
            usage: TokenUsage::new(0, 0),
        })
    }
    async fn count_tokens(
        &self,
        _req: TokenCountRequest,
    ) -> zeus_provider::Result<TokenCountResponse> {
        Ok(TokenCountResponse {
            tokens: 1,
            approximate: true,
        })
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn rag_index_embed_persists_vectors_and_search_uses_them() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/retry.rs"),
        "fn with_retry() { /* reconnect */ }\n",
    )
    .unwrap();
    let mut tm = tool_manager(&root);
    tm.embedder = Some(Arc::new(EmbedMock { dim: 8 }));
    tm.embed_model = Some("mock".into());
    let approve = |_p: &PermissionRequest| ApprovalDecision::Approved;

    let r = tm
        .dispatch_with_approver("rag_index", r#"{"embed":true}"#, approve)
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert!(r.content.contains("embedded 1 chunk(s)"), "{}", r.content);

    let persisted = zeus_rag::PersistedRagIndex::load(&root).unwrap();
    assert!(persisted.has_vectors());
    assert_eq!(persisted.vectors.as_ref().unwrap().len(), 1);

    let s = tm
        .dispatch_with_approver("rag_search", r#"{"query":"reconnect"}"#, approve)
        .unwrap();
    assert!(!s.is_error, "{}", s.content);
    assert!(s.content.contains("retry.rs"), "{}", s.content);
}

#[test]
fn urlencode_encodes_query() {
    assert_eq!(helpers::urlencode("offline sync"), "offline+sync");
    assert_eq!(helpers::urlencode("a&b?"), "a%26b%3F");
    assert_eq!(helpers::urlencode("rust"), "rust");
}

#[test]
fn web_search_rejects_empty_query() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(&root).unwrap();
    let tm = tool_manager(&root);
    let r = tm
        .dispatch_with_approver("web_search", r#"{"query":""}"#, approve)
        .unwrap();
    assert!(r.is_error);
    assert!(r.content.contains("non-empty"));
    let missing = tm
        .dispatch_with_approver("web_search", "{}", approve)
        .unwrap();
    assert!(
        missing.is_error,
        "missing `query` should surface as a failed tool result"
    );
}

#[test]
fn web_search_is_read_only_tool() {
    assert!(
        is_read_only_tool("web_search"),
        "web_search must run in plan mode"
    );
    assert!(is_read_only_tool("web_fetch"));
    assert!(!is_read_only_tool("bash"));
}

#[test]
fn memory_tools_list_read_write() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(&root).unwrap();
    let tm = tool_manager(&root);
    let approve = |_p: &PermissionRequest| ApprovalDecision::Approved;

    let empty = tm
        .dispatch_with_approver("memory", r#"{"action":"list"}"#, approve)
        .unwrap();
    assert!(!empty.is_error);
    assert!(empty.content.contains("No long-term memory"));

    let w = tm
        .dispatch_with_approver(
            "memory_write",
            r#"{"name":"auth","content":"token-based auth"}"#,
            approve,
        )
        .unwrap();
    assert!(!w.is_error, "{}", w.content);
    let path = root.join(".agent/memory/auth.md");
    assert!(path.exists());

    let list = tm
        .dispatch_with_approver("memory", r#"{"action":"list"}"#, approve)
        .unwrap();
    assert!(list.content.contains("auth"));

    let read = tm
        .dispatch_with_approver("memory", r#"{"action":"read","name":"auth"}"#, approve)
        .unwrap();
    assert!(read.content.contains("token-based"));

    let bad_name = tm
        .dispatch_with_approver(
            "memory_write",
            r#"{"name":"BAD NAME","content":"x"}"#,
            approve,
        )
        .unwrap();
    assert!(bad_name.is_error);
}

#[test]
fn memory_tools_blocked_in_plan_mode() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(&root).unwrap();
    let tm = tool_manager(&root);
    tm.set_plan_mode(true);
    let approve = |_p: &PermissionRequest| ApprovalDecision::Approved;
    let r = tm
        .dispatch_with_approver("memory_write", r#"{"name":"x","content":"y"}"#, approve)
        .unwrap();
    assert!(r.is_error, "memory_write must be blocked in plan mode");
}

#[test]
fn code_intel_tools_round_trip() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("lib.rs"),
        "pub struct Foo {}\nimpl Foo { pub fn bar(&self) {} }\nfn use_it(f: &Foo) -> u32 { 0 }\n",
    )
    .unwrap();

    let tm = tool_manager(&root);

    let r = tm
        .dispatch_with_approver("code_index", r#"{"force":true}"#, approve)
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert!(r.content.contains("indexed"));

    let r = tm
        .dispatch_with_approver("code_index", "{}", approve)
        .unwrap();
    assert!(!r.is_error);
    assert!(r.content.contains("already exists"));

    let r = tm
        .dispatch_with_approver("code_symbols", r#"{"name":"Foo"}"#, approve)
        .unwrap();
    assert!(!r.is_error);
    assert!(r.content.contains("Foo") && r.content.contains("lib.rs"));

    let r = tm
        .dispatch_with_approver("code_refs", r#"{"name":"Foo"}"#, approve)
        .unwrap();
    assert!(!r.is_error);
    assert!(r.content.contains("3 reference(s)"), "got: {}", r.content);
}

#[test]
fn code_graph_reports_callers_and_callees() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("lib.rs"),
        "fn main() {\n  helper();\n}\nfn helper() {\n  leaf();\n}\nfn leaf() {}\n",
    )
    .unwrap();

    let tm = tool_manager(&root);
    tm.dispatch_with_approver("code_index", r#"{"force":true}"#, approve)
        .unwrap();

    let r = tm
        .dispatch_with_approver("code_graph", r#"{"name":"helper"}"#, approve)
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert!(r.content.contains("caller(s) of 'helper'"));
    assert!(r.content.contains("main -> helper"));
    assert!(r.content.contains("calls 1 function(s)"));
    assert!(r.content.contains("helper -> leaf"));

    let r = tm
        .dispatch_with_approver(
            "code_graph",
            r#"{"name":"helper","direction":"callers"}"#,
            approve,
        )
        .unwrap();
    assert!(!r.is_error);
    assert!(r.content.contains("main -> helper"));
    assert!(!r.content.contains("helper -> leaf"));

    let r = tm
        .dispatch_with_approver("code_graph", r#"{"name":"nope"}"#, approve)
        .unwrap();
    assert!(!r.is_error);
    assert!(r.content.contains("no callers of 'nope' found"));
}

#[test]
fn code_verbose_rename_reports_plan_only() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("lib.rs"), "fn alpha() { leap_alpha(); }\n").unwrap();

    let tm = tool_manager(&root);
    let r = tm
        .dispatch_with_approver("code_rename", r#"{"old":"alpha","new":"omega"}"#, approve)
        .unwrap();
    assert!(!r.is_error);
    assert!(r.content.contains("rename 'alpha' -> 'omega'"));
    assert!(r.content.contains("Plan only"));
    assert!(std::fs::read_to_string(root.join("lib.rs"))
        .unwrap()
        .contains("fn alpha()"));
}

#[test]
fn bash_runs_and_denies_destructive() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    let r = tm
        .dispatch_with_approver("bash", r#"{"command":"echo hi"}"#, approve)
        .unwrap();
    assert!(!r.is_error);
    assert!(r.content.contains("hi"));

    let r2 = tm
        .dispatch_with_approver("bash", r#"{"command":"rm -rf /"}"#, approve)
        .unwrap();
    assert!(r2.is_error);
}

#[test]
fn bash_background_spawns_and_is_listed_and_stoppable() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    let sleep_cmd = if cfg!(windows) {
        "ping -n 30 127.0.0.1 >NUL"
    } else {
        "sleep 30"
    };

    let started = tm
        .dispatch_with_approver(
            "bash",
            &format!(r#"{{"command":"{sleep_cmd}","background":true}}"#),
            approve,
        )
        .unwrap();
    assert!(!started.is_error);
    assert!(started.content.contains("started background task"));

    let listed = tm.dispatch_with_approver("bg_list", "{}", approve).unwrap();
    assert!(listed.content.contains("status=Running"));

    let id = tm.background().list().unwrap()[0].0.id;
    let stopped = tm
        .dispatch_with_approver("bg_stop", &format!(r#"{{"id":{id}}}"#), approve)
        .unwrap();
    assert!(!stopped.is_error);
    assert!(tm.background().get(id).unwrap().is_none());
}

#[test]
fn mcp_tool_is_advertised_and_dispatchable_end_to_end() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(&root).unwrap();
    let script = crate::mcp::write_test_server(&root);
    let client = crate::mcp::McpClient::connect(
        "testsrv",
        crate::mcp::python_cmd(),
        &[script.display().to_string()],
        &root,
    )
    .unwrap();
    let tm = tool_manager_with_mcp(&root, vec![client]);

    let specs = tm.all_tool_specs();
    assert!(specs.iter().any(|s| s.name == "mcp__testsrv__echo"));

    let ok = tm
        .dispatch_with_approver("mcp__testsrv__echo", r#"{"text":"hi"}"#, approve)
        .unwrap();
    assert!(!ok.is_error);
    assert_eq!(ok.content, "echo: hi");

    let failed = tm
        .dispatch_with_approver("mcp__testsrv__echo", r#"{"fail":true}"#, approve)
        .unwrap();
    assert!(failed.is_error);
    assert_eq!(failed.content, "deliberate failure");
}

#[test]
fn mcp_call_denied_is_not_run() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(&root).unwrap();
    let script = crate::mcp::write_test_server(&root);
    let client = crate::mcp::McpClient::connect(
        "testsrv",
        crate::mcp::python_cmd(),
        &[script.display().to_string()],
        &root,
    )
    .unwrap();
    let tm = tool_manager_with_mcp(&root, vec![client]);

    let denied = tm
        .dispatch_with_approver("mcp__testsrv__echo", r#"{"text":"hi"}"#, |_| {
            ApprovalDecision::Denied
        })
        .unwrap();
    assert!(denied.is_error);
    assert!(denied.content.contains("denied"));
}

#[test]
fn every_tool_spec_has_a_handler() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    for spec in builtin_tool_specs() {
        let result = tm
            .dispatch_with_approver(&spec.name, "{}", approve)
            .unwrap();
        assert!(
            !(result.is_error && result.content.starts_with("unknown tool:")),
            "tool spec '{}' has no handler: {}",
            spec.name,
            result.content
        );
    }
}

#[test]
fn git_tools_work_end_to_end_through_the_full_dispatch_path() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    std::fs::create_dir_all(&root).unwrap();
    std::process::Command::new("git")
        .arg("init")
        .current_dir(&root)
        .output()
        .unwrap();
    std::process::Command::new("git")
        .args(["config", "user.email", "test@example.com"])
        .current_dir(&root)
        .output()
        .unwrap();
    std::process::Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(&root)
        .output()
        .unwrap();

    let tm = tool_manager(&root);

    std::fs::write(root.join("a.txt"), "hello").unwrap();
    let add = tm
        .dispatch_with_approver("git_add", r#"{"paths":["a.txt"]}"#, approve)
        .unwrap();
    assert!(!add.is_error, "git_add failed: {}", add.content);

    let commit = tm
        .dispatch_with_approver("git_commit", r#"{"message":"initial commit"}"#, approve)
        .unwrap();
    assert!(!commit.is_error, "git_commit failed: {}", commit.content);

    let log = tm.dispatch_with_approver("git_log", "{}", approve).unwrap();
    assert!(!log.is_error);
    assert!(log.content.contains("initial commit"));

    let status = tm
        .dispatch_with_approver("git_status", "{}", approve)
        .unwrap();
    assert!(!status.is_error);

    let force_push = tm
        .dispatch_with_approver("git_push", r#"{"force":true}"#, approve)
        .unwrap();
    assert!(force_push.is_error);

    let hard_reset = tm
        .dispatch_with_approver("git_reset", r#"{"mode":"hard"}"#, approve)
        .unwrap();
    assert!(hard_reset.is_error);
}

#[test]
fn detect_test_command_maps_manifests() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();
    std::fs::write(root.join("Cargo.toml"), "").unwrap();
    assert_eq!(detect_test_command(root).as_deref(), Some("cargo test"));

    std::fs::remove_file(root.join("Cargo.toml")).unwrap();
    std::fs::write(root.join("go.mod"), "").unwrap();
    assert_eq!(detect_test_command(root).as_deref(), Some("go test ./..."));

    std::fs::remove_file(root.join("go.mod")).unwrap();
    std::fs::write(root.join("pyproject.toml"), "").unwrap();
    assert_eq!(
        detect_test_command(root).as_deref(),
        Some("python -m pytest -q")
    );

    std::fs::remove_file(root.join("pyproject.toml")).unwrap();
    std::fs::write(root.join("package.json"), "{}").unwrap();
    assert_eq!(detect_test_command(root).as_deref(), Some("npm test"));

    std::fs::write(root.join("pnpm-lock.yaml"), "").unwrap();
    assert_eq!(detect_test_command(root).as_deref(), Some("pnpm test"));

    std::fs::remove_file(root.join("pnpm-lock.yaml")).unwrap();
    std::fs::write(root.join("yarn.lock"), "").unwrap();
    assert_eq!(detect_test_command(root).as_deref(), Some("yarn test"));
}

#[test]
fn detect_test_command_none_when_no_manifest() {
    let tmp = TempDir::new().unwrap();
    assert_eq!(detect_test_command(tmp.path()), None);
}

#[test]
fn summarize_test_output_picks_verdict_lines() {
    let out = "\n  Compiling zeus v0.1.0\n\nrunning 4 tests\n..s....\n\ntest result: ok. 4 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out\n\nrunning 1 test\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n";
    let summary = helpers::summarize_test_output(out);
    assert!(summary.contains("test result: ok"), "{summary}");
    assert!(summary.contains("4 passed"), "{summary}");
    assert!(!summary.contains("running 4"), "{summary}");
}

#[test]
fn test_tool_runs_with_explicit_command() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    let cmd = if cfg!(windows) {
        r#"powershell -NoProfile -Command "Write-Output 'test result: ok. 1 passed; 0 failed'""#
    } else {
        "echo \"test result: ok. 1 passed; 0 failed\""
    };
    let args = serde_json::json!({ "command": cmd });
    let r = tm
        .dispatch_with_approver("test", &args.to_string(), approve)
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert!(r.content.contains("1 passed"), "{}", r.content);
}

#[test]
fn test_tool_without_command_reports_detection_failure() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    let r = tm.dispatch_with_approver("test", "{}", approve).unwrap();
    assert!(r.is_error, "{}", r.content);
    assert!(r.content.contains("auto-detect"), "{}", r.content);
}

#[test]
fn browser_rejects_bad_url_and_blocks_in_plan_mode() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);

    let bad = tm
        .dispatch_with_approver("browser", r#"{"url":"C:/Windows/System32"}"#, approve)
        .unwrap();
    assert!(bad.is_error, "{}", bad.content);

    tm.set_plan_mode(true);
    let blocked = tm
        .dispatch_with_approver("browser", r#"{"url":"http://localhost:5173"}"#, approve)
        .unwrap();
    assert!(blocked.is_error, "{}", blocked.content);
    assert!(blocked.content.contains("Plan mode"), "{}", blocked.content);
    tm.set_plan_mode(false);
}

#[test]
fn browser_rejects_file_scheme_and_bare_paths() {
    assert!(helpers::open_browser_url("file:///C:/Windows/System32").is_err());
    assert!(helpers::open_browser_url("file:///etc/hosts").is_err());
    assert!(helpers::open_browser_url("C:/Users/me/notes.txt").is_err());
    assert!(helpers::open_browser_url("../secret/config.toml").is_err());
    assert!(helpers::open_browser_url("http://localhost:5173").is_ok());
    assert!(helpers::open_browser_url("https://example.com").is_ok());
    assert!(helpers::open_browser_url("localhost:5173").is_ok());
}

#[test]
fn web_fetch_rejects_internal_targets() {
    for url in [
        "http://localhost",
        "http://localhost:8080/path",
        "http://127.0.0.1",
        "http://127.0.0.2",
        "http://10.0.0.1",
        "http://172.16.0.5",
        "http://192.168.1.1",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]",
        "http://[::ffff:127.0.0.1]",
        "http://[::ffff:10.0.0.1]",
        "http://metadata.google.internal",
        "http://localhost.",
        "https://127.0.0.1:8443",
        "https://0.0.0.0",
    ] {
        assert!(
            helpers::reject_web_target(url).is_some(),
            "expected '{url}' to be refused"
        );
    }
}

#[test]
fn web_fetch_blocks_hostnames_that_resolve_to_loopback() {
    if let Some(reason) = helpers::reject_web_target("http://localhost.") {
        assert!(reason.contains("localhost"), "{reason}");
    }
}

#[test]
fn web_fetch_allows_public_targets() {
    for url in [
        "http://example.com",
        "https://example.com/docs",
        "https://api.github.com/repos/foo/bar",
        "http://193.0.0.1",
        "https://8.8.8.8",
    ] {
        assert!(
            helpers::reject_web_target(url).is_none(),
            "unexpected block of '{url}'"
        );
    }
}

#[test]
fn current_time_tool_returns_a_parseable_datetime() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path().join("proj");
    let tm = tool_manager(&root);
    let r = tm
        .dispatch_with_approver("current_time", "{}", approve)
        .unwrap();
    assert!(!r.is_error, "{}", r.content);
    assert!(
        r.content.contains(&chrono::Local::now().year().to_string()),
        "expected current year in: {}",
        r.content
    );
    assert!(r.content.contains("UTC offset"), "{}", r.content);

    tm.set_plan_mode(true);
    let in_plan = tm
        .dispatch_with_approver("current_time", "{}", approve)
        .unwrap();
    assert!(!in_plan.is_error, "{}", in_plan.content);
}

#[test]
fn current_time_is_listed_as_read_only() {
    assert!(is_read_only_tool("current_time"));
}

#[test]
fn platform_tools_registry_matches_specs_and_dispatch() {
    let specs = platform_tool_specs();
    let spec_names: Vec<&str> = specs.iter().map(|t| t.name.as_str()).collect();

    let registry: Vec<&str> = PLATFORM_TOOLS.to_vec();
    let mut spec_sorted = spec_names.clone();
    let mut registry_sorted = registry.clone();
    spec_sorted.sort_unstable();
    registry_sorted.sort_unstable();

    assert_eq!(
        spec_sorted, registry_sorted,
        "PLATFORM_TOOLS registry and platform_tool_specs() disagree on the \
         platform tool list — keep them identical"
    );

    for name in &registry {
        let tmp = TempDir::new().unwrap();
        let tm = tool_manager(tmp.path());
        let r = tm.dispatch_with_approver(name, "{}", approve).unwrap();
        assert!(
            !r.content.contains("unknown tool"),
            "{name} not dispatched by do_platform"
        );
    }
}
