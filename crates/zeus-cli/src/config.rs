//! Shared config loading, workspace construction, approval prompting, and
//! low-level TOML helpers for the CLI layer.

use anyhow::{Context, Result};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::cli::Cli;
use crate::highlight;
use zeus_agent::load_custom_personas;
use zeus_config::Config;
use zeus_fs::{ApprovalDecision, PermissionRequest, Workspace};

/// Load merged config (project override if `--project-root` given), and
/// pre-load custom specialist personas from ~/.zeus/personas/*.toml once per
/// process (cached globally).
pub fn load_config(cli: &Cli) -> Result<Config> {
    let config = if let Some(root) = &cli.project_root {
        Config::load(Some(root.as_path())).context("failed to load config")?
    } else {
        Config::load_from_cwd().context("failed to load config")?
    };
    load_custom_personas(&config.global.personas);
    Ok(config)
}

pub fn workspace(config: &Config) -> Result<Workspace> {
    Workspace::from_config(config).map_err(|e| anyhow::anyhow!(e))
}

pub fn approver(yes: bool) -> impl FnMut(&PermissionRequest) -> ApprovalDecision {
    move |req: &PermissionRequest| {
        if yes {
            eprintln!("[auto-approve] {}", req.description);
            if let Some(preview) = &req.preview {
                eprintln!(
                    "{}",
                    if highlight::looks_like_diff(preview) {
                        highlight::ansi_diff(preview)
                    } else {
                        preview.clone()
                    }
                );
            }
            return ApprovalDecision::Approved;
        }
        if let Some(preview) = &req.preview {
            eprintln!(
                "{}",
                if highlight::looks_like_diff(preview) {
                    highlight::ansi_diff(preview)
                } else {
                    preview.clone()
                }
            );
        }
        eprint!("Allow {}? [y/N/s(session)] ", req.description);
        let _ = io::stderr().flush();
        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_err() {
            return ApprovalDecision::Denied;
        }
        match line.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => ApprovalDecision::Approved,
            "s" | "session" => ApprovalDecision::ApprovedForSession,
            _ => ApprovalDecision::Denied,
        }
    }
}

/// Path to the raw on-disk settings.toml that `zeus config get/set` touches:
/// the project's (checked-in, shared) file if a project is active and `global`
/// wasn't forced, else the global one. These commands read/write *one*
/// on-disk layer directly (as raw TOML), not the fully merged view `config
/// show` prints — simpler, and correct for "I want to change this project's
/// checked-in setting" without needing typed structs for every field.
pub fn settings_file_path(config: &Config, global: bool) -> PathBuf {
    if !global {
        if let Some(project) = &config.project {
            return project.settings_toml.clone();
        }
    }
    config.global.settings_toml.clone()
}

pub fn load_toml_or_empty(path: &Path) -> Result<toml::Value> {
    if !path.exists() {
        return Ok(toml::Value::Table(toml::value::Table::new()));
    }
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

pub fn get_toml_path<'a>(root: &'a toml::Value, parts: &[&str]) -> Option<&'a toml::Value> {
    let mut current = root;
    for part in parts {
        current = current.get(part)?;
    }
    Some(current)
}

pub fn set_toml_path(root: &mut toml::Value, parts: &[&str], value: toml::Value) {
    if !root.is_table() {
        *root = toml::Value::Table(toml::value::Table::new());
    }
    let table = root.as_table_mut().expect("just ensured it's a table");
    if parts.len() == 1 {
        table.insert(parts[0].to_string(), value);
        return;
    }
    let entry = table
        .entry(parts[0].to_string())
        .or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
    set_toml_path(entry, &parts[1..], value);
}

/// Infer a TOML scalar type from a plain CLI string: "true"/"false" -> bool,
/// parses as an integer or float -> number, else a plain string.
pub fn parse_toml_scalar(s: &str) -> toml::Value {
    if let Ok(b) = s.parse::<bool>() {
        return toml::Value::Boolean(b);
    }
    if let Ok(i) = s.parse::<i64>() {
        return toml::Value::Integer(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return toml::Value::Float(f);
    }
    toml::Value::String(s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_toml_scalar_infers_types() {
        assert_eq!(parse_toml_scalar("true"), toml::Value::Boolean(true));
        assert_eq!(parse_toml_scalar("42"), toml::Value::Integer(42));
        assert_eq!(parse_toml_scalar("2.5"), toml::Value::Float(2.5));
        assert_eq!(
            parse_toml_scalar("hello"),
            toml::Value::String("hello".into())
        );
    }

    #[test]
    fn set_and_get_toml_path_roundtrip() {
        let mut root = toml::Value::Table(toml::value::Table::new());
        set_toml_path(&mut root, &["a", "b", "c"], toml::Value::Integer(7));
        let got = get_toml_path(&root, &["a", "b", "c"]).unwrap();
        assert_eq!(got, &toml::Value::Integer(7));
        // Missing path returns None rather than panicking.
        assert!(get_toml_path(&root, &["a", "nope"]).is_none());
    }

    #[test]
    fn load_toml_or_empty_returns_empty_for_missing_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let value = load_toml_or_empty(&tmp.path().join("absent.toml")).unwrap();
        assert!(value.is_table());
        assert!(value.as_table().unwrap().is_empty());
    }
}
