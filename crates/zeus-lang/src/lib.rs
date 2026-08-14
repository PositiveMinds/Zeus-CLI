//! Language support for zeus: detect the language of a project or file, look
//! up that language's standard dev commands (build / test / lint / format),
//! and scaffold a minimal project skeleton.
//!
//! This is the shared foundation used by:
//! - the CLI (`zeus project` — detect, commands, scaffold, format),
//! - the agent (`format_file` tool),
//! - the symbol indexer's per-language profile table.
//!
//! Kept dependency-free so any zeus crate can use it.

pub mod detect;
pub mod framework;
pub mod scaffold;
pub mod spec;

pub use detect::{detect_project, detect_source, Language};
pub use framework::{framework_spec, scaffold_framework, Framework};
pub use scaffold::{available_scaffold_languages, scaffold_project};
pub use spec::{dev_commands, spec, FormatStyle, LangSpec, FILE_PLACEHOLDER};
