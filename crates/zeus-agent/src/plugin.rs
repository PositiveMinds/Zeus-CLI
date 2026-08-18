//! Native plugin loader: dynamically loads compiled `.dll`/`.so`/`.dylib`
//! files from `~/.zeus/plugins/` and exposes their tools alongside built-ins
//! and MCP servers.
//!
//! This is a fundamentally larger trust boundary than MCP (a separate OS
//! process) or hooks (a subprocess): a loaded plugin runs **in this
//! process's address space**. A buggy or malicious plugin can corrupt
//! memory, crash the process, or do anything this process's OS permissions
//! allow — there is no sandboxing here, unlike every other extension
//! mechanism in this codebase. Load only plugins you trust.
//!
//! ABI (version 1, see `zeus-example-plugin` for a working reference
//! implementation): every call crosses the FFI boundary as JSON inside
//! NUL-terminated C strings, never a native Rust type. This is deliberate —
//! Rust has no stable ABI across compiler versions, so passing Rust structs,
//! `Vec`, `String`, or trait objects across a `dylib` boundary is unsound in
//! general. Restricting the boundary to `extern "C"` functions moving only
//! `u32` and `*const/*mut c_char` sidesteps that entirely: the C calling
//! convention *is* stable, and JSON text has no ABI to break.
//!
//! Required exports:
//! - `zeus_plugin_abi_version() -> u32`
//! - `zeus_plugin_tool_specs() -> *mut c_char` (JSON array of tool specs)
//! - `zeus_plugin_call_tool(name: *const c_char, args_json: *const c_char) -> *mut c_char`
//! - `zeus_plugin_free_string(s: *mut c_char)` (must be used to free
//!   anything the plugin returned — never the host's own allocator; plugin
//!   and host may use different allocators, especially across a Windows DLL
//!   boundary, making cross-allocator free undefined behavior)

use serde::Deserialize;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use tracing::warn;
use zeus_provider::ToolSpec;

/// Must match `zeus-example-plugin`'s `ABI_VERSION`. Bump only alongside a
/// coordinated change to the exported-function contract above; existing
/// compiled plugins built against an older version are rejected at load
/// time (see `PluginManager::load_all`), not silently miscalled.
const EXPECTED_ABI_VERSION: u32 = 1;

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type ToolSpecsFn = unsafe extern "C" fn() -> *mut c_char;
type CallToolFn =
    unsafe extern "C" fn(name: *const c_char, args_json: *const c_char) -> *mut c_char;
type FreeStringFn = unsafe extern "C" fn(s: *mut c_char);

#[derive(Debug, Clone, Deserialize)]
struct RawToolSpec {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default = "default_schema")]
    parameters: serde_json::Value,
}

fn default_schema() -> serde_json::Value {
    serde_json::json!({ "type": "object" })
}

#[derive(Debug, Clone, Deserialize)]
struct RawCallResult {
    #[serde(default)]
    content: String,
    #[serde(default)]
    is_error: bool,
}

pub struct PluginCallResult {
    pub content: String,
    pub is_error: bool,
}

/// A loaded native plugin. The `Library` must outlive every use of its
/// resolved function pointers — dropping it unloads the code and any
/// pointer into it becomes dangling — so it's kept alive for this struct's
/// whole lifetime rather than dropped after the initial tool-spec fetch.
pub struct LoadedPlugin {
    name: String,
    library: libloading::Library,
    tools: Vec<ToolSpec>,
}

impl LoadedPlugin {
    fn load(name: &str, path: &Path) -> Result<Self, String> {
        // Safety: loading a native library and calling into it is
        // inherently unsafe — we're trusting that whatever is at `path`
        // implements the documented ABI honestly. There is no way to
        // verify that from the host side; this is the plugin trust
        // boundary described in the module doc comment.
        let library = unsafe {
            libloading::Library::new(path).map_err(|e| format!("failed to load library: {e}"))?
        };

        let version = unsafe {
            let sym: libloading::Symbol<AbiVersionFn> = library
                .get(b"zeus_plugin_abi_version")
                .map_err(|e| format!("missing zeus_plugin_abi_version: {e}"))?;
            sym()
        };
        if version != EXPECTED_ABI_VERSION {
            return Err(format!(
                "ABI version mismatch: plugin is v{version}, host expects v{EXPECTED_ABI_VERSION}"
            ));
        }

        // Resolve the remaining symbols now so a plugin missing one of them
        // is rejected at load time rather than the first time it's called.
        unsafe {
            let _: libloading::Symbol<ToolSpecsFn> = library
                .get(b"zeus_plugin_tool_specs")
                .map_err(|e| format!("missing zeus_plugin_tool_specs: {e}"))?;
            let _: libloading::Symbol<CallToolFn> = library
                .get(b"zeus_plugin_call_tool")
                .map_err(|e| format!("missing zeus_plugin_call_tool: {e}"))?;
            let _: libloading::Symbol<FreeStringFn> = library
                .get(b"zeus_plugin_free_string")
                .map_err(|e| format!("missing zeus_plugin_free_string: {e}"))?;
        }

        let mut plugin = Self {
            name: name.to_string(),
            library,
            tools: Vec::new(),
        };
        plugin.tools = plugin.fetch_tool_specs()?;
        Ok(plugin)
    }

    fn fetch_tool_specs(&self) -> Result<Vec<ToolSpec>, String> {
        let raw = unsafe {
            let tool_specs_fn: libloading::Symbol<ToolSpecsFn> = self
                .library
                .get(b"zeus_plugin_tool_specs")
                .map_err(|e| e.to_string())?;
            let free_fn: libloading::Symbol<FreeStringFn> = self
                .library
                .get(b"zeus_plugin_free_string")
                .map_err(|e| e.to_string())?;
            take_string(tool_specs_fn(), &free_fn)
        }?;

        let specs: Vec<RawToolSpec> =
            serde_json::from_str(&raw).map_err(|e| format!("bad tool_specs JSON: {e}"))?;
        Ok(specs
            .into_iter()
            .map(|s| ToolSpec {
                name: s.name,
                description: s.description,
                parameters: s.parameters,
            })
            .collect())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tools(&self) -> &[ToolSpec] {
        &self.tools
    }

    pub fn call_tool(&self, tool_name: &str, args_json: &str) -> Result<PluginCallResult, String> {
        let name_c = CString::new(tool_name).map_err(|e| e.to_string())?;
        let args_c = CString::new(args_json).map_err(|e| e.to_string())?;

        let raw = unsafe {
            let call_fn: libloading::Symbol<CallToolFn> = self
                .library
                .get(b"zeus_plugin_call_tool")
                .map_err(|e| e.to_string())?;
            let free_fn: libloading::Symbol<FreeStringFn> = self
                .library
                .get(b"zeus_plugin_free_string")
                .map_err(|e| e.to_string())?;
            take_string(call_fn(name_c.as_ptr(), args_c.as_ptr()), &free_fn)
        }?;

        let result: RawCallResult =
            serde_json::from_str(&raw).map_err(|e| format!("bad call_tool result JSON: {e}"))?;
        Ok(PluginCallResult {
            content: result.content,
            is_error: result.is_error,
        })
    }
}

/// Copy a plugin-owned C string into an owned Rust `String`, then free the
/// original through the plugin's own free function (never the host's
/// allocator — see the module doc comment on why).
///
/// # Safety
/// `ptr` must be a valid, NUL-terminated UTF-8 C string pointer previously
/// returned by this same plugin (or null), and `free_fn` must be that
/// plugin's real `zeus_plugin_free_string` export.
unsafe fn take_string(
    ptr: *mut c_char,
    free_fn: &libloading::Symbol<FreeStringFn>,
) -> Result<String, String> {
    if ptr.is_null() {
        return Err("plugin returned a null string".to_string());
    }
    let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
    free_fn(ptr);
    Ok(s)
}

/// Platform-specific dynamic library extension.
fn dylib_extension() -> &'static str {
    if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    }
}

/// Best-effort load every dynamic library in `plugins_dir`: one that fails
/// to load, lacks a required export, or reports an incompatible ABI version
/// logs a warning and is skipped — the same "one bad extension shouldn't
/// take down the agent" policy already used for MCP servers.
pub fn load_all(plugins_dir: &Path) -> Vec<LoadedPlugin> {
    let Ok(entries) = std::fs::read_dir(plugins_dir) else {
        return Vec::new();
    };
    let ext = dylib_extension();
    let mut plugins = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some(ext) {
            continue;
        }
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        match LoadedPlugin::load(&name, &path) {
            Ok(plugin) => plugins.push(plugin),
            Err(e) => warn!(plugin = %name, error = %e, "failed to load plugin; skipping"),
        }
    }
    plugins
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Workspace root (parent of crates/zeus-agent).
    fn workspace_root() -> std::path::PathBuf {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        Path::new(&dir)
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    /// Path to the real compiled example plugin dylib, built as part of
    /// this workspace (`zeus-example-plugin`, crate-type = cdylib). This
    /// test only passes if the actual FFI boundary works — dynamic load,
    /// symbol resolution, ABI version check, JSON round-trip through raw
    /// C strings, and cross-allocator-safe string freeing — not just that
    /// the loader code compiles.
    fn example_plugin_path() -> std::path::PathBuf {
        let target_debug = workspace_root().join("target").join("debug");
        // Cargo emits cdylibs as `[lib]<crate>.<ext>`: no prefix on Windows,
        // `lib` prefix on Unix/macOS. The loader test reads the real built
        // artifact, so it must match cargo's actual output filename.
        let stem = if cfg!(windows) {
            "zeus_example_plugin"
        } else {
            "libzeus_example_plugin"
        };
        target_debug.join(format!("{stem}.{}", dylib_extension()))
    }

    /// Make sure the example plugin cdylib exists, building it via cargo if
    /// it doesn't — so the loader test is self-contained and never relies on
    /// a CI step or a prior manual build. `cargo test` itself only compiles
    /// the plugin into hashed `deps/` copies; the top-level `target/debug`
    /// artifact requires an explicit `cargo build -p zeus-example-plugin`.
    fn ensure_example_plugin() -> std::path::PathBuf {
        let path = example_plugin_path();
        if path.exists() {
            return path;
        }
        let status = std::process::Command::new("cargo")
            .args(["build", "-p", "zeus-example-plugin"])
            .current_dir(workspace_root())
            .status()
            .unwrap_or_else(|e| {
                panic!("failed to spawn `cargo build -p zeus-example-plugin`: {e}")
            });
        assert!(
            status.success(),
            "`cargo build -p zeus-example-plugin` failed ({status})"
        );
        assert!(
            path.exists(),
            "`cargo build -p zeus-example-plugin` succeeded but produced no artifact at {path:?}"
        );
        path
    }

    #[test]
    fn loads_real_compiled_plugin_and_calls_its_tool() {
        let path = ensure_example_plugin();
        let plugin = LoadedPlugin::load("example", &path).unwrap();

        assert_eq!(plugin.tools().len(), 1);
        assert_eq!(plugin.tools()[0].name, "shout");

        let result = plugin
            .call_tool("shout", r#"{"text":"hello plugin"}"#)
            .unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content, "HELLO PLUGIN");

        let unknown = plugin.call_tool("nope", "{}").unwrap();
        assert!(unknown.is_error);
    }

    #[test]
    fn load_all_skips_missing_directory_without_panicking() {
        let plugins = load_all(Path::new("this/does/not/exist"));
        assert!(plugins.is_empty());
    }

    #[test]
    fn load_all_finds_and_loads_the_example_plugin() {
        let src = ensure_example_plugin();
        let tmp = tempfile::TempDir::new().unwrap();
        let dest = tmp.path().join(format!("example.{}", dylib_extension()));
        std::fs::copy(&src, &dest).unwrap();

        let plugins = load_all(tmp.path());
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].name(), "example");
        assert_eq!(plugins[0].tools()[0].name, "shout");
    }

    #[test]
    fn load_all_skips_corrupt_dylib_without_panicking() {
        let tmp = tempfile::TempDir::new().unwrap();
        // A file with a dylib extension that is not a real dynamic library —
        // e.g. a leftover text file, a truncated download, a half-written
        // artifact. Loading it must fail gracefully and skip, not panic and
        // take down the whole agent.
        let corrupt = tmp.path().join(format!("broken.{}", dylib_extension()));
        std::fs::write(&corrupt, b"this is definitely not a PE/ELF/Mach-O binary").unwrap();

        let plugins = load_all(tmp.path());
        assert!(
            plugins.is_empty(),
            "corrupt dylib must be skipped, not loaded"
        );
    }

    #[test]
    fn load_all_skips_truncated_dylib_without_panicking() {
        let src = ensure_example_plugin();
        let tmp = tempfile::TempDir::new().unwrap();
        // A truncated real dylib: valid header-ish bytes but cut off, as a
        // crash during a plugin download/install would leave behind.
        let bytes = std::fs::read(&src).unwrap();
        let truncated = tmp.path().join(format!("cut.{}", dylib_extension()));
        std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();

        let plugins = load_all(tmp.path());
        assert!(
            plugins.is_empty(),
            "truncated dylib must be skipped, not loaded"
        );
    }

    #[test]
    fn load_all_keeps_good_plugin_alongside_bad_ones() {
        // One bad extension (a load failure) must not take down the good
        // ones — same "one bad extension shouldn't break the agent" policy
        // the loader documents.
        let src = ensure_example_plugin();
        let tmp = tempfile::TempDir::new().unwrap();
        let good = tmp.path().join(format!("good.{}", dylib_extension()));
        std::fs::copy(&src, &good).unwrap();
        let bad = tmp.path().join(format!("bad.{}", dylib_extension()));
        std::fs::write(&bad, b"not a real library either").unwrap();

        let plugins = load_all(tmp.path());
        assert_eq!(plugins.len(), 1, "only the good plugin should load");
        assert_eq!(plugins[0].name(), "good");
        assert_eq!(plugins[0].tools()[0].name, "shout");
    }
}
