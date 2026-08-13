//! Example native plugin implementing the zeus plugin ABI (version 1).
//! Exposes one tool, "shout", which upper-cases its input text.
//!
//! ABI contract (must match the host loader in `zeus-agent`'s `plugin.rs`):
//! everything crosses the FFI boundary as JSON inside NUL-terminated C
//! strings — never a native Rust type — specifically so this works
//! regardless of Rust ABI (in)stability across compiler versions: the only
//! thing that has to match on both sides is the C calling convention, which
//! `extern "C"` guarantees; no Rust struct layout is ever shared.
//!
//! - `zeus_plugin_abi_version() -> u32` — must equal the host's expected
//!   version exactly; mismatches are rejected, not tolerated.
//! - `zeus_plugin_tool_specs() -> *mut c_char` — heap-allocated JSON array
//!   of `{name, description, parameters}`. Ownership transfers to the
//!   caller, who must free it via `zeus_plugin_free_string` (never the
//!   host's own allocator — plugin and host may use different allocators,
//!   especially across a Windows DLL boundary, so cross-allocator free is
//!   undefined behavior).
//! - `zeus_plugin_call_tool(name, args_json) -> *mut c_char` — `name` and
//!   `args_json` are borrowed C strings valid only for the call; returns a
//!   heap-allocated JSON `{content, is_error}`, same ownership rule as above.
//! - `zeus_plugin_free_string(s)` — frees a string this plugin returned;
//!   null is a documented no-op.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

const ABI_VERSION: u32 = 1;

#[no_mangle]
pub extern "C" fn zeus_plugin_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn zeus_plugin_tool_specs() -> *mut c_char {
    let specs = serde_json::json!([
        {
            "name": "shout",
            "description": "Upper-cases the given text (example native plugin tool).",
            "parameters": {
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"]
            }
        }
    ]);
    to_c_string(specs.to_string())
}

/// # Safety
/// `name` and `args_json` must be valid, NUL-terminated UTF-8 C strings for
/// the duration of this call — the host guarantees this per the ABI contract.
#[no_mangle]
pub unsafe extern "C" fn zeus_plugin_call_tool(
    name: *const c_char,
    args_json: *const c_char,
) -> *mut c_char {
    let name = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(_) => return to_c_string(err_result("tool name was not valid UTF-8").to_string()),
    };
    let args_str = match CStr::from_ptr(args_json).to_str() {
        Ok(s) => s,
        Err(_) => return to_c_string(err_result("arguments were not valid UTF-8").to_string()),
    };
    let args: serde_json::Value = serde_json::from_str(args_str).unwrap_or(serde_json::Value::Null);

    let result = match name {
        "shout" => {
            let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
            ok_result(&text.to_uppercase())
        }
        other => err_result(&format!("unknown tool: {other}")),
    };
    to_c_string(result.to_string())
}

/// # Safety
/// `s` must be a pointer previously returned by `zeus_plugin_tool_specs` or
/// `zeus_plugin_call_tool` from this same plugin instance, or null.
#[no_mangle]
pub unsafe extern "C" fn zeus_plugin_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

fn ok_result(content: &str) -> serde_json::Value {
    serde_json::json!({ "content": content, "is_error": false })
}

fn err_result(message: &str) -> serde_json::Value {
    serde_json::json!({ "content": message, "is_error": true })
}

fn to_c_string(s: String) -> *mut c_char {
    // A raw NUL byte inside `s` would truncate silently when the host reads
    // it back via CStr; serde_json's text output never contains one
    // (control characters are \u-escaped), so this is safe for this
    // plugin's own output. Falling back rather than `.unwrap()`-panicking
    // across the FFI boundary on a genuine NUL, which would be UB.
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => CString::new(
            "{\"content\":\"internal error: NUL byte in plugin output\",\"is_error\":true}",
        )
        .unwrap()
        .into_raw(),
    }
}
