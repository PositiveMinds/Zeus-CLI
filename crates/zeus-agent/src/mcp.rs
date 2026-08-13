//! MCP (Model Context Protocol) client: connects to an external tool
//! server over stdio, discovers its tools, and lets the agent call them
//! alongside its own built-in tools.
//!
//! Transport is newline-delimited JSON-RPC 2.0 over the child process's
//! stdin/stdout (confirmed against the spec: MCP stdio framing is one JSON
//! value per line, NOT Content-Length/LSP-style headers). stderr is left
//! untouched (inherited) — per spec it's free-form server logs, not
//! protocol traffic, and must not be treated as an error signal.
//!
//! https://modelcontextprotocol.io/specification/2025-11-25/basic/transports

use crate::error::{AgentError, Result};
use serde::Deserialize;
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Latest spec version this client speaks. The server may negotiate an
/// older one back in its `initialize` response; per spec that's fine as
/// long as it's a version we understand (we don't currently reject any).
pub const PROTOCOL_VERSION: &str = "2025-11-25";

/// Cap on a single request/response round trip. A server that goes silent
/// (but stays alive, so `read_line` never sees EOF) must surface as an error
/// rather than hang the agent turn on a blocking stdio read.
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "inputSchema", default = "default_schema")]
    pub input_schema: Value,
}

fn default_schema() -> Value {
    serde_json::json!({ "type": "object" })
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        #[serde(rename = "mimeType", default)]
        mime_type: String,
    },
    #[serde(rename = "resource_link")]
    ResourceLink {
        #[serde(default)]
        uri: String,
        #[serde(default)]
        name: String,
    },
    #[serde(rename = "resource")]
    Resource,
    #[serde(rename = "audio")]
    Audio,
    // Forward-compatible: unknown block types (new annotations, future kinds)
    // don't break parsing of the response.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ToolCallResult {
    #[serde(default)]
    content: Vec<ContentBlock>,
    #[serde(rename = "isError", default)]
    pub is_error: bool,
}

impl ToolCallResult {
    /// Flatten the content blocks into plain text for feeding back into the
    /// conversation (images/audio/resources are summarized, not inlined).
    pub fn as_text(&self) -> String {
        let lines: Vec<String> = self
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.clone()),
                ContentBlock::ResourceLink { uri, name } => {
                    Some(format!("[resource: {name} ({uri})]"))
                }
                ContentBlock::Image { mime_type } => Some(format!("[image: {mime_type}]")),
                ContentBlock::Resource => Some("[embedded resource]".to_string()),
                ContentBlock::Audio => Some("[audio]".to_string()),
                ContentBlock::Unknown => None,
            })
            .collect();
        if lines.is_empty() {
            "(no content)".to_string()
        } else {
            lines.join("\n")
        }
    }
}

struct Conversation {
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
}

/// A connected MCP server: a spawned child process plus its discovered
/// tool list (fetched once at connect time, matching how built-in tool
/// specs are a fixed list — a long-lived agent session that needs to react
/// to `notifications/tools/list_changed` would need to re-fetch; out of
/// scope for this first client).
pub struct McpClient {
    name: String,
    child: Mutex<Child>,
    conversation: Mutex<Conversation>,
    next_id: AtomicI64,
    tools: Vec<McpTool>,
}

impl McpClient {
    /// Spawn `command` as an MCP server, perform the `initialize` handshake,
    /// fetch its tool list, and return the connected client.
    pub fn connect(name: &str, command: &str, args: &[String], cwd: &Path) -> Result<Self> {
        let mut cmd = Command::new(command);
        cmd.args(args);
        cmd.current_dir(cwd);
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        // Inherited, not piped/null: stderr is free-form server logs per
        // spec, not protocol traffic — let it pass through for visibility
        // rather than silently swallowing it or risking a full pipe buffer.
        let mut child = cmd
            .spawn()
            .map_err(|e| AgentError::Terminal(format!("mcp spawn '{name}' failed: {e}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| AgentError::Terminal(format!("mcp '{name}': no stdin handle")))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AgentError::Terminal(format!("mcp '{name}': no stdout handle")))?;

        let mut client = Self {
            name: name.to_string(),
            child: Mutex::new(child),
            conversation: Mutex::new(Conversation {
                stdin,
                reader: BufReader::new(stdout),
            }),
            next_id: AtomicI64::new(1),
            tools: Vec::new(),
        };

        client.initialize()?;
        client.tools = client.fetch_tools()?;
        Ok(client)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tools(&self) -> &[McpTool] {
        &self.tools
    }

    pub fn call_tool(&self, tool_name: &str, arguments: Value) -> Result<ToolCallResult> {
        let params = serde_json::json!({ "name": tool_name, "arguments": arguments });
        let result = self.send_request("tools/call", params)?;
        serde_json::from_value(result).map_err(|e| {
            AgentError::Terminal(format!(
                "mcp '{}': bad tools/call response: {e}",
                self.name
            ))
        })
    }

    fn initialize(&self) -> Result<()> {
        let params = serde_json::json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "zeus", "version": env!("CARGO_PKG_VERSION") },
        });
        self.send_request("initialize", params)?;
        self.send_notification("notifications/initialized")?;
        Ok(())
    }

    fn fetch_tools(&self) -> Result<Vec<McpTool>> {
        #[derive(Deserialize)]
        struct ToolsListResult {
            #[serde(default)]
            tools: Vec<McpTool>,
            #[serde(rename = "nextCursor", default)]
            next_cursor: Option<String>,
        }

        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let params = match &cursor {
                Some(c) => serde_json::json!({ "cursor": c }),
                None => serde_json::json!({}),
            };
            let result = self.send_request("tools/list", params)?;
            let page: ToolsListResult = serde_json::from_value(result).map_err(|e| {
                AgentError::Terminal(format!(
                    "mcp '{}': bad tools/list response: {e}",
                    self.name
                ))
            })?;
            tools.extend(page.tools);
            cursor = page.next_cursor;
            if cursor.is_none() {
                break;
            }
        }
        Ok(tools)
    }

    fn send_request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let mut conv = self.conversation.lock().unwrap();
        Self::write_line(&mut conv.stdin, &req)?;

        // Read the response on a scoped thread and wait bounded on the main
        // one: a server that goes silent while staying alive (read_line can
        // never EOF) must time out instead of blocking forever. On expiry we
        // kill the child — closing its stdout is what lets the reader thread
        // reach EOF, so `scope` joins cleanly. `thread::scope` returns this
        // closure's result, so `return` here flows straight out of
        // `send_request`.
        let reader = &mut conv.reader;
        let name = &self.name;
        let (result_tx, result_rx) = mpsc::channel();
        std::thread::scope(|s| {
            let _ = s.spawn(move || {
                let _ = result_tx.send(Self::read_response(reader, id, method, name));
            });
            let deadline = Instant::now() + RESPONSE_TIMEOUT;
            loop {
                if let Ok(r) = result_rx.try_recv() {
                    return r;
                }
                if Instant::now() >= deadline {
                    if let Ok(mut child) = self.child.lock() {
                        let _ = child.kill();
                    }
                    return Err(AgentError::Terminal(format!(
                        "mcp '{}': no response to '{method}' within {}s (server killed)",
                        self.name,
                        RESPONSE_TIMEOUT.as_secs()
                    )));
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        })
    }

    /// Consume server lines until the response matching `id` arrives (skipping
    /// notifications and stray responses), then return its result/error.
    fn read_response(
        reader: &mut BufReader<ChildStdout>,
        id: i64,
        method: &str,
        name: &str,
    ) -> Result<Value> {
        loop {
            let line = Self::read_line(reader)?;
            let value: Value = serde_json::from_str(&line).map_err(|e| {
                AgentError::Terminal(format!(
                    "mcp '{name}': non-JSON line from server: {e} (line: {line:?})"
                ))
            })?;
            // A request/id-bearing response we're waiting on; anything else
            // (a notification, or — shouldn't happen for a client that only
            // ever has one call in flight — a response to a different id)
            // is skipped rather than treated as fatal.
            if value.get("id").and_then(|v| v.as_i64()) != Some(id) {
                continue;
            }
            if let Some(err) = value.get("error") {
                return Err(AgentError::Terminal(format!(
                    "mcp '{name}' error calling {method}: {err}"
                )));
            }
            return Ok(value.get("result").cloned().unwrap_or(Value::Null));
        }
    }

    fn send_notification(&self, method: &str) -> Result<()> {
        let msg = serde_json::json!({ "jsonrpc": "2.0", "method": method });
        let mut conv = self.conversation.lock().unwrap();
        Self::write_line(&mut conv.stdin, &msg)
    }

    fn write_line(stdin: &mut ChildStdin, value: &Value) -> Result<()> {
        // Compact (not pretty-printed) serialization is required: MCP's
        // newline-delimited framing means the JSON itself must not contain
        // embedded newlines.
        let mut line = serde_json::to_string(value)
            .map_err(|e| AgentError::Terminal(format!("mcp serialize failed: {e}")))?;
        line.push('\n');
        stdin
            .write_all(line.as_bytes())
            .map_err(|e| AgentError::Terminal(format!("mcp write failed: {e}")))?;
        stdin
            .flush()
            .map_err(|e| AgentError::Terminal(format!("mcp flush failed: {e}")))?;
        Ok(())
    }

    fn read_line(reader: &mut BufReader<ChildStdout>) -> Result<String> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| AgentError::Terminal(format!("mcp read failed: {e}")))?;
            if n == 0 {
                return Err(AgentError::Terminal(
                    "mcp server closed stdout unexpectedly".into(),
                ));
            }
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                return Ok(trimmed.to_string());
            }
            // Blank line — skip and keep reading.
        }
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        // Best-effort teardown for a short-lived CLI process: just kill it.
        // (A more graceful shutdown would close stdin and wait first, but
        // given every call already went through a single blocking
        // request/response round trip, there's nothing in flight to let
        // finish gracefully by the time this runs.)
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// A tiny in-process "MCP server" implemented as a Python script written to
/// disk for tests: replies to `initialize`/`tools/list` with fixed data, and
/// echoes `tools/call` arguments back (or returns `isError: true` when asked
/// to fail) — enough to exercise the real client/transport code end to end,
/// not just parsing logic. `pub(crate)` so both this module's tests and
/// `tools.rs`'s integration test can spawn the same fixture.
#[cfg(test)]
pub(crate) fn write_test_server(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("test_mcp_server.py");
    let script = r#"
import sys, json

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    method = req.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": req["id"], "result": {
            "protocolVersion": "2025-11-25",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "test-server", "version": "0.0.1"},
        }})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": req["id"], "result": {"tools": [
            {"name": "echo", "description": "Echoes the input back",
             "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
        ]}})
    elif method == "tools/call":
        args = req["params"].get("arguments", {})
        if args.get("fail"):
            send({"jsonrpc": "2.0", "id": req["id"], "result": {
                "content": [{"type": "text", "text": "deliberate failure"}],
                "isError": True,
            }})
        else:
            send({"jsonrpc": "2.0", "id": req["id"], "result": {
                "content": [{"type": "text", "text": f"echo: {args.get('text', '')}"}],
                "isError": False,
            }})
"#;
    std::fs::write(&path, script).unwrap();
    path
}

#[cfg(test)]
pub(crate) fn python_cmd() -> &'static str {
    if cfg!(windows) {
        "python"
    } else {
        "python3"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn connects_lists_tools_and_calls_one() {
        let tmp = TempDir::new().unwrap();
        let script = write_test_server(tmp.path());
        let client = McpClient::connect(
            "test",
            python_cmd(),
            &[script.display().to_string()],
            tmp.path(),
        )
        .unwrap();

        assert_eq!(client.tools().len(), 1);
        assert_eq!(client.tools()[0].name, "echo");

        let result = client
            .call_tool("echo", serde_json::json!({ "text": "hello mcp" }))
            .unwrap();
        assert!(!result.is_error);
        assert_eq!(result.as_text(), "echo: hello mcp");
    }

    #[test]
    fn tool_execution_error_is_reported_via_is_error() {
        let tmp = TempDir::new().unwrap();
        let script = write_test_server(tmp.path());
        let client = McpClient::connect(
            "test",
            python_cmd(),
            &[script.display().to_string()],
            tmp.path(),
        )
        .unwrap();

        let result = client
            .call_tool("echo", serde_json::json!({ "fail": true }))
            .unwrap();
        assert!(result.is_error);
        assert_eq!(result.as_text(), "deliberate failure");
    }
}
