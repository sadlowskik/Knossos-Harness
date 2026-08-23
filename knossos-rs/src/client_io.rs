//! Optional editor I/O: `fs/*`, `terminal/*`, `elicitation/*`.
//!
//! Installed only when the client advertised the capability. Without it the
//! harness uses the workspace jail on disk, which is the CLI and the default.
//! A client that advertises a capability and then fails degrades to disk /
//! subprocess rather than aborting the turn — matching Python `acp.py`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use crate::jsonrpc::{PeerHandle, RpcError};

const FS_TIMEOUT: Duration = Duration::from_secs(30);

/// ACP paths are ordinary absolute paths, not Win32's verbatim `\\?\` form.
/// The latter is useful internally but editor buffer maps key the former, so
/// sending it over the wire makes an unsaved file look like a different file.
pub(crate) fn wire_path(path: &Path) -> String {
    let raw = path.to_string_lossy();
    #[cfg(windows)]
    {
        if let Some(rest) = raw.strip_prefix(r"\\?\UNC\") {
            return format!(r"\\{rest}");
        }
        if let Some(rest) = raw.strip_prefix(r"\\?\") {
            return rest.to_string();
        }
    }
    raw.into_owned()
}

#[derive(Debug, Clone, Default)]
pub struct ClientCaps {
    pub fs_read: bool,
    pub fs_write: bool,
    pub terminal: bool,
    pub elicit: bool,
}

impl ClientCaps {
    pub fn from_initialize(params: &Value) -> Self {
        let caps = params.get("clientCapabilities").unwrap_or(&Value::Null);
        ClientCaps {
            fs_read: advertised(caps, &["fs", "readTextFile"]),
            fs_write: advertised(caps, &["fs", "writeTextFile"]),
            terminal: advertised(caps, &["terminal"]),
            elicit: advertised(caps, &["elicitation"])
                || advertised(caps, &["elicitation", "form"]),
        }
    }

    pub fn any_fs(&self) -> bool {
        self.fs_read || self.fs_write
    }
}

/// `{}` is how ACP spells "supported". False and missing are not.
fn advertised(caps: &Value, path: &[&str]) -> bool {
    let mut cur = caps;
    for key in path {
        cur = match cur.get(*key) {
            Some(v) => v,
            None => return false,
        };
    }
    !matches!(cur, Value::Bool(false) | Value::Null)
}

#[derive(Clone)]
pub struct ClientIo {
    peer: PeerHandle,
    session: String,
    cancel: Arc<AtomicBool>,
    pub caps: ClientCaps,
}

impl std::fmt::Debug for ClientIo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientIo")
            .field("session", &self.session)
            .field("caps", &self.caps)
            .finish_non_exhaustive()
    }
}

impl ClientIo {
    pub fn new(
        peer: PeerHandle,
        session: String,
        cancel: Arc<AtomicBool>,
        caps: ClientCaps,
    ) -> Self {
        ClientIo {
            peer,
            session,
            cancel,
            caps,
        }
    }

    fn call(
        &self,
        method: &str,
        params: Value,
        timeout: Option<Duration>,
    ) -> Result<Value, RpcError> {
        self.peer
            .request_with(method, Some(params), timeout, Some(&self.cancel))
    }

    /// Editor buffer, or `None` to fall through to disk.
    pub fn read_text(&self, path: &Path) -> Option<String> {
        if !self.caps.fs_read || self.cancel.load(Ordering::SeqCst) {
            return None;
        }
        let answer = self
            .call(
                "fs/read_text_file",
                json!({"sessionId": self.session, "path": wire_path(path)}),
                Some(FS_TIMEOUT),
            )
            .ok()?;
        answer
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    /// True if the editor accepted the write. False means use disk.
    pub fn write_text(&self, path: &Path, content: &str) -> bool {
        if !self.caps.fs_write || self.cancel.load(Ordering::SeqCst) {
            return false;
        }
        self.call(
            "fs/write_text_file",
            json!({
                "sessionId": self.session,
                "path": wire_path(path),
                "content": content,
            }),
            Some(FS_TIMEOUT),
        )
        .is_ok()
    }

    /// `(exit_code, output)` from the editor terminal, or `None` to subprocess.
    pub fn run_terminal(
        &self,
        cwd: &Path,
        argv: &[String],
        timeout: Duration,
    ) -> Option<(i32, String)> {
        if !self.caps.terminal || argv.is_empty() || self.cancel.load(Ordering::SeqCst) {
            return None;
        }
        let created = self
            .call(
                "terminal/create",
                json!({
                    "sessionId": self.session,
                    "command": argv[0],
                    "args": argv[1..],
                    "cwd": wire_path(cwd),
                }),
                Some(FS_TIMEOUT),
            )
            .ok()?;
        let terminal_id = created
            .get("terminalId")
            .and_then(Value::as_str)?
            .to_string();
        let ref_params = json!({"sessionId": self.session, "terminalId": terminal_id});
        let wait = timeout + FS_TIMEOUT;
        let exit_info = self.call("terminal/wait_for_exit", ref_params.clone(), Some(wait));
        let output = self.call("terminal/output", ref_params.clone(), Some(FS_TIMEOUT));
        let _ = self.call("terminal/release", ref_params, Some(FS_TIMEOUT));

        let exit_info = exit_info.ok()?;
        let output = output.ok()?;
        let mut body = output
            .get("output")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if output
            .get("truncated")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            body.push_str("\n\n[the editor truncated this output]");
        }
        if let Some(sig) = exit_info.get("signal").and_then(Value::as_str) {
            body.push_str(&format!("\n\n[terminated by signal {sig}]"));
        }
        let code = exit_info
            .get("exitCode")
            .and_then(Value::as_i64)
            .unwrap_or(-1) as i32;
        Some((code, body))
    }

    /// Structured question. `None` if the client has no elicitation or the user declined.
    pub fn ask(&self, question: &str, choices: &[String]) -> Option<String> {
        if !self.caps.elicit || self.cancel.load(Ordering::SeqCst) {
            return None;
        }
        let mut field = json!({
            "type": "string",
            "title": "Answer",
            "description": question,
        });
        if !choices.is_empty() {
            field["enum"] = json!(choices);
        }
        let reply = self
            .call(
                "elicitation/create",
                json!({
                    "sessionId": self.session,
                    "message": question,
                    "mode": "form",
                    "requestedSchema": {
                        "type": "object",
                        "properties": {"answer": field},
                        "required": ["answer"],
                    },
                }),
                None,
            )
            .ok()?;
        if reply.get("action").and_then(Value::as_str) != Some("accept") {
            return None;
        }
        reply.pointer("/content/answer").and_then(|v| {
            v.as_str()
                .map(str::to_string)
                .or_else(|| v.as_i64().map(|n| n.to_string()))
                .or_else(|| v.as_bool().map(|b| b.to_string()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_object_means_supported() {
        let caps = json!({"fs": {"readTextFile": true, "writeTextFile": {}}, "terminal": true, "elicitation": {"form": {}}});
        let c = ClientCaps::from_initialize(&json!({"clientCapabilities": caps}));
        assert!(c.fs_read);
        assert!(c.fs_write);
        assert!(c.terminal);
        assert!(c.elicit);
    }

    #[test]
    fn missing_caps_are_off() {
        let c = ClientCaps::from_initialize(&json!({}));
        assert!(!c.fs_read && !c.terminal && !c.elicit);
    }

    #[cfg(windows)]
    #[test]
    fn verbatim_windows_paths_are_normalized_for_editors() {
        assert_eq!(
            wire_path(Path::new(r"\\?\C:\work\file.rs")),
            r"C:\work\file.rs"
        );
        assert_eq!(
            wire_path(Path::new(r"\\?\UNC\server\share\file.rs")),
            r"\\server\share\file.rs"
        );
    }
}
