//! An LSP client: exact relationships between files.
//!
//! The parser made Argus exact about what each file *defines*. It still cannot
//! say what *uses* it — [`importers_of`](crate::argus::Argus::importers_of)
//! matches module paths by name, which is a guess that works until two crates
//! both have a `config` module. A language server already knows the answer:
//! rust-analyzer, pyright and the rest maintain a real cross-file index with
//! type resolution.
//!
//! So this asks them. Three queries carry almost all the value:
//!
//! | Query | Answers |
//! |---|---|
//! | `workspace/symbol` | every symbol in the project, exactly |
//! | `textDocument/references` | who uses this, across files |
//! | `textDocument/definition` | where does this actually come from |
//!
//! LSP is JSON-RPC like ACP and MCP, but **framed differently**:
//! `Content-Length` headers rather than one message per line. That single
//! difference is why this cannot reuse [`jsonrpc::Peer`](crate::jsonrpc::Peer),
//! and getting it wrong produces a client that appears to hang — the server is
//! waiting for a header that never arrives.
//!
//! Everything here is optional and failure-tolerant. A language server that is
//! not installed, is slow to index, or does not implement a capability must
//! degrade to what Argus already does rather than break retrieval.
//! [`LspClient::available`] says whether anything is actually connected.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// Servers index in the background and answer emptily until they are ready.
/// An empty answer is indistinguishable from "no references", so give it a
/// moment before believing one.
pub const INDEX_SETTLE: Duration = Duration::from_secs(2);

// ------------------------------------------------------------------------ uris

/// Characters that survive a URI path segment unescaped (RFC 3986 unreserved,
/// plus `/` and `:` — the first separates segments, the second is the Windows
/// drive marker and must not become `%3A`).
fn is_uri_safe(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_' | '~' | '/' | ':')
}

/// Strip the extended-length prefix `canonicalize` adds on Windows.
///
/// `\\?\C:\proj` is a valid path to the OS and gibberish to a language server,
/// which will report every URI as outside the workspace it was given.
fn plain(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest) => PathBuf::from(rest),
        None => path.to_path_buf(),
    }
}

pub fn path_to_uri(path: &Path) -> String {
    let resolved = plain(&path.canonicalize().unwrap_or_else(|_| path.to_path_buf()));
    let text = resolved.to_string_lossy().replace('\\', "/");
    let mut out = String::from("file://");
    // A POSIX path already begins with the separator; a Windows one begins with
    // a drive letter and needs it, so that `file:///C:/x` parses as an absolute
    // path rather than `C:` being read as a host.
    if !text.starts_with('/') {
        out.push('/');
    }
    for c in text.chars() {
        if is_uri_safe(c) {
            out.push(c);
        } else {
            let mut buf = [0u8; 4];
            for byte in c.encode_utf8(&mut buf).as_bytes() {
                out.push_str(&format!("%{byte:02X}"));
            }
        }
    }
    out
}

pub fn uri_to_path(uri: &str) -> PathBuf {
    let rest = uri.strip_prefix("file://").unwrap_or_else(|| {
        uri.strip_prefix("file:").unwrap_or(uri)
    });
    // `file://host/path` is not something a language server emits, but an empty
    // authority is, and both leave the path starting at the first `/`.
    let raw = percent_decode(rest);

    // Windows: `file:///C:/x` decodes to `/C:/x`, and the leading slash must go.
    let bytes = raw.as_bytes();
    let drive_prefixed = bytes.len() > 2 && bytes[0] == b'/' && bytes[2] == b':';
    if cfg!(windows) && drive_prefixed {
        PathBuf::from(&raw[1..])
    } else {
        PathBuf::from(raw)
    }
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
            if let Some(byte) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ------------------------------------------------------------------- locations

/// A place in the project. Lines are 0-based, as LSP sends them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    pub path: PathBuf,
    pub line: usize,
    pub character: usize,
}

impl Location {
    /// `path:line`, counting from 1 — LSP counts from 0, humans and editors
    /// do not.
    pub fn reference(&self) -> String {
        format!("{}:{}", self.path.display(), self.line + 1)
    }

    /// Parse a `Location`, `LocationLink` or `SymbolInformation` payload.
    pub fn from_lsp(raw: &Value) -> Option<Self> {
        let uri = raw
            .get("uri")
            .or_else(|| raw.get("targetUri"))
            .and_then(Value::as_str)?;
        let range = raw
            .get("range")
            .or_else(|| raw.get("targetSelectionRange"))
            .or_else(|| raw.get("targetRange"))?;
        let start = range.get("start")?;
        Some(Location {
            path: uri_to_path(uri),
            line: start.get("line").and_then(Value::as_u64).unwrap_or(0) as usize,
            character: start.get("character").and_then(Value::as_u64).unwrap_or(0) as usize,
        })
    }
}

// --------------------------------------------------------------------- servers

/// Language servers worth trying, per file extension, cheapest first.
///
/// Nothing is installed on the user's behalf: an absent server means the
/// exact-reference queries report themselves unavailable, which is a better
/// answer than guessing at symbol positions.
pub const SERVERS: &[(&str, &[&[&str]])] = &[
    (".rs", &[&["rust-analyzer"]]),
    (
        ".py",
        &[
            &["pyright-langserver", "--stdio"],
            &["pylsp"],
            &["jedi-language-server"],
        ],
    ),
    (".ts", &[&["typescript-language-server", "--stdio"]]),
    (".go", &[&["gopls"]]),
];

/// Whether this command can actually be started.
///
/// Python needed an import check here because `pip install python-lsp-server`
/// drops `pylsp.exe` into a Scripts directory that is routinely off PATH, so a
/// PATH probe reported "no language server" for one that was installed. Rust
/// servers are single binaries, so PATH is the whole question — but the probe
/// is a real spawn rather than a PATH walk, since a binary that is present and
/// not executable fails the same way an absent one does.
fn launchable(argv: &[&str]) -> bool {
    Command::new(argv[0])
        .args(&argv[1..])
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|mut child| {
            let _ = child.kill();
            let _ = child.wait();
        })
        .is_ok()
}

/// Start a language server suited to what is actually in `root`.
///
/// Picks by counting extensions rather than by configuration, so a mixed tree
/// gets the server for its majority language and a tree with no recognised
/// source gets nothing. Returns `None` — never an error — when no candidate is
/// installed or the handshake fails: exact references are an enhancement, and an
/// agent that cannot start without one is worse than one that says so.
pub fn for_workspace(root: impl AsRef<Path>, timeout: Duration) -> Option<LspClient> {
    let root = root.as_ref();
    let mut counts: Vec<(&str, usize)> = Vec::new();
    for (suffix, _) in SERVERS {
        // Walking a large tree is slow; stop as soon as the answer is clear.
        let found = count_files(root, suffix, 25);
        if found > 0 {
            counts.push((suffix, found));
        }
    }
    let (suffix, _) = counts.into_iter().max_by_key(|(_, n)| *n)?;
    let candidates = SERVERS.iter().find(|(s, _)| *s == suffix).map(|(_, c)| *c)?;

    for argv in candidates {
        if !launchable(argv) {
            continue;
        }
        match LspClient::spawn(argv, root, timeout) {
            Some(client) => {
                eprintln!("[lsp] {} for {suffix} in {}", argv[0], root.display());
                return Some(client);
            }
            None => eprintln!("[lsp] {} failed to start", argv[0]),
        }
    }
    eprintln!("[lsp] no language server available for {suffix}");
    None
}

fn count_files(root: &Path, suffix: &str, cap: usize) -> usize {
    let mut found = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else if name.ends_with(suffix) {
                found += 1;
                if found >= cap {
                    return found;
                }
            }
        }
    }
    found
}

// ---------------------------------------------------------------------- client

#[derive(Debug, Default)]
struct Slot {
    outcome: Mutex<Option<Result<Value, String>>>,
    woken: Condvar,
}

/// One language server, spoken to over `Content-Length` framed JSON-RPC.
pub struct LspClient {
    root: PathBuf,
    timeout: Duration,
    label: String,
    capabilities: Value,
    tx: Mutex<Box<dyn Write + Send>>,
    pending: Arc<Mutex<HashMap<i64, Arc<Slot>>>>,
    next_id: AtomicI64,
    ready: Arc<AtomicBool>,
    child: Mutex<Option<Child>>,
}

impl LspClient {
    /// Launch `command` and initialise it. `None` instead of an error.
    ///
    /// A missing language server is a normal condition — most machines do not
    /// have every one installed — so it must not be a failure that every caller
    /// has to guard.
    pub fn spawn(command: &[&str], root: impl AsRef<Path>, timeout: Duration) -> Option<Self> {
        let mut child = Command::new(command[0])
            .args(&command[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| eprintln!("[lsp] {} did not start: {err}", command[0]))
            .ok()?;

        let stdout = child.stdout.take()?;
        let stdin = child.stdin.take()?;
        if let Some(stderr) = child.stderr.take() {
            let label = command[0].to_string();
            std::thread::Builder::new()
                .name(format!("lsp-stderr-{label}"))
                .spawn(move || {
                    for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                        if !line.trim().is_empty() {
                            eprintln!("[lsp:{label}] {}", line.trim_end());
                        }
                    }
                })
                .ok();
        }

        let client = Self::over(
            command[0],
            Box::new(BufReader::new(stdout)),
            Box::new(stdin),
            root,
            timeout,
        );
        match client {
            Some(client) => {
                *client.child.lock().expect("child mutex") = Some(child);
                Some(client)
            }
            None => {
                let _ = child.kill();
                let _ = child.wait();
                None
            }
        }
    }

    /// Initialise over an already-open transport.
    ///
    /// Split out from [`spawn`](Self::spawn) so the framing — the part most
    /// likely to be wrong — can be exercised without depending on a language
    /// server being installed.
    pub fn over(
        label: &str,
        rx: Box<dyn BufRead + Send>,
        tx: Box<dyn Write + Send>,
        root: impl AsRef<Path>,
        timeout: Duration,
    ) -> Option<Self> {
        let root = plain(&root.as_ref().canonicalize().unwrap_or_else(|_| root.as_ref().into()));
        let client = LspClient {
            label: label.to_string(),
            capabilities: Value::Null,
            tx: Mutex::new(tx),
            pending: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicI64::new(0),
            ready: Arc::new(AtomicBool::new(true)),
            child: Mutex::new(None),
            root,
            timeout,
        };

        let pending = Arc::clone(&client.pending);
        let ready = Arc::clone(&client.ready);
        std::thread::Builder::new()
            .name("lsp-reader".into())
            .spawn(move || read_loop(rx, &pending, &ready))
            .ok()?;

        let uri = path_to_uri(&client.root);
        let name = client.root.file_name().map(|n| n.to_string_lossy().into_owned());
        let result = client.request(
            "initialize",
            json!({
                "processId": std::process::id(),
                "rootUri": uri,
                "workspaceFolders": [{"uri": uri, "name": name}],
                "capabilities": {
                    "workspace": {"symbol": {"dynamicRegistration": false}},
                    "textDocument": {
                        "references": {"dynamicRegistration": false},
                        "definition": {"dynamicRegistration": false},
                    },
                },
            }),
        );

        let result = match result {
            Ok(result) => result,
            Err(err) => {
                eprintln!("[lsp] initialize failed: {err}");
                client.stop();
                return None;
            }
        };

        let mut client = client;
        client.capabilities = result.get("capabilities").cloned().unwrap_or(Value::Null);
        client.notify("initialized", json!({}));
        Some(client)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn capabilities(&self) -> &Value {
        &self.capabilities
    }

    /// Whether anything is actually connected and answering.
    pub fn available(&self) -> bool {
        if !self.ready.load(Ordering::SeqCst) {
            return false;
        }
        // A killed server whose pipe has not yet reached EOF still reports
        // `ready`; the exit status is the earlier signal of the two.
        let status = self.child.lock().expect("child mutex").as_mut().map(Child::try_wait);
        !matches!(status, Some(Ok(Some(_))))
    }

    pub fn stop(&self) {
        self.ready.store(false, Ordering::SeqCst);
        self.notify("exit", Value::Null);
        if let Some(mut child) = self.child.lock().expect("child mutex").take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        // Anything still waiting must be released; the reader may be parked on
        // a pipe that only the server can close.
        let stranded: Vec<Arc<Slot>> =
            self.pending.lock().expect("pending mutex").drain().map(|(_, s)| s).collect();
        for slot in stranded {
            settle(&slot, Err("client stopped".into()));
        }
    }

    /// Give a background indexer a moment.
    ///
    /// Servers answer emptily while still indexing, and an empty answer is
    /// indistinguishable from "no references" — which would silently look like
    /// a correct result.
    pub fn wait_until_indexed(&self, settle_for: Duration) {
        std::thread::sleep(settle_for);
    }

    // ------------------------------------------------------------- queries

    /// Every symbol matching `query`. An empty query means all of them.
    pub fn workspace_symbols(&self, query: &str) -> Vec<Location> {
        if !self.available() {
            return Vec::new();
        }
        let raw = match self.request("workspace/symbol", json!({"query": query})) {
            Ok(raw) => raw,
            Err(err) => {
                eprintln!("[lsp] workspace/symbol failed: {err}");
                return Vec::new();
            }
        };
        raw.as_array()
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .filter_map(|item| Location::from_lsp(item.get("location").unwrap_or(item)))
            .collect()
    }

    /// Who uses the symbol at this position, across the whole project.
    ///
    /// This is the thing Argus cannot compute. `importers_of` matches module
    /// names and is right until two modules share one.
    pub fn references(
        &self,
        path: &Path,
        line: usize,
        character: usize,
        include_declaration: bool,
    ) -> Vec<Location> {
        self.locations(
            "textDocument/references",
            path,
            line,
            character,
            Some(json!({"context": {"includeDeclaration": include_declaration}})),
        )
    }

    pub fn definition(&self, path: &Path, line: usize, character: usize) -> Vec<Location> {
        self.locations("textDocument/definition", path, line, character, None)
    }

    fn locations(
        &self,
        method: &str,
        path: &Path,
        line: usize,
        character: usize,
        extra: Option<Value>,
    ) -> Vec<Location> {
        if !self.available() {
            return Vec::new();
        }
        let mut params = Map::new();
        params.insert("textDocument".into(), json!({"uri": path_to_uri(path)}));
        params.insert("position".into(), json!({"line": line, "character": character}));
        if let Some(Value::Object(extra)) = extra {
            params.extend(extra);
        }

        let raw = match self.request(method, Value::Object(params)) {
            Ok(raw) => raw,
            Err(err) => {
                eprintln!("[lsp] {method} failed: {err}");
                return Vec::new();
            }
        };
        // LSP allows a bare object or a list; a client that assumes a list
        // silently drops every single-result answer.
        match raw {
            Value::Null => Vec::new(),
            Value::Array(items) => items.iter().filter_map(Location::from_lsp).collect(),
            one => Location::from_lsp(&one).into_iter().collect(),
        }
    }

    // ------------------------------------------------------------ protocol

    pub fn request(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst) + 1;
        let slot = Arc::new(Slot::default());
        self.pending.lock().expect("pending mutex").insert(id, Arc::clone(&slot));

        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));

        let deadline = Instant::now() + self.timeout;
        let mut outcome = slot.outcome.lock().expect("slot mutex");
        loop {
            if let Some(settled) = outcome.take() {
                return settled;
            }
            if Instant::now() >= deadline {
                self.pending.lock().expect("pending mutex").remove(&id);
                return Err(format!("{method} timed out after {:?}", self.timeout));
            }
            let (guard, _) = slot
                .woken
                .wait_timeout(outcome, Duration::from_millis(50))
                .expect("slot mutex");
            outcome = guard;
        }
    }

    pub fn notify(&self, method: &str, params: Value) {
        self.send(&json!({"jsonrpc": "2.0", "method": method, "params": params}));
    }

    fn send(&self, message: &Value) {
        let body = serde_json::to_vec(message).expect("message is serialisable");
        // Content-Length framing, not newline framing. Sending ndjson here makes
        // the server wait forever for a header, which looks exactly like a hang.
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        let mut tx = self.tx.lock().expect("write mutex");
        if tx
            .write_all(header.as_bytes())
            .and_then(|()| tx.write_all(&body))
            .and_then(|()| tx.flush())
            .is_err()
        {
            self.ready.store(false, Ordering::SeqCst);
        }
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        self.stop();
    }
}

impl std::fmt::Debug for LspClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LspClient")
            .field("label", &self.label)
            .field("root", &self.root)
            .field("available", &self.ready.load(Ordering::SeqCst))
            .finish()
    }
}

fn settle(slot: &Slot, outcome: Result<Value, String>) {
    *slot.outcome.lock().expect("slot mutex") = Some(outcome);
    slot.woken.notify_all();
}

fn read_loop(
    mut rx: Box<dyn BufRead + Send>,
    pending: &Arc<Mutex<HashMap<i64, Arc<Slot>>>>,
    ready: &Arc<AtomicBool>,
) {
    while let Some(length) = read_header(&mut rx) {
        let mut body = vec![0u8; length];
        if rx.read_exact(&mut body).is_err() {
            break;
        }
        let Ok(message) = serde_json::from_slice::<Value>(&body) else { continue };

        // Server->client requests are ignored: none of the capabilities
        // advertised above require answering one.
        let Some(id) = message.get("id").and_then(Value::as_i64) else { continue };
        if message.get("method").is_some() {
            continue;
        }
        let slot = pending.lock().expect("pending mutex").remove(&id);
        if let Some(slot) = slot {
            let outcome = match message.get("error") {
                Some(Value::Object(err)) => Err(err
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown LSP error")
                    .to_string()),
                _ => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
            };
            settle(&slot, outcome);
        }
    }

    ready.store(false, Ordering::SeqCst);
    // Nothing will answer outstanding calls now.
    let stranded: Vec<Arc<Slot>> =
        pending.lock().expect("pending mutex").drain().map(|(_, s)| s).collect();
    for slot in stranded {
        settle(&slot, Err("language server closed the connection".into()));
    }
}

/// Read one header block, returning its `Content-Length`.
fn read_header(rx: &mut Box<dyn BufRead + Send>) -> Option<usize> {
    let mut length: Option<usize> = None;
    loop {
        let mut line = String::new();
        match rx.read_line(&mut line) {
            Ok(0) | Err(_) => return None,
            Ok(_) => {}
        }
        let text = line.trim();
        if text.is_empty() {
            return length; // a blank line ends the header
        }
        if let Some(value) = text.to_ascii_lowercase().strip_prefix("content-length:") {
            match value.trim().parse() {
                Ok(n) => length = Some(n),
                Err(_) => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Shutdown, TcpListener, TcpStream};
    use tempfile::TempDir;

    /// A scripted language server speaking genuine `Content-Length` framing.
    ///
    /// The framing is real on purpose: it is the one part of this module most
    /// likely to be wrong, and a mocked transport would skip it. Send
    /// newline-framed JSON to a language server and it waits forever for a
    /// header, which looks exactly like a hang.
    struct FakeServer {
        addr: std::net::SocketAddr,
    }

    impl FakeServer {
        fn new() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let addr = listener.local_addr().expect("addr");

            std::thread::spawn(move || {
                let (stream, _) = listener.accept().expect("accept");
                let mut out = stream.try_clone().expect("clone");
                let mut rx: Box<dyn BufRead + Send> = Box::new(BufReader::new(stream));
                let mut root = "file:///proj".to_string();

                while let Some(length) = read_header(&mut rx) {
                    let mut body = vec![0u8; length];
                    if rx.read_exact(&mut body).is_err() {
                        break;
                    }
                    let Ok(msg) = serde_json::from_slice::<Value>(&body) else { break };
                    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
                    let id = msg.get("id").cloned();

                    let loc = |uri: String, line: u64| {
                        json!({"uri": uri, "range": {
                            "start": {"line": line, "character": 0},
                            "end": {"line": line, "character": 4}}})
                    };
                    let result = match method {
                        "initialize" => {
                            if let Some(uri) =
                                msg.pointer("/params/rootUri").and_then(Value::as_str)
                            {
                                root = uri.to_string();
                            }
                            json!({"capabilities": {
                                "referencesProvider": true, "definitionProvider": true,
                                "workspaceSymbolProvider": true}})
                            }
                        "workspace/symbol" => json!([{
                            "name": "Router", "kind": 5,
                            "location": loc(format!("{root}/moe.rs"), 28)}]),
                        "textDocument/references" => json!([
                            loc(format!("{root}/naiads.rs"), 41),
                            loc(format!("{root}/full.rs"), 7)]),
                        "textDocument/definition" => loc(format!("{root}/moe.rs"), 28),
                        "textDocument/nothing" => Value::Null,
                        "exit" => break,
                        "initialized" => continue,
                        _ => {
                            if let Some(id) = id {
                                let err = json!({"jsonrpc": "2.0", "id": id,
                                    "error": {"code": -32601, "message": "unsupported"}});
                                if send_framed(&mut out, &err).is_err() {
                                    break;
                                }
                            }
                            continue;
                        }
                    };
                    let Some(id) = id else { continue };
                    let reply = json!({"jsonrpc": "2.0", "id": id, "result": result});
                    if send_framed(&mut out, &reply).is_err() {
                        break;
                    }
                }
            });
            FakeServer { addr }
        }

        fn client(&self, root: &Path) -> Option<LspClient> {
            let stream = TcpStream::connect(self.addr).expect("connect");
            LspClient::over(
                "fake",
                Box::new(BufReader::new(stream.try_clone().expect("clone"))),
                Box::new(stream),
                root,
                Duration::from_secs(15),
            )
        }
    }

    fn send_framed(out: &mut TcpStream, message: &Value) -> std::io::Result<()> {
        let body = serde_json::to_vec(message).expect("serialise");
        write!(out, "Content-Length: {}\r\n\r\n", body.len())?;
        out.write_all(&body)?;
        out.flush()
    }

    fn connected() -> (TempDir, FakeServer, LspClient) {
        let dir = TempDir::new().expect("tempdir");
        let server = FakeServer::new();
        let client = server.client(dir.path()).expect("handshake");
        (dir, server, client)
    }

    // ------------------------------------------------------------- lifecycle

    /// Not having a language server installed is normal, not exceptional.
    #[test]
    fn a_missing_server_returns_none_rather_than_failing() {
        let dir = TempDir::new().expect("tempdir");
        let client = LspClient::spawn(
            &["definitely-not-a-real-language-server-xyzzy"],
            dir.path(),
            Duration::from_secs(5),
        );
        assert!(client.is_none());
    }

    #[test]
    fn capabilities_are_recorded() {
        let (_dir, _server, client) = connected();
        assert!(client.available());
        assert_eq!(client.capabilities()["referencesProvider"], json!(true));
    }

    #[test]
    fn queries_on_a_stopped_client_return_empty() {
        let (dir, _server, client) = connected();
        client.stop();

        assert!(!client.available());
        assert!(client.workspace_symbols("Router").is_empty());
        assert!(client.references(&dir.path().join("x.rs"), 0, 0, false).is_empty());
    }

    /// Otherwise a crashed server hangs whatever asked it a question.
    #[test]
    fn a_dead_server_frees_a_blocked_caller() {
        let dir = TempDir::new().expect("tempdir");
        let server = FakeServer::new();
        let stream = TcpStream::connect(server.addr).expect("connect");
        let client = LspClient::over(
            "fake",
            Box::new(BufReader::new(stream.try_clone().expect("clone"))),
            Box::new(stream.try_clone().expect("clone")),
            dir.path(),
            Duration::from_secs(15),
        )
        .expect("handshake");

        stream.shutdown(Shutdown::Both).expect("shutdown");

        // Returns empty rather than blocking to the timeout.
        let started = Instant::now();
        assert!(client.references(&dir.path().join("x.rs"), 1, 1, false).is_empty());
        assert!(started.elapsed() < Duration::from_secs(10), "it waited for the timeout");
    }

    // ---------------------------------------------------------------- queries

    #[test]
    fn workspace_symbols_are_returned() {
        let (_dir, _server, client) = connected();
        let symbols = client.workspace_symbols("Router");

        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].path.file_name().unwrap(), "moe.rs");
        assert_eq!(symbols[0].line, 28);
    }

    /// The thing Argus cannot compute: who uses this, across the project.
    #[test]
    fn references_span_files() {
        let (dir, _server, client) = connected();
        let refs = client.references(&dir.path().join("moe.rs"), 28, 6, false);

        let mut names: Vec<String> = refs
            .iter()
            .map(|r| r.path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, ["full.rs", "naiads.rs"]);
    }

    /// LSP allows either; a client that assumes a list drops the answer.
    #[test]
    fn definition_accepts_a_single_object_not_just_a_list() {
        let (dir, _server, client) = connected();
        let defs = client.definition(&dir.path().join("naiads.rs"), 41, 10);

        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].path.file_name().unwrap(), "moe.rs");
    }

    #[test]
    fn a_null_result_is_empty_not_an_error() {
        let (dir, _server, client) = connected();
        assert!(client
            .locations("textDocument/nothing", &dir.path().join("x.rs"), 0, 0, None)
            .is_empty());
    }

    #[test]
    fn an_unsupported_method_does_not_propagate_as_a_failure() {
        let (dir, _server, client) = connected();
        assert!(client
            .locations("textDocument/whatever", &dir.path().join("x.rs"), 0, 0, None)
            .is_empty());
        assert!(client.available(), "one unsupported call must not end the session");
    }

    // ------------------------------------------------------------------- uris

    #[test]
    fn uris_round_trip() {
        let dir = TempDir::new().expect("tempdir");
        let original = dir.path().join("some file.rs");
        std::fs::write(&original, "").expect("write");
        let original = original.canonicalize().expect("canonicalize");

        assert_eq!(uri_to_path(&path_to_uri(&original)), plain(&original));
    }

    #[test]
    fn a_uri_with_spaces_is_decoded() {
        assert_eq!(uri_to_path("file:///proj/my%20file.rs").file_name().unwrap(), "my file.rs");
    }

    #[test]
    fn a_space_is_escaped_on_the_way_out() {
        let uri = path_to_uri(Path::new("/proj/my file.rs"));
        assert!(uri.contains("my%20file.rs"), "{uri}");
        assert!(!uri.contains(' '), "{uri}");
    }

    #[cfg(windows)]
    #[test]
    fn windows_drive_letters_lose_the_leading_slash() {
        let path = uri_to_path("file:///C:/proj/x.rs");
        assert!(path.to_string_lossy().starts_with("C:"), "{}", path.display());
    }

    /// A drive letter must not be escaped, or the server reads `C%3A` as a host.
    #[test]
    fn a_drive_letter_survives_encoding() {
        assert!(!path_to_uri(Path::new(r"C:\proj\x.rs")).contains("%3A"));
    }

    /// LSP lines are 0-based; humans and editors count from 1.
    #[test]
    fn a_location_reports_a_one_based_reference() {
        let loc = Location { path: PathBuf::from("/a/b.rs"), line: 27, character: 0 };
        assert!(loc.reference().ends_with(":28"));
    }

    #[test]
    fn a_location_without_a_range_is_rejected() {
        assert!(Location::from_lsp(&json!({"uri": "file:///x"})).is_none());
        assert!(Location::from_lsp(&json!({"range": {"start": {"line": 1}}})).is_none());
    }

    #[test]
    fn a_location_link_is_accepted_as_well_as_a_location() {
        let link = json!({
            "targetUri": "file:///proj/moe.rs",
            "targetSelectionRange": {"start": {"line": 3, "character": 6},
                                     "end": {"line": 3, "character": 12}},
        });
        let loc = Location::from_lsp(&link).expect("a LocationLink is a location");
        assert_eq!((loc.line, loc.character), (3, 6));
    }

    // ----------------------------------------------- integration with argus

    /// `users_of` answers with resolved references, not name matches.
    #[test]
    fn argus_uses_the_language_server_for_real_references() {
        let dir = TempDir::new().expect("tempdir");
        let root = dir.path();
        std::fs::write(root.join("moe.rs"), "pub struct Router;\n").expect("write");
        std::fs::write(root.join("naiads.rs"), "use crate::moe::Router;\n").expect("write");
        std::fs::write(root.join("full.rs"), "use crate::moe::Router;\n").expect("write");

        let server = FakeServer::new();
        let client = server.client(root).expect("handshake");

        let mut argus = crate::argus::Argus::new(root).with_lsp(client);
        argus.scan();
        let router = argus
            .symbols_in("moe.rs")
            .into_iter()
            .find(|s| s.name == "Router")
            .expect("Router is indexed")
            .clone();

        let mut users = argus.users_of(&router);
        users.sort();
        assert_eq!(users, ["full.rs", "naiads.rs"]);
        assert!(!users.contains(&"moe.rs".to_string()), "a symbol is not its own user");
    }

    /// No server must mean "no exact answer", not "no retrieval".
    #[test]
    fn argus_without_a_server_falls_back_rather_than_failing() {
        let dir = TempDir::new().expect("tempdir");
        std::fs::write(dir.path().join("moe.rs"), "pub struct Router;\n").expect("write");

        let mut argus = crate::argus::Argus::new(dir.path());
        argus.scan();
        let router = argus
            .symbols_in("moe.rs")
            .into_iter()
            .find(|s| s.name == "Router")
            .expect("Router is indexed")
            .clone();

        assert!(argus.users_of(&router).is_empty());
        // The approximate path still works, which is the point of keeping it.
        assert!(argus.importers_of("moe.rs").is_empty());
    }
}
