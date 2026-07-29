//! MCP: connecting Knossos to tools it did not ship with.
//!
//! The Model Context Protocol is how an agent reaches things outside its own
//! process — databases, browsers, issue trackers, whatever someone has wrapped.
//! Like ACP it is JSON-RPC 2.0 over newline-delimited stdio, so the transport is
//! already here: [`jsonrpc::Peer`](crate::jsonrpc::Peer) does both.
//!
//! ACP passes `mcpServers` in `session/new`. Knossos accepted that parameter and
//! discarded it, which meant a client could configure tools that silently never
//! appeared. This connects them instead.
//!
//! The handshake is three calls:
//!
//! | Call | Purpose |
//! |---|---|
//! | `initialize` | exchange protocol version and capabilities |
//! | `notifications/initialized` | a notification; the server may wait for it |
//! | `tools/list` | what this server can do |
//!
//! Each remote tool is then wrapped as an ordinary [`Tool`] and registered, so
//! Talos dispatches it exactly like a local one. Names are prefixed with the
//! server's label (`github.create_issue`) because two servers may both offer
//! `search`, and silently shadowing one with the other would be a bug nobody
//! could see.
//!
//! Failure is contained by design. A server that will not start, times out, or
//! returns nonsense must not take the session with it — the agent keeps its
//! local tools and the failure is reported once. An editor that cannot open
//! because a side-car is down is worse than one missing a feature.
//!
//! # What a remote tool costs you
//!
//! [`McpTool::consequential`] is `true` for every remote tool, without asking
//! the server. The trait leaves that question undefaulted precisely so it gets
//! answered deliberately, and the honest answer here is that we cannot know: a
//! remote tool acts through its own process, so neither the path jail nor the
//! sandbox constrains it, and a server's own claim about itself is not evidence.
//! Treating one as a read would let it satisfy Talos's "did any work" test
//! without anything observable having happened.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::jsonrpc::{Peer, PeerHandle, RpcError, METHOD_NOT_FOUND};
use crate::tools::{Tool, ToolCtx, ToolOutput};

/// MCP's own protocol version, distinct from ACP's.
pub const PROTOCOL_VERSION: &str = "2025-06-18";
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
pub const CALL_TIMEOUT: Duration = Duration::from_secs(120);

/// A server declaration, as ACP delivers it in `session/new`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct McpServer {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
}

impl McpServer {
    /// Parse one entry of `mcpServers`, or `None` if it is unusable.
    ///
    /// ACP carries `env` as a list of `{name, value}` objects; a plain mapping
    /// is accepted too, because that is what most hand-written configs contain.
    pub fn from_acp(raw: &Value) -> Option<Self> {
        let name = raw.get("name")?.as_str().filter(|s| !s.is_empty())?;
        let command = raw.get("command")?.as_str().filter(|s| !s.is_empty())?;

        let mut env = BTreeMap::new();
        match raw.get("env") {
            Some(Value::Object(map)) => {
                for (k, v) in map {
                    env.insert(k.clone(), stringify(v));
                }
            }
            Some(Value::Array(entries)) => {
                for entry in entries {
                    let Some(key) = entry.get("name").and_then(Value::as_str) else { continue };
                    if key.is_empty() {
                        continue;
                    }
                    let value = entry.get("value").map(stringify).unwrap_or_default();
                    env.insert(key.to_string(), value);
                }
            }
            _ => {}
        }

        let args = raw
            .get("args")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(stringify).collect())
            .unwrap_or_default();

        Some(McpServer { name: name.to_string(), command: command.to_string(), args, env })
    }
}

/// JSON scalars keep their plain form; `"8080"` and `8080` must not differ once
/// they reach an environment block.
fn stringify(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// What a server said one of its tools is.
#[derive(Debug, Clone)]
struct RemoteSpec {
    remote_name: String,
    local_name: String,
    description: String,
    schema: Value,
}

/// One connected MCP server.
pub struct McpClient {
    server: McpServer,
    handle: PeerHandle,
    /// Held so the reader and worker threads can be joined on shutdown.
    peer: Mutex<Option<Peer>>,
    child: Mutex<Option<Child>>,
    /// One in-flight call per server. MCP servers are not required to handle
    /// concurrent requests, and a wrong answer attributed to the wrong call is
    /// harder to notice than a slow one.
    call_lock: Mutex<()>,
    specs: Vec<RemoteSpec>,
}

impl McpClient {
    /// Start `server` as a subprocess and complete the MCP handshake.
    pub fn connect(server: McpServer) -> Result<Arc<Self>> {
        let mut child = Command::new(&server.command)
            .args(&server.args)
            .envs(&server.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("starting {}", server.command))?;

        let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;

        // Drain stderr, or a chatty server fills the pipe and blocks mid-call.
        if let Some(stderr) = child.stderr.take() {
            let label = server.name.clone();
            std::thread::Builder::new()
                .name(format!("mcp-stderr-{label}"))
                .spawn(move || {
                    for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                        eprintln!("[mcp:{label}] {}", line.trim_end());
                    }
                })
                .ok();
        }

        let client = Self::over(server, Box::new(BufReader::new(stdout)), Box::new(stdin));
        match client {
            Ok(client) => {
                *client.child.lock().expect("child mutex") = Some(child);
                Ok(client)
            }
            Err(err) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(err)
            }
        }
    }

    /// Complete the MCP handshake over an already-open transport.
    ///
    /// Split out from [`connect`](Self::connect) so the protocol can be
    /// exercised without a subprocess: spawning is a few lines, the handshake
    /// is where the behaviour is.
    pub fn over(
        server: McpServer,
        rx: Box<dyn BufRead + Send>,
        tx: Box<dyn Write + Send>,
    ) -> Result<Arc<Self>> {
        let mut peer = Peer::new(rx, tx);
        let handle = peer.handle();
        // Server->client requests. Nothing is supported yet, but requests are
        // still answered — silence would block the server indefinitely.
        peer.start(|method: &str, _: Option<Value>, is_request: bool| {
            if is_request {
                Err(RpcError::new(
                    METHOD_NOT_FOUND,
                    format!("{method} is not supported by this client"),
                ))
            } else {
                Ok(Value::Null)
            }
        });

        let handshake = (|| -> Result<Vec<RemoteSpec>> {
            handle.request_with(
                "initialize",
                Some(json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "knossos", "version": "0.1.0"},
                })),
                Some(CONNECT_TIMEOUT),
                None,
            )?;
            // Some servers wait for this before answering anything else.
            handle.notify("notifications/initialized", Some(json!({})));

            let listed = handle.request_with(
                "tools/list",
                Some(json!({})),
                Some(CONNECT_TIMEOUT),
                None,
            )?;
            Ok(parse_tools(&server.name, &listed))
        })();

        let specs = match handshake {
            Ok(specs) => specs,
            Err(err) => {
                handle.close();
                drop(peer);
                return Err(err);
            }
        };

        eprintln!("[mcp] {}: {} tool(s)", server.name, specs.len());
        Ok(Arc::new(McpClient {
            server,
            handle,
            peer: Mutex::new(Some(peer)),
            child: Mutex::new(None),
            call_lock: Mutex::new(()),
            specs,
        }))
    }

    pub fn name(&self) -> &str {
        &self.server.name
    }

    /// The remote tools, wrapped so Talos cannot tell them apart from local ones.
    pub fn tools(self: &Arc<Self>) -> Vec<Box<dyn Tool>> {
        self.specs
            .iter()
            .map(|spec| {
                Box::new(McpTool { client: Arc::clone(self), spec: spec.clone() }) as Box<dyn Tool>
            })
            .collect()
    }

    /// Invoke one remote tool. Blocking; callers on the runtime go through
    /// [`McpTool::run`], which moves this off the async worker.
    pub fn call_tool(&self, name: &str, args: &Value) -> ToolOutput {
        if self.handle.is_closed() {
            return ToolOutput::error(format!("{} is not connected", self.server.name));
        }

        let raw = {
            let _serialised = self.call_lock.lock().expect("call mutex");
            self.handle.request_with(
                "tools/call",
                Some(json!({"name": name, "arguments": args})),
                Some(CALL_TIMEOUT),
                None,
            )
        };

        match raw {
            Ok(raw) => {
                let content = render(raw.get("content"));
                if raw.get("isError").and_then(Value::as_bool).unwrap_or(false) {
                    ToolOutput::error(content)
                } else {
                    ToolOutput::ok(content)
                }
            }
            Err(err) => ToolOutput::error(format!("{name}: {}", err.message)),
        }
    }

    /// Shut the server down. Idempotent.
    ///
    /// The threads are dropped rather than joined. The reader is parked in a
    /// blocking read on the server's stdout and `PeerHandle::close` cannot
    /// interrupt that — killing the child closes the pipe, which is what
    /// actually ends it. Joining here would instead wait on a thread that only
    /// the pipe can release, which is a hang whenever the server outlives us.
    pub fn close(&self) {
        self.handle.close();
        if let Some(mut child) = self.child.lock().expect("child mutex").take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        drop(self.peer.lock().expect("peer mutex").take());
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        self.close();
    }
}

fn parse_tools(server_name: &str, listed: &Value) -> Vec<RemoteSpec> {
    listed
        .get("tools")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .filter_map(|t| {
            let remote_name = t.get("name")?.as_str().filter(|s| !s.is_empty())?;
            Some(RemoteSpec {
                local_name: format!("{server_name}.{remote_name}"),
                description: t
                    .get("description")
                    .and_then(Value::as_str)
                    .filter(|d| !d.is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("{remote_name} via {server_name}")),
                schema: t
                    .get("inputSchema")
                    .filter(|s| s.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
                remote_name: remote_name.to_string(),
            })
        })
        .collect()
}

/// Flatten MCP content blocks into text.
///
/// Non-text blocks are named rather than dropped: an engine told `[image
/// content]` can ask for something else, while an engine told nothing assumes
/// the call returned empty.
fn render(content: Option<&Value>) -> String {
    let Some(content) = content else { return String::new() };
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let parts: Vec<String> = blocks.iter().map(render_block).collect();
            parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n")
        }
        block => render_block(block),
    }
}

fn render_block(block: &Value) -> String {
    let Some(obj) = block.as_object() else { return stringify(block) };
    match obj.get("type").and_then(Value::as_str) {
        Some("text") => obj.get("text").map(stringify).unwrap_or_default(),
        Some("resource") => {
            let resource = obj.get("resource").cloned().unwrap_or(Value::Null);
            let text = resource.get("text").map(stringify).unwrap_or_default();
            if text.is_empty() {
                resource.get("uri").map(stringify).unwrap_or_default()
            } else {
                text
            }
        }
        other => format!("[{} content]", other.unwrap_or("unknown")),
    }
}

/// A remote tool, wrapped so Talos cannot tell it apart from a local one.
pub struct McpTool {
    client: Arc<McpClient>,
    spec: RemoteSpec,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.spec.local_name
    }

    fn description(&self) -> &str {
        &self.spec.description
    }

    fn schema(&self) -> Value {
        self.spec.schema.clone()
    }

    /// Always. See the module docs: a remote tool acts through its own process,
    /// so nothing here can bound what it touches.
    fn consequential(&self) -> bool {
        true
    }

    /// `ctx` is unused, and that is the point worth stating: a remote tool acts
    /// through its own server, so the workspace sandbox cannot constrain it.
    /// Connecting an MCP server widens what the agent can reach.
    async fn run(&self, input: &Value, _ctx: &ToolCtx) -> Result<ToolOutput> {
        let client = Arc::clone(&self.client);
        let remote_name = self.spec.remote_name.clone();
        let local_name = self.spec.local_name.clone();
        let input = input.clone();
        // The peer is thread-blocking by design, so the call must leave the
        // async worker or a slow server stalls every other task on it.
        tokio::task::spawn_blocking(move || client.call_tool(&remote_name, &input))
            .await
            .or_else(|err| Ok(ToolOutput::error(format!("{local_name} failed: {err}"))))
    }
}

/// Connect every declared server. Returns the clients and the failures.
///
/// One bad server must not take the session with it: the agent keeps its local
/// tools, and the failure is reported once rather than on every later call.
pub fn connect_all(raw_servers: &[Value]) -> (Vec<Arc<McpClient>>, Vec<String>) {
    let mut clients = Vec::new();
    let mut errors = Vec::new();

    for raw in raw_servers {
        let Some(server) = McpServer::from_acp(raw) else {
            let mut shown = raw.to_string();
            shown.truncate(120);
            errors.push(format!("unusable mcpServers entry: {shown}"));
            continue;
        };
        let name = server.name.clone();
        match McpClient::connect(server) {
            Ok(client) => clients.push(client),
            Err(err) => {
                eprintln!("[mcp] {name} failed to start: {err}");
                errors.push(format!("{name}: {err}"));
            }
        }
    }

    (clients, errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A scripted MCP server on a loopback socket.
    ///
    /// Real enough to exercise the handshake — it answers `initialize`,
    /// `tools/list` and `tools/call` in order — without depending on an
    /// interpreter being installed to run a side-car.
    struct FakeServer {
        addr: std::net::SocketAddr,
        calls: Arc<Mutex<Vec<Value>>>,
    }

    impl FakeServer {
        fn new(listed: Value) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
            let addr = listener.local_addr().expect("addr");
            let calls = Arc::new(Mutex::new(Vec::new()));
            let server = FakeServer { addr, calls: Arc::clone(&calls) };

            std::thread::spawn(move || {
                let (stream, _) = listener.accept().expect("accept");
                let mut out = stream.try_clone().expect("clone");
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    let Ok(msg) = serde_json::from_str::<Value>(&line) else { continue };
                    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
                    let Some(id) = msg.get("id") else { continue };

                    let result = match method {
                        "initialize" => json!({"protocolVersion": PROTOCOL_VERSION}),
                        "tools/list" => listed.clone(),
                        "tools/call" => {
                            let params = msg.get("params").cloned().unwrap_or(Value::Null);
                            calls.lock().expect("calls").push(params.clone());
                            let name =
                                params.get("name").and_then(Value::as_str).unwrap_or_default();
                            if name == "explode" {
                                json!({"content": [{"type": "text", "text": "kaboom"}],
                                       "isError": true})
                            } else {
                                json!({"content": [{"type": "text", "text": "done"}]})
                            }
                        }
                        _ => json!({}),
                    };
                    let reply = json!({"jsonrpc": "2.0", "id": id, "result": result});
                    if writeln!(out, "{reply}").is_err() {
                        break;
                    }
                    let _ = out.flush();
                }
            });
            server
        }

        fn client(&self, name: &str) -> Result<Arc<McpClient>> {
            let stream = TcpStream::connect(self.addr).expect("connect");
            let rx = Box::new(BufReader::new(stream.try_clone().expect("clone")));
            McpClient::over(
                McpServer { name: name.into(), ..Default::default() },
                rx,
                Box::new(stream),
            )
        }
    }

    fn one_tool() -> Value {
        json!({"tools": [{
            "name": "create_issue",
            "description": "Open an issue",
            "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}},
        }]})
    }

    // ------------------------------------------------------------- declarations

    #[test]
    fn an_acp_declaration_becomes_a_server() {
        let raw = json!({
            "name": "github", "command": "npx",
            "args": ["-y", "server-github"],
            "env": [{"name": "TOKEN", "value": "abc"}],
        });
        let server = McpServer::from_acp(&raw).expect("usable");
        assert_eq!(server.name, "github");
        assert_eq!(server.args, ["-y", "server-github"]);
        assert_eq!(server.env.get("TOKEN").map(String::as_str), Some("abc"));
    }

    /// Most hand-written configs use a mapping, whatever ACP's own shape is.
    #[test]
    fn env_is_accepted_as_a_mapping_too() {
        let raw = json!({"name": "db", "command": "srv", "env": {"PORT": 8080}});
        let server = McpServer::from_acp(&raw).expect("usable");
        assert_eq!(server.env.get("PORT").map(String::as_str), Some("8080"));
    }

    #[test]
    fn a_declaration_without_a_command_is_unusable() {
        for raw in [
            json!({"name": "x"}),
            json!({"command": "y"}),
            json!({"name": "", "command": "y"}),
            json!({"name": "x", "command": ""}),
            json!("not an object"),
        ] {
            assert!(McpServer::from_acp(&raw).is_none(), "{raw} should be rejected");
        }
    }

    // ---------------------------------------------------------------- handshake

    #[test]
    fn connecting_discovers_the_servers_tools() {
        let server = FakeServer::new(one_tool());
        let client = server.client("github").expect("handshake");
        let tools = client.tools();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "github.create_issue", "the server label must prefix it");
        assert_eq!(tools[0].description(), "Open an issue");
        assert_eq!(tools[0].schema()["properties"]["title"]["type"], json!("string"));
    }

    /// Two servers may both offer `search`; shadowing one would be invisible.
    #[test]
    fn two_servers_offering_the_same_tool_do_not_collide() {
        let listed = json!({"tools": [{"name": "search"}]});
        let (a, b) = (FakeServer::new(listed.clone()), FakeServer::new(listed));
        let (ca, cb) = (a.client("github").expect("a"), b.client("jira").expect("b"));

        assert_eq!(ca.tools()[0].name(), "github.search");
        assert_eq!(cb.tools()[0].name(), "jira.search");
    }

    #[test]
    fn a_tool_without_a_description_still_gets_one() {
        let server = FakeServer::new(json!({"tools": [{"name": "ping"}]}));
        let client = server.client("net").expect("handshake");
        let tools = client.tools();

        assert_eq!(tools[0].description(), "ping via net");
        assert_eq!(tools[0].schema(), json!({"type": "object", "properties": {}}));
    }

    #[test]
    fn a_nameless_tool_entry_is_skipped_rather_than_fatal() {
        let listed = json!({"tools": [{"description": "no name"}, {"name": "real"}, 7]});
        let server = FakeServer::new(listed);
        let client = server.client("srv").expect("handshake");

        let tools = client.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "srv.real");
    }

    /// A server that lists nothing is a connected server with no tools, not an
    /// error — it may gain them on a later reconnect.
    #[test]
    fn a_server_with_no_tools_is_not_a_failure() {
        let server = FakeServer::new(json!({}));
        let client = server.client("empty").expect("handshake");
        assert!(client.tools().is_empty());
    }

    // -------------------------------------------------------------------- calls

    #[test]
    fn calling_a_tool_forwards_its_arguments() {
        let server = FakeServer::new(one_tool());
        let client = server.client("github").expect("handshake");

        let out = client.call_tool("create_issue", &json!({"title": "bug"}));
        assert!(!out.is_error, "{}", out.content);
        assert_eq!(out.content, "done");

        let calls = server.calls.lock().expect("calls");
        assert_eq!(calls[0], json!({"name": "create_issue", "arguments": {"title": "bug"}}));
    }

    #[test]
    fn a_tool_reporting_an_error_is_an_error_not_a_result() {
        let server = FakeServer::new(one_tool());
        let client = server.client("github").expect("handshake");

        let out = client.call_tool("explode", &json!({}));
        assert!(out.is_error);
        assert_eq!(out.content, "kaboom");
    }

    #[test]
    fn calling_a_closed_client_reports_rather_than_hangs() {
        let server = FakeServer::new(one_tool());
        let client = server.client("github").expect("handshake");
        client.close();

        let out = client.call_tool("create_issue", &json!({}));
        assert!(out.is_error);
        assert!(out.content.contains("not connected"), "{}", out.content);
    }

    /// A remote tool acts through its own process, so nothing local bounds it.
    #[test]
    fn every_remote_tool_is_consequential() {
        let server = FakeServer::new(one_tool());
        let client = server.client("github").expect("handshake");
        assert!(client.tools().iter().all(|t| t.consequential()));
    }

    #[tokio::test]
    async fn a_remote_tool_runs_through_the_ordinary_tool_trait() {
        let server = FakeServer::new(one_tool());
        let client = server.client("github").expect("handshake");
        let tools = client.tools();

        let dir = tempfile::TempDir::new().expect("tempdir");
        let ctx = ToolCtx::new(dir.path());
        let out = tools[0].run(&json!({"title": "bug"}), &ctx).await.expect("run");

        assert_eq!(out.content, "done");
        assert!(!out.is_error);
    }

    // ------------------------------------------------------------------ content

    #[test]
    fn text_blocks_are_flattened_in_order() {
        let content = json!([{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]);
        assert_eq!(render(Some(&content)), "one\ntwo");
    }

    /// An engine told `[image content]` can ask for something else; an engine
    /// told nothing assumes the call returned empty.
    #[test]
    fn a_non_text_block_is_named_rather_than_dropped() {
        let content = json!([{"type": "image", "data": "..."}]);
        assert_eq!(render(Some(&content)), "[image content]");
        assert_eq!(render(Some(&json!([{"data": "..."}]))), "[unknown content]");
    }

    #[test]
    fn a_resource_block_prefers_its_text_and_falls_back_to_its_uri() {
        let with_text = json!([{"type": "resource", "resource": {"text": "body", "uri": "u"}}]);
        assert_eq!(render(Some(&with_text)), "body");

        let bare = json!([{"type": "resource", "resource": {"uri": "file:///x"}}]);
        assert_eq!(render(Some(&bare)), "file:///x");
    }

    #[test]
    fn absent_and_empty_content_render_to_nothing() {
        assert_eq!(render(None), "");
        assert_eq!(render(Some(&Value::Null)), "");
        assert_eq!(render(Some(&json!([]))), "");
        assert_eq!(render(Some(&json!("plain string"))), "plain string");
    }

    // ------------------------------------------------------------- containment

    /// An editor that cannot open because a side-car is down is worse than one
    /// missing a feature.
    #[test]
    fn one_server_that_will_not_start_does_not_stop_the_others() {
        let (clients, errors) = connect_all(&[
            json!({"name": "ghost", "command": "definitely-not-a-real-binary-xyzzy"}),
            json!({"name": "broken"}),
        ]);

        assert!(clients.is_empty());
        assert_eq!(errors.len(), 2);
        assert!(errors[0].starts_with("ghost:"), "{:?}", errors);
        assert!(errors[1].contains("unusable mcpServers entry"), "{:?}", errors);
    }

    #[test]
    fn connecting_nothing_is_not_an_error() {
        let (clients, errors) = connect_all(&[]);
        assert!(clients.is_empty() && errors.is_empty());
    }

    /// Closing twice, and closing a dropped client, must both be safe — `Drop`
    /// calls `close` and callers reasonably call it themselves too.
    #[test]
    fn closing_is_idempotent() {
        let server = FakeServer::new(one_tool());
        let client = server.client("github").expect("handshake");
        client.close();
        client.close();
        assert!(client.handle.is_closed());
    }

    /// The lock exists so two calls cannot interleave on a server that does not
    /// expect it; this pins that it serialises rather than deadlocks.
    #[test]
    fn concurrent_calls_are_serialised_not_dropped() {
        let server = FakeServer::new(one_tool());
        let client = server.client("github").expect("handshake");
        let done = Arc::new(AtomicUsize::new(0));

        let workers: Vec<_> = (0..4)
            .map(|i| {
                let (client, done) = (Arc::clone(&client), Arc::clone(&done));
                std::thread::spawn(move || {
                    let out = client.call_tool("create_issue", &json!({"n": i}));
                    if !out.is_error {
                        done.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();
        for w in workers {
            w.join().expect("worker");
        }

        assert_eq!(done.load(Ordering::SeqCst), 4);
        assert_eq!(server.calls.lock().expect("calls").len(), 4);
    }
}
