//! The workspace routes of the Rust Field server end to end: the file tree,
//! file reads and writes, path containment, git state, and the operator
//! terminal streamed over the WebSocket hub.

use futures_util::StreamExt;
use knossos::field::{Running, ServerOptions};
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use std::path::Path;
use std::time::Duration;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

const BOOT: &str = "bootstrap-fixture-token";
const BROWSER: &str = "browser-fixture-token";

struct Fixture {
    dir: tempfile::TempDir,
    running: Running,
    client: Client,
    base: String,
}

impl Fixture {
    fn cookie(&self) -> String {
        format!("field_session={BROWSER}")
    }

    fn workspace(&self) -> std::path::PathBuf {
        self.dir.path().join("workspace")
    }

    async fn get(&self, path: &str) -> reqwest::Response {
        self.client
            .get(format!("{}{path}", self.base))
            .header("Cookie", self.cookie())
            .send()
            .await
            .unwrap()
    }

    async fn send(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Value,
        with_origin: bool,
    ) -> reqwest::Response {
        let mut req = self
            .client
            .request(method, format!("{}{path}", self.base))
            .header("Cookie", self.cookie())
            .header("Content-Type", "application/json")
            .body(body.to_string());
        if with_origin {
            req = req.header("Origin", self.base.clone());
        }
        req.send().await.unwrap()
    }

    async fn post(&self, path: &str, body: Value) -> reqwest::Response {
        self.send(reqwest::Method::POST, path, body, true).await
    }

    async fn put(&self, path: &str, body: Value, with_origin: bool) -> reqwest::Response {
        self.send(reqwest::Method::PUT, path, body, with_origin)
            .await
    }

    async fn socket(
        &self,
    ) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>
    {
        let ws_url = format!("ws://127.0.0.1:{}/ws", self.running.addr.port());
        let mut request = ws_url.into_client_request().unwrap();
        request
            .headers_mut()
            .insert("Origin", self.base.parse().unwrap());
        request
            .headers_mut()
            .insert("Cookie", self.cookie().parse().unwrap());
        let (socket, _) = tokio_tungstenite::connect_async(request).await.unwrap();
        socket
    }
}

async fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let field_dir = dir.path().join("field");
    let dist = dir.path().join("dist");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir_all(&field_dir).unwrap();
    std::fs::create_dir_all(&dist).unwrap();
    std::fs::create_dir_all(workspace.join("src")).unwrap();
    std::fs::create_dir_all(workspace.join("node_modules")).unwrap();
    std::fs::write(workspace.join("src").join("main.rs"), "fn main() {}\n").unwrap();
    std::fs::write(workspace.join("README.md"), "# Fixture\n").unwrap();
    std::fs::write(workspace.join(".env"), "SECRET=canary\n").unwrap();
    std::fs::write(workspace.join("ignored.txt"), "ignored\n").unwrap();
    std::fs::write(workspace.join(".gitignore"), "ignored.txt\n").unwrap();
    std::fs::write(workspace.join("node_modules").join("x.js"), "").unwrap();
    std::fs::write(
        field_dir.join("field.yaml"),
        "field:\n  name: Test Field\nworkspaces:\n  - id: here\n    name: Here\n    path: ../workspace\n  - id: ghost\n    name: Ghost\n    path: missing\nendpoints: []\nwebsites: []\n",
    )
    .unwrap();
    std::fs::write(dist.join("index.html"), "<!doctype html><h1>Field</h1>").unwrap();
    let running = knossos::field::server::start(ServerOptions {
        field_dir,
        state_dir: dir.path().join("state"),
        dist_dir: dist,
        port: Some(0),
        ui_origin: None,
        backend: None,
        bootstrap_token: Some(BOOT.into()),
        browser_token: Some(BROWSER.into()),
        probe_endpoints: false,
    })
    .await
    .unwrap();
    let base = format!("http://127.0.0.1:{}", running.addr.port());
    let client = Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let boot = client
        .get(format!("{base}/?bootstrap={BOOT}"))
        .send()
        .await
        .unwrap();
    assert_eq!(boot.status(), StatusCode::SEE_OTHER);
    Fixture {
        dir,
        running,
        client,
        base,
    }
}

fn git(cwd: &Path, args: &[&str]) -> bool {
    std::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

async fn next_text<S>(socket: &mut S) -> String
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match socket.next().await {
                Some(Ok(Message::Text(text))) => return text.to_string(),
                Some(Ok(_)) => continue,
                other => panic!("socket ended before a text frame: {other:?}"),
            }
        }
    })
    .await
    .expect("a text frame within ten seconds")
}

/// Terminal frames for `id`, up to and including its exit frame.
async fn terminal_frames<S>(socket: &mut S, id: &str) -> Vec<Value>
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    let mut frames = Vec::new();
    loop {
        let message: Value = serde_json::from_str(&next_text(socket).await).unwrap();
        if message["type"] != "terminal" || message["id"] != id {
            continue;
        }
        let done = message.get("exit").is_some();
        frames.push(message);
        if done {
            return frames;
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tree_file_write_and_containment() {
    let f = fixture().await;

    // The tree: directories first, skip lists and the security policy applied.
    let tree = f.get("/api/fs/tree?ws=here").await;
    assert_eq!(tree.status(), StatusCode::OK);
    let tree: Value = tree.json().await.unwrap();
    assert_eq!(tree["ws"], "here");
    assert_eq!(tree["dir"], "");
    let names: Vec<&str> = tree["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["src", ".gitignore", "README.md"], "{tree}");
    assert_eq!(tree["entries"][0]["dir"], true);
    assert_eq!(tree["entries"][0]["size"], Value::Null);
    assert_eq!(tree["entries"][2]["size"], 10);
    let nested: Value = f
        .get("/api/fs/tree?ws=here&dir=src")
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(nested["entries"][0]["path"], "src/main.rs");

    // An unmounted or unknown workspace is a plain 400.
    let ghost = f.get("/api/fs/tree?ws=ghost").await;
    assert_eq!(ghost.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        ghost.json::<Value>().await.unwrap(),
        json!({ "error": "workspace ghost is not mounted", "code": "bad_request" })
    );
    let unknown: Value = f.get("/api/fs/tree?ws=nope").await.json().await.unwrap();
    assert_eq!(unknown["error"], "unknown workspace: nope");

    // Read.
    let file = f.get("/api/fs/file?ws=here&path=src/main.rs").await;
    assert_eq!(file.status(), StatusCode::OK);
    assert_eq!(
        file.json::<Value>().await.unwrap(),
        json!({ "ws": "here", "path": "src/main.rs", "size": 13, "tooLarge": false, "content": "fn main() {}\n" })
    );

    // Write needs the same-origin gate every unsafe method gets.
    let csrf = f
        .put(
            "/api/fs/file",
            json!({ "ws": "here", "path": "src/new.rs", "content": "x" }),
            false,
        )
        .await;
    assert_eq!(csrf.status(), StatusCode::FORBIDDEN);
    let put = f
        .put(
            "/api/fs/file",
            json!({ "ws": "here", "path": "src/new.rs", "content": "pub fn x() {}\n" }),
            true,
        )
        .await;
    assert_eq!(put.status(), StatusCode::OK);
    assert_eq!(
        put.json::<Value>().await.unwrap(),
        json!({ "ok": true, "bytes": 14 })
    );
    assert_eq!(
        std::fs::read_to_string(f.workspace().join("src").join("new.rs")).unwrap(),
        "pub fn x() {}\n"
    );
    let back: Value = f
        .get("/api/fs/file?ws=here&path=src/new.rs")
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(back["content"], "pub fn x() {}\n");

    // Containment: climbing out, secrets, and ignored files are all refused.
    let escape = f.get("/api/fs/file?ws=here&path=../field/field.yaml").await;
    assert_eq!(escape.status(), StatusCode::BAD_REQUEST);
    let escape: Value = escape.json().await.unwrap();
    assert_eq!(escape["code"], "bad_request");
    assert_eq!(escape["error"], "path escapes the workspace root");
    let put_escape = f
        .put(
            "/api/fs/file",
            json!({ "ws": "here", "path": "../escape.txt", "content": "x" }),
            true,
        )
        .await;
    assert_eq!(put_escape.status(), StatusCode::BAD_REQUEST);
    assert!(!f.dir.path().join("escape.txt").exists());
    let hidden: Value = f
        .get("/api/fs/file?ws=here&path=.env")
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(
        hidden["error"],
        "path is hidden by the workspace security policy"
    );
    let ignored: Value = f
        .get("/api/fs/file?ws=here&path=ignored.txt")
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(
        ignored["error"],
        "path is hidden by the workspace security policy"
    );
    let dir_as_file: Value = f
        .get("/api/fs/file?ws=here&path=src")
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(
        dir_as_file["error"],
        "workspace target is not a regular file"
    );

    f.running.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn git_status_log_and_diff() {
    let f = fixture().await;
    let ws = f.workspace();

    // Not a repository yet: status fails the way execFile did, log is empty.
    let status = f.get("/api/git/status?ws=here").await;
    if status.status() == StatusCode::BAD_REQUEST {
        let error: Value = status.json().await.unwrap();
        assert_eq!(error["code"], "bad_request");
        assert!(
            error["error"]
                .as_str()
                .unwrap()
                .starts_with("Command failed: git ")
                || error["error"].as_str().unwrap().starts_with("spawn git "),
            "{error}"
        );
    } else {
        // The temp dir sits inside a larger repository on this machine.
        assert_eq!(status.status(), StatusCode::OK);
    }
    let log: Value = f.get("/api/git/log?ws=here").await.json().await.unwrap();
    assert_eq!(log["ws"], "here");
    assert!(log["commits"].is_array());
    let missing = f.get("/api/git/diff?ws=here&path=nope.txt").await;
    assert_eq!(missing.status(), StatusCode::BAD_REQUEST);

    let ready = git(&ws, &["init", "-q"])
        && git(&ws, &["config", "user.email", "field@example.invalid"])
        && git(&ws, &["config", "user.name", "Field"])
        && git(&ws, &["config", "commit.gpgsign", "false"])
        && git(&ws, &["add", "."])
        && git(&ws, &["commit", "-q", "-m", "initial import"]);
    if !ready {
        eprintln!("git is not usable here; skipping the repository half");
        f.running.stop().await;
        return;
    }
    std::fs::write(ws.join("src").join("fresh.rs"), "// new\n").unwrap();
    std::fs::write(ws.join("README.md"), "# Fixture\n\nchanged\n").unwrap();

    let status: Value = f.get("/api/git/status?ws=here").await.json().await.unwrap();
    assert_eq!(status["ws"], "here");
    assert!(status["branch"].is_string(), "{status}");
    assert_eq!(status["ahead"], 0);
    assert_eq!(status["behind"], 0);
    let files = status["files"].as_array().unwrap();
    assert!(
        files.contains(&json!({ "path": "README.md", "status": "modified", "staged": false })),
        "{status}"
    );
    assert!(
        files.contains(&json!({ "path": "src/fresh.rs", "status": "untracked", "staged": false })),
        "{status}"
    );

    let log: Value = f.get("/api/git/log?ws=here").await.json().await.unwrap();
    assert_eq!(log["commits"].as_array().unwrap().len(), 1);
    assert_eq!(log["commits"][0]["subject"], "initial import");
    assert_eq!(log["commits"][0]["author"], "Field");
    assert!(log["commits"][0]["hash"].as_str().unwrap().len() >= 7);

    let diff: Value = f
        .get("/api/git/diff?ws=here&path=README.md")
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(diff["ws"], "here");
    assert_eq!(diff["path"], "README.md");
    assert!(
        diff["diff"].as_str().unwrap().contains("+changed"),
        "{diff}"
    );
    let fresh: Value = f
        .get("/api/git/diff?ws=here&path=src/fresh.rs")
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(fresh["diff"], "(untracked — no diff yet)");
    let same: Value = f
        .get("/api/git/diff?ws=here&path=src/main.rs")
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(same["diff"], "(no changes)");

    f.running.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn terminal_runs_streams_logs_and_kills() {
    let f = fixture().await;
    let mut socket = f.socket().await;
    let first: Value = serde_json::from_str(&next_text(&mut socket).await).unwrap();
    assert_eq!(first["type"], "snapshot");

    // Validation before anything is spawned.
    let empty = f
        .post(
            "/api/terminal/run",
            json!({ "ws": "here", "command": "  " }),
        )
        .await;
    assert_eq!(empty.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        empty.json::<Value>().await.unwrap(),
        json!({ "error": "terminal command is required", "code": "bad_request" })
    );
    let ghost = f
        .post(
            "/api/terminal/run",
            json!({ "ws": "ghost", "command": "echo hi" }),
        )
        .await;
    assert_eq!(ghost.status(), StatusCode::BAD_REQUEST);

    // A short command: header, output, exit, in that order, then the event.
    let run = f
        .post(
            "/api/terminal/run",
            json!({ "ws": "here", "command": "echo hi" }),
        )
        .await;
    assert_eq!(run.status(), StatusCode::OK);
    let run: Value = run.json().await.unwrap();
    let id = run["terminalId"].as_str().unwrap().to_string();
    assert!(!id.is_empty());
    let frames = terminal_frames(&mut socket, &id).await;
    assert_eq!(frames[0]["stream"], "meta");
    assert_eq!(frames[0]["data"], "[here:.] $ echo hi\n");
    let output: String = frames
        .iter()
        .filter(|fr| fr["stream"] == "out")
        .map(|fr| fr["data"].as_str().unwrap())
        .collect();
    assert!(output.contains("hi"), "{frames:?}");
    let exit = frames.last().unwrap();
    assert_eq!(exit["stream"], "meta");
    assert_eq!(exit["exit"], 0);
    assert_eq!(exit["data"], "\n[exit 0]\n");
    assert!(!f.running.state.terminals.contains(&id));

    let events: Value = f.get("/api/events").await.json().await.unwrap();
    let logged = events["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["kind"] == "terminal.run")
        .expect("terminal.run event");
    assert_eq!(
        logged["data"],
        json!({ "workspaceId": "here", "command": "echo hi", "terminalId": id })
    );

    // The command is logged redacted.
    let secret = f
        .post("/api/terminal/run", json!({ "ws": "here", "command": "echo TOKEN=hunter2secret", "terminalId": "redacted" }))
        .await;
    assert_eq!(secret.status(), StatusCode::OK);
    let frames = terminal_frames(&mut socket, "redacted").await;
    assert_eq!(frames[0]["data"], "[here:.] $ echo TOKEN=[REDACTED]\n");
    let events: Value = f.get("/api/events").await.json().await.unwrap();
    let logged = events["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["kind"] == "terminal.run" && e["data"]["terminalId"] == "redacted")
        .expect("redacted terminal.run event");
    assert_eq!(logged["data"]["command"], "echo TOKEN=[REDACTED]");

    // A long-running one: the id is reserved while it runs, and kill ends it.
    let sleep = if cfg!(windows) {
        "Start-Sleep -Seconds 30"
    } else {
        "sleep 30"
    };
    let long = f
        .post(
            "/api/terminal/run",
            json!({ "ws": "here", "cwd": "src", "command": sleep, "terminalId": "long" }),
        )
        .await;
    assert_eq!(long.status(), StatusCode::OK);
    assert_eq!(
        long.json::<Value>().await.unwrap(),
        json!({ "terminalId": "long" })
    );
    assert!(f.running.state.terminals.contains("long"));
    let duplicate = f
        .post(
            "/api/terminal/run",
            json!({ "ws": "here", "command": sleep, "terminalId": "long" }),
        )
        .await;
    assert_eq!(duplicate.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        duplicate.json::<Value>().await.unwrap(),
        json!({ "error": "terminal id is already active", "code": "terminal_exists" })
    );
    let killed = f
        .post("/api/terminal/kill", json!({ "terminalId": "long" }))
        .await;
    assert_eq!(killed.status(), StatusCode::OK);
    assert_eq!(killed.json::<Value>().await.unwrap(), json!({ "ok": true }));
    let frames = terminal_frames(&mut socket, "long").await;
    assert_eq!(frames[0]["data"], format!("[here:src] $ {sleep}\n"));
    let exit = frames.last().unwrap();
    assert!(
        exit["data"].as_str().unwrap().starts_with("\n[exit "),
        "{exit}"
    );
    assert_ne!(
        exit["exit"], 0,
        "a killed shell does not exit cleanly: {exit}"
    );
    assert!(!f.running.state.terminals.contains("long"));
    let gone = f
        .post("/api/terminal/kill", json!({ "terminalId": "long" }))
        .await;
    assert_eq!(gone.json::<Value>().await.unwrap(), json!({ "ok": false }));

    let _ = socket.close(None).await;
    f.running.stop().await;
}
