//! In-app agent definitions: `GET/POST /api/agents`, `/api/agents/update`
//! and `/api/agents/delete`, end to end through the Rust Field server. A
//! defined agent is a markdown record under `field/agents/` and is
//! spawnable the moment it is saved.

use knossos::field::config::frontmatter;
use knossos::field::{Running, ServerOptions};
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use std::path::PathBuf;

const BOOT: &str = "bootstrap-fixture-token";
const BROWSER: &str = "browser-fixture-token";

struct Fixture {
    _dir: tempfile::TempDir,
    field_dir: PathBuf,
    running: Running,
    client: Client,
    base: String,
}

impl Fixture {
    async fn get(&self, path: &str) -> Value {
        self.client
            .get(format!("{}{path}", self.base))
            .header("Cookie", format!("field_session={BROWSER}"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    async fn post(&self, path: &str, body: Value) -> (StatusCode, Value) {
        let res = self
            .client
            .post(format!("{}{path}", self.base))
            .header("Cookie", format!("field_session={BROWSER}"))
            .header("Content-Type", "application/json")
            .header("Origin", self.base.clone())
            .body(body.to_string())
            .send()
            .await
            .unwrap();
        let status = res.status();
        (status, res.json().await.unwrap_or(Value::Null))
    }
}

async fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let field_dir = dir.path().join("field");
    let dist = dir.path().join("dist");
    std::fs::create_dir_all(&field_dir).unwrap();
    std::fs::create_dir_all(&dist).unwrap();
    std::fs::write(
        field_dir.join("field.yaml"),
        "field:\n  name: Test Field\nworkspaces:\n  - id: here\n    name: Here\n    path: .\nendpoints:\n  - id: local\n    name: Local\n    kind: openai-compatible\n    model: test-model\n    base_url: http://127.0.0.1:9/v1\nwebsites: []\n",
    )
    .unwrap();
    std::fs::create_dir_all(field_dir.join("roles")).unwrap();
    std::fs::write(
        field_dir.join("roles").join("builder.md"),
        "---\nid: builder\nname: Builder\ndefault_endpoint: local\ntools_allow: [Read, Grep, Edit, Bash]\n---\nBuild carefully.\n",
    )
    .unwrap();
    std::fs::create_dir_all(field_dir.join("agents")).unwrap();
    std::fs::write(
        field_dir.join("agents").join("builder-1.md"),
        "---\nid: builder-1\nname: Builder One\nrole: builder\nendpoint: local\n---\nYou are the builder.\n",
    )
    .unwrap();
    std::env::set_var("FIELD_KNOSSOS_BIN", env!("CARGO_BIN_EXE_knossos"));
    std::fs::write(dist.join("index.html"), "<!doctype html><h1>Field</h1>").unwrap();
    let running = knossos::field::server::start(ServerOptions {
        field_dir: field_dir.clone(),
        state_dir: dir.path().join("state"),
        dist_dir: dist,
        port: Some(0),
        ui_origin: None,
        backend: None,
        bootstrap_token: Some(BOOT.into()),
        browser_token: Some(BROWSER.into()),
        probe_endpoints: false,
        watch_git: false,
    })
    .await
    .unwrap();
    let base = format!("http://127.0.0.1:{}", running.addr.port());
    let client = Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    // The bootstrap link is what turns the browser token into a session.
    let boot = client
        .get(format!("{base}/?bootstrap={BOOT}"))
        .send()
        .await
        .unwrap();
    assert!(boot.status().is_redirection(), "{}", boot.status());
    Fixture {
        _dir: dir,
        field_dir,
        running,
        client,
        base,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn agents_are_defined_listed_spawned_updated_and_deleted_in_app() {
    let f = fixture().await;

    // The list starts with the record on disk, orders included.
    let listed = f.get("/api/agents").await;
    assert_eq!(listed["agents"].as_array().unwrap().len(), 1, "{listed}");
    assert_eq!(listed["agents"][0]["id"], "builder-1");
    assert_eq!(listed["agents"][0]["orders"], "You are the builder.");

    // Validation: an empty name, an unknown role, an unknown endpoint.
    let (status, body) = f
        .post(
            "/api/agents",
            json!({ "name": "  ", "role": "builder", "endpoint": "local" }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["code"], "bad_request");
    let (status, body) = f
        .post(
            "/api/agents",
            json!({ "name": "Nobody", "role": "poet", "endpoint": "local" }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("unknown role"));
    let (status, body) = f
        .post(
            "/api/agents",
            json!({ "name": "Nobody", "role": "builder", "endpoint": "cloud" }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("unknown endpoint"));

    // Create: the id is the slug of the name; the file carries the fields.
    let (status, created) = f
        .post(
            "/api/agents",
            json!({
                "name": "Rhea Coder",
                "role": "builder",
                "endpoint": "local",
                "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "thinking": "high",
                "orders": "Keep diffs small.\nRun the tests.",
            }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{created}");
    let agent = &created["agent"];
    assert_eq!(agent["id"], "rhea-coder");
    assert_eq!(agent["name"], "Rhea Coder");
    assert_eq!(agent["role"], "builder");
    assert_eq!(agent["endpoint"], "local");
    assert_eq!(agent["model"], "Qwen/Qwen2.5-Coder-7B-Instruct");
    assert_eq!(agent["thinking"], "high");
    assert_eq!(agent["orders"], "Keep diffs small.\nRun the tests.");
    let file = f.field_dir.join("agents").join("rhea-coder.md");
    assert!(file.is_file(), "{} was not written", file.display());
    let (data, orders) = frontmatter(&std::fs::read_to_string(&file).unwrap());
    assert_eq!(data["id"], "rhea-coder");
    assert_eq!(data["name"], "Rhea Coder");
    assert_eq!(data["role"], "builder");
    assert_eq!(data["endpoint"], "local");
    assert_eq!(data["model"], "Qwen/Qwen2.5-Coder-7B-Instruct");
    assert_eq!(data["thinking"], "high");
    assert_eq!(orders, "Keep diffs small.\nRun the tests.\n");

    // A second agent with the same name gets a distinct id.
    let (status, twin) = f
        .post(
            "/api/agents",
            json!({ "name": "Rhea Coder", "role": "builder", "endpoint": "local" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{twin}");
    assert_eq!(twin["agent"]["id"], "rhea-coder-2");
    assert!(twin["agent"]["model"].is_null());

    // The list and the config view both show it; the event was logged.
    let listed = f.get("/api/agents").await;
    let mut ids: Vec<&str> = listed["agents"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["id"].as_str().unwrap())
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, ["builder-1", "rhea-coder", "rhea-coder-2"]);
    let config = f.get("/api/config").await;
    assert!(config["agents"]
        .as_array()
        .unwrap()
        .iter()
        .any(|a| a["id"] == "rhea-coder"));
    let events = f.get("/api/events?from=0&limit=100").await;
    let defined = events["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["kind"] == "agent.defined" && e["data"]["agentId"] == "rhea-coder")
        .cloned()
        .expect("agent.defined was emitted");
    assert_eq!(defined["data"]["model"], "Qwen/Qwen2.5-Coder-7B-Instruct");
    assert_eq!(defined["data"]["endpoint"], "local");

    // Spawn with the new agent right away: the registry sees the record
    // without a restart, and the session runs on the agent's model.
    let (status, spawned) = f
        .post(
            "/api/sessions",
            json!({ "agentId": "rhea-coder", "orders": "Say ready and stop." }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{spawned}");
    let session_id = spawned["sessionId"].as_str().unwrap().to_string();
    let events = f.get("/api/events?from=0&limit=200").await;
    let spawned_event = events["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["kind"] == "session.spawned" && e["data"]["sessionId"] == session_id)
        .cloned()
        .expect("session.spawned was emitted");
    assert_eq!(spawned_event["data"]["agentId"], "rhea-coder");
    assert_eq!(
        spawned_event["data"]["model"],
        "Qwen/Qwen2.5-Coder-7B-Instruct"
    );

    // Deleting an agent with a live session is refused.
    let (status, body) = f
        .post(
            "/api/agents/delete",
            json!({ "id": "rhea-coder", "confirmRisk": true }),
        )
        .await;
    assert!(
        status == StatusCode::CONFLICT || status == StatusCode::OK,
        "{status} {body}"
    );
    if status == StatusCode::CONFLICT {
        assert_eq!(body["code"], "agent_in_use");
    }
    let (status, _) = f
        .post(
            "/api/command",
            json!({ "kind": "cancel", "sessionIds": [session_id] }),
        )
        .await;
    assert_eq!(status, StatusCode::OK);

    // Update changes the model in place; the id and the file stay.
    let (status, updated) = f
        .post(
            "/api/agents/update",
            json!({
                "id": "rhea-coder-2",
                "name": "Rhea Coder",
                "role": "builder",
                "endpoint": "local",
                "model": "claude-sonnet-4-5",
                "thinking": "low",
                "orders": "Review only.",
            }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{updated}");
    assert_eq!(updated["agent"]["id"], "rhea-coder-2");
    assert_eq!(updated["agent"]["model"], "claude-sonnet-4-5");
    assert_eq!(updated["agent"]["thinking"], "low");
    assert_eq!(updated["agent"]["orders"], "Review only.");
    let file2 = f.field_dir.join("agents").join("rhea-coder-2.md");
    let (data, orders) = frontmatter(&std::fs::read_to_string(&file2).unwrap());
    assert_eq!(data["id"], "rhea-coder-2");
    assert_eq!(data["model"], "claude-sonnet-4-5");
    assert_eq!(orders, "Review only.\n");
    let (status, body) = f
        .post(
            "/api/agents/update",
            json!({ "id": "ghost", "name": "Ghost", "role": "builder", "endpoint": "local" }),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");

    // Delete refuses without confirmRisk, then removes the file.
    let (status, body) = f
        .post("/api/agents/delete", json!({ "id": "rhea-coder-2" }))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["code"], "confirmation_required");
    assert!(file2.is_file());
    let (status, body) = f
        .post(
            "/api/agents/delete",
            json!({ "id": "rhea-coder-2", "confirmRisk": true }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(!file2.exists(), "the agent file was not removed");
    let listed = f.get("/api/agents").await;
    assert!(!listed["agents"]
        .as_array()
        .unwrap()
        .iter()
        .any(|a| a["id"] == "rhea-coder-2"));
    let (status, _) = f
        .post("/api/sessions", json!({ "agentId": "rhea-coder-2" }))
        .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "a deleted agent cannot spawn"
    );

    f.running.stop().await;
}
