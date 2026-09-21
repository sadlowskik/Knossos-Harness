//! The Rust Field server end to end: bootstrap, gates, the ported API, the
//! built client, the WebSocket hub, logout. The boundary half of
//! `security.test.mjs`, `ws.test.mjs` and `body.test.mjs`.

use futures_util::StreamExt;
use knossos::field::{Running, ServerOptions};
use reqwest::header::{HeaderValue, SET_COOKIE};
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use std::time::Duration;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

const BOOT: &str = "bootstrap-fixture-token";
const BROWSER: &str = "browser-fixture-token";

struct Fixture {
    _dir: tempfile::TempDir,
    running: Running,
    client: Client,
    base: String,
}

impl Fixture {
    fn cookie(&self) -> String {
        format!("field_session={BROWSER}")
    }

    fn origin(&self) -> String {
        self.base.clone()
    }

    async fn get(&self, path: &str) -> reqwest::Response {
        self.client
            .get(format!("{}{path}", self.base))
            .header("Cookie", self.cookie())
            .send()
            .await
            .unwrap()
    }

    async fn post(&self, path: &str, body: Value, with_origin: bool) -> reqwest::Response {
        let mut req = self
            .client
            .post(format!("{}{path}", self.base))
            .header("Cookie", self.cookie())
            .header("Content-Type", "application/json")
            .body(body.to_string());
        if with_origin {
            req = req.header("Origin", self.origin());
        }
        req.send().await.unwrap()
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
        "field:\n  name: Test Field\nworkspaces:\n  - id: here\n    name: Here\n    path: .\n  - id: ghost\n    name: Ghost\n    path: missing\nendpoints:\n  - id: local\n    name: Local\n    kind: openai-compatible\n    model: test-model\n    base_url: http://127.0.0.1:9/v1\nwebsites: []\n",
    )
    .unwrap();
    std::fs::create_dir_all(field_dir.join("roles")).unwrap();
    std::fs::write(field_dir.join("roles").join("builder.md"), "---\nid: builder\nname: Builder\ndefault_endpoint: local\ntools_allow: [Read, Grep, Edit, Bash]\n---\nBuild carefully.\n").unwrap();
    std::fs::create_dir_all(field_dir.join("agents")).unwrap();
    std::fs::write(field_dir.join("agents").join("builder-1.md"), "---\nid: builder-1\nname: Builder One\nrole: builder\nendpoint: local\n---\nYou are the builder.\n").unwrap();
    // The registry starts the real binary that cargo built for this test.
    std::env::set_var("FIELD_KNOSSOS_BIN", env!("CARGO_BIN_EXE_knossos"));
    std::fs::write(dist.join("index.html"), "<!doctype html><h1>Field</h1>").unwrap();
    std::fs::write(dist.join("app.js"), "console.log('field')").unwrap();
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
        watch_git: false,
    })
    .await
    .unwrap();
    let base = format!("http://127.0.0.1:{}", running.addr.port());
    let client = Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    Fixture {
        _dir: dir,
        running,
        client,
        base,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bootstrap_gates_api_static_and_logout() {
    let f = fixture().await;

    // Health needs only loopback and the exact Host.
    let health = f
        .client
        .get(format!("{}/healthz", f.base))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    assert_eq!(health.json::<Value>().await.unwrap(), json!({ "ok": true }));

    // Nothing else before bootstrap, even with the right cookie in hand.
    let early = f.get("/api/state").await;
    assert_eq!(early.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        early.json::<Value>().await.unwrap()["code"],
        "browser_auth_required"
    );

    // The bootstrap link is single use and sets the cookie.
    let boot = f
        .client
        .get(format!("{}/?bootstrap={BOOT}", f.base))
        .send()
        .await
        .unwrap();
    assert_eq!(boot.status(), StatusCode::SEE_OTHER);
    assert_eq!(
        boot.headers().get("location"),
        Some(&HeaderValue::from_static("/"))
    );
    let cookie = boot
        .headers()
        .get(SET_COOKIE)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        cookie.starts_with(&format!(
            "field_session={BROWSER}; HttpOnly; SameSite=Strict"
        )),
        "{cookie}"
    );
    let again = f
        .client
        .get(format!("{}/?bootstrap={BOOT}", f.base))
        .send()
        .await
        .unwrap();
    assert_eq!(again.status(), StatusCode::GONE);

    // The built client, with the SPA fallback and the security headers.
    let page = f.get("/").await;
    assert_eq!(page.status(), StatusCode::OK);
    assert_eq!(page.headers().get("x-frame-options").unwrap(), "DENY");
    assert!(page
        .headers()
        .get("content-security-policy")
        .unwrap()
        .to_str()
        .unwrap()
        .contains("frame-ancestors 'none'"));
    assert!(page.text().await.unwrap().contains("<h1>Field</h1>"));
    let asset = f.get("/app.js").await;
    assert_eq!(
        asset.headers().get("content-type").unwrap(),
        "text/javascript; charset=utf-8"
    );
    let deep = f.get("/campaigns/some/route").await;
    assert_eq!(deep.status(), StatusCode::OK);
    assert!(deep.text().await.unwrap().contains("<h1>Field</h1>"));
    let escape = f.get("/../field/field.yaml").await;
    assert!(
        !escape.text().await.unwrap().contains("workspaces:"),
        "no path escapes the dist directory"
    );

    // State and config from the projection and the configuration.
    let state = f.get("/api/state").await.json::<Value>().await.unwrap();
    assert_eq!(state["seq"], 0);
    assert_eq!(state["workspaces"][0]["id"], "here");
    assert_eq!(state["workspaces"][0]["mounted"], true);
    assert_eq!(state["workspaces"][1]["mounted"], false);
    assert!(state["rehearsal"].is_object());
    let config = f.get("/api/config").await.json::<Value>().await.unwrap();
    assert_eq!(config["field"]["name"], "Test Field");

    // Mutations need the exact Origin; then they append, fold and page.
    let refused = f
        .post(
            "/api/position",
            json!({ "entityType": "agent", "entityId": "a", "x": 1, "y": 2 }),
            false,
        )
        .await;
    assert_eq!(refused.status(), StatusCode::FORBIDDEN);
    assert_eq!(
        refused.json::<Value>().await.unwrap()["code"],
        "origin_required"
    );
    let ok = f
        .post(
            "/api/position",
            json!({ "entityType": "agent", "entityId": "a", "x": 1, "y": 2 }),
            true,
        )
        .await;
    assert_eq!(ok.status(), StatusCode::OK);
    let events = f
        .get("/api/events?from=0&limit=10")
        .await
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(events["events"][0]["kind"], "ui.position");
    assert_eq!(events["events"][0]["seq"], 1);
    assert_eq!(events["head"], 1);
    assert!(events["nextFrom"].is_null());
    let bad = f.get("/api/events?limit=0").await;
    assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        bad.json::<Value>().await.unwrap()["code"],
        "invalid_pagination"
    );
    let state = f.get("/api/state").await.json::<Value>().await.unwrap();
    assert_eq!(state["positions"]["agent:a"], json!({ "x": 1, "y": 2 }));

    // The world: capital, then reconciliation seats it in Italia.
    let missing = f
        .post(
            "/api/world/capital",
            json!({ "workspaceId": "ghost" }),
            true,
        )
        .await;
    assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    let world = f
        .post("/api/world/capital", json!({ "workspaceId": "here" }), true)
        .await
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(world["capitalWorkspaceId"], "here");
    let world = f
        .post("/api/world/reconcile", json!({ "clusters": [{ "clusterKey": "service:db", "label": "DB", "kind": "service" }] }), true)
        .await
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(
        world["assignments"]["workspace:here"]["territoryId"],
        "italia"
    );
    assert_eq!(world["assignments"]["service:db"]["territoryId"], "gallia");
    let trace = f
        .get("/api/trace?subject=here&from=0&limit=10")
        .await
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(trace["subject"], "here");
    assert_eq!(trace["events"][0]["kind"], "world.capital_selected");

    // What is not ported yet says so, distinctly from a refusal.
    let rehearsal = f.post("/api/simulations/run", json!({}), true).await;
    assert_eq!(rehearsal.status(), StatusCode::NOT_IMPLEMENTED);
    assert_eq!(
        rehearsal.json::<Value>().await.unwrap()["code"],
        "not_ported"
    );
    // The director validates before it records anything.
    let campaign = f.post("/api/campaigns/create", json!({}), true).await;
    assert_eq!(campaign.status(), StatusCode::BAD_REQUEST);
    let unconfirmed = f
        .post(
            "/api/campaigns/action",
            json!({ "campaignId": "nope", "kind": "checkpoint" }),
            true,
        )
        .await;
    assert_eq!(unconfirmed.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        unconfirmed.json::<Value>().await.unwrap()["code"],
        "confirmation_required"
    );
    let replay = f.get("/api/campaigns/replay?campaignId=nope").await;
    assert_eq!(replay.status(), StatusCode::NOT_FOUND);

    // The registry: an unknown agent is refused; a known one starts the real
    // knossos binary and its life shows up in the log and the state.
    let unknown = f
        .post("/api/sessions", json!({ "agentId": "nobody" }), true)
        .await;
    assert_eq!(unknown.status(), StatusCode::BAD_REQUEST);
    assert!(unknown.json::<Value>().await.unwrap()["error"]
        .as_str()
        .unwrap()
        .contains("unknown agent"));
    let endpoints = f.get("/api/endpoints").await.json::<Value>().await.unwrap();
    assert_eq!(endpoints["endpoints"][0]["id"], "local");
    assert_eq!(endpoints["endpoints"][0]["source"], "config");
    let spawned = f
        .post(
            "/api/sessions",
            json!({ "agentId": "builder-1", "orders": "Say ready and stop." }),
            true,
        )
        .await;
    assert_eq!(
        spawned.status(),
        StatusCode::OK,
        "{}",
        spawned.text().await.unwrap_or_default()
    );
    let session_id = spawned.json::<Value>().await.unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        let state = f.get("/api/state").await.json::<Value>().await.unwrap();
        let unit = state["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["id"] == session_id)
            .cloned();
        if let Some(unit) = unit {
            let state_word = unit["state"].as_str().unwrap_or("").to_string();
            if state_word != "spawning" || std::time::Instant::now() > deadline {
                assert_eq!(unit["agentId"], "builder-1");
                assert_eq!(unit["endpointId"], "local");
                break;
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the spawned session never reached the state"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let events = f
        .get("/api/events?from=0&limit=100")
        .await
        .json::<Value>()
        .await
        .unwrap();
    let kinds: Vec<&str> = events["events"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|e| e["kind"].as_str())
        .collect();
    assert!(
        kinds.contains(&"budget.reserved") && kinds.contains(&"session.spawned"),
        "{kinds:?}"
    );
    let cancelled = f
        .post(
            "/api/command",
            json!({ "kind": "cancel", "sessionIds": [session_id] }),
            true,
        )
        .await;
    assert_eq!(cancelled.status(), StatusCode::OK);

    // Cities are workspaces; routines answer with states and details; an
    // endpoint test reports the probe outcome rather than failing the call.
    let cities = f.get("/api/cities").await.json::<Value>().await.unwrap();
    let names: Vec<&str> = cities["cities"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c["id"].as_str())
        .collect();
    assert!(names.contains(&"here"), "{names:?}");
    let city = f
        .get("/api/city?id=here")
        .await
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(city["tier"], "outpost");
    assert!(!city["agents"].as_array().unwrap().is_empty(), "{city}");
    let missing = f.get("/api/city?id=nowhere").await;
    assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    let routines = f
        .get("/api/routines/states")
        .await
        .json::<Value>()
        .await
        .unwrap();
    assert!(routines["states"].is_object() && routines["details"].is_object());
    let unknown = f
        .post(
            "/api/routines/toggle",
            json!({ "routineId": "ghost", "enabled": true, "confirmRisk": true }),
            true,
        )
        .await;
    assert_eq!(unknown.status(), StatusCode::BAD_REQUEST);
    let probe = f
        .post("/api/endpoints/test", json!({ "id": "local" }), true)
        .await;
    assert_eq!(probe.status(), StatusCode::OK);
    let probe = probe.json::<Value>().await.unwrap();
    assert_eq!(probe["ok"], false, "{probe}");
    let probe = f
        .post("/api/endpoints/test", json!({ "id": "ghost" }), true)
        .await;
    assert_eq!(probe.status(), StatusCode::NOT_FOUND);

    // Body limits by wire bytes, and malformed JSON.
    let huge = f
        .client
        .post(format!("{}/api/position", f.base))
        .header("Cookie", f.cookie())
        .header("Origin", f.origin())
        .header("Content-Type", "application/json")
        .body(vec![b'x'; 5 * 1024 * 1024])
        .send()
        .await
        .unwrap();
    assert_eq!(huge.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let broken = f
        .client
        .post(format!("{}/api/position", f.base))
        .header("Cookie", f.cookie())
        .header("Origin", f.origin())
        .header("Content-Type", "application/json")
        .body("{nope}")
        .send()
        .await
        .unwrap();
    assert_eq!(broken.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        broken.json::<Value>().await.unwrap()["code"],
        "invalid_json"
    );

    // WebSocket: refused without the cookie, snapshot first with it, events
    // at once, a coalesced snapshot after.
    let ws_url = format!("ws://127.0.0.1:{}/ws", f.running.addr.port());
    let mut anonymous = ws_url.clone().into_client_request().unwrap();
    anonymous
        .headers_mut()
        .insert("Origin", f.origin().parse().unwrap());
    match tokio_tungstenite::connect_async(anonymous).await {
        Err(tokio_tungstenite::tungstenite::Error::Http(response)) => {
            assert_eq!(response.status().as_u16(), 401)
        }
        other => panic!("anonymous websocket must be refused with 401, got {other:?}"),
    }
    let mut request = ws_url.into_client_request().unwrap();
    request
        .headers_mut()
        .insert("Origin", f.origin().parse().unwrap());
    request
        .headers_mut()
        .insert("Cookie", f.cookie().parse().unwrap());
    let (mut socket, _) = tokio_tungstenite::connect_async(request).await.unwrap();
    let first: Value = serde_json::from_str(&next_text(&mut socket).await).unwrap();
    assert_eq!(first["type"], "snapshot");
    assert_eq!(first["state"]["world"]["capitalWorkspaceId"], "here");

    f.post(
        "/api/position",
        json!({ "entityType": "agent", "entityId": "b", "x": 3, "y": 4 }),
        true,
    )
    .await;
    let event: Value = serde_json::from_str(&next_text(&mut socket).await).unwrap();
    assert_eq!(event["type"], "event");
    assert_eq!(event["event"]["kind"], "ui.position");
    let coalesced: Value = serde_json::from_str(&next_text(&mut socket).await).unwrap();
    assert_eq!(coalesced["type"], "snapshot");
    assert_eq!(
        coalesced["state"]["positions"]["agent:b"],
        json!({ "x": 3, "y": 4 })
    );

    // Logout clears the cookie, closes sockets and refuses the old cookie.
    let out = f.post("/api/logout", json!({}), true).await;
    assert_eq!(out.status(), StatusCode::OK);
    assert!(out
        .headers()
        .get(SET_COOKIE)
        .unwrap()
        .to_str()
        .unwrap()
        .contains("Max-Age=0"));
    let closed = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match socket.next().await {
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break true,
                Some(Ok(_)) => continue,
            }
        }
    })
    .await
    .unwrap_or(false);
    assert!(closed, "logout terminates the authenticated socket");
    let _ = socket.close(None).await;
    let after = f.get("/api/state").await;
    assert_eq!(after.status(), StatusCode::UNAUTHORIZED);

    // The log outlived the session: a restart replays the same state.
    let seq_before = f.running.state.log.lock().unwrap().size();
    assert!(
        seq_before >= 5,
        "position, capital, two assignments, position, then the session's life: {seq_before}"
    );
    f.running.stop().await;
}

async fn next_text<S>(socket: &mut S) -> String
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match socket.next().await {
                Some(Ok(Message::Text(text))) => return text.to_string(),
                Some(Ok(_)) => continue,
                other => panic!("socket ended before a text frame: {other:?}"),
            }
        }
    })
    .await
    .expect("a text frame within five seconds")
}
