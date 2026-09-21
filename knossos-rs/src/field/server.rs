//! The Field server in the `knossos` binary. Port of
//! `field/server/src/index.js`, the request pipeline of `api.js`, and the
//! body reader in `body.js`.
//!
//! One handler sees every request in the reference's order: security
//! headers, the bootstrap link, the health probe, authorization, logout, the
//! API table, then the built web client with an SPA fallback. Routes whose
//! Node implementation still depends on the harness registry, the director
//! or process spawning answer `501 not_ported` until their port lands, so a
//! client can tell "not here yet" from "refused".

use super::config::{load_config, update_frontmatter_file, FieldSettings};
use super::eventlog::{AppendOptions, Backend, EventLog};
use super::hub::{Hub, Outbound};
use super::js::{get, get_arr, get_str, js_string};
use super::model::DomainError;
use super::page::{page_result, parse_event_page};
use super::projection::Projection;
use super::registry::{Capabilities, PermissionOutcome, Registry, RegistryOptions};
use super::replay::{read_event_range, replay_into, Range};
use super::security::{
    Authority, Bootstrap, ControlSecurity, Denied, RequestFacts, SecurityOptions,
};
use super::stores::{EndpointsStore, KeyStore};
use axum::body::Body;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{ConnectInfo, FromRequestParts, State};
use axum::http::{header, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Router;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tokio::sync::oneshot;

const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;
const BODY_TIMEOUT: Duration = Duration::from_secs(10);
const TERRITORY_IDS: [&str; 12] = [
    "italia",
    "gallia",
    "hispania",
    "africa",
    "aegyptus",
    "britannia",
    "dacia",
    "balkans",
    "anatolia",
    "levant",
    "mesopotamia",
    "cyrenaica",
];

/// Where the server finds its inputs.
#[derive(Debug, Clone)]
pub struct ServerOptions {
    /// `field/` configuration directory (holds `field.yaml`).
    pub field_dir: PathBuf,
    /// Data directory for the event log and stores.
    pub state_dir: PathBuf,
    /// Built web client, served with an SPA fallback when present.
    pub dist_dir: PathBuf,
    /// Listening port; `None` uses `field.api_port` or 7749.
    pub port: Option<u16>,
    /// Extra trusted origin and bootstrap redirect for a dev web server.
    pub ui_origin: Option<String>,
    pub backend: Option<Backend>,
    /// Fixed tokens, for tests; production mints random ones.
    pub bootstrap_token: Option<String>,
    pub browser_token: Option<String>,
}

pub struct AppState {
    pub settings: Arc<RwLock<FieldSettings>>,
    pub log: Arc<Mutex<EventLog>>,
    pub projection: Arc<Mutex<Projection>>,
    pub security: Arc<Mutex<ControlSecurity>>,
    pub hub: Arc<Hub>,
    pub registry: Arc<Mutex<Registry>>,
    pub keys: Arc<Mutex<KeyStore>>,
    pub endpoints_store: Arc<Mutex<EndpointsStore>>,
    pub dist_dir: PathBuf,
    headers: Vec<(HeaderName, HeaderValue)>,
}

impl AppState {
    /// Append, fold, fan out: the one path every event takes, live or replayed.
    pub fn emit(
        &self,
        kind: &str,
        data: Value,
        options: AppendOptions,
    ) -> Result<super::Event, ApiError> {
        let event = self
            .log
            .lock()
            .map_err(|_| ApiError::internal("event log lock poisoned"))?
            .append(kind, data, options)
            .map_err(|e| ApiError::internal(format!("event log append failed: {e}")))?;
        if let Ok(mut projection) = self.projection.lock() {
            projection.apply(&event);
        }
        self.hub.push_event(&event);
        Ok(event)
    }

    /// Run `f` under the security lock; `None` only if the lock is poisoned.
    fn with_security<R>(&self, f: impl FnOnce(&mut ControlSecurity) -> R) -> Option<R> {
        self.security.lock().ok().map(|mut guard| f(&mut guard))
    }

    fn snapshot(&self) -> Value {
        let now = now_ms();
        self.projection
            .lock()
            .map(|p| p.snapshot(now))
            .unwrap_or_else(|_| json!({}))
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// A JSON error reply: `{ error, code, ...detail }`.
#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: String,
    pub error: String,
    pub detail: Value,
}

impl ApiError {
    fn new(status: StatusCode, code: &str, error: impl Into<String>) -> Self {
        ApiError {
            status,
            code: code.into(),
            error: error.into(),
            detail: json!({}),
        }
    }

    fn internal(error: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", error)
    }

    fn not_ported(route: &str) -> Self {
        Self::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_ported",
            format!(
                "{route} is not served by the Rust Field server yet; run the Node server for it"
            ),
        )
    }
}

impl From<Denied> for ApiError {
    fn from(d: Denied) -> Self {
        ApiError::new(
            StatusCode::from_u16(d.status).unwrap_or(StatusCode::FORBIDDEN),
            d.code,
            d.error,
        )
    }
}

impl From<DomainError> for ApiError {
    fn from(e: DomainError) -> Self {
        let status = match e.code.as_str() {
            "not_found" => StatusCode::NOT_FOUND,
            "gate_blocked" | "invalid_transition" | "role_conflict" => StatusCode::CONFLICT,
            _ => StatusCode::BAD_REQUEST,
        };
        ApiError {
            status,
            code: e.code,
            error: e.message,
            detail: e.detail,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut body = json!({ "error": self.error, "code": self.code });
        if let (Value::Object(out), Value::Object(detail)) = (&mut body, &self.detail) {
            for (k, v) in detail {
                out.insert(k.clone(), v.clone());
            }
        }
        json_response(self.status, &body)
    }
}

fn json_response(status: StatusCode, body: &Value) -> Response {
    let text = body.to_string();
    (
        status,
        [
            (header::CONTENT_TYPE, "application/json; charset=utf-8"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        text,
    )
        .into_response()
}

fn facts(req: &Request<Body>, addr: SocketAddr) -> RequestFacts {
    let h = |name: header::HeaderName| {
        req.headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    };
    let bootstrap = req.uri().query().and_then(|q| {
        url::form_urlencoded::parse(q.as_bytes())
            .find(|(k, _)| k == "bootstrap")
            .map(|(_, v)| v.into_owned())
    });
    RequestFacts {
        method: req.method().as_str().to_string(),
        path: req.uri().path().to_string(),
        bootstrap,
        host: h(header::HOST),
        origin: h(header::ORIGIN),
        cookie: h(header::COOKIE),
        authorization: h(header::AUTHORIZATION),
        content_type: h(header::CONTENT_TYPE),
        remote_address: addr.ip().to_string(),
    }
}

fn query(req: &Request<Body>) -> Vec<(String, String)> {
    req.uri()
        .query()
        .map(|q| {
            url::form_urlencoded::parse(q.as_bytes())
                .map(|(k, v)| (k.into_owned(), v.into_owned()))
                .collect()
        })
        .unwrap_or_default()
}

fn param<'a>(query: &'a [(String, String)], key: &str) -> Option<&'a str> {
    query
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// One bounded JSON request body, by wire bytes.
///
/// An oversized body is drained rather than abandoned: closing the socket
/// with unread bytes makes some stacks (macOS) reset the connection before
/// the client has read the 413, which the reference avoids with
/// `req.resume()`. A hard ceiling still bounds the drain.
async fn read_json_body(body: Body) -> Result<Value, ApiError> {
    use futures_util::StreamExt;
    const DRAIN_CEILING: usize = 64 * 1024 * 1024;
    let mut stream = body.into_data_stream();
    let mut bytes: Vec<u8> = Vec::new();
    let mut total = 0usize;
    let mut too_large = false;
    let read = tokio::time::timeout(BODY_TIMEOUT, async {
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => {
                    return Err(ApiError::new(
                        StatusCode::BAD_REQUEST,
                        "body_aborted",
                        "request body was aborted",
                    ))
                }
            };
            total += chunk.len();
            if total > MAX_BODY_BYTES {
                too_large = true;
                bytes.clear();
                if total > DRAIN_CEILING {
                    break;
                }
                continue;
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(())
    })
    .await;
    match read {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(ApiError::new(
                StatusCode::REQUEST_TIMEOUT,
                "body_timeout",
                "request body timed out",
            ))
        }
    }
    if too_large {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "body_too_large",
            "request body exceeds the allowed size",
        ));
    }
    if bytes.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_slice(&bytes)
        .map_err(|_| ApiError::new(StatusCode::BAD_REQUEST, "invalid_json", "invalid JSON body"))
}

async fn handle(
    State(state): State<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response {
    let mut response = dispatch(&state, addr, req).await;
    for (name, value) in &state.headers {
        response.headers_mut().insert(name.clone(), value.clone());
    }
    response
}

async fn dispatch(state: &Arc<AppState>, addr: SocketAddr, req: Request<Body>) -> Response {
    let facts = facts(&req, addr);

    let bootstrap = state
        .with_security(|s| s.consume_bootstrap(&facts))
        .unwrap_or(Bootstrap::NotHandled);
    match bootstrap {
        Bootstrap::NotHandled => {}
        Bootstrap::Denied(denied) => return ApiError::from(denied).into_response(),
        Bootstrap::Granted { redirect, cookie } => {
            let mut response =
                (StatusCode::SEE_OTHER, [(header::LOCATION, redirect)]).into_response();
            if let Some(cookie) = cookie.and_then(|c| HeaderValue::from_str(&c).ok()) {
                response.headers_mut().insert(header::SET_COOKIE, cookie);
            }
            return response;
        }
    }

    if req.method() == Method::GET && facts.path == "/healthz" {
        if let Some(Err(denied)) = state.with_security(|s| s.authorize_network(&facts)) {
            return ApiError::from(denied).into_response();
        }
        return json_response(StatusCode::OK, &json!({ "ok": true }));
    }

    let decided: Option<Result<Authority, Denied>> = if facts.path == "/ws" {
        state.with_security(|s| s.authorize_upgrade(&facts))
    } else {
        state.with_security(|s| s.authorize_request(&facts))
    };
    let authority = match decided {
        Some(Ok(a)) => a,
        Some(Err(denied)) => return ApiError::from(denied).into_response(),
        None => return ApiError::internal("security lock poisoned").into_response(),
    };

    if facts.path == "/ws" {
        return websocket(state, req).await;
    }

    if req.method() == Method::POST && facts.path == "/api/logout" {
        let abandoned = state
            .registry
            .lock()
            .map(|mut r| r.abandon_operator())
            .unwrap_or_else(|_| json!({}));
        let cookie: Option<String> = state.with_security(|s| {
            s.revoke_all_harness_tokens();
            s.revoke_browser_session()
        });
        state.hub.revoke_clients();
        let mut body = json!({ "ok": true });
        if let (Value::Object(o), Value::Object(a)) = (&mut body, &abandoned) {
            for (k, v) in a {
                o.insert(k.clone(), v.clone());
            }
        }
        let mut response = json_response(StatusCode::OK, &body);
        if let Some(cookie) = cookie.and_then(|c| HeaderValue::from_str(&c).ok()) {
            response.headers_mut().insert(header::SET_COOKIE, cookie);
        }
        return response;
    }

    if facts.path.starts_with("/api/") {
        let method = req.method().clone();
        let params = query(&req);
        let body = if method == Method::POST || method == Method::PUT {
            match read_json_body(req.into_body()).await {
                Ok(v) => v,
                Err(e) => return e.into_response(),
            }
        } else {
            Value::Null
        };
        if method == Method::POST && facts.path == "/api/internal/permission" {
            return match internal_permission(state, body, &authority).await {
                Ok(value) => json_response(StatusCode::OK, &value),
                Err(e) => e.into_response(),
            };
        }
        return match api(state, &method, &facts.path, &params, body, &authority) {
            Ok(value) => json_response(StatusCode::OK, &value),
            Err(e) => e.into_response(),
        };
    }

    static_file(state, req.method(), &facts.path).await
}

fn api(
    state: &Arc<AppState>,
    method: &Method,
    path: &str,
    params: &[(String, String)],
    body: Value,
    _authority: &Authority,
) -> Result<Value, ApiError> {
    let key = format!("{method} {path}");
    match key.as_str() {
        "GET /api/state" => Ok(state.snapshot()),
        "GET /api/world" => Ok(state
            .projection
            .lock()
            .map(|p| Value::Object(p.world().clone()))
            .unwrap_or_else(|_| json!({}))),
        "GET /api/config" => Ok(state
            .settings
            .read()
            .map(|s| s.view())
            .unwrap_or_else(|_| json!({}))),
        "GET /api/events" => {
            let page = parse_event_page(param(params, "from"), param(params, "limit"))
                .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, e.code, e.message))?;
            let log = state
                .log
                .lock()
                .map_err(|_| ApiError::internal("event log lock poisoned"))?;
            let events = log
                .read(page.from, page.limit + 1)
                .map_err(|e| ApiError::internal(e.to_string()))?;
            let has_more = events.len() > page.limit;
            Ok(
                serde_json::to_value(page_result(events, page, Some(log.size()), Some(has_more)))
                    .unwrap_or_default(),
            )
        }
        "GET /api/trace" => {
            let Some(subject) = param(params, "subject").filter(|s| !s.is_empty()) else {
                return Err(DomainError::new("invalid_id", "subject is required").into());
            };
            let page = parse_event_page(param(params, "from"), param(params, "limit"))
                .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, e.code, e.message))?;
            let log = state
                .log
                .lock()
                .map_err(|_| ApiError::internal("event log lock poisoned"))?;
            let events = log
                .by_subject(subject, page.from, page.limit + 1)
                .map_err(|e| ApiError::internal(e.to_string()))?;
            let has_more = events.len() > page.limit;
            let mut out =
                serde_json::to_value(page_result(events, page, Some(log.size()), Some(has_more)))
                    .unwrap_or_default();
            out["subject"] = json!(subject);
            Ok(out)
        }
        "GET /api/campaigns" => {
            let projection = state
                .projection
                .lock()
                .map_err(|_| ApiError::internal("projection lock poisoned"))?;
            Ok(json!({ "campaigns": projection.campaigns.snapshot()["campaigns"] }))
        }
        "GET /api/campaigns/trace" => {
            let Some(campaign_id) = param(params, "campaignId").filter(|s| !s.is_empty()) else {
                return Err(DomainError::new("invalid_id", "campaignId is required").into());
            };
            let ids: Vec<String> = {
                let projection = state
                    .projection
                    .lock()
                    .map_err(|_| ApiError::internal("projection lock poisoned"))?;
                let Some(campaign) = projection.campaigns.campaigns.get(campaign_id) else {
                    return Err(DomainError::new(
                        "not_found",
                        format!("campaign {campaign_id} does not exist"),
                    )
                    .into());
                };
                let mut ids = vec![campaign_id.to_string()];
                for key in [
                    "objectiveIds",
                    "findingIds",
                    "mitigationIds",
                    "verdictIds",
                    "checkpointIds",
                ] {
                    if let Some(Value::Array(items)) = campaign.get(key) {
                        ids.extend(items.iter().map(js_string));
                    }
                }
                ids
            };
            let page = parse_event_page(param(params, "from"), param(params, "limit"))
                .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, e.code, e.message))?;
            let log = state
                .log
                .lock()
                .map_err(|_| ApiError::internal("event log lock poisoned"))?;
            let events: Vec<_> = read_event_range(&*log, Range::default())
                .into_iter()
                .filter(|evt| {
                    evt.seq > page.from
                        && (evt.subject.as_ref().is_some_and(|s| ids.contains(s))
                            || get_str(&evt.data, "campaignId") == Some(campaign_id))
                })
                .take(page.limit + 1)
                .collect();
            let has_more = events.len() > page.limit;
            let mut out =
                serde_json::to_value(page_result(events, page, Some(log.size()), Some(has_more)))
                    .unwrap_or_default();
            out["campaignId"] = json!(campaign_id);
            Ok(out)
        }
        "GET /api/campaigns/replay" => {
            let Some(campaign_id) = param(params, "campaignId").filter(|s| !s.is_empty()) else {
                return Err(DomainError::new("invalid_id", "campaignId is required").into());
            };
            {
                let projection = state
                    .projection
                    .lock()
                    .map_err(|_| ApiError::internal("projection lock poisoned"))?;
                if !projection.campaigns.campaigns.contains(campaign_id) {
                    return Err(DomainError::new(
                        "not_found",
                        format!("campaign {campaign_id} does not exist"),
                    )
                    .into());
                }
            }
            let log = state
                .log
                .lock()
                .map_err(|_| ApiError::internal("event log lock poisoned"))?;
            let size = log.size();
            let requested = match param(params, "seq") {
                None => size,
                Some(raw) => match raw.trim().parse::<u64>() {
                    Ok(n) if n <= size => n,
                    _ => {
                        return Err(DomainError::new(
                            "invalid_seq",
                            format!("seq must be an integer between 0 and {size}"),
                        )
                        .into());
                    }
                },
            };
            let seed = state
                .settings
                .read()
                .map(|s| s.projection_config())
                .unwrap_or_default();
            let mut historical = Projection::new(seed);
            let replayed = replay_into(&*log, &mut historical, Range::default().to(requested));
            let now = replayed
                .last_event
                .as_ref()
                .map(|e| e.ts)
                .unwrap_or_else(now_ms);
            let campaign = historical
                .campaigns
                .campaigns
                .get(campaign_id)
                .map(|c| historical.campaigns.campaign_view(c));
            Ok(json!({
                "campaignId": campaign_id,
                "requestedSeq": requested,
                "actualSeq": replayed.last_event.as_ref().map(|e| e.seq).unwrap_or(0),
                "replayedEvents": replayed.count,
                "campaign": campaign,
                "graph": historical.graph.snapshot(now, Some(4000), Some(8000)),
            }))
        }
        "POST /api/world/capital" => {
            let workspace_id = get(&body, "workspaceId")
                .map(js_string)
                .unwrap_or_default()
                .trim()
                .to_string();
            let mounted = state
                .settings
                .read()
                .ok()
                .and_then(|s| s.workspace(&workspace_id).cloned())
                .is_some_and(|w| w.get("mounted") == Some(&Value::Bool(true)));
            if !mounted {
                let shown = if workspace_id.is_empty() {
                    "(empty)".to_string()
                } else {
                    workspace_id.clone()
                };
                return Err(DomainError::new(
                    "not_found",
                    format!("mounted workspace {shown} does not exist"),
                )
                .into());
            }
            state.emit(
                "world.capital_selected",
                json!({ "workspaceId": workspace_id }),
                AppendOptions::subject(workspace_id),
            )?;
            Ok(state
                .projection
                .lock()
                .map(|p| Value::Object(p.world().clone()))
                .unwrap_or_else(|_| json!({})))
        }
        "POST /api/world/reconcile" => reconcile_world(state, &body),
        "POST /api/position" => {
            state.emit("ui.position", body, AppendOptions::default())?;
            Ok(json!({ "ok": true }))
        }
        "POST /api/control-group" => {
            state.emit(
                "ui.control_group",
                json!({ "group": body.get("group"), "sessionIds": get_arr(&body, "sessionIds").cloned().unwrap_or_default() }),
                AppendOptions::default(),
            )?;
            Ok(json!({ "ok": true }))
        }
        "POST /api/sessions" => {
            let id = state
                .registry
                .lock()
                .map_err(|_| ApiError::internal("registry lock poisoned"))?
                .spawn(&body)
                .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, "bad_request", e.0))?;
            Ok(json!({ "sessionId": id }))
        }
        "POST /api/assign" => state
            .registry
            .lock()
            .map_err(|_| ApiError::internal("registry lock poisoned"))?
            .assign(&body)
            .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, "bad_request", e.0)),
        "POST /api/command" => {
            let kind = get_str(&body, "kind").unwrap_or("").to_string();
            state
                .registry
                .lock()
                .map_err(|_| ApiError::internal("registry lock poisoned"))?
                .command(&kind, &body)
                .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, "bad_request", e.0))
        }
        "POST /api/permission/decide" => {
            let id = get_str(&body, "permissionId").unwrap_or("").to_string();
            let decision = get_str(&body, "decision").unwrap_or("deny").to_string();
            let message = get_str(&body, "message").map(str::to_string);
            let ok = state
                .registry
                .lock()
                .map_err(|_| ApiError::internal("registry lock poisoned"))?
                .decide_permission(&id, &decision, message.as_deref());
            Ok(json!({ "ok": ok }))
        }
        "POST /api/agents/settings" => agent_settings(state, &body),
        "GET /api/endpoints" => {
            let settings = state
                .settings
                .read()
                .map_err(|_| ApiError::internal("settings lock poisoned"))?;
            let store = state
                .endpoints_store
                .lock()
                .map_err(|_| ApiError::internal("endpoints store lock poisoned"))?;
            let snapshot = state.snapshot();
            let status_of = |id: &str| -> Value {
                snapshot["endpoints"]
                    .as_array()
                    .and_then(|list| list.iter().find(|e| e["id"] == id))
                    .and_then(|e| e.get("status").cloned())
                    .unwrap_or(json!("unknown"))
            };
            Ok(json!({
                "endpoints": settings.endpoints.iter().map(|e| {
                    let id = get_str(e, "id").unwrap_or("");
                    json!({
                        "id": id, "name": get(e, "name").cloned().unwrap_or(json!(id)), "kind": e.get("kind"),
                        "model": or_null_value(e.get("model")),
                        "base_url": get(e, "base_url").or(get(e, "baseUrl")).cloned().unwrap_or(Value::Null),
                        "source": if store.get(id).is_some() { "user" } else { "config" },
                        "hasKey": get(e, "secretRef").is_some() || get(e, "credential_env").is_some(),
                        "status": status_of(id),
                    })
                }).collect::<Vec<_>>(),
            }))
        }
        "POST /api/endpoints" => add_endpoint(state, &body),
        "POST /api/endpoints/test" => Err(ApiError::not_ported(&key)),
        "POST /api/endpoints/delete" => {
            let id = get_str(&body, "id").unwrap_or("").to_string();
            let desc = state
                .endpoints_store
                .lock()
                .ok()
                .and_then(|s| s.get(&id).cloned());
            let Some(desc) = desc else {
                let shown = if id.is_empty() {
                    "(empty)".to_string()
                } else {
                    id.clone()
                };
                return Err(DomainError::new(
                    "not_found",
                    format!("{shown} is not a user-added endpoint"),
                )
                .into());
            };
            if let Some(secret) = get_str(&desc, "secretRef") {
                if let Ok(mut keys) = state.keys.lock() {
                    let _ = keys.delete(secret);
                }
            }
            if let Ok(mut store) = state.endpoints_store.lock() {
                let _ = store.remove(&id);
            }
            if let Ok(mut settings) = state.settings.write() {
                settings.endpoints.retain(|e| get_str(e, "id") != Some(&id));
            }
            if let Ok(mut projection) = state.projection.lock() {
                projection.remove_endpoint(&id);
            }
            Ok(json!({ "ok": true }))
        }
        "GET /api/simulations" => {
            Ok(json!({ "enabled": false, "active": Value::Null, "scenarios": [] }))
        }
        "GET /api/routines/states" => {
            let snapshot = state.snapshot();
            Ok(json!({ "states": snapshot["routines"], "details": [] }))
        }
        _ => Err(ApiError::not_ported(&key)),
    }
}

fn or_null_value(v: Option<&Value>) -> Value {
    v.cloned().unwrap_or(Value::Null)
}

/// `POST /api/internal/permission`: called by a harness with its session
/// capability; held open until the operator or the policy decides.
async fn internal_permission(
    state: &Arc<AppState>,
    body: Value,
    authority: &Authority,
) -> Result<Value, ApiError> {
    let Authority::Harness { session_id } = authority else {
        return Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "internal_auth_required",
            "Valid harness authorization is required.",
        ));
    };
    let outcome = state
        .registry
        .lock()
        .map_err(|_| ApiError::internal("registry lock poisoned"))?
        .request_permission(
            session_id,
            get_str(&body, "toolName").unwrap_or(""),
            body.get("input").cloned().unwrap_or(Value::Null),
            body.get("toolUseId").cloned(),
            Some(session_id),
        );
    let decision = match outcome {
        PermissionOutcome::Immediate(d) => d,
        PermissionOutcome::Pending(rx) => rx.await.unwrap_or(super::registry::Decision {
            decision: "deny".into(),
            message: Some("the permission request was dropped".into()),
        }),
    };
    Ok(json!({ "decision": decision.decision, "message": decision.message }))
}

fn agent_settings(state: &Arc<AppState>, body: &Value) -> Result<Value, ApiError> {
    let agent_id = get_str(body, "agentId").unwrap_or("").trim().to_string();
    let mut settings = state
        .settings
        .write()
        .map_err(|_| ApiError::internal("settings lock poisoned"))?;
    let Some(index) = settings
        .agents
        .iter()
        .position(|a| get_str(a, "id") == Some(&agent_id))
    else {
        let shown = if agent_id.is_empty() {
            "(empty)".to_string()
        } else {
            agent_id.clone()
        };
        return Err(DomainError::new("not_found", format!("agent {shown} does not exist")).into());
    };
    let agent = settings.agents[index].clone();
    let role = get_str(&agent, "role").and_then(|r| {
        settings
            .roles
            .iter()
            .find(|x| get_str(x, "id") == Some(r))
            .cloned()
    });
    let role_tools: Vec<String> = role
        .as_ref()
        .and_then(|r| get_arr(r, "tools_allow"))
        .map(|t| t.iter().map(js_string).collect())
        .unwrap_or_default();
    let requested: Vec<String> = match get_arr(body, "toolsAllow") {
        Some(items) => items.iter().map(js_string).collect(),
        None => get_arr(&agent, "tools_allow")
            .map(|t| t.iter().map(js_string).collect())
            .unwrap_or_else(|| role_tools.clone()),
    };
    let mut tools_allow: Vec<String> = Vec::new();
    for tool in requested {
        if role_tools.contains(&tool) && !tools_allow.contains(&tool) {
            tools_allow.push(tool);
        }
    }
    let name: String = get(body, "name")
        .map(js_string)
        .or_else(|| get_str(&agent, "name").map(str::to_string))
        .unwrap_or_else(|| agent_id.clone())
        .trim()
        .chars()
        .take(64)
        .collect();
    let name = if name.is_empty() {
        agent_id.clone()
    } else {
        name
    };
    let updates = json!({ "name": name, "tools_allow": tools_allow });
    if let Some(file) = get_str(&agent, "file") {
        update_frontmatter_file(Path::new(file), &updates)
            .map_err(|e| ApiError::internal(format!("cannot update {file}: {e}")))?;
    }
    if let Value::Object(a) = &mut settings.agents[index] {
        a.insert("name".into(), json!(name));
        a.insert("tools_allow".into(), json!(tools_allow));
    }
    let mut stripped = settings.agents[index]
        .as_object()
        .cloned()
        .unwrap_or_default();
    stripped.remove("body");
    stripped.remove("file");
    drop(settings);
    state.emit(
        "agent.settings_updated",
        json!({ "agentId": agent_id, "name": name, "toolsAllow": tools_allow }),
        AppendOptions::subject(agent_id.clone()),
    )?;
    Ok(json!({ "agent": Value::Object(stripped), "appliesTo": "future_sessions" }))
}

fn add_endpoint(state: &Arc<AppState>, body: &Value) -> Result<Value, ApiError> {
    let raw_id = get(body, "id").map(js_string).unwrap_or_default();
    let id: String = raw_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .take(48)
        .collect();
    if id.is_empty() {
        return Err(
            DomainError::new("bad_request", "id is required (letters, digits, - _)").into(),
        );
    }
    let kind = get_str(body, "kind").unwrap_or("").to_string();
    if !matches!(kind.as_str(), "anthropic" | "openai-compatible") {
        return Err(DomainError::new(
            "bad_request",
            "kind must be \"anthropic\" or \"openai-compatible\"",
        )
        .into());
    }
    let name: String = get(body, "name")
        .map(js_string)
        .unwrap_or_else(|| id.clone())
        .chars()
        .take(64)
        .collect();
    let model: Option<String> = get(body, "model")
        .filter(|m| super::js::truthy(m))
        .map(|m| js_string(m).chars().take(128).collect());
    let base_url: Option<String> = get(body, "base_url")
        .filter(|b| super::js::truthy(b))
        .map(|b| js_string(b).trim().to_string());
    if let Some(b) = &base_url {
        let lower = b.to_ascii_lowercase();
        if !(lower.starts_with("http://") || lower.starts_with("https://")) {
            return Err(DomainError::new(
                "bad_request",
                "base_url must start with http:// or https://",
            )
            .into());
        }
    }
    let mut desc =
        json!({ "id": id, "name": name, "kind": kind, "model": model, "base_url": base_url });
    let key = get(body, "key")
        .filter(|k| super::js::truthy(k))
        .map(js_string);
    if let Some(key) = &key {
        let secret_ref = format!("ep:{id}");
        state
            .keys
            .lock()
            .map_err(|_| ApiError::internal("key store lock poisoned"))?
            .set(&secret_ref, key)
            .map_err(|e| ApiError::internal(format!("cannot store the key: {e}")))?;
        if let Ok(mut log) = state.log.lock() {
            log.add_secret(key.clone());
        }
        desc["secretRef"] = json!(secret_ref);
    }
    state
        .endpoints_store
        .lock()
        .map_err(|_| ApiError::internal("endpoints store lock poisoned"))?
        .add(desc.clone())
        .map_err(|e| ApiError::internal(format!("cannot store the endpoint: {e}")))?;
    if let Ok(mut settings) = state.settings.write() {
        match settings
            .endpoints
            .iter()
            .position(|e| get_str(e, "id") == Some(&id))
        {
            Some(i) => settings.endpoints[i] = desc.clone(),
            None => settings.endpoints.push(desc.clone()),
        }
    }
    if let Ok(mut projection) = state.projection.lock() {
        projection.ensure_endpoint(&id, &name, &kind, model.as_deref(), base_url.as_deref());
    }
    Ok(json!({ "ok": true, "id": id, "hasKey": key.is_some() }))
}

fn world_cluster(input: &Value) -> Result<Value, DomainError> {
    let clip = |s: String, n: usize| s.chars().take(n).collect::<String>();
    let cluster_key = clip(
        get(input, "clusterKey")
            .map(js_string)
            .unwrap_or_default()
            .trim()
            .to_string(),
        160,
    );
    if cluster_key.is_empty() {
        return Err(DomainError::new("invalid_id", "clusterKey is required"));
    }
    let label = clip(
        get(input, "label")
            .map(js_string)
            .unwrap_or_else(|| cluster_key.clone())
            .trim()
            .to_string(),
        96,
    );
    let kind = clip(
        get(input, "kind")
            .map(js_string)
            .unwrap_or_else(|| "infrastructure".into())
            .trim()
            .to_string(),
        48,
    );
    Ok(json!({
        "clusterKey": cluster_key,
        "label": if label.is_empty() { cluster_key.clone() } else { label },
        "kind": if kind.is_empty() { "infrastructure".to_string() } else { kind },
        "workspaceId": get(input, "workspaceId").filter(|w| super::js::truthy(w)).map(|w| json!(clip(js_string(w), 96))).unwrap_or(Value::Null),
    }))
}

fn reconcile_world(state: &Arc<AppState>, body: &Value) -> Result<Value, ApiError> {
    let raw = get_arr(body, "clusters").cloned().unwrap_or_default();
    if raw.len() > 96 {
        return Err(DomainError::new(
            "invalid_request",
            "at most 96 infrastructure clusters may be reconciled",
        )
        .into());
    }
    let clusters: Vec<Value> = raw.iter().map(world_cluster).collect::<Result<_, _>>()?;
    let world = || -> Value {
        state
            .projection
            .lock()
            .map(|p| Value::Object(p.world().clone()))
            .unwrap_or_else(|_| json!({}))
    };
    let capital_workspace_id = get(&world(), "capitalWorkspaceId").map(js_string);
    let Some(capital_workspace_id) = capital_workspace_id.filter(|c| !c.is_empty()) else {
        return Err(DomainError::new(
            "invalid_state",
            "choose a capital project before reconciling the world",
        )
        .into());
    };
    let capital_key = format!("workspace:{capital_workspace_id}");
    let capital_cluster = clusters
        .iter()
        .find(|c| get_str(c, "clusterKey") == Some(&capital_key))
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "clusterKey": capital_key,
                "label": state.settings.read().ok().and_then(|s| s.workspace(&capital_workspace_id).and_then(|w| get(w, "name")).cloned()).unwrap_or(json!(capital_workspace_id)),
                "kind": "project",
                "workspaceId": capital_workspace_id,
            })
        });
    let assignments_of = |w: &Value| -> Vec<Value> {
        w.get("assignments")
            .and_then(Value::as_object)
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    };
    let initial = world();
    let assignments = assignments_of(&initial);
    let mut current_keys: Vec<String> = clusters
        .iter()
        .filter_map(|c| get_str(c, "clusterKey").map(str::to_string))
        .collect();
    current_keys.push(capital_key.clone());
    for assignment in &assignments {
        let key = get_str(assignment, "clusterKey").unwrap_or("").to_string();
        if current_keys.contains(&key) {
            continue;
        }
        state.emit(
            "world.territory_released",
            json!({ "clusterKey": key }),
            AppendOptions::subject(key.clone()),
        )?;
    }
    let current_italia = assignments
        .iter()
        .find(|a| get_str(a, "territoryId") == Some("italia"))
        .cloned();
    let capital_previous = initial["assignments"]
        .get(&capital_key)
        .and_then(|a| get_str(a, "territoryId"))
        .map(str::to_string);
    if let Some(italia) = current_italia.filter(|i| get_str(i, "clusterKey") != Some(&capital_key))
    {
        let occupied: Vec<String> = assignments
            .iter()
            .filter_map(|a| get_str(a, "territoryId").map(str::to_string))
            .collect();
        let destination = match capital_previous.as_deref() {
            Some(prev) if prev != "italia" => Some(prev.to_string()),
            _ => TERRITORY_IDS
                .iter()
                .find(|id| **id != "italia" && !occupied.contains(&id.to_string()))
                .map(|s| s.to_string()),
        };
        let cluster_key = get_str(&italia, "clusterKey").unwrap_or("").to_string();
        match destination {
            Some(territory) => {
                let mut data = italia.clone();
                data["territoryId"] = json!(territory);
                state.emit(
                    "world.territory_assigned",
                    data,
                    AppendOptions::subject(cluster_key),
                )?;
            }
            None => {
                state.emit(
                    "world.territory_released",
                    json!({ "clusterKey": cluster_key }),
                    AppendOptions::subject(cluster_key),
                )?;
            }
        }
    }
    if world()["assignments"]
        .get(&capital_key)
        .and_then(|a| get_str(a, "territoryId"))
        != Some("italia")
    {
        let mut data = capital_cluster.clone();
        data["territoryId"] = json!("italia");
        state.emit(
            "world.territory_assigned",
            data,
            AppendOptions::subject(capital_key.clone()),
        )?;
    }
    let mut ordered: Vec<Value> = clusters
        .iter()
        .filter(|c| get_str(c, "clusterKey") != Some(&capital_key))
        .cloned()
        .collect();
    ordered.sort_by(|a, b| {
        get_str(a, "clusterKey")
            .unwrap_or("")
            .cmp(get_str(b, "clusterKey").unwrap_or(""))
    });
    for cluster in ordered {
        let key = get_str(&cluster, "clusterKey").unwrap_or("").to_string();
        let current = world();
        if current["assignments"].get(&key).is_some() {
            continue;
        }
        let occupied: Vec<String> = assignments_of(&current)
            .iter()
            .filter_map(|a| get_str(a, "territoryId").map(str::to_string))
            .collect();
        let Some(territory) = TERRITORY_IDS
            .iter()
            .find(|id| **id != "italia" && !occupied.contains(&id.to_string()))
        else {
            break;
        };
        let mut data = cluster.clone();
        data["territoryId"] = json!(territory);
        state.emit(
            "world.territory_assigned",
            data,
            AppendOptions::subject(key),
        )?;
    }
    Ok(world())
}

async fn websocket(state: &Arc<AppState>, req: Request<Body>) -> Response {
    let (mut parts, _body) = req.into_parts();
    let upgrade = match WebSocketUpgrade::from_request_parts(&mut parts, &()).await {
        Ok(u) => u,
        Err(_) => {
            return ApiError::new(
                StatusCode::BAD_REQUEST,
                "upgrade_required",
                "expected a WebSocket upgrade",
            )
            .into_response()
        }
    };
    let state = Arc::clone(state);
    upgrade.on_upgrade(move |socket| client(socket, state))
}

async fn client(mut socket: WebSocket, state: Arc<AppState>) {
    let mut rx = state.hub.subscribe();
    if socket
        .send(Message::Text(state.hub.snapshot_message().into()))
        .await
        .is_err()
    {
        return;
    }
    loop {
        tokio::select! {
            outbound = rx.recv() => match outbound {
                Ok(Outbound::Text(text)) => {
                    if socket.send(Message::Text(text.to_string().into())).await.is_err() {
                        break;
                    }
                }
                Ok(Outbound::Revoke) => {
                    let _ = socket.send(Message::Close(None)).await;
                    break;
                }
                // Too far behind: terminate rather than buffer without bound.
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => break,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            },
            incoming = socket.recv() => match incoming {
                // The client sends nothing meaningful; commands go over HTTP.
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                Some(Ok(_)) => {}
            },
        }
    }
}

fn mime_for(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "woff2" => "font/woff2",
        "png" => "image/png",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        _ => "application/octet-stream",
    }
}

fn inside(root: &Path, candidate: &Path) -> bool {
    let (root, candidate) = if cfg!(windows) {
        (
            PathBuf::from(root.to_string_lossy().to_lowercase()),
            PathBuf::from(candidate.to_string_lossy().to_lowercase()),
        )
    } else {
        (root.to_path_buf(), candidate.to_path_buf())
    };
    candidate == root || candidate.starts_with(&root)
}

async fn static_file(state: &Arc<AppState>, method: &Method, path: &str) -> Response {
    let dist = &state.dist_dir;
    if (method == Method::GET || method == Method::HEAD) && dist.is_dir() {
        let rel = if path == "/" {
            "index.html"
        } else {
            path.trim_start_matches('/')
        };
        let file = super::config::dunce_canonicalize(&dist.join(rel));
        let root = super::config::dunce_canonicalize(dist).unwrap_or_else(|| dist.clone());
        if let Some(file) = file.filter(|f| inside(&root, f) && f.is_file()) {
            if let Ok(bytes) = tokio::fs::read(&file).await {
                return (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, mime_for(&file))],
                    bytes,
                )
                    .into_response();
            }
        }
        let index = dist.join("index.html");
        if let Ok(bytes) = tokio::fs::read(&index).await {
            return (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
                bytes,
            )
                .into_response();
        }
    }
    json_response(
        StatusCode::NOT_FOUND,
        &json!({ "error": "not found", "hint": "run `npm run web` for the dev interface" }),
    )
}

/// A running server, for the caller to stop.
pub struct Running {
    pub addr: SocketAddr,
    pub bootstrap_url: String,
    pub state: Arc<AppState>,
    shutdown: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl Running {
    pub async fn stop(mut self) {
        if let Ok(mut registry) = self.state.registry.lock() {
            registry.shutdown();
        }
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        let _ = self.task.await;
    }
}

/// Real health for real inference connections, on a slow heartbeat. Port of
/// `field/server/src/endpoints.js`: an `openai-compatible` endpoint answers
/// `GET /models`; an `anthropic` one is probed for reachability of the
/// provider host, which proves the route out and nothing about credentials.
fn spawn_endpoint_probes(state: Arc<AppState>) {
    const INTERVAL: Duration = Duration::from_secs(20);
    const HEARTBEAT: Duration = Duration::from_secs(5 * 60);
    const TIMEOUT: Duration = Duration::from_millis(2500);
    tokio::spawn(async move {
        let client = reqwest::Client::builder().timeout(TIMEOUT).build().ok();
        let mut last: std::collections::HashMap<String, (String, std::time::Instant)> =
            std::collections::HashMap::new();
        loop {
            let endpoints: Vec<Value> = state
                .settings
                .read()
                .map(|s| s.endpoints.clone())
                .unwrap_or_default();
            for ep in endpoints {
                let id = get_str(&ep, "id").unwrap_or("").to_string();
                let kind = get_str(&ep, "kind").unwrap_or("");
                let base = get_str(&ep, "base_url")
                    .or(get_str(&ep, "baseUrl"))
                    .map(|b| b.trim_end_matches('/').to_string());
                let (status, latency, detail) = match (kind, base, &client) {
                    ("openai-compatible" | "cameo", Some(base), Some(client)) => {
                        probe_http(client, &format!("{base}/models")).await
                    }
                    ("anthropic", _, Some(client)) => {
                        let (s, l, d) =
                            probe_http(client, "https://api.anthropic.com/v1/models").await;
                        let d = if s == "up" {
                            format!("reachable · {d}")
                        } else {
                            d
                        };
                        (s, l, d)
                    }
                    _ => (
                        "unknown".to_string(),
                        None,
                        format!("no probe defined for kind \"{kind}\""),
                    ),
                };
                let now = std::time::Instant::now();
                let previous = last.get(&id).cloned();
                let changed = previous.as_ref().map(|(s, _)| s != &status).unwrap_or(true);
                let stale = previous
                    .as_ref()
                    .map(|(_, at)| now.duration_since(*at) > HEARTBEAT)
                    .unwrap_or(true);
                let at = if changed || stale {
                    now
                } else {
                    previous.as_ref().map(|(_, at)| *at).unwrap_or(now)
                };
                last.insert(id.clone(), (status.clone(), at));
                if changed || stale {
                    let _ = state.emit(
                        "endpoint.health",
                        json!({ "endpointId": id, "status": status, "latencyMs": latency, "detail": detail }),
                        AppendOptions::subject(id.clone()),
                    );
                }
                if changed {
                    if let Ok(mut registry) = state.registry.lock() {
                        registry.on_endpoint_health(&id, &status);
                    }
                }
            }
            tokio::time::sleep(INTERVAL).await;
        }
    });
}

async fn probe_http(client: &reqwest::Client, url: &str) -> (String, Option<u64>, String) {
    let started = std::time::Instant::now();
    match client.get(url).send().await {
        Ok(res) => {
            let code = res.status().as_u16();
            let status = if res.status().is_success() || code == 401 || code == 403 {
                "up"
            } else {
                "degraded"
            };
            (
                status.to_string(),
                Some(started.elapsed().as_millis() as u64),
                format!("HTTP {code}"),
            )
        }
        Err(e) => {
            let detail = if e.is_timeout() {
                "no response in 2500ms".to_string()
            } else {
                e.to_string()
            };
            ("down".to_string(), None, detail)
        }
    }
}

/// Load configuration, replay the log, bind loopback and serve.
pub async fn start(options: ServerOptions) -> anyhow::Result<Running> {
    use anyhow::Context;
    let settings = load_config(&options.field_dir).context("loading field configuration")?;
    let requested = options.port.or_else(|| settings.api_port()).unwrap_or(7749);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", requested))
        .await
        .with_context(|| format!("binding 127.0.0.1:{requested}"))?;
    let addr = listener.local_addr()?;
    let port = addr.port();

    let frame_sources: Vec<String> = settings
        .websites
        .iter()
        .filter_map(|s| get_str(s, "domain"))
        .flat_map(|d| [format!("https://{d}"), format!("http://{d}")])
        .collect();
    let mut security_options = SecurityOptions::new(port);
    security_options.trusted_origins = options.ui_origin.iter().cloned().collect();
    security_options.bootstrap_redirect = options.ui_origin.clone().unwrap_or_else(|| "/".into());
    security_options.frame_sources = frame_sources;
    security_options.bootstrap_token = options.bootstrap_token.clone();
    security_options.browser_token = options.browser_token.clone();
    let security = ControlSecurity::new(security_options).context("field security")?;

    // Every credential-shaped environment value and both tokens are redaction
    // secrets before the first event is written.
    let secret_name =
        regex::Regex::new(r"(?i)(api[_-]?key|token|secret|password|authorization|cookie)")
            .expect("static regex");
    let mut secrets: Vec<String> = std::env::vars()
        .filter(|(k, v)| !v.is_empty() && secret_name.is_match(k))
        .map(|(_, v)| v)
        .collect();
    secrets.push(security.bootstrap_token().to_string());
    secrets.push(security.browser_token().to_string());
    let log = EventLog::open(&options.state_dir, secrets, options.backend)
        .context("opening the event log")?;

    // UI-added models: keys live only in the key store; endpoint descriptors
    // merge over field.yaml. Every stored key is a redaction secret before
    // anything runs.
    let keys = KeyStore::open(&options.state_dir);
    let endpoints_store = EndpointsStore::open(&options.state_dir);
    let mut log = log;
    for value in keys.values() {
        log.add_secret(value);
    }
    let mut settings = settings;
    for desc in endpoints_store.all() {
        let id = get_str(&desc, "id").unwrap_or("").to_string();
        match settings
            .endpoints
            .iter()
            .position(|e| get_str(e, "id") == Some(&id))
        {
            Some(i) => settings.endpoints[i] = desc,
            None => settings.endpoints.push(desc),
        }
    }

    let mut projection = Projection::new(settings.projection_config());
    let replay_started = std::time::Instant::now();
    let replayed = replay_into(&log, &mut projection, Range::default());
    let projection = Arc::new(Mutex::new(projection));

    let snapshot_source = Arc::clone(&projection);
    let hub = Hub::new(Arc::new(move || {
        let now = now_ms();
        snapshot_source
            .lock()
            .map(|p| p.snapshot(now))
            .unwrap_or_else(|_| json!({}))
    }));

    let headers = security
        .headers()
        .into_iter()
        .filter_map(|(k, v)| Some((HeaderName::from_static(k), HeaderValue::from_str(&v).ok()?)))
        .collect();
    let bootstrap_url = security.bootstrap_url();
    let settings = Arc::new(RwLock::new(settings));
    let log = Arc::new(Mutex::new(log));
    let security = Arc::new(Mutex::new(security));
    let keys = Arc::new(Mutex::new(keys));

    // The registry emits through the same append-fold-fanout path as the API.
    let emit_log = Arc::clone(&log);
    let emit_projection = Arc::clone(&projection);
    let emit_hub = Arc::clone(&hub);
    let emit: super::registry::Emit =
        Arc::new(move |kind: &str, data: Value, options: AppendOptions| {
            let event = emit_log.lock().ok()?.append(kind, data, options).ok()?;
            if let Ok(mut p) = emit_projection.lock() {
                p.apply(&event);
            }
            emit_hub.push_event(&event);
            Some(event)
        });
    let policy_projection = Arc::clone(&projection);
    let campaign_policy: super::registry::CampaignPolicy = Arc::new(move |id: &str| {
        let p = policy_projection.lock().ok()?;
        let campaign = p.campaigns.campaigns.get(id)?;
        Some(p.campaigns.campaign_view(campaign))
    });
    let mint_security = Arc::clone(&security);
    let revoke_security = Arc::clone(&security);
    let secret_log = Arc::clone(&log);
    let registry = Registry::new(RegistryOptions {
        settings: Arc::clone(&settings),
        emit,
        api_base: format!("http://127.0.0.1:{port}"),
        keys: Some(Arc::clone(&keys)),
        capabilities: Some(Capabilities {
            mint: Arc::new(move |session_id: &str| {
                mint_security
                    .lock()
                    .ok()
                    .map(|mut s| s.mint_harness_token(session_id))
            }),
            revoke: Arc::new(move |session_id: &str| {
                if let Ok(mut s) = revoke_security.lock() {
                    s.revoke_harness_token(session_id);
                }
            }),
        }),
        register_secret: Some(Arc::new(move |secret: &str| {
            if let Ok(mut l) = secret_log.lock() {
                l.add_secret(secret.to_string());
            }
        })),
        campaign_policy: Some(campaign_policy),
        factory: None,
    });

    let state = Arc::new(AppState {
        settings,
        log,
        projection,
        security,
        hub,
        registry,
        keys,
        endpoints_store: Arc::new(Mutex::new(endpoints_store)),
        dist_dir: options.dist_dir.clone(),
        headers,
    });
    spawn_endpoint_probes(Arc::clone(&state));

    let app = Router::new()
        .fallback(handle)
        .with_state(Arc::clone(&state));
    let (tx, rx) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(async {
            let _ = rx.await;
        })
        .await;
    });
    {
        let log = state
            .log
            .lock()
            .map_err(|_| anyhow::anyhow!("event log lock poisoned"))?;
        let settings = state
            .settings
            .read()
            .map_err(|_| anyhow::anyhow!("settings lock poisoned"))?;
        let mounted = settings
            .workspaces
            .iter()
            .filter(|w| w.get("mounted") == Some(&Value::Bool(true)))
            .count();
        eprintln!("\n  Field server  http://127.0.0.1:{}", addr.port());
        eprintln!("  open          {bootstrap_url}");
        eprintln!(
            "  store         {:?} ({} events, replayed {} in {}ms)",
            log.backend(),
            log.size(),
            replayed.count,
            replay_started.elapsed().as_millis()
        );
        eprintln!(
            "  workspaces    {mounted}/{} mounted",
            settings.workspaces.len()
        );
        eprintln!(
            "  agents        {} defined · {} roles",
            settings.agents.len(),
            settings.roles.len()
        );
        eprintln!("  endpoints     {} configured", settings.endpoints.len());
        eprintln!(
            "  runtime       rust (knossos field --native); routes not yet ported answer 501"
        );
        for w in settings
            .workspaces
            .iter()
            .filter(|w| w.get("mounted") != Some(&Value::Bool(true)))
        {
            eprintln!(
                "  ! workspace \"{}\" is not mounted: {} does not exist",
                get_str(w, "id").unwrap_or("?"),
                get_str(w, "path").unwrap_or("?")
            );
        }
        eprintln!();
    }
    Ok(Running {
        addr,
        bootstrap_url,
        state,
        shutdown: Some(tx),
        task,
    })
}

/// Serve until Ctrl+C.
pub async fn serve(options: ServerOptions) -> anyhow::Result<()> {
    let running = start(options).await?;
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("\n  stopping…");
    running.stop().await;
    Ok(())
}
