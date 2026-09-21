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

use super::config::{load_config, FieldSettings};
use super::eventlog::{AppendOptions, Backend, EventLog};
use super::hub::{Hub, Outbound};
use super::js::{get, get_arr, get_str, js_string};
use super::model::DomainError;
use super::page::{page_result, parse_event_page};
use super::projection::Projection;
use super::replay::{read_event_range, replay_into, Range};
use super::security::{
    Authority, Bootstrap, ControlSecurity, Denied, RequestFacts, SecurityOptions,
};
use axum::body::{to_bytes, Body};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{ConnectInfo, FromRequestParts, State};
use axum::http::{header, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Router;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
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
    pub settings: FieldSettings,
    pub log: Mutex<EventLog>,
    pub projection: Arc<Mutex<Projection>>,
    pub security: Mutex<ControlSecurity>,
    pub hub: Arc<Hub>,
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
async fn read_json_body(body: Body) -> Result<Value, ApiError> {
    let bytes = match tokio::time::timeout(BODY_TIMEOUT, to_bytes(body, MAX_BODY_BYTES)).await {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(_)) => {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "body_too_large",
                "request body exceeds the allowed size",
            ))
        }
        Err(_) => {
            return Err(ApiError::new(
                StatusCode::REQUEST_TIMEOUT,
                "body_timeout",
                "request body timed out",
            ))
        }
    };
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
        let cookie: Option<String> = state.with_security(|s| {
            s.revoke_all_harness_tokens();
            s.revoke_browser_session()
        });
        state.hub.revoke_clients();
        let mut response = json_response(StatusCode::OK, &json!({ "ok": true, "abandoned": 0 }));
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
        "GET /api/config" => Ok(state.settings.view()),
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
            let mut historical = Projection::new(state.settings.projection_config());
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
                .workspace(&workspace_id)
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
                "label": state.settings.workspace(&capital_workspace_id).and_then(|w| get(w, "name")).cloned().unwrap_or(json!(capital_workspace_id)),
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
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        let _ = self.task.await;
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
    let state = Arc::new(AppState {
        settings,
        log: Mutex::new(log),
        projection,
        security: Mutex::new(security),
        hub,
        dist_dir: options.dist_dir.clone(),
        headers,
    });

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
        let mounted = state
            .settings
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
            state.settings.workspaces.len()
        );
        eprintln!(
            "  runtime       rust (knossos field --native); routes not yet ported answer 501"
        );
        for w in state
            .settings
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
