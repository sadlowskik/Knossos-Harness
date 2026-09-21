//! Live operational state as a fold over the event log. Port of
//! `field/server/src/store/projection.js`.
//!
//! Rebuilt by replaying every event on boot, so the running state and a
//! historical replay are produced by exactly the same code path. Synthetic
//! (rehearsal) events fold into a child projection and never touch real
//! state; `simulation.started` resets that child.

use super::budget_ledger::BudgetLedger;
use super::campaign_projection::CampaignProjection;
use super::eventlog::{Event, Source};
use super::graph_projection::GraphProjection;
use super::js::{
    assign, finite, get, get_arr, get_str, jnum, js_string, num, number, onum, or_null, set,
    truthy, Obj, OrderedMap,
};
use super::model::is_content_revision;
use super::replay::ApplyEvent;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const MAX_CONTEXT_TOKENS: f64 = 200_000.0;
const ACTIVITY_WINDOW_MS: i64 = 60 * 60 * 1000;
const ACTIVITY_EVENTS_FOR_FULL_SIGNAL: f64 = 20.0;

/// The parts of `field.yaml` the projection seeds itself from.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct FieldConfig {
    #[serde(default)]
    pub workspaces: Vec<WorkspaceConfig>,
    #[serde(default)]
    pub endpoints: Vec<EndpointConfig>,
    #[serde(default)]
    pub websites: Vec<WebsiteConfig>,
    #[serde(default)]
    pub routines: Vec<RoutineConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct WorkspaceConfig {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub mounted: Option<bool>,
    #[serde(default)]
    pub region: Option<Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EndpointConfig {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub cost_per_mtok: Option<Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct WebsiteConfig {
    pub domain: String,
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RoutineConfig {
    pub id: String,
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Totals {
    pub cost_usd: f64,
    pub input_tokens: f64,
    pub output_tokens: f64,
    pub sessions_spawned: i64,
}

#[derive(Debug, Clone)]
pub struct Projection {
    cfg: FieldConfig,
    partition_synthetic: bool,
    pub campaigns: CampaignProjection,
    pub graph: GraphProjection,
    pub budgets: BudgetLedger,
    synthetic: Option<Box<Projection>>,
    sessions: OrderedMap<Obj>,
    folders: OrderedMap<Obj>,
    files: OrderedMap<Obj>,
    websites: OrderedMap<Obj>,
    endpoints: OrderedMap<Obj>,
    assignments: OrderedMap<Obj>,
    permissions: OrderedMap<Obj>,
    workspaces: OrderedMap<Obj>,
    control_groups: Obj,
    positions: OrderedMap<Value>,
    routines: OrderedMap<Obj>,
    world: Obj,
    pub totals: Totals,
    last_seq: u64,
}

fn opt_str(value: &Option<String>) -> Value {
    value.clone().map(Value::String).unwrap_or(Value::Null)
}

impl Projection {
    pub fn new(cfg: FieldConfig) -> Self {
        Self::build(cfg, true)
    }

    fn build(cfg: FieldConfig, partition_synthetic: bool) -> Self {
        let mut p = Projection {
            cfg: cfg.clone(),
            partition_synthetic,
            campaigns: CampaignProjection::new(),
            graph: GraphProjection::new(),
            budgets: BudgetLedger::new(),
            synthetic: None,
            sessions: OrderedMap::new(),
            folders: OrderedMap::new(),
            files: OrderedMap::new(),
            websites: OrderedMap::new(),
            endpoints: OrderedMap::new(),
            assignments: OrderedMap::new(),
            permissions: OrderedMap::new(),
            workspaces: OrderedMap::new(),
            control_groups: Obj::new(),
            positions: OrderedMap::new(),
            routines: OrderedMap::new(),
            world: Obj::new(),
            totals: Totals::default(),
            last_seq: 0,
        };
        p.reset();
        if partition_synthetic {
            p.synthetic = Some(Box::new(Projection::build(cfg, false)));
        }
        p
    }

    pub fn reset(&mut self) {
        self.budgets = BudgetLedger::new();
        self.campaigns.reset();
        self.graph.reset();
        self.sessions.clear();
        self.folders.clear();
        self.files.clear();
        self.websites.clear();
        self.endpoints.clear();
        self.assignments.clear();
        self.permissions.clear();
        self.workspaces.clear();
        self.control_groups = Obj::new();
        self.positions.clear();
        self.routines.clear();
        for routine in &self.cfg.routines {
            let mut r = Obj::new();
            set(&mut r, "id", routine.id.clone());
            set(&mut r, "enabled", routine.enabled);
            set(&mut r, "lastRunAt", Value::Null);
            set(&mut r, "lastOutcome", Value::Null);
            set(&mut r, "activeRunId", Value::Null);
            set(&mut r, "activeRunIds", json!([]));
            set(&mut r, "currentOwner", Value::Null);
            set(&mut r, "queuedRuns", json!([]));
            set(&mut r, "cooldownUntil", 0);
            set(&mut r, "history", json!([]));
            self.routines.insert(routine.id.clone(), r);
        }
        self.world = json!({
            "capitalWorkspaceId": Value::Null,
            "capitalSelectedAt": Value::Null,
            "assignments": {},
            "revision": 0,
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        self.totals = Totals::default();
        self.last_seq = 0;
        if let Some(synthetic) = self.synthetic.as_mut() {
            synthetic.reset();
        }
        for w in &self.cfg.workspaces {
            let mut ws = Obj::new();
            set(&mut ws, "id", w.id.clone());
            assign(&mut ws, "name", w.name.clone().map(Value::String));
            assign(&mut ws, "path", w.path.clone().map(Value::String));
            assign(&mut ws, "mounted", w.mounted.map(Value::Bool));
            set(
                &mut ws,
                "region",
                w.region
                    .clone()
                    .unwrap_or(json!({ "x": 0, "y": 0, "w": 700, "h": 440 })),
            );
            set(&mut ws, "git", Value::Null);
            set(&mut ws, "changeCount", 0);
            set(&mut ws, "lastTs", 0);
            set(&mut ws, "activityEvents", json!([]));
            self.workspaces.insert(w.id.clone(), ws);
        }
        for e in &self.cfg.endpoints {
            let mut ep = Obj::new();
            set(&mut ep, "id", e.id.clone());
            assign(&mut ep, "name", e.name.clone().map(Value::String));
            assign(&mut ep, "kind", e.kind.clone().map(Value::String));
            assign(&mut ep, "model", e.model.clone().map(Value::String));
            set(&mut ep, "baseUrl", opt_str(&e.base_url));
            set(
                &mut ep,
                "costPerMtok",
                e.cost_per_mtok
                    .clone()
                    .unwrap_or(json!({ "input": 0, "output": 0 })),
            );
            set(&mut ep, "status", "unknown");
            set(&mut ep, "latencyMs", Value::Null);
            set(&mut ep, "lastCheck", 0);
            set(&mut ep, "detail", Value::Null);
            set(&mut ep, "failures", 0);
            self.endpoints.insert(e.id.clone(), ep);
        }
        for s in &self.cfg.websites {
            let mut site = Obj::new();
            set(&mut site, "domain", s.domain.clone());
            set(
                &mut site,
                "label",
                s.label.clone().unwrap_or_else(|| s.domain.clone()),
            );
            set(&mut site, "sessions", json!([]));
            set(&mut site, "lastUrl", Value::Null);
            set(&mut site, "lastTs", 0);
            set(&mut site, "hits", 0);
            self.websites.insert(s.domain.clone(), site);
        }
    }

    pub fn last_seq(&self) -> u64 {
        self.last_seq
    }

    pub fn session_record(&self, id: &str) -> Option<&Obj> {
        self.sessions.get(id)
    }

    pub fn world(&self) -> &Obj {
        &self.world
    }

    pub fn apply(&mut self, evt: &Event) {
        self.last_seq = self.last_seq.max(evt.seq);
        let synthetic = evt.source == Source::Synthetic
            || evt.data.get("simulated") == Some(&Value::Bool(true));
        if self.partition_synthetic && synthetic {
            if let Some(child) = self.synthetic.as_mut() {
                if evt.kind == "simulation.started" {
                    child.reset();
                }
                let mut relabelled = evt.clone();
                relabelled.source = Source::Synthetic;
                child.apply(&relabelled);
            }
            return;
        }
        self.campaigns.apply(evt);
        self.budgets.apply(evt);
        self.graph.apply(evt);
        self.handle(evt);
        if let Some(id) = get(&evt.data, "sessionId") {
            if let Some(s) = self.sessions.get_mut(&js_string(id)) {
                set(s, "lastEventTs", evt.ts);
            }
        }
    }

    fn session(&mut self, id: &str) -> &mut Obj {
        self.sessions.entry_or_insert_with(id, || {
            let mut s = Obj::new();
            set(&mut s, "id", id);
            set(&mut s, "agentId", Value::Null);
            set(&mut s, "name", id.chars().take(6).collect::<String>());
            set(&mut s, "role", "builder");
            set(&mut s, "model", Value::Null);
            set(&mut s, "endpointId", Value::Null);
            set(&mut s, "thinking", "medium");
            set(&mut s, "cwd", Value::Null);
            set(&mut s, "workspaceId", Value::Null);
            set(&mut s, "state", "spawning");
            set(&mut s, "parentId", Value::Null);
            set(&mut s, "children", json!([]));
            set(&mut s, "depth", 0);
            set(&mut s, "target", Value::Null);
            set(&mut s, "assignmentId", Value::Null);
            set(&mut s, "lastTool", Value::Null);
            set(&mut s, "toolCount", 0);
            set(&mut s, "editCount", 0);
            set(
                &mut s,
                "tokens",
                json!({ "input": 0, "output": 0, "cacheRead": 0 }),
            );
            set(&mut s, "contextPct", 0);
            set(&mut s, "costUsd", 0);
            set(&mut s, "startedAt", 0);
            set(&mut s, "endedAt", Value::Null);
            set(&mut s, "lastEventTs", 0);
            set(&mut s, "messageCount", 0);
            set(&mut s, "verified", "unverified");
            set(&mut s, "browser", Value::Null);
            set(&mut s, "error", Value::Null);
            set(&mut s, "progress", Value::Null);
            set(&mut s, "campaignId", Value::Null);
            set(&mut s, "team", Value::Null);
            set(&mut s, "objectiveId", Value::Null);
            set(&mut s, "touched", json!([]));
            s
        })
    }

    fn touch_folder(
        &mut self,
        workspace_id: &str,
        dir: &Value,
        ts: i64,
        session_id: Option<&Value>,
    ) {
        let key = format!("{workspace_id}:{}", js_string(dir));
        let folder = self.folders.entry_or_insert_with(&key, || {
            let mut f = Obj::new();
            set(&mut f, "key", key.clone());
            set(&mut f, "workspaceId", workspace_id);
            set(&mut f, "dir", dir.clone());
            set(&mut f, "hits", 0);
            set(&mut f, "lastTs", 0);
            set(&mut f, "agents", json!([]));
            f
        });
        let hits = onum(folder, "hits") + 1.0;
        set(folder, "hits", jnum(hits));
        let last = number(folder.get("lastTs"));
        set(folder, "lastTs", jnum(last.max(ts as f64)));
        if let Some(session_id) = session_id.filter(|s| truthy(s)) {
            if let Some(Value::Array(agents)) = folder.get_mut("agents") {
                if !agents.contains(session_id) {
                    agents.push(session_id.clone());
                }
            }
        }
    }

    /// The exact payload the WebSocket and `GET /api/state` return.
    pub fn snapshot(&self, now: i64) -> Value {
        let campaign = self.campaigns.snapshot();
        let campaign_views = campaign
            .get("campaigns")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let workspaces: Vec<Value> = self
            .workspaces
            .values()
            .map(|workspace| {
                let mut view = workspace.clone();
                view.remove("activityEvents");
                view.insert(
                    "maturity".into(),
                    workspace_maturity(workspace, &campaign_views, now),
                );
                Value::Object(view)
            })
            .collect();
        let sessions: Vec<Value> = self
            .sessions
            .values()
            .map(|s| {
                let mut view = s.clone();
                if let Some(Value::Array(touched)) = view.get_mut("touched") {
                    let keep = touched.len().saturating_sub(40);
                    touched.drain(..keep);
                }
                Value::Object(view)
            })
            .collect();
        let mut state = json!({
            "seq": self.last_seq,
            "now": now,
            "sessions": sessions,
            "workspaces": workspaces,
            "folders": self.folders.values().map(|f| Value::Object(f.clone())).collect::<Vec<_>>(),
            "files": self.files.values().map(|f| Value::Object(f.clone())).collect::<Vec<_>>(),
            "websites": self.websites.values().map(|s| Value::Object(s.clone())).collect::<Vec<_>>(),
            "endpoints": self.endpoints.values().map(|e| Value::Object(e.clone())).collect::<Vec<_>>(),
            "assignments": self.assignments.values().map(|a| Value::Object(a.clone())).collect::<Vec<_>>(),
            "permissions": self.permissions.values().filter(|p| p.get("status").and_then(Value::as_str) == Some("pending")).map(|p| Value::Object(p.clone())).collect::<Vec<_>>(),
            "controlGroups": Value::Object(self.control_groups.clone()),
            "positions": Value::Object(self.positions.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
            "world": Value::Object(self.world.clone()),
            "totals": {
                "costUsd": jnum(self.totals.cost_usd),
                "inputTokens": jnum(self.totals.input_tokens),
                "outputTokens": jnum(self.totals.output_tokens),
                "sessionsSpawned": self.totals.sessions_spawned,
            },
            "budgetReservations": self.budgets.snapshot(),
            "campaigns": campaign_views,
            "capabilities": campaign.get("capabilities").cloned().unwrap_or(json!([])),
            "checkpoints": campaign.get("checkpoints").cloned().unwrap_or(json!([])),
            "routines": self.routines.values().map(|routine| {
                let mut view = routine.clone();
                if let Some(Value::Array(history)) = view.get_mut("history") {
                    let keep = history.len().saturating_sub(50);
                    history.drain(..keep);
                }
                Value::Object(view)
            }).collect::<Vec<_>>(),
            // WebSocket snapshots are bounded; the complete graph remains
            // reconstructible from the event log.
            "graph": self.graph.snapshot(now, Some(4000), Some(8000)),
        });
        if let Some(synthetic) = &self.synthetic {
            state["rehearsal"] = synthetic.snapshot(now);
        }
        state
    }

    fn handle(&mut self, evt: &Event) {
        let d = evt.data.clone();
        let session_id = get(&d, "sessionId").map(js_string);
        match evt.kind.as_str() {
            "session.spawned" => {
                let Some(id) = session_id.clone() else {
                    return;
                };
                let parent = get(&d, "parentSessionId").map(js_string);
                {
                    let s = self.session(&id);
                    assign(s, "agentId", d.get("agentId").cloned());
                    let name = get(&d, "name").cloned().or_else(|| s.get("name").cloned());
                    assign(s, "name", name);
                    for key in [
                        "role",
                        "model",
                        "endpointId",
                        "thinking",
                        "cwd",
                        "workspaceId",
                    ] {
                        assign(s, key, d.get(key).cloned());
                    }
                    set(s, "parentId", or_null(d.get("parentSessionId")));
                    set(s, "state", "spawning");
                    set(s, "startedAt", evt.ts);
                    set(s, "target", or_null(d.get("target")));
                    set(s, "assignmentId", or_null(d.get("assignmentId")));
                    set(s, "campaignId", or_null(d.get("campaignId")));
                    set(s, "team", or_null(d.get("team")));
                    set(s, "objectiveId", or_null(d.get("objectiveId")));
                    set(s, "simulated", d.get("simulated").is_some_and(truthy));
                    set(s, "simulationRunId", or_null(d.get("simulationRunId")));
                }
                if let Some(parent_id) = parent.filter(|p| !p.is_empty()) {
                    let depth = {
                        let p = self.session(&parent_id);
                        if let Some(Value::Array(children)) = p.get_mut("children") {
                            if !children.iter().any(|c| c.as_str() == Some(&id)) {
                                children.push(json!(id));
                            }
                        }
                        number(p.get("depth"))
                    };
                    let s = self.session(&id);
                    set(s, "depth", jnum(depth + 1.0));
                }
                self.totals.sessions_spawned += 1;
            }
            "session.state" => {
                let Some(id) = session_id else {
                    return;
                };
                let s = self.session(&id);
                assign(s, "state", d.get("state").cloned());
                if let Some(detail) = d.get("detail") {
                    set(s, "stateDetail", detail.clone());
                }
            }
            "session.message" => {
                let Some(id) = session_id else {
                    return;
                };
                let s = self.session(&id);
                let count = onum(s, "messageCount") + 1.0;
                set(s, "messageCount", jnum(count));
                if get_str(&d, "role") == Some("assistant") {
                    if let Some(text) = get_str(&d, "text").filter(|t| !t.is_empty()) {
                        set(s, "lastSay", text.chars().take(400).collect::<String>());
                    }
                }
            }
            "session.progress" => {
                let Some(id) = session_id else {
                    return;
                };
                let s = self.session(&id);
                let bounded = |key: &str| {
                    let n = num(&d, key);
                    jnum(if n.is_nan() { 0.0 } else { n.max(0.0) })
                };
                set(
                    s,
                    "progress",
                    json!({
                        "done": bounded("done"),
                        "total": bounded("total"),
                        "steps": get_arr(&d, "steps").map(|steps| steps.iter().take(100).cloned().collect::<Vec<_>>()).unwrap_or_default(),
                    }),
                );
            }
            "session.tool_use" => {
                let Some(id) = session_id else {
                    return;
                };
                let workspace_id = get(&d, "workspaceId").map(js_string);
                {
                    let s = self.session(&id);
                    let count = onum(s, "toolCount") + 1.0;
                    set(s, "toolCount", jnum(count));
                    set(
                        s,
                        "lastTool",
                        json!({ "name": d.get("name"), "summary": get(&d, "summary").cloned().unwrap_or(json!("")), "ts": evt.ts }),
                    );
                    set(s, "state", "working");
                }
                if let (Some(ws), Some(dir)) = (
                    workspace_id.as_deref(),
                    d.get("dir").filter(|v| !v.is_null()),
                ) {
                    self.touch_folder(ws, dir, evt.ts, d.get("sessionId"));
                    if let Some(w) = self.workspaces.get_mut(ws) {
                        record_workspace_activity(w, evt);
                    }
                    // Reading or writing inside a workspace attaches the agent to that region.
                    let s = self.session(&id);
                    set(s, "workspaceId", ws);
                    set(s, "focusDir", dir.clone());
                }
                if let Some(path) = get(&d, "path").filter(|p| truthy(p)).cloned() {
                    // File-level position for the canvas Field: which file this
                    // unit is on right now. Additive to focusDir.
                    let name = get_str(&d, "name").unwrap_or("").to_string();
                    let s = self.session(&id);
                    set(s, "focusPath", path.clone());
                    if let Some(Value::Array(touched)) = s.get_mut("touched") {
                        touched.push(json!({ "path": path, "ts": evt.ts, "tool": d.get("name") }));
                    }
                    if matches!(name.as_str(), "Edit" | "Write" | "NotebookEdit") {
                        let edits = onum(s, "editCount") + 1.0;
                        set(s, "editCount", jnum(edits));
                    }
                }
            }
            "session.tool_result" => {
                let Some(id) = session_id else {
                    return;
                };
                if d.get("ok") == Some(&Value::Bool(false)) {
                    let preview = get(&d, "preview")
                        .map(js_string)
                        .unwrap_or_else(|| "tool failed".into());
                    let s = self.session(&id);
                    set(
                        s,
                        "lastError",
                        preview.chars().take(200).collect::<String>(),
                    );
                }
            }
            "session.usage" => {
                let Some(id) = session_id else {
                    return;
                };
                let s = self.session(&id);
                let cumulative = |value: Option<f64>, previous: f64| match value {
                    Some(v) if v >= 0.0 => previous.max(v),
                    _ => previous,
                };
                let previous_tokens = s.get("tokens").cloned().unwrap_or(json!({}));
                let prev_in = num(&previous_tokens, "input");
                let prev_out = num(&previous_tokens, "output");
                let prev_cache = num(&previous_tokens, "cacheRead");
                let input = cumulative(finite(&d, "inputTokens"), prev_in);
                let output = cumulative(finite(&d, "outputTokens"), prev_out);
                let cache = cumulative(finite(&d, "cacheRead"), prev_cache);
                set(
                    s,
                    "tokens",
                    json!({ "input": jnum(input), "output": jnum(output), "cacheRead": jnum(cache) }),
                );
                let used = match finite(&d, "contextTokens") {
                    Some(c) if c >= 0.0 => c,
                    _ => input + cache + output,
                };
                let pct = super::js::round((used / MAX_CONTEXT_TOKENS) * 100.0).clamp(0.0, 100.0);
                set(s, "contextPct", jnum(pct));
                let previous_cost = onum(s, "costUsd");
                let cost = cumulative(finite(&d, "costUsd"), previous_cost);
                set(s, "costUsd", jnum(cost));
                self.totals.cost_usd += cost - previous_cost;
                self.totals.input_tokens += input - prev_in;
                self.totals.output_tokens += output - prev_out;
            }
            "session.ended" => {
                let Some(id) = session_id else {
                    return;
                };
                let s = self.session(&id);
                let state = match get_str(&d, "reason") {
                    Some("error") => "error",
                    Some("cancelled") => "cancelled",
                    _ => "done",
                };
                set(s, "state", state);
                set(s, "endedAt", evt.ts);
                set(s, "error", or_null(d.get("error")));
                set(s, "result", or_null(d.get("result")));
            }
            "session.delegated" => {
                // A subagent created by the agent's own Task tool runs inside
                // the harness process. Field records that the delegation
                // happened, but does not fabricate a session it cannot observe.
                let Some(parent_id) = get(&d, "parentSessionId").map(js_string) else {
                    return;
                };
                let child = or_null(d.get("childSessionId"));
                let p = self.session(&parent_id);
                let delegations = p
                    .entry("delegations".to_string())
                    .or_insert_with(|| json!([]));
                if let Value::Array(items) = delegations {
                    if !items.iter().any(|x| x.get("id") == Some(&child)) {
                        items.push(json!({
                            "id": child,
                            "description": or_null(d.get("description")),
                            "type": or_null(d.get("subagentType")),
                            "ts": evt.ts,
                        }));
                    }
                }
            }
            "assignment.created" => {
                let assignment_id = get(&d, "assignmentId")
                    .map(js_string)
                    .unwrap_or_else(|| "undefined".into());
                let session_ids = get_arr(&d, "sessionIds").cloned().unwrap_or_default();
                let mut a = Obj::new();
                set(&mut a, "id", assignment_id.clone());
                assign(&mut a, "sessionIds", d.get("sessionIds").cloned());
                for key in ["targetType", "targetId", "targetLabel", "orders"] {
                    assign(&mut a, key, d.get(key).cloned());
                }
                set(&mut a, "workspaceId", or_null(d.get("workspaceId")));
                set(&mut a, "status", "active");
                set(&mut a, "createdAt", evt.ts);
                self.assignments.insert(assignment_id.clone(), a);
                for id in session_ids {
                    let s = self.session(&js_string(&id));
                    set(s, "assignmentId", assignment_id.clone());
                    set(
                        s,
                        "target",
                        json!({
                            "type": d.get("targetType"), "id": d.get("targetId"),
                            "label": d.get("targetLabel"), "workspaceId": d.get("workspaceId"),
                        }),
                    );
                    if let Some(ws) = get(&d, "workspaceId").filter(|w| truthy(w)) {
                        set(s, "workspaceId", ws.clone());
                    }
                }
            }
            "assignment.completed" | "assignment.failed" | "assignment.interrupted" => {
                let status = evt.kind.trim_start_matches("assignment.");
                if let Some(a) = self
                    .assignments
                    .get_mut(&get(&d, "assignmentId").map(js_string).unwrap_or_default())
                {
                    set(a, "status", status);
                    set(a, "reason", or_null(d.get("reason")));
                    if status == "failed" {
                        set(a, "failedSessionId", or_null(d.get("sessionId")));
                    }
                    set(a, "endedAt", evt.ts);
                }
            }
            "assignment.cancelled" => {
                let key = get(&d, "assignmentId").map(js_string).unwrap_or_default();
                let mut session_ids = Vec::new();
                if let Some(a) = self.assignments.get_mut(&key) {
                    set(a, "status", "cancelled");
                    set(a, "reason", or_null(d.get("reason")));
                    set(a, "endedAt", evt.ts);
                    session_ids = a
                        .get("sessionIds")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                }
                for id in session_ids {
                    if let Some(s) = self.sessions.get_mut(&js_string(&id)) {
                        set(s, "target", Value::Null);
                    }
                }
            }
            "fs.changed" => {
                let ws_text = d
                    .get("workspaceId")
                    .map(js_string)
                    .unwrap_or_else(|| "undefined".into());
                let key = format!(
                    "{ws_text}:{}",
                    d.get("path")
                        .map(js_string)
                        .unwrap_or_else(|| "undefined".into())
                );
                let mut f = Obj::new();
                set(&mut f, "key", key.clone());
                for k in ["workspaceId", "path", "dir", "change"] {
                    assign(&mut f, k, d.get(k).cloned());
                }
                set(&mut f, "lastTs", evt.ts);
                set(&mut f, "bySession", or_null(d.get("sessionId")));
                self.files.insert(key, f);
                let dir = d.get("dir").cloned().unwrap_or(Value::Null);
                self.touch_folder(&ws_text, &dir, evt.ts, d.get("sessionId"));
                if let Some(w) = self.workspaces.get_mut(&ws_text) {
                    let count = onum(w, "changeCount") + 1.0;
                    set(w, "changeCount", jnum(count));
                    record_workspace_activity(w, evt);
                }
            }
            "git.status" => {
                if let Some(w) =
                    get(&d, "workspaceId").and_then(|id| self.workspaces.get_mut(&js_string(id)))
                {
                    set(
                        w,
                        "git",
                        json!({ "branch": d.get("branch"), "ahead": d.get("ahead"), "behind": d.get("behind"), "files": d.get("files") }),
                    );
                }
            }
            "browser.navigated" => {
                let Some(id) = session_id else {
                    return;
                };
                let domain = d
                    .get("domain")
                    .map(js_string)
                    .unwrap_or_else(|| "undefined".into());
                {
                    let s = self.session(&id);
                    set(
                        s,
                        "browser",
                        json!({ "url": d.get("url"), "domain": d.get("domain"), "ts": evt.ts }),
                    );
                }
                let site = self.websites.entry_or_insert_with(&domain, || {
                    let mut site = Obj::new();
                    set(
                        &mut site,
                        "domain",
                        d.get("domain").cloned().unwrap_or(Value::Null),
                    );
                    set(
                        &mut site,
                        "label",
                        d.get("domain").cloned().unwrap_or(Value::Null),
                    );
                    set(&mut site, "sessions", json!([]));
                    set(&mut site, "lastUrl", Value::Null);
                    set(&mut site, "lastTs", 0);
                    set(&mut site, "hits", 0);
                    set(&mut site, "discovered", true);
                    site
                });
                if let Some(Value::Array(sessions)) = site.get_mut("sessions") {
                    if !sessions.iter().any(|s| s.as_str() == Some(&id)) {
                        sessions.push(json!(id));
                    }
                }
                set(site, "lastUrl", or_null(d.get("url")));
                set(site, "lastTs", evt.ts);
                let hits = onum(site, "hits") + 1.0;
                set(site, "hits", jnum(hits));
            }
            "browser.closed" => {
                let Some(id) = session_id else {
                    return;
                };
                if let Some(s) = self.sessions.get_mut(&id) {
                    set(s, "browser", Value::Null);
                }
                for site in self.websites.values_mut() {
                    if let Some(Value::Array(sessions)) = site.get_mut("sessions") {
                        sessions.retain(|s| s.as_str() != Some(&id));
                    }
                }
            }
            "endpoint.health" => {
                let Some(e) =
                    get(&d, "endpointId").and_then(|id| self.endpoints.get_mut(&js_string(id)))
                else {
                    return;
                };
                assign(e, "status", d.get("status").cloned());
                set(e, "latencyMs", or_null(d.get("latencyMs")));
                set(e, "detail", or_null(d.get("detail")));
                set(e, "lastCheck", evt.ts);
                if get_str(&d, "status") == Some("down") {
                    let failures = onum(e, "failures") + 1.0;
                    set(e, "failures", jnum(failures));
                } else {
                    set(e, "failures", 0);
                }
            }
            "endpoint.routed" => {
                let Some(id) = session_id else {
                    return;
                };
                let s = self.session(&id);
                assign(s, "endpointId", d.get("endpointId").cloned());
                let model = get(&d, "model")
                    .cloned()
                    .or_else(|| s.get("model").cloned());
                assign(s, "model", model);
                set(s, "rerouted", or_null(d.get("reason")));
            }
            "permission.requested" => {
                let permission_id = get(&d, "permissionId")
                    .map(js_string)
                    .unwrap_or_else(|| "undefined".into());
                let mut p = Obj::new();
                set(&mut p, "id", permission_id.clone());
                for key in ["sessionId", "toolName", "input"] {
                    assign(&mut p, key, d.get(key).cloned());
                }
                set(&mut p, "status", "pending");
                set(&mut p, "requestedAt", evt.ts);
                set(&mut p, "decidedAt", Value::Null);
                self.permissions.insert(permission_id.clone(), p);
                let Some(id) = session_id else {
                    return;
                };
                let s = self.session(&id);
                set(s, "state", "waiting_permission");
                set(s, "pendingPermission", permission_id);
            }
            "permission.decided" => {
                let key = get(&d, "permissionId").map(js_string).unwrap_or_default();
                let mut owner: Option<String> = None;
                if let Some(p) = self.permissions.get_mut(&key) {
                    assign(p, "status", d.get("decision").cloned());
                    set(p, "decidedAt", evt.ts);
                    owner = p.get("sessionId").filter(|v| !v.is_null()).map(js_string);
                }
                if let Some(id) = owner.or(session_id) {
                    if let Some(s) = self.sessions.get_mut(&id) {
                        set(s, "pendingPermission", Value::Null);
                        if s.get("state").and_then(Value::as_str) == Some("waiting_permission") {
                            set(s, "state", "working");
                        }
                    }
                }
            }
            "work.verified" => {
                if let Some(id) = session_id {
                    if let Some(s) = self.sessions.get_mut(&id) {
                        set(
                            s,
                            "verified",
                            if get_str(&d, "result") == Some("verified") {
                                "verified"
                            } else {
                                "rejected"
                            },
                        );
                    }
                }
                if let Some(a) =
                    get(&d, "assignmentId").and_then(|id| self.assignments.get_mut(&js_string(id)))
                {
                    assign(a, "verified", d.get("result").cloned());
                }
            }
            "ui.position" => {
                let key = format!(
                    "{}:{}",
                    d.get("entityType")
                        .map(js_string)
                        .unwrap_or_else(|| "undefined".into()),
                    d.get("entityId")
                        .map(js_string)
                        .unwrap_or_else(|| "undefined".into())
                );
                self.positions
                    .insert(key, json!({ "x": d.get("x"), "y": d.get("y") }));
            }
            "world.capital_selected" => {
                assign(
                    &mut self.world,
                    "capitalWorkspaceId",
                    d.get("workspaceId").cloned(),
                );
                set(&mut self.world, "capitalSelectedAt", evt.ts);
                bump_revision(&mut self.world);
            }
            "world.territory_assigned" => {
                let cluster_key = d
                    .get("clusterKey")
                    .map(js_string)
                    .unwrap_or_else(|| "undefined".into());
                let label = get(&d, "label")
                    .filter(|l| truthy(l))
                    .cloned()
                    .unwrap_or(json!(cluster_key));
                let kind = get(&d, "kind")
                    .filter(|k| truthy(k))
                    .cloned()
                    .unwrap_or(json!("infrastructure"));
                let workspace_id = get(&d, "workspaceId")
                    .filter(|w| truthy(w))
                    .cloned()
                    .unwrap_or(Value::Null);
                if let Some(Value::Object(assignments)) = self.world.get_mut("assignments") {
                    assignments.insert(
                        cluster_key,
                        json!({
                            "clusterKey": d.get("clusterKey"), "territoryId": d.get("territoryId"),
                            "label": label, "kind": kind, "workspaceId": workspace_id, "assignedAt": evt.ts,
                        }),
                    );
                }
                bump_revision(&mut self.world);
            }
            "world.territory_released" => {
                let cluster_key = d
                    .get("clusterKey")
                    .map(js_string)
                    .unwrap_or_else(|| "undefined".into());
                if let Some(Value::Object(assignments)) = self.world.get_mut("assignments") {
                    assignments.remove(&cluster_key);
                }
                bump_revision(&mut self.world);
            }
            "ui.control_group" => {
                let group = d
                    .get("group")
                    .map(js_string)
                    .unwrap_or_else(|| "undefined".into());
                assign(
                    &mut self.control_groups,
                    &group,
                    d.get("sessionIds").cloned(),
                );
            }
            "routine.schedule_claimed" => {
                let key = get(&d, "routineId").map(js_string).unwrap_or_default();
                if let Some(routine) = self.routines.get_mut(&key) {
                    if let Some(slot) = get_str(&d, "slot") {
                        let last = routine
                            .get("lastScheduleSlot")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        if slot > last {
                            set(routine, "lastScheduleSlot", slot);
                        }
                    }
                }
            }
            "routine.triggered" => {
                let routine_id = d.get("routineId").cloned();
                let session_ids = get_arr(&d, "sessionIds").cloned().unwrap_or_default();
                for id in &session_ids {
                    let s = self.session(&js_string(id));
                    assign(s, "routineId", routine_id.clone());
                }
                let key = routine_id.as_ref().map(js_string).unwrap_or_default();
                if let Some(routine) = self.routines.get_mut(&key) {
                    let last_run = get(&d, "startedAt").cloned().unwrap_or(json!(evt.ts));
                    set(routine, "lastRunAt", last_run.clone());
                    set(routine, "activeRunId", or_null(d.get("runId")));
                    if let Some(run_id) = get(&d, "runId").filter(|r| truthy(r)).cloned() {
                        let mut ids = routine
                            .get("activeRunIds")
                            .and_then(Value::as_array)
                            .cloned()
                            .unwrap_or_default();
                        if !ids.contains(&run_id) {
                            ids.push(run_id);
                        }
                        set(routine, "activeRunIds", Value::Array(ids));
                    }
                    set(
                        routine,
                        "currentOwner",
                        session_ids.first().cloned().unwrap_or(Value::Null),
                    );
                    if let Some(Value::Array(queued)) = routine.get_mut("queuedRuns") {
                        queued.retain(|item| item.get("runId") != d.get("runId"));
                    }
                    if let Some(Value::Array(history)) = routine.get_mut("history") {
                        history.push(json!({
                            "runId": or_null(d.get("runId")), "status": "running", "at": last_run, "reason": or_null(d.get("reason")),
                        }));
                    }
                }
            }
            "routine.enabled" => {
                if let Some(routine) = self
                    .routines
                    .get_mut(&get(&d, "routineId").map(js_string).unwrap_or_default())
                {
                    set(routine, "enabled", d.get("enabled").is_some_and(truthy));
                }
            }
            "routine.queued" => {
                let Some(run_id) = get(&d, "runId").filter(|r| truthy(r)).cloned() else {
                    return;
                };
                if let Some(routine) = self
                    .routines
                    .get_mut(&get(&d, "routineId").map(js_string).unwrap_or_default())
                {
                    if let Some(Value::Array(queued)) = routine.get_mut("queuedRuns") {
                        if !queued.iter().any(|item| item.get("runId") == Some(&run_id)) {
                            queued.push(json!({
                                "runId": run_id, "at": get(&d, "queuedAt").cloned().unwrap_or(json!(evt.ts)),
                                "reason": or_null(d.get("reason")), "paths": get(&d, "paths").cloned().unwrap_or(json!([])),
                                "authorityHash": or_null(d.get("authorityHash")), "claimed": false,
                            }));
                            let keep = queued.len().saturating_sub(1);
                            queued.drain(..keep);
                        }
                    }
                }
            }
            "routine.queue_claimed" => {
                if let Some(routine) = self
                    .routines
                    .get_mut(&get(&d, "routineId").map(js_string).unwrap_or_default())
                {
                    if let Some(Value::Array(queued)) = routine.get_mut("queuedRuns") {
                        if let Some(Value::Object(item)) = queued
                            .iter_mut()
                            .find(|item| item.get("runId") == d.get("runId"))
                        {
                            set(item, "claimed", true);
                        }
                    }
                }
            }
            "routine.cooldown_started" => {
                if let Some(routine) = self
                    .routines
                    .get_mut(&get(&d, "routineId").map(js_string).unwrap_or_default())
                {
                    let until = num(&d, "until");
                    set(
                        routine,
                        "cooldownUntil",
                        jnum(if until.is_nan() { 0.0 } else { until }),
                    );
                }
            }
            "routine.completed" => {
                if let Some(routine) = self
                    .routines
                    .get_mut(&get(&d, "routineId").map(js_string).unwrap_or_default())
                {
                    settle_routine(routine, &d, evt, "completed");
                }
            }
            "routine.failed" => {
                if let Some(routine) = self
                    .routines
                    .get_mut(&get(&d, "routineId").map(js_string).unwrap_or_default())
                {
                    settle_routine(routine, &d, evt, "failed");
                }
            }
            "routine.skipped" => {
                if let Some(routine) = self
                    .routines
                    .get_mut(&get(&d, "routineId").map(js_string).unwrap_or_default())
                {
                    if let Some(run_id) = get(&d, "runId").filter(|r| truthy(r)) {
                        if let Some(Value::Array(queued)) = routine.get_mut("queuedRuns") {
                            queued.retain(|item| item.get("runId") != Some(run_id));
                        }
                    }
                    if let Some(Value::Array(history)) = routine.get_mut("history") {
                        history.push(json!({ "runId": or_null(d.get("runId")), "status": "skipped", "at": evt.ts, "reason": or_null(d.get("reason")) }));
                    }
                }
            }
            _ => {}
        }
    }
}

impl ApplyEvent for Projection {
    fn apply(&mut self, event: &Event) {
        Projection::apply(self, event)
    }
}

fn bump_revision(world: &mut Obj) {
    let revision = onum(world, "revision") + 1.0;
    set(world, "revision", jnum(revision));
}

fn settle_routine(routine: &mut Obj, data: &Value, event: &Event, status: &str) {
    set(routine, "lastOutcome", status);
    let run_id = data.get("runId").cloned();
    let mut active: Vec<Value> = routine
        .get("activeRunIds")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    active.retain(|id| Some(id) != run_id.as_ref());
    set(routine, "activeRunIds", Value::Array(active.clone()));
    let current = routine.get("activeRunId").cloned().unwrap_or(Value::Null);
    if !truthy(&current) || Some(&current) == run_id.as_ref() {
        let next = active.last().cloned().unwrap_or(Value::Null);
        set(routine, "activeRunId", next.clone());
        if !truthy(&next) {
            set(routine, "currentOwner", Value::Null);
        }
    }
    if let Some(cooldown) = finite(data, "cooldownUntil") {
        set(routine, "cooldownUntil", jnum(cooldown));
    }
    if let Some(Value::Array(queued)) = routine.get_mut("queuedRuns") {
        queued.retain(|item| item.get("runId") != run_id.as_ref());
    }
    if let Some(Value::Array(history)) = routine.get_mut("history") {
        let existing = run_id
            .as_ref()
            .filter(|r| truthy(r))
            .and_then(|r| history.iter_mut().find(|item| item.get("runId") == Some(r)));
        match existing {
            Some(Value::Object(row)) => {
                set(row, "status", status);
                set(row, "endedAt", event.ts);
                set(row, "reason", or_null(data.get("reason")));
            }
            _ => history.push(json!({
                "runId": or_null(data.get("runId")), "status": status, "at": event.ts, "endedAt": event.ts,
                "reason": or_null(data.get("reason")),
            })),
        }
    }
}

fn record_workspace_activity(workspace: &mut Obj, event: &Event) {
    if event.source != Source::Observed {
        return;
    }
    set(workspace, "lastTs", event.ts);
    if let Some(Value::Array(activity)) = workspace.get_mut("activityEvents") {
        activity.push(json!({ "seq": event.seq, "ts": event.ts, "kind": event.kind, "subject": event.subject }));
        if activity.len() > 200 {
            let drop = activity.len() - 200;
            activity.drain(..drop);
        }
    }
}

fn campaign_targets_workspace(campaign: &Value, workspace_id: &str) -> bool {
    let mut targets: Vec<&Value> = Vec::new();
    if let Some(t) = get(campaign, "target") {
        targets.push(t);
    }
    for objective in get_arr(campaign, "objectives")
        .map(|o| o.as_slice())
        .unwrap_or(&[])
    {
        if let Some(t) = get(objective, "target") {
            targets.push(t);
        }
    }
    targets.iter().any(|t| {
        get_str(t, "workspaceId") == Some(workspace_id)
            || (get_str(t, "type") == Some("workspace") && get_str(t, "id") == Some(workspace_id))
    })
}

pub fn workspace_maturity(workspace: &Obj, campaigns: &[Value], now: i64) -> Value {
    let workspace_id = workspace.get("id").map(js_string).unwrap_or_default();
    let activity: Vec<&Value> = workspace
        .get("activityEvents")
        .and_then(Value::as_array)
        .map(|events| {
            events
                .iter()
                .filter(|e| {
                    now - e
                        .get("ts")
                        .map(|t| num(&json!({ "t": t }), "t"))
                        .unwrap_or(0.0) as i64
                        <= ACTIVITY_WINDOW_MS
                })
                .collect()
        })
        .unwrap_or_default();
    let relevant: Vec<&Value> = campaigns
        .iter()
        .filter(|c| campaign_targets_workspace(c, &workspace_id))
        .collect();
    struct Criterion<'a> {
        campaign: &'a Value,
        objective: &'a Value,
        criterion: &'a Value,
    }
    let mut criteria: Vec<Criterion<'_>> = Vec::new();
    for campaign in &relevant {
        for objective in get_arr(campaign, "objectives")
            .map(|o| o.as_slice())
            .unwrap_or(&[])
        {
            if objective.get("required") == Some(&Value::Bool(false)) {
                continue;
            }
            for criterion in get_arr(objective, "definitionOfDone")
                .map(|c| c.as_slice())
                .unwrap_or(&[])
            {
                criteria.push(Criterion {
                    campaign,
                    objective,
                    criterion,
                });
            }
        }
    }
    let completed: Vec<&Criterion<'_>> = criteria
        .iter()
        .filter(|c| {
            get_str(c.objective, "status") == Some("satisfied")
                && get_arr(c.objective, "criteriaEvidence").is_some_and(|items| {
                    items.iter().any(|item| {
                        item.get("criterion") == Some(c.criterion)
                            && item.get("evidence").is_some_and(truthy)
                    })
                })
        })
        .collect();
    let last_verdict = |campaign: &Value| {
        get_arr(campaign, "verdicts")
            .and_then(|v| v.last())
            .cloned()
    };
    let verified: Vec<&Criterion<'_>> = completed
        .iter()
        .copied()
        .filter(|c| {
            matches!(get_str(c.campaign, "phase"), Some("verified" | "promoted"))
                && last_verdict(c.campaign)
                    .as_ref()
                    .and_then(|v| get_str(v, "verdict"))
                    == Some("verified")
        })
        .collect();
    let content_checkpoint = |campaign: &Value| -> Option<Value> {
        get_arr(campaign, "checkpoints")?
            .iter()
            .find(|cp| cp.get("revision").is_some_and(is_content_revision))
            .cloned()
    };
    let persisted: Vec<&Criterion<'_>> = verified
        .iter()
        .copied()
        .filter(|c| content_checkpoint(c.campaign).is_some())
        .collect();
    let terminal: Vec<&Value> = relevant
        .iter()
        .copied()
        .filter(|c| {
            matches!(
                get_str(c, "phase"),
                Some("verified" | "promoted" | "failed" | "cancelled" | "rolled_back")
            )
        })
        .collect();
    let reliable = terminal
        .iter()
        .filter(|c| matches!(get_str(c, "phase"), Some("verified" | "promoted")))
        .count();
    let ratio = |count: usize| -> f64 {
        if criteria.is_empty() {
            0.0
        } else {
            super::js::round((count as f64 / criteria.len() as f64) * 100.0)
        }
    };
    let complete = ratio(completed.len());
    let verification = ratio(verified.len());
    let persistence = ratio(persisted.len());
    let score = super::js::round(complete * 0.3 + verification * 0.4 + persistence * 0.3);
    let tier = if score >= 85.0 {
        4
    } else if score >= 62.0 {
        3
    } else if score >= 34.0 {
        2
    } else if score > 0.0 {
        1
    } else {
        0
    };
    let mut evidence: Vec<Value> = Vec::new();
    let recent = activity.len().saturating_sub(5);
    for event in &activity[recent..] {
        let mut item = json!({ "type": "activity" });
        if let (Value::Object(out), Value::Object(src)) = (&mut item, *event) {
            for (k, v) in src {
                out.insert(k.clone(), v.clone());
            }
        }
        evidence.push(item);
    }
    for c in &completed {
        evidence.push(json!({
            "type": "completion", "campaignId": c.campaign.get("id"), "objectiveId": c.objective.get("id"), "criterion": c.criterion,
        }));
    }
    for c in &verified {
        evidence.push(json!({
            "type": "verification", "campaignId": c.campaign.get("id"),
            "verdictId": last_verdict(c.campaign).and_then(|v| v.get("id").cloned()),
        }));
    }
    for c in &persisted {
        evidence.push(json!({
            "type": "persistence", "campaignId": c.campaign.get("id"),
            "checkpointId": content_checkpoint(c.campaign).and_then(|cp| cp.get("id").cloned()),
        }));
    }
    let keep = evidence.len().saturating_sub(40);
    evidence.drain(..keep);
    json!({
        "score": jnum(score),
        "tier": tier,
        "activity": jnum((super::js::round((activity.len() as f64 / ACTIVITY_EVENTS_FOR_FULL_SIGNAL) * 100.0)).min(100.0)),
        "complete": jnum(complete),
        "verified": jnum(verification),
        "persisted": jnum(persistence),
        "reliability": jnum(if terminal.is_empty() { 0.0 } else { super::js::round((reliable as f64 / terminal.len() as f64) * 100.0) }),
        "evidence": evidence,
    })
}
