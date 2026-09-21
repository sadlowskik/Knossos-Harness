//! Typed topology folded from events. Port of
//! `field/server/src/orchestration/graph-projection.js`.
//!
//! Nodes and edges are observed, never declared: every event that mentions
//! an agent, folder, file, endpoint or campaign artefact strengthens it. World
//! nodes (folders, websites, services, endpoints, workspaces) go through a
//! lifecycle with hysteresis, so a place an agent worked at stays visible for
//! a while after the last touch and fades to dormant rather than vanishing.

use super::eventlog::Event;
use super::js::{get, get_arr, get_str, jnum, js_string, Obj, OrderedMap};
use serde_json::{json, Value};

const WORLD_TYPES: [&str; 5] = ["folder", "website", "service", "endpoint", "workspace"];
const ACTIVE_STATES: [&str; 3] = ["active_worksite", "established", "promoted"];

fn is_world(node_type: &str) -> bool {
    WORLD_TYPES.contains(&node_type)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub key: String,
    pub node_type: String,
    pub id: String,
    pub label: String,
    pub first_seen: i64,
    pub last_seen: i64,
    pub observations: i64,
    pub first_seq: u64,
    pub last_seq: u64,
    pub state: String,
    pub attrs: Obj,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub key: String,
    pub edge_type: String,
    pub from: String,
    pub to: String,
    pub first_seen: i64,
    pub last_seen: i64,
    pub observations: i64,
    pub first_seq: u64,
    pub last_seq: u64,
    pub active: bool,
    pub attrs: Obj,
}

#[derive(Debug, Clone, Default)]
pub struct GraphProjection {
    nodes: OrderedMap<Node>,
    edges: OrderedMap<Edge>,
}

pub fn node_key(node_type: &str, id: &str) -> String {
    format!("{node_type}:{id}")
}

/// Attribute set for an observation: `undefined` fields are simply not
/// inserted, `null` ones are.
fn attrs(pairs: &[(&str, Option<&Value>)]) -> Obj {
    let mut out = Obj::new();
    for (key, value) in pairs {
        if let Some(v) = value {
            out.insert((*key).to_string(), (*v).clone());
        }
    }
    out
}

fn lifecycle(node: &Node, now: i64) -> &'static str {
    if node.state == "promoted" {
        return "promoted";
    }
    let age = (now - node.last_seen).max(0);
    let span = (node.last_seen - node.first_seen).max(0);
    if age > 30 * 60_000 {
        return "dormant";
    }
    if node.observations >= 8 || (node.observations >= 4 && span >= 5 * 60_000) {
        return "established";
    }
    if node.observations >= 3 || span >= 60_000 {
        return "active_worksite";
    }
    "contact"
}

fn node_priority(node_type: &str, state: &str, last_seen: i64, observations: i64, now: i64) -> i64 {
    if state == "promoted" {
        return 1_000_000;
    }
    let active = !is_world(node_type) || ACTIVE_STATES.contains(&state);
    let recent = (100_000 - ((now - last_seen) as f64 / 1000.0).floor() as i64).max(0);
    (if active { 500_000 } else { 0 }) + recent + observations.min(10_000)
}

impl GraphProjection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.edges.clear();
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn node(&self, key: &str) -> Option<&Node> {
        self.nodes.get(key)
    }

    pub fn apply(&mut self, evt: &Event) {
        let d = &evt.data;
        let s = |key: &str| get(d, key);
        let campaign = s("campaignId");
        let session = s("sessionId");
        match evt.kind.as_str() {
            "campaign.created" => {
                self.observe_node(
                    "campaign",
                    campaign,
                    attrs(&[("label", s("name")), ("phase", Some(&json!("draft")))]),
                    evt,
                    1,
                );
                if let Some(ws) = d.get("target").and_then(|t| get(t, "workspaceId")) {
                    self.observe_node("workspace", Some(ws), attrs(&[("label", Some(ws))]), evt, 1);
                    self.observe_edge(
                        "working_on",
                        Some(node_key(
                            "campaign",
                            &js_string(campaign.unwrap_or(&Value::Null)),
                        )),
                        Some(node_key("workspace", &js_string(ws))),
                        Obj::new(),
                        evt,
                        1,
                    );
                }
            }
            "campaign.phase_changed" => {
                self.observe_node("campaign", campaign, attrs(&[("phase", s("to"))]), evt, 1);
            }
            "objective.created" => {
                let objective = s("objectiveId");
                self.observe_node(
                    "objective",
                    objective,
                    attrs(&[
                        ("label", s("statement")),
                        ("campaignId", campaign),
                        ("status", Some(&json!("queued"))),
                    ]),
                    evt,
                    1,
                );
                self.observe_edge(
                    "depends_on",
                    key_of("objective", objective),
                    key_of("campaign", campaign),
                    Obj::new(),
                    evt,
                    1,
                );
                if let Some(ws) = d.get("target").and_then(|t| get(t, "workspaceId")) {
                    self.observe_node("workspace", Some(ws), attrs(&[("label", Some(ws))]), evt, 1);
                    self.observe_edge(
                        "working_on",
                        key_of("objective", objective),
                        Some(node_key("workspace", &js_string(ws))),
                        Obj::new(),
                        evt,
                        1,
                    );
                }
            }
            "objective.assigned" => {
                let objective = s("objectiveId");
                self.observe_node(
                    "objective",
                    objective,
                    attrs(&[("status", Some(&json!("active"))), ("team", s("team"))]),
                    evt,
                    1,
                );
                for id in get_arr(d, "sessionIds").cloned().unwrap_or_default() {
                    self.observe_edge(
                        "assigned_to",
                        Some(node_key("agent", &js_string(&id))),
                        key_of("objective", objective),
                        attrs(&[("team", s("team"))]),
                        evt,
                        1,
                    );
                }
            }
            "objective.satisfied" => {
                self.observe_node(
                    "objective",
                    s("objectiveId"),
                    attrs(&[("status", Some(&json!("satisfied")))]),
                    evt,
                    1,
                );
            }
            "objective.blocked" => {
                self.observe_node(
                    "objective",
                    s("objectiveId"),
                    attrs(&[("status", Some(&json!("blocked")))]),
                    evt,
                    1,
                );
            }
            "team.member_assigned" => {
                let team_id = json!(format!(
                    "{}:{}",
                    js_string(campaign.unwrap_or(&Value::Null)),
                    js_string(s("team").unwrap_or(&Value::Null))
                ));
                self.observe_node(
                    "team",
                    Some(&team_id),
                    attrs(&[
                        ("label", s("team")),
                        ("campaignId", campaign),
                        ("kind", s("team")),
                    ]),
                    evt,
                    1,
                );
                let label = s("agentId").or(session);
                self.observe_node(
                    "agent",
                    session,
                    attrs(&[("label", label), ("role", s("role")), ("team", s("team"))]),
                    evt,
                    1,
                );
                self.observe_edge(
                    "member_of",
                    key_of("agent", session),
                    Some(node_key("team", &js_string(&team_id))),
                    attrs(&[("role", s("role"))]),
                    evt,
                    1,
                );
                self.observe_edge(
                    "member_of",
                    Some(node_key("team", &js_string(&team_id))),
                    key_of("campaign", campaign),
                    Obj::new(),
                    evt,
                    1,
                );
            }
            "session.spawned" => {
                let label = s("name").or(s("agentId")).or(session);
                self.observe_node(
                    "agent",
                    session,
                    attrs(&[
                        ("label", label),
                        ("role", s("role")),
                        ("state", Some(&json!("spawning"))),
                        ("campaignId", campaign),
                        ("team", s("team")),
                    ]),
                    evt,
                    1,
                );
                if let Some(ws) = s("workspaceId") {
                    self.observe_node("workspace", Some(ws), attrs(&[("label", Some(ws))]), evt, 1);
                    self.observe_edge(
                        "located_at",
                        key_of("agent", session),
                        Some(node_key("workspace", &js_string(ws))),
                        Obj::new(),
                        evt,
                        1,
                    );
                }
                if let Some(endpoint) = s("endpointId") {
                    self.observe_node(
                        "endpoint",
                        Some(endpoint),
                        attrs(&[("label", Some(endpoint)), ("model", s("model"))]),
                        evt,
                        1,
                    );
                    self.observe_edge(
                        "routed_through",
                        key_of("agent", session),
                        Some(node_key("endpoint", &js_string(endpoint))),
                        Obj::new(),
                        evt,
                        1,
                    );
                }
            }
            "session.state" => {
                self.observe_node("agent", session, attrs(&[("state", s("state"))]), evt, 1);
            }
            "session.tool_use" => {
                self.observe_node("agent", session, Obj::new(), evt, 1);
                let name = s("name");
                self.observe_node("tool", name, attrs(&[("label", name)]), evt, 1);
                self.observe_edge(
                    "used",
                    key_of("agent", session),
                    key_of("tool", name),
                    Obj::new(),
                    evt,
                    1,
                );
                let workspace = s("workspaceId");
                if let (Some(ws), Some(dir)) = (workspace, d.get("dir").filter(|v| !v.is_null())) {
                    let id = json!(format!("{}:{}", js_string(ws), js_string(dir)));
                    let dir_label = if super::js::truthy(dir) {
                        dir.clone()
                    } else {
                        json!("/")
                    };
                    self.observe_node(
                        "folder",
                        Some(&id),
                        attrs(&[
                            ("label", Some(&dir_label)),
                            ("workspaceId", Some(ws)),
                            ("path", Some(dir)),
                        ]),
                        evt,
                        1,
                    );
                    self.observe_edge(
                        "working_on",
                        key_of("agent", session),
                        Some(node_key("folder", &js_string(&id))),
                        Obj::new(),
                        evt,
                        1,
                    );
                    self.observe_edge(
                        "located_at",
                        Some(node_key("folder", &js_string(&id))),
                        Some(node_key("workspace", &js_string(ws))),
                        Obj::new(),
                        evt,
                        1,
                    );
                }
                if let Some(path) = s("path").filter(|p| super::js::truthy(p)) {
                    let ws_text = workspace
                        .map(js_string)
                        .unwrap_or_else(|| "external".to_string());
                    let id = json!(format!("{ws_text}:{}", js_string(path)));
                    self.observe_node(
                        "file",
                        Some(&id),
                        attrs(&[
                            ("label", Some(path)),
                            ("workspaceId", workspace),
                            ("path", Some(path)),
                        ]),
                        evt,
                        1,
                    );
                    self.observe_edge(
                        "working_on",
                        key_of("agent", session),
                        Some(node_key("file", &js_string(&id))),
                        Obj::new(),
                        evt,
                        1,
                    );
                }
            }
            "session.delegated" => {
                let parent = s("parentSessionId");
                let child = s("childSessionId");
                self.observe_node("agent", parent, Obj::new(), evt, 1);
                let label = s("description").or(child);
                self.observe_node(
                    "agent",
                    child,
                    attrs(&[("label", label), ("delegated", Some(&json!(true)))]),
                    evt,
                    1,
                );
                self.observe_edge(
                    "communicates_with",
                    key_of("agent", parent),
                    key_of("agent", child),
                    attrs(&[("delegation", Some(&json!(true)))]),
                    evt,
                    1,
                );
            }
            "agent.communication" => {
                let from = s("fromSessionId");
                let to = s("toSessionId");
                self.observe_node("agent", from, Obj::new(), evt, 1);
                self.observe_node("agent", to, Obj::new(), evt, 1);
                self.observe_edge(
                    "communicates_with",
                    key_of("agent", from),
                    key_of("agent", to),
                    attrs(&[("channel", s("channel"))]),
                    evt,
                    1,
                );
            }
            "browser.navigated" => {
                let domain = s("domain");
                self.observe_node(
                    "website",
                    domain,
                    attrs(&[("label", domain), ("url", s("url"))]),
                    evt,
                    1,
                );
                self.observe_edge(
                    "visited",
                    key_of("agent", session),
                    key_of("website", domain),
                    attrs(&[("url", s("url"))]),
                    evt,
                    1,
                );
            }
            "endpoint.health" => {
                let endpoint = s("endpointId");
                self.observe_node(
                    "endpoint",
                    endpoint,
                    attrs(&[
                        ("label", endpoint),
                        ("status", s("status")),
                        ("latencyMs", s("latencyMs")),
                    ]),
                    evt,
                    1,
                );
            }
            "endpoint.routed" => {
                let endpoint = s("endpointId");
                self.observe_node(
                    "endpoint",
                    endpoint,
                    attrs(&[("label", endpoint), ("model", s("model"))]),
                    evt,
                    1,
                );
                self.observe_edge(
                    "routed_through",
                    key_of("agent", session),
                    key_of("endpoint", endpoint),
                    attrs(&[("reason", s("reason"))]),
                    evt,
                    1,
                );
            }
            "finding.reported" => {
                let author = s("authorSessionId");
                let finding = s("findingId");
                self.observe_node(
                    "agent",
                    author,
                    attrs(&[("team", Some(&json!("red")))]),
                    evt,
                    1,
                );
                self.observe_node(
                    "finding",
                    finding,
                    attrs(&[
                        ("label", s("claim")),
                        ("severity", s("severity")),
                        ("status", Some(&json!("open"))),
                    ]),
                    evt,
                    1,
                );
                self.observe_edge(
                    "found",
                    key_of("agent", author),
                    key_of("finding", finding),
                    Obj::new(),
                    evt,
                    1,
                );
                if let Some(objective) = s("objectiveId") {
                    self.observe_edge(
                        "challenged_by",
                        Some(node_key("objective", &js_string(objective))),
                        key_of("finding", finding),
                        Obj::new(),
                        evt,
                        1,
                    );
                }
            }
            "mitigation.proposed" => {
                let mitigation = s("mitigationId");
                self.observe_node(
                    "mitigation",
                    mitigation,
                    attrs(&[("label", s("claim")), ("status", Some(&json!("proposed")))]),
                    evt,
                    1,
                );
                for id in get_arr(d, "findingIds").cloned().unwrap_or_default() {
                    self.observe_edge(
                        "mitigated_by",
                        Some(node_key("finding", &js_string(&id))),
                        key_of("mitigation", mitigation),
                        Obj::new(),
                        evt,
                        1,
                    );
                }
            }
            "retest.completed" => {
                self.observe_node(
                    "agent",
                    session,
                    attrs(&[("team", Some(&json!("red")))]),
                    evt,
                    1,
                );
                self.observe_edge(
                    "retested_by",
                    key_of("finding", s("findingId")),
                    key_of("agent", session),
                    attrs(&[("result", s("result"))]),
                    evt,
                    1,
                );
            }
            "referee.verdict" => {
                let verdict = s("verdictId");
                self.observe_node(
                    "agent",
                    session,
                    attrs(&[("team", Some(&json!("referee")))]),
                    evt,
                    1,
                );
                self.observe_node(
                    "verdict",
                    verdict,
                    attrs(&[("label", s("verdict")), ("verdict", s("verdict"))]),
                    evt,
                    1,
                );
                self.observe_edge(
                    "verified_by",
                    key_of("campaign", campaign),
                    key_of("verdict", verdict),
                    Obj::new(),
                    evt,
                    1,
                );
                self.observe_edge(
                    "produced",
                    key_of("agent", session),
                    key_of("verdict", verdict),
                    Obj::new(),
                    evt,
                    1,
                );
            }
            "campaign.checkpoint_created" => {
                let checkpoint = s("checkpointId");
                self.observe_node(
                    "checkpoint",
                    checkpoint,
                    attrs(&[("label", s("name")), ("campaignId", campaign)]),
                    evt,
                    1,
                );
                self.observe_edge(
                    "produced",
                    key_of("campaign", campaign),
                    key_of("checkpoint", checkpoint),
                    Obj::new(),
                    evt,
                    1,
                );
            }
            "campaign.promoted" => {
                for capability in get_arr(d, "capabilities").cloned().unwrap_or_default() {
                    let id = get(&capability, "id");
                    let label = get(&capability, "name").or(id);
                    self.promote("capability", id, evt, attrs(&[("label", label)]));
                    self.observe_edge(
                        "promoted_into",
                        key_of("campaign", campaign),
                        key_of("capability", id),
                        attrs(&[("checkpointId", s("checkpointId"))]),
                        evt,
                        1,
                    );
                }
            }
            _ => {}
        }
    }

    pub fn observe_node(
        &mut self,
        node_type: &str,
        id: Option<&Value>,
        attrs: Obj,
        evt: &Event,
        weight: i64,
    ) -> Option<&mut Node> {
        let id = id?;
        let id_text = js_string(id);
        if id.is_null() || id_text.is_empty() {
            return None;
        }
        let key = node_key(node_type, &id_text);
        let label_attr = attrs
            .get("label")
            .filter(|l| super::js::truthy(l))
            .map(js_string);
        let node = self.nodes.entry_or_insert_with(&key, || Node {
            key: key.clone(),
            node_type: node_type.to_string(),
            id: id_text.clone(),
            label: attrs
                .get("label")
                .map(js_string)
                .unwrap_or_else(|| id_text.clone()),
            first_seen: evt.ts,
            last_seen: evt.ts,
            observations: 0,
            first_seq: evt.seq,
            last_seq: evt.seq,
            state: if is_world(node_type) {
                "contact"
            } else {
                "active"
            }
            .to_string(),
            attrs: Obj::new(),
        });
        node.last_seen = node.last_seen.max(evt.ts);
        node.last_seq = node.last_seq.max(evt.seq);
        node.observations += weight;
        for (k, v) in attrs {
            node.attrs.insert(k, v);
        }
        if let Some(label) = label_attr {
            node.label = label;
        }
        if is_world(node_type) {
            node.state = lifecycle(node, evt.ts).to_string();
        }
        Some(node)
    }

    pub fn observe_edge(
        &mut self,
        edge_type: &str,
        from: Option<String>,
        to: Option<String>,
        attrs: Obj,
        evt: &Event,
        weight: i64,
    ) -> Option<&mut Edge> {
        let (from, to) = (from?, to?);
        if from.is_empty() || to.is_empty() {
            return None;
        }
        let key = format!("{edge_type}:{from}->{to}");
        let edge = self.edges.entry_or_insert_with(&key, || Edge {
            key: key.clone(),
            edge_type: edge_type.to_string(),
            from: from.clone(),
            to: to.clone(),
            first_seen: evt.ts,
            last_seen: evt.ts,
            observations: 0,
            first_seq: evt.seq,
            last_seq: evt.seq,
            active: true,
            attrs: Obj::new(),
        });
        edge.last_seen = edge.last_seen.max(evt.ts);
        edge.last_seq = edge.last_seq.max(evt.seq);
        edge.observations += weight;
        for (k, v) in attrs {
            edge.attrs.insert(k, v);
        }
        Some(edge)
    }

    pub fn promote(&mut self, node_type: &str, id: Option<&Value>, evt: &Event, attrs: Obj) {
        if let Some(node) = self.observe_node(node_type, id, attrs, evt, 1) {
            node.state = "promoted".to_string();
        }
    }

    /// Bounded view: at most `max_nodes` and `max_edges`, the most relevant
    /// first, with lifecycle states computed for `now`.
    pub fn snapshot(&self, now: i64, max_nodes: Option<usize>, max_edges: Option<usize>) -> Value {
        struct View<'a> {
            node: &'a Node,
            state: String,
        }
        let mut all: Vec<View<'_>> = self
            .nodes
            .values()
            .map(|node| View {
                state: if is_world(&node.node_type) && node.state != "promoted" {
                    lifecycle(node, now).to_string()
                } else {
                    node.state.clone()
                },
                node,
            })
            .collect();
        let total_nodes = all.len();
        if let Some(max) = max_nodes.filter(|m| total_nodes > *m) {
            all.sort_by(|a, b| {
                let pa = node_priority(
                    &a.node.node_type,
                    &a.state,
                    a.node.last_seen,
                    a.node.observations,
                    now,
                );
                let pb = node_priority(
                    &b.node.node_type,
                    &b.state,
                    b.node.last_seen,
                    b.node.observations,
                    now,
                );
                pb.cmp(&pa).then(b.node.last_seen.cmp(&a.node.last_seen))
            });
            all.truncate(max);
        }
        let visible: Vec<&str> = all
            .iter()
            .filter(|v| !is_world(&v.node.node_type) || ACTIVE_STATES.contains(&v.state.as_str()))
            .map(|v| v.node.key.as_str())
            .collect();
        let visible_set: std::collections::HashSet<&str> = visible.iter().copied().collect();
        struct EdgeView<'a> {
            edge: &'a Edge,
            active: bool,
        }
        let mut edges: Vec<EdgeView<'_>> = self
            .edges
            .values()
            .filter(|e| {
                visible_set.contains(e.from.as_str()) && visible_set.contains(e.to.as_str())
            })
            .map(|edge| EdgeView {
                active: now - edge.last_seen < 30 * 60_000,
                edge,
            })
            .collect();
        if let Some(max) = max_edges.filter(|m| edges.len() > *m) {
            edges.sort_by(|a, b| {
                (b.active as u8)
                    .cmp(&(a.active as u8))
                    .then(b.edge.last_seen.cmp(&a.edge.last_seen))
                    .then(b.edge.observations.cmp(&a.edge.observations))
            });
            edges.truncate(max);
        }
        let nodes_json: Vec<Value> = all
            .iter()
            .map(|v| {
                json!({
                    "key": v.node.key, "type": v.node.node_type, "id": v.node.id, "label": v.node.label,
                    "firstSeen": v.node.first_seen, "lastSeen": v.node.last_seen,
                    "observations": v.node.observations, "firstSeq": v.node.first_seq, "lastSeq": v.node.last_seq,
                    "state": v.state, "attrs": v.node.attrs,
                })
            })
            .collect();
        let edges_json: Vec<Value> = edges
            .iter()
            .map(|e| {
                json!({
                    "key": e.edge.key, "type": e.edge.edge_type, "from": e.edge.from, "to": e.edge.to,
                    "firstSeen": e.edge.first_seen, "lastSeen": e.edge.last_seen, "observations": e.edge.observations,
                    "firstSeq": e.edge.first_seq, "lastSeq": e.edge.last_seq, "active": e.active, "attrs": e.edge.attrs,
                })
            })
            .collect();
        let truncated = nodes_json.len() < total_nodes || edges_json.len() < self.edges.len();
        json!({
            "nodes": nodes_json,
            "edges": edges_json,
            "visibleNodeKeys": visible,
            "totals": { "nodes": jnum(total_nodes as f64), "edges": jnum(self.edges.len() as f64) },
            "truncated": truncated,
        })
    }
}

fn key_of(node_type: &str, id: Option<&Value>) -> Option<String> {
    id.map(|v| node_key(node_type, &js_string(v)))
}

#[allow(dead_code)]
fn _unused(_: &str) -> Option<&str> {
    get_str(&Value::Null, "")
}
