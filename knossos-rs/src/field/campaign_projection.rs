//! Campaign state folded from events. Port of
//! `field/server/src/orchestration/campaign-projection.js`, every handler
//! including the ones nothing emits yet.

use super::eventlog::Event;
use super::js::{
    assign, finite, get, get_arr, get_str, is_truthy, jnum, js_string, or_null, set, Obj,
    OrderedMap,
};
use super::model::{legal_actions, transition_options, Context};
use serde_json::{json, Value};

#[derive(Debug, Clone, Default)]
pub struct CampaignProjection {
    pub campaigns: OrderedMap<Obj>,
    pub objectives: OrderedMap<Obj>,
    pub findings: OrderedMap<Obj>,
    pub mitigations: OrderedMap<Obj>,
    pub verdicts: OrderedMap<Obj>,
    pub checkpoints: OrderedMap<Obj>,
    pub capabilities: OrderedMap<Obj>,
    /// Idempotency ledger for command replay: `commandId -> { seq, campaignId }`.
    pub command_results: OrderedMap<Value>,
    session_costs: std::collections::HashMap<String, f64>,
}

fn empty_teams() -> Value {
    let team = |kind: &str| json!({ "kind": kind, "members": [], "ready": false });
    json!({ "blue": team("blue"), "red": team("red"), "purple": team("purple"), "referee": team("referee") })
}

fn id_of(d: &Value, key: &str) -> String {
    d.get(key)
        .map(js_string)
        .unwrap_or_else(|| "undefined".to_string())
}

fn push_unique(obj: &mut Obj, key: &str, value: Value) {
    let list = obj.entry(key.to_string()).or_insert_with(|| json!([]));
    if let Value::Array(items) = list {
        if !items.contains(&value) {
            items.push(value);
        }
    }
}

fn push(obj: &mut Obj, key: &str, value: Value) {
    let list = obj.entry(key.to_string()).or_insert_with(|| json!([]));
    if let Value::Array(items) = list {
        items.push(value);
    }
}

fn members_mut(team: &mut Value) -> Option<&mut Vec<Value>> {
    team.get_mut("members").and_then(Value::as_array_mut)
}

fn team_ready(team: &mut Value) {
    let ready = members_mut(team).is_some_and(|members| {
        members
            .iter()
            .any(|m| matches!(get_str(m, "status"), Some("active" | "external")))
    });
    if let Value::Object(t) = team {
        t.insert("ready".into(), Value::Bool(ready));
    }
}

fn find_member<'a>(members: &'a mut [Value], session_id: &str) -> Option<&'a mut Value> {
    members
        .iter_mut()
        .find(|m| get_str(m, "sessionId") == Some(session_id))
}

impl CampaignProjection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    fn team_mut(&mut self, campaign_id: &str, team: &str) -> Option<&mut Value> {
        self.campaigns
            .get_mut(campaign_id)?
            .get_mut("teams")?
            .get_mut(team)
    }

    pub fn apply(&mut self, evt: &Event) {
        let d = &evt.data;
        self.handle(evt);
        if let Some(command_id) = get(d, "commandId") {
            self.command_results.insert(
                js_string(command_id),
                json!({ "seq": evt.seq, "campaignId": or_null(d.get("campaignId")) }),
            );
        }
        if let Some(campaign) =
            get(d, "campaignId").and_then(|id| self.campaigns.get_mut(&js_string(id)))
        {
            set(campaign, "updatedAt", evt.ts);
            set(campaign, "lastSeq", evt.seq);
        }
    }

    pub fn context(&self, campaign_id: &str) -> Option<Context> {
        let campaign = self.campaigns.get(campaign_id)?;
        let collect = |key: &str, map: &OrderedMap<Obj>| -> Vec<Value> {
            campaign
                .get(key)
                .and_then(Value::as_array)
                .map(|ids| {
                    ids.iter()
                        .filter_map(|id| map.get(&js_string(id)))
                        .map(|o| Value::Object(o.clone()))
                        .collect()
                })
                .unwrap_or_default()
        };
        let objectives = collect("objectiveIds", &self.objectives);
        let findings = collect("findingIds", &self.findings);
        let verdicts = collect("verdictIds", &self.verdicts);
        let checkpoints = collect("checkpointIds", &self.checkpoints);
        Some(Context {
            latest_verdict: verdicts
                .last()
                .and_then(|v| get_str(v, "verdict"))
                .map(str::to_string),
            checkpoint_id: checkpoints
                .last()
                .and_then(|c| get_str(c, "id"))
                .map(str::to_string),
            checkpoint_revision: checkpoints.last().and_then(|c| get(c, "revision")).cloned(),
            objectives,
            findings,
            verdicts,
            checkpoints,
        })
    }

    pub fn campaign_view(&self, campaign: &Obj) -> Value {
        let id = campaign.get("id").map(js_string).unwrap_or_default();
        let ctx = self.context(&id).unwrap_or_default();
        let collect = |key: &str, map: &OrderedMap<Obj>| -> Vec<Value> {
            campaign
                .get(key)
                .and_then(Value::as_array)
                .map(|ids| {
                    ids.iter()
                        .filter_map(|id| map.get(&js_string(id)))
                        .map(|o| Value::Object(o.clone()))
                        .collect()
                })
                .unwrap_or_default()
        };
        let campaign_value = Value::Object(campaign.clone());
        let mut view = campaign.clone();
        view.insert("objectives".into(), Value::Array(ctx.objectives.clone()));
        view.insert("findings".into(), Value::Array(ctx.findings.clone()));
        view.insert(
            "mitigations".into(),
            Value::Array(collect("mitigationIds", &self.mitigations)),
        );
        view.insert("verdicts".into(), Value::Array(ctx.verdicts.clone()));
        view.insert("checkpoints".into(), Value::Array(ctx.checkpoints.clone()));
        view.insert(
            "capabilities".into(),
            Value::Array(collect("capabilityIds", &self.capabilities)),
        );
        view.insert(
            "legalActions".into(),
            json!(legal_actions(&campaign_value, &ctx)),
        );
        view.insert(
            "transitionOptions".into(),
            Value::Array(transition_options(&campaign_value, &ctx)),
        );
        Value::Object(view)
    }

    pub fn snapshot(&self) -> Value {
        json!({
            "campaigns": self.campaigns.values().map(|c| self.campaign_view(c)).collect::<Vec<_>>(),
            "capabilities": self.capabilities.values().map(|c| Value::Object(c.clone())).collect::<Vec<_>>(),
            "checkpoints": self.checkpoints.values().map(|c| Value::Object(c.clone())).collect::<Vec<_>>(),
        })
    }

    fn handle(&mut self, evt: &Event) {
        let d = &evt.data;
        let campaign_id = id_of(d, "campaignId");
        match evt.kind.as_str() {
            "campaign.created" => {
                if self.campaigns.contains(&campaign_id) {
                    return;
                }
                let mut c = Obj::new();
                set(&mut c, "id", campaign_id.clone());
                assign(&mut c, "name", d.get("name").cloned());
                assign(&mut c, "intent", d.get("intent").cloned());
                assign(&mut c, "scope", d.get("scope").cloned());
                assign(&mut c, "concurrency", d.get("concurrency").cloned());
                assign(&mut c, "budgetUsd", d.get("budgetUsd").cloned());
                assign(&mut c, "doctrine", d.get("doctrine").cloned());
                set(&mut c, "target", or_null(d.get("target")));
                set(&mut c, "phase", "draft");
                set(&mut c, "paused", false);
                set(&mut c, "createdAt", evt.ts);
                set(&mut c, "updatedAt", evt.ts);
                set(&mut c, "lastSeq", evt.seq);
                set(&mut c, "teams", empty_teams());
                for key in [
                    "objectiveIds",
                    "findingIds",
                    "mitigationIds",
                    "verdictIds",
                    "checkpointIds",
                    "capabilityIds",
                    "staffingGaps",
                ] {
                    set(&mut c, key, json!([]));
                }
                set(&mut c, "costUsd", 0);
                set(&mut c, "budgetExhausted", false);
                set(&mut c, "failure", Value::Null);
                self.campaigns.insert(campaign_id, c);
            }
            "campaign.phase_changed" => {
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    assign(c, "phase", d.get("to").cloned());
                    set(c, "phaseReason", or_null(d.get("reason")));
                }
            }
            "campaign.paused" => {
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    set(c, "paused", true);
                }
            }
            "campaign.resumed" => {
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    set(c, "paused", false);
                }
            }
            "campaign.cancelled" => {
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    set(c, "phase", "cancelled");
                    set(c, "paused", false);
                    set(c, "cancelReason", or_null(d.get("reason")));
                }
            }
            "campaign.failed" => {
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    set(c, "phase", "failed");
                    set(
                        c,
                        "failure",
                        get(d, "reason")
                            .cloned()
                            .unwrap_or(json!("campaign failed")),
                    );
                }
            }
            "campaign.doctrine_changed" => {
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    let mut doctrine = c
                        .get("doctrine")
                        .and_then(Value::as_object)
                        .cloned()
                        .unwrap_or_default();
                    if let Some(Value::Object(patch)) = d.get("doctrine") {
                        for (k, v) in patch {
                            doctrine.insert(k.clone(), v.clone());
                        }
                    }
                    set(c, "doctrine", Value::Object(doctrine));
                }
            }
            "objective.created" => {
                let objective_id = id_of(d, "objectiveId");
                if self.objectives.contains(&objective_id) {
                    return;
                }
                let mut o = Obj::new();
                set(&mut o, "id", objective_id.clone());
                assign(&mut o, "campaignId", d.get("campaignId").cloned());
                assign(&mut o, "statement", d.get("statement").cloned());
                set(
                    &mut o,
                    "definitionOfDone",
                    get(d, "definitionOfDone").cloned().unwrap_or(json!([])),
                );
                assign(&mut o, "priority", d.get("priority").cloned());
                assign(&mut o, "risk", d.get("risk").cloned());
                set(&mut o, "target", or_null(d.get("target")));
                set(
                    &mut o,
                    "dependencies",
                    get(d, "dependencies").cloned().unwrap_or(json!([])),
                );
                set(
                    &mut o,
                    "required",
                    d.get("required") != Some(&Value::Bool(false)),
                );
                set(&mut o, "status", "queued");
                set(&mut o, "team", "blue");
                set(&mut o, "sessionIds", json!([]));
                set(&mut o, "progress", Value::Null);
                set(&mut o, "evidence", json!([]));
                set(&mut o, "blockedReason", Value::Null);
                set(&mut o, "createdAt", evt.ts);
                set(&mut o, "updatedAt", evt.ts);
                set(&mut o, "criteriaEvidence", json!([]));
                self.objectives.insert(objective_id.clone(), o);
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    push_unique(c, "objectiveIds", json!(objective_id));
                }
            }
            "objective.assigned" => {
                let Some(o) = self.objectives.get_mut(&id_of(d, "objectiveId")) else {
                    return;
                };
                if let Some(team) = get(d, "team") {
                    set(o, "team", team.clone());
                }
                let mut ids: Vec<Value> = o
                    .get("sessionIds")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                for id in get_arr(d, "sessionIds").cloned().unwrap_or_default() {
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
                set(o, "sessionIds", Value::Array(ids));
                set(o, "status", "active");
                set(o, "updatedAt", evt.ts);
            }
            "objective.progress" => {
                let Some(o) = self.objectives.get_mut(&id_of(d, "objectiveId")) else {
                    return;
                };
                set(o, "progress", or_null(d.get("progress")));
                if let Some(evidence) = d.get("evidence").filter(|e| super::js::truthy(e)) {
                    push(o, "evidence", evidence.clone());
                }
                set(o, "updatedAt", evt.ts);
            }
            "objective.blocked" => {
                if let Some(o) = self.objectives.get_mut(&id_of(d, "objectiveId")) {
                    set(o, "status", "blocked");
                    assign(o, "blockedReason", d.get("reason").cloned());
                    set(o, "updatedAt", evt.ts);
                }
            }
            "objective.satisfied" => {
                if let Some(o) = self.objectives.get_mut(&id_of(d, "objectiveId")) {
                    set(o, "status", "satisfied");
                    let evidence = get_arr(d, "evidence").cloned().unwrap_or_default();
                    for item in &evidence {
                        push(o, "evidence", item.clone());
                    }
                    let criteria = match get_arr(d, "criteriaEvidence") {
                        Some(items) => Value::Array(items.clone()),
                        None => {
                            let done = o
                                .get("definitionOfDone")
                                .and_then(Value::as_array)
                                .cloned()
                                .unwrap_or_default();
                            Value::Array(
                                done.iter()
                                    .enumerate()
                                    .filter_map(|(index, criterion)| {
                                        let evidence = d.get("evidence").and_then(|e| e.get(index));
                                        evidence.filter(|e| super::js::truthy(e)).map(
                                            |e| json!({ "criterion": criterion, "evidence": e }),
                                        )
                                    })
                                    .collect(),
                            )
                        }
                    };
                    set(o, "criteriaEvidence", criteria);
                    set(o, "updatedAt", evt.ts);
                }
            }
            "objective.failed" => {
                if let Some(o) = self.objectives.get_mut(&id_of(d, "objectiveId")) {
                    set(o, "status", "failed");
                    assign(o, "blockedReason", d.get("reason").cloned());
                    set(o, "updatedAt", evt.ts);
                }
            }
            "team.member_assigned" => {
                let team_name = id_of(d, "team");
                let session_id = id_of(d, "sessionId");
                let Some(team) = self.team_mut(&campaign_id, &team_name) else {
                    return;
                };
                if let Some(members) = members_mut(team) {
                    if find_member(members, &session_id).is_none() {
                        members.push(json!({
                            "sessionId": or_null(d.get("sessionId")),
                            "agentId": or_null(d.get("agentId")),
                            "role": or_null(d.get("role")),
                            "objectiveId": or_null(d.get("objectiveId")),
                            "status": get(d, "status").cloned().unwrap_or(json!("active")),
                            "external": d.get("external") == Some(&Value::Bool(true)),
                            "assignedAt": or_null(d.get("assignedAt")),
                        }));
                    }
                }
                team_ready(team);
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    if let Some(Value::Array(gaps)) = c.get_mut("staffingGaps") {
                        gaps.retain(|g| {
                            !(g.get("team") == d.get("team") && g.get("role") == d.get("role"))
                        });
                    }
                }
            }
            "team.member_removed" => {
                let Some(team) = self.team_mut(&campaign_id, &id_of(d, "team")) else {
                    return;
                };
                if let Some(members) = members_mut(team) {
                    members.retain(|m| m.get("sessionId") != d.get("sessionId"));
                }
                team_ready(team);
            }
            "team.member_state" => {
                let session_id = id_of(d, "sessionId");
                let Some(team) = self.team_mut(&campaign_id, &id_of(d, "team")) else {
                    return;
                };
                let Some(member) = members_mut(team).and_then(|m| find_member(m, &session_id))
                else {
                    return;
                };
                if let Value::Object(m) = member {
                    assign(m, "status", d.get("status").cloned());
                    let session_state = get(d, "sessionState")
                        .cloned()
                        .or_else(|| m.get("sessionState").cloned())
                        .unwrap_or(Value::Null);
                    set(m, "sessionState", session_state);
                    let external = get_str(d, "status") == Some("external")
                        || m.get("external") == Some(&Value::Bool(true));
                    set(m, "external", external);
                }
                team_ready(team);
            }
            "team.staffing_gap" => {
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    let exists =
                        c.get("staffingGaps")
                            .and_then(Value::as_array)
                            .is_some_and(|gaps| {
                                gaps.iter().any(|g| {
                                    g.get("team") == d.get("team") && g.get("role") == d.get("role")
                                })
                            });
                    if !exists {
                        push(
                            c,
                            "staffingGaps",
                            json!({ "team": d.get("team"), "role": d.get("role"), "reason": d.get("reason") }),
                        );
                    }
                }
            }
            "team.handoff_started" => {
                let from = id_of(d, "fromSessionId");
                let team_name = id_of(d, "team");
                if let Some(Value::Object(m)) = self
                    .team_mut(&campaign_id, &team_name)
                    .and_then(members_mut)
                    .and_then(|members| find_member(members, &from))
                {
                    set(m, "status", "handing_off");
                }
            }
            "team.handoff_completed" => {
                let from = id_of(d, "fromSessionId");
                let to = id_of(d, "toSessionId");
                if let Some(team) = self.team_mut(&campaign_id, &id_of(d, "team")) {
                    if let Some(members) = members_mut(team) {
                        if let Some(Value::Object(m)) = find_member(members, &from) {
                            set(m, "status", "retired");
                        }
                        if let Some(Value::Object(m)) = find_member(members, &to) {
                            set(m, "status", "active");
                        }
                    }
                }
            }
            "session.state" => {
                let state = get(d, "state").map(js_string);
                self.synchronize_session_state(get_str(d, "sessionId"), state.as_deref(), evt);
            }
            "session.ended" => {
                let state = if get_str(d, "reason") == Some("error") {
                    "error"
                } else {
                    "done"
                };
                self.synchronize_session_state(get_str(d, "sessionId"), Some(state), evt);
            }
            "session.usage" => {
                let Some(cost) = finite(d, "costUsd").filter(|c| *c >= 0.0) else {
                    return;
                };
                if get(d, "campaignId").is_none() {
                    return;
                }
                let Some(c) = self.campaigns.get_mut(&campaign_id) else {
                    return;
                };
                let key = format!("{campaign_id}:{}", id_of(d, "sessionId"));
                let prior = self.session_costs.get(&key).copied().unwrap_or(0.0);
                let current = prior.max(cost);
                self.session_costs.insert(key, current);
                let cost_usd = super::js::number(c.get("costUsd")) + (current - prior);
                set(c, "costUsd", jnum(cost_usd));
                let budget = super::js::number(c.get("budgetUsd"));
                set(c, "budgetExhausted", budget > 0.0 && cost_usd >= budget);
            }
            "finding.reported" => {
                let finding_id = id_of(d, "findingId");
                if self.findings.contains(&finding_id) {
                    return;
                }
                let mut f = Obj::new();
                set(&mut f, "id", finding_id.clone());
                for key in [
                    "campaignId",
                    "objectiveId",
                    "authorSessionId",
                    "category",
                    "severity",
                    "claim",
                    "scope",
                    "confidence",
                ] {
                    assign(&mut f, key, d.get(key).cloned());
                }
                set(
                    &mut f,
                    "evidence",
                    get(d, "evidence").cloned().unwrap_or(json!([])),
                );
                set(
                    &mut f,
                    "reproduction",
                    get(d, "reproduction").cloned().unwrap_or(json!([])),
                );
                set(
                    &mut f,
                    "noReproductionReason",
                    or_null(d.get("noReproductionReason")),
                );
                set(&mut f, "status", "open");
                set(&mut f, "mitigationIds", json!([]));
                set(&mut f, "retests", json!([]));
                set(&mut f, "createdAt", evt.ts);
                set(&mut f, "updatedAt", evt.ts);
                self.findings.insert(finding_id.clone(), f);
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    push_unique(c, "findingIds", json!(finding_id));
                }
            }
            "finding.acknowledged" => {
                self.set_finding_state(&id_of(d, "findingId"), "acknowledged", evt, &[])
            }
            "finding.disputed" => self.set_finding_state(
                &id_of(d, "findingId"),
                "disputed",
                evt,
                &[("dispute", d.get("evidence"))],
            ),
            "finding.waived" => self.set_finding_state(
                &id_of(d, "findingId"),
                "waived",
                evt,
                &[
                    ("waiver", d.get("reason")),
                    ("authority", d.get("authority")),
                ],
            ),
            "mitigation.proposed" => {
                let mitigation_id = id_of(d, "mitigationId");
                if self.mitigations.contains(&mitigation_id) {
                    return;
                }
                let finding_ids = get_arr(d, "findingIds").cloned().unwrap_or_default();
                let mut m = Obj::new();
                set(&mut m, "id", mitigation_id.clone());
                assign(&mut m, "campaignId", d.get("campaignId").cloned());
                set(&mut m, "findingIds", Value::Array(finding_ids.clone()));
                assign(&mut m, "ownerSessionId", d.get("ownerSessionId").cloned());
                assign(&mut m, "claim", d.get("claim").cloned());
                set(
                    &mut m,
                    "artifacts",
                    get(d, "artifacts").cloned().unwrap_or(json!([])),
                );
                set(
                    &mut m,
                    "evidence",
                    get(d, "evidence").cloned().unwrap_or(json!([])),
                );
                set(&mut m, "status", "proposed");
                set(&mut m, "createdAt", evt.ts);
                set(&mut m, "updatedAt", evt.ts);
                self.mitigations.insert(mitigation_id.clone(), m);
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    push_unique(c, "mitigationIds", json!(mitigation_id));
                }
                for id in finding_ids {
                    if let Some(f) = self.findings.get_mut(&js_string(&id)) {
                        push_unique(f, "mitigationIds", json!(mitigation_id));
                    }
                }
            }
            "mitigation.started" => {
                let mitigation_id = id_of(d, "mitigationId");
                let mut finding_ids = Vec::new();
                if let Some(m) = self.mitigations.get_mut(&mitigation_id) {
                    set(m, "status", "active");
                    set(m, "updatedAt", evt.ts);
                    finding_ids = m
                        .get("findingIds")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                }
                for id in finding_ids {
                    self.set_finding_state(&js_string(&id), "mitigating", evt, &[]);
                }
            }
            "mitigation.ready" => {
                let mitigation_id = id_of(d, "mitigationId");
                let mut finding_ids = Vec::new();
                if let Some(m) = self.mitigations.get_mut(&mitigation_id) {
                    set(m, "status", "ready");
                    for item in get_arr(d, "evidence").cloned().unwrap_or_default() {
                        push(m, "evidence", item);
                    }
                    set(m, "updatedAt", evt.ts);
                    finding_ids = m
                        .get("findingIds")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                }
                for id in finding_ids {
                    self.set_finding_state(&js_string(&id), "ready_for_retest", evt, &[]);
                }
            }
            "retest.completed" => {
                let finding_id = id_of(d, "findingId");
                let result = get_str(d, "result");
                let mut mitigation_ids = Vec::new();
                {
                    let Some(f) = self.findings.get_mut(&finding_id) else {
                        return;
                    };
                    push(
                        f,
                        "retests",
                        json!({
                            "result": d.get("result"), "sessionId": d.get("sessionId"),
                            "evidence": get(d, "evidence").cloned().unwrap_or(json!([])), "ts": evt.ts,
                        }),
                    );
                    let status = match result {
                        Some("fixed") => "confirmed",
                        Some("false_positive") => "rejected",
                        Some("persists") => "acknowledged",
                        _ => "ready_for_retest",
                    };
                    set(f, "status", status);
                    if result == Some("persists") {
                        mitigation_ids = f
                            .get("mitigationIds")
                            .and_then(Value::as_array)
                            .cloned()
                            .unwrap_or_default();
                    }
                    set(f, "updatedAt", evt.ts);
                }
                for id in mitigation_ids {
                    if let Some(m) = self.mitigations.get_mut(&js_string(&id)) {
                        if m.get("status").and_then(Value::as_str) == Some("ready") {
                            set(m, "status", "failed");
                            set(m, "updatedAt", evt.ts);
                        }
                    }
                }
            }
            "referee.review_started" => {
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    set(
                        c,
                        "review",
                        json!({ "sessionId": d.get("sessionId"), "startedAt": evt.ts, "status": "active" }),
                    );
                }
            }
            "referee.verdict" => {
                let verdict_id = id_of(d, "verdictId");
                if self.verdicts.contains(&verdict_id) {
                    return;
                }
                let mut v = Obj::new();
                set(&mut v, "id", verdict_id.clone());
                for key in ["campaignId", "sessionId", "verdict", "rationale"] {
                    assign(&mut v, key, d.get(key).cloned());
                }
                set(
                    &mut v,
                    "evidence",
                    get(d, "evidence").cloned().unwrap_or(json!([])),
                );
                set(&mut v, "createdAt", evt.ts);
                self.verdicts.insert(verdict_id.clone(), v);
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    push(c, "verdictIds", json!(verdict_id));
                    let mut review = c
                        .get("review")
                        .and_then(Value::as_object)
                        .cloned()
                        .unwrap_or_default();
                    review.insert("status".into(), json!("complete"));
                    review.insert("verdictId".into(), json!(verdict_id));
                    set(c, "review", Value::Object(review));
                }
            }
            "campaign.checkpoint_created" => {
                let checkpoint_id = id_of(d, "checkpointId");
                if self.checkpoints.contains(&checkpoint_id) {
                    return;
                }
                let mut cp = Obj::new();
                set(&mut cp, "id", checkpoint_id.clone());
                assign(&mut cp, "campaignId", d.get("campaignId").cloned());
                assign(&mut cp, "name", d.get("name").cloned());
                set(
                    &mut cp,
                    "eventSeq",
                    get(d, "eventSeq").cloned().unwrap_or(json!(evt.seq)),
                );
                set(&mut cp, "revision", or_null(d.get("revision")));
                set(&mut cp, "workspaceId", or_null(d.get("workspaceId")));
                set(&mut cp, "branch", or_null(d.get("branch")));
                set(
                    &mut cp,
                    "mode",
                    get(d, "mode").cloned().unwrap_or(json!("record_only")),
                );
                set(
                    &mut cp,
                    "capabilityIds",
                    get(d, "capabilityIds").cloned().unwrap_or(json!([])),
                );
                set(&mut cp, "scene", or_null(d.get("scene")));
                set(&mut cp, "createdAt", evt.ts);
                self.checkpoints.insert(checkpoint_id.clone(), cp);
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    push(c, "checkpointIds", json!(checkpoint_id));
                }
            }
            "campaign.promoted" => {
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    set(c, "phase", "promoted");
                    set(c, "promotedAt", evt.ts);
                    assign(c, "promotedCheckpointId", d.get("checkpointId").cloned());
                }
                for capability in get_arr(d, "capabilities").cloned().unwrap_or_default() {
                    let cap_id = id_of(&capability, "id");
                    if self.capabilities.contains(&cap_id) {
                        continue;
                    }
                    let mut cap = capability.as_object().cloned().unwrap_or_default();
                    assign(&mut cap, "campaignId", d.get("campaignId").cloned());
                    assign(&mut cap, "checkpointId", d.get("checkpointId").cloned());
                    set(&mut cap, "status", "promoted");
                    set(&mut cap, "promotedAt", evt.ts);
                    self.capabilities.insert(cap_id.clone(), cap);
                    if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                        push(c, "capabilityIds", json!(cap_id));
                    }
                }
            }
            "campaign.rolled_back" => {
                if let Some(c) = self.campaigns.get_mut(&campaign_id) {
                    set(c, "phase", "rolled_back");
                    assign(c, "rollbackCheckpointId", d.get("checkpointId").cloned());
                    assign(c, "rollbackReason", d.get("reason").cloned());
                    set(c, "rolledBackAt", evt.ts);
                    set(
                        c,
                        "rollbackMode",
                        get(d, "mode").cloned().unwrap_or(json!("record_only")),
                    );
                    set(
                        c,
                        "rollbackWorkspaceChanged",
                        d.get("workspaceChanged") == Some(&Value::Bool(true)),
                    );
                }
            }
            _ => {}
        }
    }

    fn set_finding_state(
        &mut self,
        finding_id: &str,
        state: &str,
        evt: &Event,
        extra: &[(&str, Option<&Value>)],
    ) {
        if let Some(f) = self.findings.get_mut(finding_id) {
            for (key, value) in extra {
                assign(f, key, value.cloned());
            }
            set(f, "status", state);
            set(f, "updatedAt", evt.ts);
        }
    }

    fn synchronize_session_state(
        &mut self,
        session_id: Option<&str>,
        state: Option<&str>,
        evt: &Event,
    ) {
        let Some(session_id) = session_id else {
            return;
        };
        let Some(state) = state else {
            return;
        };
        let member_status = if matches!(
            state,
            "spawning" | "ready" | "thinking" | "working" | "waiting_permission"
        ) {
            "active"
        } else if state == "paused" {
            "paused"
        } else {
            "unavailable"
        };
        let campaign_ids: Vec<String> = self.campaigns.keys().cloned().collect();
        for campaign_id in campaign_ids {
            let mut touched = false;
            let objective_ids: Vec<Value>;
            let blue_members: Vec<Value>;
            {
                let Some(campaign) = self.campaigns.get_mut(&campaign_id) else {
                    continue;
                };
                if let Some(Value::Object(teams)) = campaign.get_mut("teams") {
                    for team in teams.values_mut() {
                        let Some(Value::Object(member)) =
                            members_mut(team).and_then(|m| find_member(m, session_id))
                        else {
                            continue;
                        };
                        set(member, "status", member_status);
                        set(member, "sessionState", state);
                        team_ready(team);
                        touched = true;
                    }
                }
                if !touched {
                    continue;
                }
                objective_ids = campaign
                    .get("objectiveIds")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                blue_members = campaign
                    .get("teams")
                    .and_then(|t| t.get("blue"))
                    .and_then(|b| get_arr(b, "members"))
                    .cloned()
                    .unwrap_or_default();
            }
            // Only blue availability can block build readiness. Red/referee
            // outages are visible in their formations and staffing gates
            // without rewriting already-proven blue evidence.
            for objective_id in objective_ids {
                let Some(objective) = self.objectives.get_mut(&js_string(&objective_id)) else {
                    continue;
                };
                if objective.get("status").and_then(Value::as_str) == Some("satisfied") {
                    continue;
                }
                let blue_available = blue_members.iter().any(|m| {
                    m.get("objectiveId") == Some(&objective_id)
                        && matches!(get_str(m, "status"), Some("active" | "external"))
                });
                let status = objective
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let reason = objective
                    .get("blockedReason")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if !blue_available && matches!(status.as_str(), "queued" | "active") {
                    set(objective, "status", "blocked");
                    set(
                        objective,
                        "blockedReason",
                        "no live blue session remains assigned",
                    );
                    set(objective, "updatedAt", evt.ts);
                } else if blue_available
                    && status == "blocked"
                    && reason == "no live blue session remains assigned"
                {
                    set(objective, "status", "active");
                    set(objective, "blockedReason", Value::Null);
                    set(objective, "updatedAt", evt.ts);
                }
            }
            if let Some(campaign) = self.campaigns.get_mut(&campaign_id) {
                set(campaign, "updatedAt", evt.ts);
                set(campaign, "lastSeq", evt.seq);
            }
        }
    }
}

#[allow(dead_code)]
fn _probe(v: &Value) -> bool {
    is_truthy(v, "x")
}
