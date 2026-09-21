//! Strategic command boundary for campaigns. Port of
//! `field/server/src/orchestration/director.js`.
//!
//! The director emits facts into the canonical log and asks the registry to
//! perform real harness work. It never mutates a projection directly: the
//! server's fanout folds every emitted fact, which keeps live commands and
//! replay on one path. Nothing here holds the projection lock while emitting
//! or while calling the registry, because both fold back into the projection
//! (the registry emits its own facts); every handler reads what it needs,
//! drops the lock, then acts.
//!
//! Campaign records are read as snapshots. Where the Node director relied on
//! the projection's live object mutating underneath it mid-command, the
//! Rust port re-reads the record at that point, so the observable behaviour
//! is unchanged.

use super::campaign_projection::CampaignProjection;
use super::eventlog::AppendOptions;
use super::js::{coalesce, get, get_arr, get_str, js_string, number, or_null, truthy, Obj};
use super::model::{
    assert_id, assert_team, is_content_revision, normalize_doctrine, validate_campaign_input,
    validate_finding, validate_transition, Context, DomainError, TEAM_KINDS, VERDICTS,
};
use super::projection::Projection;
use super::registry::{uuid, Emit, Registry, ReportHandler};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex, PoisonError};

const TERMINAL_SESSION_STATES: [&str; 4] = ["done", "cancelled", "error", "interrupted"];

pub type EventHead = Arc<dyn Fn() -> u64 + Send + Sync>;
pub type IdSource = Arc<dyn Fn() -> String + Send + Sync>;

/// What the director asks of the registry. `Mutex<Registry>` is the live
/// implementation; tests stub it exactly as the Node test stubs `registry`.
pub trait Harness: Send + Sync {
    /// Spawn a session and return its live info (`id`, `agentId`, `role`,
    /// `state`, ...). The error is the message a staffing gap records.
    fn spawn(&self, body: &Value) -> Result<Value, String>;
    /// Live metadata for a session, `None` when it is not live.
    fn info(&self, session_id: &str) -> Option<Value>;
    /// Returns `{ assignmentId, skipped }`.
    fn assign(&self, body: &Value) -> Result<Value, String>;
    fn command(&self, kind: &str, payload: &Value) -> Result<Value, String>;
}

impl Harness for Mutex<Registry> {
    fn spawn(&self, body: &Value) -> Result<Value, String> {
        let mut registry = self.lock().unwrap_or_else(PoisonError::into_inner);
        let id = registry.spawn(body).map_err(|e| e.0)?;
        Ok(registry.info(&id).unwrap_or_else(|| json!({ "id": id })))
    }

    fn info(&self, session_id: &str) -> Option<Value> {
        self.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .info(session_id)
    }

    fn assign(&self, body: &Value) -> Result<Value, String> {
        self.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .assign(body)
            .map_err(|e| e.0)
    }

    fn command(&self, kind: &str, payload: &Value) -> Result<Value, String> {
        self.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .command(kind, payload)
            .map_err(|e| e.0)
    }
}

pub struct CampaignDirector {
    projection: Arc<Mutex<Projection>>,
    registry: Arc<dyn Harness>,
    emit: Emit,
    event_head: EventHead,
    id: IdSource,
}

// ---------------------------------------------------------------- helpers

/// `${value}` in a template literal: `undefined` for an absent key.
fn text(value: Option<&Value>) -> String {
    value
        .map(js_string)
        .unwrap_or_else(|| "undefined".to_string())
}

fn field(value: &Value, key: &str) -> String {
    text(value.get(key))
}

/// `String(value ?? fallback).slice(0, n)`; `value` is already `??`-filtered.
fn clip(value: Option<&Value>, fallback: &str, n: usize) -> String {
    value
        .map(js_string)
        .unwrap_or_else(|| fallback.to_string())
        .chars()
        .take(n)
        .collect()
}

/// `String(value ?? '').trim()`.
fn trimmed(value: Option<&Value>) -> String {
    value.map(js_string).unwrap_or_default().trim().to_string()
}

/// `Array.isArray(value) ? value.filter((x) => x != null && x !== '').slice(0, 200) : []`.
fn list(value: Option<&Value>) -> Vec<Value> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter(|x| !x.is_null() && x.as_str() != Some(""))
                .take(200)
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

/// An object literal whose `undefined` entries are dropped, as
/// `JSON.stringify` drops them.
fn object(pairs: Vec<(&str, Option<Value>)>) -> Value {
    let mut obj = Obj::new();
    for (key, value) in pairs {
        if let Some(value) = value {
            obj.insert(key.to_string(), value);
        }
    }
    Value::Object(obj)
}

/// `{ ...base, ...extra }`.
fn merge(mut base: Value, extra: Value) -> Value {
    if let (Value::Object(b), Value::Object(e)) = (&mut base, extra) {
        for (k, v) in e {
            b.insert(k, v);
        }
    }
    base
}

fn nested<'a>(value: &'a Value, a: &str, b: &str) -> Option<&'a Value> {
    get(value, a).and_then(|x| get(x, b))
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn cid(campaign: &Value) -> &str {
    get_str(campaign, "id").unwrap_or("")
}

fn phase(campaign: &Value) -> &str {
    get_str(campaign, "phase").unwrap_or("")
}

fn members(campaign: &Value, team: &str) -> Vec<Value> {
    campaign
        .get("teams")
        .and_then(|t| t.get(team))
        .and_then(|f| get_arr(f, "members"))
        .cloned()
        .unwrap_or_default()
}

/// `member.sessionId === sessionId` (`undefined` never matches a stored id).
fn is_session(member: &Value, session_id: Option<&Value>) -> bool {
    session_id.is_some() && member.get("sessionId") == session_id
}

fn has_status(member: &Value, status: &str) -> bool {
    get_str(member, "status") == Some(status)
}

fn not_retired(member: &Value) -> bool {
    !has_status(member, "retired")
}

/// `input.objectiveId ?? campaign.objectiveIds[0]`.
fn objective_or_first<'a>(campaign: &'a Value, input: &'a Value) -> Option<&'a Value> {
    get(input, "objectiveId")
        .or_else(|| get_arr(campaign, "objectiveIds").and_then(|ids| ids.first()))
}

fn bad_request(message: String) -> DomainError {
    DomainError::new("bad_request", message)
}

// ---------------------------------------------------------------- director

impl CampaignDirector {
    pub fn new(projection: Arc<Mutex<Projection>>, registry: Arc<dyn Harness>, emit: Emit) -> Self {
        CampaignDirector {
            projection,
            registry,
            emit,
            event_head: Arc::new(|| 0),
            id: Arc::new(uuid),
        }
    }

    /// The log head reported in every command result.
    pub fn with_event_head(mut self, event_head: EventHead) -> Self {
        self.event_head = event_head;
        self
    }

    /// The id source for generated campaign, objective, finding and
    /// command ids (random UUIDs by default).
    pub fn with_id(mut self, id: IdSource) -> Self {
        self.id = id;
        self
    }

    /// The handler the registry calls with every validated `FIELD_REPORT`.
    pub fn report_handler(self: &Arc<Self>) -> ReportHandler {
        let director = Arc::clone(self);
        Arc::new(move |envelope: Value| director.ingest_session_report(envelope))
    }

    pub fn create(&self, input: &Value) -> Result<Value, DomainError> {
        let command_id = self.command_id(input.get("commandId"))?;
        if let Some(prior) = self.prior(&command_id) {
            return Ok(prior);
        }

        let spec = validate_campaign_input(input)?;
        let campaign_id = match input.get("campaignId").filter(|v| truthy(v)) {
            Some(value) => assert_id(value, "campaignId")?,
            None => (self.id)(),
        };
        if self.read(|p| p.campaigns.contains(&campaign_id)) {
            return Err(DomainError::new(
                "already_exists",
                format!("campaign {campaign_id} already exists"),
            ));
        }

        self.append(
            "campaign.created",
            merge(
                json!({ "campaignId": campaign_id, "commandId": command_id }),
                spec.clone(),
            ),
            &campaign_id,
        );
        let mut objective_ids: Vec<String> = Vec::new();
        for objective in get_arr(&spec, "objectives").into_iter().flatten() {
            let objective_id = (self.id)();
            objective_ids.push(objective_id.clone());
            self.append(
                "objective.created",
                merge(
                    json!({ "campaignId": campaign_id, "objectiveId": objective_id, "commandId": command_id }),
                    objective.clone(),
                ),
                &objective_id,
            );
        }
        Ok(self.result(
            json!(campaign_id),
            &command_id,
            json!({ "objectiveIds": objective_ids }),
        ))
    }

    pub fn action(&self, input: &Value) -> Result<Value, DomainError> {
        let command_id = self.command_id(input.get("commandId"))?;
        if let Some(prior) = self.prior(&command_id) {
            return Ok(prior);
        }
        let campaign_id = assert_id(
            input.get("campaignId").unwrap_or(&Value::Null),
            "campaignId",
        )?;
        let campaign = self.require_campaign(&campaign_id)?;
        let kind = get(input, "kind").map(js_string).unwrap_or_default();

        if campaign.get("paused").is_some_and(truthy)
            && !matches!(kind.as_str(), "resume" | "cancel" | "rollback")
        {
            return Err(DomainError {
                code: "campaign_paused".into(),
                message: "campaign is paused".into(),
                detail: json!({ "currentPhase": campaign.get("phase") }),
            });
        }
        if campaign.get("budgetExhausted").is_some_and(truthy)
            && !matches!(kind.as_str(), "pause" | "cancel" | "rollback")
        {
            return Err(DomainError::new(
                "budget_exhausted",
                format!(
                    "campaign budget exhausted (${:.2} of ${:.2})",
                    number(campaign.get("costUsd")),
                    number(campaign.get("budgetUsd"))
                ),
            ));
        }

        let c = &campaign;
        let id = command_id.as_str();
        let detail = match kind.as_str() {
            "mobilize" => self.mobilize(c, input, id)?,
            "advance" => self.advance(c, input, id)?,
            "assign_team" => self.assign_team(c, input, id)?,
            "issue_orders" => self.issue_orders(c, input, id)?,
            "reinforce" => self.mobilize_into_existing(c, input, id, "team.reinforced")?,
            "retreat" => self.retreat(c, input, id)?,
            "pause" => self.pause(c, input, id)?,
            "resume" => self.resume(c, input, id)?,
            "cancel" => self.cancel(c, input, id)?,
            "change_doctrine" => self.change_doctrine(c, input, id)?,
            "objective_progress" => self.objective_progress(c, input, id)?,
            "satisfy_objective" => self.satisfy_objective(c, input, id)?,
            "block_objective" => self.block_objective(c, input, id)?,
            "report_finding" => self.report_finding(c, input, id)?,
            "acknowledge_finding" => self.finding_state(c, input, id, "acknowledged")?,
            "dispute_finding" => self.finding_state(c, input, id, "disputed")?,
            "waive_finding" => self.waive_finding(c, input, id)?,
            "propose_mitigation" => self.propose_mitigation(c, input, id)?,
            "start_mitigation" => self.mitigation_state(c, input, id, "started")?,
            "mark_mitigation_ready" => self.mitigation_state(c, input, id, "ready")?,
            "record_retest" => self.record_retest(c, input, id)?,
            "begin_referee_review" => self.begin_review(c, input, id)?,
            "record_verdict" => self.record_verdict(c, input, id)?,
            "checkpoint" => self.checkpoint(c, input, id)?,
            "promote" => self.promote(c, input, id)?,
            "rollback" => self.rollback(c, input, id)?,
            _ => {
                return Err(DomainError::new(
                    "unknown_action",
                    format!("unknown campaign action: {kind}"),
                ))
            }
        };
        Ok(self.result(json!(campaign_id), &command_id, detail))
    }

    /// Convert a validated, explicit final-message sentinel into ordinary
    /// campaign actions.
    pub fn ingest(&self, envelope: &Value) -> Result<Value, DomainError> {
        let seq = match get(envelope, "eventSeq") {
            Some(seq) => js_string(seq),
            None => (self.id)(),
        };
        let base = object(vec![
            (
                "commandId",
                Some(json!(format!(
                    "report:{}:{seq}",
                    field(envelope, "sessionId")
                ))),
            ),
            ("campaignId", envelope.get("campaignId").cloned()),
            ("objectiveId", envelope.get("objectiveId").cloned()),
            ("sessionId", envelope.get("sessionId").cloned()),
        ]);
        let empty = json!({});
        let report = get(envelope, "report").unwrap_or(&empty);
        let team = get_str(envelope, "team");
        let pick = |key: &str| report.get(key).cloned();
        let forbid = |message: &str| DomainError::new("forbidden", message);
        let action = |kind: &str, extra: Vec<(&str, Option<Value>)>| {
            let mut pairs = vec![("kind", Some(json!(kind)))];
            pairs.extend(extra);
            self.action(&merge(base.clone(), object(pairs)))
        };
        match get_str(report, "kind") {
            Some("objective_progress") => action(
                "objective_progress",
                vec![
                    ("progress", pick("progress")),
                    ("evidence", pick("evidence")),
                ],
            ),
            Some("objective_satisfied") => {
                if team != Some("blue") {
                    return Err(forbid("only blue can submit build-readiness evidence"));
                }
                action("satisfy_objective", vec![("evidence", pick("evidence"))])
            }
            Some("finding") => {
                if team != Some("red") {
                    return Err(forbid("only red can submit a finding report"));
                }
                action(
                    "report_finding",
                    vec![
                        ("authorSessionId", envelope.get("sessionId").cloned()),
                        ("category", pick("category")),
                        ("severity", pick("severity")),
                        ("claim", pick("claim")),
                        ("scope", pick("scope")),
                        ("evidence", pick("evidence")),
                        ("reproduction", pick("reproduction")),
                        ("noReproductionReason", pick("noReproductionReason")),
                        ("confidence", pick("confidence")),
                    ],
                )
            }
            Some("mitigation_ready") => {
                if team != Some("blue") {
                    return Err(forbid("only blue can submit mitigation readiness"));
                }
                action(
                    "mark_mitigation_ready",
                    vec![
                        ("mitigationId", pick("mitigationId")),
                        ("evidence", pick("evidence")),
                    ],
                )
            }
            Some("retest") => {
                if team != Some("red") {
                    return Err(forbid("only red can submit a retest"));
                }
                action(
                    "record_retest",
                    vec![
                        ("findingId", pick("findingId")),
                        ("result", pick("result")),
                        ("evidence", pick("evidence")),
                    ],
                )
            }
            Some("verdict") => {
                if team != Some("referee") {
                    return Err(forbid("only a referee can submit a verdict"));
                }
                action(
                    "record_verdict",
                    vec![
                        ("verdict", pick("verdict")),
                        ("evidence", pick("evidence")),
                        ("rationale", pick("rationale")),
                    ],
                )
            }
            _ => Err(DomainError::new(
                "invalid_report",
                format!(
                    "unknown FIELD_REPORT kind: {}",
                    get(report, "kind")
                        .map(js_string)
                        .unwrap_or_else(|| "(missing)".to_string())
                ),
            )),
        }
    }

    /// [`Self::ingest`] in the shape the registry's report handler takes;
    /// the error is the rejection reason it records.
    pub fn ingest_session_report(&self, envelope: Value) -> Result<(), String> {
        self.ingest(&envelope).map(|_| ()).map_err(|e| e.message)
    }

    // ------------------------------------------------------------ actions

    fn mobilize(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        let from = phase(campaign);
        if from != "draft" && from != "failed" {
            return Err(DomainError::new(
                "invalid_transition",
                format!("cannot mobilize from {}", field(campaign, "phase")),
            ));
        }
        let roster = get_arr(input, "roster").cloned().unwrap_or_default();
        if roster.is_empty() && members(campaign, "blue").is_empty() {
            return Err(DomainError::new(
                "empty_roster",
                "mobilization needs at least one blue roster entry",
            ));
        }
        let campaign_id = cid(campaign);

        let mut spawned: Vec<Value> = Vec::new();
        let mut gaps: Vec<Value> = Vec::new();
        let remaining =
            (number(campaign.get("concurrency")) - live_member_ids(campaign).len() as f64).max(0.0);
        let admitted = (remaining as usize).min(roster.len());
        let gap_of = |entry: &Value, team: Option<Value>, reason: &str| {
            object(vec![
                ("team", team),
                (
                    "role",
                    Some(get(entry, "role").cloned().unwrap_or(json!("unspecified"))),
                ),
                ("agentId", entry.get("agentId").cloned()),
                ("reason", Some(json!(reason))),
            ])
        };
        for entry in &roster[admitted..] {
            gaps.push(gap_of(
                entry,
                entry.get("team").cloned(),
                "campaign concurrency limit",
            ));
        }
        for entry in &roster[..admitted] {
            let team = assert_team(&field(entry, "team"))?.to_string();
            let objective =
                self.require_objective(campaign, objective_or_first(campaign, entry))?;
            match self.spawn_member(campaign, entry, &team, &objective, command_id) {
                Ok(session_id) => spawned.push(session_id),
                Err(reason) => {
                    let gap = gap_of(entry, Some(json!(team)), &reason);
                    gaps.push(gap.clone());
                    self.append(
                        "team.staffing_gap",
                        merge(
                            json!({ "campaignId": campaign_id, "commandId": command_id }),
                            gap,
                        ),
                        campaign_id,
                    );
                }
            }
        }

        // The Node director reads its live campaign object here; the
        // assignments above have folded in by now.
        let campaign = self.require_campaign(campaign_id)?;
        let context = self.context(campaign_id);
        let has_blue = members(&campaign, "blue")
            .iter()
            .any(|m| has_status(m, "active"));
        if !has_blue {
            if !spawned.is_empty() {
                self.registry
                    .command("pause", &json!({ "sessionIds": spawned }))
                    .map_err(bad_request)?;
            }
            self.append(
                "campaign.failed",
                json!({
                    "campaignId": campaign_id, "commandId": command_id,
                    "reason": "mobilization produced no active blue team; spawned non-blue sessions were paused",
                }),
                campaign_id,
            );
            return Ok(json!({ "spawned": spawned, "gaps": gaps, "phase": "failed" }));
        }
        validate_transition(&campaign, "mobilizing", &context)?;
        let reason = if gaps.is_empty() {
            "roster mobilized".to_string()
        } else {
            format!("mobilized with {} staffing gaps", gaps.len())
        };
        self.append(
            "campaign.phase_changed",
            json!({
                "campaignId": campaign_id, "commandId": command_id,
                "from": campaign.get("phase"), "to": "mobilizing", "reason": reason,
            }),
            campaign_id,
        );
        Ok(json!({ "spawned": spawned, "gaps": gaps, "phase": "mobilizing" }))
    }

    /// One roster entry of `mobilize`: the Node `try` block. Any failure is
    /// the staffing gap's reason.
    fn spawn_member(
        &self,
        campaign: &Value,
        entry: &Value,
        team: &str,
        objective: &Value,
        command_id: &str,
    ) -> Result<Value, String> {
        let campaign_id = cid(campaign);
        let agent_id = assert_id(entry.get("agentId").unwrap_or(&Value::Null), "agentId")
            .map_err(|e| e.message)?;
        let session = self.registry.spawn(&object(vec![
            ("agentId", Some(json!(agent_id))),
            (
                "workspaceId",
                coalesce(&[
                    get(entry, "workspaceId"),
                    nested(objective, "target", "workspaceId"),
                    nested(campaign, "target", "workspaceId"),
                ])
                .cloned(),
            ),
            ("missionId", entry.get("missionId").cloned()),
            (
                "orders",
                Some(json!(team_orders(
                    campaign,
                    objective,
                    team,
                    get(entry, "orders")
                ))),
            ),
            ("endpointId", entry.get("endpointId").cloned()),
            ("thinking", entry.get("thinking").cloned()),
            (
                "target",
                Some(
                    get(objective, "target")
                        .cloned()
                        .unwrap_or_else(|| or_null(campaign.get("target"))),
                ),
            ),
            ("campaignId", Some(json!(campaign_id))),
            ("team", Some(json!(team))),
            ("objectiveId", objective.get("id").cloned()),
            ("doctrine", campaign.get("doctrine").cloned()),
            ("environmentScope", campaign.get("scope").cloned()),
        ]))?;
        let session_id = or_null(session.get("id"));
        let live = self.require_campaign(campaign_id).map_err(|e| e.message)?;
        assert_independent_membership(&live, team, Some(&session_id)).map_err(|e| e.message)?;
        self.append(
            "team.member_assigned",
            object(vec![
                ("campaignId", Some(json!(campaign_id))),
                ("commandId", Some(json!(command_id))),
                ("team", Some(json!(team))),
                ("sessionId", Some(session_id.clone())),
                ("agentId", entry.get("agentId").cloned()),
                (
                    "role",
                    Some(or_null(coalesce(&[
                        get(entry, "role"),
                        get(&session, "role"),
                    ]))),
                ),
                ("objectiveId", objective.get("id").cloned()),
                ("status", Some(json!("active"))),
                ("assignedAt", Some(json!(now_ms()))),
            ]),
            campaign_id,
        );
        self.append(
            "objective.assigned",
            json!({
                "campaignId": campaign_id, "commandId": command_id, "objectiveId": objective.get("id"),
                "team": team, "sessionIds": [session_id],
            }),
            &field(objective, "id"),
        );
        Ok(session_id)
    }

    fn advance(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        let campaign_id = cid(campaign);
        let to = get(input, "to").map(js_string).unwrap_or_default();
        let context = self.context(campaign_id);
        validate_transition(campaign, &to, &context)?;
        self.append(
            "campaign.phase_changed",
            json!({
                "campaignId": campaign_id, "commandId": command_id,
                "from": campaign.get("phase"), "to": to,
                "reason": clip(get(input, "reason"), "operator advanced campaign", 2000),
            }),
            campaign_id,
        );
        // The Node director builds this result from its live campaign
        // object after the phase change has folded in, so `from` reports
        // the phase as it stands now.
        let after = self.require_campaign(campaign_id)?;
        Ok(json!({ "from": after.get("phase"), "to": to }))
    }

    fn assign_team(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        let campaign_id = cid(campaign);
        let team = assert_team(&field(input, "team"))?.to_string();
        let session_id = assert_id(input.get("sessionId").unwrap_or(&Value::Null), "sessionId")?;
        let session_value = json!(session_id);
        assert_independent_membership(campaign, &team, Some(&session_value))?;
        let already_assigned = TEAM_KINDS.iter().any(|kind| {
            members(campaign, kind)
                .iter()
                .any(|m| is_session(m, Some(&session_value)) && not_retired(m))
        });
        if !already_assigned {
            assert_campaign_capacity(campaign)?;
        }
        let objective = self.require_objective(campaign, objective_or_first(campaign, input))?;
        let session = self.registry.info(&session_id);
        if session.is_none() && !input.get("external").is_some_and(truthy) {
            return Err(DomainError::new(
                "session_not_found",
                format!("session {session_id} is not live"),
            ));
        }
        let status = match &session {
            Some(s) => {
                if get_str(s, "state").is_some_and(|state| TERMINAL_SESSION_STATES.contains(&state))
                {
                    "unavailable"
                } else {
                    "active"
                }
            }
            None => "external",
        };
        let from_session = |key: &str| session.as_ref().and_then(|s| get(s, key));
        self.append(
            "team.member_assigned",
            json!({
                "campaignId": campaign_id, "commandId": command_id, "team": team, "sessionId": session_id,
                "agentId": or_null(coalesce(&[get(input, "agentId"), from_session("agentId")])),
                "role": or_null(coalesce(&[get(input, "role"), from_session("role")])),
                "objectiveId": objective.get("id"),
                "status": status,
                "external": session.is_none(),
                "assignedAt": now_ms(),
            }),
            campaign_id,
        );
        self.append(
            "objective.assigned",
            json!({
                "campaignId": campaign_id, "commandId": command_id, "objectiveId": objective.get("id"),
                "team": team, "sessionIds": [session_id],
            }),
            &field(&objective, "id"),
        );
        Ok(json!({ "team": team, "sessionId": session_id, "objectiveId": objective.get("id") }))
    }

    fn issue_orders(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        let campaign_id = cid(campaign);
        let team = assert_team(&field(input, "team"))?.to_string();
        let objective = self.require_objective(campaign, objective_or_first(campaign, input))?;
        let requested = get_arr(input, "sessionIds").filter(|ids| !ids.is_empty());
        let session_ids: Vec<Value> = members(campaign, &team)
            .iter()
            .filter(|m| has_status(m, "active"))
            .map(|m| or_null(m.get("sessionId")))
            .filter(|id| requested.is_none_or(|ids| ids.contains(id)))
            .collect();
        if session_ids.is_empty() {
            return Err(DomainError::new(
                "empty_team",
                format!("team {team} has no active selected members"),
            ));
        }
        let orders = team_orders(campaign, &objective, &team, get(input, "orders"));
        let target = coalesce(&[get(&objective, "target"), get(campaign, "target")])
            .cloned()
            .unwrap_or_else(|| json!({ "type": "workspace", "id": objective.get("id") }));
        let assignment = self
            .registry
            .assign(&object(vec![
                ("sessionIds", Some(json!(session_ids))),
                ("target", Some(target)),
                ("orders", Some(json!(orders))),
                ("endpointId", input.get("endpointId").cloned()),
                ("thinking", input.get("thinking").cloned()),
            ]))
            .map_err(bad_request)?;
        self.append(
            "team.orders_issued",
            json!({
                "campaignId": campaign_id, "commandId": command_id, "team": team, "objectiveId": objective.get("id"),
                "sessionIds": session_ids, "assignmentId": assignment.get("assignmentId"), "orders": orders,
            }),
            campaign_id,
        );
        Ok(json!({
            "team": team, "sessionIds": session_ids, "assignmentId": assignment.get("assignmentId"),
            "skipped": get(&assignment, "skipped").cloned().unwrap_or(json!([])),
        }))
    }

    fn mobilize_into_existing(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
        event_kind: &str,
    ) -> Result<Value, DomainError> {
        let campaign_id = cid(campaign);
        assert_campaign_capacity(campaign)?;
        let team = assert_team(&field(input, "team"))?.to_string();
        let objective = self.require_objective(campaign, objective_or_first(campaign, input))?;
        let agent_id = assert_id(input.get("agentId").unwrap_or(&Value::Null), "agentId")?;
        let session = self
            .registry
            .spawn(&object(vec![
                ("agentId", Some(json!(agent_id))),
                (
                    "workspaceId",
                    coalesce(&[
                        get(input, "workspaceId"),
                        nested(&objective, "target", "workspaceId"),
                        nested(campaign, "target", "workspaceId"),
                    ])
                    .cloned(),
                ),
                (
                    "orders",
                    Some(json!(team_orders(
                        campaign,
                        &objective,
                        &team,
                        get(input, "orders")
                    ))),
                ),
                ("endpointId", input.get("endpointId").cloned()),
                ("thinking", input.get("thinking").cloned()),
                (
                    "target",
                    Some(
                        get(&objective, "target")
                            .cloned()
                            .unwrap_or_else(|| or_null(campaign.get("target"))),
                    ),
                ),
                ("campaignId", Some(json!(campaign_id))),
                ("team", Some(json!(team))),
                ("objectiveId", objective.get("id").cloned()),
                ("doctrine", campaign.get("doctrine").cloned()),
                ("environmentScope", campaign.get("scope").cloned()),
            ]))
            .map_err(bad_request)?;
        let session_id = or_null(session.get("id"));
        assert_independent_membership(campaign, &team, Some(&session_id))?;
        self.append(
            "team.member_assigned",
            json!({
                "campaignId": campaign_id, "commandId": command_id, "team": team, "sessionId": session_id,
                "agentId": input.get("agentId"),
                "role": or_null(coalesce(&[get(input, "role"), get(&session, "role")])),
                "objectiveId": objective.get("id"), "status": "active", "assignedAt": now_ms(),
            }),
            campaign_id,
        );
        self.append(
            "objective.assigned",
            json!({
                "campaignId": campaign_id, "commandId": command_id, "objectiveId": objective.get("id"),
                "team": team, "sessionIds": [session_id],
            }),
            &field(&objective, "id"),
        );
        self.append(
            event_kind,
            json!({ "campaignId": campaign_id, "commandId": command_id, "team": team, "sessionIds": [session_id] }),
            campaign_id,
        );
        Ok(json!({ "team": team, "sessionId": session_id }))
    }

    fn retreat(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        let campaign_id = cid(campaign);
        let team = assert_team(&field(input, "team"))?.to_string();
        let member = members(campaign, &team)
            .into_iter()
            .find(|m| is_session(m, input.get("sessionId")))
            .ok_or_else(|| {
                DomainError::new(
                    "not_a_member",
                    format!("{} is not on team {team}", field(input, "sessionId")),
                )
            })?;
        let session_id = or_null(member.get("sessionId"));
        self.registry
            .command("pause", &json!({ "sessionIds": [session_id] }))
            .map_err(bad_request)?;
        self.append(
            "team.member_removed",
            json!({
                "campaignId": campaign_id, "commandId": command_id, "team": team, "sessionId": session_id,
                "reason": clip(get(input, "reason"), "operator retreat", 2000),
            }),
            campaign_id,
        );
        self.append(
            "team.retreated",
            json!({ "campaignId": campaign_id, "commandId": command_id, "team": team, "sessionIds": [session_id] }),
            campaign_id,
        );
        Ok(json!({ "team": team, "sessionId": session_id }))
    }

    fn pause(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        if campaign.get("paused").is_some_and(truthy) {
            return Ok(json!({ "paused": true }));
        }
        let campaign_id = cid(campaign);
        let ids = live_member_ids(campaign);
        if !ids.is_empty() {
            self.registry
                .command("pause", &json!({ "sessionIds": ids }))
                .map_err(bad_request)?;
        }
        self.append(
            "campaign.paused",
            json!({ "campaignId": campaign_id, "commandId": command_id, "reason": or_null(get(input, "reason")) }),
            campaign_id,
        );
        Ok(json!({ "paused": true, "sessionIds": ids }))
    }

    fn resume(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        if !campaign.get("paused").is_some_and(truthy) {
            return Err(DomainError::new("not_paused", "campaign is not paused"));
        }
        let campaign_id = cid(campaign);
        let ids = live_member_ids(campaign);
        if !ids.is_empty() {
            self.registry
                .command(
                    "resume",
                    &object(vec![
                        ("sessionIds", Some(json!(ids))),
                        ("orders", input.get("orders").cloned()),
                    ]),
                )
                .map_err(bad_request)?;
        }
        self.append(
            "campaign.resumed",
            json!({ "campaignId": campaign_id, "commandId": command_id }),
            campaign_id,
        );
        Ok(json!({ "resumed": true, "sessionIds": ids }))
    }

    fn cancel(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        if phase(campaign) == "cancelled" {
            return Ok(json!({ "cancelled": true }));
        }
        let campaign_id = cid(campaign);
        let ids = live_member_ids(campaign);
        if !ids.is_empty() {
            self.registry
                .command("cancel", &json!({ "sessionIds": ids }))
                .map_err(bad_request)?;
        }
        self.append(
            "campaign.cancelled",
            json!({
                "campaignId": campaign_id, "commandId": command_id,
                "reason": clip(get(input, "reason"), "operator cancelled", 2000),
            }),
            campaign_id,
        );
        Ok(json!({ "cancelled": true, "sessionIds": ids }))
    }

    fn change_doctrine(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        let campaign_id = cid(campaign);
        let mut merged = get(campaign, "doctrine")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        if let Some(Value::Object(patch)) = get(input, "doctrine") {
            for (k, v) in patch {
                merged.insert(k.clone(), v.clone());
            }
        }
        let doctrine = normalize_doctrine(Some(&Value::Object(merged)))?;
        self.append(
            "campaign.doctrine_changed",
            json!({ "campaignId": campaign_id, "commandId": command_id, "doctrine": doctrine }),
            campaign_id,
        );
        Ok(json!({ "doctrine": doctrine }))
    }

    fn objective_progress(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        let objective = self.require_objective(campaign, input.get("objectiveId"))?;
        self.append(
            "objective.progress",
            json!({
                "campaignId": cid(campaign), "commandId": command_id, "objectiveId": objective.get("id"),
                "progress": or_null(get(input, "progress")), "evidence": or_null(get(input, "evidence")),
            }),
            &field(&objective, "id"),
        );
        Ok(json!({ "objectiveId": objective.get("id") }))
    }

    fn satisfy_objective(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        let objective = self.require_objective(campaign, input.get("objectiveId"))?;
        let evidence = list(input.get("evidence"));
        if evidence.is_empty() {
            return Err(DomainError::new(
                "missing_evidence",
                "satisfying an objective requires evidence",
            ));
        }
        let done = get_arr(&objective, "definitionOfDone")
            .cloned()
            .unwrap_or_default();
        if evidence.len() < done.len() {
            return Err(DomainError::new(
                "missing_evidence",
                format!(
                    "objective has {} completion criteria but only {} evidence entries",
                    done.len(),
                    evidence.len()
                ),
            ));
        }
        let criteria_evidence: Vec<Value> = done
            .iter()
            .enumerate()
            .map(|(index, criterion)| json!({ "criterion": criterion, "evidence": evidence.get(index) }))
            .collect();
        self.append(
            "objective.satisfied",
            json!({
                "campaignId": cid(campaign), "commandId": command_id, "objectiveId": objective.get("id"),
                "evidence": evidence, "criteriaEvidence": criteria_evidence,
                "sessionId": or_null(get(input, "sessionId")),
            }),
            &field(&objective, "id"),
        );
        Ok(json!({ "objectiveId": objective.get("id"), "status": "satisfied" }))
    }

    fn block_objective(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        let objective = self.require_objective(campaign, input.get("objectiveId"))?;
        self.append(
            "objective.blocked",
            json!({
                "campaignId": cid(campaign), "commandId": command_id, "objectiveId": objective.get("id"),
                "reason": clip(get(input, "reason"), "blocked", 4000),
            }),
            &field(&objective, "id"),
        );
        Ok(json!({ "objectiveId": objective.get("id"), "status": "blocked" }))
    }

    fn report_finding(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        if !matches!(phase(campaign), "red_challenging" | "red_retesting") {
            return Err(DomainError::new(
                "wrong_phase",
                format!(
                    "findings cannot be reported during {}",
                    field(campaign, "phase")
                ),
            ));
        }
        let objective = self.require_objective(campaign, input.get("objectiveId"))?;
        assert_member(campaign, "red", input.get("authorSessionId"))?;
        let finding = validate_finding(input)?;
        let finding_id = match input.get("findingId").filter(|v| truthy(v)) {
            Some(value) => assert_id(value, "findingId")?,
            None => (self.id)(),
        };
        if self.read(|p| p.findings.contains(&finding_id)) {
            return Err(DomainError::new(
                "already_exists",
                format!("finding {finding_id} exists"),
            ));
        }
        self.append(
            "finding.reported",
            merge(
                object(vec![
                    ("campaignId", Some(json!(cid(campaign)))),
                    ("commandId", Some(json!(command_id))),
                    ("findingId", Some(json!(finding_id))),
                    ("objectiveId", objective.get("id").cloned()),
                    ("authorSessionId", input.get("authorSessionId").cloned()),
                ]),
                finding,
            ),
            &finding_id,
        );
        Ok(json!({ "findingId": finding_id }))
    }

    fn finding_state(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
        state: &str,
    ) -> Result<Value, DomainError> {
        let finding = self.require_finding(campaign, input.get("findingId"))?;
        let kind = if state == "disputed" {
            "finding.disputed"
        } else {
            "finding.acknowledged"
        };
        if state == "disputed" && list(input.get("evidence")).is_empty() {
            return Err(DomainError::new(
                "missing_evidence",
                "disputing a finding requires evidence",
            ));
        }
        self.append(
            kind,
            json!({
                "campaignId": cid(campaign), "commandId": command_id, "findingId": finding.get("id"),
                "sessionId": or_null(get(input, "sessionId")), "evidence": list(input.get("evidence")),
            }),
            &field(&finding, "id"),
        );
        Ok(json!({ "findingId": finding.get("id"), "status": state }))
    }

    fn waive_finding(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        let finding = self.require_finding(campaign, input.get("findingId"))?;
        if !matches!(get_str(input, "authority"), Some("operator" | "referee")) {
            return Err(DomainError::new(
                "forbidden",
                "only an operator or referee can waive a finding",
            ));
        }
        let reason = trimmed(get(input, "reason"));
        if reason.is_empty() {
            return Err(DomainError::new(
                "missing_reason",
                "waiving a finding requires a reason",
            ));
        }
        self.append(
            "finding.waived",
            json!({
                "campaignId": cid(campaign), "commandId": command_id, "findingId": finding.get("id"),
                "authority": input.get("authority"), "sessionId": or_null(get(input, "sessionId")), "reason": reason,
            }),
            &field(&finding, "id"),
        );
        Ok(json!({ "findingId": finding.get("id"), "status": "waived" }))
    }

    fn propose_mitigation(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        let finding_ids = list(input.get("findingIds"));
        if finding_ids.is_empty() {
            return Err(DomainError::new(
                "missing_findings",
                "a mitigation must reference findings",
            ));
        }
        for id in &finding_ids {
            self.require_finding(campaign, Some(id))?;
        }
        if input.get("ownerSessionId").is_some_and(truthy) {
            assert_member(campaign, "blue", input.get("ownerSessionId"))?;
        }
        let claim = trimmed(get(input, "claim"));
        if claim.is_empty() {
            return Err(DomainError::new(
                "invalid_mitigation",
                "mitigation claim is required",
            ));
        }
        let mitigation_id = match input.get("mitigationId").filter(|v| truthy(v)) {
            Some(value) => assert_id(value, "mitigationId")?,
            None => (self.id)(),
        };
        self.append(
            "mitigation.proposed",
            json!({
                "campaignId": cid(campaign), "commandId": command_id, "mitigationId": mitigation_id,
                "findingIds": finding_ids, "ownerSessionId": or_null(get(input, "ownerSessionId")), "claim": claim,
                "artifacts": list(input.get("artifacts")), "evidence": list(input.get("evidence")),
            }),
            &mitigation_id,
        );
        Ok(json!({ "mitigationId": mitigation_id }))
    }

    fn mitigation_state(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
        state: &str,
    ) -> Result<Value, DomainError> {
        let mitigation = self.require_mitigation(campaign, input.get("mitigationId"))?;
        let prior_evidence = get_arr(&mitigation, "evidence").is_some_and(|e| !e.is_empty());
        if state == "ready" && list(input.get("evidence")).is_empty() && !prior_evidence {
            return Err(DomainError::new(
                "missing_evidence",
                "a ready mitigation requires evidence",
            ));
        }
        let kind = if state == "started" {
            "mitigation.started"
        } else {
            "mitigation.ready"
        };
        self.append(
            kind,
            json!({
                "campaignId": cid(campaign), "commandId": command_id, "mitigationId": mitigation.get("id"),
                "sessionId": or_null(get(input, "sessionId")), "evidence": list(input.get("evidence")),
            }),
            &field(&mitigation, "id"),
        );
        let status = if state == "started" {
            "active"
        } else {
            "ready"
        };
        Ok(json!({ "mitigationId": mitigation.get("id"), "status": status }))
    }

    fn record_retest(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        let finding = self.require_finding(campaign, input.get("findingId"))?;
        assert_member(campaign, "red", input.get("sessionId"))?;
        if !matches!(
            get_str(input, "result"),
            Some("fixed" | "persists" | "false_positive" | "inconclusive")
        ) {
            return Err(DomainError::new(
                "invalid_retest",
                format!("unknown retest result: {}", field(input, "result")),
            ));
        }
        let evidence = list(input.get("evidence"));
        if evidence.is_empty() {
            return Err(DomainError::new(
                "missing_evidence",
                "a retest requires evidence",
            ));
        }
        self.append(
            "retest.completed",
            json!({
                "campaignId": cid(campaign), "commandId": command_id, "findingId": finding.get("id"),
                "sessionId": input.get("sessionId"), "result": input.get("result"), "evidence": evidence,
            }),
            &field(&finding, "id"),
        );
        Ok(json!({ "findingId": finding.get("id"), "result": input.get("result") }))
    }

    fn begin_review(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        if phase(campaign) != "referee_review" {
            return Err(DomainError::new(
                "wrong_phase",
                format!(
                    "referee review cannot begin during {}",
                    field(campaign, "phase")
                ),
            ));
        }
        assert_member(campaign, "referee", input.get("sessionId"))?;
        let campaign_id = cid(campaign);
        self.append(
            "referee.review_started",
            json!({ "campaignId": campaign_id, "commandId": command_id, "sessionId": input.get("sessionId") }),
            campaign_id,
        );
        Ok(json!({ "sessionId": input.get("sessionId") }))
    }

    fn record_verdict(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        if phase(campaign) != "referee_review" {
            return Err(DomainError::new(
                "wrong_phase",
                format!(
                    "verdict cannot be recorded during {}",
                    field(campaign, "phase")
                ),
            ));
        }
        assert_member(campaign, "referee", input.get("sessionId"))?;
        assert_independent_membership(campaign, "referee", input.get("sessionId"))?;
        let review = get(campaign, "review");
        let review_active = review.is_some_and(|r| get_str(r, "status") == Some("active"));
        if !review_active || review.and_then(|r| r.get("sessionId")) != input.get("sessionId") {
            return Err(DomainError::new(
                "review_not_started",
                "this referee must begin an independent review before recording a verdict",
            ));
        }
        if !get_str(input, "verdict").is_some_and(|v| VERDICTS.contains(&v)) {
            return Err(DomainError::new(
                "invalid_verdict",
                format!("unknown verdict: {}", field(input, "verdict")),
            ));
        }
        let evidence = list(input.get("evidence"));
        if evidence.is_empty() {
            return Err(DomainError::new(
                "missing_evidence",
                "a referee verdict requires evidence",
            ));
        }
        let rationale = trimmed(get(input, "rationale"));
        if rationale.is_empty() {
            return Err(DomainError::new(
                "missing_rationale",
                "a referee verdict requires rationale",
            ));
        }
        let verdict_id = (self.id)();
        self.append(
            "referee.verdict",
            json!({
                "campaignId": cid(campaign), "commandId": command_id, "verdictId": verdict_id,
                "sessionId": input.get("sessionId"), "verdict": input.get("verdict"),
                "evidence": evidence, "rationale": rationale,
            }),
            &verdict_id,
        );
        Ok(json!({ "verdictId": verdict_id, "verdict": input.get("verdict") }))
    }

    fn checkpoint(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        if !matches!(phase(campaign), "verified" | "promoted") {
            return Err(DomainError::new(
                "wrong_phase",
                format!(
                    "checkpoint cannot be promoted from {}",
                    field(campaign, "phase")
                ),
            ));
        }
        let checkpoint_id = match input.get("checkpointId").filter(|v| truthy(v)) {
            Some(value) => assert_id(value, "checkpointId")?,
            None => (self.id)(),
        };
        let default_name = format!("{} checkpoint", field(campaign, "name"));
        self.append(
            "campaign.checkpoint_created",
            json!({
                "campaignId": cid(campaign), "commandId": command_id, "checkpointId": checkpoint_id,
                "name": clip(get(input, "name"), &default_name, 200),
                "eventSeq": (self.event_head)(), "revision": or_null(get(input, "revision")),
                "workspaceId": or_null(coalesce(&[get(input, "workspaceId"), nested(campaign, "target", "workspaceId")])),
                "branch": or_null(get(input, "branch")),
                "mode": get(input, "checkpointMode").cloned().unwrap_or(json!("record_only")),
                "capabilityIds": list(input.get("capabilityIds")), "scene": or_null(get(input, "scene")),
            }),
            &checkpoint_id,
        );
        Ok(json!({ "checkpointId": checkpoint_id }))
    }

    fn promote(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        let campaign_id = cid(campaign);
        let mut context = self.context(campaign_id);
        let checkpoint_id: Option<Value> = get(input, "checkpointId")
            .cloned()
            .or_else(|| context.checkpoint_id.clone().map(Value::String));
        context.checkpoint_id = checkpoint_id.as_ref().map(js_string);
        validate_transition(campaign, "promoted", &context)?;
        let key = text(checkpoint_id.as_ref());
        let checkpoint = self.read(|p| p.checkpoints.get(&key).cloned());
        let belongs = checkpoint
            .as_ref()
            .is_some_and(|c| c.get("campaignId").and_then(Value::as_str) == Some(campaign_id));
        if !belongs {
            return Err(DomainError::new(
                "invalid_checkpoint",
                "checkpoint does not belong to this campaign",
            ));
        }
        let revision = checkpoint
            .as_ref()
            .and_then(|c| c.get("revision"))
            .cloned()
            .unwrap_or(Value::Null);
        if !is_content_revision(&revision) {
            return Err(DomainError::new(
                "invalid_checkpoint",
                "promotion requires a checkpoint bound to a Git or SHA-256 content revision",
            ));
        }
        let mut capabilities: Vec<Value> = Vec::new();
        for capability in get_arr(input, "capabilities").into_iter().flatten() {
            let id_value = match get(capability, "id") {
                Some(id) => id.clone(),
                None => json!((self.id)()),
            };
            let id = assert_id(&id_value, "capabilityId")?;
            capabilities.push(json!({
                "id": id,
                "name": clip(coalesce(&[get(capability, "name"), get(capability, "id")]), "capability", 300),
                "evidence": list(capability.get("evidence")),
                "target": get(capability, "target").cloned().unwrap_or_else(|| or_null(campaign.get("target"))),
            }));
        }
        if capabilities.is_empty() {
            return Err(DomainError::new(
                "missing_capabilities",
                "promotion requires at least one capability",
            ));
        }
        let capability_ids: Vec<Value> =
            capabilities.iter().map(|c| or_null(c.get("id"))).collect();
        self.append(
            "campaign.promoted",
            json!({
                "campaignId": campaign_id, "commandId": command_id, "checkpointId": checkpoint_id,
                "capabilities": capabilities,
            }),
            campaign_id,
        );
        Ok(json!({ "checkpointId": checkpoint_id, "capabilityIds": capability_ids }))
    }

    fn rollback(
        &self,
        campaign: &Value,
        input: &Value,
        command_id: &str,
    ) -> Result<Value, DomainError> {
        if phase(campaign) != "promoted" {
            return Err(DomainError::new(
                "wrong_phase",
                format!(
                    "rollback requires a promoted campaign, not {}",
                    field(campaign, "phase")
                ),
            ));
        }
        let campaign_id = cid(campaign);
        let checkpoint_id = assert_id(
            input.get("checkpointId").unwrap_or(&Value::Null),
            "checkpointId",
        )?;
        let belongs = self.read(|p| {
            p.checkpoints
                .get(&checkpoint_id)
                .is_some_and(|c| c.get("campaignId").and_then(Value::as_str) == Some(campaign_id))
        });
        if !belongs {
            return Err(DomainError::new(
                "invalid_checkpoint",
                "rollback checkpoint does not belong to this campaign",
            ));
        }
        let reason = trimmed(get(input, "reason"));
        if reason.is_empty() {
            return Err(DomainError::new(
                "missing_reason",
                "rollback requires a reason",
            ));
        }
        self.append(
            "campaign.rolled_back",
            json!({
                "campaignId": campaign_id, "commandId": command_id, "checkpointId": checkpoint_id, "reason": reason,
                "mode": "record_only", "workspaceChanged": false,
            }),
            campaign_id,
        );
        Ok(json!({
            "checkpointId": checkpoint_id, "rolledBack": true, "mode": "record_only", "workspaceChanged": false,
        }))
    }

    // ------------------------------------------------------------ plumbing

    fn read<T>(&self, f: impl FnOnce(&CampaignProjection) -> T) -> T {
        let guard = self
            .projection
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        f(&guard.campaigns)
    }

    fn context(&self, campaign_id: &str) -> Context {
        self.read(|p| p.context(campaign_id)).unwrap_or_default()
    }

    fn append(&self, kind: &str, data: Value, subject: &str) {
        (self.emit)(kind, data, AppendOptions::subject(subject));
    }

    fn command_id(&self, value: Option<&Value>) -> Result<String, DomainError> {
        match value.filter(|v| truthy(v)) {
            Some(value) => assert_id(value, "commandId"),
            None => Ok((self.id)()),
        }
    }

    fn prior(&self, command_id: &str) -> Option<Value> {
        let prior = self.read(|p| p.command_results.get(command_id).cloned())?;
        Some(self.result(
            or_null(prior.get("campaignId")),
            command_id,
            json!({ "duplicate": true, "originalSeq": prior.get("seq") }),
        ))
    }

    fn result(&self, campaign_id: Value, command_id: &str, detail: Value) -> Value {
        let campaign = self.read(|p| {
            campaign_id
                .as_str()
                .and_then(|id| p.campaigns.get(id))
                .map(|c| p.campaign_view(c))
        });
        merge(
            json!({
                "ok": true, "commandId": command_id, "campaignId": campaign_id,
                "head": (self.event_head)(), "campaign": campaign.unwrap_or(Value::Null),
            }),
            detail,
        )
    }

    fn require_campaign(&self, id: &str) -> Result<Value, DomainError> {
        self.read(|p| p.campaigns.get(id).cloned())
            .map(Value::Object)
            .ok_or_else(|| DomainError::new("not_found", format!("campaign {id} does not exist")))
    }

    fn require_objective(
        &self,
        campaign: &Value,
        id: Option<&Value>,
    ) -> Result<Value, DomainError> {
        self.require_owned("objective", campaign, id, |p| &p.objectives)
    }

    fn require_finding(&self, campaign: &Value, id: Option<&Value>) -> Result<Value, DomainError> {
        self.require_owned("finding", campaign, id, |p| &p.findings)
    }

    fn require_mitigation(
        &self,
        campaign: &Value,
        id: Option<&Value>,
    ) -> Result<Value, DomainError> {
        self.require_owned("mitigation", campaign, id, |p| &p.mitigations)
    }

    /// A record that exists and belongs to this campaign.
    fn require_owned(
        &self,
        what: &str,
        campaign: &Value,
        id: Option<&Value>,
        table: impl FnOnce(&CampaignProjection) -> &super::js::OrderedMap<Obj>,
    ) -> Result<Value, DomainError> {
        let key = text(id);
        let campaign_id = cid(campaign);
        self.read(|p| {
            table(p)
                .get(&key)
                .filter(|record| {
                    record.get("campaignId").and_then(Value::as_str) == Some(campaign_id)
                })
                .cloned()
        })
        .map(Value::Object)
        .ok_or_else(|| {
            DomainError::new(
                "not_found",
                format!("{what} {key} does not belong to campaign {campaign_id}"),
            )
        })
    }
}

fn team_orders(campaign: &Value, objective: &Value, team: &str, extra: Option<&Value>) -> String {
    let mut lines: Vec<String> = vec![
        format!("Campaign: {}", field(campaign, "name")),
        format!("Intent: {}", field(campaign, "intent")),
        format!("Your team: {}", team.to_uppercase()),
        format!("Environment scope: {}", field(campaign, "scope")),
        format!("Objective: {}", field(objective, "statement")),
        "Definition of done:".to_string(),
    ];
    for item in get_arr(objective, "definitionOfDone").into_iter().flatten() {
        lines.push(format!("- {}", js_string(item)));
    }
    let push = |lines: &mut Vec<String>, items: &[&str]| {
        lines.extend(items.iter().map(|s| s.to_string()));
    };
    match team {
        "blue" => push(&mut lines, &["Build or operate the result and attach concrete evidence for every completion claim."]),
        "red" => {
            let categories = nested(campaign, "doctrine", "redCategories")
                .and_then(Value::as_array)
                .map(|items| items.iter().map(js_string).collect::<Vec<_>>().join(", "))
                .unwrap_or_default();
            lines.push(format!("Challenge categories: {categories}."));
            push(&mut lines, &[
                "Do not repair what you find. Report claim, severity, scope, evidence, reproduction, and confidence.",
                "Stay inside the declared environment scope and use decoy credentials for injection tests.",
            ]);
        }
        "referee" => push(&mut lines, &["Reproduce the definition of done independently. Neither blue nor red summaries are facts until you verify them."]),
        "purple" => push(&mut lines, &["Synthesize lessons only after the referee verdict. You cannot waive retest or verification."]),
        _ => {}
    }
    push(&mut lines, &[
        "",
        "Structured campaign reporting:",
        "End the final message with `FIELD_REPORT:` followed by exactly one compact JSON object.",
    ]);
    match team {
        "blue" => push(
            &mut lines,
            &[
                r#"Use {"kind":"objective_satisfied","evidence":[...]} only when every definition-of-done item has concrete evidence."#,
                r#"For partial work use {"kind":"objective_progress","progress":{"done":N,"total":N},"evidence":"..."}."#,
            ],
        ),
        "red" => push(
            &mut lines,
            &[
                r#"Use {"kind":"finding","category":"...","severity":"high","claim":"...","scope":"...","evidence":[...],"reproduction":[...],"confidence":0.9}."#,
                r#"During retest use {"kind":"retest","findingId":"...","result":"fixed|persists|false_positive|inconclusive","evidence":[...]}."#,
            ],
        ),
        "referee" => push(
            &mut lines,
            &[
                r#"Use {"kind":"verdict","verdict":"verified|rejected|inconclusive","evidence":[...],"rationale":"..."}."#,
            ],
        ),
        _ => {}
    }
    if let Some(extra) = extra.filter(|e| truthy(e)) {
        lines.push(String::new());
        lines.push(js_string(extra).chars().take(20_000).collect());
    }
    lines.join("\n")
}

fn assert_member(
    campaign: &Value,
    team: &str,
    session_id: Option<&Value>,
) -> Result<Value, DomainError> {
    members(campaign, team)
        .into_iter()
        .find(|m| is_session(m, session_id) && not_retired(m))
        .ok_or_else(|| {
            DomainError::new(
                "forbidden",
                format!(
                    "session {} is not an active {team} team member",
                    text(session_id)
                ),
            )
        })
}

fn assert_independent_membership(
    campaign: &Value,
    team: &str,
    session_id: Option<&Value>,
) -> Result<(), DomainError> {
    let conflicts: &[&str] = if team == "referee" {
        &["blue", "red"]
    } else {
        &["referee"]
    };
    for other in conflicts {
        let conflict = members(campaign, other)
            .iter()
            .any(|m| is_session(m, session_id) && not_retired(m));
        if conflict {
            return Err(DomainError::new(
                "role_conflict",
                format!(
                    "{} cannot serve on both {team} and {other}",
                    text(session_id)
                ),
            ));
        }
    }
    Ok(())
}

/// Distinct session ids of every active member, in team order.
fn live_member_ids(campaign: &Value) -> Vec<Value> {
    let mut ids: Vec<Value> = Vec::new();
    for team in TEAM_KINDS {
        for member in members(campaign, team) {
            if !has_status(&member, "active") {
                continue;
            }
            let id = or_null(member.get("sessionId"));
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

fn assert_campaign_capacity(campaign: &Value) -> Result<(), DomainError> {
    let active = live_member_ids(campaign).len();
    if active as f64 >= number(campaign.get("concurrency")) {
        return Err(DomainError::new(
            "campaign_capacity",
            format!(
                "campaign concurrency limit is {}; {active} sessions are already active",
                field(campaign, "concurrency")
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Port of `field/server/test/director.test.mjs`, with the registry
    //! stubbed exactly as that test stubs it.

    use super::*;
    use crate::field::eventlog::{Event, Source};
    use crate::field::projection::FieldConfig;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct FakeRegistry {
        sessions: Mutex<Vec<Value>>,
        calls: Mutex<Vec<(String, Value)>>,
    }

    impl Harness for FakeRegistry {
        fn spawn(&self, input: &Value) -> Result<Value, String> {
            let mut sessions = self.sessions.lock().unwrap();
            let id = format!("session-{}", sessions.len() + 1);
            let role = match get_str(input, "team") {
                Some("referee") => "verifier",
                Some("red") => "scout",
                _ => "builder",
            };
            let session = merge(
                json!({ "id": id, "agentId": input.get("agentId"), "role": role, "state": "working" }),
                input.clone(),
            );
            sessions.push(session.clone());
            self.calls
                .lock()
                .unwrap()
                .push(("spawn".into(), input.clone()));
            Ok(session)
        }

        fn info(&self, session_id: &str) -> Option<Value> {
            self.sessions
                .lock()
                .unwrap()
                .iter()
                .find(|s| get_str(s, "id") == Some(session_id))
                .cloned()
        }

        fn assign(&self, input: &Value) -> Result<Value, String> {
            let mut calls = self.calls.lock().unwrap();
            calls.push(("assign".into(), input.clone()));
            Ok(json!({ "assignmentId": format!("assign-{}", calls.len()), "skipped": [] }))
        }

        fn command(&self, kind: &str, input: &Value) -> Result<Value, String> {
            self.calls
                .lock()
                .unwrap()
                .push((kind.to_string(), input.clone()));
            Ok(json!({ "ok": true }))
        }
    }

    struct Rig {
        projection: Arc<Mutex<Projection>>,
        stored: Arc<Mutex<Vec<Event>>>,
        registry: Arc<FakeRegistry>,
        director: CampaignDirector,
    }

    fn rig() -> Rig {
        let projection = Arc::new(Mutex::new(Projection::new(FieldConfig::default())));
        let stored: Arc<Mutex<Vec<Event>>> = Arc::new(Mutex::new(Vec::new()));
        let seq = Arc::new(AtomicU64::new(0));
        let id_seq = Arc::new(AtomicU64::new(0));
        let registry = Arc::new(FakeRegistry {
            sessions: Mutex::new(Vec::new()),
            calls: Mutex::new(Vec::new()),
        });
        let emit: Emit = {
            let projection = Arc::clone(&projection);
            let stored = Arc::clone(&stored);
            let seq = Arc::clone(&seq);
            Arc::new(move |kind: &str, data: Value, options: AppendOptions| {
                let seq = seq.fetch_add(1, Ordering::SeqCst) + 1;
                let event = Event {
                    seq,
                    ts: 1_700_000_000_000 + seq as i64,
                    kind: kind.to_string(),
                    actor: options.actor,
                    subject: options.subject,
                    source: options.source.unwrap_or(Source::Observed),
                    data,
                };
                stored.lock().unwrap().push(event.clone());
                projection.lock().unwrap().campaigns.apply(&event);
                Some(event)
            })
        };
        let harness: Arc<dyn Harness> = Arc::clone(&registry) as Arc<dyn Harness>;
        let director = CampaignDirector::new(Arc::clone(&projection), harness, emit)
            .with_event_head(Arc::new(move || seq.load(Ordering::SeqCst)))
            .with_id(Arc::new(move || {
                format!("id-{}", id_seq.fetch_add(1, Ordering::SeqCst) + 1)
            }));
        Rig {
            projection,
            stored,
            registry,
            director,
        }
    }

    impl Rig {
        fn act(&self, command_id: &str, kind: &str, rest: Value) -> Result<Value, DomainError> {
            self.director.action(&merge(
                json!({ "commandId": command_id, "campaignId": "campaign-1", "kind": kind }),
                rest,
            ))
        }

        fn ok(&self, command_id: &str, kind: &str, rest: Value) -> Value {
            self.act(command_id, kind, rest).unwrap_or_else(|e| {
                panic!("{kind} ({command_id}) failed: {} [{}]", e.message, e.code)
            })
        }

        fn campaign(&self) -> Value {
            let p = self.projection.lock().unwrap();
            Value::Object(p.campaigns.campaigns.get("campaign-1").unwrap().clone())
        }

        fn view(&self) -> Value {
            let p = self.projection.lock().unwrap();
            p.campaigns
                .campaign_view(p.campaigns.campaigns.get("campaign-1").unwrap())
        }

        fn calls(&self) -> Vec<(String, Value)> {
            self.registry.calls.lock().unwrap().clone()
        }

        fn create_campaign(&self) -> Value {
            self.director
                .create(&json!({
                    "commandId": "cmd-create", "campaignId": "campaign-1", "name": "Release gateway",
                    "intent": "Prove rollback under concurrency.", "scope": "sandbox",
                    "objectives": [{
                        "statement": "Rollback remains safe during restart",
                        "definitionOfDone": ["race harness passes 1000 iterations"],
                        "target": { "type": "folder", "id": "cameod/src", "workspaceId": "cameo" },
                    }],
                }))
                .expect("create")
        }
    }

    fn err_of(result: Result<Value, DomainError>) -> DomainError {
        match result {
            Ok(value) => panic!("expected an error, got {value}"),
            Err(e) => e,
        }
    }

    #[test]
    fn full_red_blue_referee_vertical_slice() {
        let rig = rig();
        let created = rig.create_campaign();
        assert_eq!(get_str(&created["campaign"], "phase"), Some("draft"));
        let objective_id = js_string(&created["objectiveIds"][0]);

        {
            let mut p = rig.projection.lock().unwrap();
            let objective = p.campaigns.objectives.get_mut(&objective_id).unwrap();
            objective["definitionOfDone"]
                .as_array_mut()
                .unwrap()
                .push(json!("restart check passes"));
        }
        let short = err_of(rig.act(
            "cmd-short-evidence",
            "satisfy_objective",
            json!({ "objectiveId": objective_id, "evidence": ["only one proof"] }),
        ));
        assert!(
            short
                .message
                .contains("2 completion criteria but only 1 evidence entries"),
            "{}",
            short.message
        );
        {
            let mut p = rig.projection.lock().unwrap();
            let objective = p.campaigns.objectives.get_mut(&objective_id).unwrap();
            objective["definitionOfDone"].as_array_mut().unwrap().pop();
        }

        let duplicate_create = rig
            .director
            .create(&json!({
                "commandId": "cmd-create", "name": "ignored duplicate", "intent": "ignored",
                "objectives": [{ "statement": "ignored", "definitionOfDone": ["ignored"] }],
            }))
            .expect("duplicate create");
        assert_eq!(duplicate_create["duplicate"], json!(true));
        assert_eq!(rig.projection.lock().unwrap().campaigns.campaigns.len(), 1);

        let mobilized = rig.ok(
            "cmd-mobilize",
            "mobilize",
            json!({ "roster": [
                { "team": "blue", "agentId": "blue-agent", "objectiveId": objective_id },
                { "team": "red", "agentId": "red-agent", "objectiveId": objective_id },
                { "team": "referee", "agentId": "ref-agent", "objectiveId": objective_id },
            ] }),
        );
        assert_eq!(mobilized["phase"], json!("mobilizing"));
        let spawned = mobilized["spawned"].as_array().unwrap().clone();
        assert_eq!(spawned.len(), 3);
        let (blue_id, red_id, referee_id) =
            (spawned[0].clone(), spawned[1].clone(), spawned[2].clone());
        let first_spawn = rig
            .calls()
            .into_iter()
            .find(|(kind, _)| kind == "spawn")
            .unwrap()
            .1;
        assert!(get_str(&first_spawn, "orders")
            .unwrap()
            .contains("Your team: BLUE"));

        rig.ok("cmd-blue", "advance", json!({ "to": "blue_building" }));
        rig.ok(
            "cmd-satisfy",
            "satisfy_objective",
            json!({ "objectiveId": objective_id, "sessionId": blue_id, "evidence": ["stress: pass"] }),
        );
        rig.ok("cmd-red", "advance", json!({ "to": "red_challenging" }));
        let reported = rig.ok(
            "cmd-find",
            "report_finding",
            json!({
                "objectiveId": objective_id, "authorSessionId": red_id, "category": "rollback", "severity": "high",
                "claim": "restart and rollback interleave", "scope": "release gateway",
                "evidence": ["trace://41"], "reproduction": ["start restart", "issue rollback"], "confidence": 0.95,
            }),
        );
        let finding_id = reported["findingId"].clone();
        rig.ok("cmd-contested", "advance", json!({ "to": "contested" }));
        rig.ok(
            "cmd-ack",
            "acknowledge_finding",
            json!({ "findingId": finding_id, "sessionId": blue_id }),
        );
        rig.ok(
            "cmd-mitigating",
            "advance",
            json!({ "to": "blue_mitigating" }),
        );
        let mitigation = rig.ok(
            "cmd-mitigation",
            "propose_mitigation",
            json!({
                "findingIds": [finding_id], "ownerSessionId": blue_id,
                "claim": "serialize gateway state transitions", "artifacts": ["cameod/src/app.rs"],
            }),
        );
        let mitigation_id = mitigation["mitigationId"].clone();
        rig.ok(
            "cmd-mitigation-start",
            "start_mitigation",
            json!({ "mitigationId": mitigation_id, "sessionId": blue_id }),
        );
        rig.ok(
            "cmd-mitigation-ready",
            "mark_mitigation_ready",
            json!({ "mitigationId": mitigation_id, "sessionId": blue_id, "evidence": ["stress: 1000 pass"] }),
        );
        rig.ok(
            "cmd-retest-phase",
            "advance",
            json!({ "to": "red_retesting" }),
        );
        rig.ok(
            "cmd-retest",
            "record_retest",
            json!({ "findingId": finding_id, "sessionId": red_id, "result": "fixed", "evidence": ["original repro: pass"] }),
        );
        rig.ok(
            "cmd-review-phase",
            "advance",
            json!({ "to": "referee_review" }),
        );
        rig.ok(
            "cmd-review",
            "begin_referee_review",
            json!({ "sessionId": referee_id }),
        );
        rig.ok(
            "cmd-verdict",
            "record_verdict",
            json!({
                "sessionId": referee_id, "verdict": "verified", "evidence": ["independent harness: pass"],
                "rationale": "definition of done reproduced independently",
            }),
        );
        rig.ok("cmd-verified", "advance", json!({ "to": "verified" }));
        let checkpoint = rig.ok(
            "cmd-checkpoint",
            "checkpoint",
            json!({ "name": "gateway verified", "revision": "abc1234" }),
        );
        let checkpoint_id = checkpoint["checkpointId"].clone();
        let early = err_of(rig.act(
            "cmd-early-rollback",
            "rollback",
            json!({ "checkpointId": checkpoint_id, "reason": "too early" }),
        ));
        assert!(
            early.message.contains("requires a promoted campaign"),
            "{}",
            early.message
        );
        rig.ok(
            "cmd-promote",
            "promote",
            json!({
                "checkpointId": checkpoint_id,
                "capabilities": [{ "id": "gateway-race-safe", "name": "Race-safe release gateway", "evidence": ["verdict"] }],
            }),
        );

        let final_view = rig.view();
        assert_eq!(final_view["phase"], json!("promoted"));
        for team in ["blue", "red", "referee"] {
            assert_eq!(
                final_view["teams"][team]["members"]
                    .as_array()
                    .unwrap()
                    .len(),
                1,
                "{team}"
            );
        }
        assert_eq!(final_view["findings"][0]["status"], json!("confirmed"));
        assert_eq!(
            final_view["capabilities"][0]["id"],
            json!("gateway-race-safe")
        );

        let before = rig.stored.lock().unwrap().len();
        let duplicate = rig.ok("cmd-promote", "promote", json!({}));
        assert_eq!(duplicate["duplicate"], json!(true));
        assert_eq!(rig.stored.lock().unwrap().len(), before);

        let cross = err_of(rig.act(
            "cmd-cross-campaign",
            "assign_team",
            json!({ "team": "referee", "sessionId": blue_id, "objectiveId": objective_id, "external": true }),
        ));
        assert_eq!(cross.code, "role_conflict");
        assert!(
            cross.message.contains("cannot serve on both"),
            "{}",
            cross.message
        );

        rig.ok(
            "cmd-rollback",
            "rollback",
            json!({ "checkpointId": checkpoint_id, "reason": "injected post-promotion regression" }),
        );
        let campaign = rig.campaign();
        assert_eq!(campaign["phase"], json!("rolled_back"));
        assert_eq!(campaign["rollbackCheckpointId"], checkpoint_id);
    }

    #[test]
    fn session_reports_become_campaign_actions_with_team_gates() {
        let rig = rig();
        let created = rig.create_campaign();
        let objective_id = js_string(&created["objectiveIds"][0]);
        let mobilized = rig.ok(
            "cmd-mobilize",
            "mobilize",
            json!({ "roster": [
                { "team": "blue", "agentId": "blue-agent" },
                { "team": "red", "agentId": "red-agent" },
            ] }),
        );
        let blue_id = mobilized["spawned"][0].clone();
        rig.ok("cmd-blue", "advance", json!({ "to": "blue_building" }));

        let envelope = |team: &str, report: Value| {
            json!({
                "sessionId": blue_id, "eventSeq": 7, "campaignId": "campaign-1",
                "objectiveId": objective_id, "team": team, "report": report,
            })
        };
        rig.director
            .ingest_session_report(envelope(
                "blue",
                json!({ "kind": "objective_satisfied", "evidence": ["stress: pass"] }),
            ))
            .expect("blue evidence is accepted");
        {
            let p = rig.projection.lock().unwrap();
            let objective = p.campaigns.objectives.get(&objective_id).unwrap();
            assert_eq!(objective["status"], json!("satisfied"));
            assert!(p.campaigns.command_results.contains("report:session-1:7"));
        }
        // The same report replayed is the same command: no second fold.
        let before = rig.stored.lock().unwrap().len();
        let replay = rig
            .director
            .ingest(&envelope(
                "blue",
                json!({ "kind": "objective_satisfied", "evidence": ["stress: pass"] }),
            ))
            .unwrap();
        assert_eq!(replay["duplicate"], json!(true));
        assert_eq!(rig.stored.lock().unwrap().len(), before);

        let rejected = rig
            .director
            .ingest_session_report(envelope(
                "blue",
                json!({ "kind": "finding", "claim": "x", "evidence": ["e"], "reproduction": ["r"] }),
            ))
            .unwrap_err();
        assert_eq!(rejected, "only red can submit a finding report");
        let unknown = rig
            .director
            .ingest_session_report(envelope("blue", json!({})))
            .unwrap_err();
        assert_eq!(unknown, "unknown FIELD_REPORT kind: (missing)");
    }

    #[test]
    fn pause_gates_actions_and_resume_reaches_live_members() {
        let rig = rig();
        rig.create_campaign();
        let mobilized = rig.ok(
            "cmd-mobilize",
            "mobilize",
            json!({ "roster": [{ "team": "blue", "agentId": "blue-agent" }] }),
        );
        let blue_id = mobilized["spawned"][0].clone();

        let paused = rig.ok("cmd-pause", "pause", json!({ "reason": "operator review" }));
        assert_eq!(paused["sessionIds"], json!([blue_id]));
        let refused = err_of(rig.act("cmd-advance", "advance", json!({ "to": "blue_building" })));
        assert_eq!(refused.code, "campaign_paused");
        assert_eq!(refused.detail["currentPhase"], json!("mobilizing"));
        // The paused gate runs before dispatch and admits only resume,
        // cancel and rollback, so a second pause is refused rather than
        // short-circuited, and no second registry command is issued.
        assert_eq!(
            err_of(rig.act("cmd-pause-2", "pause", json!({}))).code,
            "campaign_paused"
        );

        let resumed = rig.ok("cmd-resume", "resume", json!({ "orders": "carry on" }));
        assert_eq!(resumed["resumed"], json!(true));
        let calls = rig.calls();
        let resume = calls.iter().find(|(kind, _)| kind == "resume").unwrap();
        assert_eq!(resume.1["orders"], json!("carry on"));
        assert_eq!(calls.iter().filter(|(kind, _)| kind == "pause").count(), 1);
        assert_eq!(
            err_of(rig.act("cmd-resume-2", "resume", json!({}))).code,
            "not_paused"
        );
    }
}
