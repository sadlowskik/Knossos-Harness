//! The campaign state machine and its validators. Port of
//! `field/server/src/orchestration/model.js`.

use super::js::{get, get_arr, get_str, is_integer, jnum, js_string, number, truthy};
use regex::Regex;
use serde_json::{json, Value};
use std::sync::OnceLock;

pub const CAMPAIGN_PHASES: [&str; 13] = [
    "draft",
    "mobilizing",
    "blue_building",
    "red_challenging",
    "contested",
    "blue_mitigating",
    "red_retesting",
    "referee_review",
    "verified",
    "promoted",
    "failed",
    "cancelled",
    "rolled_back",
];

pub const TEAM_KINDS: [&str; 4] = ["blue", "red", "purple", "referee"];
pub const SEVERITIES: [&str; 5] = ["info", "low", "medium", "high", "critical"];
pub const FINDING_STATES: [&str; 8] = [
    "open",
    "acknowledged",
    "disputed",
    "mitigating",
    "ready_for_retest",
    "confirmed",
    "rejected",
    "waived",
];
pub const VERDICTS: [&str; 3] = ["verified", "rejected", "inconclusive"];
pub const ENVIRONMENT_SCOPES: [&str; 5] = [
    "snapshot",
    "sandbox",
    "staging",
    "production-readonly",
    "production-approved",
];

/// `paused` is a campaign flag rather than a stored phase; it appears here
/// only as an operator-facing legal action.
pub fn transitions(phase: &str) -> &'static [&'static str] {
    match phase {
        "draft" => &["mobilizing", "cancelled"],
        "mobilizing" => &["blue_building", "paused", "failed", "cancelled"],
        "blue_building" => &["red_challenging", "paused", "failed", "cancelled"],
        "red_challenging" => &[
            "contested",
            "referee_review",
            "paused",
            "failed",
            "cancelled",
        ],
        "contested" => &["blue_mitigating", "paused", "failed", "cancelled"],
        "blue_mitigating" => &["red_retesting", "paused", "failed", "cancelled"],
        "red_retesting" => &[
            "referee_review",
            "blue_mitigating",
            "paused",
            "failed",
            "cancelled",
        ],
        "referee_review" => &[
            "verified",
            "blue_building",
            "blue_mitigating",
            "paused",
            "failed",
            "cancelled",
        ],
        "verified" => &["promoted", "blue_building", "cancelled"],
        "promoted" => &["rolled_back"],
        "failed" => &["mobilizing", "cancelled"],
        "cancelled" => &[],
        "rolled_back" => &["blue_building", "cancelled"],
        _ => &[],
    }
}

fn allowed_from(phase: &str) -> Vec<&'static str> {
    transitions(phase)
        .iter()
        .copied()
        .filter(|t| *t != "paused")
        .collect()
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct DomainError {
    pub code: String,
    pub message: String,
    pub detail: Value,
}

impl DomainError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        DomainError {
            code: code.to_string(),
            message: message.into(),
            detail: json!({}),
        }
    }

    fn with_detail(mut self, detail: Value) -> Self {
        self.detail = detail;
        self
    }
}

fn gate(message: impl Into<String>) -> DomainError {
    DomainError::new("gate_blocked", message)
}

pub fn assert_id(value: &Value, name: &str) -> Result<String, DomainError> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"^[a-zA-Z0-9._:-]+$").expect("static regex"));
    match value.as_str() {
        Some(s) if !s.is_empty() && s.len() <= 160 && re.is_match(s) => Ok(s.to_string()),
        _ => Err(DomainError::new(
            "invalid_id",
            format!("{name} must be 1-160 URL-safe characters"),
        )
        .with_detail(json!({ "name": name }))),
    }
}

fn text(value: Option<&Value>) -> String {
    value.map(js_string).unwrap_or_default()
}

fn bounded_int(
    value: Option<&Value>,
    default: f64,
    min: f64,
    max: f64,
    name: &str,
) -> Result<f64, DomainError> {
    let n = match value.filter(|v| !v.is_null()) {
        Some(v) => number(Some(v)),
        None => default,
    };
    let integer = n.is_finite() && n.fract() == 0.0;
    if !integer || n < min || n > max {
        return Err(DomainError::new(
            "invalid_number",
            format!(
                "{name} must be an integer between {} and {}",
                super::js::format_number(min),
                super::js::format_number(max)
            ),
        ));
    }
    Ok(n)
}

fn bounded_number(
    value: Option<&Value>,
    default: f64,
    min: f64,
    max: f64,
    name: &str,
) -> Result<f64, DomainError> {
    let n = match value.filter(|v| !v.is_null()) {
        Some(v) => number(Some(v)),
        None => default,
    };
    if !n.is_finite() || n < min || n > max {
        return Err(DomainError::new(
            "invalid_number",
            format!(
                "{name} must be between {} and {}",
                super::js::format_number(min),
                super::js::format_number(max)
            ),
        ));
    }
    Ok(n)
}

pub fn validate_campaign_input(input: &Value) -> Result<Value, DomainError> {
    let name = text(input.get("name")).trim().to_string();
    let intent = text(input.get("intent")).trim().to_string();
    if name.is_empty() || name.chars().count() > 160 {
        return Err(DomainError::new(
            "invalid_campaign",
            "campaign name is required and must be at most 160 characters",
        ));
    }
    if intent.is_empty() || intent.chars().count() > 8000 {
        return Err(DomainError::new(
            "invalid_campaign",
            "campaign intent is required and must be at most 8000 characters",
        ));
    }
    let scope = get(input, "scope").cloned().unwrap_or(json!("sandbox"));
    let scope_text = js_string(&scope);
    if !scope.is_string() || !ENVIRONMENT_SCOPES.contains(&scope_text.as_str()) {
        return Err(DomainError::new(
            "invalid_scope",
            format!("unknown campaign scope: {scope_text}"),
        ));
    }
    let concurrency = bounded_int(input.get("concurrency"), 8.0, 1.0, 100.0, "concurrency")?;
    let budget_usd = bounded_number(input.get("budgetUsd"), 25.0, 0.0, 1_000_000.0, "budgetUsd")?;
    let objectives = get_arr(input, "objectives").cloned().unwrap_or_default();
    if objectives.is_empty() {
        return Err(DomainError::new(
            "missing_objectives",
            "a campaign requires at least one objective",
        ));
    }
    let mut validated = Vec::with_capacity(objectives.len());
    for (i, objective) in objectives.iter().enumerate() {
        validated.push(validate_objective(objective, i)?);
    }
    Ok(json!({
        "name": name,
        "intent": intent,
        "scope": scope_text,
        "concurrency": jnum(concurrency),
        "budgetUsd": jnum(budget_usd),
        "doctrine": normalize_doctrine(input.get("doctrine"))?,
        "target": normalize_target(input.get("target")),
        "objectives": validated,
    }))
}

pub fn validate_objective(input: &Value, index: usize) -> Result<Value, DomainError> {
    let statement = text(get(input, "statement").or(get(input, "name")))
        .trim()
        .to_string();
    if statement.is_empty() || statement.chars().count() > 2000 {
        return Err(DomainError::new(
            "invalid_objective",
            format!(
                "objective {} needs a statement of at most 2000 characters",
                index + 1
            ),
        ));
    }
    let done: Vec<String> = get_arr(input, "definitionOfDone")
        .map(|items| {
            items
                .iter()
                .map(|x| js_string(x).trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if done.is_empty() {
        return Err(DomainError::new(
            "invalid_objective",
            format!("objective {} needs a definition of done", index + 1),
        ));
    }
    let priority = bounded_int(
        input.get("priority"),
        (index + 1) as f64,
        1.0,
        10_000.0,
        "priority",
    )?;
    let dependencies: Vec<String> = get_arr(input, "dependencies")
        .map(|items| items.iter().map(js_string).take(100).collect())
        .unwrap_or_default();
    Ok(json!({
        "statement": statement,
        "definitionOfDone": done.into_iter().take(40).collect::<Vec<_>>(),
        "priority": jnum(priority),
        "risk": get(input, "risk").map(js_string).unwrap_or_else(|| "medium".to_string()),
        "target": normalize_target(input.get("target")),
        "dependencies": dependencies,
    }))
}

pub fn validate_finding(input: &Value) -> Result<Value, DomainError> {
    let severity = get(input, "severity").cloned().unwrap_or(json!("medium"));
    let severity_text = js_string(&severity);
    if !severity.is_string() || !SEVERITIES.contains(&severity_text.as_str()) {
        return Err(DomainError::new(
            "invalid_severity",
            format!("unknown finding severity: {severity_text}"),
        ));
    }
    let claim = text(input.get("claim")).trim().to_string();
    if claim.is_empty() || claim.chars().count() > 8000 {
        return Err(DomainError::new(
            "invalid_finding",
            "finding claim is required and must be at most 8000 characters",
        ));
    }
    let evidence: Vec<Value> = get_arr(input, "evidence")
        .map(|items| {
            items
                .iter()
                .filter(|v| truthy(v))
                .take(100)
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    let reproduction: Vec<String> = get_arr(input, "reproduction")
        .map(|items| {
            items
                .iter()
                .map(js_string)
                .filter(|s| !s.is_empty())
                .take(100)
                .collect()
        })
        .unwrap_or_default();
    let no_reproduction_reason = text(input.get("noReproductionReason")).trim().to_string();
    if evidence.is_empty() {
        return Err(DomainError::new(
            "missing_evidence",
            "a finding requires evidence",
        ));
    }
    if reproduction.is_empty() && no_reproduction_reason.is_empty() {
        return Err(DomainError::new(
            "missing_reproduction",
            "a finding requires reproduction steps or a reason they are unavailable",
        ));
    }
    let clip = |s: String, n: usize| s.chars().take(n).collect::<String>();
    Ok(json!({
        "severity": severity_text,
        "category": clip(get(input, "category").map(js_string).unwrap_or_else(|| "correctness".into()), 80),
        "claim": claim,
        "scope": clip(get(input, "scope").map(js_string).unwrap_or_default(), 1000),
        "evidence": evidence,
        "reproduction": reproduction,
        "noReproductionReason": if no_reproduction_reason.is_empty() { Value::Null } else { json!(no_reproduction_reason) },
        "confidence": jnum(bounded_number(input.get("confidence"), 0.8, 0.0, 1.0, "confidence")?),
    }))
}

/// What the projection knows about a campaign when a gate is evaluated.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Context {
    pub objectives: Vec<Value>,
    pub findings: Vec<Value>,
    pub verdicts: Vec<Value>,
    pub checkpoints: Vec<Value>,
    pub latest_verdict: Option<String>,
    pub checkpoint_id: Option<String>,
    pub checkpoint_revision: Option<Value>,
}

fn member_available(member: &Value) -> bool {
    matches!(get_str(member, "status"), Some("active" | "external"))
}

pub fn validate_transition(
    campaign: &Value,
    to: &str,
    context: &Context,
) -> Result<(), DomainError> {
    if campaign.is_null() {
        return Err(DomainError::new("not_found", "campaign does not exist"));
    }
    if !CAMPAIGN_PHASES.contains(&to) || to == "paused" {
        return Err(DomainError::new(
            "invalid_phase",
            format!("unknown campaign phase: {to}"),
        ));
    }
    let from = get_str(campaign, "phase").unwrap_or("");
    let allowed = allowed_from(from);
    if !allowed.contains(&to) {
        return Err(DomainError::new(
            "invalid_transition",
            format!("cannot advance campaign from {from} to {to}"),
        )
        .with_detail(json!({ "currentPhase": from, "allowedPhases": allowed })));
    }
    let objectives = &context.objectives;
    let findings = &context.findings;
    let doctrine = campaign.get("doctrine").cloned().unwrap_or(json!({}));
    let blocking = |f: &Value| is_blocking_finding(f, &doctrine);
    if (from == "draft" || from == "failed") && to == "mobilizing" {
        if objectives.is_empty() {
            return Err(gate("campaign has no objectives"));
        }
        let blue_ready = campaign
            .get("teams")
            .and_then(|t| t.get("blue"))
            .and_then(|b| get_arr(b, "members"))
            .is_some_and(|members| members.iter().any(member_available));
        if !blue_ready {
            return Err(gate(
                "campaign has no active or explicitly external blue team members",
            ));
        }
    }
    if from == "blue_building" && to == "red_challenging" {
        let missing = objectives
            .iter()
            .filter(|o| {
                get_bool_ne_false(o, "required") && get_str(o, "status") != Some("satisfied")
            })
            .count();
        if missing > 0 {
            return Err(gate(format!(
                "{missing} required objectives are not ready for challenge"
            )));
        }
    }
    if from == "red_challenging" && to == "contested" && !findings.iter().any(blocking) {
        return Err(gate("no blocking red finding exists"));
    }
    if from == "red_challenging" && to == "referee_review" && findings.iter().any(blocking) {
        return Err(gate("blocking findings require mitigation first"));
    }
    if from == "contested" && to == "blue_mitigating" {
        let untriaged = findings
            .iter()
            .filter(|f| blocking(f) && get_str(f, "status") == Some("open"))
            .count();
        if untriaged > 0 {
            return Err(gate(format!(
                "{untriaged} blocking findings are not acknowledged or disputed"
            )));
        }
    }
    if from == "blue_mitigating" && to == "red_retesting" {
        let unready = findings
            .iter()
            .filter(|f| {
                blocking(f)
                    && !matches!(
                        get_str(f, "status"),
                        Some("ready_for_retest" | "rejected" | "waived")
                    )
            })
            .count();
        if unready > 0 {
            return Err(gate(format!(
                "{unready} blocking findings have no ready mitigation"
            )));
        }
    }
    if from == "red_retesting" && to == "referee_review" {
        let unresolved = findings
            .iter()
            .filter(|f| {
                blocking(f)
                    && !matches!(
                        get_str(f, "status"),
                        Some("confirmed" | "rejected" | "waived")
                    )
            })
            .count();
        if unresolved > 0 {
            return Err(gate(format!(
                "{unresolved} blocking findings lack a completed retest"
            )));
        }
    }
    if from == "referee_review"
        && to == "verified"
        && context.latest_verdict.as_deref() != Some("verified")
    {
        return Err(gate("an independent verified referee verdict is required"));
    }
    if from == "verified"
        && to == "promoted"
        && (context.checkpoint_id.as_deref().is_none_or(str::is_empty)
            || !context
                .checkpoint_revision
                .as_ref()
                .is_some_and(is_content_revision))
    {
        return Err(gate("promotion requires a revision-bound checkpoint"));
    }
    Ok(())
}

/// `x.required !== false`.
fn get_bool_ne_false(value: &Value, key: &str) -> bool {
    value.get(key) != Some(&Value::Bool(false))
}

pub fn is_content_revision(value: &Value) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re =
        RE.get_or_init(|| Regex::new(r"(?i)^(?:sha256:)?[a-f0-9]{7,64}$").expect("static regex"));
    value.as_str().is_some_and(|s| re.is_match(s.trim()))
}

pub fn legal_actions(campaign: &Value, context: &Context) -> Vec<String> {
    if campaign.is_null() {
        return Vec::new();
    }
    let paused = campaign.get("paused").is_some_and(truthy);
    let mut out: Vec<String> = if paused {
        vec!["resume".into(), "cancel".into()]
    } else {
        vec!["pause".into(), "cancel".into()]
    };
    let phase = get_str(campaign, "phase").unwrap_or("");
    for next in allowed_from(phase) {
        if validate_transition(campaign, next, context).is_ok() {
            out.push(format!("advance:{next}"));
        }
    }
    if !paused && !matches!(phase, "cancelled" | "promoted") {
        for action in ["assign_team", "issue_orders", "reinforce", "retreat"] {
            out.push(action.into());
        }
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|a| seen.insert(a.clone()));
    out
}

/// Every phase reachable from the current phase, including the reason a
/// gate is closed.
pub fn transition_options(campaign: &Value, context: &Context) -> Vec<Value> {
    if campaign.is_null() {
        return Vec::new();
    }
    let phase = get_str(campaign, "phase").unwrap_or("");
    allowed_from(phase)
        .into_iter()
        .map(|to| match validate_transition(campaign, to, context) {
            Ok(()) => json!({ "to": to, "legal": true, "reason": Value::Null }),
            Err(error) => {
                json!({ "to": to, "legal": false, "reason": error.message, "code": error.code })
            }
        })
        .collect()
}

pub fn is_blocking_finding(finding: &Value, doctrine: &Value) -> bool {
    if finding.is_null()
        || matches!(
            get_str(finding, "status"),
            Some("confirmed" | "rejected" | "waived")
        )
    {
        return false;
    }
    let threshold = get_str(doctrine, "blockingSeverity").unwrap_or("high");
    let index = |s: Option<&str>| {
        s.and_then(|s| SEVERITIES.iter().position(|x| *x == s))
            .map(|i| i as i64)
            .unwrap_or(-1)
    };
    index(get_str(finding, "severity")) >= index(Some(threshold))
}

pub fn assert_team(kind: &str) -> Result<&str, DomainError> {
    if TEAM_KINDS.contains(&kind) {
        Ok(kind)
    } else {
        Err(DomainError::new(
            "invalid_team",
            format!("unknown team: {kind}"),
        ))
    }
}

pub fn normalize_doctrine(input: Option<&Value>) -> Result<Value, DomainError> {
    let input = input.filter(|v| !v.is_null());
    let blocking = input
        .and_then(|i| get(i, "blockingSeverity"))
        .cloned()
        .unwrap_or(json!("high"));
    let blocking_text = js_string(&blocking);
    if !blocking.is_string() || !SEVERITIES.contains(&blocking_text.as_str()) {
        return Err(DomainError::new(
            "invalid_doctrine",
            format!("unknown blocking severity: {blocking_text}"),
        ));
    }
    let max_age = bounded_int(
        input.and_then(|i| i.get("maxFindingAgeMs")),
        86_400_000.0,
        1000.0,
        31_536_000_000.0,
        "maxFindingAgeMs",
    )?;
    let red_categories: Vec<String> = match input.and_then(|i| get_arr(i, "redCategories")) {
        Some(items) => items.iter().map(js_string).take(30).collect(),
        None => ["security", "reliability", "correctness", "rollback"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    Ok(json!({
        "blockingSeverity": blocking_text,
        "requireRed": input.and_then(|i| i.get("requireRed")) != Some(&Value::Bool(false)),
        "requireReferee": input.and_then(|i| i.get("requireReferee")) != Some(&Value::Bool(false)),
        "maxFindingAgeMs": jnum(max_age),
        "redCategories": red_categories,
    }))
}

pub fn normalize_target(input: Option<&Value>) -> Value {
    let Some(input) = input.filter(|v| truthy(v)) else {
        return Value::Null;
    };
    let clip = |s: String, n: usize| s.chars().take(n).collect::<String>();
    json!({
        "type": clip(get(input, "type").map(js_string).unwrap_or_else(|| "workspace".into()), 40),
        "id": clip(get(input, "id").map(js_string).unwrap_or_default(), 1000),
        "label": get(input, "label").map(|l| json!(clip(js_string(l), 300))).unwrap_or(Value::Null),
        "workspaceId": get(input, "workspaceId").map(|w| json!(clip(js_string(w), 160))).unwrap_or(Value::Null),
    })
}

#[allow(dead_code)]
fn _integer_probe(v: &Value) -> bool {
    is_integer(v)
}
