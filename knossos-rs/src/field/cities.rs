//! Cities: a city IS a workspace. Port of `field/server/src/cities.js`.
//!
//! Everything here is derived from the live projection and the event log;
//! nothing is written back to `field/` (operational state stays in the log).
//! Leveling reuses the per-workspace `maturity` score (campaign completion,
//! verification, persistence) rather than inventing a parallel XP system, so
//! a city's rank reflects the same evidence the rest of Field trusts.

use super::config::FieldSettings;
use super::eventlog::EventLog;
use super::js::{get, get_str, js_string, Obj};
use super::projection::{workspace_maturity, Projection};
use serde_json::{json, Value};
use std::collections::HashSet;

const ACTIVE_STATES: &[&str] = &["spawning", "thinking", "working", "waiting_permission"];

pub fn is_active_state(state: &str) -> bool {
    ACTIVE_STATES.contains(&state)
}

/// `maturity.tier` is 0..4; name the ranks a civilization would recognise.
const TIER_NAMES: &[&str] = &["outpost", "outpost", "town", "city", "capital"];

fn state_of(session: &Obj) -> &str {
    session.get("state").and_then(Value::as_str).unwrap_or("")
}

fn city_sessions<'a>(projection: &'a Projection, workspace_id: &str) -> Vec<&'a Obj> {
    projection
        .sessions()
        .filter(|s| s.get("workspaceId").and_then(Value::as_str) == Some(workspace_id))
        .collect()
}

fn rank_of(maturity: &Value) -> Vec<(&'static str, Value)> {
    let tier = maturity
        .get("tier")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        .clamp(0, i64::MAX) as usize;
    vec![
        (
            "tier",
            json!(TIER_NAMES.get(tier).copied().unwrap_or("outpost")),
        ),
        ("level", json!(tier)),
        ("score", maturity.get("score").cloned().unwrap_or(json!(0))),
    ]
}

fn summarise_session(s: &Obj) -> Value {
    let field = |k: &str| s.get(k).cloned().unwrap_or(Value::Null);
    json!({
        "sessionId": field("id"), "name": field("name"), "role": field("role"), "state": field("state"),
        "endpointId": field("endpointId"), "contextPct": field("contextPct"), "costUsd": field("costUsd"),
        "verified": field("verified"), "lastSay": field("lastSay"), "active": is_active_state(state_of(s)),
    })
}

fn campaign_list(projection: &Projection) -> Vec<Value> {
    projection
        .campaigns
        .snapshot()
        .get("campaigns")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn with_rank(mut card: Value, maturity: &Value) -> Value {
    if let Some(obj) = card.as_object_mut() {
        for (k, v) in rank_of(maturity) {
            obj.insert(k.to_string(), v);
        }
    }
    card
}

/// Compact card for every city; cheap enough for a snapshot summary.
pub fn list_cities(projection: &Projection, now: i64) -> Vec<Value> {
    let campaigns = campaign_list(projection);
    projection
        .workspaces()
        .map(|w| {
            let maturity = workspace_maturity(w, &campaigns, now);
            let id = w.get("id").map(js_string).unwrap_or_default();
            let sessions = city_sessions(projection, &id);
            let card = json!({
                "id": id, "name": w.get("name").cloned().unwrap_or(Value::Null),
                "mounted": w.get("mounted").cloned().unwrap_or(Value::Null),
                "agentCount": sessions.iter().filter(|s| is_active_state(state_of(s))).count(),
                "sessionCount": sessions.len(),
            });
            with_rank(card, &maturity)
        })
        .collect()
}

/// Full city view including roster and the chat/order feed.
pub fn city_detail(projection: &Projection, log: &EventLog, id: &str, now: i64) -> Option<Value> {
    let w = projection.workspace(id)?;
    let campaigns = campaign_list(projection);
    let maturity = workspace_maturity(w, &campaigns, now);
    let sessions = city_sessions(projection, id);
    let card = json!({
        "id": w.get("id").cloned().unwrap_or(Value::Null), "name": w.get("name").cloned().unwrap_or(Value::Null),
        "path": w.get("path").cloned().unwrap_or(Value::Null), "mounted": w.get("mounted").cloned().unwrap_or(Value::Null),
        "maturity": maturity,
        "agents": sessions.iter().map(|s| summarise_session(s)).collect::<Vec<_>>(),
        "agentCount": sessions.iter().filter(|s| is_active_state(state_of(s))).count(),
        "feed": city_feed(projection, log, id, 60),
    });
    Some(with_rank(card, &maturity))
}

pub fn active_city_session_ids(projection: &Projection, workspace_id: &str) -> Vec<String> {
    city_sessions(projection, workspace_id)
        .into_iter()
        .filter(|s| is_active_state(state_of(s)))
        .filter_map(|s| s.get("id").map(js_string))
        .collect()
}

/// Resolve a default agent for a role, so the operator can deploy without
/// picking one.
pub fn resolve_city_agent(settings: &FieldSettings, role: Option<&str>) -> Option<String> {
    let want = role
        .filter(|r| !r.is_empty())
        .map(str::to_string)
        .or_else(|| get_str(&settings.defaults, "role").map(str::to_string))
        .unwrap_or_else(|| "builder".to_string());
    settings
        .agents
        .iter()
        .find(|a| get_str(a, "role") == Some(&want))
        .or_else(|| settings.agents.first())
        .and_then(|a| get_str(a, "id").map(str::to_string))
}

// ---- feed: a bounded tail of the event log, filtered to this city's sessions ----

const FEED_KINDS: &[&str] = &[
    "session.message",
    "session.spawned",
    "session.ended",
    "work.verified",
    "command.issued",
];
const FEED_TAIL_EVENTS: u64 = 1500;

fn feed_text(kind: &str, d: &Value) -> String {
    let s = |k: &str| get(d, k).map(js_string);
    match kind {
        "session.message" => s("text").unwrap_or_default().chars().take(600).collect(),
        "session.spawned" => format!(
            "deployed {}",
            s("name")
                .or_else(|| s("agentId"))
                .unwrap_or_else(|| "agent".into())
        ),
        "session.ended" => format!("ended ({})", s("reason").unwrap_or_else(|| "done".into())),
        "work.verified" => format!(
            "verification: {}",
            s("result").unwrap_or_else(|| "?".into())
        ),
        "command.issued" => {
            let text = s("text")
                .map(|t| format!(" — {}", t.chars().take(200).collect::<String>()))
                .unwrap_or_default();
            format!("order: {}{}", s("kind").unwrap_or_default(), text)
        }
        _ => String::new(),
    }
}

fn city_feed(
    projection: &Projection,
    log: &EventLog,
    workspace_id: &str,
    limit: usize,
) -> Vec<Value> {
    let ids: HashSet<String> = city_sessions(projection, workspace_id)
        .into_iter()
        .filter_map(|s| s.get("id").map(js_string))
        .collect();
    if ids.is_empty() {
        return Vec::new();
    }
    let from = log.size().saturating_sub(FEED_TAIL_EVENTS);
    let events = log
        .read(from, FEED_TAIL_EVENTS as usize)
        .unwrap_or_default();
    let mut feed = Vec::new();
    for e in events {
        if !FEED_KINDS.contains(&e.kind.as_str()) {
            continue;
        }
        let d = &e.data;
        let sid = get(d, "sessionId").map(js_string).or_else(|| {
            get(d, "sessionIds")
                .and_then(Value::as_array)
                .and_then(|a| a.first())
                .map(js_string)
        });
        let Some(sid) = sid.filter(|s| ids.contains(s)) else {
            continue;
        };
        feed.push(json!({
            "at": e.ts, "seq": e.seq, "kind": e.kind, "sessionId": sid,
            "role": get(d, "role").cloned().unwrap_or(Value::Null),
            "text": feed_text(&e.kind, d),
        }));
    }
    let skip = feed.len().saturating_sub(limit);
    feed.into_iter().skip(skip).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feed_text_matches_the_node_wording() {
        assert_eq!(
            feed_text("session.spawned", &json!({ "agentId": "builder-1" })),
            "deployed builder-1"
        );
        assert_eq!(feed_text("session.ended", &json!({})), "ended (done)");
        assert_eq!(
            feed_text("command.issued", &json!({ "kind": "say", "text": "go" })),
            "order: say — go"
        );
        assert_eq!(feed_text("work.verified", &json!({})), "verification: ?");
        assert_eq!(feed_text("other", &json!({})), "");
    }

    #[test]
    fn ranks_follow_maturity_tiers() {
        let names: Vec<_> = (0..6)
            .map(|t| rank_of(&json!({ "tier": t, "score": t * 10 }))[0].1.clone())
            .collect();
        assert_eq!(
            names,
            vec![
                json!("outpost"),
                json!("outpost"),
                json!("town"),
                json!("city"),
                json!("capital"),
                json!("outpost")
            ]
        );
        assert_eq!(rank_of(&Value::Null)[2].1, json!(0));
    }
}
