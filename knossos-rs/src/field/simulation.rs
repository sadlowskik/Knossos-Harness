//! Synthetic world rehearsals. Port of
//! `field/server/src/simulation/field-simulator.js`.
//!
//! A rehearsal replays a pure, pre-authored timeline of `simulated` events
//! into the log at a chosen speed: twelve agents muster, fan out,
//! coordinate, recover from drift, verify and report. Nothing here calls a
//! provider, spawns a process or touches a workspace; every event carries
//! `simulated: true` and lands in the synthetic partition of the projection.

use super::eventlog::AppendOptions;
use super::registry::Emit;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const TIMELINE_SCALE: i64 = 3;
const OPERATIONS_DURATION_MS: i64 = 24_400 * TIMELINE_SCALE;

/// (key, name, role, thinking)
const UNITS: &[(&str, &str, &str, &str)] = &[
    ("praetor", "Praetor", "architect", "high"),
    ("forge-a", "Forge I", "builder", "medium"),
    ("forge-b", "Forge II", "builder", "medium"),
    ("forge-c", "Forge III", "builder", "low"),
    ("pathfinder", "Pathfinder", "scout", "low"),
    ("courier", "Courier", "scout", "low"),
    ("sentinel-a", "Sentinel I", "verifier", "high"),
    ("sentinel-b", "Sentinel II", "verifier", "medium"),
    ("challenger", "Adversary", "challenger", "high"),
    ("archivist", "Archivist", "archivist", "low"),
    ("relay-a", "Relay I", "builder", "low"),
    ("relay-b", "Relay II", "builder", "low"),
];

/// key -> (dir, path, tool)
fn work(key: &str) -> (&'static str, &'static str, &'static str) {
    match key {
        "praetor" => ("docs", "docs/architecture.md", "Read"),
        "forge-a" => ("src/core", "src/core/service.ts", "Read"),
        "forge-b" => ("src/ui", "src/ui/App.tsx", "Edit"),
        "forge-c" => ("src/agents", "src/agents/orchestrator.ts", "Edit"),
        "pathfinder" => ("docs", "docs/architecture.md", "Grep"),
        "courier" => ("site", "site/index.html", "Read"),
        "sentinel-a" => ("tests", "tests/integration.test.ts", "Bash"),
        "sentinel-b" => ("tests/e2e", "tests/e2e/release.test.ts", "Bash"),
        "challenger" => ("src/core", "src/core/api.ts", "Read"),
        "archivist" => ("docs", "docs/operations.md", "Write"),
        "relay-a" => ("infra", "infra/compose.yaml", "Read"),
        _ => ("deploy", "deploy/service.yaml", "Read"),
    }
}

const STEPS: [&str; 5] = ["survey", "plan", "execute", "verify", "report"];

#[derive(Debug, Clone)]
pub struct TimelineItem {
    /// Milliseconds after the run starts, at speed 1.
    pub at: i64,
    pub kind: String,
    pub data: Value,
    pub subject: String,
}

fn item(at: i64, kind: &str, mut data: Value, subject: &str) -> TimelineItem {
    if let Some(obj) = data.as_object_mut() {
        obj.insert("simulated".into(), json!(true));
    }
    TimelineItem {
        at,
        kind: kind.to_string(),
        data,
        subject: subject.to_string(),
    }
}

/// Pure timeline so safety can be tested without clocks or provider access.
pub fn operations_timeline(
    run_id: &str,
    workspace_id: &str,
    workspace_name: &str,
) -> Vec<TimelineItem> {
    let id = |key: &str| format!("sim-{run_id}-{key}");
    let mut events = Vec::new();
    let progress = |at: i64, session_id: &str, done: i64| {
        item(
            at,
            "session.progress",
            json!({ "sessionId": session_id, "done": done, "total": 5, "steps": STEPS, "simulationRunId": run_id }),
            session_id,
        )
    };

    for (i, (key, name, role, thinking)) in UNITS.iter().enumerate() {
        let i = i as i64;
        let sid = id(key);
        events.push(item(i * 110, "session.spawned", json!({
            "sessionId": sid, "agentId": format!("simulation-{key}"), "name": name, "role": role,
            "model": format!("rehearsal-{role}"), "endpointId": "rehearsal", "thinking": thinking,
            "workspaceId": workspace_id, "simulationRunId": run_id,
            "initialOrders": format!("Serve as {name}, the {role} assigned to {workspace_name}. Survey your lane, coordinate with the cohort, verify evidence, and report a durable result."),
            "systemPrompt": format!("You are a synthetic {role} in Cameo's safe operations rehearsal. Make no provider calls or filesystem writes. Report simulated observations clearly to the other agents."),
        }), &sid));
        events.push(item(900 + i * 45, "session.state", json!({
            "sessionId": sid, "state": "ready", "detail": "Awaiting formation orders", "simulationRunId": run_id,
        }), &sid));
    }

    events.push(item(
        1_700,
        "session.message",
        json!({
            "sessionId": id("praetor"), "role": "assistant",
            "text": "Formation online. Splitting reconnaissance, build, and verification lanes.",
            "simulationRunId": run_id,
        }),
        &id("praetor"),
    ));

    for (i, (key, _, _, _)) in UNITS.iter().enumerate() {
        let i = i as i64;
        let (dir, path, tool) = work(key);
        let sid = id(key);
        let assignment_id = format!("sim-{run_id}-assignment-{key}");
        events.push(item(2_100 + i * 95, "assignment.created", json!({
            "assignmentId": assignment_id, "sessionIds": [sid], "targetType": "folder", "targetId": dir,
            "targetLabel": dir, "workspaceId": workspace_id,
            "orders": format!("Advance {workspace_name} through {dir}"), "simulationRunId": run_id,
        }), &assignment_id));
        events.push(item(3_500 + i * 150, "session.state", json!({
            "sessionId": sid, "state": "moving", "detail": format!("Moving to {dir}"), "simulationRunId": run_id,
        }), &sid));
        events.push(item(5_500 + i * 170, "session.tool_use", json!({
            "sessionId": sid, "name": tool, "workspaceId": workspace_id, "dir": dir, "path": path,
            "summary": format!("Surveying {path}"), "simulationRunId": run_id,
        }), &sid));
        events.push(progress(6_300 + i * 145, &sid, 1));
    }

    for (at, key, url, domain) in [
        (
            5_200,
            "pathfinder",
            "https://docs.github.com/actions",
            "docs.github.com",
        ),
        (
            5_750,
            "courier",
            "https://developer.mozilla.org/",
            "developer.mozilla.org",
        ),
        (
            6_250,
            "challenger",
            "https://docs.docker.com/",
            "docs.docker.com",
        ),
    ] {
        let sid = id(key);
        events.push(item(
            at,
            "browser.navigated",
            json!({
                "sessionId": sid, "url": url, "domain": domain, "simulationRunId": run_id,
            }),
            &sid,
        ));
    }

    let links = [
        ("pathfinder", "praetor"),
        ("courier", "praetor"),
        ("praetor", "forge-a"),
        ("praetor", "forge-b"),
        ("praetor", "forge-c"),
        ("forge-a", "sentinel-a"),
        ("forge-c", "sentinel-b"),
        ("challenger", "sentinel-a"),
        ("challenger", "sentinel-b"),
        ("relay-a", "forge-a"),
        ("relay-b", "forge-a"),
        ("archivist", "praetor"),
    ];
    for (i, (from, to)) in links.iter().enumerate() {
        let from_id = id(from);
        events.push(item(
            6_500 + i as i64 * 230,
            "agent.communication",
            json!({
                "fromSessionId": from_id, "toSessionId": id(to), "channel": "operations",
                "summary": "Shared status and evidence", "simulationRunId": run_id,
            }),
            &from_id,
        ));
    }

    for (i, (key, _, _, _)) in UNITS.iter().enumerate() {
        events.push(progress(8_500 + i as i64 * 185, &id(key), 2));
    }

    // A visible incident keeps the scenario from being a perfect canned march.
    let comm = |at: i64, from: &str, to: &str, channel: &str, summary: &str| {
        let from_id = id(from);
        item(
            at,
            "agent.communication",
            json!({
                "fromSessionId": from_id, "toSessionId": id(to), "channel": channel,
                "summary": summary, "simulationRunId": run_id,
            }),
            &from_id,
        )
    };
    let state = |at: i64, key: &str, state: &str, detail: &str| {
        let sid = id(key);
        item(
            at,
            "session.state",
            json!({
                "sessionId": sid, "state": state, "detail": detail, "simulationRunId": run_id,
            }),
            &sid,
        )
    };
    events.push(state(
        10_900,
        "relay-b",
        "blocked",
        "Deployment manifest drift detected",
    ));
    events.push(state(
        11_250,
        "forge-a",
        "blocked",
        "Waiting on deployment topology",
    ));
    events.push(comm(
        11_700,
        "relay-b",
        "praetor",
        "incident",
        "Escalated deployment drift",
    ));
    events.push(comm(
        12_150,
        "praetor",
        "sentinel-b",
        "incident",
        "Requested independent manifest comparison",
    ));
    events.push(item(12_900, "session.tool_use", json!({
        "sessionId": id("sentinel-b"), "name": "Diff", "workspaceId": workspace_id, "dir": "deploy",
        "path": "deploy/service.yaml", "summary": "Comparing desired and observed topology", "simulationRunId": run_id,
    }), &id("sentinel-b")));
    events.push(item(13_800, "session.message", json!({
        "sessionId": id("sentinel-b"), "role": "assistant",
        "text": "Drift isolated to one stale service selector. Safe correction prepared.", "simulationRunId": run_id,
    }), &id("sentinel-b")));
    events.push(state(
        14_350,
        "relay-b",
        "working",
        "Applying verified correction",
    ));
    events.push(state(14_650, "forge-a", "working", "Dependency restored"));

    let changed = [
        ("forge-a", "src/core", "src/core/service.ts"),
        ("forge-b", "src/ui", "src/ui/App.tsx"),
        ("forge-c", "src/agents", "src/agents/orchestrator.ts"),
        ("relay-b", "deploy", "deploy/service.yaml"),
        ("archivist", "docs", "docs/operations.md"),
    ];
    for (i, (key, dir, path)) in changed.iter().enumerate() {
        let sid = id(key);
        events.push(item(
            15_000 + i as i64 * 430,
            "fs.changed",
            json!({
                "sessionId": sid, "workspaceId": workspace_id, "dir": dir, "path": path,
                "change": if i == 4 { "add" } else { "change" }, "simulationRunId": run_id,
            }),
            &sid,
        ));
    }

    for (i, (key, _, role, _)) in UNITS.iter().enumerate() {
        let done = if *role == "verifier" { 3 } else { 4 };
        events.push(progress(17_400 + i as i64 * 165, &id(key), done));
    }
    events.push(comm(
        19_650,
        "challenger",
        "sentinel-a",
        "red-blue",
        "Challenge case delivered for independent replay",
    ));
    events.push(comm(
        20_300,
        "sentinel-a",
        "praetor",
        "verification",
        "Red-team challenge reproduced and closed",
    ));

    for (i, (key, _, _, _)) in UNITS.iter().enumerate() {
        let i = i as i64;
        let sid = id(key);
        events.push(progress(21_000 + i * 120, &sid, 5));
        events.push(item(22_800 + i * 120, "session.ended", json!({
            "sessionId": sid, "reason": "completed", "result": "Synthetic objective completed", "simulationRunId": run_id,
        }), &sid));
    }
    let session_ids: Vec<String> = UNITS.iter().map(|(key, _, _, _)| id(key)).collect();
    events.push(item(
        24_400,
        "simulation.completed",
        json!({
            "simulationRunId": run_id, "scenario": "operations-cycle", "sessionIds": session_ids,
        }),
        run_id,
    ));

    events.sort_by_key(|e| e.at);
    for e in &mut events {
        e.at *= TIMELINE_SCALE;
    }
    events
}

/// Every event a rehearsal emits must be marked, cost nothing and execute
/// nothing.
pub fn assert_safe_timeline(timeline: &[TimelineItem]) -> Result<(), String> {
    for event in timeline {
        if event.data.get("simulated") != Some(&Value::Bool(true)) {
            return Err(format!("unmarked simulation event: {}", event.kind));
        }
        if event.kind == "session.usage"
            && event
                .data
                .get("costUsd")
                .and_then(Value::as_f64)
                .is_some_and(|c| c > 0.0)
        {
            return Err("simulation may not emit cost".into());
        }
        if event.kind == "terminal.run" || event.kind == "permission.requested" {
            return Err(format!(
                "simulation may not emit executable event {}",
                event.kind
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Active {
    run_id: String,
    scenario: String,
    speed: f64,
    started_at: i64,
    workspace_id: String,
    duration_ms: i64,
    session_ids: Vec<String>,
}

impl Active {
    fn status(&self) -> Value {
        json!({
            "runId": self.run_id, "scenario": self.scenario, "speed": self.speed,
            "startedAt": self.started_at, "workspaceId": self.workspace_id,
            "durationMs": self.duration_ms, "sessionIds": self.session_ids,
        })
    }
}

pub struct FieldSimulator {
    emit: Emit,
    /// (id, name) of every mounted workspace.
    workspaces: Vec<(String, String)>,
    active: Mutex<Option<Active>>,
    /// Bumped on every run and stop; a scheduled event whose generation is
    /// stale never fires (the Node code cleared timers instead).
    generation: AtomicU64,
    now: Arc<dyn Fn() -> i64 + Send + Sync>,
    new_run_id: Arc<dyn Fn() -> String + Send + Sync>,
}

fn simulated(subject: &str) -> AppendOptions {
    AppendOptions {
        actor: None,
        subject: Some(subject.to_string()),
        source: None,
        simulated: true,
    }
}

impl FieldSimulator {
    pub fn new(
        emit: Emit,
        workspaces: Vec<(String, String)>,
        now: Arc<dyn Fn() -> i64 + Send + Sync>,
        new_run_id: Arc<dyn Fn() -> String + Send + Sync>,
    ) -> Arc<FieldSimulator> {
        Arc::new(FieldSimulator {
            emit,
            workspaces,
            active: Mutex::new(None),
            generation: AtomicU64::new(0),
            now,
            new_run_id,
        })
    }

    pub fn scenarios(&self) -> Value {
        json!([{
            "id": "operations-cycle", "name": "Long-horizon operations cycle", "durationMs": OPERATIONS_DURATION_MS,
            "description": "Twelve agents muster, fan out, coordinate, recover from drift, verify, and report.",
        }])
    }

    pub fn status(&self) -> Value {
        self.active
            .lock()
            .ok()
            .and_then(|a| a.as_ref().map(Active::status))
            .unwrap_or(Value::Null)
    }

    /// Starts a rehearsal, replacing any running one. Needs a Tokio runtime
    /// on the calling thread: the timeline is played by a spawned task.
    pub fn run(
        self: &Arc<Self>,
        scenario: &str,
        speed: Option<f64>,
        workspace_id: Option<&str>,
    ) -> Result<Value, String> {
        if scenario != "operations-cycle" {
            return Err(format!("unknown simulation scenario: {scenario}"));
        }
        self.stop("replaced");
        let safe_speed = speed
            .filter(|s| s.is_finite() && *s != 0.0)
            .unwrap_or(1.0)
            .clamp(0.25, 4.0);
        let (ws_id, ws_name) = self
            .workspaces
            .iter()
            .find(|(id, _)| Some(id.as_str()) == workspace_id)
            .or_else(|| self.workspaces.first())
            .cloned()
            .unwrap_or_else(|| {
                (
                    workspace_id
                        .filter(|w| !w.is_empty())
                        .unwrap_or("cameo")
                        .to_string(),
                    "the project".to_string(),
                )
            });
        let run_id: String = (self.new_run_id)().chars().take(8).collect();
        let timeline = operations_timeline(&run_id, &ws_id, &ws_name);
        let mut session_ids: Vec<String> = Vec::new();
        for e in &timeline {
            if let Some(sid) = e.data.get("sessionId").and_then(Value::as_str) {
                if !session_ids.iter().any(|s| s == sid) {
                    session_ids.push(sid.to_string());
                }
            }
        }
        let last_at = timeline.last().map(|e| e.at).unwrap_or(0);
        let active = Active {
            run_id: run_id.clone(),
            scenario: scenario.to_string(),
            speed: safe_speed,
            started_at: (self.now)(),
            workspace_id: ws_id,
            duration_ms: (last_at as f64 / safe_speed).ceil() as i64,
            session_ids: session_ids.clone(),
        };
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        if let Ok(mut slot) = self.active.lock() {
            *slot = Some(active.clone());
        }
        (self.emit)(
            "simulation.started",
            json!({
                "simulationRunId": run_id, "scenario": scenario, "speed": safe_speed,
                "sessionIds": session_ids, "simulated": true,
            }),
            simulated(&run_id),
        );

        let me = Arc::clone(self);
        tokio::spawn(async move {
            let started = tokio::time::Instant::now();
            for event in timeline {
                let due =
                    started + Duration::from_millis((event.at as f64 / safe_speed).ceil() as u64);
                tokio::time::sleep_until(due).await;
                if me.generation.load(Ordering::SeqCst) != generation {
                    return;
                }
                (me.emit)(&event.kind, event.data, simulated(&event.subject));
                if event.kind == "simulation.completed" {
                    if let Ok(mut slot) = me.active.lock() {
                        if slot.as_ref().is_some_and(|a| a.run_id == run_id) {
                            *slot = None;
                        }
                    }
                }
            }
        });
        Ok(active.status())
    }

    pub fn stop(&self, reason: &str) -> Value {
        self.generation.fetch_add(1, Ordering::SeqCst);
        let previous = self.active.lock().ok().and_then(|mut a| a.take());
        let Some(previous) = previous else {
            return json!({ "stopped": false });
        };
        for session_id in &previous.session_ids {
            (self.emit)(
                "session.ended",
                json!({
                    "sessionId": session_id, "reason": "cancelled", "simulated": true,
                    "simulationRunId": previous.run_id,
                }),
                simulated(session_id),
            );
        }
        (self.emit)(
            "simulation.stopped",
            json!({
                "simulationRunId": previous.run_id, "scenario": previous.scenario,
                "reason": reason, "simulated": true,
            }),
            simulated(&previous.run_id),
        );
        json!({ "stopped": true, "runId": previous.run_id })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_operations_timeline_is_safe_ordered_and_scaled() {
        let timeline = operations_timeline("abc", "cameo", "Cameo");
        assert_safe_timeline(&timeline).unwrap();
        assert!(timeline.windows(2).all(|w| w[0].at <= w[1].at));
        assert_eq!(timeline.last().unwrap().kind, "simulation.completed");
        assert_eq!(timeline.last().unwrap().at, OPERATIONS_DURATION_MS);
        let spawned = timeline
            .iter()
            .filter(|e| e.kind == "session.spawned")
            .count();
        assert_eq!(spawned, 12);
        assert!(timeline
            .iter()
            .all(|e| e.data["simulated"] == true && !e.subject.is_empty()));
        let mut unsafe_timeline = timeline.clone();
        unsafe_timeline.push(item(0, "terminal.run", json!({}), "x"));
        assert!(assert_safe_timeline(&unsafe_timeline)
            .unwrap_err()
            .contains("executable"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_and_stop_emit_marked_events_and_a_stale_run_never_fires() {
        let events: Arc<Mutex<Vec<(String, bool)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        let emit: Emit = Arc::new(move |kind, _data, options| {
            sink.lock()
                .unwrap()
                .push((kind.to_string(), options.simulated));
            None
        });
        let sim = FieldSimulator::new(
            emit,
            vec![("cameo".into(), "Cameo".into())],
            Arc::new(|| 1_700_000_000_000),
            Arc::new(|| "run-1234-xyz".into()),
        );
        assert_eq!(sim.status(), Value::Null);
        let status = sim.run("operations-cycle", Some(4.0), None).unwrap();
        assert_eq!(status["runId"], "run-1234");
        assert_eq!(status["speed"], 4.0);
        assert_eq!(status["sessionIds"].as_array().unwrap().len(), 12);
        assert!(sim.run("nope", None, None).is_err());
        tokio::time::sleep(Duration::from_millis(120)).await;
        let stopped = sim.stop("operator");
        assert_eq!(stopped["stopped"], true);
        assert_eq!(sim.status(), Value::Null);
        tokio::time::sleep(Duration::from_millis(400)).await;
        let seen = events.lock().unwrap().clone();
        assert_eq!(seen[0].0, "simulation.started");
        assert!(seen.iter().all(|(_, simulated)| *simulated), "{seen:?}");
        let ended = seen.iter().filter(|(k, _)| k == "session.ended").count();
        assert_eq!(ended, 12, "stop ends every synthetic session once");
        assert_eq!(
            seen.last().unwrap().0,
            "simulation.stopped",
            "nothing fires after stop: {seen:?}"
        );
        assert_eq!(sim.stop("again")["stopped"], false);
    }
}
