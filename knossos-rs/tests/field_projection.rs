//! Parity oracle for the projection port: `projection-focus.test.mjs`,
//! `world.test.mjs`, `budget-ledger.test.mjs`, `graph.test.mjs`,
//! `transitions.test.mjs`, `campaign.test.mjs` and the projection half of
//! `stress.test.mjs`, case for case.

use knossos::field::budget_ledger::BudgetLedger;
use knossos::field::campaign_projection::CampaignProjection;
use knossos::field::graph_projection::GraphProjection;
use knossos::field::model::{
    is_blocking_finding, legal_actions, transition_options, transitions, validate_campaign_input,
    validate_finding, validate_transition, Context, CAMPAIGN_PHASES,
};
use knossos::field::projection::{FieldConfig, Projection};
use knossos::field::{Event, Source};
use serde_json::{json, Value};
use std::time::Instant;

fn ev(seq: u64, ts: i64, kind: &str, data: Value) -> Event {
    Event {
        seq,
        ts,
        kind: kind.into(),
        actor: None,
        subject: None,
        source: Source::Observed,
        data,
    }
}

fn synthetic(seq: u64, ts: i64, kind: &str, data: Value) -> Event {
    Event {
        source: Source::Synthetic,
        ..ev(seq, ts, kind, data)
    }
}

fn cfg(value: Value) -> FieldConfig {
    serde_json::from_value(value).unwrap()
}

fn alpha() -> FieldConfig {
    cfg(json!({
        "workspaces": [{ "id": "alpha", "name": "Alpha", "mounted": true }],
        "endpoints": [],
        "websites": [],
    }))
}

fn session<'a>(snap: &'a Value, id: &str) -> &'a Value {
    snap["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["id"] == id)
        .unwrap()
}

#[test]
fn tool_use_path_sets_focus_path_additively_and_absence_leaves_it_undefined() {
    let mut projection = Projection::new(alpha());
    projection.apply(&ev(
        1,
        1000,
        "session.spawned",
        json!({ "sessionId": "unit-1", "workspaceId": "alpha", "role": "builder" }),
    ));
    let snap = projection.snapshot(2000);
    let unit = session(&snap, "unit-1");
    assert!(
        unit.get("focusPath").is_none(),
        "a fresh session has no focusPath"
    );
    assert!(
        unit.get("focusDir").is_none(),
        "a fresh session has no focusDir"
    );

    projection.apply(&ev(
        2,
        1100,
        "session.tool_use",
        json!({ "sessionId": "unit-1", "name": "Edit", "workspaceId": "alpha", "dir": "src/core", "path": "src/core/service.ts", "summary": "Editing service" }),
    ));
    let snap = projection.snapshot(2000);
    let unit = session(&snap, "unit-1");
    assert_eq!(unit["focusPath"], "src/core/service.ts");
    assert_eq!(unit["focusDir"], "src/core");

    projection.apply(&ev(
        3,
        1200,
        "session.tool_use",
        json!({ "sessionId": "unit-1", "name": "Bash", "workspaceId": "alpha", "dir": "src/core", "summary": "Running tests" }),
    ));
    let snap = projection.snapshot(2000);
    assert_eq!(
        session(&snap, "unit-1")["focusPath"],
        "src/core/service.ts",
        "a pathless tool_use does not erase focusPath"
    );

    projection.apply(&ev(
        4,
        1300,
        "session.spawned",
        json!({ "sessionId": "unit-2", "workspaceId": "alpha", "role": "verifier" }),
    ));
    projection.apply(&ev(
        5,
        1400,
        "session.tool_use",
        json!({ "sessionId": "unit-2", "name": "Bash", "workspaceId": "alpha", "dir": "tests", "summary": "Running suite" }),
    ));
    let snap = projection.snapshot(2000);
    let unit = session(&snap, "unit-2");
    assert!(unit.get("focusPath").is_none());
    assert_eq!(unit["focusDir"], "tests");
}

#[test]
fn durable_territory_synthetic_partition_and_evidence_derived_maturity() {
    let mut projection = Projection::new(alpha());
    projection.apply(&ev(
        1,
        1000,
        "world.capital_selected",
        json!({ "workspaceId": "alpha" }),
    ));
    projection.apply(&ev(
        2,
        1001,
        "world.territory_assigned",
        json!({ "clusterKey": "workspace:alpha", "territoryId": "italia", "label": "Alpha", "kind": "project", "workspaceId": "alpha" }),
    ));
    let mut snap = projection.snapshot(2000);
    assert_eq!(snap["world"]["capitalWorkspaceId"], "alpha");
    assert_eq!(snap["world"]["capitalSelectedAt"], 1000);
    assert_eq!(
        snap["world"]["assignments"]["workspace:alpha"]["territoryId"],
        "italia"
    );
    assert_eq!(snap["world"]["revision"], 2);

    snap["world"]["assignments"]["workspace:alpha"]["territoryId"] = json!("mutated");
    assert_eq!(
        projection.world()["assignments"]["workspace:alpha"]["territoryId"],
        "italia"
    );

    projection.apply(&ev(
        3,
        1002,
        "world.territory_released",
        json!({ "clusterKey": "workspace:alpha" }),
    ));
    assert!(projection.snapshot(2000)["world"]["assignments"]
        .get("workspace:alpha")
        .is_none());

    projection.apply(&ev(
        4,
        1003,
        "session.spawned",
        json!({ "sessionId": "real-session", "workspaceId": "alpha", "role": "builder" }),
    ));
    projection.apply(&ev(
        5,
        1004,
        "fs.changed",
        json!({ "sessionId": "real-session", "workspaceId": "alpha", "dir": "src", "path": "src/real.js", "change": "change" }),
    ));
    projection.apply(&synthetic(
        6,
        1005,
        "simulation.started",
        json!({ "simulationRunId": "demo-1", "simulated": true }),
    ));
    projection.apply(&synthetic(
        7,
        1006,
        "session.spawned",
        json!({ "sessionId": "sim-session", "workspaceId": "alpha", "role": "verifier", "simulated": true, "simulationRunId": "demo-1" }),
    ));
    projection.apply(&synthetic(
        8,
        1007,
        "fs.changed",
        json!({ "sessionId": "sim-session", "workspaceId": "alpha", "dir": "demo", "path": "demo/fake.js", "change": "add", "simulated": true }),
    ));
    let ids = |list: &Value, key: &str| -> Vec<String> {
        list.as_array()
            .unwrap()
            .iter()
            .map(|x| x[key].as_str().unwrap().to_string())
            .collect()
    };
    let partitioned = projection.snapshot(2000);
    assert_eq!(ids(&partitioned["sessions"], "id"), ["real-session"]);
    assert_eq!(ids(&partitioned["files"], "path"), ["src/real.js"]);
    assert_eq!(partitioned["workspaces"][0]["changeCount"], 1);
    assert_eq!(
        ids(&partitioned["rehearsal"]["sessions"], "id"),
        ["sim-session"]
    );
    assert_eq!(
        ids(&partitioned["rehearsal"]["files"], "path"),
        ["demo/fake.js"]
    );
    assert_eq!(partitioned["rehearsal"]["workspaces"][0]["changeCount"], 1);

    projection.apply(&synthetic(
        9,
        1008,
        "simulation.started",
        json!({ "simulationRunId": "demo-2", "simulated": true }),
    ));
    let partitioned = projection.snapshot(2000);
    assert_eq!(
        partitioned["rehearsal"]["sessions"]
            .as_array()
            .unwrap()
            .len(),
        0,
        "a new rehearsal resets prior synthetic state"
    );
    assert_eq!(
        partitioned["sessions"].as_array().unwrap().len(),
        1,
        "resetting a rehearsal cannot alter production state"
    );

    projection.apply(&ev(
        10,
        1010,
        "session.ended",
        json!({ "sessionId": "real-session", "reason": "completed" }),
    ));
    assert_eq!(
        projection.snapshot(2000)["workspaces"][0]["maturity"]["score"],
        0
    );
    projection.apply(&ev(
        11,
        1011,
        "campaign.created",
        json!({ "campaignId": "release", "name": "Release", "intent": "Ship", "scope": "sandbox", "target": { "type": "workspace", "id": "alpha", "workspaceId": "alpha" }, "doctrine": {}, "concurrency": 1, "budgetUsd": 1 }),
    ));
    projection.apply(&ev(
        12,
        1012,
        "objective.created",
        json!({ "campaignId": "release", "objectiveId": "criterion", "statement": "Pass release check", "definitionOfDone": ["test passes"], "required": true, "target": { "type": "workspace", "id": "alpha", "workspaceId": "alpha" } }),
    ));
    projection.apply(&ev(
        13,
        1013,
        "objective.satisfied",
        json!({ "campaignId": "release", "objectiveId": "criterion", "evidence": ["test://pass"], "criteriaEvidence": [{ "criterion": "test passes", "evidence": "test://pass" }] }),
    ));
    let maturity = projection.snapshot(2000)["workspaces"][0]["maturity"].clone();
    assert_eq!(maturity["complete"], 100);
    assert_eq!(
        maturity["verified"], 0,
        "completion evidence is not an independent verdict"
    );
    assert_eq!(maturity["persisted"], 0);
    projection.apply(&ev(
        14,
        1014,
        "referee.verdict",
        json!({ "campaignId": "release", "verdictId": "verdict-1", "sessionId": "independent-referee", "verdict": "verified", "evidence": ["replay://pass"], "rationale": "independent replay passed" }),
    ));
    projection.apply(&ev(
        15,
        1015,
        "campaign.phase_changed",
        json!({ "campaignId": "release", "from": "referee_review", "to": "verified" }),
    ));
    let maturity = projection.snapshot(2000)["workspaces"][0]["maturity"].clone();
    assert_eq!(maturity["verified"], 100);
    assert_eq!(maturity["persisted"], 0);
    projection.apply(&ev(
        16,
        1016,
        "campaign.checkpoint_created",
        json!({ "campaignId": "release", "checkpointId": "checkpoint-1", "revision": "sha256:abcdef12", "eventSeq": 15 }),
    ));
    let maturity = projection.snapshot(2000)["workspaces"][0]["maturity"].clone();
    assert_eq!(maturity["score"], 100);
    assert_eq!(maturity["persisted"], 100);
    let evidence = maturity["evidence"].as_array().unwrap();
    assert!(evidence
        .iter()
        .any(|i| i["type"] == "verification" && i["verdictId"] == "verdict-1"));
    assert!(evidence
        .iter()
        .any(|i| i["type"] == "persistence" && i["checkpointId"] == "checkpoint-1"));
}

#[test]
fn concurrent_reservation_unknown_and_zero_cost_monotonic_usage_and_replay() {
    let mut ledger = BudgetLedger::new();
    let mut events = Vec::new();
    let mut apply = |ledger: &mut BudgetLedger, kind: &str, data: Value| {
        let e = ev(0, 0, kind, data);
        ledger.apply(&e);
        events.push(e);
    };
    apply(
        &mut ledger,
        "budget.reserved",
        json!({ "sessionId": "a", "campaignId": "campaign", "limitUsd": 4 }),
    );
    apply(
        &mut ledger,
        "budget.reserved",
        json!({ "sessionId": "b", "campaignId": "campaign", "limitUsd": 4 }),
    );
    assert_eq!(ledger.campaign("campaign", 0.0, 10.0).remaining_usd, 2.0);
    assert_eq!(
        ledger.campaign("campaign", 0.0, 10.0).unknown_cost_sessions,
        2
    );
    apply(
        &mut ledger,
        "session.usage",
        json!({ "sessionId": "a", "costUsd": 1 }),
    );
    apply(
        &mut ledger,
        "session.usage",
        json!({ "sessionId": "a", "costUsd": 0.5 }),
    );
    assert_eq!(ledger.campaign("campaign", 1.0, 10.0).remaining_usd, 2.0);
    apply(
        &mut ledger,
        "session.turn_complete",
        json!({ "sessionId": "a" }),
    );
    apply(&mut ledger, "session.ended", json!({ "sessionId": "a" }));
    assert_eq!(ledger.campaign("campaign", 1.0, 10.0).remaining_usd, 5.0);
    apply(&mut ledger, "session.ended", json!({ "sessionId": "b" }));
    assert_eq!(
        ledger.campaign("campaign", 1.0, 10.0).remaining_usd,
        5.0,
        "missing telemetry is not refunded"
    );
    apply(
        &mut ledger,
        "budget.reactivated",
        json!({ "sessionId": "a" }),
    );
    assert_eq!(ledger.campaign("campaign", 1.0, 10.0).remaining_usd, 2.0);
    let mut replay = BudgetLedger::new();
    for e in &events {
        replay.apply(e);
        replay.apply(e);
    }
    assert_eq!(
        replay.snapshot(),
        ledger.snapshot(),
        "replay and duplicate delivery retain the same reservations"
    );

    let mut zero = BudgetLedger::new();
    zero.apply(&ev(
        0,
        0,
        "budget.reserved",
        json!({ "sessionId": "zero", "limitUsd": 1 }),
    ));
    zero.apply(&ev(
        0,
        0,
        "session.usage",
        json!({ "sessionId": "zero", "costUsd": 0 }),
    ));
    assert_eq!(zero.snapshot()[0]["costStatus"], "reported_zero");
    // NaN cannot travel through JSON; a non-number cost is the same "not finite" case.
    zero.apply(&ev(
        0,
        0,
        "session.usage",
        json!({ "sessionId": "zero", "costUsd": "NaN" }),
    ));
    assert_eq!(zero.snapshot()[0]["spentUsd"], 0);

    // UI and campaign totals must agree with the replayed reservation ledger.
    let mut projection = Projection::new(cfg(
        json!({ "endpoints": [], "workspaces": [], "websites": [] }),
    ));
    for (i, value) in [
        json!(2),
        json!(1),
        json!(2),
        json!("NaN"),
        json!("Infinity"),
        json!(-1),
        json!(3),
    ]
    .into_iter()
    .enumerate()
    {
        let seq = i as u64 + 1;
        projection.apply(&ev(
            seq,
            seq as i64,
            "session.usage",
            json!({ "sessionId": "cumulative", "costUsd": value, "inputTokens": value, "outputTokens": value, "deltaInput": 99, "deltaOutput": 99 }),
        ));
    }
    assert_eq!(projection.totals.cost_usd, 3.0);
    assert_eq!(projection.totals.input_tokens, 3.0);
    assert_eq!(projection.totals.output_tokens, 3.0);
    assert_eq!(
        projection.session_record("cumulative").unwrap()["costUsd"],
        3
    );
}

#[test]
fn typed_topology_lifecycle_hysteresis_and_contest_edges() {
    let t0: i64 = 1_700_000_000_000;
    let mut seq = 0u64;
    let mut event = |kind: &str, data: Value, ts: Option<i64>| {
        seq += 1;
        ev(seq, ts.unwrap_or(t0 + seq as i64 * 1000), kind, data)
    };
    let mut g = GraphProjection::new();
    g.apply(&event(
        "campaign.created",
        json!({ "campaignId": "c1", "name": "Gateway", "target": { "workspaceId": "cameo" } }),
        None,
    ));
    g.apply(&event(
        "session.spawned",
        json!({ "sessionId": "blue-1", "agentId": "rhea", "name": "Rhea", "role": "builder", "workspaceId": "cameo", "endpointId": "local", "model": "ornith", "campaignId": "c1", "team": "blue" }),
        None,
    ));
    g.apply(&event("team.member_assigned", json!({ "campaignId": "c1", "team": "blue", "sessionId": "blue-1", "agentId": "rhea", "role": "builder" }), None));
    for i in 0..3 {
        g.apply(&event(
            "session.tool_use",
            json!({ "sessionId": "blue-1", "name": "Read", "workspaceId": "cameo", "dir": "cameod/src", "path": format!("cameod/src/f{i}.rs") }),
            Some(t0 + i * 31_000),
        ));
    }
    g.apply(&event(
        "finding.reported",
        json!({ "campaignId": "c1", "objectiveId": "o1", "findingId": "f1", "authorSessionId": "red-1", "claim": "rollback race", "severity": "high" }),
        None,
    ));
    g.apply(&event("mitigation.proposed", json!({ "campaignId": "c1", "mitigationId": "m1", "findingIds": ["f1"], "claim": "serialize rollback" }), None));
    g.apply(&event(
        "retest.completed",
        json!({ "campaignId": "c1", "findingId": "f1", "sessionId": "red-1", "result": "fixed" }),
        None,
    ));

    let find_node = |snap: &Value, key: &str| -> Value {
        snap["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["key"] == key)
            .cloned()
            .unwrap()
    };
    let has_edge =
        |snap: &Value, f: &dyn Fn(&Value) -> bool| snap["edges"].as_array().unwrap().iter().any(f);

    let snap = g.snapshot(t0 + 70_000, None, None);
    let folder = find_node(&snap, "folder:cameo:cameod/src");
    assert_eq!(folder["state"], "active_worksite");
    assert!(has_edge(&snap, &|e| e["type"] == "working_on"
        && e["to"] == folder["key"]));
    assert!(has_edge(&snap, &|e| e["type"] == "mitigated_by"));
    assert!(has_edge(&snap, &|e| e["type"] == "retested_by"));

    let snap = g.snapshot(t0 + 33 * 60_000, None, None);
    assert_eq!(
        find_node(&snap, "folder:cameo:cameod/src")["state"],
        "dormant"
    );
    assert!(!snap["visibleNodeKeys"]
        .as_array()
        .unwrap()
        .iter()
        .any(|k| k == "folder:cameo:cameod/src"));

    g.apply(&event(
        "session.tool_use",
        json!({ "sessionId": "blue-1", "name": "Read", "workspaceId": "cameo", "dir": "cameod/src", "path": "cameod/src/app.rs" }),
        Some(t0 + 34 * 60_000),
    ));
    let snap = g.snapshot(t0 + 34 * 60_000, None, None);
    let state = find_node(&snap, "folder:cameo:cameod/src")["state"].clone();
    assert!(
        state == "active_worksite" || state == "established",
        "{state}"
    );
}

fn base(phase: &str) -> Value {
    json!({
        "id": "transition-campaign", "phase": phase, "paused": false,
        "doctrine": { "blockingSeverity": "high" },
        "teams": {
            "blue": { "members": [{ "sessionId": "blue-1", "status": "active" }] }, "red": { "members": [] },
            "referee": { "members": [] }, "purple": { "members": [] },
        },
    })
}

fn context_for(from: &str, to: &str) -> Context {
    let mut context = Context {
        objectives: vec![json!({ "id": "o1", "required": true, "status": "satisfied" })],
        ..Default::default()
    };
    let finding = |status: &str| vec![json!({ "id": "f1", "severity": "high", "status": status })];
    match (from, to) {
        ("red_challenging", "contested") => context.findings = finding("open"),
        ("contested", "blue_mitigating") => context.findings = finding("acknowledged"),
        ("blue_mitigating", "red_retesting") => context.findings = finding("ready_for_retest"),
        ("red_retesting", "referee_review") => context.findings = finding("confirmed"),
        ("referee_review", "verified") => context.latest_verdict = Some("verified".into()),
        ("verified", "promoted") => {
            context.checkpoint_id = Some("cp1".into());
            context.checkpoint_revision = Some(json!("abcdef1"));
        }
        _ => {}
    }
    context
}

#[test]
fn every_legal_transition_passes_and_an_illegal_one_per_phase_is_refused() {
    let mut positive = 0;
    let mut negative = 0;
    for from in CAMPAIGN_PHASES {
        let legal: Vec<&str> = transitions(from)
            .iter()
            .copied()
            .filter(|t| *t != "paused")
            .collect();
        for to in &legal {
            assert!(
                validate_transition(&base(from), to, &context_for(from, to)).is_ok(),
                "{from} -> {to}"
            );
            positive += 1;
        }
        if let Some(illegal) = CAMPAIGN_PHASES
            .iter()
            .find(|to| **to != from && !legal.contains(to))
        {
            let err =
                validate_transition(&base(from), illegal, &context_for(from, illegal)).unwrap_err();
            assert!(
                err.message.contains("cannot advance"),
                "{from} -> {illegal}: {}",
                err.message
            );
            negative += 1;
        }
    }
    assert!(positive > 0 && negative > 0);

    let options = transition_options(
        &base("blue_building"),
        &Context {
            objectives: vec![json!({ "id": "o1", "required": true, "status": "active" })],
            ..Default::default()
        },
    );
    let red = options
        .iter()
        .find(|o| o["to"] == "red_challenging")
        .unwrap();
    assert_eq!(red["legal"], false);
    assert!(red["reason"].as_str().unwrap().contains("not ready"));
}

#[test]
fn campaign_state_machine_evidence_gates_and_projection_assertions() {
    let seq = std::cell::Cell::new(0u64);
    let event = |kind: &str, data: Value| {
        seq.set(seq.get() + 1);
        ev(
            seq.get(),
            1_700_000_000_000 + seq.get() as i64 * 1000,
            kind,
            data,
        )
    };
    let valid = validate_campaign_input(&json!({
        "name": "Release gateway",
        "intent": "Prove restart and rollback safety.",
        "scope": "sandbox",
        "objectives": [{ "statement": "Rollback is race-safe", "definitionOfDone": ["stress check passes"] }],
    }))
    .unwrap();
    assert_eq!(valid["objectives"].as_array().unwrap().len(), 1);
    assert!(
        validate_campaign_input(&json!({ "name": "x", "intent": "y", "objectives": [] })).is_err()
    );
    let err =
        validate_finding(&json!({ "claim": "race", "severity": "high", "evidence": ["trace"] }))
            .unwrap_err();
    assert!(err.message.contains("reproduction"));

    let mut p = CampaignProjection::new();
    let mut created = valid.as_object().unwrap().clone();
    created.insert("campaignId".into(), json!("c1"));
    let events = vec![
        event("campaign.created", Value::Object(created)),
        event(
            "objective.created",
            json!({ "campaignId": "c1", "objectiveId": "o1", "statement": "Rollback is race-safe", "definitionOfDone": ["stress check passes"], "priority": 1, "risk": "high", "target": null }),
        ),
        event(
            "team.member_assigned",
            json!({ "campaignId": "c1", "team": "blue", "sessionId": "blue-1", "agentId": "rhea", "role": "builder" }),
        ),
    ];
    for e in &events {
        p.apply(e);
    }
    let campaign = |p: &CampaignProjection| Value::Object(p.campaigns.get("c1").unwrap().clone());
    assert!(validate_transition(&campaign(&p), "mobilizing", &p.context("c1").unwrap()).is_ok());
    assert!(legal_actions(&campaign(&p), &p.context("c1").unwrap())
        .contains(&"advance:mobilizing".to_string()));

    p.apply(&event(
        "campaign.phase_changed",
        json!({ "campaignId": "c1", "from": "draft", "to": "mobilizing" }),
    ));
    p.apply(&event(
        "campaign.phase_changed",
        json!({ "campaignId": "c1", "from": "mobilizing", "to": "blue_building" }),
    ));
    let err = validate_transition(&campaign(&p), "red_challenging", &p.context("c1").unwrap())
        .unwrap_err();
    assert!(err.message.contains("not ready"));

    p.apply(&event(
        "objective.satisfied",
        json!({ "campaignId": "c1", "objectiveId": "o1", "evidence": ["check: pass"] }),
    ));
    assert!(
        validate_transition(&campaign(&p), "red_challenging", &p.context("c1").unwrap()).is_ok()
    );
    p.apply(&event(
        "campaign.phase_changed",
        json!({ "campaignId": "c1", "from": "blue_building", "to": "red_challenging" }),
    ));

    let finding = validate_finding(&json!({
        "claim": "Restart can interleave with rollback.", "severity": "high", "category": "rollback",
        "scope": "release gateway", "evidence": ["trace://race"], "reproduction": ["start restart", "issue rollback"],
    }))
    .unwrap();
    let mut reported = finding.as_object().unwrap().clone();
    for (k, v) in json!({ "campaignId": "c1", "objectiveId": "o1", "findingId": "f1", "authorSessionId": "red-1" }).as_object().unwrap() {
        reported.insert(k.clone(), v.clone());
    }
    p.apply(&event("finding.reported", Value::Object(reported)));
    let ctx = p.context("c1").unwrap();
    assert!(is_blocking_finding(
        &ctx.findings[0],
        &campaign(&p)["doctrine"]
    ));
    assert!(validate_transition(&campaign(&p), "contested", &ctx).is_ok());
    p.apply(&event(
        "campaign.phase_changed",
        json!({ "campaignId": "c1", "from": "red_challenging", "to": "contested" }),
    ));
    let err = validate_transition(&campaign(&p), "blue_mitigating", &p.context("c1").unwrap())
        .unwrap_err();
    assert!(err.message.contains("not acknowledged"));

    p.apply(&event(
        "finding.acknowledged",
        json!({ "campaignId": "c1", "findingId": "f1" }),
    ));
    assert!(
        validate_transition(&campaign(&p), "blue_mitigating", &p.context("c1").unwrap()).is_ok()
    );
    p.apply(&event(
        "campaign.phase_changed",
        json!({ "campaignId": "c1", "from": "contested", "to": "blue_mitigating" }),
    ));
    p.apply(&event(
        "mitigation.proposed",
        json!({ "campaignId": "c1", "mitigationId": "m1", "findingIds": ["f1"], "ownerSessionId": "blue-1", "claim": "serialize transitions", "artifacts": ["gateway.rs"], "evidence": [] }),
    ));
    p.apply(&event(
        "mitigation.started",
        json!({ "campaignId": "c1", "mitigationId": "m1" }),
    ));
    p.apply(&event(
        "mitigation.ready",
        json!({ "campaignId": "c1", "mitigationId": "m1", "evidence": ["stress: pass"] }),
    ));
    assert!(validate_transition(&campaign(&p), "red_retesting", &p.context("c1").unwrap()).is_ok());
    p.apply(&event(
        "campaign.phase_changed",
        json!({ "campaignId": "c1", "from": "blue_mitigating", "to": "red_retesting" }),
    ));
    p.apply(&event(
        "retest.completed",
        json!({ "campaignId": "c1", "findingId": "f1", "sessionId": "red-1", "result": "fixed", "evidence": ["repro no longer fails"] }),
    ));
    assert!(
        validate_transition(&campaign(&p), "referee_review", &p.context("c1").unwrap()).is_ok()
    );
    p.apply(&event(
        "campaign.phase_changed",
        json!({ "campaignId": "c1", "from": "red_retesting", "to": "referee_review" }),
    ));
    let err =
        validate_transition(&campaign(&p), "verified", &p.context("c1").unwrap()).unwrap_err();
    assert!(err.message.contains("verdict"));
    p.apply(&event(
        "referee.verdict",
        json!({ "campaignId": "c1", "verdictId": "v1", "sessionId": "ref-1", "verdict": "verified", "evidence": ["independent check"], "rationale": "definition of done passed" }),
    ));
    assert!(validate_transition(&campaign(&p), "verified", &p.context("c1").unwrap()).is_ok());
    p.apply(&event(
        "campaign.phase_changed",
        json!({ "campaignId": "c1", "from": "referee_review", "to": "verified" }),
    ));
    let err =
        validate_transition(&campaign(&p), "promoted", &p.context("c1").unwrap()).unwrap_err();
    assert!(err.message.contains("checkpoint"));
    let at = seq.get();
    p.apply(&event(
        "campaign.checkpoint_created",
        json!({ "campaignId": "c1", "checkpointId": "cp1", "name": "verified gateway", "eventSeq": at, "revision": "abcdef1" }),
    ));
    assert!(validate_transition(&campaign(&p), "promoted", &p.context("c1").unwrap()).is_ok());
    p.apply(&event(
        "campaign.promoted",
        json!({ "campaignId": "c1", "checkpointId": "cp1", "capabilities": [{ "id": "cap:gateway", "name": "Race-safe release gateway" }] }),
    ));

    let view = p.campaign_view(p.campaigns.get("c1").unwrap());
    assert_eq!(view["phase"], "promoted");
    assert_eq!(view["findings"][0]["status"], "confirmed");
    assert_eq!(view["capabilities"][0]["status"], "promoted");

    // A prefix replay into a clean projection is coherent and shares nothing.
    let mut replay = CampaignProjection::new();
    for e in &events {
        replay.apply(e);
    }
    assert_eq!(replay.campaigns.get("c1").unwrap()["phase"], "draft");
    assert_eq!(p.campaigns.get("c1").unwrap()["phase"], "promoted");
}

#[test]
fn a_hundred_thousand_events_fold_deterministically_within_bounds() {
    let config = cfg(json!({
        "workspaces": [{ "id": "cameo", "name": "Cameo", "path": "/cameo", "mounted": true, "region": { "x": 0, "y": 0, "w": 900, "h": 500 } }],
        "endpoints": [
            { "id": "local-a", "name": "Local A", "kind": "openai-compatible", "model": "ornith", "cost_per_mtok": { "input": 0, "output": 0 } },
            { "id": "cloud-b", "name": "Cloud B", "kind": "anthropic", "model": "frontier", "cost_per_mtok": { "input": 3, "output": 15 } },
        ],
        "websites": [],
    }));
    let mut events: Vec<Event> = Vec::new();
    let t0: i64 = 1_700_000_000_000 - 120_000;
    let push = |events: &mut Vec<Event>, kind: &str, data: Value, subject: Option<&str>| {
        let seq = events.len() as u64 + 1;
        events.push(Event {
            seq,
            ts: t0 + seq as i64,
            kind: kind.into(),
            actor: data
                .get("sessionId")
                .and_then(Value::as_str)
                .map(str::to_string),
            subject: subject.map(str::to_string),
            source: Source::Observed,
            data,
        });
    };
    push(
        &mut events,
        "campaign.created",
        json!({
            "campaignId": "stress-campaign", "commandId": "stress-create", "name": "Thirty-agent operation",
            "intent": "Exercise field density and replay.", "scope": "sandbox", "concurrency": 30,
            "budgetUsd": 200, "doctrine": { "blockingSeverity": "high", "requireRed": true, "requireReferee": true, "redCategories": ["reliability"] },
            "target": { "type": "workspace", "id": "cameo", "workspaceId": "cameo" }, "objectives": [],
        }),
        Some("stress-campaign"),
    );
    for o in 0..6 {
        push(
            &mut events,
            "objective.created",
            json!({
                "campaignId": "stress-campaign", "objectiveId": format!("objective-{o}"),
                "statement": format!("Objective {o}"), "definitionOfDone": [format!("check {o}")], "priority": o + 1,
                "risk": "medium", "target": { "type": "workspace", "id": "cameo", "workspaceId": "cameo" },
            }),
            Some(&format!("objective-{o}")),
        );
    }
    for i in 0..30 {
        let team = if i < 18 {
            "blue"
        } else if i < 27 {
            "red"
        } else if i < 29 {
            "referee"
        } else {
            "purple"
        };
        let role = if team == "referee" {
            "verifier"
        } else if team == "red" {
            "challenger"
        } else {
            "builder"
        };
        let sid = format!("agent-{i}");
        push(
            &mut events,
            "session.spawned",
            json!({
                "sessionId": sid, "agentId": sid, "name": format!("Agent {i}"), "role": role,
                "model": "ornith", "endpointId": if i % 2 == 1 { "cloud-b" } else { "local-a" }, "thinking": "medium",
                "workspaceId": "cameo", "campaignId": "stress-campaign", "team": team, "objectiveId": format!("objective-{}", i % 6),
            }),
            Some(&sid),
        );
        push(
            &mut events,
            "team.member_assigned",
            json!({ "campaignId": "stress-campaign", "team": team, "sessionId": sid, "agentId": sid, "role": role, "objectiveId": format!("objective-{}", i % 6), "status": "active" }),
            Some("stress-campaign"),
        );
        push(
            &mut events,
            "objective.assigned",
            json!({ "campaignId": "stress-campaign", "objectiveId": format!("objective-{}", i % 6), "team": team, "sessionIds": [sid] }),
            Some(&format!("objective-{}", i % 6)),
        );
        push(
            &mut events,
            "session.state",
            json!({ "sessionId": sid, "campaignId": "stress-campaign", "state": "working" }),
            Some(&sid),
        );
    }
    push(
        &mut events,
        "campaign.phase_changed",
        json!({ "campaignId": "stress-campaign", "from": "draft", "to": "mobilizing" }),
        Some("stress-campaign"),
    );
    push(
        &mut events,
        "campaign.phase_changed",
        json!({ "campaignId": "stress-campaign", "from": "mobilizing", "to": "blue_building" }),
        Some("stress-campaign"),
    );

    while events.len() < 100_000 {
        let i = events.len();
        let sid = format!("agent-{}", i % 30);
        if i.is_multiple_of(97) {
            push(
                &mut events,
                "agent.communication",
                json!({ "campaignId": "stress-campaign", "fromSessionId": sid, "toSessionId": format!("agent-{}", (i + 7) % 30), "channel": "handoff" }),
                Some("stress-campaign"),
            );
        } else if i.is_multiple_of(211) {
            push(
                &mut events,
                "session.usage",
                json!({ "sessionId": sid, "campaignId": "stress-campaign", "inputTokens": i, "outputTokens": i as f64 / 4.0, "contextTokens": i % 180_000, "costUsd": (i % 1000) as f64 / 1000.0 }),
                Some(&sid),
            );
        } else {
            push(
                &mut events,
                "session.tool_use",
                json!({
                    "sessionId": sid, "campaignId": "stress-campaign", "name": if i.is_multiple_of(5) { "Grep" } else { "Read" },
                    "workspaceId": "cameo", "dir": format!("zone-{}", i % 200), "path": format!("zone-{}/file-{}.rs", i % 200, i % 10000),
                    "summary": format!("inspect file {}", i % 10000),
                }),
                Some(&sid),
            );
        }
    }
    push(
        &mut events,
        "endpoint.health",
        json!({ "endpointId": "local-a", "status": "down", "detail": "injected outage" }),
        Some("local-a"),
    );
    push(
        &mut events,
        "endpoint.routed",
        json!({ "sessionId": "agent-0", "endpointId": "cloud-b", "model": "frontier", "reason": "local-a went down" }),
        Some("agent-0"),
    );
    push(
        &mut events,
        "team.handoff_started",
        json!({ "campaignId": "stress-campaign", "team": "blue", "fromSessionId": "agent-0", "toSessionId": "agent-1" }),
        Some("stress-campaign"),
    );
    push(
        &mut events,
        "team.handoff_completed",
        json!({ "campaignId": "stress-campaign", "team": "blue", "fromSessionId": "agent-0", "toSessionId": "agent-1" }),
        Some("stress-campaign"),
    );

    let replay = |events: &[Event]| {
        let mut p = Projection::new(config.clone());
        for e in events {
            p.apply(e);
        }
        p
    };
    let start = Instant::now();
    let live = replay(&events);
    let first = start.elapsed();
    let start = Instant::now();
    let rebuilt = replay(&events);
    let second = start.elapsed();
    let now = t0 + events.len() as i64 + 1;
    let start = Instant::now();
    let a = live.snapshot(now);
    let snapshot_ms = start.elapsed().as_millis();
    let b = rebuilt.snapshot(now);

    assert_eq!(a["seq"], events.last().unwrap().seq);
    assert_eq!(a["sessions"].as_array().unwrap().len(), 30);
    let blue_active = a["campaigns"][0]["teams"]["blue"]["members"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["status"] == "active")
        .count();
    assert_eq!(blue_active, 17);
    assert_eq!(
        a["endpoints"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["id"] == "local-a")
            .unwrap()["status"],
        "down"
    );
    assert_eq!(session(&a, "agent-0")["endpointId"], "cloud-b");
    assert_eq!(a["graph"]["totals"]["nodes"], b["graph"]["totals"]["nodes"]);
    assert_eq!(a["graph"]["totals"]["edges"], b["graph"]["totals"]["edges"]);
    assert_eq!(a["campaigns"], b["campaigns"]);
    assert_eq!(a["graph"]["nodes"].as_array().unwrap().len(), 4000);
    assert_eq!(a["graph"]["truncated"], true);
    assert!(a["graph"]["edges"].as_array().unwrap().len() <= 8000);
    assert!(
        a["graph"]["totals"]["nodes"].as_u64().unwrap() >= 10_000,
        "expected 10k graph nodes, got {}",
        a["graph"]["totals"]["nodes"]
    );
    eprintln!(
        "stress: {} events, replay {:?} / {:?}, snapshot {snapshot_ms}ms, {} graph nodes",
        events.len(),
        first,
        second,
        a["graph"]["totals"]["nodes"]
    );
    assert!(
        snapshot_ms < 5_000,
        "bounded snapshot too slow: {snapshot_ms}ms"
    );
    assert!(
        first.as_secs() < 60 && second.as_secs() < 60,
        "replay too slow: {first:?} / {second:?}"
    );
}
