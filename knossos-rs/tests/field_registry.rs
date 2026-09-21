//! Parity oracle for the registry port: `registry-resilience.test.mjs`,
//! with a fake adapter standing in for the external child exactly as the
//! Node test stubs `start`.

use knossos::field::adapter::{Adapter, BeforeStart, EventSink, SessionInfo};
use knossos::field::config::FieldSettings;
use knossos::field::eventlog::{AppendOptions, Event, Source};
use knossos::field::knossos_session::KnossosOptions;
use knossos::field::registry::{Meta, Registry, RegistryOptions};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

struct Fake {
    info: Mutex<SessionInfo>,
    running: Mutex<bool>,
    pauses: AtomicUsize,
    resumes: AtomicUsize,
    cancels: AtomicUsize,
    launches: Arc<AtomicUsize>,
    before: Mutex<Option<BeforeStart>>,
}

impl Fake {
    fn new(id: &str, endpoint: &str, running: bool, launches: Arc<AtomicUsize>) -> Arc<Fake> {
        Arc::new(Fake {
            info: Mutex::new(SessionInfo {
                id: id.into(),
                agent_id: Some("builder-1".into()),
                name: id.into(),
                role: Some("builder".into()),
                model: Some("ornith".into()),
                endpoint_id: Some(endpoint.into()),
                effort: "medium".into(),
                cwd: PathBuf::from("."),
                workspace_id: Some("cameo".into()),
                state: "working".into(),
            }),
            running: Mutex::new(running),
            pauses: AtomicUsize::new(0),
            resumes: AtomicUsize::new(0),
            cancels: AtomicUsize::new(0),
            launches,
            before: Mutex::new(None),
        })
    }
}

impl Adapter for Fake {
    fn info(&self) -> SessionInfo {
        self.info.lock().unwrap().clone()
    }
    fn set_endpoint(&self, endpoint_id: Option<String>, model: Option<String>) {
        let mut i = self.info.lock().unwrap();
        i.endpoint_id = endpoint_id;
        i.model = model;
    }
    fn set_effort(&self, effort: &str) {
        self.info.lock().unwrap().effort = effort.into();
    }
    fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }
    fn owned_processes(&self) -> usize {
        usize::from(self.is_running())
    }
    fn set_before_start(&self, hook: BeforeStart) {
        *self.before.lock().unwrap() = Some(hook);
    }
    fn start(&self, _orders: Option<String>) -> Result<(), String> {
        if let Some(h) = self.before.lock().unwrap().clone() {
            h()?;
        }
        self.launches.fetch_add(1, Ordering::SeqCst);
        *self.running.lock().unwrap() = true;
        Ok(())
    }
    fn send(&self, _text: &str) -> bool {
        true
    }
    fn pause(&self) -> bool {
        self.pauses.fetch_add(1, Ordering::SeqCst);
        *self.running.lock().unwrap() = false;
        true
    }
    fn resume(&self, _orders: Option<String>) -> Result<bool, String> {
        if let Some(h) = self.before.lock().unwrap().clone() {
            h()?;
        }
        self.resumes.fetch_add(1, Ordering::SeqCst);
        *self.running.lock().unwrap() = true;
        Ok(true)
    }
    fn cancel(&self) -> bool {
        self.cancels.fetch_add(1, Ordering::SeqCst);
        *self.running.lock().unwrap() = false;
        true
    }
    fn decide_permission(&self, _id: &Value, _decision: &str) -> bool {
        true
    }
    fn capabilities(&self) -> Value {
        json!({ "kind": "fake" })
    }
}

type Log = Arc<Mutex<Vec<Event>>>;

fn emitter(log: Log) -> knossos::field::registry::Emit {
    Arc::new(move |kind: &str, data: Value, options: AppendOptions| {
        let mut events = log.lock().unwrap();
        let event = Event {
            seq: events.len() as u64 + 1,
            ts: 0,
            kind: kind.into(),
            actor: options.actor,
            subject: options.subject,
            source: options.source.unwrap_or(Source::Observed),
            data,
        };
        events.push(event.clone());
        Some(event)
    })
}

fn settings(defaults: Value) -> Arc<RwLock<FieldSettings>> {
    let cwd = std::env::current_dir().unwrap();
    Arc::new(RwLock::new(FieldSettings {
        field_dir: cwd.clone(),
        field: json!({}),
        defaults,
        workspaces: vec![
            json!({ "id": "cameo", "name": "Cameo", "path": cwd.to_string_lossy(), "mounted": true }),
        ],
        endpoints: vec![
            json!({ "id": "local", "kind": "openai-compatible", "model": "ornith" }),
            json!({ "id": "cloud", "kind": "openai-compatible", "model": "frontier" }),
        ],
        websites: vec![],
        roles: vec![
            json!({ "id": "builder", "default_endpoint": "local", "default_thinking": "medium", "body": "" }),
        ],
        agents: vec![
            json!({ "id": "builder-1", "role": "builder", "endpoint": "local", "thinking": "medium" }),
        ],
        missions: vec![],
        constitutions: vec![],
        skills: vec![],
        routines: vec![],
        memory: vec![],
    }))
}

fn count(log: &Log, kind: &str) -> usize {
    log.lock()
        .unwrap()
        .iter()
        .filter(|e| e.kind == kind)
        .count()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn endpoint_failover_budget_pause_and_exactly_once_settlement() {
    let log: Log = Arc::new(Mutex::new(Vec::new()));
    let registry = Registry::new(RegistryOptions {
        settings: settings(json!({ "budget_usd_per_session": 1 })),
        emit: emitter(Arc::clone(&log)),
        api_base: "http://127.0.0.1:1".into(),
        keys: None,
        capabilities: None,
        register_secret: None,
        campaign_policy: None,
        factory: None,
    });
    let launches = Arc::new(AtomicUsize::new(0));
    let session = Fake::new("s1", "local", true, Arc::clone(&launches));
    {
        let mut r = registry.lock().unwrap();
        r.endpoint_status.insert("local".into(), "up".into());
        r.endpoint_status.insert("cloud".into(), "up".into());
        r.sessions.insert("s1".into(), session.clone());
        r.meta.insert(
            "s1".into(),
            Meta {
                budget_usd: 1.0,
                campaign_id: Some("c1".into()),
                team: Some("blue".into()),
                objective_id: Some("o1".into()),
                ..Default::default()
            },
        );
        r.on_endpoint_health("local", "down");
    }
    assert_eq!(session.info().endpoint_id.as_deref(), Some("cloud"));
    assert_eq!(session.pauses.load(Ordering::SeqCst), 1);
    assert_eq!(session.resumes.load(Ordering::SeqCst), 1);
    assert_eq!(count(&log, "endpoint.routed"), 1);

    // Repeating the same down status cannot produce a reroute storm.
    registry.lock().unwrap().on_endpoint_health("local", "down");
    assert_eq!(count(&log, "endpoint.routed"), 1);

    // When the last endpoint goes down: fail closed, blocked, no fake reroute.
    registry.lock().unwrap().on_endpoint_health("cloud", "down");
    assert_eq!(session.pauses.load(Ordering::SeqCst), 2);
    assert_eq!(session.resumes.load(Ordering::SeqCst), 1);
    {
        let events = log.lock().unwrap();
        let last = events.last().unwrap();
        assert_eq!(last.kind, "session.state");
        assert_eq!(last.data["state"], "blocked");
        assert!(last.data["detail"]
            .as_str()
            .unwrap()
            .contains("no alternative"));
    }
    assert!(registry
        .lock()
        .unwrap()
        .route(Some("auto"), Some("builder"))["endpointId"]
        .is_null());

    // Budget exhaustion pauses once the cumulative cost crosses the cap.
    *session.running.lock().unwrap() = true;
    registry.lock().unwrap().on_session_event(
        "s1",
        "session.usage",
        json!({ "sessionId": "s1", "costUsd": 1.25 }),
    );
    assert!(log.lock().unwrap().iter().any(|e| e.kind == "session.state"
        && e.data["detail"]
            .as_str()
            .is_some_and(|d| d.contains("budget exhausted"))));
    assert_eq!(registry.lock().unwrap().meta["s1"].cost_usd, 1.25);

    // Assignment settlement converges exactly once.
    {
        let mut r = registry.lock().unwrap();
        r.meta.get_mut("s1").unwrap().assignment_id = Some("assignment-success".into());
        r.meta.insert(
            "s2".into(),
            Meta {
                assignment_id: Some("assignment-success".into()),
                ..Default::default()
            },
        );
        r.assignments.insert(
            "assignment-success".into(),
            knossos::field::registry::Assignment {
                members: vec!["s1".into(), "s2".into()],
                outcomes: vec![],
                settled: false,
            },
        );
        r.on_session_event(
            "s1",
            "session.ended",
            json!({ "sessionId": "s1", "reason": "exit" }),
        );
        assert_eq!(count(&log, "assignment.completed"), 0);
        r.on_session_event(
            "s2",
            "session.ended",
            json!({ "sessionId": "s2", "reason": "exit" }),
        );
        r.on_session_event(
            "s2",
            "session.ended",
            json!({ "sessionId": "s2", "reason": "exit" }),
        );
    }
    assert_eq!(
        count(&log, "assignment.completed"),
        1,
        "completion converges exactly once"
    );
    {
        let mut r = registry.lock().unwrap();
        r.meta.insert(
            "s3".into(),
            Meta {
                assignment_id: Some("assignment-failure".into()),
                ..Default::default()
            },
        );
        r.assignments.insert(
            "assignment-failure".into(),
            knossos::field::registry::Assignment {
                members: vec!["s3".into()],
                outcomes: vec![],
                settled: false,
            },
        );
        r.on_session_event(
            "s3",
            "session.ended",
            json!({ "sessionId": "s3", "reason": "error" }),
        );
    }
    assert_eq!(count(&log, "assignment.failed"), 1);

    // A shared campaign cap pauses the cohort only once.
    *session.running.lock().unwrap() = true;
    let pauses_before = session.pauses.load(Ordering::SeqCst);
    {
        let mut r = registry.lock().unwrap();
        let meta = r.meta.get_mut("s1").unwrap();
        meta.campaign_id = Some("shared-budget".into());
        meta.budget_usd = 10.0;
        r.campaign_policy = Arc::new(|_| {
            Some(
                json!({ "id": "shared-budget", "costUsd": 2.1, "budgetUsd": 2, "budgetExhausted": true }),
            )
        });
        r.on_session_event(
            "s1",
            "session.usage",
            json!({ "sessionId": "s1", "costUsd": 1.4 }),
        );
        r.on_session_event(
            "s1",
            "session.usage",
            json!({ "sessionId": "s1", "costUsd": 1.5 }),
        );
    }
    assert_eq!(count(&log, "campaign.budget_exhausted"), 1);
    assert_eq!(session.pauses.load(Ordering::SeqCst), pauses_before + 1);

    // Completion tokens are a budget too.
    let completion = Fake::new("s4", "local", true, Arc::clone(&launches));
    {
        let mut r = registry.lock().unwrap();
        r.sessions.insert("s4".into(), completion.clone());
        r.meta.insert(
            "s4".into(),
            Meta {
                budget_usd: 10.0,
                max_output_tokens: 100.0,
                ..Default::default()
            },
        );
        r.on_session_event(
            "s4",
            "session.usage",
            json!({ "sessionId": "s4", "outputTokens": 100 }),
        );
        r.on_session_event(
            "s4",
            "session.usage",
            json!({ "sessionId": "s4", "outputTokens": 120 }),
        );
    }
    assert_eq!(completion.pauses.load(Ordering::SeqCst), 1);
    assert_eq!(
        log.lock()
            .unwrap()
            .iter()
            .filter(|e| e.kind == "budget.exhausted" && e.data["sessionId"] == "s4")
            .count(),
        1
    );

    // Exhausted sessions refuse every command that would do work.
    {
        let mut r = registry.lock().unwrap();
        for kind in ["resume", "escalate", "say", "redirect"] {
            let err = r
                .command(kind, &json!({ "sessionIds": ["s1"] }))
                .unwrap_err();
            assert!(err.0.contains("exhausted"), "{kind}: {}", err.0);
        }
        r.on_session_event(
            "s1",
            "session.usage",
            json!({ "sessionId": "s1", "costUsd": 0.1 }),
        );
        assert!(
            r.meta["s1"].cost_usd >= 1.0,
            "out-of-order usage never refunds spent cost"
        );
        r.meta.insert(
            "deadline".into(),
            Meta {
                deadline_at: Some(1),
                budget_usd: 10.0,
                ..Default::default()
            },
        );
        r.sessions.insert(
            "deadline".into(),
            Fake::new("deadline", "local", true, Arc::clone(&launches)),
        );
        assert!(r
            .command("resume", &json!({ "sessionIds": ["deadline"] }))
            .unwrap_err()
            .0
            .contains("exhausted"));
    }

    // Logout denies every pending approval and cancels every session.
    let leaving = Fake::new("leave-1", "local", true, Arc::clone(&launches));
    {
        let mut r = registry.lock().unwrap();
        r.sessions.insert("leave-1".into(), leaving.clone());
        let outcome = r.request_permission(
            "leave-1",
            "Bash",
            json!({ "command": "npm test" }),
            None,
            None,
        );
        assert!(matches!(
            outcome,
            knossos::field::registry::PermissionOutcome::Pending(_)
        ));
        assert_eq!(r.pending_permissions(), 1);
        let abandoned = r.abandon_operator();
        assert!(abandoned["denied"].as_u64().unwrap() >= 1);
        assert!(abandoned["cancelled"].as_u64().unwrap() >= 1);
        assert_eq!(r.pending_permissions(), 0);
    }
    assert_eq!(leaving.cancels.load(Ordering::SeqCst), 1);
    assert!(log
        .lock()
        .unwrap()
        .iter()
        .any(|e| e.kind == "permission.decided"
            && e.data["by"] == "operator"
            && e.data["decision"] == "deny"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admission_reserves_campaign_budget_and_denied_work_never_starts() {
    let log: Log = Arc::new(Mutex::new(Vec::new()));
    let launches = Arc::new(AtomicUsize::new(0));
    let factory_launches = Arc::clone(&launches);
    let factory: knossos::field::registry::AdapterFactory =
        Arc::new(move |opts: KnossosOptions, _sink: EventSink| {
            let f = Fake::new(
                &opts.id,
                opts.endpoint_id.as_deref().unwrap_or("local"),
                false,
                Arc::clone(&factory_launches),
            );
            f as Arc<dyn Adapter>
        });
    let registry = Registry::new(RegistryOptions {
        settings: settings(json!({ "budget_usd_per_session": 1 })),
        emit: emitter(Arc::clone(&log)),
        api_base: "http://127.0.0.1:1".into(),
        keys: None,
        capabilities: None,
        register_secret: None,
        campaign_policy: Some(Arc::new(|_| {
            Some(json!({ "budgetUsd": 2.5, "costUsd": 0 }))
        })),
        factory: Some(factory),
    });
    let mut r = registry.lock().unwrap();
    let first = r
        .spawn(&json!({ "agentId": "builder-1", "campaignId": "limited", "budgetUsd": 2 }))
        .unwrap();
    let second = r
        .spawn(&json!({ "agentId": "builder-1", "campaignId": "limited", "budgetUsd": 2 }))
        .unwrap();
    assert_eq!(r.meta[&first].budget_usd, 2.0);
    assert_eq!(r.meta[&second].budget_usd, 0.5);
    assert!(r
        .spawn(&json!({ "agentId": "builder-1", "campaignId": "limited" }))
        .unwrap_err()
        .0
        .contains("reserved"));
    assert_eq!(
        launches.load(Ordering::SeqCst),
        2,
        "denied work never starts a child"
    );
    assert!(r
        .command(
            "verify",
            &json!({ "sessionIds": [first], "verifierAgentId": "builder-1" })
        )
        .unwrap_err()
        .0
        .contains("reserved"));
    r.assignments.insert(
        "follow-up".into(),
        knossos::field::registry::Assignment {
            members: vec![first.clone()],
            outcomes: vec![],
            settled: false,
        },
    );
    assert!(r
        .command(
            "reinforce",
            &json!({ "assignmentId": "follow-up", "agentIds": ["builder-1"], "target": { "type": "workspace", "id": "cameo", "workspaceId": "cameo" } })
        )
        .unwrap_err()
        .0
        .contains("reserved"));
    assert_eq!(
        launches.load(Ordering::SeqCst),
        2,
        "verification and reinforcement cannot escape campaign reservations"
    );
    assert!(r
        .spawn(&json!({ "agentId": "builder-1", "verifyFor": first, "campaignId": "bypass" }))
        .unwrap_err()
        .0
        .contains("cannot differ"));
    r.campaign_policy = Arc::new(|_| Some(json!({ "budgetUsd": 10, "costUsd": 0 })));
    let reinforced = r
        .spawn(&json!({ "agentId": "builder-1", "assignmentId": "follow-up" }))
        .unwrap();
    assert_eq!(r.meta[&reinforced].campaign_id.as_deref(), Some("limited"));
    assert!(
        r.assignments["follow-up"].members.contains(&reinforced),
        "reinforcement joins settlement membership"
    );
    r.campaign_policy = Arc::new(|_| Some(json!({ "budgetUsd": 2.5, "costUsd": 0 })));
    r.on_session_event(
        &first,
        "session.ended",
        json!({ "sessionId": first, "reason": "error" }),
    );
    assert!(
        r.spawn(&json!({ "agentId": "builder-1", "campaignId": "limited" }))
            .unwrap_err()
            .0
            .contains("reserved"),
        "unbilled failure cannot refund the budget"
    );
    assert!(log
        .lock()
        .unwrap()
        .iter()
        .any(|e| e.kind == "session.spawned" && e.data["sessionId"] == first));
    r.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capacity_limits_count_owned_processes() {
    let log: Log = Arc::new(Mutex::new(Vec::new()));
    let launches = Arc::new(AtomicUsize::new(0));
    let registry = Registry::new(RegistryOptions {
        settings: settings(json!({ "budget_usd_per_session": 1, "max_concurrent_sessions": 2 })),
        emit: emitter(Arc::clone(&log)),
        api_base: "http://127.0.0.1:1".into(),
        keys: None,
        capabilities: None,
        register_secret: None,
        campaign_policy: Some(Arc::new(|_| Some(json!({ "concurrency": 1 })))),
        factory: None,
    });
    let mut r = registry.lock().unwrap();
    let running = Fake::new("running", "local", true, Arc::clone(&launches));
    r.sessions.insert("running".into(), running.clone());
    r.meta.insert(
        "running".into(),
        Meta {
            campaign_id: Some("one".into()),
            ..Default::default()
        },
    );
    assert!(r
        .assert_capacity(None, Some("cloud"), Some("one"))
        .unwrap_err()
        .0
        .contains("campaign session"));
    r.assert_capacity(Some("running"), Some("local"), Some("one"))
        .unwrap();
    r.sessions.insert(
        "other".into(),
        Fake::new("other", "cloud", true, Arc::clone(&launches)),
    );
    assert!(r
        .spawn(&json!({ "agentId": "builder-1" }))
        .unwrap_err()
        .0
        .contains("global session"));
    let paused = Fake::new("paused", "local", false, Arc::clone(&launches));
    r.sessions.insert("paused".into(), paused.clone());
    assert!(r
        .command("resume", &json!({ "sessionIds": ["paused"] }))
        .unwrap_err()
        .0
        .contains("global session"));
    assert_eq!(paused.resumes.load(Ordering::SeqCst), 0);
    r.sessions.remove("other");
    {
        let mut s = r.settings.write().unwrap();
        for e in s.endpoints.iter_mut() {
            e["max_concurrent_sessions"] = json!(1);
        }
    }
    assert!(r
        .spawn(&json!({ "agentId": "builder-1" }))
        .unwrap_err()
        .0
        .contains("endpoint session"));
    *running.running.lock().unwrap() = false;
    r.command("resume", &json!({ "sessionIds": ["paused"] }))
        .unwrap();
    assert_eq!(
        paused.resumes.load(Ordering::SeqCst),
        1,
        "drained processes release capacity"
    );
}
