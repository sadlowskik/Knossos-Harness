//! Parity oracle for the event log port: `field/server/test/eventlog.test.mjs`,
//! `replay.test.mjs` and the log half of `api-pagination.test.mjs`, case for
//! case.

use knossos::field::eventlog::{AppendOptions, Backend, EventLogError};
use knossos::field::{
    parse_event_page, read_event_range, replay_into, Event, EventLog, Health, Range, ReadEvents,
    Source, EVENT_LOG_SCHEMA_VERSION,
};
use serde_json::json;

fn kinds(events: &[Event]) -> Vec<&str> {
    events.iter().map(|e| e.kind.as_str()).collect()
}

fn seqs(events: &[Event]) -> Vec<u64> {
    events.iter().map(|e| e.seq).collect()
}

#[test]
fn versioned_durability_redaction_integrity_and_tail_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let configured_secret = "configured-secret-value-92".to_string();

    let mut first = EventLog::open(dir.path(), vec![configured_secret.clone()], None).unwrap();
    first
        .append(
            "campaign.created",
            json!({"campaignId": "c1", "name": "restart"}),
            AppendOptions::subject("c1"),
        )
        .unwrap();
    first
        .append(
            "campaign.phase_changed",
            json!({"campaignId": "c1", "from": "draft", "to": "mobilizing"}),
            AppendOptions::subject("c1"),
        )
        .unwrap();
    first
        .append(
            "simulation.started",
            json!({"simulated": true}),
            AppendOptions::subject("sim-1").simulated(),
        )
        .unwrap();
    first
        .append(
            "session.tool_use",
            json!({
                "sessionId": "s1", "inputTokens": 42,
                "input": {"api_key": "must-not-persist", "nested": {"authorization": "Bearer must-not-persist"}},
                "preview": format!("tool accidentally echoed {configured_secret} and sk-examplecredential99"),
            }),
            AppendOptions::subject("s1"),
        )
        .unwrap();
    assert_eq!(first.size(), 4);
    first.close();

    let second = EventLog::open(dir.path(), vec![], None).unwrap();
    assert_eq!(second.size(), 4);
    assert_eq!(
        second.health(),
        Health {
            backend: Backend::Sqlite,
            schema_version: EVENT_LOG_SCHEMA_VERSION,
            integrity: "ok",
            events: 4,
        }
    );
    let all = second.read(0, 100_000).unwrap();
    assert_eq!(
        kinds(&all),
        [
            "campaign.created",
            "campaign.phase_changed",
            "simulation.started",
            "session.tool_use"
        ]
    );
    let sources: Vec<Source> = all.iter().map(|e| e.source).collect();
    assert_eq!(
        sources,
        [
            Source::Observed,
            Source::Observed,
            Source::Synthetic,
            Source::Observed
        ]
    );
    assert_eq!(second.by_subject("c1", 0, 100_000).unwrap().len(), 2);
    let safe = second.by_subject("s1", 0, 100_000).unwrap()[0].data.clone();
    assert_eq!(
        safe["inputTokens"], 42,
        "ordinary token counters are not credentials"
    );
    assert_eq!(safe["input"]["api_key"], "[REDACTED]");
    assert_eq!(safe["input"]["nested"]["authorization"], "[REDACTED]");
    let text = safe.to_string();
    assert!(
        !text.contains("configured-secret-value-92") && !text.contains("sk-examplecredential99")
    );
    assert!(safe["preview"].as_str().unwrap().contains("[REDACTED]"));
    second.close();

    // JSONL: one torn tail record is ignored during recovery.
    let jsonl_dir = tempfile::tempdir().unwrap();
    let mut jsonl = EventLog::open(jsonl_dir.path(), vec![], Some(Backend::Jsonl)).unwrap();
    jsonl
        .append("fixture.one", json!({"ok": true}), AppendOptions::default())
        .unwrap();
    jsonl
        .append("fixture.two", json!({"ok": true}), AppendOptions::default())
        .unwrap();
    jsonl.close();
    let jsonl_path = jsonl_dir.path().join("events.jsonl");
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&jsonl_path)
            .unwrap();
        f.write_all(b"{\"seq\":3").unwrap();
    }
    let jsonl = EventLog::open(jsonl_dir.path(), vec![], Some(Backend::Jsonl)).unwrap();
    assert_eq!(
        jsonl.size(),
        2,
        "one torn tail record is ignored during recovery"
    );
    assert_eq!(jsonl.backend(), Backend::Jsonl);
    jsonl.close();

    // Malformed committed history must stop startup instead of silently losing state.
    let committed = std::fs::read_to_string(&jsonl_path).unwrap();
    let mut lines: Vec<&str> = committed.split('\n').collect();
    lines[0] = "{broken committed history";
    std::fs::write(&jsonl_path, lines.join("\n")).unwrap();
    let err = EventLog::open(jsonl_dir.path(), vec![], Some(Backend::Jsonl)).unwrap_err();
    assert!(
        matches!(err, EventLogError::JsonlMalformed { line: 1, .. }),
        "{err}"
    );
    assert!(err.to_string().contains("malformed at line 1"));
}

#[test]
fn a_gap_in_committed_jsonl_history_is_fatal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.jsonl");
    std::fs::write(
        &path,
        "{\"seq\":1,\"ts\":1,\"kind\":\"a\",\"data\":{}}\n{\"seq\":3,\"ts\":1,\"kind\":\"b\",\"data\":{}}\n",
    )
    .unwrap();
    let err = EventLog::open(dir.path(), vec![], Some(Backend::Jsonl)).unwrap_err();
    assert!(
        matches!(
            err,
            EventLogError::JsonlSequence {
                line: 2,
                expected: 2
            }
        ),
        "{err}"
    );
}

#[test]
fn a_schema_one_database_is_migrated_in_place() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = rusqlite::Connection::open(dir.path().join("field.db")).unwrap();
        db.execute_batch(
            "CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, kind TEXT NOT NULL, actor TEXT, subject TEXT, data TEXT NOT NULL);
             INSERT INTO events (ts, kind, actor, subject, data) VALUES (1, 'legacy.event', NULL, 's', '{\"simulated\":true}');
             PRAGMA user_version = 1;",
        )
        .unwrap();
    }
    let log = EventLog::open(dir.path(), vec![], Some(Backend::Sqlite)).unwrap();
    assert_eq!(log.size(), 1);
    let rows = log.read(0, 10).unwrap();
    assert_eq!(rows[0].kind, "legacy.event");
    // The migration adds the column with its default, so committed history is
    // `observed` whatever its payload says: the reference does the same, and
    // the `data.simulated` inference only covers a NULL or unknown source.
    assert_eq!(rows[0].source, Source::Observed);
    let version: i64 = rusqlite::Connection::open(dir.path().join("field.db"))
        .unwrap()
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .unwrap();
    assert_eq!(version, EVENT_LOG_SCHEMA_VERSION);
}

#[test]
fn a_newer_schema_refuses_to_open() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = rusqlite::Connection::open(dir.path().join("field.db")).unwrap();
        db.execute_batch("PRAGMA user_version = 99;").unwrap();
    }
    let err = EventLog::open(dir.path(), vec![], Some(Backend::Sqlite)).unwrap_err();
    assert!(
        matches!(err, EventLogError::SchemaTooNew { found: 99, .. }),
        "{err}"
    );
}

struct FakeLog(Vec<Event>);

impl ReadEvents for FakeLog {
    fn read(&self, from_seq: u64, limit: usize) -> Vec<Event> {
        self.0
            .iter()
            .filter(|e| e.seq > from_seq)
            .take(limit)
            .cloned()
            .collect()
    }
}

#[test]
fn unbounded_paging_and_exact_historical_ceilings() {
    let events: Vec<Event> = (0..25_017u64)
        .map(|index| Event {
            seq: index + 1,
            ts: 1_700_000_000_000 + index as i64,
            kind: "fixture.tick".into(),
            actor: None,
            subject: Some("paging".into()),
            source: Source::Observed,
            data: json!({"index": index}),
        })
        .collect();
    let log = FakeLog(events.clone());

    let all = read_event_range(&log, Range::default().page_size(997));
    assert_eq!(all.len(), events.len());
    assert_eq!(all.last().unwrap().seq, events.len() as u64);

    let prefix = read_event_range(&log, Range::default().to(10_003).page_size(512));
    assert_eq!(prefix.len(), 10_003);
    assert_eq!(prefix.last().unwrap().seq, 10_003);

    let mut seen = Vec::new();
    let result = replay_into(
        &log,
        &mut |e: &Event| seen.push(e.seq),
        Range::default().to(20_001).page_size(333),
    );
    assert_eq!(result.count, 20_001);
    assert_eq!(result.last_event.unwrap().seq, 20_001);
    assert_eq!(&seen[seen.len() - 3..], &[19_999, 20_000, 20_001]);
}

#[test]
fn bounded_event_and_subject_windows_preserve_cursor_order() {
    assert!(parse_event_page(None, None).is_ok());
    for backend in [Backend::Jsonl, Backend::Sqlite] {
        let dir = tempfile::tempdir().unwrap();
        let mut log = EventLog::open(dir.path(), vec![], Some(backend)).unwrap();
        for i in 0..7 {
            log.append(
                "session.message",
                json!({"i": i}),
                AppendOptions::subject("s1"),
            )
            .unwrap();
        }
        log.append(
            "other.event",
            json!({"i": 8}),
            AppendOptions::subject("other"),
        )
        .unwrap();
        assert_eq!(seqs(&log.read(0, 3).unwrap()), [1, 2, 3]);
        assert_eq!(seqs(&log.read(3, 3).unwrap()), [4, 5, 6]);
        let i = |events: Vec<Event>| -> Vec<i64> {
            events
                .iter()
                .map(|e| e.data["i"].as_i64().unwrap())
                .collect()
        };
        assert_eq!(i(log.by_subject("s1", 3, 3).unwrap()), [3, 4, 5]);
        assert_eq!(i(log.by_subject("s1", 6, 3).unwrap()), [6]);
        assert!(log.by_subject("missing", 0, 3).unwrap().is_empty());

        // Listeners fire after the durable write, with the stored event.
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = seen.clone();
        let id = log.subscribe(Box::new(move |e| sink.lock().unwrap().push(e.seq)));
        log.append("listened", json!({}), AppendOptions::default())
            .unwrap();
        log.unsubscribe(id);
        log.append("unheard", json!({}), AppendOptions::default())
            .unwrap();
        assert_eq!(*seen.lock().unwrap(), vec![9]);

        // The wire shape the client reads: the same field names as the Node server.
        let last = log.read(9, 1).unwrap().remove(0);
        let wire = serde_json::to_value(&last).unwrap();
        assert_eq!(wire["seq"], 10);
        assert_eq!(wire["kind"], "unheard");
        assert_eq!(wire["source"], "observed");
        assert!(wire["actor"].is_null() && wire["subject"].is_null());
        assert!(wire["ts"].as_i64().unwrap() > 1_700_000_000_000);
    }
}
