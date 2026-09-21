//! The event log: the single source of truth for operational state. Port of
//! `field/server/src/store/db.js`.
//!
//! Two backends with identical semantics. SQLite is the default (WAL, one
//! implicit transaction per insert, `PRAGMA user_version` carries the schema
//! version and `quick_check` runs at open). JSONL is the strong-durability
//! path: one line per event, `fsync` per append, the whole file replayed at
//! open under strict rules. A process can stop between write and fsync, so
//! exactly one malformed *trailing* record is tolerated; a malformed record
//! inside committed history refuses to open rather than silently losing
//! state, because a fold over a log with a hole is a fold over a lie.
//!
//! Every payload passes through [`super::sanitize`] before it reaches disk.

use super::replay::ReadEvents;
use super::sanitize::sanitize_event_data;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const EVENT_LOG_SCHEMA_VERSION: i64 = 2;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS events (
  seq     INTEGER PRIMARY KEY AUTOINCREMENT,
  ts      INTEGER NOT NULL,
  kind    TEXT    NOT NULL,
  actor   TEXT,
  subject TEXT,
  source  TEXT    NOT NULL DEFAULT 'observed',
  data    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_kind    ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_subject ON events(subject);
CREATE INDEX IF NOT EXISTS idx_events_ts      ON events(ts);
";

/// Where an event came from. `synthetic` events belong to a rehearsal and
/// are folded into a separate projection so they never touch real state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Observed,
    Derived,
    Manual,
    Synthetic,
}

impl Source {
    pub const ALL: [Source; 4] = [
        Source::Observed,
        Source::Derived,
        Source::Manual,
        Source::Synthetic,
    ];

    pub fn parse(text: &str) -> Option<Source> {
        match text {
            "observed" => Some(Source::Observed),
            "derived" => Some(Source::Derived),
            "manual" => Some(Source::Manual),
            "synthetic" => Some(Source::Synthetic),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Source::Observed => "observed",
            Source::Derived => "derived",
            Source::Manual => "manual",
            Source::Synthetic => "synthetic",
        }
    }
}

/// One stored event. The field names are the wire format the web client and
/// the JSONL file use; nothing here is renamed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub seq: u64,
    /// Milliseconds since the Unix epoch, as `Date.now()` reports.
    pub ts: i64,
    pub kind: String,
    #[serde(default)]
    pub actor: Option<String>,
    #[serde(default)]
    pub subject: Option<String>,
    pub source: Source,
    #[serde(default)]
    pub data: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Sqlite,
    Jsonl,
}

impl Backend {
    pub fn parse(text: &str) -> Option<Backend> {
        match text {
            "sqlite" => Some(Backend::Sqlite),
            "jsonl" => Some(Backend::Jsonl),
            _ => None,
        }
    }
}

/// What `health()` reports; the shape the API returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Health {
    pub backend: Backend,
    pub schema_version: i64,
    pub integrity: &'static str,
    pub events: u64,
}

/// Options for one append. All optional; `source` overrides the
/// `simulated` inference.
#[derive(Debug, Default, Clone)]
pub struct AppendOptions {
    pub actor: Option<String>,
    pub subject: Option<String>,
    pub source: Option<Source>,
    pub simulated: bool,
}

impl AppendOptions {
    pub fn subject(subject: impl Into<String>) -> Self {
        AppendOptions {
            subject: Some(subject.into()),
            ..Default::default()
        }
    }

    pub fn simulated(mut self) -> Self {
        self.simulated = true;
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EventLogError {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("Field event database failed integrity check: {0}")]
    Corrupt(String),
    #[error("Field event database has invalid schema version: {0}")]
    InvalidSchemaVersion(i64),
    #[error("Field event database schema {found} is newer than supported schema {supported}")]
    SchemaTooNew { found: i64, supported: i64 },
    #[error("Field event database migration failed: {0}")]
    Migration(rusqlite::Error),
    #[error("{0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("Field JSONL event log is malformed at line {line}: {cause}")]
    JsonlMalformed { line: usize, cause: String },
    #[error("Field JSONL event log has a non-object record at line {0}")]
    JsonlNotObject(usize),
    #[error("Field JSONL event log sequence is invalid at line {line}: expected {expected}")]
    JsonlSequence { line: usize, expected: u64 },
    #[error("Field JSONL event log record is invalid at line {0}")]
    JsonlRecord(usize),
    #[error("Field JSONL event log has invalid source at sequence {0}")]
    JsonlSource(u64),
}

type Listener = Box<dyn Fn(&Event) + Send + Sync>;

enum Store {
    Sqlite(Connection),
    Jsonl { file: PathBuf, rows: Vec<Event> },
}

pub struct EventLog {
    dir: PathBuf,
    store: Store,
    secrets: Vec<String>,
    listeners: Vec<(u64, Listener)>,
    next_listener: u64,
    integrity: &'static str,
    last_seq: u64,
}

impl std::fmt::Debug for EventLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventLog")
            .field("dir", &self.dir)
            .field("backend", &self.backend())
            .field("events", &self.last_seq)
            .field("listeners", &self.listeners.len())
            .finish()
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn keep_secrets(secrets: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for secret in secrets {
        if secret.chars().count() >= 8 && !out.contains(&secret) {
            out.push(secret);
        }
    }
    out
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0 && !f.is_nan()),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

fn simulated_flag(data: &Value) -> bool {
    data.get("simulated").is_some_and(truthy)
}

impl EventLog {
    /// Open (or create) the log under `dir`. `backend` `None` picks SQLite.
    pub fn open(
        dir: &Path,
        secrets: Vec<String>,
        backend: Option<Backend>,
    ) -> Result<EventLog, EventLogError> {
        fs::create_dir_all(dir)?;
        let secrets = keep_secrets(secrets);
        let backend = backend.unwrap_or(Backend::Sqlite);
        let (store, last_seq) = match backend {
            Backend::Sqlite => open_sqlite(dir)?,
            Backend::Jsonl => open_jsonl(dir)?,
        };
        Ok(EventLog {
            dir: dir.to_path_buf(),
            store,
            secrets,
            listeners: Vec::new(),
            next_listener: 1,
            integrity: "ok",
            last_seq,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn backend(&self) -> Backend {
        match self.store {
            Store::Sqlite(_) => Backend::Sqlite,
            Store::Jsonl { .. } => Backend::Jsonl,
        }
    }

    /// Append one event. Returns the stored event with its assigned seq.
    pub fn append(
        &mut self,
        kind: &str,
        data: Value,
        options: AppendOptions,
    ) -> Result<Event, EventLogError> {
        let ts = now_ms();
        let safe = sanitize_event_data(&data, &self.secrets);
        let source = options
            .source
            .unwrap_or(if options.simulated || simulated_flag(&safe) {
                Source::Synthetic
            } else {
                Source::Observed
            });
        let event = match &mut self.store {
            Store::Sqlite(db) => {
                db.execute(
                    "INSERT INTO events (ts, kind, actor, subject, source, data) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![ts, kind, options.actor, options.subject, source.as_str(), safe.to_string()],
                )?;
                self.last_seq = db.last_insert_rowid() as u64;
                Event {
                    seq: self.last_seq,
                    ts,
                    kind: kind.to_string(),
                    actor: options.actor,
                    subject: options.subject,
                    source,
                    data: safe,
                }
            }
            Store::Jsonl { file, rows } => {
                self.last_seq += 1;
                let event = Event {
                    seq: self.last_seq,
                    ts,
                    kind: kind.to_string(),
                    actor: options.actor,
                    subject: options.subject,
                    source,
                    data: safe,
                };
                rows.push(event.clone());
                let mut line = serde_json::to_string(&event).expect("event serializes");
                line.push('\n');
                let mut handle = append_handle(file)?;
                handle.write_all(line.as_bytes())?;
                handle.sync_all()?;
                event
            }
        };
        for (_, listener) in &self.listeners {
            listener(&event);
        }
        Ok(event)
    }

    /// Read events after `from_seq`, oldest first.
    pub fn read(&self, from_seq: u64, limit: usize) -> Result<Vec<Event>, EventLogError> {
        match &self.store {
            Store::Sqlite(db) => {
                let mut stmt = db.prepare_cached("SELECT seq, ts, kind, actor, subject, source, data FROM events WHERE seq > ?1 ORDER BY seq ASC LIMIT ?2")?;
                let rows = stmt.query_map(params![from_seq as i64, limit as i64], hydrate)?;
                rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
            }
            Store::Jsonl { rows, .. } => Ok(rows
                .iter()
                .filter(|r| r.seq > from_seq)
                .take(limit)
                .cloned()
                .collect()),
        }
    }

    /// Every event about one subject, in order, after `from_seq`.
    pub fn by_subject(
        &self,
        subject: &str,
        from_seq: u64,
        limit: usize,
    ) -> Result<Vec<Event>, EventLogError> {
        match &self.store {
            Store::Sqlite(db) => {
                let mut stmt = db.prepare_cached("SELECT seq, ts, kind, actor, subject, source, data FROM events WHERE subject = ?1 AND seq > ?2 ORDER BY seq ASC LIMIT ?3")?;
                let rows =
                    stmt.query_map(params![subject, from_seq as i64, limit as i64], hydrate)?;
                rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
            }
            Store::Jsonl { rows, .. } => Ok(rows
                .iter()
                .filter(|r| r.subject.as_deref() == Some(subject) && r.seq > from_seq)
                .take(limit)
                .cloned()
                .collect()),
        }
    }

    /// Listen to every append, after it is durable. Returns a handle for
    /// [`EventLog::unsubscribe`].
    pub fn subscribe(&mut self, listener: Listener) -> u64 {
        let id = self.next_listener;
        self.next_listener += 1;
        self.listeners.push((id, listener));
        id
    }

    pub fn unsubscribe(&mut self, id: u64) {
        self.listeners.retain(|(other, _)| *other != id);
    }

    /// Register a value to scrub from every later payload.
    pub fn add_secret(&mut self, value: impl Into<String>) {
        let secret = value.into();
        if secret.chars().count() >= 8 && !self.secrets.contains(&secret) {
            self.secrets.push(secret);
        }
    }

    /// The highest sequence number stored.
    pub fn size(&self) -> u64 {
        self.last_seq
    }

    pub fn health(&self) -> Health {
        Health {
            backend: self.backend(),
            schema_version: EVENT_LOG_SCHEMA_VERSION,
            integrity: self.integrity,
            events: self.last_seq,
        }
    }

    /// Drop the listeners and close the store.
    pub fn close(self) {
        drop(self);
    }
}

impl ReadEvents for EventLog {
    fn read(&self, from_seq: u64, limit: usize) -> Vec<Event> {
        EventLog::read(self, from_seq, limit).unwrap_or_default()
    }
}

fn append_handle(file: &Path) -> std::io::Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options.append(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(file)
}

fn open_sqlite(dir: &Path) -> Result<(Store, u64), EventLogError> {
    let db = Connection::open(dir.join("field.db"))?;
    db.execute_batch("PRAGMA journal_mode = WAL;")?;
    let checks: Vec<String> = {
        let mut stmt = db.prepare("PRAGMA quick_check")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<Result<Vec<_>, _>>()?
    };
    if checks.is_empty() || checks.iter().any(|c| c != "ok") {
        return Err(EventLogError::Corrupt(format!("{checks:?}")));
    }
    let stored: i64 = db.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if stored < 0 {
        return Err(EventLogError::InvalidSchemaVersion(stored));
    }
    if stored > EVENT_LOG_SCHEMA_VERSION {
        return Err(EventLogError::SchemaTooNew {
            found: stored,
            supported: EVENT_LOG_SCHEMA_VERSION,
        });
    }
    if stored < EVENT_LOG_SCHEMA_VERSION {
        db.execute_batch("BEGIN IMMEDIATE;")?;
        let migrate = || -> rusqlite::Result<()> {
            db.execute_batch(SCHEMA)?;
            let has_source = {
                let mut stmt = db.prepare("PRAGMA table_info(events)")?;
                let names: Vec<String> = stmt
                    .query_map([], |row| row.get::<_, String>(1))?
                    .collect::<Result<_, _>>()?;
                names.iter().any(|name| name == "source")
            };
            if !has_source {
                db.execute_batch(
                    "ALTER TABLE events ADD COLUMN source TEXT NOT NULL DEFAULT 'observed';",
                )?;
            }
            db.execute_batch(&format!(
                "PRAGMA user_version = {EVENT_LOG_SCHEMA_VERSION}; COMMIT;"
            ))
        };
        if let Err(error) = migrate() {
            let _ = db.execute_batch("ROLLBACK;");
            return Err(EventLogError::Migration(error));
        }
    } else {
        db.execute_batch(SCHEMA)?;
    }
    let last: i64 = db.query_row("SELECT COALESCE(MAX(seq),0) FROM events", [], |row| {
        row.get(0)
    })?;
    Ok((Store::Sqlite(db), last.max(0) as u64))
}

fn hydrate(row: &rusqlite::Row<'_>) -> rusqlite::Result<Event> {
    let seq: i64 = row.get(0)?;
    let ts: i64 = row.get(1)?;
    let kind: String = row.get(2)?;
    let actor: Option<String> = row.get(3)?;
    let subject: Option<String> = row.get(4)?;
    let source: Option<String> = row.get(5)?;
    let raw: String = row.get(6)?;
    let data = serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| json!({ "_unparsed": raw }));
    let source = source
        .as_deref()
        .and_then(Source::parse)
        .unwrap_or(if simulated_flag(&data) {
            Source::Synthetic
        } else {
            Source::Observed
        });
    Ok(Event {
        seq: seq.max(0) as u64,
        ts,
        kind,
        actor,
        subject,
        source,
        data,
    })
}

fn open_jsonl(dir: &Path) -> Result<(Store, u64), EventLogError> {
    let file = dir.join("events.jsonl");
    let mut rows: Vec<Event> = Vec::new();
    if file.exists() {
        let text = fs::read_to_string(&file)?;
        let lines: Vec<(usize, &str)> = text
            .split('\n')
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .collect();
        for (position, (index, line)) in lines.iter().enumerate() {
            let value: Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(error) => {
                    // Only a torn tail is recoverable.
                    if position == lines.len() - 1 {
                        break;
                    }
                    return Err(EventLogError::JsonlMalformed {
                        line: index + 1,
                        cause: error.to_string(),
                    });
                }
            };
            let expected = rows.last().map_or(1, |r| r.seq + 1);
            let event = validate_jsonl_row(value, expected, index + 1)?;
            rows.push(event);
        }
    }
    let last = rows.last().map_or(0, |r| r.seq);
    Ok((Store::Jsonl { file, rows }, last))
}

fn validate_jsonl_row(value: Value, expected: u64, line: usize) -> Result<Event, EventLogError> {
    let Value::Object(mut fields) = value else {
        return Err(EventLogError::JsonlNotObject(line));
    };
    let seq = fields
        .get("seq")
        .and_then(Value::as_u64)
        .filter(|s| *s <= 9_007_199_254_740_991);
    if seq != Some(expected) {
        return Err(EventLogError::JsonlSequence { line, expected });
    }
    let ts = fields
        .get("ts")
        .and_then(Value::as_f64)
        .filter(|t| t.is_finite());
    let kind = fields
        .get("kind")
        .and_then(Value::as_str)
        .filter(|k| !k.is_empty());
    let (Some(ts), Some(kind)) = (ts, kind) else {
        return Err(EventLogError::JsonlRecord(line));
    };
    let kind = kind.to_string();
    let data = fields
        .remove("data")
        .unwrap_or(Value::Object(Default::default()));
    let source = match fields.get("source") {
        None | Some(Value::Null) => {
            if simulated_flag(&data) {
                Source::Synthetic
            } else {
                Source::Observed
            }
        }
        Some(Value::String(s)) => Source::parse(s).ok_or(EventLogError::JsonlSource(expected))?,
        Some(_) => return Err(EventLogError::JsonlSource(expected)),
    };
    let text = |key: &str| fields.get(key).and_then(Value::as_str).map(str::to_string);
    Ok(Event {
        seq: expected,
        ts: ts as i64,
        kind,
        actor: text("actor"),
        subject: text("subject"),
        source,
        data,
    })
}
