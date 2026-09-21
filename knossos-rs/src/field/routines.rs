//! Persistent and scheduled agent work. Port of `field/server/src/routines.js`.
//!
//! A routine fires on a real clock tick or a real filesystem change, and when
//! it fires it spawns a real session. Disabled routines are inert.
//!
//! Timers are not owned here. The Node controller armed `setTimeout`s; this
//! port keeps every due time as data and does the work in [`Routines::poll`],
//! which the server calls every five seconds and tests call by hand. Events
//! reach it through [`Routines::on_event`], fed from the server's fanout task
//! rather than re-entered from inside `emit`, so a routine can never observe
//! its own trigger mid-flight.

use super::config::FieldSettings;
use super::eventlog::{AppendOptions, Event, Source};
use super::js::{get, get_arr, get_str, js_string, truthy, Obj};
use super::policy::glob_to_regex;
use super::projection::Projection;
use super::registry::{Emit, Registry};
use chrono::{DateTime, Datelike, NaiveDateTime, TimeZone, Timelike, Utc};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

// --- minimal 5-field cron: minute hour day-of-month month day-of-week -------

fn match_field(spec: &str, value: i64, min: i64, max: i64) -> bool {
    for part in spec.split(',') {
        if part == "*" {
            return true;
        }
        let (range, step) = match part.split_once('/') {
            Some((range, step)) => (range, step.parse::<i64>().unwrap_or(0)),
            None => (part, 1),
        };
        if step == 0 {
            continue;
        }
        if range == "*" {
            if (value - min) % step == 0 {
                return true;
            }
            continue;
        }
        if let Some((a, b)) = range.split_once('-') {
            let (a, b) = (a.parse::<i64>().unwrap_or(0), b.parse::<i64>().unwrap_or(0));
            if value >= a && value <= b && (value - a) % step == 0 {
                return true;
            }
            continue;
        }
        let start = range.parse::<i64>().unwrap_or(i64::MIN);
        let hit = if part.contains('/') {
            value >= start && value <= max && (value - start) % step == 0
        } else {
            start == value
        };
        if hit {
            return true;
        }
    }
    false
}

fn cron_parts(expr: &str) -> Vec<String> {
    expr.split_whitespace().map(str::to_string).collect()
}

fn day_gate(dom: &str, dow: &str, date: &DateTime<Utc>) -> bool {
    let day = match_field(dom, date.day() as i64, 1, 31);
    let weekday = match_field(dow, date.weekday().num_days_from_sunday() as i64, 0, 6);
    if dom != "*" && dow != "*" {
        day || weekday
    } else {
        day && weekday
    }
}

pub fn cron_matches(expr: &str, at_ms: i64) -> Result<bool, String> {
    validate_cron(expr)?;
    let p = cron_parts(expr);
    let date = Utc
        .timestamp_millis_opt(at_ms)
        .single()
        .ok_or_else(|| "time out of range".to_string())?;
    Ok(match_field(&p[0], date.minute() as i64, 0, 59)
        && match_field(&p[1], date.hour() as i64, 0, 23)
        && day_gate(&p[2], &p[4], &date)
        && match_field(&p[3], date.month() as i64, 1, 12))
}

pub fn validate_cron(expr: &str) -> Result<(), String> {
    let parts = cron_parts(expr);
    if parts.len() != 5 {
        return Err("cron must contain exactly five fields".into());
    }
    let limits = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)];
    for (field, (min, max)) in parts.iter().zip(limits) {
        validate_cron_field(field, min, max)?;
    }
    Ok(())
}

fn validate_cron_field(field: &str, min: i64, max: i64) -> Result<(), String> {
    if field.is_empty() {
        return Err("cron field is empty".into());
    }
    let is_digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    for part in field.split(',') {
        let (range, step_text) = match part.split_once('/') {
            Some((r, s)) => (r, Some(s)),
            None => (part, None),
        };
        let shape_ok = step_text.is_none_or(is_digits)
            && (range == "*"
                || match range.split_once('-') {
                    Some((a, b)) => is_digits(a) && is_digits(b),
                    None => is_digits(range),
                });
        if !shape_ok {
            return Err(format!("unsupported cron field: {part}"));
        }
        let step = match step_text {
            None => 1,
            Some(s) => s.parse::<i64>().unwrap_or(0),
        };
        if step < 1 || step > max - min + 1 {
            return Err(format!("invalid cron step: {part}"));
        }
        if range == "*" {
            continue;
        }
        let (start, end) = match range.split_once('-') {
            Some((a, b)) => (
                a.parse::<i64>().unwrap_or(-1),
                b.parse::<i64>().unwrap_or(-1),
            ),
            None => {
                let v = range.parse::<i64>().unwrap_or(-1);
                (v, v)
            }
        };
        if start < min || end > max || start > end {
            return Err(format!("cron value is outside {min}-{max}: {part}"));
        }
    }
    Ok(())
}

fn iso_ms(date: &DateTime<Utc>) -> String {
    date.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

/// Strictly future UTC occurrence, bounded to eight years (including leap gaps).
pub fn next_cron_run(expr: &str, after_ms: i64) -> Result<Option<String>, String> {
    validate_cron(expr)?;
    let p = cron_parts(expr);
    let (mi, ho, dom, mo, dow) = (&p[0], &p[1], &p[2], &p[3], &p[4]);
    let hours: Vec<u32> = (0..24)
        .filter(|h| match_field(ho, *h as i64, 0, 23))
        .collect();
    let minutes: Vec<u32> = (0..60)
        .filter(|m| match_field(mi, *m as i64, 0, 59))
        .collect();
    let after = Utc
        .timestamp_millis_opt(after_ms)
        .single()
        .ok_or_else(|| "time out of range".to_string())?;
    let mut day = after
        .date_naive()
        .and_hms_opt(0, 0, 0)
        .map(|d| Utc.from_utc_datetime(&d))
        .ok_or_else(|| "time out of range".to_string())?;
    for _ in 0..(366 * 8) {
        if match_field(mo, day.month() as i64, 1, 12) && day_gate(dom, dow, &day) {
            for hour in &hours {
                for minute in &minutes {
                    if let Some(candidate) =
                        day.with_hour(*hour).and_then(|d| d.with_minute(*minute))
                    {
                        if candidate > after {
                            return Ok(Some(iso_ms(&candidate)));
                        }
                    }
                }
            }
        }
        day += chrono::Duration::days(1);
    }
    Ok(None)
}

// --- configuration ----------------------------------------------------------

/// The slice of `field/` a routine needs: the routine records and what they
/// reference. Read fresh through a [`ConfigSource`] on every decision so a
/// live configuration change invalidates queued work, exactly as the Node
/// controller observed its shared `cfg`.
#[derive(Debug, Clone, Default)]
pub struct RoutineConfig {
    pub defaults: Value,
    pub routines: Vec<Value>,
    pub roles: Vec<Value>,
    pub agents: Vec<Value>,
    pub endpoints: Vec<Value>,
    pub workspaces: Vec<Value>,
}

impl RoutineConfig {
    pub fn from_settings(settings: &FieldSettings) -> RoutineConfig {
        RoutineConfig {
            defaults: settings.defaults.clone(),
            routines: settings.routines.clone(),
            roles: settings.roles.clone(),
            agents: settings.agents.clone(),
            endpoints: settings.endpoints.clone(),
            workspaces: settings.workspaces.clone(),
        }
    }

    fn routine(&self, id: &str) -> Option<&Value> {
        self.routines.iter().find(|r| get_str(r, "id") == Some(id))
    }

    fn find<'a>(list: &'a [Value], key: &str, want: Option<&str>) -> Option<&'a Value> {
        let want = want?;
        list.iter().find(|item| get_str(item, key) == Some(want))
    }
}

fn finite(v: Option<&Value>) -> Option<f64> {
    match v {
        Some(Value::Number(n)) => n.as_f64().filter(|f| f.is_finite()),
        Some(Value::String(s)) => s.trim().parse::<f64>().ok().filter(|f| f.is_finite()),
        Some(Value::Bool(b)) => Some(if *b { 1.0 } else { 0.0 }),
        Some(Value::Null) | None => None,
        _ => None,
    }
}

pub fn validate_routine(routine: &Value, cfg: &RoutineConfig) -> Result<(), String> {
    let id = match get(routine, "id") {
        Some(Value::String(id)) if !id.is_empty() => id.clone(),
        _ => return Err("routine id is required".into()),
    };
    let role = get_str(routine, "role");
    if RoutineConfig::find(&cfg.roles, "id", role).is_none() {
        return Err(format!(
            "routine {id} has unknown role {}",
            role.unwrap_or("undefined")
        ));
    }
    let endpoint = get_str(routine, "endpoint");
    if RoutineConfig::find(&cfg.endpoints, "id", endpoint).is_none() {
        return Err(format!(
            "routine {id} has unknown endpoint {}",
            endpoint.unwrap_or("undefined")
        ));
    }
    let workspace = get_str(routine, "workspace");
    let mounted = RoutineConfig::find(&cfg.workspaces, "id", workspace)
        .map(|w| get(w, "mounted").is_some_and(truthy))
        .unwrap_or(false);
    if !mounted {
        return Err(format!(
            "routine {id} has unavailable workspace {}",
            workspace.unwrap_or("undefined")
        ));
    }
    let trigger = get(routine, "trigger").cloned().unwrap_or(Value::Null);
    let kind = get_str(&trigger, "kind").unwrap_or("");
    if !matches!(kind, "schedule" | "fs_change") {
        return Err(format!("routine {id} has unsupported trigger"));
    }
    if kind == "schedule" {
        validate_cron(get_str(&trigger, "cron").unwrap_or(""))?;
    }
    if let Some(tz) = get(&trigger, "timezone") {
        if tz != "UTC" {
            return Err("routine schedules support UTC only".into());
        }
    }
    if let Some(overlap) = get(routine, "overlap") {
        if !matches!(
            overlap.as_str(),
            Some("skip") | Some("queue-one") | Some("replace") | Some("parallel")
        ) {
            return Err(
                "routine overlap policy supports skip, queue-one, replace, or parallel".into(),
            );
        }
    }
    if let Some(max) = get(routine, "max_queued_runs") {
        if max.as_f64() != Some(1.0) {
            return Err("routine max_queued_runs currently supports only 1".into());
        }
    }
    if let Some(cooldown) = get(routine, "cooldown_seconds") {
        match finite(Some(cooldown)) {
            Some(s) if (0.0..=86_400.0).contains(&s) => {}
            _ => return Err(format!("routine {id} has invalid cooldown")),
        }
    }
    if kind == "fs_change" {
        let seconds = match get(&trigger, "debounce_seconds") {
            None => Some(60.0),
            some => finite(some),
        };
        match seconds {
            Some(s) if (1.0..=86_400.0).contains(&s) => {}
            _ => return Err(format!("routine {id} has invalid debounce")),
        }
    }
    if get(routine, "completion")
        .and_then(|c| get(c, "kind"))
        .is_none_or(|k| !truthy(k))
    {
        return Err(format!("routine {id} requires a completion policy"));
    }
    if let Some(budget) = get(routine, "budget_usd") {
        match budget.as_f64() {
            Some(b) if b.is_finite() && b > 0.0 => {}
            _ => return Err(format!("routine {id} has invalid dollar budget")),
        }
    }
    Ok(())
}

// --- the controller ----------------------------------------------------------

/// What a routine needs from the session registry. The real registry
/// implements it; tests substitute a recorder.
pub trait RunSpawner: Send + Sync {
    /// Spawns a session for `body` (the `/api/sessions` shape) and returns
    /// its id.
    fn spawn(&self, body: Value) -> Result<String, String>;
    /// Cancels sessions a `replace` routine superseded.
    fn cancel(&self, _session_ids: &[String]) {}
}

impl RunSpawner for Arc<Mutex<Registry>> {
    fn spawn(&self, body: Value) -> Result<String, String> {
        let mut registry = self
            .lock()
            .map_err(|_| "registry lock poisoned".to_string())?;
        registry.spawn(&body).map_err(|e| e.0)
    }

    fn cancel(&self, session_ids: &[String]) {
        if let Ok(mut registry) = self.lock() {
            let _ = registry.command("cancel", &json!({ "sessionIds": session_ids }));
        }
    }
}

pub type Clock = Arc<dyn Fn() -> i64 + Send + Sync>;
/// Where the controller reads its configuration from, every time.
pub type ConfigSource = Arc<dyn Fn() -> RoutineConfig + Send + Sync>;
pub type IdSource = Arc<dyn Fn() -> String + Send + Sync>;

pub struct RoutinesOptions {
    pub cfg: ConfigSource,
    pub spawner: Arc<dyn RunSpawner>,
    pub emit: Emit,
    pub projection: Arc<Mutex<Projection>>,
    /// Milliseconds since the Unix epoch.
    pub now: Clock,
    pub create_id: IdSource,
}

#[derive(Debug, Clone)]
struct Run {
    run_id: String,
    routine_id: String,
    session_id: String,
}

#[derive(Debug, Clone)]
struct Pending {
    paths: Vec<String>,
    due_at: i64,
}

pub struct Routines {
    cfg: ConfigSource,
    spawner: Arc<dyn RunSpawner>,
    emit: Emit,
    projection: Arc<Mutex<Projection>>,
    now: Clock,
    create_id: IdSource,
    pending: HashMap<String, Pending>,
    enabled: BTreeMap<String, bool>,
    /// session id -> run
    active_runs: HashMap<String, Run>,
    active_by_routine: HashMap<String, Run>,
    queued_by_routine: HashMap<String, Obj>,
    cooldown_until: HashMap<String, i64>,
    last_schedule_slot: HashMap<String, String>,
    started: bool,
    stopped: bool,
}

fn canonical(value: &Value, out: &mut String) {
    match value {
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let sorted: BTreeMap<&String, &Value> = map.iter().collect();
            out.push('{');
            for (i, (k, v)) in sorted.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push(':');
                canonical(v, out);
            }
            out.push('}');
        }
        other => out.push_str(&other.to_string()),
    }
}

fn derived(subject: &str) -> AppendOptions {
    AppendOptions {
        actor: None,
        subject: Some(subject.to_string()),
        source: Some(Source::Derived),
        simulated: false,
    }
}

fn slot_of(ms: i64) -> String {
    Utc.timestamp_millis_opt(ms)
        .single()
        .map(|d| d.format("%Y-%m-%dT%H:%M").to_string())
        .unwrap_or_default()
}

fn slot_to_ms(slot: &str) -> Option<i64> {
    NaiveDateTime::parse_from_str(&format!("{slot}:00"), "%Y-%m-%dT%H:%M:%S")
        .ok()
        .map(|d| Utc.from_utc_datetime(&d).timestamp_millis())
}

impl Routines {
    pub fn new(options: RoutinesOptions) -> Result<Routines, String> {
        let cfg = (options.cfg)();
        let previous: HashMap<String, Obj> = {
            let projection = options
                .projection
                .lock()
                .map_err(|_| "projection lock poisoned".to_string())?;
            cfg.routines
                .iter()
                .filter_map(|r| get_str(r, "id"))
                .filter_map(|id| {
                    projection
                        .routine(id)
                        .map(|state| (id.to_string(), state.clone()))
                })
                .collect()
        };
        let mut enabled = BTreeMap::new();
        for routine in &cfg.routines {
            let id = get_str(routine, "id").unwrap_or("").to_string();
            let on = previous
                .get(&id)
                .and_then(|s| s.get("enabled"))
                .map(truthy)
                .unwrap_or_else(|| get(routine, "enabled").is_some_and(truthy));
            // Only validate routines that are actually enabled. A disabled
            // routine (e.g. a Cameo-specific sweep on a box that isn't
            // mounted) must never be able to crash Field at boot.
            if on {
                validate_routine(routine, &cfg)?;
            }
            enabled.insert(id, on);
        }
        let mut me = Routines {
            cfg: options.cfg,
            spawner: options.spawner,
            emit: options.emit,
            projection: options.projection,
            now: options.now,
            create_id: options.create_id,
            pending: HashMap::new(),
            enabled,
            active_runs: HashMap::new(),
            active_by_routine: HashMap::new(),
            queued_by_routine: HashMap::new(),
            cooldown_until: HashMap::new(),
            last_schedule_slot: HashMap::new(),
            started: false,
            stopped: false,
        };
        for routine in &cfg.routines {
            let id = get_str(routine, "id").unwrap_or("").to_string();
            let prev = previous.get(&id);
            me.last_schedule_slot.insert(
                id.clone(),
                prev.and_then(|p| p.get("lastScheduleSlot"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            );
            me.cooldown_until.insert(
                id.clone(),
                prev.and_then(|p| finite(p.get("cooldownUntil")))
                    .unwrap_or(0.0) as i64,
            );
            let queued_runs: Vec<Value> = prev
                .and_then(|p| p.get("queuedRuns"))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            for queued in queued_runs {
                let claimed = get(&queued, "claimed").is_some_and(truthy);
                let reason = if claimed {
                    Some("interrupted_after_queue_claim")
                } else if !me.enabled.get(&id).copied().unwrap_or(false) {
                    Some("disabled")
                } else if get_str(&queued, "authorityHash") != Some(&me.queue_authority(routine)) {
                    Some("queued_configuration_changed")
                } else if me.queued_by_routine.contains_key(&id) {
                    Some("queue_full")
                } else {
                    None
                };
                match reason {
                    Some(reason) => {
                        (me.emit)(
                            if claimed {
                                "routine.failed"
                            } else {
                                "routine.skipped"
                            },
                            json!({ "routineId": id, "runId": queued.get("runId"), "reason": reason }),
                            derived(&id),
                        );
                    }
                    None => {
                        let mut entry = queued.as_object().cloned().unwrap_or_default();
                        entry.insert("routineId".into(), json!(id));
                        me.queued_by_routine.insert(id.clone(), entry);
                    }
                }
            }
            let interrupted: Vec<Value> = match prev
                .and_then(|p| get_arr(&Value::Object(p.clone()), "activeRunIds").cloned())
            {
                Some(ids) if !ids.is_empty() => ids,
                _ => prev
                    .and_then(|p| p.get("activeRunId"))
                    .filter(|v| truthy(v))
                    .map(|v| vec![v.clone()])
                    .unwrap_or_default(),
            };
            for run_id in interrupted {
                (me.emit)(
                    "routine.failed",
                    json!({
                        "routineId": id, "runId": run_id,
                        "reason": "Field restarted while this run was active; outcome is interrupted, not completed",
                    }),
                    derived(&id),
                );
            }
        }
        Ok(me)
    }

    fn config(&self) -> RoutineConfig {
        (self.cfg)()
    }

    fn queue_authority(&self, routine: &Value) -> String {
        let cfg = self.config();
        let mut doc = serde_json::Map::new();
        doc.insert("routine".into(), routine.clone());
        doc.insert("defaults".into(), cfg.defaults.clone());
        let refs = [
            ("role", &cfg.roles, "id", get_str(routine, "role")),
            ("agent", &cfg.agents, "role", get_str(routine, "role")),
            (
                "endpoint",
                &cfg.endpoints,
                "id",
                get_str(routine, "endpoint"),
            ),
            (
                "workspace",
                &cfg.workspaces,
                "id",
                get_str(routine, "workspace"),
            ),
        ];
        for (name, list, key, want) in refs {
            if let Some(found) = RoutineConfig::find(list, key, want) {
                doc.insert(name.into(), found.clone());
            }
        }
        let mut text = String::new();
        canonical(&Value::Object(doc), &mut text);
        let digest = Sha256::digest(text.as_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn runs_for(&self, id: &str) -> Vec<Run> {
        let mut runs: Vec<Run> = self
            .active_runs
            .values()
            .filter(|r| r.routine_id == id)
            .cloned()
            .collect();
        runs.sort_by(|a, b| a.run_id.cmp(&b.run_id));
        runs
    }

    fn is_on(&self, id: &str) -> bool {
        self.enabled.get(id).copied().unwrap_or(false)
    }

    fn interrupt_runs(&mut self, routine_id: &str, reason: &str) {
        for run in self.runs_for(routine_id) {
            (self.emit)(
                "routine.failed",
                json!({
                    "routineId": routine_id, "runId": run.run_id,
                    "sessionId": run.session_id, "reason": reason,
                }),
                derived(routine_id),
            );
            self.active_runs.remove(&run.session_id);
            self.spawner.cancel(std::slice::from_ref(&run.session_id));
        }
        self.active_by_routine.remove(routine_id);
    }

    fn fire(
        &mut self,
        routine: &Value,
        reason: &str,
        changed_paths: &[String],
        run_id_override: Option<&str>,
    ) -> Result<Value, String> {
        let id = get_str(routine, "id").unwrap_or("").to_string();
        if self.stopped {
            return Ok(json!({ "admitted": false, "reason": "stopped" }));
        }
        if !self.is_on(&id) {
            return Ok(json!({ "admitted": false, "reason": "disabled" }));
        }
        let cooldown = self.cooldown_until.get(&id).copied().unwrap_or(0);
        let active = !self.runs_for(&id).is_empty();
        let cooling = (self.now)() < cooldown;
        let queued = run_id_override.is_none() && self.queued_by_routine.contains_key(&id);
        let overlap = get_str(routine, "overlap").unwrap_or("skip");
        if active || cooling || queued {
            if overlap == "queue-one" {
                return self.enqueue(routine, reason, changed_paths);
            }
            if overlap == "replace" && active && !cooling && !queued {
                self.interrupt_runs(&id, "replaced");
            } else if !(overlap == "parallel" && active && !cooling && !queued) {
                (self.emit)(
                    "routine.skipped",
                    json!({
                        "routineId": id,
                        "reason": if active { "overlap" } else { "cooldown" },
                        "activeRunId": self.active_by_routine.get(&id).map(|r| r.run_id.clone()),
                    }),
                    derived(&id),
                );
                return Ok(json!({ "admitted": false, "reason": "overlap" }));
            }
        }
        let role = get_str(routine, "role").unwrap_or("");
        let cfg = self.config();
        let Some(agent) = RoutineConfig::find(&cfg.agents, "role", Some(role)).cloned() else {
            let reason = format!("no agent has role \"{role}\"");
            if let Some(run_id) = run_id_override {
                (self.emit)(
                    "routine.failed",
                    json!({ "routineId": id, "runId": run_id, "reason": reason }),
                    derived(&id),
                );
            }
            return Ok(json!({ "admitted": false, "reason": reason }));
        };
        let mut orders = get(routine, "orders").map(js_string).unwrap_or_default();
        if !changed_paths.is_empty() {
            orders.push_str("\n\nFiles that changed since the last run:\n");
            orders.push_str(
                &changed_paths
                    .iter()
                    .take(40)
                    .map(|p| format!("- {p}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        let run_id = run_id_override
            .map(str::to_string)
            .unwrap_or_else(|| format!("{id}:{}", (self.create_id)()));
        let workspace = get(routine, "workspace").cloned().unwrap_or(Value::Null);
        let body = json!({
            "agentId": agent.get("id"),
            "workspaceId": workspace,
            "orders": orders,
            "endpointId": routine.get("endpoint"),
            "thinking": routine.get("thinking"),
            "budgetUsd": routine.get("budget_usd"),
            "target": { "type": "workspace", "id": workspace, "workspaceId": workspace },
        });
        match self.spawner.spawn(body) {
            Ok(session_id) => {
                let started_at = (self.now)();
                let run = Run {
                    run_id: run_id.clone(),
                    routine_id: id.clone(),
                    session_id: session_id.clone(),
                };
                self.active_runs.insert(session_id.clone(), run.clone());
                self.active_by_routine.insert(id.clone(), run);
                self.cooldown_until.insert(id.clone(), 0);
                (self.emit)(
                    "routine.triggered",
                    json!({
                        "routineId": id, "runId": run_id, "sessionIds": [session_id], "reason": reason,
                        "budgetUsd": routine.get("budget_usd").cloned().unwrap_or(Value::Null),
                        "startedAt": started_at,
                    }),
                    derived(&id),
                );
                Ok(json!({
                    "admitted": true, "runId": run_id, "routineId": id,
                    "sessionId": session_id, "startedAt": started_at,
                }))
            }
            Err(message) => {
                (self.emit)(
                    "routine.failed",
                    json!({ "routineId": id, "runId": run_id, "reason": format!("spawn failed: {message}") }),
                    derived(&id),
                );
                Ok(json!({ "admitted": false, "reason": message }))
            }
        }
    }

    fn enqueue(
        &mut self,
        routine: &Value,
        reason: &str,
        paths: &[String],
    ) -> Result<Value, String> {
        let id = get_str(routine, "id").unwrap_or("").to_string();
        if self.queued_by_routine.contains_key(&id) {
            (self.emit)(
                "routine.skipped",
                json!({ "routineId": id, "reason": "queue_full" }),
                derived(&id),
            );
            return Ok(json!({ "admitted": false, "reason": "queue_full" }));
        }
        let queued = json!({
            "runId": format!("{id}:{}", (self.create_id)()), "routineId": id,
            "paths": paths.iter().take(40).collect::<Vec<_>>(),
            "queuedAt": (self.now)(), "reason": reason,
            "authorityHash": self.queue_authority(routine),
        });
        // Persist before remembering: a queue entry the log does not hold is
        // a phantom the next process would never see.
        if (self.emit)("routine.queued", queued.clone(), derived(&id)).is_none() {
            return Err("routine.queued could not be recorded".into());
        }
        self.queued_by_routine
            .insert(id.clone(), queued.as_object().cloned().unwrap_or_default());
        self.pump(routine)?;
        let mut result = queued;
        result["admitted"] = json!(true);
        result["queued"] = json!(true);
        Ok(result)
    }

    fn pump(&mut self, routine: &Value) -> Result<(), String> {
        let id = get_str(routine, "id").unwrap_or("").to_string();
        let Some(queued) = self.queued_by_routine.get(&id).cloned() else {
            return Ok(());
        };
        let claimed = queued.get("claimed").is_some_and(truthy);
        if self.stopped || claimed || self.active_by_routine.contains_key(&id) || !self.is_on(&id) {
            return Ok(());
        }
        if queued.get("authorityHash").and_then(Value::as_str)
            != Some(&self.queue_authority(routine))
        {
            (self.emit)(
                "routine.skipped",
                json!({ "routineId": id, "runId": queued.get("runId"), "reason": "queued_configuration_changed" }),
                derived(&id),
            );
            self.queued_by_routine.remove(&id);
            return Ok(());
        }
        let wait = self.cooldown_until.get(&id).copied().unwrap_or(0) - (self.now)();
        if wait > 0 {
            // The next poll after the cooldown ends picks this up.
            return Ok(());
        }
        let run_id = queued.get("runId").map(js_string).unwrap_or_default();
        (self.emit)(
            "routine.queue_claimed",
            json!({ "routineId": id, "runId": run_id }),
            derived(&id),
        );
        if let Some(entry) = self.queued_by_routine.get_mut(&id) {
            entry.insert("claimed".into(), json!(true));
        }
        let paths: Vec<String> = queued
            .get("paths")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(js_string).collect())
            .unwrap_or_default();
        let reason = format!(
            "queued after {}",
            queued
                .get("reason")
                .map(js_string)
                .unwrap_or_else(|| "run".into())
        );
        let outcome = self.fire(routine, &reason, &paths, Some(&run_id));
        self.queued_by_routine.remove(&id);
        outcome.map(|_| ())
    }

    /// Schedule triggers: checked against the real clock. Missed slots are
    /// skipped; the watermark is persisted before spawning so a restart or a
    /// backwards clock cannot duplicate a consequential scheduled run.
    fn clock_tick(&mut self) {
        let current = (self.now)();
        let slot = slot_of(current);
        for r in self.config().routines {
            let id = get_str(&r, "id").unwrap_or("").to_string();
            let trigger = get(&r, "trigger").cloned().unwrap_or(Value::Null);
            if !self.is_on(&id) || get_str(&trigger, "kind") != Some("schedule") {
                continue;
            }
            let cron = get_str(&trigger, "cron").unwrap_or("").to_string();
            let matches = cron_matches(&cron, current).unwrap_or(false);
            let last = self
                .last_schedule_slot
                .get(&id)
                .cloned()
                .unwrap_or_default();
            if !matches || last >= slot {
                continue;
            }
            (self.emit)(
                "routine.schedule_claimed",
                json!({ "routineId": id, "slot": slot }),
                derived(&id),
            );
            self.last_schedule_slot.insert(id.clone(), slot.clone());
            let _ = self.fire(&r, &format!("cron {cron} UTC"), &[], None);
        }
    }

    /// Runs everything that has come due: debounced filesystem triggers,
    /// cooled-down queued runs and the schedule clock. Call every few seconds.
    pub fn poll(&mut self) {
        if self.stopped {
            return;
        }
        let now = (self.now)();
        let due: Vec<(String, Vec<String>)> = self
            .pending
            .iter()
            .filter(|(_, p)| p.due_at <= now)
            .map(|(id, p)| (id.clone(), p.paths.clone()))
            .collect();
        let cfg = self.config();
        for (id, paths) in due {
            self.pending.remove(&id);
            if let Some(routine) = cfg.routine(&id).cloned() {
                if self.is_on(&id) {
                    let _ = self.fire(
                        &routine,
                        &format!("{} file(s) changed", paths.len()),
                        &paths,
                        None,
                    );
                }
            }
        }
        for routine in &cfg.routines {
            let _ = self.pump(routine);
        }
        self.clock_tick();
    }

    /// Feeds one log event: run settlement and filesystem triggers.
    pub fn on_event(&mut self, evt: &Event) {
        let session_id = get(&evt.data, "sessionId").map(js_string);
        let run = session_id
            .as_ref()
            .and_then(|sid| self.active_runs.get(sid).cloned());
        if let Some(run) = &run {
            if matches!(evt.kind.as_str(), "session.turn_complete" | "session.ended") {
                let cfg = self.config();
                let routine = cfg.routine(&run.routine_id).cloned();
                let cooldown_seconds = routine
                    .as_ref()
                    .and_then(|r| finite(get(r, "cooldown_seconds")))
                    .unwrap_or(0.0);
                let until = (self.now)() + (cooldown_seconds * 1000.0) as i64;
                let failed = if evt.kind == "session.ended" {
                    matches!(
                        get_str(&evt.data, "reason"),
                        Some("error") | Some("cancelled")
                    )
                } else {
                    get(&evt.data, "isError") == Some(&Value::Bool(true))
                };
                let reason = if failed {
                    get(&evt.data, "error")
                        .or(get(&evt.data, "reason"))
                        .map(js_string)
                        .unwrap_or_else(|| "run failed".into())
                } else {
                    "run completed".into()
                };
                (self.emit)(
                    if failed {
                        "routine.failed"
                    } else {
                        "routine.completed"
                    },
                    json!({
                        "routineId": run.routine_id, "runId": run.run_id, "sessionId": run.session_id,
                        "cooldownUntil": until, "reason": reason,
                    }),
                    derived(&run.routine_id),
                );
                self.active_runs.remove(&run.session_id);
                let remaining = self.runs_for(&run.routine_id);
                match remaining.last() {
                    Some(last) => {
                        self.active_by_routine
                            .insert(run.routine_id.clone(), last.clone());
                    }
                    None => {
                        self.active_by_routine.remove(&run.routine_id);
                    }
                }
                self.cooldown_until.insert(run.routine_id.clone(), until);
                if self.queued_by_routine.contains_key(&run.routine_id) {
                    if let Some(routine) = routine {
                        let _ = self.pump(&routine);
                    }
                }
                return;
            }
        }
        if evt.kind != "fs.changed" || evt.source == Source::Synthetic {
            return;
        }
        let changed_workspace = get(&evt.data, "workspaceId").map(js_string);
        let path = get(&evt.data, "path").map(js_string).unwrap_or_default();
        let now = (self.now)();
        for r in self.config().routines {
            let id = get_str(&r, "id").unwrap_or("").to_string();
            if !self.is_on(&id) {
                continue;
            }
            let t = get(&r, "trigger").cloned().unwrap_or(Value::Null);
            if get_str(&t, "kind") != Some("fs_change") {
                continue;
            }
            if let Some(ws) = get(&t, "workspace").map(js_string) {
                if Some(ws) != changed_workspace {
                    continue;
                }
            }
            let patterns: Vec<String> = get_arr(&t, "paths")
                .map(|a| a.iter().map(js_string).collect())
                .unwrap_or_else(|| vec!["**".to_string()]);
            if !patterns.iter().any(|p| glob_to_regex(p).is_match(&path)) {
                continue;
            }
            if run.as_ref().is_some_and(|run| run.routine_id == id) {
                (self.emit)(
                    "routine.skipped",
                    json!({
                        "routineId": id, "runId": run.as_ref().map(|r| r.run_id.clone()),
                        "reason": "self_trigger", "path": path,
                    }),
                    derived(&id),
                );
                continue;
            }
            let debounce = finite(get(&t, "debounce_seconds")).unwrap_or(60.0);
            let entry = self.pending.entry(id).or_insert(Pending {
                paths: Vec::new(),
                due_at: now,
            });
            if !entry.paths.contains(&path) {
                entry.paths.push(path.clone());
            }
            entry.due_at = now + (debounce * 1000.0) as i64;
        }
    }

    pub fn start(&mut self) -> bool {
        if self.started || self.stopped {
            return false;
        }
        self.started = true;
        for routine in self.config().routines {
            let _ = self.pump(&routine);
        }
        self.clock_tick();
        true
    }

    pub fn set_enabled(&mut self, id: &str, on: bool, confirm_risk: bool) -> Result<bool, String> {
        if !self.enabled.contains_key(id) {
            return Ok(false);
        }
        if on && !confirm_risk {
            return Err("enabling a routine requires explicit operator confirmation".into());
        }
        (self.emit)(
            "routine.enabled",
            json!({ "routineId": id, "enabled": on }),
            AppendOptions {
                actor: None,
                subject: Some(id.to_string()),
                source: None,
                simulated: false,
            },
        );
        self.enabled.insert(id.to_string(), on);
        if !on {
            self.pending.remove(id);
            if let Some(queued) = self.queued_by_routine.remove(id) {
                (self.emit)(
                    "routine.skipped",
                    json!({ "routineId": id, "runId": queued.get("runId"), "reason": "disabled" }),
                    derived(id),
                );
            }
        }
        Ok(true)
    }

    /// An operator run: admitted even while the routine is disabled, without
    /// enabling it.
    pub fn run(&mut self, id: &str) -> Result<Value, String> {
        let Some(routine) = self.config().routine(id).cloned() else {
            return Ok(json!({ "admitted": false, "reason": "unknown routine" }));
        };
        let was_enabled = self.is_on(id);
        if !was_enabled {
            self.enabled.insert(id.to_string(), true);
        }
        let result = self.fire(&routine, "operator", &[], None);
        if !was_enabled {
            self.enabled.insert(id.to_string(), false);
        }
        result
    }

    pub fn is_enabled(&self, id: &str) -> bool {
        self.is_on(id)
    }

    pub fn states(&self) -> Value {
        Value::Object(
            self.enabled
                .iter()
                .map(|(k, v)| (k.clone(), json!(v)))
                .collect(),
        )
    }

    pub fn details(&self) -> Value {
        let cfg = self.config();
        let states: HashMap<String, Obj> = self
            .projection
            .lock()
            .map(|p| {
                cfg.routines
                    .iter()
                    .filter_map(|r| get_str(r, "id"))
                    .filter_map(|id| p.routine(id).map(|s| (id.to_string(), s.clone())))
                    .collect()
            })
            .unwrap_or_default();
        let now = (self.now)();
        let mut out = serde_json::Map::new();
        for routine in &cfg.routines {
            let id = get_str(routine, "id").unwrap_or("").to_string();
            let state = states.get(&id).cloned().unwrap_or_default();
            let trigger = get(routine, "trigger").cloned().unwrap_or(Value::Null);
            let mut next_run_at = Value::Null;
            if self.is_on(&id) && get_str(&trigger, "kind") == Some("schedule") {
                let watermark = state
                    .get("lastScheduleSlot")
                    .and_then(Value::as_str)
                    .and_then(slot_to_ms)
                    .unwrap_or(0);
                next_run_at =
                    next_cron_run(get_str(&trigger, "cron").unwrap_or(""), now.max(watermark))
                        .ok()
                        .flatten()
                        .map(Value::String)
                        .unwrap_or(Value::Null);
            }
            let runs = self.runs_for(&id);
            let history: Vec<Value> = state
                .get("history")
                .and_then(Value::as_array)
                .map(|h| h.iter().rev().take(50).rev().cloned().collect())
                .unwrap_or_default();
            let field = |k: &str| state.get(k).cloned().unwrap_or(Value::Null);
            out.insert(
                id.clone(),
                json!({
                    "enabled": self.is_on(&id), "timezone": "UTC",
                    "activeRunId": runs.last().map(|r| r.run_id.clone()),
                    "activeRunIds": runs.iter().map(|r| r.run_id.clone()).collect::<Vec<_>>(),
                    "queuedRuns": self.queued_by_routine.get(&id).map(|q| vec![Value::Object(q.clone())]).unwrap_or_default(),
                    "lastRunAt": field("lastRunAt"), "lastOutcome": field("lastOutcome"),
                    "nextRunAt": next_run_at, "currentOwner": field("currentOwner"),
                    "history": history,
                }),
            );
        }
        Value::Object(out)
    }

    pub fn stop(&mut self) {
        self.stopped = true;
        self.pending.clear();
        // Unclaimed work remains durable for the next process. Only an explicit
        // disable cancels it; a claimed run is reconciled as interrupted on restart.
        self.queued_by_routine.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::projection::FieldConfig;
    use super::*;
    use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
    use std::sync::RwLock;

    fn ms(iso: &str) -> i64 {
        DateTime::parse_from_rfc3339(iso)
            .unwrap()
            .timestamp_millis()
    }

    #[test]
    fn cron_semantics_match_the_node_port() {
        assert!(cron_matches("0 5 * * *", ms("2024-01-01T00:00:00-05:00")).unwrap());
        assert!(cron_matches("0 0 1 * 0", ms("2024-01-01T00:00:00Z")).unwrap());
        assert!(cron_matches("0 0 2 * 1", ms("2024-01-01T00:00:00Z")).unwrap());
        assert!(validate_cron("* * * *")
            .unwrap_err()
            .contains("five fields"));
        assert!(validate_cron("*/0 * * * *")
            .unwrap_err()
            .contains("invalid cron step"));
        assert!(validate_cron("60 * * * *")
            .unwrap_err()
            .contains("outside 0-59"));
        assert!(validate_cron("* * 8-2 * *")
            .unwrap_err()
            .contains("outside 1-31"));
        assert!(cron_matches("5/15 * * * *", ms("2024-01-01T00:20:00Z")).unwrap());
        assert_eq!(
            next_cron_run("0 0 29 2 *", ms("2025-03-01T00:00:00Z")).unwrap(),
            Some("2028-02-29T00:00:00.000Z".into())
        );
        assert_eq!(
            next_cron_run("0 0 31 2 *", ms("2025-03-01T00:00:00Z")).unwrap(),
            None
        );
        assert_eq!(
            next_cron_run("0 5 * * *", ms("2024-03-10T05:00:00Z")).unwrap(),
            Some("2024-03-11T05:00:00.000Z".into())
        );
    }

    fn routine() -> Value {
        json!({
            "id": "watch-source", "name": "Watch source", "enabled": true,
            "role": "builder", "endpoint": "local", "workspace": "cameo", "thinking": "medium",
            "budget_usd": 0.5, "orders": "Inspect changed source files.",
            "trigger": { "kind": "fs_change", "workspace": "cameo", "paths": ["src/**"], "debounce_seconds": 1 },
            "completion": { "kind": "report" },
        })
    }

    fn config(routines: Vec<Value>) -> RoutineConfig {
        RoutineConfig {
            defaults: json!({}),
            routines,
            roles: vec![json!({ "id": "builder" })],
            agents: vec![json!({ "id": "builder-1", "role": "builder" })],
            endpoints: vec![
                json!({ "id": "local", "name": "Local", "kind": "openai-compatible", "model": "test" }),
            ],
            workspaces: vec![
                json!({ "id": "cameo", "name": "Cameo", "path": ".", "mounted": true }),
            ],
        }
    }

    fn with(base: &Value, patch: Value) -> Value {
        let mut v = base.clone();
        for (k, val) in patch.as_object().unwrap() {
            v[k] = val.clone();
        }
        v
    }

    #[test]
    fn validation_matches_the_node_port() {
        let cfg = config(vec![routine()]);
        assert!(validate_routine(&routine(), &cfg).is_ok());
        let err = |patch: Value| validate_routine(&with(&routine(), patch), &cfg).unwrap_err();
        assert!(err(json!({ "role": "missing" })).contains("unknown role"));
        assert!(err(json!({ "endpoint": "missing" })).contains("unknown endpoint"));
        assert!(err(json!({ "workspace": "missing" })).contains("unavailable workspace"));
        assert!(err(json!({ "trigger": { "kind": "manual" } })).contains("unsupported trigger"));
        assert!(err(json!({ "budget_usd": 0 })).contains("invalid dollar budget"));
        let queue = with(
            &routine(),
            json!({ "id": "queue-one", "overlap": "queue-one", "max_queued_runs": 1, "cooldown_seconds": 2 }),
        );
        assert!(validate_routine(&queue, &config(vec![queue.clone()])).is_ok());
    }

    /// The Node test harness: a recording spawner, a fake clock, an emit that
    /// folds into a projection and records every event.
    struct Fixture {
        cfg: Arc<RwLock<RoutineConfig>>,
        projection: Arc<Mutex<Projection>>,
        events: Arc<Mutex<Vec<Event>>>,
        clock: Arc<AtomicI64>,
        spawns: Arc<Mutex<Vec<Value>>>,
        cancelled: Arc<Mutex<Vec<String>>>,
        fail_spawn: Arc<Mutex<bool>>,
        fail_emit: Arc<Mutex<Option<String>>>,
        ids: Arc<AtomicUsize>,
    }

    struct Recorder {
        spawns: Arc<Mutex<Vec<Value>>>,
        cancelled: Arc<Mutex<Vec<String>>>,
        fail: Arc<Mutex<bool>>,
        prefix: String,
    }

    impl RunSpawner for Recorder {
        fn spawn(&self, body: Value) -> Result<String, String> {
            if *self.fail.lock().unwrap() {
                return Err("budget exhausted".into());
            }
            let mut spawns = self.spawns.lock().unwrap();
            spawns.push(body);
            Ok(format!("{}-{}", self.prefix, spawns.len()))
        }
        fn cancel(&self, ids: &[String]) {
            self.cancelled.lock().unwrap().extend(ids.iter().cloned());
        }
    }

    impl Fixture {
        fn new(routines: Vec<Value>, start_ms: i64) -> Fixture {
            let cfg = config(routines);
            let fc: FieldConfig = serde_json::from_value(json!({
                "workspaces": cfg.workspaces, "endpoints": cfg.endpoints, "websites": [], "routines": cfg.routines,
            }))
            .unwrap();
            Fixture {
                cfg: Arc::new(RwLock::new(cfg)),
                projection: Arc::new(Mutex::new(Projection::new(fc))),
                events: Arc::new(Mutex::new(Vec::new())),
                clock: Arc::new(AtomicI64::new(start_ms)),
                spawns: Arc::new(Mutex::new(Vec::new())),
                cancelled: Arc::new(Mutex::new(Vec::new())),
                fail_spawn: Arc::new(Mutex::new(false)),
                fail_emit: Arc::new(Mutex::new(None)),
                ids: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn now(&self) -> i64 {
            self.clock.load(Ordering::SeqCst)
        }

        /// Appends an event the way the server's emit does: fold into the
        /// projection and remember it.
        fn emit(&self, kind: &str, data: Value, options: AppendOptions) -> Option<Event> {
            if self.fail_emit.lock().unwrap().as_deref() == Some(kind) {
                return None;
            }
            let mut events = self.events.lock().unwrap();
            let event = Event {
                seq: events.len() as u64 + 1,
                ts: self.now(),
                kind: kind.to_string(),
                actor: options.actor,
                subject: options.subject,
                source: options.source.unwrap_or(Source::Observed),
                data,
            };
            self.projection.lock().unwrap().apply(&event);
            events.push(event.clone());
            Some(event)
        }

        fn boot(&self, prefix: &str) -> Routines {
            let events = Arc::clone(&self.events);
            let projection = Arc::clone(&self.projection);
            let clock = Arc::clone(&self.clock);
            let fail_emit = Arc::clone(&self.fail_emit);
            let emitter = Fixture {
                cfg: Arc::clone(&self.cfg),
                projection,
                events,
                clock,
                spawns: Arc::clone(&self.spawns),
                cancelled: Arc::clone(&self.cancelled),
                fail_spawn: Arc::clone(&self.fail_spawn),
                fail_emit,
                ids: Arc::clone(&self.ids),
            };
            let emit: Emit = Arc::new(move |kind, data, options| emitter.emit(kind, data, options));
            let clock = Arc::clone(&self.clock);
            let ids = Arc::clone(&self.ids);
            let cfg_source = Arc::clone(&self.cfg);
            Routines::new(RoutinesOptions {
                cfg: Arc::new(move || cfg_source.read().unwrap().clone()),
                spawner: Arc::new(Recorder {
                    spawns: Arc::clone(&self.spawns),
                    cancelled: Arc::clone(&self.cancelled),
                    fail: Arc::clone(&self.fail_spawn),
                    prefix: prefix.to_string(),
                }),
                emit,
                projection: Arc::clone(&self.projection),
                now: Arc::new(move || clock.load(Ordering::SeqCst)),
                create_id: Arc::new(move || {
                    format!("run{}", ids.fetch_add(1, Ordering::SeqCst) + 1)
                }),
            })
            .unwrap()
        }

        fn advance(&self, controller: &mut Routines, delta: i64) {
            self.clock.fetch_add(delta, Ordering::SeqCst);
            controller.poll();
        }

        /// A log event from outside the controller (the harness, the watcher).
        fn publish(&self, controller: &mut Routines, kind: &str, data: Value) {
            let evt = self.emit(kind, data, AppendOptions::default()).unwrap();
            controller.on_event(&evt);
        }

        fn has(&self, kind: &str, reason: &str) -> bool {
            self.events
                .lock()
                .unwrap()
                .iter()
                .any(|e| e.kind == kind && get_str(&e.data, "reason") == Some(reason))
        }

        fn count(&self, kind: &str) -> usize {
            self.events
                .lock()
                .unwrap()
                .iter()
                .filter(|e| e.kind == kind)
                .count()
        }

        fn starts(&self) -> usize {
            self.spawns.lock().unwrap().len()
        }

        fn state(&self, id: &str) -> Obj {
            self.projection
                .lock()
                .unwrap()
                .routine(id)
                .cloned()
                .unwrap_or_default()
        }
    }

    #[test]
    fn fs_change_routine_debounces_skips_self_triggers_and_respects_overlap() {
        let f = Fixture::new(vec![routine()], ms("2024-01-01T00:00:00Z"));
        f.emit(
            "routine.enabled",
            json!({ "routineId": "watch-source", "enabled": false }),
            AppendOptions::default(),
        );
        let mut c = f.boot("session");
        assert!(c.start());
        assert!(!c.start(), "the scheduler clock starts once");
        assert!(
            !c.is_enabled("watch-source"),
            "durable replayed state overrides the Git default"
        );
        assert!(c
            .set_enabled("watch-source", true, false)
            .unwrap_err()
            .contains("explicit operator confirmation"));
        assert!(c.set_enabled("watch-source", true, true).unwrap());

        f.publish(
            &mut c,
            "fs.changed",
            json!({ "workspaceId": "cameo", "path": "src/changed.js" }),
        );
        c.set_enabled("watch-source", false, false).unwrap();
        f.advance(&mut c, 1000);
        assert_eq!(f.starts(), 0, "disabling cancels pending debounced work");

        c.set_enabled("watch-source", true, true).unwrap();
        let first = c.run("watch-source").unwrap();
        assert_eq!(first["admitted"], true);
        assert_eq!(
            f.spawns.lock().unwrap()[0]["budgetUsd"],
            0.5,
            "routine budget reaches the session registry"
        );
        assert_eq!(c.run("watch-source").unwrap()["reason"], "overlap");
        assert!(f.has("routine.skipped", "overlap"));

        let sid = first["sessionId"].as_str().unwrap().to_string();
        f.publish(
            &mut c,
            "fs.changed",
            json!({ "sessionId": sid, "workspaceId": "cameo", "path": "src/self.js" }),
        );
        f.advance(&mut c, 1000);
        assert_eq!(
            f.starts(),
            1,
            "a routine cannot trigger itself from its own filesystem writes"
        );
        assert!(f.has("routine.skipped", "self_trigger"));

        f.publish(
            &mut c,
            "session.turn_complete",
            json!({ "sessionId": sid, "isError": false }),
        );
        assert_eq!(c.details()["watch-source"]["lastOutcome"], "completed");
        assert_eq!(
            c.run("watch-source").unwrap()["admitted"],
            true,
            "completion releases the overlap lock"
        );
        c.stop();
        assert_eq!(c.run("watch-source").unwrap()["reason"], "stopped");
    }

    #[test]
    fn schedule_watermark_survives_restart_and_interrupted_runs_fail() {
        let scheduled = with(
            &routine(),
            json!({ "id": "scheduled", "trigger": { "kind": "schedule", "cron": "* * * * *" } }),
        );
        let f = Fixture::new(vec![scheduled], ms("2024-01-01T00:00:00Z"));
        let mut before = f.boot("scheduled");
        before.start();
        assert_eq!(f.starts(), 1);
        before.stop();
        let mut after = f.boot("scheduled");
        after.start();
        assert_eq!(
            f.starts(),
            1,
            "persisted schedule watermark prevents same-minute duplicate after restart"
        );
        let details = after.details();
        assert_eq!(
            details["scheduled"]["lastOutcome"], "failed",
            "interrupted run is never left running"
        );
        assert_eq!(
            details["scheduled"]["nextRunAt"],
            "2024-01-01T00:01:00.000Z"
        );
        f.advance(&mut after, 60_000);
        assert_eq!(f.starts(), 2, "the next minute fires once");
    }

    #[test]
    fn queue_one_holds_a_single_trigger_through_cooldown() {
        let q = with(
            &routine(),
            json!({ "id": "queue-one", "overlap": "queue-one", "max_queued_runs": 1, "cooldown_seconds": 2 }),
        );
        let f = Fixture::new(vec![q], ms("2024-01-01T00:00:00Z"));
        let mut c = f.boot("queue-session");
        let active = c.run("queue-one").unwrap();
        assert_eq!(active["admitted"], true);
        assert_eq!(
            c.run("queue-one").unwrap()["queued"],
            true,
            "queue-one retains one trigger during overlap"
        );
        assert_eq!(
            c.run("queue-one").unwrap()["reason"],
            "queue_full",
            "second queued trigger is bounded"
        );
        f.publish(
            &mut c,
            "session.turn_complete",
            json!({ "sessionId": active["sessionId"], "isError": false }),
        );
        f.advance(&mut c, 1999);
        assert_eq!(f.starts(), 1, "cooldown holds the queued run");
        f.advance(&mut c, 1);
        assert_eq!(f.starts(), 2, "queued run dispatches after cooldown");
        assert_eq!(f.count("routine.triggered"), 2);
        let queued_active = f
            .events
            .lock()
            .unwrap()
            .iter()
            .rev()
            .find(|e| e.kind == "routine.triggered")
            .map(|e| e.data["sessionIds"][0].as_str().unwrap().to_string())
            .unwrap();
        c.run("queue-one").unwrap();
        c.set_enabled("queue-one", false, false).unwrap();
        assert!(
            f.has("routine.skipped", "disabled"),
            "disable drops queued work truthfully"
        );
        f.publish(
            &mut c,
            "session.ended",
            json!({ "sessionId": queued_active, "reason": "error", "error": "budget exhausted" }),
        );
        assert_eq!(f.state("queue-one")["lastOutcome"], "failed");
    }

    #[test]
    fn replace_cancels_the_active_run_and_parallel_admits_both() {
        let r = with(&routine(), json!({ "id": "replace", "overlap": "replace" }));
        let f = Fixture::new(vec![r], ms("2024-01-01T00:00:00Z"));
        let mut c = f.boot("replace-session");
        let first = c.run("replace").unwrap();
        let second = c.run("replace").unwrap();
        assert_eq!(
            second["admitted"], true,
            "replace starts a new run while one is active"
        );
        assert_eq!(f.starts(), 2);
        assert_eq!(
            *f.cancelled.lock().unwrap(),
            vec![first["sessionId"].as_str().unwrap().to_string()]
        );
        assert!(f.has("routine.failed", "replaced"));
        assert_eq!(f.state("replace")["activeRunId"], second["runId"]);

        let p = with(
            &routine(),
            json!({ "id": "parallel", "overlap": "parallel" }),
        );
        let f = Fixture::new(vec![p], ms("2024-01-01T00:00:00Z"));
        let mut c = f.boot("parallel-session");
        let first = c.run("parallel").unwrap();
        let second = c.run("parallel").unwrap();
        assert!(
            first["admitted"] == true && second["admitted"] == true,
            "parallel admits concurrent runs"
        );
        assert_eq!(f.starts(), 2);
        let mut ids: Vec<String> = c.details()["parallel"]["activeRunIds"]
            .as_array()
            .unwrap()
            .iter()
            .map(js_string)
            .collect();
        ids.sort();
        let mut want = vec![js_string(&first["runId"]), js_string(&second["runId"])];
        want.sort();
        assert_eq!(ids, want);
        f.publish(
            &mut c,
            "session.turn_complete",
            json!({ "sessionId": first["sessionId"], "isError": false }),
        );
        assert_eq!(f.state("parallel")["activeRunId"], second["runId"]);
        assert_eq!(
            c.details()["parallel"]["activeRunIds"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    fn recovery_routine() -> Value {
        json!({ "id": "r", "enabled": true, "role": "builder", "endpoint": "local", "workspace": "cameo",
            "orders": "Review", "budget_usd": 1, "overlap": "queue-one", "cooldown_seconds": 2,
            "trigger": { "kind": "fs_change", "paths": ["src/**"] }, "completion": { "kind": "report" } })
    }

    fn settle(f: &Fixture, c: &mut Routines, run: &Value) {
        f.publish(
            c,
            "session.turn_complete",
            json!({ "sessionId": run["sessionId"] }),
        );
    }

    #[test]
    fn queued_work_resumes_once_after_restart_and_respects_persisted_cooldown() {
        let f = Fixture::new(vec![recovery_routine()], 1_700_000_000_000);
        let mut c = f.boot("s");
        let active = c.run("r").unwrap();
        settle(&f, &mut c, &active);
        assert_eq!(
            c.run("r").unwrap()["queued"],
            true,
            "enqueue during idle cooldown schedules a pump"
        );
        assert_eq!(c.run("r").unwrap()["reason"], "queue_full");
        c.stop();
        let mut c = f.boot("s");
        c.start();
        assert_eq!(f.starts(), 1, "restart respects persisted cooldown");
        f.advance(&mut c, 1999);
        assert_eq!(f.starts(), 1);
        f.advance(&mut c, 1);
        assert_eq!(f.starts(), 2, "unclaimed queued work resumes once");
        assert_eq!(f.state("r")["queuedRuns"].as_array().unwrap().len(), 0);
        let events = f.events.lock().unwrap();
        let claim = events
            .iter()
            .position(|e| e.kind == "routine.queue_claimed")
            .unwrap();
        let triggered = events
            .iter()
            .enumerate()
            .position(|(i, e)| i > claim && e.kind == "routine.triggered")
            .unwrap();
        assert!(triggered > claim, "durable claim precedes dispatch result");
        drop(events);
        c.stop();
        f.advance(&mut c, 10_000);
        assert_eq!(f.starts(), 2);
        assert_eq!(c.run("r").unwrap()["reason"], "stopped");
    }

    #[test]
    fn failed_persistence_creates_no_phantom_queue_and_disable_cancels_dispatch() {
        let f = Fixture::new(vec![recovery_routine()], 1_700_000_000_000);
        let mut c = f.boot("s");
        let active = c.run("r").unwrap();
        *f.fail_emit.lock().unwrap() = Some("routine.queued".into());
        assert!(c.run("r").is_err());
        assert_eq!(
            c.details()["r"]["queuedRuns"].as_array().unwrap().len(),
            0,
            "failed persistence creates no phantom queue"
        );
        *f.fail_emit.lock().unwrap() = None;
        assert_eq!(c.run("r").unwrap()["queued"], true);
        settle(&f, &mut c, &active);
        assert_eq!(
            c.run("r").unwrap()["reason"],
            "queue_full",
            "waiting cooldown retains the single queue slot"
        );
        c.set_enabled("r", false, false).unwrap();
        f.advance(&mut c, 2000);
        assert_eq!(f.starts(), 1, "disable cancels cooldown dispatch");
    }

    #[test]
    fn a_claimed_queue_entry_is_never_replayed_after_a_crash() {
        let f = Fixture::new(vec![recovery_routine()], 1_700_000_000_000);
        let mut c = f.boot("s");
        let active = c.run("r").unwrap();
        c.run("r").unwrap();
        settle(&f, &mut c, &active);
        let queued = f.state("r")["queuedRuns"][0].clone();
        f.emit(
            "routine.queue_claimed",
            json!({ "routineId": "r", "runId": queued["runId"] }),
            AppendOptions::default(),
        );
        c.stop();
        let mut c = f.boot("s");
        c.start();
        f.advance(&mut c, 2000);
        assert_eq!(
            f.starts(),
            1,
            "crash after claim never blindly replays consequential work"
        );
        assert!(f.has("routine.failed", "interrupted_after_queue_claim"));
        assert_eq!(f.state("r")["queuedRuns"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn queued_work_is_dropped_when_configuration_changes() {
        // Across a restart.
        let f = Fixture::new(vec![recovery_routine()], 1_700_000_000_000);
        let mut c = f.boot("s");
        let active = c.run("r").unwrap();
        c.run("r").unwrap();
        settle(&f, &mut c, &active);
        c.stop();
        f.cfg.write().unwrap().routines[0]["orders"] = json!("Different authority");
        let mut c = f.boot("s");
        c.start();
        f.advance(&mut c, 2000);
        assert_eq!(
            f.starts(),
            1,
            "restart cannot execute queued work under changed configuration"
        );
        assert!(f.has("routine.skipped", "queued_configuration_changed"));

        // During a cooldown, live.
        let f = Fixture::new(vec![recovery_routine()], 1_700_000_000_000);
        let mut c = f.boot("s");
        let active = c.run("r").unwrap();
        c.run("r").unwrap();
        settle(&f, &mut c, &active);
        f.cfg.write().unwrap().routines[0]["budget_usd"] = json!(20);
        f.advance(&mut c, 2000);
        assert_eq!(
            f.starts(),
            1,
            "configuration changes during cooldown invalidate queued authority"
        );
        assert_eq!(f.state("r")["queuedRuns"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn failed_admission_settles_the_claim_and_is_not_retried() {
        let f = Fixture::new(vec![recovery_routine()], 1_700_000_000_000);
        let mut c = f.boot("s");
        let active = c.run("r").unwrap();
        c.run("r").unwrap();
        settle(&f, &mut c, &active);
        *f.fail_spawn.lock().unwrap() = true;
        f.advance(&mut c, 2000);
        assert_eq!(
            f.state("r")["queuedRuns"].as_array().unwrap().len(),
            0,
            "failed admission settles the claimed queue"
        );
        assert_eq!(f.state("r")["lastOutcome"], "failed");
        *f.fail_spawn.lock().unwrap() = false;
        c.stop();
        let mut restarted = f.boot("s");
        restarted.start();
        f.advance(&mut restarted, 5000);
        assert_eq!(
            f.starts(),
            1,
            "failed admission is not silently retried after restart"
        );
    }
}
