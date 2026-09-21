//! The registry owns every live harness session and turns operator commands
//! into real harness actions. It is the only place allowed to start or stop
//! a session. Port of `field/server/src/harness/registry.js`.
//!
//! Every adapter event arrives through one channel and is folded by one
//! task, so the registry lock is never re-entered from inside an emit, and
//! the order of events is the order the harness produced them.

use super::adapter::{Adapter, EventSink};
use super::budget_ledger::BudgetLedger;
use super::config::{compose_prompt, FieldSettings};
use super::eventlog::{AppendOptions, Event, Source};
use super::js::{get, get_arr, get_str, js_string, number, or_null, truthy};
use super::knossos_session::{KnossosOptions, KnossosSession};
use super::policy::{evaluate_permission_policy, parse_campaign_report, PolicyInput};
use super::stores::KeyStore;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

const EFFORT_ORDER: [&str; 5] = ["low", "medium", "high", "xhigh", "max"];
const PERMISSION_TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// Append to the log and fan out; returns the stored event when it was.
pub type Emit = Arc<dyn Fn(&str, Value, AppendOptions) -> Option<Event> + Send + Sync>;
/// The campaign view for an id: `budgetUsd`, `costUsd`, `budgetExhausted`, `concurrency`.
pub type CampaignPolicy = Arc<dyn Fn(&str) -> Option<Value> + Send + Sync>;
pub type RegisterSecret = Arc<dyn Fn(&str) + Send + Sync>;
pub type ReportHandler = Arc<dyn Fn(Value) -> Result<(), String> + Send + Sync>;
/// Builds the adapter for a session; tests substitute a fake.
pub type AdapterBuilder = dyn Fn(KnossosOptions, EventSink) -> Arc<dyn Adapter> + Send + Sync;
pub type AdapterFactory = Arc<AdapterBuilder>;

/// Session-scoped internal capabilities, minted per session and revoked at exit.
#[derive(Clone)]
pub struct Capabilities {
    pub mint: Arc<CapabilityMint>,
    pub revoke: Arc<CapabilityRevoke>,
}

/// Mints a session-scoped capability token; `None` when minting is unavailable.
pub type CapabilityMint = dyn Fn(&str) -> Option<String> + Send + Sync;
/// Revokes every capability minted for a session id.
pub type CapabilityRevoke = dyn Fn(&str) + Send + Sync;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub decision: String,
    pub message: Option<String>,
}

/// What a permission request produced: an immediate policy verdict, or a
/// wait on the operator.
pub enum PermissionOutcome {
    Immediate(Decision),
    Pending(oneshot::Receiver<Decision>),
}

#[derive(Debug, Clone, Default)]
pub struct Meta {
    pub assignment_id: Option<String>,
    pub verify_for: Option<String>,
    pub budget_usd: f64,
    pub cost_usd: f64,
    pub mission_id: Option<String>,
    pub campaign_id: Option<String>,
    pub team: Option<String>,
    pub objective_id: Option<String>,
    pub environment_scope: Option<String>,
    pub tools_allow: Option<Vec<String>>,
    pub write_scope: Vec<String>,
    pub output_tokens: f64,
    pub max_output_tokens: f64,
    pub started_at: i64,
    pub runtime_ms: i64,
    pub deadline_at: Option<i64>,
    pub budget_stopped: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Assignment {
    pub members: Vec<String>,
    pub outcomes: Vec<(String, String)>,
    pub settled: bool,
}

struct PendingPermission {
    session_id: String,
    tx: oneshot::Sender<Decision>,
    timeout: Option<tokio::task::AbortHandle>,
}

/// One adapter event, queued for the fold.
pub type QueuedEvent = (String, String, Value);

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct RegistryError(pub String);

fn err(message: impl Into<String>) -> RegistryError {
    RegistryError(message.into())
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub(crate) fn uuid() -> String {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("the OS random source is available");
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let h: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}{}{}{}-{}{}-{}{}-{}{}-{}{}{}{}{}{}",
        h[0],
        h[1],
        h[2],
        h[3],
        h[4],
        h[5],
        h[6],
        h[7],
        h[8],
        h[9],
        h[10],
        h[11],
        h[12],
        h[13],
        h[14],
        h[15]
    )
}

pub struct Registry {
    pub settings: Arc<RwLock<FieldSettings>>,
    keys: Option<Arc<Mutex<KeyStore>>>,
    emit: Emit,
    /// Handed to the direct Claude CLI adapter's permission bridge; unused
    /// until that adapter is ported.
    #[allow(dead_code)]
    api_base: String,
    capabilities: Option<Capabilities>,
    #[allow(dead_code)]
    register_secret: Option<RegisterSecret>,
    pub campaign_policy: CampaignPolicy,
    pub budgets: BudgetLedger,
    pub sessions: HashMap<String, Arc<dyn Adapter>>,
    session_order: Vec<String>,
    pub meta: HashMap<String, Meta>,
    pub assignments: HashMap<String, Assignment>,
    permissions: HashMap<String, PendingPermission>,
    pub endpoint_status: HashMap<String, String>,
    report_handler: Option<ReportHandler>,
    exhausted_campaigns: HashSet<String>,
    deadlines: HashMap<String, tokio::task::AbortHandle>,
    me: Weak<Mutex<Registry>>,
    queue: mpsc::UnboundedSender<QueuedEvent>,
    factory: Option<AdapterFactory>,
}

pub struct RegistryOptions {
    pub settings: Arc<RwLock<FieldSettings>>,
    pub emit: Emit,
    pub api_base: String,
    pub keys: Option<Arc<Mutex<KeyStore>>>,
    pub capabilities: Option<Capabilities>,
    pub register_secret: Option<RegisterSecret>,
    pub campaign_policy: Option<CampaignPolicy>,
    pub factory: Option<AdapterFactory>,
}

impl Registry {
    /// Build the registry and its event fold. The fold task runs on the
    /// current tokio runtime and folds every adapter event in order.
    pub fn new(options: RegistryOptions) -> Arc<Mutex<Registry>> {
        let (queue, rx) = mpsc::unbounded_channel::<QueuedEvent>();
        let registry = Arc::new_cyclic(|me: &Weak<Mutex<Registry>>| {
            Mutex::new(Registry {
                settings: options.settings,
                keys: options.keys,
                emit: options.emit,
                api_base: options.api_base,
                capabilities: options.capabilities,
                register_secret: options.register_secret,
                campaign_policy: options
                    .campaign_policy
                    .unwrap_or_else(|| Arc::new(|_| None)),
                budgets: BudgetLedger::new(),
                sessions: HashMap::new(),
                session_order: Vec::new(),
                meta: HashMap::new(),
                assignments: HashMap::new(),
                permissions: HashMap::new(),
                endpoint_status: HashMap::new(),
                report_handler: None,
                exhausted_campaigns: HashSet::new(),
                deadlines: HashMap::new(),
                me: me.clone(),
                queue,
                factory: options.factory,
            })
        });
        Self::start_fold(Arc::downgrade(&registry), rx);
        registry
    }

    fn start_fold(me: Weak<Mutex<Registry>>, mut rx: mpsc::UnboundedReceiver<QueuedEvent>) {
        tokio::spawn(async move {
            while let Some((session_id, kind, data)) = rx.recv().await {
                let Some(registry) = me.upgrade() else {
                    break;
                };
                if kind == "harness.permission_requested" {
                    let (outcome, adapter, request_id) = {
                        let guard = registry.lock();
                        let Ok(mut r) = guard else {
                            continue;
                        };
                        let adapter = r.sessions.get(&session_id).cloned();
                        let request_id = data.get("requestId").cloned().unwrap_or(Value::Null);
                        let outcome = r.request_permission(
                            &session_id,
                            get_str(&data, "toolName").unwrap_or(""),
                            data.get("input").cloned().unwrap_or(Value::Null),
                            Some(request_id.clone()),
                            None,
                        );
                        (outcome, adapter, request_id)
                    };
                    let decision = match outcome {
                        PermissionOutcome::Immediate(d) => d,
                        PermissionOutcome::Pending(rx) => match rx.await {
                            Ok(d) => d,
                            Err(_) => Decision {
                                decision: "deny".into(),
                                message: None,
                            },
                        },
                    };
                    if let Some(adapter) = adapter {
                        adapter.decide_permission(&request_id, &decision.decision);
                    }
                    continue;
                }
                let guard = registry.lock();
                if let Ok(mut r) = guard {
                    r.on_session_event(&session_id, &kind, data);
                }
            }
        });
    }

    /// The sink an adapter emits into.
    pub fn sink(&self) -> EventSink {
        let queue = self.queue.clone();
        Arc::new(move |kind: &str, data: Value| {
            let session_id = get_str(&data, "sessionId").unwrap_or("").to_string();
            let _ = queue.send((session_id, kind.to_string(), data));
        })
    }

    fn emit(
        &self,
        kind: &str,
        data: Value,
        subject: Option<&str>,
        actor: Option<&str>,
        source: Option<Source>,
    ) -> Option<Event> {
        (self.emit)(
            kind,
            data,
            AppendOptions {
                actor: actor.map(str::to_string),
                subject: subject.map(str::to_string),
                source,
                simulated: false,
            },
        )
    }

    pub fn set_campaign_report_handler(&mut self, handler: Option<ReportHandler>) {
        self.report_handler = handler;
    }

    fn endpoint(&self, id: &str) -> Option<Value> {
        self.settings.read().ok().and_then(|s| {
            s.endpoints
                .iter()
                .find(|e| get_str(e, "id") == Some(id))
                .cloned()
        })
    }

    fn workspace(&self, id: &str) -> Option<Value> {
        self.settings
            .read()
            .ok()
            .and_then(|s| s.workspace(id).cloned())
    }

    fn role(&self, id: &str) -> Option<Value> {
        self.settings.read().ok().and_then(|s| {
            s.roles
                .iter()
                .find(|r| get_str(r, "id") == Some(id))
                .cloned()
        })
    }

    fn healthy(&self, id: &str) -> bool {
        matches!(
            self.endpoint_status.get(id).map(String::as_str),
            Some("up") | Some("unknown") | None
        )
    }

    /// Pick a real endpoint. `auto` prefers the role default, then any healthy peer.
    pub fn route(&self, endpoint_id: Option<&str>, role_id: Option<&str>) -> Value {
        let endpoints: Vec<Value> = self
            .settings
            .read()
            .map(|s| s.endpoints.clone())
            .unwrap_or_default();
        if let Some(requested) = endpoint_id.filter(|e| !e.is_empty() && *e != "auto") {
            if self.healthy(requested) {
                return json!({ "endpointId": requested, "reason": "operator choice" });
            }
            let kind = self
                .endpoint(requested)
                .and_then(|e| get_str(&e, "kind").map(str::to_string));
            if let Some(alt) = endpoints.iter().find(|e| {
                get_str(e, "kind").map(str::to_string) == kind
                    && get_str(e, "id").is_some_and(|id| self.healthy(id))
            }) {
                return json!({ "endpointId": alt["id"], "reason": format!("{requested} is down") });
            }
            return json!({ "endpointId": Value::Null, "requestedEndpointId": requested, "reason": "no healthy alternative" });
        }
        let preferred = role_id
            .and_then(|r| self.role(r))
            .and_then(|r| get_str(&r, "default_endpoint").map(str::to_string));
        if let Some(p) = preferred.as_deref().filter(|p| self.healthy(p)) {
            return json!({ "endpointId": p, "reason": "role default" });
        }
        if let Some(any_up) = endpoints
            .iter()
            .find(|e| get_str(e, "id").is_some_and(|id| self.healthy(id)))
        {
            return json!({
                "endpointId": any_up["id"],
                "reason": match &preferred { Some(p) => format!("{p} unavailable"), None => "first healthy endpoint".into() },
            });
        }
        json!({
            "endpointId": Value::Null,
            "requestedEndpointId": preferred.map(Value::String).unwrap_or_else(|| endpoints.first().and_then(|e| e.get("id").cloned()).unwrap_or(Value::Null)),
            "reason": "no endpoint reporting healthy",
        })
    }

    /// The credential and base-url environment a session needs for its
    /// endpoint. The key is resolved from the key store at spawn time only.
    fn endpoint_env(&self, ep: Option<&Value>) -> std::collections::BTreeMap<String, String> {
        let mut env = std::collections::BTreeMap::new();
        let Some(ep) = ep else {
            return env;
        };
        let key = get_str(ep, "secretRef").and_then(|r| {
            self.keys
                .as_ref()
                .and_then(|k| k.lock().ok().and_then(|k| k.get(r).map(str::to_string)))
        });
        let base = get_str(ep, "base_url")
            .or(get_str(ep, "baseUrl"))
            .map(str::to_string);
        match get_str(ep, "kind") {
            Some("openai-compatible") | Some("cameo") => {
                if let Some(base) = base {
                    env.insert("CAMEO_BASE_URL".into(), base);
                    env.insert(
                        "CAMEO_MODEL".into(),
                        get_str(ep, "model").unwrap_or("").to_string(),
                    );
                }
                if let Some(key) = key {
                    env.insert("CAMEO_SERVE_KEY".into(), key);
                }
            }
            Some("anthropic") => {
                if let Some(key) = key {
                    env.insert("ANTHROPIC_API_KEY".into(), key);
                }
                if let Some(base) = base {
                    env.insert("ANTHROPIC_BASE_URL".into(), base);
                }
            }
            Some("ollama") => {
                if let Some(base) = base {
                    env.insert("OLLAMA_HOST".into(), base);
                }
            }
            _ => {}
        }
        env
    }

    /// `adaptive` is resolved from the real size of the target, not a guess.
    pub fn resolve_effort(
        &self,
        thinking: Option<&str>,
        role_id: Option<&str>,
        workspace_id: Option<&str>,
        target_path: Option<&str>,
    ) -> (String, String) {
        if let Some(t) = thinking.filter(|t| !t.is_empty() && *t != "adaptive") {
            return (t.to_string(), "operator choice".into());
        }
        let floor = role_id
            .and_then(|r| self.role(r))
            .and_then(|r| get_str(&r, "default_thinking").map(str::to_string))
            .unwrap_or_else(|| "medium".into());
        let mut measured = "medium".to_string();
        let mut reason = "no measurable target".to_string();
        if let (Some(ws), Some(target)) =
            (workspace_id.and_then(|id| self.workspace(id)), target_path)
        {
            if let Some(root) = get_str(&ws, "path") {
                let abs = Path::new(root).join(target);
                if let Ok(meta) = std::fs::metadata(&abs) {
                    if meta.is_file() {
                        let lines = std::fs::read_to_string(&abs)
                            .map(|t| t.split('\n').count())
                            .unwrap_or(0);
                        measured = if lines < 200 {
                            "low"
                        } else if lines < 800 {
                            "medium"
                        } else {
                            "high"
                        }
                        .into();
                        reason = format!("target file is {lines} lines");
                    } else if meta.is_dir() {
                        let n = count_files(&abs, 400);
                        measured = if n <= 12 {
                            "low"
                        } else if n <= 80 {
                            "medium"
                        } else {
                            "high"
                        }
                        .into();
                        reason = format!(
                            "target holds {} files",
                            if n >= 400 {
                                "400+".to_string()
                            } else {
                                n.to_string()
                            }
                        );
                    }
                }
            }
        }
        let index = |s: &str| {
            EFFORT_ORDER
                .iter()
                .position(|e| *e == s)
                .map(|i| i as i64)
                .unwrap_or(-1)
        };
        let chosen = EFFORT_ORDER
            .get(index(&measured).max(index(&floor)).max(0) as usize)
            .unwrap_or(&"medium")
            .to_string();
        (chosen, format!("adaptive: {reason}"))
    }

    /// Start a session. `body` is the `POST /api/sessions` payload.
    pub fn spawn(&mut self, body: &Value) -> Result<String, RegistryError> {
        let s = |key: &str| get_str(body, key).map(str::to_string);
        let agent_id = s("agentId").unwrap_or_default();
        let mut workspace_id = s("workspaceId");
        let mut mission_id = s("missionId");
        let orders = s("orders");
        let endpoint_id = s("endpointId");
        let thinking = s("thinking");
        let target = get(body, "target").cloned();
        let assignment_id = s("assignmentId");
        let verify_for = s("verifyFor");
        let mut campaign_id = s("campaignId");
        let mut team = s("team");
        let mut objective_id = s("objectiveId");
        let doctrine = get(body, "doctrine").cloned();
        let mut environment_scope = s("environmentScope");
        let budget_usd = get(body, "budgetUsd").map(|v| number(Some(v)));

        // Follow-up work belongs to its original campaign even when the
        // client omits or supplies a different campaign id.
        let inherited: Option<Meta> = if let Some(source) = &verify_for {
            Some(
                self.meta
                    .get(source)
                    .cloned()
                    .ok_or_else(|| err("verification requires a known source session"))?,
            )
        } else if let Some(assignment) = &assignment_id {
            let run = self
                .assignments
                .get(assignment)
                .filter(|a| !a.settled)
                .ok_or_else(|| err("reinforcement requires an active assignment"))?;
            let members: Vec<Option<Meta>> = run
                .members
                .iter()
                .map(|id| self.meta.get(id).cloned())
                .collect();
            if members.is_empty() || members.iter().any(Option::is_none) {
                return Err(err("assignment source policy is unavailable"));
            }
            let campaigns: HashSet<Option<String>> = members
                .iter()
                .flatten()
                .map(|m| m.campaign_id.clone())
                .collect();
            if campaigns.len() != 1 {
                return Err(err(
                    "cannot reinforce an assignment spanning different campaigns",
                ));
            }
            members.into_iter().flatten().next()
        } else {
            None
        };
        if let Some(inherited) = inherited {
            if campaign_id.is_some() && campaign_id != inherited.campaign_id {
                return Err(err("follow-up campaign cannot differ from its source"));
            }
            campaign_id = inherited.campaign_id;
            mission_id = inherited.mission_id;
            team = inherited.team;
            objective_id = inherited.objective_id;
            environment_scope = inherited.environment_scope;
        }

        let settings = self
            .settings
            .read()
            .map_err(|_| err("settings lock poisoned"))?
            .clone();
        let agent = settings
            .agents
            .iter()
            .find(|a| get_str(a, "id") == Some(&agent_id))
            .cloned()
            .ok_or_else(|| err(format!("unknown agent: {agent_id}")))?;
        let role_id = get_str(&agent, "role").map(str::to_string);
        let role = role_id.as_deref().and_then(|r| {
            settings
                .roles
                .iter()
                .find(|x| get_str(x, "id") == Some(r))
                .cloned()
        });
        let ws_id = workspace_id
            .take()
            .or_else(|| {
                settings
                    .workspaces
                    .first()
                    .and_then(|w| get_str(w, "id").map(str::to_string))
            })
            .ok_or_else(|| err("no workspace mounted"))?;
        let ws = settings
            .workspace(&ws_id)
            .cloned()
            .ok_or_else(|| err("no workspace mounted"))?;
        if ws.get("mounted") != Some(&Value::Bool(true)) {
            return Err(err(format!(
                "workspace {ws_id} is not mounted at {}",
                get_str(&ws, "path").unwrap_or("?")
            )));
        }

        let routed = self.route(
            endpoint_id.as_deref().or(get_str(&agent, "endpoint")),
            role_id.as_deref(),
        );
        let Some(routed_id) = get_str(&routed, "endpointId").map(str::to_string) else {
            return Err(err(format!(
                "no healthy inference endpoint for {agent_id}: {}",
                get_str(&routed, "reason").unwrap_or("")
            )));
        };
        let ep = self.endpoint(&routed_id);
        self.assert_capacity(None, Some(&routed_id), campaign_id.as_deref())?;

        let requested_budget = budget_usd
            .or_else(|| get(&agent, "budget_usd").map(|v| number(Some(v))))
            .or_else(|| get(&settings.defaults, "budget_usd_per_session").map(|v| number(Some(v))))
            .unwrap_or(5.0);
        if !requested_budget.is_finite() || requested_budget <= 0.0 {
            return Err(err("session dollar budget must be positive and finite"));
        }
        let campaign = campaign_id
            .as_deref()
            .and_then(|c| (self.campaign_policy)(c));
        if campaign_id.is_some() && campaign.is_none() {
            return Err(err("campaign policy is unavailable"));
        }
        if let Some(c) = &campaign {
            let budget = number(c.get("budgetUsd"));
            let cost = c.get("costUsd").map(|v| number(Some(v))).unwrap_or(0.0);
            if !budget.is_finite() || budget <= 0.0 || !cost.is_finite() || cost < 0.0 {
                return Err(err("campaign budget policy is invalid"));
            }
        }
        let remaining = match (&campaign, &campaign_id) {
            (Some(c), Some(id)) if number(c.get("budgetUsd")) > 0.0 => {
                self.budgets
                    .campaign(
                        id,
                        c.get("costUsd").map(|v| number(Some(v))).unwrap_or(0.0),
                        number(c.get("budgetUsd")),
                    )
                    .remaining_usd
            }
            _ => requested_budget,
        };
        let admitted_budget = requested_budget.min(remaining);
        if admitted_budget <= 0.0 {
            return Err(err(
                "campaign budget is spent or reserved by other sessions",
            ));
        }
        let target_path = target
            .as_ref()
            .and_then(|t| get_str(t, "path").or(get_str(t, "id")).map(str::to_string));
        let (effort, effort_reason) = self.resolve_effort(
            thinking.as_deref().or(get_str(&agent, "thinking")),
            role_id.as_deref(),
            Some(&ws_id),
            target_path.as_deref(),
        );

        let id = uuid();
        let max_children = get(&agent, "max_children")
            .or(get(&settings.defaults, "max_children_per_session"))
            .map(|v| number(Some(v)))
            .unwrap_or(2.0);
        let effective_tools: Vec<String> = get_arr(&agent, "tools_allow")
            .or_else(|| role.as_ref().and_then(|r| get_arr(r, "tools_allow")))
            .map(|items| items.iter().map(js_string).collect())
            .unwrap_or_default();
        let mut system_prompt = compose_prompt(
            &settings,
            &agent_id,
            role_id.as_deref(),
            mission_id.as_deref(),
            None,
        );
        system_prompt.push_str(&format!(
            "\n\n---\n\n# Delegation limit\n\nYou may create at most {} subagents. Maximum delegation depth for this Field is {}.",
            super::js::format_number(max_children),
            super::js::format_number(get(&settings.defaults, "max_delegation_depth").map(|v| number(Some(v))).unwrap_or(2.0))
        ));

        // Every endpoint kind runs through the Knossos adapter with the
        // matching engine; the direct Claude CLI adapter is the next port.
        let kind = ep
            .as_ref()
            .and_then(|e| get_str(e, "kind").map(str::to_string));
        let engine = match kind.as_deref() {
            Some("openai-compatible") | Some("cameo") => "cameo",
            Some("ollama") => "ollama",
            _ => "anthropic",
        };
        let credential_env: Vec<String> = ep
            .as_ref()
            .and_then(|e| e.get("credential_env"))
            .map(|v| match v {
                Value::Array(items) => items.iter().map(js_string).collect(),
                other => vec![js_string(other)],
            })
            .unwrap_or_default();
        let options = KnossosOptions {
            id: id.clone(),
            agent_id: Some(agent_id.clone()),
            name: Some(get_str(&agent, "name").unwrap_or(&agent_id).to_string()),
            role: role_id.clone(),
            model: ep
                .as_ref()
                .and_then(|e| get_str(e, "model").map(str::to_string)),
            endpoint_id: Some(routed_id.clone()),
            effort: Some(effort.clone()),
            cwd: PathBuf::from(get_str(&ws, "path").unwrap_or(".")),
            workspace_id: Some(ws_id.clone()),
            system_prompt: system_prompt.clone(),
            env: self.endpoint_env(ep.as_ref()),
            engine: Some(engine.to_string()),
            provider_kind: kind.clone(),
            read_only: role
                .as_ref()
                .is_some_and(|r| r.get("read_only") == Some(&Value::Bool(true))),
            environment_scope: environment_scope.clone(),
            credential_env_keys: credential_env,
            binary: None,
        };
        let sink = self.sink();
        let adapter: Arc<dyn Adapter> = match &self.factory {
            Some(factory) => factory(options, sink),
            None => KnossosSession::new(options, sink),
        };
        self.sessions.insert(id.clone(), Arc::clone(&adapter));
        self.session_order.push(id.clone());

        let max_output_tokens = bounded_positive(
            get(&agent, "completion_tokens")
                .or(get(&settings.defaults, "completion_tokens_per_session")),
            32_768.0,
            true,
        );
        let runtime_minutes = bounded_positive(
            get(&agent, "wall_minutes").or(get(&settings.defaults, "wall_minutes_per_session")),
            120.0,
            false,
        );
        let started_at = now_ms();
        let runtime_ms = (runtime_minutes * 60_000.0) as i64;
        self.meta.insert(
            id.clone(),
            Meta {
                assignment_id: assignment_id.clone(),
                verify_for: verify_for.clone(),
                budget_usd: admitted_budget,
                cost_usd: 0.0,
                mission_id: mission_id.clone(),
                campaign_id: campaign_id.clone(),
                team: team.clone(),
                objective_id: objective_id.clone(),
                environment_scope: environment_scope.clone(),
                tools_allow: Some(effective_tools.clone()),
                write_scope: role
                    .as_ref()
                    .and_then(|r| get_arr(r, "write_scope"))
                    .map(|items| items.iter().map(js_string).collect())
                    .unwrap_or_default(),
                output_tokens: 0.0,
                max_output_tokens,
                started_at,
                runtime_ms,
                deadline_at: Some(started_at + runtime_ms),
                budget_stopped: false,
            },
        );
        self.arm_session_deadline(&id);

        let reservation = self.emit(
            "budget.reserved",
            json!({ "sessionId": id, "campaignId": campaign_id, "limitUsd": admitted_budget }),
            Some(&id),
            None,
            Some(Source::Derived),
        );
        self.budgets.apply(&reservation.unwrap_or_else(|| {
            synthetic_event(
                "budget.reserved",
                json!({ "sessionId": id, "campaignId": campaign_id, "limitUsd": admitted_budget }),
            )
        }));

        let initial_orders = orders
            .clone()
            .unwrap_or_else(|| "Await orders. Report ready and take no action yet.".into());
        self.emit(
            "session.spawned",
            json!({
                "sessionId": id, "agentId": agent_id, "name": get_str(&agent, "name").unwrap_or(&agent_id),
                "role": role_id, "model": ep.as_ref().and_then(|e| e.get("model").cloned()).unwrap_or(Value::Null),
                "endpointId": routed_id, "routeReason": routed.get("reason"),
                "thinking": effort, "thinkingReason": effort_reason,
                "cwd": ws.get("path"), "workspaceId": ws_id,
                "missionId": mission_id, "assignmentId": assignment_id,
                "target": target, "campaignId": campaign_id, "team": team, "objectiveId": objective_id,
                "doctrine": doctrine, "environmentScope": environment_scope,
                "systemPrompt": system_prompt, "initialOrders": initial_orders,
            }),
            Some(&id),
            None,
            None,
        );
        if let Some(assignment) = assignment_id
            .as_deref()
            .and_then(|a| self.assignments.get_mut(a))
        {
            if !assignment.members.contains(&id) {
                assignment.members.push(id.clone());
            }
        }
        adapter.start(Some(initial_orders)).map_err(err)?;
        Ok(id)
    }

    /// Fold one adapter event: budgets, verification verdicts, settlement,
    /// campaign reports, then the durable record.
    pub fn on_session_event(&mut self, session_id: &str, kind: &str, data: Value) {
        let meta_snapshot = self.meta.get(session_id).cloned().unwrap_or_default();
        let agent_id = self
            .sessions
            .get(session_id)
            .and_then(|s| s.info().agent_id);
        let mut contextual = data.clone();
        if let Value::Object(o) = &mut contextual {
            for (key, fallback) in [
                ("campaignId", meta_snapshot.campaign_id.clone()),
                ("team", meta_snapshot.team.clone()),
                ("objectiveId", meta_snapshot.objective_id.clone()),
            ] {
                if o.get(key).is_none_or(Value::is_null) {
                    o.insert(
                        key.into(),
                        fallback.map(Value::String).unwrap_or(Value::Null),
                    );
                }
            }
        }

        if kind == "session.usage" {
            if let Some(cost) = super::js::finite(&contextual, "costUsd").filter(|c| *c >= 0.0) {
                let (used, limit) = {
                    let meta = self.meta.entry(session_id.to_string()).or_default();
                    meta.cost_usd = meta.cost_usd.max(cost);
                    (meta.cost_usd, meta.budget_usd)
                };
                if limit > 0.0 && used >= limit {
                    self.stop_session_for_budget(session_id, "dollar_cost", used, limit);
                }
            }
            if let Some(tokens) =
                super::js::finite(&contextual, "outputTokens").filter(|t| *t >= 0.0)
            {
                let (used, limit) = {
                    let meta = self.meta.entry(session_id.to_string()).or_default();
                    meta.output_tokens = meta.output_tokens.max(tokens);
                    (meta.output_tokens, meta.max_output_tokens)
                };
                if limit > 0.0 && used >= limit {
                    self.stop_session_for_budget(session_id, "completion_tokens", used, limit);
                }
            }
        }

        if kind == "session.turn_complete" {
            if let Some(verify_for) = meta_snapshot.verify_for.clone() {
                let text = get(&contextual, "result")
                    .map(js_string)
                    .unwrap_or_default();
                static VERDICT: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
                let re = VERDICT.get_or_init(|| {
                    Regex::new(r"(?i)VERDICT:\s*(verified|rejected)").expect("static regex")
                });
                if let Some(caps) = re.captures(&text) {
                    let result = caps[1].to_ascii_lowercase();
                    let assignment = self
                        .meta
                        .get(&verify_for)
                        .and_then(|m| m.assignment_id.clone());
                    self.emit(
                        "work.verified",
                        json!({
                            "sessionId": verify_for, "verifierSessionId": session_id, "assignmentId": assignment,
                            "result": result, "evidence": text.chars().take(2000).collect::<String>(),
                        }),
                        Some(&verify_for),
                        None,
                        None,
                    );
                }
            }
        }

        let stored = self.emit(
            kind,
            contextual.clone(),
            Some(session_id),
            agent_id.as_deref(),
            None,
        );
        let seq = stored.as_ref().map(|e| e.seq);
        self.budgets
            .apply(&stored.unwrap_or_else(|| synthetic_event(kind, contextual.clone())));
        if kind == "session.usage" {
            if let Some(campaign) = meta_snapshot.campaign_id.clone() {
                self.enforce_campaign_budget(&campaign);
            }
        }
        if kind == "session.ended" {
            if let Some(c) = &self.capabilities {
                (c.revoke)(session_id);
            }
            self.clear_session_deadline(session_id);
            let reason = get_str(&contextual, "reason").unwrap_or("exit").to_string();
            self.settle_assignment_member(session_id, &reason);
        }
        if kind == "session.turn_complete" {
            self.settle_assignment_member(session_id, "completed");
            if let (Some(campaign), Some(handler)) = (
                meta_snapshot.campaign_id.clone(),
                self.report_handler.clone(),
            ) {
                if let Some(report) = parse_campaign_report(contextual.get("result")) {
                    let report_kind = report.get("kind").cloned();
                    let role = self.sessions.get(session_id).and_then(|s| s.info().role);
                    let payload = json!({
                        "sessionId": session_id, "agentId": agent_id, "role": role, "eventSeq": seq,
                        "campaignId": campaign, "team": meta_snapshot.team, "objectiveId": meta_snapshot.objective_id,
                        "report": report,
                    });
                    if let Err(message) = handler(payload) {
                        self.emit(
                            "campaign.report_rejected",
                            json!({
                                "campaignId": campaign, "sessionId": session_id, "team": meta_snapshot.team,
                                "objectiveId": meta_snapshot.objective_id, "reason": message, "reportKind": report_kind,
                            }),
                            Some(&campaign),
                            agent_id.as_deref(),
                            None,
                        );
                    }
                }
            }
        }
    }

    /// Compose real orders from a real target.
    pub fn orders_for(&self, target: &Value, extra: Option<&str>) -> String {
        let ws = get_str(target, "workspaceId").and_then(|id| self.workspace(id));
        let ws_name = ws
            .as_ref()
            .and_then(|w| get_str(w, "name"))
            .unwrap_or("undefined")
            .to_string();
        let ws_path = ws
            .as_ref()
            .and_then(|w| get_str(w, "path"))
            .unwrap_or("undefined")
            .to_string();
        let id = get(target, "id").map(js_string).unwrap_or_default();
        let mut lines: Vec<String> = Vec::new();
        match get_str(target, "type") {
            Some("workspace") => lines.push(format!("Your workspace is {ws_name} at {ws_path}. Survey it before acting.")),
            Some("folder") => lines.push(format!(
                "Work inside `{}` in the {ws_name} workspace. Stay in that subtree unless the orders say otherwise.",
                if id.is_empty() { ".".to_string() } else { id.clone() }
            )),
            Some("file") => {
                if id.to_lowercase().ends_with(".md") {
                    lines.push(format!("Read `{id}` in the {ws_name} workspace. It is an instruction document: execute the plan it contains, in order."));
                } else {
                    lines.push(format!("Your target is `{id}` in the {ws_name} workspace."));
                }
            }
            Some("mission") => {
                let mission = self
                    .settings
                    .read()
                    .ok()
                    .and_then(|s| s.missions.iter().find(|m| get_str(m, "id") == Some(&id)).cloned());
                match mission {
                    None => lines.push(format!("Mission {id} is not defined.")),
                    Some(m) => {
                        lines.push(format!("Execute mission \"{}\".", get_str(&m, "name").unwrap_or(&id)));
                        if let Some(t) = get_str(&m, "target") {
                            lines.push(format!("Its target is `{t}` in the {ws_name} workspace."));
                        }
                        lines.push(String::new());
                        lines.push(get_str(&m, "body").unwrap_or("").trim().to_string());
                        if let Some(done) = get_arr(&m, "definition_of_done").filter(|d| !d.is_empty()) {
                            lines.push(String::new());
                            lines.push("Definition of done:".into());
                            for d in done {
                                lines.push(format!("- {}", js_string(d)));
                            }
                        }
                    }
                }
            }
            Some("website") => lines.push(format!(
                "Open {} and study it. Record the exact URL behind every claim you make.",
                get_str(target, "url").map(str::to_string).unwrap_or_else(|| format!("https://{id}"))
            )),
            _ => lines.push(format!("Target: {}", get(target, "label").map(js_string).unwrap_or(id))),
        }
        if let Some(extra) = extra.map(str::trim).filter(|e| !e.is_empty()) {
            lines.push(String::new());
            lines.push(extra.to_string());
        }
        lines.join("\n")
    }

    /// Assign one or more live sessions to a real target.
    pub fn assign(&mut self, body: &Value) -> Result<Value, RegistryError> {
        let session_ids: Vec<String> = get_arr(body, "sessionIds")
            .map(|v| v.iter().map(js_string).collect())
            .unwrap_or_default();
        let target = get(body, "target").cloned().unwrap_or(json!({}));
        let orders = get_str(body, "orders").map(str::to_string);
        let endpoint_id = get_str(body, "endpointId").map(str::to_string);
        let thinking = get_str(body, "thinking").map(str::to_string);
        let assignment_id = uuid();
        let composed = self.orders_for(&target, orders.as_deref());
        let mut skipped: Vec<Value> = Vec::new();
        let mut admitted: Vec<String> = Vec::new();
        let target_ws = get_str(&target, "workspaceId").map(str::to_string);
        for id in &session_ids {
            let Some(session) = self.sessions.get(id).cloned() else {
                skipped.push(json!({ "sessionId": id, "reason": "session is not live" }));
                continue;
            };
            if !self.meta.contains_key(id) {
                skipped.push(json!({ "sessionId": id, "reason": "session is not live" }));
                continue;
            }
            if let Err(e) = self.assert_budget_allows_work(id) {
                skipped.push(json!({ "sessionId": id, "reason": e.0 }));
                continue;
            }
            let info = session.info();
            if let (Some(t), Some(s)) = (&target_ws, &info.workspace_id) {
                if t != s {
                    skipped.push(json!({ "sessionId": id, "reason": format!("rooted in {s}") }));
                    self.emit(
                        "session.state",
                        json!({
                            "sessionId": id, "state": "blocked",
                            "detail": format!("cannot be assigned to {t}: this session is rooted in {s}. Spawn a new agent there instead."),
                        }),
                        Some(id),
                        None,
                        None,
                    );
                    continue;
                }
            }
            admitted.push(id.clone());
        }
        self.emit(
            "assignment.created",
            json!({
                "assignmentId": assignment_id, "sessionIds": admitted, "targetType": target.get("type"),
                "targetId": target.get("id"), "targetLabel": get(&target, "label").cloned().unwrap_or_else(|| or_null(target.get("id"))),
                "workspaceId": target_ws, "orders": composed,
            }),
            Some(&assignment_id),
            None,
            None,
        );
        self.assignments.insert(
            assignment_id.clone(),
            Assignment {
                members: admitted.clone(),
                outcomes: Vec::new(),
                settled: false,
            },
        );
        if admitted.is_empty() {
            self.emit(
                "assignment.cancelled",
                json!({ "assignmentId": assignment_id, "reason": "no eligible live sessions" }),
                Some(&assignment_id),
                None,
                None,
            );
            if let Some(a) = self.assignments.get_mut(&assignment_id) {
                a.settled = true;
            }
            return Ok(json!({ "assignmentId": assignment_id, "skipped": skipped }));
        }
        for id in admitted {
            let Some(session) = self.sessions.get(&id).cloned() else {
                continue;
            };
            let previous = self.meta.get(&id).and_then(|m| m.assignment_id.clone());
            if previous.as_deref().is_some_and(|p| p != assignment_id) {
                self.settle_assignment_member(&id, "reassigned");
            }
            if let Some(meta) = self.meta.get_mut(&id) {
                meta.assignment_id = Some(assignment_id.clone());
            }
            let info = session.info();
            if let Some(requested) = endpoint_id.as_deref().filter(|e| *e != "auto") {
                let routed = self.route(Some(requested), info.role.as_deref());
                if let Some(rid) = get_str(&routed, "endpointId") {
                    if let Some(ep) = self.endpoint(rid) {
                        if info.endpoint_id.as_deref() != Some(rid) {
                            let model = get_str(&ep, "model").map(str::to_string);
                            session.set_endpoint(Some(rid.to_string()), model.clone());
                            self.emit(
                                "endpoint.routed",
                                json!({ "sessionId": id, "endpointId": rid, "model": model, "reason": routed.get("reason") }),
                                Some(&id),
                                None,
                                None,
                            );
                        }
                    }
                }
            }
            if let Some(t) = &thinking {
                let (effort, _) = self.resolve_effort(
                    Some(t),
                    info.role.as_deref(),
                    target_ws.as_deref(),
                    get_str(&target, "id"),
                );
                session.set_effort(&effort);
            }
            let outcome = self
                .admit_existing(&id, None)
                .map_err(|e| e.0)
                .and_then(|_| {
                    self.arm_session_deadline(&id);
                    if !session.is_running() {
                        session.resume(Some(composed.clone())).map(|_| ())
                    } else {
                        session.send(&composed);
                        Ok(())
                    }
                });
            if let Err(message) = outcome {
                skipped.push(json!({ "sessionId": id, "reason": message }));
                self.settle_assignment_member(&id, "error");
            }
        }
        Ok(json!({ "assignmentId": assignment_id, "skipped": skipped }))
    }

    pub fn settle_assignment_member(&mut self, session_id: &str, reason: &str) -> bool {
        let Some(assignment_id) = self
            .meta
            .get(session_id)
            .and_then(|m| m.assignment_id.clone())
        else {
            return false;
        };
        let Some(run) = self.assignments.get_mut(&assignment_id) else {
            return false;
        };
        if run.settled || !run.members.iter().any(|m| m == session_id) {
            return false;
        }
        run.outcomes.retain(|(id, _)| id != session_id);
        run.outcomes
            .push((session_id.to_string(), reason.to_string()));
        let failed = run.outcomes.iter().find(|(_, o)| o == "error").cloned();
        let cancelled = run
            .outcomes
            .iter()
            .find(|(_, o)| o == "cancelled" || o == "reassigned")
            .cloned();
        if let Some((sid, outcome)) = failed.clone().or(cancelled) {
            run.settled = true;
            let kind = if failed.is_some() {
                "assignment.failed"
            } else {
                "assignment.cancelled"
            };
            self.emit(
                kind,
                json!({ "assignmentId": assignment_id, "sessionId": sid, "reason": outcome }),
                Some(&assignment_id),
                None,
                None,
            );
            return true;
        }
        if run.outcomes.len() == run.members.len() {
            run.settled = true;
            let members = run.members.clone();
            self.emit(
                "assignment.completed",
                json!({ "assignmentId": assignment_id, "sessionIds": members, "reason": "all assigned sessions exited successfully" }),
                Some(&assignment_id),
                None,
                None,
            );
            return true;
        }
        false
    }

    pub fn enforce_campaign_budget(&mut self, campaign_id: &str) -> bool {
        let Some(campaign) = (self.campaign_policy)(campaign_id) else {
            return false;
        };
        if !campaign.get("budgetExhausted").is_some_and(truthy)
            || self.exhausted_campaigns.contains(campaign_id)
        {
            return false;
        }
        self.exhausted_campaigns.insert(campaign_id.to_string());
        let session_ids: Vec<String> = self
            .meta
            .iter()
            .filter(|(_, m)| m.campaign_id.as_deref() == Some(campaign_id))
            .map(|(id, _)| id.clone())
            .collect();
        self.emit(
            "campaign.budget_exhausted",
            json!({ "campaignId": campaign_id, "costUsd": campaign.get("costUsd"), "budgetUsd": campaign.get("budgetUsd"), "sessionIds": session_ids }),
            Some(campaign_id),
            None,
            Some(Source::Derived),
        );
        for id in session_ids {
            if let Some(s) = self.sessions.get(&id) {
                s.pause();
            }
        }
        true
    }

    pub fn stop_session_for_budget(
        &mut self,
        session_id: &str,
        budget: &str,
        used: f64,
        limit: f64,
    ) -> bool {
        let Some(meta) = self.meta.get_mut(session_id) else {
            return false;
        };
        if meta.budget_stopped {
            return false;
        }
        meta.budget_stopped = true;
        let (campaign, team, objective) = (
            meta.campaign_id.clone(),
            meta.team.clone(),
            meta.objective_id.clone(),
        );
        self.emit(
            "budget.exhausted",
            json!({ "sessionId": session_id, "campaignId": campaign, "budget": budget, "used": super::js::jnum(used), "limit": super::js::jnum(limit) }),
            Some(session_id),
            None,
            Some(Source::Derived),
        );
        self.emit(
            "session.state",
            json!({
                "sessionId": session_id, "state": "blocked", "campaignId": campaign, "team": team, "objectiveId": objective,
                "detail": format!("{} budget exhausted ({} of {})", budget.replacen('_', " ", 1), super::js::format_number(used), super::js::format_number(limit)),
            }),
            Some(session_id),
            None,
            Some(Source::Derived),
        );
        if let Some(s) = self.sessions.get(session_id) {
            s.pause();
        }
        true
    }

    pub fn arm_session_deadline(&mut self, session_id: &str) {
        self.clear_session_deadline(session_id);
        let Some(meta) = self.meta.get(session_id) else {
            return;
        };
        let Some(deadline_at) = meta.deadline_at else {
            return;
        };
        let (started_at, runtime_ms) = (meta.started_at, meta.runtime_ms);
        let wait = Duration::from_millis((deadline_at - now_ms()).max(1) as u64);
        let me = self.me.clone();
        let id = session_id.to_string();
        let Ok(handle) = std::panic::catch_unwind(|| {
            tokio::spawn(async move {
                tokio::time::sleep(wait).await;
                if let Some(registry) = me.upgrade() {
                    if let Ok(mut r) = registry.lock() {
                        r.deadlines.remove(&id);
                        let used = (now_ms() - started_at) as f64;
                        r.stop_session_for_budget(&id, "wall_time_ms", used, runtime_ms as f64);
                    }
                }
            })
            .abort_handle()
        }) else {
            return;
        };
        self.deadlines.insert(session_id.to_string(), handle);
    }

    pub fn clear_session_deadline(&mut self, session_id: &str) {
        if let Some(handle) = self.deadlines.remove(session_id) {
            handle.abort();
        }
    }

    pub fn has_deadline(&self, session_id: &str) -> bool {
        self.deadlines.contains_key(session_id)
    }

    /// Count owned processes, including idle harnesses and children still draining.
    pub fn assert_capacity(
        &mut self,
        session_id: Option<&str>,
        endpoint_id: Option<&str>,
        campaign_id: Option<&str>,
    ) -> Result<(), RegistryError> {
        let limit = |value: Option<&Value>, fallback: f64| -> Result<f64, RegistryError> {
            match value.filter(|v| !v.is_null()) {
                None => Ok(fallback),
                Some(v) => {
                    let n = number(Some(v));
                    if !n.is_finite() || n.fract() != 0.0 || n < 1.0 {
                        Err(err("session concurrency limits must be positive integers"))
                    } else {
                        Ok(n)
                    }
                }
            }
        };
        let (defaults, endpoint) = {
            let s = self
                .settings
                .read()
                .map_err(|_| err("settings lock poisoned"))?;
            (
                s.defaults.clone(),
                endpoint_id.and_then(|e| {
                    s.endpoints
                        .iter()
                        .find(|x| get_str(x, "id") == Some(e))
                        .cloned()
                }),
            )
        };
        let global_limit = limit(defaults.get("max_concurrent_sessions"), 16.0)?;
        let endpoint_limit = limit(
            endpoint
                .as_ref()
                .and_then(|e| e.get("max_concurrent_sessions")),
            4.0,
        )?;
        let campaign = campaign_id.and_then(|c| (self.campaign_policy)(c));
        let campaign_limit = limit(
            campaign.as_ref().and_then(|c| c.get("concurrency")),
            global_limit,
        )?;
        let active: Vec<(String, Option<String>)> = self
            .sessions
            .iter()
            .filter(|(id, s)| {
                Some(id.as_str()) != session_id && (s.is_running() || s.owned_processes() > 0)
            })
            .map(|(id, s)| (id.clone(), s.info().endpoint_id))
            .collect();
        let reason = if active.len() as f64 >= global_limit {
            Some("global session concurrency limit")
        } else if active
            .iter()
            .filter(|(_, e)| e.as_deref() == endpoint_id)
            .count() as f64
            >= endpoint_limit
        {
            Some("endpoint session concurrency limit")
        } else if campaign_id.is_some()
            && active
                .iter()
                .filter(|(id, _)| {
                    self.meta.get(id).and_then(|m| m.campaign_id.as_deref()) == campaign_id
                })
                .count() as f64
                >= campaign_limit
        {
            Some("campaign session concurrency limit")
        } else {
            None
        };
        if let Some(reason) = reason {
            self.emit(
                "capacity.denied",
                json!({ "sessionId": session_id, "endpointId": endpoint_id, "campaignId": campaign_id, "reason": reason }),
                session_id.or(campaign_id).or(endpoint_id),
                None,
                Some(Source::Derived),
            );
            return Err(err(reason));
        }
        Ok(())
    }

    pub fn admit_existing(
        &mut self,
        session_id: &str,
        endpoint_id: Option<&str>,
    ) -> Result<(), RegistryError> {
        let Some(session) = self.sessions.get(session_id).cloned() else {
            return Ok(());
        };
        let endpoint = endpoint_id
            .map(str::to_string)
            .or(session.info().endpoint_id);
        let campaign = self
            .meta
            .get(session_id)
            .and_then(|m| m.campaign_id.clone());
        self.assert_capacity(Some(session_id), endpoint.as_deref(), campaign.as_deref())
    }

    pub fn assert_budget_allows_work(&mut self, session_id: &str) -> Result<(), RegistryError> {
        let Some(meta) = self.meta.get(session_id).cloned() else {
            return Ok(());
        };
        if meta.budget_stopped
            || (meta.budget_usd > 0.0 && meta.cost_usd >= meta.budget_usd)
            || meta.deadline_at.is_some_and(|d| now_ms() >= d)
            || (meta.max_output_tokens > 0.0 && meta.output_tokens >= meta.max_output_tokens)
        {
            return Err(err(format!("session {session_id} has exhausted its budget; create a new explicitly budgeted session")));
        }
        let campaign = meta
            .campaign_id
            .as_deref()
            .and_then(|c| (self.campaign_policy)(c));
        if campaign
            .as_ref()
            .is_some_and(|c| c.get("budgetExhausted").is_some_and(truthy))
        {
            return Err(err("campaign budget is exhausted"));
        }
        let reservation = self.budgets.reservation(session_id).cloned();
        if let Some(r) = reservation.filter(|r| r.terminal && r.settled && r.spent_usd.is_some()) {
            let needed = (r.limit_usd - r.spent_usd.unwrap_or(0.0)).max(0.0);
            if let (Some(c), Some(id)) = (&campaign, &meta.campaign_id) {
                let budget = number(c.get("budgetUsd"));
                if budget > 0.0
                    && self
                        .budgets
                        .campaign(
                            id,
                            c.get("costUsd").map(|v| number(Some(v))).unwrap_or(0.0),
                            budget,
                        )
                        .remaining_usd
                        < needed
                {
                    return Err(err(
                        "campaign budget is reserved by other sessions; resume denied",
                    ));
                }
            }
            let event = self.emit(
                "budget.reactivated",
                json!({ "sessionId": session_id }),
                Some(session_id),
                None,
                Some(Source::Derived),
            );
            self.budgets.apply(&event.unwrap_or_else(|| {
                synthetic_event("budget.reactivated", json!({ "sessionId": session_id }))
            }));
        }
        Ok(())
    }

    pub fn command(&mut self, kind: &str, payload: &Value) -> Result<Value, RegistryError> {
        let ids: Vec<String> = get_arr(payload, "sessionIds")
            .map(|v| v.iter().map(js_string).collect())
            .unwrap_or_default();
        if matches!(kind, "resume" | "escalate" | "say" | "redirect") {
            for id in &ids {
                self.assert_budget_allows_work(id)?;
            }
        }
        let mut issued = json!({ "kind": kind });
        if let (Value::Object(o), Value::Object(p)) = (&mut issued, payload) {
            for (k, v) in p {
                o.insert(k.clone(), v.clone());
            }
        }
        self.emit(
            "command.issued",
            issued,
            ids.first().map(String::as_str),
            None,
            None,
        );
        let orders = get_str(payload, "orders").map(str::to_string);
        match kind {
            "pause" => {
                for id in &ids {
                    if let Some(s) = self.sessions.get(id) {
                        s.pause();
                    }
                }
                Ok(json!({ "paused": ids.len() }))
            }
            "resume" => {
                for id in &ids {
                    self.admit_existing(id, None)?;
                    self.arm_session_deadline(id);
                    if let Some(s) = self.sessions.get(id).cloned() {
                        s.resume(orders.clone()).map_err(err)?;
                    }
                }
                Ok(json!({ "resumed": ids.len() }))
            }
            "cancel" => {
                for id in &ids {
                    if let Some(s) = self.sessions.get(id) {
                        s.cancel();
                    }
                }
                Ok(json!({ "cancelled": ids.len() }))
            }
            "redirect" => self.assign(
                &json!({ "sessionIds": ids, "target": payload.get("target"), "orders": orders }),
            ),
            "reinforce" => {
                let target = get(payload, "target").cloned().unwrap_or(json!({}));
                let mut spawned = Vec::new();
                for agent_id in get_arr(payload, "agentIds")
                    .map(|v| v.iter().map(js_string).collect::<Vec<_>>())
                    .unwrap_or_default()
                {
                    let id = self.spawn(&json!({
                        "agentId": agent_id, "workspaceId": target.get("workspaceId"),
                        "orders": self.orders_for(&target, orders.as_deref()),
                        "assignmentId": payload.get("assignmentId"), "target": target,
                        "thinking": payload.get("thinking"), "endpointId": payload.get("endpointId"),
                    }))?;
                    spawned.push(id);
                }
                Ok(json!({ "spawned": spawned }))
            }
            "verify" => {
                let verifier = get_str(payload, "verifierAgentId")
                    .map(str::to_string)
                    .or_else(|| {
                        self.settings.read().ok().and_then(|s| {
                            s.agents
                                .iter()
                                .find(|a| get_str(a, "role") == Some("verifier"))
                                .and_then(|a| get_str(a, "id").map(str::to_string))
                        })
                    });
                let Some(verifier) = verifier else {
                    return Err(err("no verifier agent is defined in field/agents"));
                };
                let mut out = Vec::new();
                for id in &ids {
                    let Some(s) = self.sessions.get(id).cloned() else {
                        continue;
                    };
                    let info = s.info();
                    let ws = info.workspace_id.clone().unwrap_or_default();
                    let orders = [
                        format!("Verify the work done by session {} ({id}) in workspace {ws}.", info.name),
                        "Re-derive the result yourself. Run the project checks and paste the real output.".into(),
                        "Do not fix anything you find.".into(),
                        "End your final message with exactly one line: `VERDICT: verified` or `VERDICT: rejected`.".into(),
                    ]
                    .join("\n");
                    let v = self.spawn(&json!({
                        "agentId": verifier, "workspaceId": ws, "orders": orders, "verifyFor": id, "thinking": "high",
                        "target": { "type": "workspace", "id": ws, "workspaceId": ws },
                    }))?;
                    out.push(v);
                }
                Ok(json!({ "verifiers": out }))
            }
            "escalate" => {
                let best = self.settings.read().ok().and_then(|s| {
                    s.endpoints
                        .iter()
                        .find(|e| {
                            get_str(e, "id").is_some_and(|id| {
                                self.endpoint_status.get(id).map(String::as_str) != Some("down")
                            })
                        })
                        .cloned()
                });
                for id in &ids {
                    let Some(s) = self.sessions.get(id).cloned() else {
                        continue;
                    };
                    let best_id = best
                        .as_ref()
                        .and_then(|b| get_str(b, "id").map(str::to_string));
                    self.admit_existing(id, best_id.as_deref())?;
                    let info = s.info();
                    let current = EFFORT_ORDER
                        .iter()
                        .position(|e| *e == info.effort)
                        .unwrap_or(1);
                    let next = EFFORT_ORDER[(current + 1).min(EFFORT_ORDER.len() - 1)];
                    s.set_effort(next);
                    if let Some(b) = &best {
                        let model = get_str(b, "model").map(str::to_string);
                        s.set_endpoint(best_id.clone(), model.clone());
                        self.emit(
                            "endpoint.routed",
                            json!({ "sessionId": id, "endpointId": best_id, "model": model, "reason": "escalated by operator" }),
                            Some(id),
                            None,
                            None,
                        );
                    }
                    self.emit(
                        "session.state",
                        json!({ "sessionId": id, "state": "thinking", "detail": format!("escalated to {next}") }),
                        Some(id),
                        None,
                        None,
                    );
                    if s.is_running() {
                        s.pause();
                    }
                    self.arm_session_deadline(id);
                    s.resume(Some(orders.clone().unwrap_or_else(|| {
                        "Escalated. Re-approach with more care and report what changed in your assessment.".into()
                    })))
                    .map_err(err)?;
                }
                Ok(json!({ "escalated": ids.len() }))
            }
            "say" => {
                let text = get_str(payload, "text").unwrap_or("").to_string();
                for id in &ids {
                    self.admit_existing(id, None)?;
                    if let Some(s) = self.sessions.get(id) {
                        s.send(&text);
                    }
                }
                Ok(json!({ "sent": ids.len() }))
            }
            other => Err(err(format!("unknown command: {other}"))),
        }
    }

    pub fn set_control_group(&mut self, group: &Value, session_ids: Vec<Value>) {
        self.emit(
            "ui.control_group",
            json!({ "group": group, "sessionIds": session_ids }),
            None,
            None,
            None,
        );
    }

    /// Safe live metadata for the campaign director. Never exposes the child.
    pub fn info(&self, session_id: &str) -> Option<Value> {
        let session = self.sessions.get(session_id)?;
        let meta = self.meta.get(session_id)?;
        let info = session.info();
        Some(json!({
            "id": info.id, "agentId": info.agent_id, "role": info.role, "state": info.state,
            "workspaceId": info.workspace_id, "endpointId": info.endpoint_id, "model": info.model,
            "campaignId": meta.campaign_id, "team": meta.team, "objectiveId": meta.objective_id,
        }))
    }

    /// A harness wants to do something consequential.
    pub fn request_permission(
        &mut self,
        session_id: &str,
        tool_name: &str,
        input: Value,
        tool_use_id: Option<Value>,
        capability_session_id: Option<&str>,
    ) -> PermissionOutcome {
        let session_id = capability_session_id.unwrap_or(session_id).to_string();
        let permission_id = uuid();
        self.emit(
            "permission.requested",
            json!({ "permissionId": permission_id, "sessionId": session_id, "toolName": tool_name, "input": input, "toolUseId": tool_use_id }),
            Some(&session_id),
            None,
            None,
        );
        let session = self.sessions.get(&session_id).cloned();
        let info = session.as_ref().map(|s| s.info());
        let meta = self.meta.get(&session_id).cloned().unwrap_or_default();
        let role = info
            .as_ref()
            .and_then(|i| i.role.as_deref())
            .and_then(|r| self.role(r));
        let websites: Vec<String> = self
            .settings
            .read()
            .map(|s| {
                s.websites
                    .iter()
                    .filter_map(|w| get_str(w, "domain").map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let allowed_domains = role
            .as_ref()
            .and_then(|r| get_arr(r, "network_domains"))
            .map(|d| d.iter().map(js_string).collect())
            .unwrap_or(websites);
        let cwd = info.as_ref().map(|i| i.cwd.clone());
        let policy = evaluate_permission_policy(&PolicyInput {
            tool_name,
            input,
            workspace_path: cwd.as_deref(),
            read_only: role
                .as_ref()
                .is_some_and(|r| r.get("read_only") == Some(&Value::Bool(true))),
            environment_scope: meta.environment_scope.as_deref(),
            allowed_tools: meta.tools_allow.clone().or_else(|| {
                role.as_ref()
                    .and_then(|r| get_arr(r, "tools_allow"))
                    .map(|t| t.iter().map(js_string).collect())
            }),
            write_scope: if meta.write_scope.is_empty() {
                role.as_ref()
                    .and_then(|r| get_arr(r, "write_scope"))
                    .map(|s| s.iter().map(js_string).collect())
            } else {
                Some(meta.write_scope.clone())
            },
            validate_write_path: None,
            allowed_domains,
        });
        if let Some(denial) = policy {
            self.emit(
                "permission.decided",
                json!({ "permissionId": permission_id, "sessionId": session_id, "decision": "deny", "by": "scope-policy", "reason": denial.message }),
                Some(&session_id),
                None,
                None,
            );
            return PermissionOutcome::Immediate(Decision {
                decision: "deny".into(),
                message: Some(denial.message),
            });
        }
        let (tx, rx) = oneshot::channel();
        let me = self.me.clone();
        let pid = permission_id.clone();
        let timeout = std::panic::catch_unwind(|| {
            tokio::spawn(async move {
                tokio::time::sleep(PERMISSION_TIMEOUT).await;
                if let Some(registry) = me.upgrade() {
                    if let Ok(mut r) = registry.lock() {
                        if let Some(entry) = r.permissions.remove(&pid) {
                            r.emit(
                                "permission.decided",
                                json!({ "permissionId": pid, "sessionId": entry.session_id, "decision": "deny", "by": "timeout" }),
                                Some(&entry.session_id),
                                None,
                                None,
                            );
                            let _ = entry.tx.send(Decision {
                                decision: "deny".into(),
                                message: Some("No operator decision within 10 minutes.".into()),
                            });
                        }
                    }
                }
            })
            .abort_handle()
        })
        .ok();
        self.permissions.insert(
            permission_id,
            PendingPermission {
                session_id,
                tx,
                timeout,
            },
        );
        PermissionOutcome::Pending(rx)
    }

    pub fn pending_permissions(&self) -> usize {
        self.permissions.len()
    }

    pub fn abandon_operator(&mut self) -> Value {
        let pending: Vec<String> = self.permissions.keys().cloned().collect();
        for id in &pending {
            self.decide_permission(id, "deny", Some("Operator signed out."));
        }
        let running: Vec<String> = self.sessions.keys().cloned().collect();
        for id in &running {
            if let Some(c) = &self.capabilities {
                (c.revoke)(id);
            }
            if let Some(s) = self.sessions.get(id) {
                s.cancel();
            }
        }
        json!({ "denied": pending.len(), "cancelled": running.len() })
    }

    pub fn decide_permission(
        &mut self,
        permission_id: &str,
        decision: &str,
        message: Option<&str>,
    ) -> bool {
        let Some(entry) = self.permissions.remove(permission_id) else {
            return false;
        };
        if let Some(t) = entry.timeout {
            t.abort();
        }
        self.emit(
            "permission.decided",
            json!({ "permissionId": permission_id, "sessionId": entry.session_id, "decision": decision, "by": "operator" }),
            Some(&entry.session_id),
            None,
            None,
        );
        let _ = entry.tx.send(Decision {
            decision: decision.to_string(),
            message: message.map(str::to_string),
        });
        true
    }

    /// A real endpoint failure pauses the sessions it powers and reroutes them.
    pub fn on_endpoint_health(&mut self, endpoint_id: &str, status: &str) {
        let previous = self
            .endpoint_status
            .insert(endpoint_id.to_string(), status.to_string());
        if status != "down" || previous.as_deref() == Some("down") {
            return;
        }
        let affected: Vec<(String, Arc<dyn Adapter>)> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.info().endpoint_id.as_deref() == Some(endpoint_id) && s.is_running())
            .map(|(id, s)| (id.clone(), Arc::clone(s)))
            .collect();
        for (id, s) in affected {
            if self.assert_budget_allows_work(&id).is_err() {
                s.pause();
                continue;
            }
            let routed = self.route(Some("auto"), s.info().role.as_deref());
            match get_str(&routed, "endpointId").map(str::to_string) {
                Some(rid) if rid != endpoint_id => {
                    if self.admit_existing(&id, Some(&rid)).is_err() {
                        s.pause();
                        continue;
                    }
                    s.pause();
                    self.arm_session_deadline(&id);
                    let model = self
                        .endpoint(&rid)
                        .and_then(|e| get_str(&e, "model").map(str::to_string))
                        .or(s.info().model);
                    s.set_endpoint(Some(rid.clone()), model.clone());
                    self.emit(
                        "endpoint.routed",
                        json!({ "sessionId": id, "endpointId": rid, "model": model, "reason": format!("{endpoint_id} went down") }),
                        Some(&id),
                        None,
                        None,
                    );
                    let _ = s.resume(Some("Your endpoint failed and you were rerouted. Re-state where you were and continue.".into()));
                }
                _ => {
                    s.pause();
                    self.emit(
                        "session.state",
                        json!({ "sessionId": id, "state": "blocked", "detail": format!("endpoint {endpoint_id} is down, no alternative") }),
                        Some(&id),
                        None,
                        None,
                    );
                }
            }
        }
    }

    pub fn shutdown(&mut self) {
        for (_, handle) in self.deadlines.drain() {
            handle.abort();
        }
        for s in self.sessions.values() {
            if s.is_running() {
                s.cancel();
            }
        }
    }
}

fn bounded_positive(value: Option<&Value>, fallback: f64, integer: bool) -> f64 {
    let n = number(value);
    if n.is_finite() && n > 0.0 && (!integer || n.fract() == 0.0) {
        n
    } else {
        fallback
    }
}

fn synthetic_event(kind: &str, data: Value) -> Event {
    Event {
        seq: 0,
        ts: now_ms(),
        kind: kind.to_string(),
        actor: None,
        subject: None,
        source: Source::Derived,
        data,
    }
}

fn count_files(dir: &Path, cap: usize) -> usize {
    let mut n = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        if n >= cap {
            break;
        }
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name == ".git" || name == "node_modules" || name == "target" {
                continue;
            }
            match entry.file_type() {
                Ok(t) if t.is_dir() => stack.push(entry.path()),
                Ok(_) => {
                    n += 1;
                    if n >= cap {
                        break;
                    }
                }
                Err(_) => {}
            }
        }
    }
    n
}
