//! Provider-independent mission state and its crash-recoverable journal.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

pub const MISSION_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_SNAPSHOT_INTERVAL: u64 = 32;
pub const MAX_MISSION_EVENT_BYTES: usize = 1024 * 1024;
pub const MAX_MISSION_STATE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_MISSION_JOURNAL_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MissionPhase {
    Intake,
    Recon,
    Contract,
    Baseline,
    Plan,
    Execute,
    Evaluate,
    Replan,
    Challenge,
    Verify,
    Handoff,
    Accepted,
    Revise,
    Reverted,
    Paused,
    Blocked,
    Cancelled,
    Failed,
    Recovering,
}

impl MissionPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Intake => "intake",
            Self::Recon => "recon",
            Self::Contract => "contract",
            Self::Baseline => "baseline",
            Self::Plan => "plan",
            Self::Execute => "execute",
            Self::Evaluate => "evaluate",
            Self::Replan => "replan",
            Self::Challenge => "challenge",
            Self::Verify => "verify",
            Self::Handoff => "handoff",
            Self::Accepted => "accepted",
            Self::Revise => "revise",
            Self::Reverted => "reverted",
            Self::Paused => "paused",
            Self::Blocked => "blocked",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
            Self::Recovering => "recovering",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeKind {
    Observed,
    Inferred,
    UserSupplied,
    AcceptedProjectMemory,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProofStatus {
    Required,
    Passed,
    Failed,
    Invalidated,
    Unavailable,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinalVerdict {
    Unverified,
    PartiallyVerified,
    Verified,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MissionIdentity {
    pub mission_id: String,
    pub parent_id: Option<String>,
    pub revision: u64,
    pub created_at: String,
    pub last_safe_checkpoint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct MissionContract {
    pub outcome: String,
    pub scope: Vec<String>,
    pub exclusions: Vec<String>,
    pub constraints: Vec<String>,
    pub risk: String,
    pub definition_of_done: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MissionPolicy {
    pub execution_mode: String,
    pub tool_grants: BTreeSet<String>,
    pub network_allowed: bool,
    pub secret_handles_only: bool,
    pub approval_leases: BTreeMap<String, String>,
}

impl Default for MissionPolicy {
    fn default() -> Self {
        Self {
            execution_mode: "ask".into(),
            tool_grants: BTreeSet::new(),
            network_allowed: false,
            secret_handles_only: true,
            approval_leases: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct MissionBudget {
    pub token_limit: Option<u64>,
    pub request_limit: Option<u64>,
    pub time_limit_ms: Option<u64>,
    pub dollar_micros_limit: Option<u64>,
    pub tool_limit: Option<u64>,
    pub child_limit: Option<u64>,
    pub retry_limit: Option<u64>,
    pub tokens_used: u64,
    pub requests_used: u64,
    pub time_used_ms: u64,
    pub dollar_micros_used: u64,
    pub tools_used: u64,
    pub children_used: u64,
    pub retries_used: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanNode {
    pub id: String,
    pub dependencies: Vec<String>,
    pub status: String,
    pub owner: Option<String>,
    pub expected_output: String,
    pub proof: String,
    pub rollback: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MissionFocus {
    pub phase: MissionPhase,
    pub active_node: Option<String>,
    pub current_hypothesis: Option<String>,
    pub next_action: Option<String>,
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KnowledgeRecord {
    pub id: String,
    pub kind: KnowledgeKind,
    pub content: String,
    pub provenance: String,
    pub fresh_at_revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub id: String,
    pub source: String,
    pub content_hash: String,
    pub workspace_revision: u64,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceState {
    pub root_identity: String,
    pub base_revision: Option<String>,
    pub revision: u64,
    pub changed_paths: BTreeSet<String>,
    pub locks: BTreeMap<String, String>,
    pub snapshots: Vec<String>,
}

/// Facts discovered before planning that make a host-run mission reproducible.
///
/// This is deliberately descriptive. A recipe never grants installation,
/// network, or mutation authority merely because it was discovered.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct EnvironmentRecord {
    pub adapter: String,
    pub isolation: String,
    pub os: String,
    pub architecture: String,
    pub runtime_versions: BTreeMap<String, String>,
    /// SHA-256 digests of bounded, user-controlled instruction, lock, and
    /// manifest files.
    /// Values are fingerprints only; their contents never enter mission state.
    pub source_hashes: BTreeMap<String, String>,
    pub instruction_files: Vec<String>,
    pub lockfiles: Vec<String>,
    pub workspace_markers: Vec<String>,
    pub setup_recipe: Vec<String>,
    pub verification_recipe: Vec<String>,
    #[serde(default = "environment_setup_not_run")]
    pub setup_status: String,
    #[serde(default = "environment_verification_pending")]
    pub verification_status: String,
    #[serde(default)]
    pub verification_evidence_hash: Option<String>,
    pub discovery_notes: Vec<String>,
}

fn environment_setup_not_run() -> String {
    "not_run_requires_explicit_capability".into()
}

fn environment_verification_pending() -> String {
    "pending".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActionIntent {
    pub call_id: String,
    pub tool: String,
    pub input_hash: String,
    pub at_workspace_revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailureRecord {
    pub family: String,
    pub attempt: String,
    pub retryable: bool,
    pub recovery: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChildRecord {
    pub id: String,
    pub contract: String,
    pub owner_paths: Vec<String>,
    pub status: String,
    pub handoff_evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProofRecord {
    pub id: String,
    pub status: ProofStatus,
    pub bound_revision: Option<u64>,
    pub evidence_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerificationState {
    pub baseline: Vec<String>,
    pub required_checks: BTreeMap<String, ProofRecord>,
    pub final_verdict: FinalVerdict,
}

impl Default for VerificationState {
    fn default() -> Self {
        Self {
            baseline: Vec::new(),
            required_checks: BTreeMap::new(),
            final_verdict: FinalVerdict::Unverified,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ConversationState {
    pub provider_cursor: Option<String>,
    pub recent_message_start: u64,
    pub recent_message_end: u64,
    pub compaction_generation: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MissionState {
    pub schema_version: u32,
    pub identity: MissionIdentity,
    pub contract: MissionContract,
    pub policy: MissionPolicy,
    pub budget: MissionBudget,
    pub plan: BTreeMap<String, PlanNode>,
    pub focus: MissionFocus,
    pub knowledge: BTreeMap<String, KnowledgeRecord>,
    pub evidence: BTreeMap<String, EvidenceRecord>,
    pub workspace: WorkspaceState,
    #[serde(default)]
    pub environment: Option<EnvironmentRecord>,
    pub pending_actions: BTreeMap<String, ActionIntent>,
    pub failures: Vec<FailureRecord>,
    pub children: BTreeMap<String, ChildRecord>,
    pub verification: VerificationState,
    pub conversation: ConversationState,
}

impl MissionState {
    pub fn new(
        mission_id: impl Into<String>,
        contract: MissionContract,
        root_identity: impl Into<String>,
    ) -> Result<Self> {
        let mission_id = mission_id.into();
        validate_id(&mission_id)?;
        Ok(Self {
            schema_version: MISSION_SCHEMA_VERSION,
            identity: MissionIdentity {
                mission_id,
                parent_id: None,
                revision: 0,
                created_at: chrono::Utc::now().to_rfc3339(),
                last_safe_checkpoint: None,
            },
            contract,
            policy: MissionPolicy::default(),
            budget: MissionBudget::default(),
            plan: BTreeMap::new(),
            focus: MissionFocus {
                phase: MissionPhase::Intake,
                active_node: None,
                current_hypothesis: None,
                next_action: None,
                stop_reason: None,
            },
            knowledge: BTreeMap::new(),
            evidence: BTreeMap::new(),
            workspace: WorkspaceState {
                root_identity: root_identity.into(),
                base_revision: None,
                revision: 0,
                changed_paths: BTreeSet::new(),
                locks: BTreeMap::new(),
                snapshots: Vec::new(),
            },
            environment: None,
            pending_actions: BTreeMap::new(),
            failures: Vec::new(),
            children: BTreeMap::new(),
            verification: VerificationState::default(),
            conversation: ConversationState::default(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum MissionEvent {
    Created {
        state: Box<MissionState>,
    },
    Transition {
        from: MissionPhase,
        to: MissionPhase,
        cause: String,
    },
    PlanRevised {
        nodes: Vec<PlanNode>,
        active_node: Option<String>,
        cause: String,
    },
    ConsequentialIntent {
        intent: ActionIntent,
    },
    ConsequentialResult {
        call_id: String,
        tool: String,
        succeeded: bool,
        result_hash: String,
        changed_paths: Vec<String>,
    },
    ApprovalDecision {
        lease_id: String,
        capability: String,
        allowed: bool,
    },
    KnowledgeRecorded {
        record: KnowledgeRecord,
    },
    Compacted {
        generation: u64,
        source_start: u64,
        source_end: u64,
        manifest_hash: String,
    },
    ChildHandoff {
        child: ChildRecord,
    },
    VerifierVerdict {
        proof: ProofRecord,
        verdict: FinalVerdict,
    },
    EnvironmentVerification {
        status: String,
        evidence_hash: String,
    },
    Checkpoint {
        id: String,
        safe: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Envelope {
    sequence: u64,
    at: String,
    previous_hash: String,
    hash: String,
    #[serde(flatten)]
    event: MissionEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Snapshot {
    schema_version: u32,
    sequence: u64,
    journal_hash: String,
    state_hash: String,
    state: MissionState,
}

pub struct MissionStore {
    dir: PathBuf,
    journal: PathBuf,
    state: MissionState,
    sequence: u64,
    last_hash: String,
    snapshot_interval: u64,
}

impl MissionStore {
    /// Held by a checkpoint-enabled executor for its entire ownership period.
    pub fn lock_execution(&self) -> Result<std::fs::File> {
        let path = self.dir.join("execution.lock");
        reject_symlink_components(&path)?;
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(&path)?;
        fs2::FileExt::try_lock_exclusive(&file)
            .context("mission already has an active executor")?;
        Ok(file)
    }

    pub fn journal_hash(&self) -> &str {
        &self.last_hash
    }

    /// Write immutable, bounded conversation bytes before anchoring their hash
    /// in the journal. Unreferenced files after a crash never become executable.
    pub fn save_conversation(&mut self, payload: &serde_json::Value) -> Result<()> {
        anyhow::ensure!(
            self.state.pending_actions.is_empty(),
            "unresolved actions cannot be checkpointed"
        );
        let envelope = serde_json::json!({ "schema": "knossos-conversation/v1",
            "mission_id": self.mission_id(), "revision": self.sequence,
            "journal_hash": self.last_hash, "payload": payload });
        let bytes = serde_json::to_vec(&envelope)?;
        anyhow::ensure!(
            bytes.len() <= MAX_MISSION_STATE_BYTES,
            "conversation checkpoint exceeds 8 MiB"
        );
        let hash = format!("{:x}", sha2::Sha256::digest(&bytes));
        let id = format!("conversation-{hash}");
        let path = self.dir.join(format!("{id}.json"));
        reject_symlink_components(&path)?;
        let mut file = secure_create_new(&path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        #[cfg(unix)]
        std::fs::File::open(&self.dir)?.sync_all()?;
        self.append(MissionEvent::Checkpoint { id, safe: true })?;
        Ok(())
    }

    pub fn load_conversation(&self) -> Result<serde_json::Value> {
        use std::io::Read;
        anyhow::ensure!(
            self.state.pending_actions.is_empty(),
            "unresolved action requires operator reconciliation"
        );
        let id = self
            .state
            .identity
            .last_safe_checkpoint
            .as_deref()
            .context("mission has no conversation checkpoint")?;
        let hash = id
            .strip_prefix("conversation-")
            .context("mission has no portable conversation checkpoint")?;
        anyhow::ensure!(
            hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid checkpoint identity"
        );
        let path = self.dir.join(format!("{id}.json"));
        reject_symlink_components(&path)?;
        anyhow::ensure!(
            std::fs::metadata(&path)?.is_file(),
            "checkpoint is not a regular file"
        );
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(MAX_MISSION_STATE_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        anyhow::ensure!(
            bytes.len() <= MAX_MISSION_STATE_BYTES,
            "checkpoint exceeds size limit"
        );
        anyhow::ensure!(
            format!("{:x}", sha2::Sha256::digest(&bytes)) == hash,
            "conversation checkpoint hash mismatch"
        );
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        anyhow::ensure!(
            value["schema"] == "knossos-conversation/v1"
                && value["mission_id"] == self.mission_id(),
            "checkpoint identity mismatch"
        );
        let revision = value["revision"]
            .as_u64()
            .context("checkpoint revision missing")?;
        anyhow::ensure!(
            revision.checked_add(1) == Some(self.sequence),
            "mission advanced beyond this checkpoint; review interrupted work"
        );
        // The hash-anchoring checkpoint must be the next journal event. The
        // saved preceding hash is also checked against that event's envelope.
        let last = std::fs::read_to_string(&self.journal)?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .last()
            .map(str::to_owned)
            .context("mission journal is empty")?;
        let event: Envelope = serde_json::from_str(&last)?;
        anyhow::ensure!(
            value["journal_hash"] == event.previous_hash,
            "checkpoint journal binding mismatch"
        );
        Ok(value["payload"].clone())
    }

    pub fn create(workspace: impl AsRef<Path>, state: MissionState) -> Result<Self> {
        validate_state(&state)?;
        let dir = mission_dir(workspace.as_ref(), &state.identity.mission_id)?;
        reject_symlink_components(&dir)?;
        std::fs::create_dir_all(&dir)?;
        secure_directory(&dir)?;
        let journal = dir.join("journal.jsonl");
        if journal.exists() {
            bail!("mission `{}` already exists", state.identity.mission_id);
        }
        let mut store = Self {
            dir,
            journal,
            state: state.clone(),
            sequence: 0,
            last_hash: String::new(),
            snapshot_interval: DEFAULT_SNAPSHOT_INTERVAL,
        };
        store.append(MissionEvent::Created {
            state: Box::new(state),
        })?;
        Ok(store)
    }

    /// Read and integrity-check historical evidence. This never makes a
    /// mission actionable, so legacy or drifted journals remain inspectable.
    pub fn open(workspace: impl AsRef<Path>, mission_id: &str) -> Result<Self> {
        validate_id(mission_id)?;
        let dir = mission_dir(workspace.as_ref(), mission_id)?;
        reject_symlink_components(&dir)?;
        let journal = dir.join("journal.jsonl");
        if std::fs::metadata(&journal)?.len() > MAX_MISSION_JOURNAL_BYTES {
            bail!(
                "mission journal exceeds its {} byte limit",
                MAX_MISSION_JOURNAL_BYTES
            );
        }
        let file = std::fs::File::open(&journal)
            .with_context(|| format!("opening mission `{mission_id}`"))?;
        let lines: Vec<String> = std::io::BufReader::new(file)
            .lines()
            .collect::<std::io::Result<_>>()?;
        let mut state = None;
        let mut sequence = 0;
        let mut last_hash = String::new();
        for (index, line) in lines.iter().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let envelope: Envelope = match serde_json::from_str(line) {
                Ok(value) => value,
                Err(_) if index + 1 == lines.len() => break, // tolerate one torn crash tail
                Err(error) => {
                    return Err(error).context("mission journal contains malformed history")
                }
            };
            if envelope.sequence != sequence + 1 || envelope.previous_hash != last_hash {
                bail!("mission journal sequence/hash chain is invalid");
            }
            if envelope_hash(
                envelope.sequence,
                &envelope.at,
                &envelope.previous_hash,
                &envelope.event,
            )? != envelope.hash
            {
                bail!("mission journal event hash is invalid");
            }
            match &envelope.event {
                MissionEvent::Created { state: initial } if state.is_none() => {
                    validate_state(initial)?;
                    state = Some(initial.as_ref().clone());
                }
                MissionEvent::Created { .. } => {
                    bail!("mission journal contains a second creation event")
                }
                event => apply_event(
                    state
                        .as_mut()
                        .context("mission journal does not begin with creation")?,
                    event,
                )?,
            }
            sequence = envelope.sequence;
            state
                .as_mut()
                .context("mission journal does not begin with creation")?
                .identity
                .revision = sequence;
            last_hash = envelope.hash;
        }
        let state = state.context("mission journal has no creation event")?;
        Ok(Self {
            dir,
            journal,
            state,
            sequence,
            last_hash,
            snapshot_interval: DEFAULT_SNAPSHOT_INTERVAL,
        })
    }

    /// Open a mission for execution after a process restart. Historical reads
    /// use [`Self::open`]; only this path rejects missing or drifted evidence.
    pub fn open_for_resume(workspace: impl AsRef<Path>, mission_id: &str) -> Result<Self> {
        let store = Self::open(workspace.as_ref(), mission_id)?;
        let environment = store.state.environment.as_ref().context(
            "mission lacks environment evidence and cannot resume; start a revised mission",
        )?;
        if !crate::environment::EnvironmentFingerprint::unchanged_since(
            workspace.as_ref(),
            environment,
        )? {
            bail!(
                "mission environment drifted since it was recorded; start a revised mission after reviewing instructions and lockfiles"
            );
        }
        Ok(store)
    }

    pub fn state(&self) -> &MissionState {
        &self.state
    }
    pub fn mission_id(&self) -> &str {
        &self.state.identity.mission_id
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn with_snapshot_interval(mut self, interval: u64) -> Self {
        self.snapshot_interval = interval.max(1);
        self
    }

    pub fn append(&mut self, event: MissionEvent) -> Result<()> {
        if matches!(event, MissionEvent::Created { .. }) && self.sequence != 0 {
            bail!("creation can only be the first mission event");
        }
        let mut next = self.state.clone();
        match &event {
            MissionEvent::Created { state } => {
                validate_state(state)?;
                if state.as_ref() != &self.state {
                    bail!("creation event does not match the mission state");
                }
            }
            _ => apply_event(&mut next, &event)?,
        }
        let sequence = self.sequence + 1;
        let at = chrono::Utc::now().to_rfc3339();
        let hash = envelope_hash(sequence, &at, &self.last_hash, &event)?;
        let envelope = Envelope {
            sequence,
            at,
            previous_hash: self.last_hash.clone(),
            hash: hash.clone(),
            event,
        };
        let encoded = serde_json::to_vec(&envelope)?;
        if encoded.len() > MAX_MISSION_EVENT_BYTES {
            bail!(
                "mission event exceeds its {} byte limit",
                MAX_MISSION_EVENT_BYTES
            );
        }
        reject_symlink_components(&self.journal)?;
        let existing = std::fs::metadata(&self.journal).map_or(0, |metadata| metadata.len());
        if existing.saturating_add(encoded.len() as u64 + 1) > MAX_MISSION_JOURNAL_BYTES {
            bail!(
                "mission journal would exceed its {} byte limit",
                MAX_MISSION_JOURNAL_BYTES
            );
        }
        let mut file = secure_append(&self.journal)?;
        file.write_all(&encoded)?;
        file.write_all(b"\n")?;
        file.flush()?;
        file.sync_data()?;
        self.state = next;
        self.state.identity.revision = sequence;
        self.sequence = sequence;
        self.last_hash = hash;
        if sequence.rem_euclid(self.snapshot_interval) == 0 {
            self.snapshot()?;
        }
        Ok(())
    }

    pub fn snapshot(&self) -> Result<PathBuf> {
        let snapshots = self.dir.join("snapshots");
        reject_symlink_components(&snapshots)?;
        std::fs::create_dir_all(&snapshots)?;
        secure_directory(&snapshots)?;
        let state_json = serde_json::to_vec(&self.state)?;
        if state_json.len() > MAX_MISSION_STATE_BYTES {
            bail!(
                "mission state exceeds its {} byte limit",
                MAX_MISSION_STATE_BYTES
            );
        }
        let snap = Snapshot {
            schema_version: MISSION_SCHEMA_VERSION,
            sequence: self.sequence,
            journal_hash: self.last_hash.clone(),
            state_hash: digest(&state_json),
            state: self.state.clone(),
        };
        let target = snapshots.join(format!("{:020}.json", self.sequence));
        if target.exists() {
            return Ok(target);
        }
        let temp = snapshots.join(format!(".{:020}.tmp", self.sequence));
        let mut file = secure_create_new(&temp)?;
        file.write_all(&serde_json::to_vec_pretty(&snap)?)?;
        file.flush()?;
        file.sync_all()?;
        std::fs::rename(&temp, &target)?;
        Ok(target)
    }
}

fn apply_event(state: &mut MissionState, event: &MissionEvent) -> Result<()> {
    match event {
        MissionEvent::Created { .. } => bail!("creation can only be the first event"),
        MissionEvent::Transition { from, to, cause } => {
            if state.focus.phase != *from {
                bail!("transition source does not match current phase");
            }
            if cause.trim().is_empty() || !transition_allowed(*from, *to) {
                bail!("mission phase transition is not allowed");
            }
            if *from == MissionPhase::Verify
                && *to == MissionPhase::Handoff
                && !proofs_current(state)
            {
                bail!("verified handoff requires every proof at the current workspace revision");
            }
            state.focus.phase = *to;
            state.focus.stop_reason = matches!(
                to,
                MissionPhase::Blocked | MissionPhase::Cancelled | MissionPhase::Failed
            )
            .then(|| cause.clone());
        }
        MissionEvent::PlanRevised {
            nodes,
            active_node,
            cause,
        } => {
            if cause.trim().is_empty() {
                bail!("plan revision needs a cause");
            }
            let ids: BTreeSet<_> = nodes.iter().map(|node| node.id.as_str()).collect();
            if ids.len() != nodes.len() {
                bail!("plan node ids must be unique");
            }
            for node in nodes {
                if node
                    .dependencies
                    .iter()
                    .any(|dependency| !ids.contains(dependency.as_str()))
                {
                    bail!("plan dependency does not exist");
                }
            }
            if active_node
                .as_ref()
                .is_some_and(|id| !ids.contains(id.as_str()))
            {
                bail!("active plan node does not exist");
            }
            state.plan = nodes
                .iter()
                .cloned()
                .map(|node| (node.id.clone(), node))
                .collect();
            state.focus.active_node = active_node.clone();
        }
        MissionEvent::ConsequentialIntent { intent } => {
            if intent.call_id.trim().is_empty()
                || intent.tool.trim().is_empty()
                || intent.input_hash.trim().is_empty()
                || intent.at_workspace_revision != state.workspace.revision
                || state.pending_actions.contains_key(&intent.call_id)
            {
                bail!("invalid or duplicate consequential action intent");
            }
            state
                .pending_actions
                .insert(intent.call_id.clone(), intent.clone());
        }
        MissionEvent::ConsequentialResult {
            call_id,
            tool,
            succeeded,
            result_hash,
            changed_paths,
        } => {
            if tool.trim().is_empty() || result_hash.trim().is_empty() {
                bail!("consequential result needs tool and hash");
            }
            let intent = state
                .pending_actions
                .remove(call_id)
                .context("consequential result has no matching intent")?;
            if intent.tool != *tool {
                bail!("consequential result tool does not match its intent");
            }
            state.budget.tools_used = state.budget.tools_used.saturating_add(1);
            if *succeeded {
                state.workspace.revision = state.workspace.revision.saturating_add(1);
                state
                    .workspace
                    .changed_paths
                    .extend(changed_paths.iter().cloned());
                for proof in state.verification.required_checks.values_mut() {
                    if proof.bound_revision != Some(state.workspace.revision) {
                        proof.status = ProofStatus::Invalidated;
                    }
                }
                state.verification.final_verdict = FinalVerdict::Unverified;
            }
        }
        MissionEvent::ApprovalDecision {
            lease_id,
            capability,
            allowed,
        } => {
            if *allowed {
                state
                    .policy
                    .approval_leases
                    .insert(lease_id.clone(), capability.clone());
            } else {
                state.policy.approval_leases.remove(lease_id);
            }
        }
        MissionEvent::KnowledgeRecorded { record } => {
            state.knowledge.insert(record.id.clone(), record.clone());
        }
        MissionEvent::Compacted {
            generation,
            source_start,
            source_end,
            manifest_hash,
        } => {
            if *generation != state.conversation.compaction_generation + 1
                || source_start > source_end
                || manifest_hash.is_empty()
            {
                bail!("invalid compaction manifest");
            }
            state.conversation.compaction_generation = *generation;
            state.conversation.recent_message_start = source_end.saturating_add(1);
        }
        MissionEvent::ChildHandoff { child } => {
            state.children.insert(child.id.clone(), child.clone());
        }
        MissionEvent::VerifierVerdict { proof, verdict } => {
            if proof.status == ProofStatus::Passed
                && proof.bound_revision != Some(state.workspace.revision)
            {
                bail!("proof is not bound to the current workspace revision");
            }
            state
                .verification
                .required_checks
                .insert(proof.id.clone(), proof.clone());
            state.verification.final_verdict = *verdict;
        }
        MissionEvent::EnvironmentVerification {
            status,
            evidence_hash,
        } => {
            if !matches!(status.as_str(), "passed" | "failed" | "partial")
                || evidence_hash.trim().is_empty()
            {
                bail!("invalid environment verification result");
            }
            let environment = state
                .environment
                .as_mut()
                .context("environment verification has no captured environment")?;
            environment.verification_status = status.clone();
            environment.verification_evidence_hash = Some(evidence_hash.clone());
        }
        MissionEvent::Checkpoint { id, safe } => {
            if *safe {
                state.identity.last_safe_checkpoint = Some(id.clone());
            }
            state.workspace.snapshots.push(id.clone());
        }
    }
    Ok(())
}

fn proofs_current(state: &MissionState) -> bool {
    state.verification.final_verdict == FinalVerdict::Verified
        && !state.verification.required_checks.is_empty()
        && state.verification.required_checks.values().all(|proof| {
            proof.status == ProofStatus::Passed
                && proof.bound_revision == Some(state.workspace.revision)
        })
}

fn transition_allowed(from: MissionPhase, to: MissionPhase) -> bool {
    use MissionPhase::*;
    if matches!(to, Paused | Blocked | Cancelled | Failed | Recovering) {
        return !matches!(from, Accepted | Reverted | Cancelled | Failed);
    }
    matches!(
        (from, to),
        (Intake, Recon)
            | (Recon, Contract)
            | (Contract, Baseline)
            | (Baseline, Plan)
            | (Plan, Execute)
            | (Execute, Evaluate)
            | (Evaluate, Execute)
            | (Evaluate, Replan)
            | (Replan, Plan)
            | (Evaluate, Challenge)
            | (Challenge, Verify)
            | (Verify, Replan)
            | (Verify, Handoff)
            | (Handoff, Accepted)
            | (Handoff, Revise)
            | (Handoff, Reverted)
            | (Revise, Plan)
            | (Recovering, Recon)
            | (Paused, Execute)
            | (Blocked, Recon)
    )
}

fn validate_state(state: &MissionState) -> Result<()> {
    validate_id(&state.identity.mission_id)?;
    if state.schema_version != MISSION_SCHEMA_VERSION {
        bail!(
            "unsupported mission schema version {}",
            state.schema_version
        );
    }
    if state.contract.outcome.trim().is_empty() {
        bail!("mission outcome cannot be empty");
    }
    if serde_json::to_vec(state)?.len() > MAX_MISSION_STATE_BYTES {
        bail!(
            "mission state exceeds its {} byte limit",
            MAX_MISSION_STATE_BYTES
        );
    }
    Ok(())
}

fn validate_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id.len() > 96
        || !id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
    {
        bail!("mission id must be 1-96 safe ASCII characters");
    }
    Ok(())
}

fn mission_dir(workspace: &Path, mission_id: &str) -> Result<PathBuf> {
    validate_id(mission_id)?;
    Ok(workspace.join(".knossos").join("missions").join(mission_id))
}

fn reject_symlink_components(path: &Path) -> Result<()> {
    for ancestor in path.ancestors() {
        if std::fs::symlink_metadata(ancestor)
            .is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            bail!(
                "mission storage path contains a symlink: {}",
                ancestor.display()
            );
        }
    }
    Ok(())
}

fn secure_directory(_path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(_path, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn secure_append(path: &Path) -> Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

fn secure_create_new(path: &Path) -> Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

fn digest(bytes: &[u8]) -> String {
    let mut hash = Sha1::new();
    hash.update(bytes);
    format!("sha1:{:x}", hash.finalize())
}

/// Hash evidence before it enters the mission journal.
///
/// Tool output can contain source, credentials, or other operator data. The
/// durable mission keeps only this fingerprint; full content remains in the
/// explicitly opt-in trace path.
pub fn evidence_digest(bytes: &[u8]) -> String {
    digest(bytes)
}

fn envelope_hash(sequence: u64, at: &str, previous: &str, event: &MissionEvent) -> Result<String> {
    Ok(digest(&serde_json::to_vec(&(
        sequence, at, previous, event,
    ))?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> MissionContract {
        MissionContract {
            outcome: "ship safely".into(),
            definition_of_done: vec!["tests pass".into()],
            ..MissionContract::default()
        }
    }

    fn state_with_environment(dir: &Path, id: &str) -> MissionState {
        let mut state = MissionState::new(id, contract(), "workspace-a").unwrap();
        state.environment = Some(
            crate::environment::EnvironmentFingerprint::discover(dir)
                .unwrap()
                .into_record(),
        );
        state
    }

    fn intent(call_id: &str, tool: &str, revision: u64) -> MissionEvent {
        MissionEvent::ConsequentialIntent {
            intent: ActionIntent {
                call_id: call_id.into(),
                tool: tool.into(),
                input_hash: "sha1:input".into(),
                at_workspace_revision: revision,
            },
        }
    }

    #[test]
    fn journal_replays_and_snapshots_without_provider_messages() {
        let dir = tempfile::tempdir().unwrap();
        let state = state_with_environment(dir.path(), "mission-1");
        let mut store = MissionStore::create(dir.path(), state)
            .unwrap()
            .with_snapshot_interval(2);
        store
            .append(MissionEvent::Transition {
                from: MissionPhase::Intake,
                to: MissionPhase::Recon,
                cause: "inspect".into(),
            })
            .unwrap();
        assert!(store
            .dir
            .join("snapshots")
            .join("00000000000000000002.json")
            .exists());
        store
            .append(MissionEvent::Transition {
                from: MissionPhase::Recon,
                to: MissionPhase::Contract,
                cause: "scope known".into(),
            })
            .unwrap();
        let reopened = MissionStore::open(dir.path(), "mission-1").unwrap();
        assert_eq!(reopened.sequence(), 3);
        assert_eq!(reopened.state(), store.state());
        assert_eq!(reopened.state().focus.phase, MissionPhase::Contract);
    }

    #[test]
    fn a_write_invalidates_old_proof_and_blocks_handoff() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = MissionState::new("mission-2", contract(), "workspace-a").unwrap();
        state.focus.phase = MissionPhase::Verify;
        state.verification.final_verdict = FinalVerdict::Verified;
        state.verification.required_checks.insert(
            "tests".into(),
            ProofRecord {
                id: "tests".into(),
                status: ProofStatus::Passed,
                bound_revision: Some(0),
                evidence_id: Some("e1".into()),
            },
        );
        let mut store = MissionStore::create(dir.path(), state).unwrap();
        store.append(intent("call-1", "edit_file", 0)).unwrap();
        store
            .append(MissionEvent::ConsequentialResult {
                call_id: "call-1".into(),
                tool: "edit_file".into(),
                succeeded: true,
                result_hash: "sha1:result".into(),
                changed_paths: vec!["src/lib.rs".into()],
            })
            .unwrap();
        assert_eq!(
            store.state().verification.required_checks["tests"].status,
            ProofStatus::Invalidated
        );
        assert!(store
            .append(MissionEvent::Transition {
                from: MissionPhase::Verify,
                to: MissionPhase::Handoff,
                cause: "done".into()
            })
            .is_err());
    }

    #[test]
    fn an_unfinished_intent_survives_replay_as_uncertain_work() {
        let dir = tempfile::tempdir().unwrap();
        let state = state_with_environment(dir.path(), "mission-pending");
        let mut store = MissionStore::create(dir.path(), state).unwrap();
        store.append(intent("call-1", "run", 0)).unwrap();

        let reopened = MissionStore::open(dir.path(), "mission-pending").unwrap();
        assert_eq!(reopened.state().pending_actions["call-1"].tool, "run");
        assert_eq!(reopened.state().workspace.revision, 0);
    }

    #[test]
    fn a_successful_command_without_changed_paths_advances_the_revision() {
        let dir = tempfile::tempdir().unwrap();
        let state = MissionState::new("mission-command", contract(), "workspace-a").unwrap();
        let mut store = MissionStore::create(dir.path(), state).unwrap();
        store.append(intent("call-1", "run", 0)).unwrap();
        store
            .append(MissionEvent::ConsequentialResult {
                call_id: "call-1".into(),
                tool: "run".into(),
                succeeded: true,
                result_hash: "sha1:result".into(),
                changed_paths: Vec::new(),
            })
            .unwrap();

        assert!(store.state().pending_actions.is_empty());
        assert_eq!(store.state().workspace.revision, 1);
    }

    #[test]
    fn only_current_complete_proof_allows_verified_handoff() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = MissionState::new("mission-3", contract(), "workspace-a").unwrap();
        state.focus.phase = MissionPhase::Verify;
        let mut store = MissionStore::create(dir.path(), state).unwrap();
        store
            .append(MissionEvent::VerifierVerdict {
                proof: ProofRecord {
                    id: "tests".into(),
                    status: ProofStatus::Passed,
                    bound_revision: Some(0),
                    evidence_id: Some("e1".into()),
                },
                verdict: FinalVerdict::Verified,
            })
            .unwrap();
        store
            .append(MissionEvent::Transition {
                from: MissionPhase::Verify,
                to: MissionPhase::Handoff,
                cause: "all proof current".into(),
            })
            .unwrap();
        assert_eq!(store.state().focus.phase, MissionPhase::Handoff);
    }

    #[test]
    fn a_torn_tail_is_ignored_but_tampering_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let state = state_with_environment(dir.path(), "mission-4");
        let store = MissionStore::create(dir.path(), state).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&store.journal)
            .unwrap();
        write!(file, "{{\"sequence\":").unwrap();
        drop(file);
        assert_eq!(
            MissionStore::open(dir.path(), "mission-4")
                .unwrap()
                .sequence(),
            1
        );

        let text = std::fs::read_to_string(&store.journal)
            .unwrap()
            .replace("ship safely", "ship unsafely");
        std::fs::write(&store.journal, text).unwrap();
        assert!(MissionStore::open(dir.path(), "mission-4").is_err());
    }

    #[test]
    fn restart_refuses_a_persisted_mission_when_its_instruction_drifted() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "first instruction\n").unwrap();
        let mut state = MissionState::new("mission-drift", contract(), "workspace-a").unwrap();
        state.environment = Some(
            crate::environment::EnvironmentFingerprint::discover(dir.path())
                .unwrap()
                .into_record(),
        );
        MissionStore::create(dir.path(), state).unwrap();
        assert!(MissionStore::open_for_resume(dir.path(), "mission-drift").is_ok());

        std::fs::write(dir.path().join("AGENTS.md"), "changed instruction\n").unwrap();
        assert!(MissionStore::open(dir.path(), "mission-drift").is_ok());
        let error = MissionStore::open_for_resume(dir.path(), "mission-drift")
            .err()
            .expect("drift must reject a persisted mission");
        assert!(error.to_string().contains("environment drifted"));
    }

    #[test]
    fn legacy_mission_without_environment_evidence_cannot_resume() {
        let dir = tempfile::tempdir().unwrap();
        let state = MissionState::new("mission-legacy", contract(), "workspace-a").unwrap();
        MissionStore::create(dir.path(), state).unwrap();
        assert!(MissionStore::open(dir.path(), "mission-legacy").is_ok());
        let error = MissionStore::open_for_resume(dir.path(), "mission-legacy")
            .err()
            .expect("legacy mission must not resume");
        assert!(error.to_string().contains("lacks environment evidence"));
    }
}
