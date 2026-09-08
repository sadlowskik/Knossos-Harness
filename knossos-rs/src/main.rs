use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use sha1::{Digest, Sha1};

use knossos::acp;
use knossos::argus::Argus;
use knossos::ariadne::Ariadne;
use knossos::config::{Config, EngineKind};
use knossos::delegate;
use knossos::engine::{self, Message, Request};
use knossos::environment::EnvironmentFingerprint;
use knossos::gate::RetrievalGate;
use knossos::mcp;
use knossos::mnemosyne::Mnemosyne;
use knossos::oracle::Oracle;
use knossos::scribe::SymbolIndex;
use knossos::session::{Session, TraceEvent};
use knossos::talos::Talos;
use knossos::themis::Themis;
use knossos::tools::{ToolCtx, ToolRegistry};
use knossos::{diff, metis, repl};

#[derive(Parser)]
#[command(
    name = "knossos",
    version,
    about = "The Knossos agentic coding harness",
    long_about = "Exact symbol memory (Scribe), an always-on constitution (Themis), \
                  tiered verification (Oracle) and an explicit halting policy (Ariadne) \
                  around a swappable engine."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Workspace root. Every file operation is jailed inside it.
    #[arg(long, short = 'w', global = true, default_value = ".")]
    workspace: PathBuf,

    #[arg(
        long,
        short = 'e',
        global = true,
        value_enum,
        default_value = "anthropic"
    )]
    engine: EngineKind,

    /// Model id. Defaults per engine; also read from KNOSSOS_MODEL.
    #[arg(long, short = 'm', global = true)]
    model: Option<String>,

    /// OpenAI-compat provider (`groq`, `gemini`, `openrouter`, `qwen`, …).
    /// Used with `--engine openai`; the named `--engine groq` forms set it
    /// for you.
    #[arg(long, global = true)]
    provider: Option<String>,

    /// Override the provider's base URL.
    #[arg(long, global = true)]
    base_url: Option<String>,

    #[arg(long, global = true)]
    verbose: bool,
}

/// Options shared by `task` and `repl`.
#[derive(clap::Args, Clone)]
struct LoopArgs {
    /// Hard ceiling on engine turns.
    ///
    /// This is the value the binary actually ships with — it shadows
    /// `Config::default()`, so raising the ceiling in one place and not the
    /// other changes nothing for anyone running `knossos`.
    #[arg(long, default_value = "20")]
    max_steps: usize,
    /// Where budget pressure begins.
    #[arg(long, default_value = "6")]
    target_steps: usize,
    /// JSONL trajectory log. Defaults to `.knossos/trace-<pid>.jsonl`.
    #[arg(long)]
    trace: Option<PathBuf>,
    /// Stage edits in memory and show diffs instead of writing to disk.
    #[arg(long)]
    dry_run: bool,
    /// Skip Oracle tier 4 (model judgement against the constitution).
    #[arg(long)]
    no_judge: bool,
    /// Do not offer repository context unasked.
    ///
    /// The `search_code` tool still works; this only turns off the proactive
    /// path, which costs an index build at startup and a gate decision per turn.
    #[arg(long)]
    no_context: bool,
    /// Disable durable failure/attempt memory for an ablation.
    #[arg(long)]
    no_memory: bool,
    /// Disable conversation compaction for an ablation.
    #[arg(long)]
    no_compaction: bool,
    /// Replay provider responses from a collected trace; performs no network I/O.
    #[arg(long)]
    replay_responses: Option<PathBuf>,
    /// MCP server declarations. Defaults to `.knossos/mcp.json`, if present.
    #[arg(long)]
    mcp_config: Option<PathBuf>,
    /// Output-token ceiling per engine turn.
    ///
    /// Worth raising for a reasoning model: thinking is generated before the
    /// answer and counts against the same budget, so a ceiling that looks
    /// generous can be spent entirely on reasoning and truncate the reply
    /// mid-sentence — which the front end reports as an output limit rather
    /// than as "it thought too long".
    #[arg(long, default_value = "8192")]
    max_tokens: u32,
    /// Assign this agent a context window. Defaults to a safe fraction of the
    /// engine/server window discovered at turn time.
    #[arg(long)]
    context_window: Option<u32>,
    /// Compact the working transcript at this many prompt tokens.
    #[arg(long)]
    compact_at: Option<u32>,
    /// Maximum logical provider calls for this task or entire eval run.
    #[arg(long)]
    max_requests: Option<u64>,
    /// Strict input+maximum-output reservation across this task or eval run.
    #[arg(long)]
    max_total_tokens: Option<u64>,
    /// Maximum simultaneous engine calls, including delegated work.
    #[arg(long, default_value = "1")]
    max_concurrency: usize,
    /// Context window for a local model, in tokens.
    ///
    /// The ceiling `--max-tokens` lives inside: prompt and completion share it.
    /// Ollama's own default is 4096 whatever the model declares, which is why
    /// this defaults to something usable instead. Raising it costs VRAM, since
    /// the KV cache scales with it — so this is the knob where your GPU, rather
    /// than a configuration value, is what actually stops you.
    #[arg(long, default_value_t = knossos::engine::ollama::DEFAULT_NUM_CTX)]
    num_ctx: u32,
    /// Tell a reasoning model to answer without reasoning first.
    ///
    /// Reasoning is generated before the answer and billed as output, so it is
    /// paid for on every step. Much of what it buys — choosing an approach,
    /// checking the result — this harness already does with a plan you can read
    /// and a verifier that compiles the code. Worth measuring rather than
    /// assuming: `coding_eval` is what settles whether the quality is worth the
    /// wall-clock.
    #[arg(long)]
    no_think: bool,
    /// Let the agent hand scoped subtasks to child agents.
    ///
    /// Off by default because it spends engine turns: a caller measuring the
    /// loop needs to be able to compare with and without. The win is context,
    /// not speed — a child's reading is discarded when it finishes, so the
    /// parent pays for a paragraph instead of ten files it will never reread.
    #[arg(long)]
    delegate: bool,
    /// Record the full prompt and completion at every engine call.
    ///
    /// Turns the trace from an audit log into training data: each step gains
    /// the exact request sent and the exact reply, which together are an SFT
    /// example, while the verdict and halt events already present say whether
    /// it worked. Costs a great deal of disk — a trace grows roughly with the
    /// square of the run length — so it is off unless a run exists to produce
    /// a corpus.
    #[arg(long)]
    collect_exchanges: bool,
    /// Write the discovered mission environment and explicit recipes as JSON.
    /// Discovery never runs those commands or installs dependencies.
    #[arg(long)]
    environment_export: Option<PathBuf>,
}

#[derive(clap::Args, Clone, Default)]
struct RecoveryArgs {
    /// Retain prompts/tool results in an owner-private portable checkpoint.
    #[arg(long)]
    persist_conversation: bool,
    /// Restore a checkpointed mission before accepting the next turn.
    #[arg(long)]
    resume_mission: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Launch Knossos Field, the local Roman multi-agent command surface.
    Field {
        /// Field installation root. Otherwise use KNOSSOS_FIELD_DIR or discover a sibling bundle.
        #[arg(long)]
        dir: Option<PathBuf>,
    },

    /// One turn against the configured engine. Smoke test for the engine slot.
    Chat { prompt: String },

    /// Build and print Scribe's exact symbol index.
    Index {
        /// Print every symbol rather than a summary.
        #[arg(long)]
        full: bool,
        /// Look up one name and print its exact declarations.
        #[arg(long)]
        lookup: Option<String>,
    },

    /// Run Oracle's deterministic tiers over the workspace.
    Verify,

    /// Produce a plan without executing it.
    Plan { task: String },

    /// Plan and execute a task, verifying as it goes.
    Task {
        task: String,
        #[command(flatten)]
        recovery: RecoveryArgs,
        #[command(flatten)]
        opts: LoopArgs,
    },

    /// Interactive session that keeps context between turns.
    Repl {
        /// Optional first task. Omit it to start at the prompt.
        task: Option<String>,
        #[command(flatten)]
        recovery: RecoveryArgs,
        #[command(flatten)]
        opts: LoopArgs,
    },

    /// Grade coding cases (fail_to_pass / pass_to_pass / restored tests).
    ///
    /// Faithful port of `codeval.py`. Point at a JSON list of cases; each is
    /// materialised, run through Talos, then graded. The agent never sees
    /// held-out tests.
    Eval {
        /// JSON file of cases. Omit to run the bundled core suite.
        #[arg(long)]
        cases: Option<PathBuf>,
        /// Run only these case ids. Repeat for a deterministic canary subset.
        #[arg(long = "case-id")]
        case_ids: Vec<String>,
        /// Run only the first N selected cases.
        #[arg(long)]
        limit: Option<usize>,
        /// Continue after an engine/agent error. By default eval stops so a bad
        /// key, exhausted quota, or dead provider cannot burn the rest of a suite.
        #[arg(long)]
        continue_after_agent_error: bool,
        /// Experiment arm written into every trace (for example
        /// `knossos-full` or `knossos-no-context`).
        #[arg(long, default_value = "unspecified")]
        experiment_arm: String,
        /// Persist completed cases here after every successful grading step.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Resume from --checkpoint, rejecting a different suite or arm.
        #[arg(long, requires = "checkpoint")]
        resume: bool,
        #[command(flatten)]
        opts: LoopArgs,
    },

    /// Speak the Agent Client Protocol on stdio, for Zed and other ACP editors.
    ///
    /// Not meant to be driven by hand. Point an editor's agent configuration at
    /// this — the workspace comes from the editor in `session/new`, so
    /// `--workspace` is ignored here.
    Acp {
        #[command(flatten)]
        recovery: RecoveryArgs,
        #[command(flatten)]
        opts: LoopArgs,
    },

    /// Long-lived NDJSON server for editor front ends.
    ///
    /// Commands in on stdin, events out on stdout, one JSON object per line.
    /// Not meant to be driven by hand — see the VS Code extension.
    Serve {
        #[command(flatten)]
        recovery: RecoveryArgs,
        #[command(flatten)]
        opts: LoopArgs,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    // Folded into the config rather than passed alongside it, because
    // `cfg.max_tokens` is what every call site already reads — planning, the
    // executor, the judge and any child agent. Threading a second value would
    // mean finding all of them and getting one wrong.
    let cfg = Config {
        engine: cli.engine,
        model: cli.model.clone(),
        provider: cli.provider.clone(),
        openai_base_url: cli.base_url.clone(),
        workspace: cli.workspace.clone(),
        max_tokens: match &cli.command {
            Command::Task { opts, .. }
            | Command::Repl { opts, .. }
            | Command::Serve { opts, .. }
            | Command::Acp { opts, .. }
            | Command::Eval { opts, .. } => opts.max_tokens,
            _ => Config::default().max_tokens,
        },
        context_window: match &cli.command {
            Command::Task { opts, .. }
            | Command::Repl { opts, .. }
            | Command::Serve { opts, .. }
            | Command::Acp { opts, .. }
            | Command::Eval { opts, .. } => opts.context_window,
            _ => None,
        },
        compact_at: match &cli.command {
            Command::Task { opts, .. }
            | Command::Repl { opts, .. }
            | Command::Serve { opts, .. }
            | Command::Acp { opts, .. }
            | Command::Eval { opts, .. } => opts.compact_at,
            _ => None,
        },
        ollama_num_ctx: match &cli.command {
            Command::Task { opts, .. }
            | Command::Repl { opts, .. }
            | Command::Serve { opts, .. }
            | Command::Acp { opts, .. }
            | Command::Eval { opts, .. } => Some(opts.num_ctx),
            _ => Config::default().ollama_num_ctx,
        },
        // `None` rather than `Some(true)` when the flag is absent, so the
        // model's own default stands instead of being overridden to match it.
        ollama_think: match &cli.command {
            Command::Task { opts, .. }
            | Command::Repl { opts, .. }
            | Command::Serve { opts, .. }
            | Command::Acp { opts, .. }
            | Command::Eval { opts, .. } => opts.no_think.then_some(false),
            _ => Config::default().ollama_think,
        },
        ..Config::default()
    };

    match cli.command {
        Command::Field { ref dir } => run_field(dir.as_deref()),
        Command::Chat { ref prompt } => chat(&cfg, prompt).await,
        Command::Index { full, ref lookup } => index(&cfg, full, lookup.as_deref()),
        Command::Verify => verify(&cfg).await,
        Command::Plan { ref task } => plan_only(&cfg, task).await,
        Command::Task {
            ref task,
            ref opts,
            ref recovery,
        } => run_task(&cfg, task, opts, recovery).await,
        Command::Repl {
            ref task,
            ref opts,
            ref recovery,
        } => run_repl(&cfg, task.clone(), opts, recovery).await,
        Command::Serve {
            ref opts,
            ref recovery,
        } => run_serve(&cfg, opts, recovery).await,
        Command::Acp {
            ref opts,
            ref recovery,
        } => run_acp(&cfg, opts, recovery).await,
        Command::Eval {
            ref cases,
            ref case_ids,
            limit,
            continue_after_agent_error,
            ref experiment_arm,
            ref checkpoint,
            resume,
            ref opts,
        } => {
            run_eval(
                &cfg,
                cases.as_deref(),
                case_ids,
                limit,
                continue_after_agent_error,
                experiment_arm,
                checkpoint.as_deref(),
                resume,
                opts,
            )
            .await
        }
    }
}

fn field_root(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return validate_field_root(path).with_context(|| {
            format!(
                "--dir does not point to a Knossos Field installation: {}",
                path.display()
            )
        });
    }
    if let Some(path) = std::env::var_os("KNOSSOS_FIELD_DIR").filter(|value| !value.is_empty()) {
        let path = PathBuf::from(path);
        return validate_field_root(&path).with_context(|| {
            format!(
                "KNOSSOS_FIELD_DIR does not point to a Knossos Field installation: {}",
                path.display()
            )
        });
    }

    let executable = std::env::current_exe().context("cannot locate the knossos executable")?;
    let mut candidates = Vec::new();
    for ancestor in executable.ancestors().take(8) {
        candidates.push(ancestor.to_path_buf());
        candidates.push(ancestor.join("field"));
    }
    if let Some(prefix) = executable.parent().and_then(Path::parent) {
        candidates.push(prefix.join("share").join("knossos").join("field"));
    }
    for candidate in candidates {
        if let Ok(root) = validate_field_root(&candidate) {
            return Ok(root);
        }
    }
    bail!(
        "Knossos Field is not installed beside this binary; extract the knossos-field release and run `knossos field --dir <path>`, or set KNOSSOS_FIELD_DIR"
    )
}

fn validate_field_root(path: &Path) -> Result<PathBuf> {
    let package_path = path.join("package.json");
    let server_path = path.join("server").join("src").join("index.js");
    let package: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&package_path)
            .with_context(|| format!("missing {}", package_path.display()))?,
    )
    .with_context(|| format!("invalid {}", package_path.display()))?;
    if package.get("name").and_then(serde_json::Value::as_str) != Some("knossos-field") {
        bail!(
            "{} is not the knossos-field package",
            package_path.display()
        );
    }
    if !server_path.is_file() {
        bail!(
            "missing Field server entry point: {}",
            server_path.display()
        );
    }
    path.canonicalize()
        .with_context(|| format!("cannot resolve Field installation: {}", path.display()))
}

fn run_field(explicit: Option<&Path>) -> Result<()> {
    let root = field_root(explicit)?;
    let dependency = root.join("node_modules").join("yaml");
    if !dependency.is_dir() {
        bail!(
            "Field dependencies are not installed in {}; run `npm ci --omit=dev` there first",
            root.display()
        );
    }
    let node = std::env::var_os("KNOSSOS_NODE_BIN").unwrap_or_else(|| "node".into());
    let executable = std::env::current_exe()
        .context("cannot locate the knossos executable")?
        .canonicalize()
        .context("cannot resolve the knossos executable")?;
    eprintln!("Launching Knossos Field from {}", root.display());
    let status = std::process::Command::new(node)
        .args([
            "--disable-warning=ExperimentalWarning",
            "server/src/index.js",
        ])
        .current_dir(&root)
        .env("FIELD_KNOSSOS_BIN", executable)
        .status()
        .context("failed to start Node.js; install Node 20+ or set KNOSSOS_NODE_BIN")?;
    if !status.success() {
        bail!("Knossos Field exited with {status}");
    }
    Ok(())
}

async fn chat(cfg: &Config, prompt: &str) -> Result<()> {
    let eng = cfg.build_engine()?;
    let req = Request::new(
        "You are Knossos, a precise coding assistant.",
        vec![Message::user_text(prompt)],
    )
    .with_max_tokens(cfg.max_tokens);

    let resp = engine::complete(eng.as_ref(), &req).await?;
    println!("{}", resp.text());
    eprintln!(
        "[{} | in {} out {}]",
        eng.name(),
        resp.usage.input_tokens,
        resp.usage.output_tokens
    );
    Ok(())
}

fn index(cfg: &Config, full: bool, lookup: Option<&str>) -> Result<()> {
    let root = cfg.workspace_root()?;
    let idx = SymbolIndex::build(&root)?;

    if let Some(name) = lookup {
        let hits = idx.lookup(name);
        if hits.is_empty() {
            println!("`{name}` is not declared anywhere in this workspace.");
            std::process::exit(1);
        }
        for s in hits {
            println!(
                "{}:{}: {} {}",
                s.file.display(),
                s.line,
                s.kind.label(),
                s.signature
            );
        }
        return Ok(());
    }

    println!(
        "{} symbols across {} files ({})",
        idx.symbol_count(),
        idx.file_count(),
        idx.adapter().name()
    );
    if full {
        print!("{}", idx.render(usize::MAX));
    }
    Ok(())
}

async fn verify(cfg: &Config) -> Result<()> {
    let root = cfg.workspace_root()?;
    let idx = SymbolIndex::build(&root)?;
    let oracle = Oracle::new(&root);

    // No specific edits to check, so tier 0 covers every indexed file.
    let files: Vec<PathBuf> = idx
        .iter()
        .map(|s| s.file.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    let verdict = oracle.verify(idx.adapter(), &files).await?;
    for tier in &verdict.tiers {
        let mark = if tier.passed { "PASS" } else { "FAIL" };
        println!("[{mark}] tier {} — {}", tier.tier, tier.label);
        if !tier.passed {
            println!("\n{}\n", tier.detail);
        }
    }
    println!("{}", verdict.summary());

    if !verdict.passed {
        std::process::exit(1);
    }
    Ok(())
}

async fn plan_only(cfg: &Config, task: &str) -> Result<()> {
    let root = cfg.workspace_root()?;
    let eng = cfg.build_engine()?;
    let idx = SymbolIndex::build(&root)?;
    let themis = Themis::load(&root);

    let plan = metis::plan(eng.as_ref(), &themis, &idx, task, cfg.max_tokens).await?;
    println!("{}", plan.render());
    Ok(())
}

/// Assemble a Talos and announce the configuration.
///
/// `stream` makes the session mirror every trace event to stdout, which is how
/// `serve` gives a front end live progress. Announcements stay on stderr in
/// every mode, so enabling it cannot corrupt the protocol channel.
fn build_talos(cfg: &Config, opts: &LoopArgs, stream: bool) -> Result<(Talos, PathBuf)> {
    let quota = knossos::engine::budget::Quota::new(
        opts.max_requests,
        opts.max_total_tokens,
        opts.max_concurrency,
    );
    build_talos_with_quota(cfg, opts, stream, quota)
}

fn build_talos_with_quota(
    cfg: &Config,
    opts: &LoopArgs,
    stream: bool,
    quota: Arc<knossos::engine::budget::Quota>,
) -> Result<(Talos, PathBuf)> {
    let root = cfg.workspace_root()?;
    let raw: Box<dyn knossos::engine::Engine> = match &opts.replay_responses {
        Some(path) => Box::new(knossos::engine::mock::MockEngine::from_trace(path)?),
        None => cfg.build_engine()?,
    };
    let eng: Box<dyn knossos::engine::Engine> = Box::new(
        knossos::engine::budget::BudgetedEngine::new(raw, quota.clone()),
    );
    let engine_name = eng.name().to_string();

    let idx = SymbolIndex::build(&root)?;
    let themis = Themis::load(&root);
    // Built before `idx` is moved into Talos; both read the same adapter.
    let retrieval = std::sync::Arc::new(Mnemosyne::build(&root, idx.adapter())?);

    let trace_path = opts.trace.clone().unwrap_or_else(|| {
        root.join(".knossos")
            .join(format!("trace-{}.jsonl", std::process::id()))
    });
    let mut session = Session::new(&root, &engine_name).with_trace(&trace_path)?;
    if stream {
        session = session.streaming();
    }
    if opts.collect_exchanges {
        session = session.collecting();
    }

    // Everything that has something to announce runs before anything is
    // announced, so the block below stays contiguous rather than interleaving
    // with whatever a side-car decided to say while starting.
    let mut registry = ToolRegistry::with_retrieval(retrieval.clone());
    let mcp_report = connect_mcp(&mut registry, &root, opts.mcp_config.as_deref());
    if opts.delegate {
        registry.extend([Box::new(delegate::Delegate::new(
            child_factory(cfg.clone(), root.clone(), trace_path.clone()),
            opts.max_steps,
        )) as Box<dyn knossos::tools::Tool>]);
    }

    let gate = (!opts.no_context).then(|| {
        let mut argus = Argus::new(&root);
        let report = argus.scan();
        (RetrievalGate::new(std::sync::Arc::new(argus)), report)
    });

    eprintln!("engine       {engine_name}");
    eprintln!("workspace    {}", root.display());
    eprintln!("constitution {}", themis.source());
    eprintln!(
        "symbols      {} across {} files",
        idx.symbol_count(),
        idx.file_count()
    );
    eprintln!("retrieval    {} chunks indexed", retrieval.chunk_count());
    match &gate {
        Some((_, report)) => eprintln!("context      {report}"),
        None => eprintln!("context      off (--no-context)"),
    }
    for line in &mcp_report {
        eprintln!("mcp          {line}");
    }
    eprintln!(
        "budget       {} target / {} max",
        opts.target_steps, opts.max_steps
    );
    if opts.dry_run {
        eprintln!("mode         DRY RUN — nothing is written to disk");
    }
    eprintln!("trace        {}\n", trace_path.display());

    let mut ctx = ToolCtx::new(&root);
    if opts.dry_run {
        ctx = ctx.dry_run();
    }

    let mut talos = Talos::new(
        eng,
        registry,
        ctx,
        Oracle::new(&root),
        idx,
        themis,
        Ariadne::new(opts.max_steps, opts.target_steps),
        session,
        cfg.max_tokens,
        !opts.no_judge,
    )
    .with_context_limits(cfg.context_window, cfg.compact_at);
    talos.attach_quota(quota);
    if let Some((gate, _)) = gate {
        talos = talos.with_retrieval(gate);
    }
    if opts.no_memory {
        talos.disable_memory();
        eprintln!("memory       off (--no-memory)");
    }
    if opts.no_compaction {
        talos.disable_compaction();
        eprintln!("compaction   off (--no-compaction)");
    }

    Ok((talos, trace_path))
}

fn recovery_requested(recovery: &RecoveryArgs) -> bool {
    recovery.persist_conversation || recovery.resume_mission.is_some()
}

fn recovery_identity(cfg: &Config, opts: &LoopArgs) -> String {
    format!(
        "{cfg:?};steps={};target={};requests={:?};tokens={:?};concurrency={};memory={};context={};compact={}",
        opts.max_steps,
        opts.target_steps,
        opts.max_requests,
        opts.max_total_tokens,
        opts.max_concurrency,
        !opts.no_memory,
        !opts.no_context,
        !opts.no_compaction,
    )
}

fn ensure_recovery_supported(root: &Path, opts: &LoopArgs) -> Result<()> {
    anyhow::ensure!(
        !opts.delegate
            && opts.mcp_config.is_none()
            && !root.join(".knossos/mcp.json").exists()
            && !root.join(".daedalus/mcp.json").exists(),
        "conversation restore does not yet support MCP or delegated tools"
    );
    anyhow::ensure!(
        !opts.dry_run,
        "preview overlays cannot yet be restored from conversation checkpoints"
    );
    Ok(())
}

/// Speak ACP on stdio until the editor goes away.
///
/// Nothing is printed. stdout is the protocol channel, so every announcement
/// the other commands make goes to stderr here — an editor shows it as agent
/// logs, and a stray line on stdout would corrupt the stream.
async fn prefer_cameo(mut cfg: Config) -> Config {
    if cfg.engine != EngineKind::Anthropic {
        return cfg;
    }
    let Ok(model) = std::env::var("CAMEO_MODEL") else {
        return cfg;
    };
    if model.trim().is_empty() {
        return cfg;
    }
    let base =
        std::env::var("CAMEO_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:9090/v1".into());
    let key = std::env::var("CAMEO_SERVE_KEY").ok();
    let cameo =
        knossos::engine::cameo::CameoEngine::new(model.clone(), base.clone(), key, cfg.max_tokens);
    if cameo.is_resident().await {
        cfg.engine = EngineKind::Cameo;
        cfg.model = Some(model);
        cfg.openai_base_url = Some(base);
    }
    cfg
}

async fn run_acp(cfg: &Config, opts: &LoopArgs, recovery: &RecoveryArgs) -> Result<()> {
    let cfg = prefer_cameo(cfg.clone()).await;
    anyhow::ensure!(
        recovery.resume_mission.is_none(),
        "ACP restores per workspace: start with --persist-conversation and pass resumeMission to session/new"
    );
    eprintln!(
        "knossos acp — {:?}/{} — waiting for an editor",
        cfg.engine,
        cfg.model.as_deref().unwrap_or("default"),
    );

    let opts = opts.clone();
    let recovery_identity = recovery
        .persist_conversation
        .then(|| recovery_identity(&cfg, &opts));
    let scripted = knossos::engine::mock::MockEngine::from_env();
    let write = std::env::var("KNOSSOS_WRITE").ok().as_deref() == Some("1");
    let execute = std::env::var("KNOSSOS_EXECUTE").ok().as_deref() == Some("1");
    let default_dry = scripted.is_some() && !write;
    let _ = execute;

    let build: Arc<acp::BuildAgent> = Arc::new(move |root: &Path, cancel, approver| {
        // The editor names the workspace, so everything is built per session
        // rather than once at startup.
        let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        if recovery_identity.is_some() {
            ensure_recovery_supported(&root, &opts)?;
        }
        let idx = SymbolIndex::build(&root)?;
        let retrieval = std::sync::Arc::new(Mnemosyne::build(&root, idx.adapter())?);
        let trace = root
            .join(".knossos")
            .join(format!("acp-{}.jsonl", std::process::id()));

        let mut registry = ToolRegistry::with_retrieval(retrieval);
        connect_mcp(&mut registry, &root, opts.mcp_config.as_deref());
        if opts.delegate {
            registry.extend([Box::new(delegate::Delegate::new(
                child_factory(cfg.clone(), root.clone(), trace.clone()),
                opts.max_steps,
            )) as Box<dyn knossos::tools::Tool>]);
        }

        let engine: Box<dyn knossos::engine::Engine> =
            if let Some(scripted) = knossos::engine::mock::MockEngine::from_env() {
                Box::new(scripted)
            } else {
                // Cannot await here; Cameo-if-resident is applied in `run_task`.
                cfg.build_engine()?
            };
        let mut ctx = ToolCtx::new(&root);
        if default_dry {
            ctx.set_dry_run(true);
        }
        let mut talos = Talos::new(
            engine,
            registry,
            ctx,
            Oracle::new(&root),
            idx,
            Themis::load(&root),
            Ariadne::new(opts.max_steps, opts.target_steps),
            Session::new(&root, "acp").with_trace(&trace)?,
            cfg.max_tokens,
            !opts.no_judge,
        )
        .with_context_limits(cfg.context_window, cfg.compact_at);
        talos.cancel = cancel;
        talos.approver = Some(approver);
        if let Some(identity) = &recovery_identity {
            talos.enable_conversation_checkpoints(identity.clone())?;
        }
        if !opts.no_context {
            let mut argus = Argus::new(&root);
            argus.scan();
            talos = talos.with_retrieval(RetrievalGate::new(std::sync::Arc::new(argus)));
        }
        Ok(talos)
    });

    let rx = Box::new(std::io::BufReader::new(std::io::stdin()));
    let mut peer = knossos::jsonrpc::Peer::new(rx, Box::new(std::io::stdout()))
        .with_fast_path(acp::is_fast_path);
    // Captured on this thread, which has the runtime. The peer's worker does
    // not, and is what will block on each turn.
    let agent = Arc::new(acp::Agent::new(
        peer.handle(),
        build,
        tokio::runtime::Handle::current(),
    ));

    // On its own thread: `Peer::wait` joins the reader, which parks on stdin
    // for the life of the process, and a runtime worker held that long is one
    // the rest of the process never gets back.
    tokio::task::spawn_blocking(move || {
        peer.start(move |method: &str, params, _| agent.handle(method, params));
        peer.wait();
    })
    .await?;
    Ok(())
}

/// Build the closure that spawns child agents.
///
/// This is where a role becomes a capability. `reviewer` and `investigator` get
/// a registry with no writing tools at all — not a prompt asking them not to
/// write, an inability to. A reviewer that can edit will fix what it finds and
/// report success, which destroys the only thing an independent reviewer was
/// for; asking politely is not a control.
///
/// A child never gets the delegate tool, so `MAX_DEPTH` is enforced by
/// construction rather than by a check the child could reach.
fn child_factory(
    cfg: Config,
    root: PathBuf,
    trace: PathBuf,
) -> std::sync::Arc<delegate::SpawnChild> {
    std::sync::Arc::new(move |req: delegate::ChildRequest| {
        let engine = cfg.build_engine()?;
        let can_write = req.role.name == "general";
        let tools = if can_write {
            ToolRegistry::standard()
        } else {
            ToolRegistry::read_only()
        };

        Ok(Talos::new(
            engine,
            tools,
            req.ctx,
            // Shares the parent's baseline decision, but not its baseline: a
            // child verifies the same workspace, so re-running the ladder to
            // establish what was already broken would cost the same again.
            Oracle::new(&root).without_baseline(),
            SymbolIndex::build(&root)?,
            Themis::load(&root),
            Ariadne::new(req.max_steps, req.max_steps.div_ceil(2)),
            // The same trace file: a child's steps belong in the record of the
            // run that caused them, not in a file nobody knows to open.
            Session::new(&root, "child").with_trace(&trace)?,
            cfg.max_tokens,
            // Tier 4 costs an engine turn, and the parent judges the whole task
            // once the child's work is folded into it.
            false,
        )
        .with_context_limits(cfg.context_window, cfg.compact_at)
        .with_role(req.role.prompt.clone()))
    })
}

/// Connect declared MCP servers and register what they offer.
///
/// Every failure here is reported and survived. A side-car that will not start
/// must not stop the harness from opening: an editor missing a feature is a far
/// better outcome than one that will not launch.
fn connect_mcp(
    registry: &mut ToolRegistry,
    root: &std::path::Path,
    configured: Option<&std::path::Path>,
) -> Vec<String> {
    let path = configured.map(PathBuf::from).unwrap_or_else(|| {
        let canonical = root.join(".knossos").join("mcp.json");
        let legacy = root.join(".daedalus").join("mcp.json");
        if canonical.exists() || !legacy.exists() {
            canonical
        } else {
            legacy
        }
    });
    if !path.exists() {
        // Silence when nobody asked for MCP; an explicit path that is missing is
        // a mistake worth naming.
        return match configured {
            Some(_) => vec![format!("no such file: {}", path.display())],
            None => Vec::new(),
        };
    }

    let declared = match mcp::declarations_from(&path) {
        Ok(declared) => declared,
        Err(err) => return vec![format!("{err:#}")],
    };

    let (clients, errors) = mcp::connect_all(&declared);
    let mut report = errors;
    for client in &clients {
        // A duplicate name would shadow silently, which is the exact bug the
        // server-label prefix exists to prevent — so say so rather than let the
        // second definition disappear.
        let taken: Vec<String> = registry.names().iter().map(|n| n.to_string()).collect();
        let (fresh, clashing): (Vec<_>, Vec<_>) = client
            .tools()
            .into_iter()
            .partition(|t| !taken.iter().any(|n| n == t.name()));
        for tool in &clashing {
            report.push(format!("{} already exists; skipped", tool.name()));
        }
        report.push(format!("{} — {} tool(s)", client.name(), fresh.len()));
        registry.extend(fresh);
    }
    report
}

#[allow(clippy::too_many_arguments)]
async fn run_eval(
    cfg: &Config,
    cases: Option<&Path>,
    case_ids: &[String],
    limit: Option<usize>,
    continue_after_agent_error: bool,
    experiment_arm: &str,
    checkpoint_path: Option<&Path>,
    resume: bool,
    opts: &LoopArgs,
) -> Result<()> {
    let mut cfg = prefer_cameo(cfg.clone()).await;
    cfg.workspace = cfg.workspace_root()?;
    let mut suite = match cases {
        Some(path) => knossos::eval::load_cases(path)?,
        None => knossos::eval::bundled_core()?,
    };
    if !case_ids.is_empty() {
        let available: std::collections::BTreeSet<_> =
            suite.iter().map(|case| case.id.as_str()).collect();
        let missing: Vec<_> = case_ids
            .iter()
            .filter(|id| !available.contains(id.as_str()))
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Err(anyhow::anyhow!(
                "unknown eval case id(s): {}; available: {}",
                missing.join(", "),
                available.into_iter().collect::<Vec<_>>().join(", ")
            ));
        }
        suite.retain(|case| case_ids.iter().any(|id| id == &case.id));
    }
    if let Some(limit) = limit {
        if limit == 0 {
            return Err(anyhow::anyhow!("--limit must be positive"));
        }
        suite.truncate(limit);
    }
    let suite_bytes = match cases {
        Some(path) => std::fs::read(path)?,
        None => include_bytes!("../cases/core.json").to_vec(),
    };
    let suite_digest = format!("sha1:{:x}", Sha1::digest(&suite_bytes));
    let harness_commit = harness_revision();
    let selected_ids: std::collections::BTreeSet<String> =
        suite.iter().map(|case| case.id.clone()).collect();
    let selected_total = selected_ids.len();
    let mut checkpoint = match checkpoint_path {
        Some(path) if resume => Some(knossos::eval_checkpoint::EvalCheckpoint::load(
            path,
            &suite_digest,
            experiment_arm,
        )?),
        Some(_) => Some(knossos::eval_checkpoint::EvalCheckpoint::new(
            &suite_digest,
            experiment_arm,
        )),
        None => None,
    };
    let mut passed = checkpoint
        .as_ref()
        .map(|state| state.passed.intersection(&selected_ids).count())
        .unwrap_or(0);
    if let Some(state) = &checkpoint {
        suite.retain(|case| !state.completed.contains(&case.id));
    }
    let eval_run = format!(
        "{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    );
    let persistent_root = cfg
        .workspace_root()?
        .join(".knossos")
        .join("eval")
        .join(&eval_run);
    std::fs::create_dir_all(&persistent_root)?;
    println!(
        "eval {} case(s) from {}",
        selected_total,
        cases
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "bundled core".into())
    );
    let spent = checkpoint
        .as_ref()
        .map(|state| knossos::engine::budget::QuotaSnapshot {
            requests: state.requests_spent,
            tokens: state.tokens_spent,
        })
        .unwrap_or_default();
    let eval_quota = knossos::engine::budget::Quota::with_spent(
        opts.max_requests,
        opts.max_total_tokens,
        opts.max_concurrency,
        spent,
    );
    for case in &suite {
        let safe_id: String = case
            .id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let root = std::env::temp_dir().join(format!("knossos-eval-{eval_run}-{safe_id}"));
        let _ = std::fs::remove_dir_all(&root);
        knossos::eval::materialise(case, &root)?;
        let mut case_cfg = cfg.clone();
        case_cfg.workspace = root.clone();
        let mut case_opts = opts.clone();
        case_opts.trace = Some(match (&opts.trace, suite.len()) {
            (Some(path), 1) => path.clone(),
            (Some(path), _) => {
                let parent = path.parent().unwrap_or_else(|| Path::new("."));
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("trace");
                parent.join(format!("{stem}-{safe_id}.jsonl"))
            }
            (None, _) => persistent_root.join(format!("{safe_id}.jsonl")),
        });
        let (mut talos, trace_path) =
            build_talos_with_quota(&case_cfg, &case_opts, false, Arc::clone(&eval_quota))?;
        talos.set_require_action(matches!(
            case.expected_action.as_str(),
            "edit" | "edit_preserve"
        ));
        talos.session.log(&TraceEvent::ExperimentMetadata {
            arm: experiment_arm.to_string(),
            case_id: case.id.clone(),
            expected_action: case.expected_action.clone(),
            provider: cfg
                .provider
                .clone()
                .unwrap_or_else(|| format!("{:?}", cfg.engine).to_ascii_lowercase()),
            model: cfg
                .model
                .clone()
                .unwrap_or_else(|| "provider-default".into()),
            harness_commit: harness_commit.clone(),
            suite_digest: suite_digest.clone(),
            max_steps: opts.max_steps,
            max_output_tokens: opts.max_tokens,
            started_at: chrono::Utc::now().to_rfc3339(),
        });
        talos.session.log(&TraceEvent::TaskStarted {
            task: case.prompt.clone(),
            engine: talos.session.engine_name.clone(),
            workspace: root.display().to_string(),
            max_steps: opts.max_steps,
            target_steps: opts.target_steps,
        });
        let result = async {
            let plan = if knossos::metis::worth_planning(&case.prompt) {
                knossos::metis::plan(
                    talos.engine.as_ref(),
                    &talos.themis,
                    &talos.scribe,
                    &case.prompt,
                    case_cfg.max_tokens,
                )
                .await?
            } else {
                knossos::metis::Plan {
                    steps: vec![case.prompt.clone()],
                }
            };
            let outcome = talos.run(&case.prompt, &plan).await?;
            anyhow::Ok(outcome)
        }
        .await;
        if let Err(ref e) = result {
            let engine_error = e.downcast_ref::<knossos::engine::EngineError>();
            let budget_error = matches!(
                engine_error,
                Some(knossos::engine::EngineError::Budget { .. })
            );
            let provider_error = engine_error.is_some() && !budget_error;
            talos.session.log(&TraceEvent::EvaluationFinished {
                case_id: case.id.clone(),
                grader_pass: false,
                verifier_pass: false,
                halt_reason: if budget_error {
                    "budget_exhausted"
                } else if provider_error {
                    "provider_error"
                } else {
                    "harness_error"
                }
                .into(),
                provider_status: if provider_error { "error" } else { "ok" }.into(),
                infrastructure_status: if budget_error {
                    "budget_exhausted"
                } else if provider_error {
                    "not_graded"
                } else {
                    "harness_error"
                }
                .into(),
                tamper: false,
            });
            eprintln!(
                "FAIL {} (agent: {e:#}; trace {})",
                case.id,
                trace_path.display()
            );
            if let Some(state) = &mut checkpoint {
                let spent = eval_quota.snapshot();
                state.requests_spent = spent.requests;
                state.tokens_spent = spent.tokens;
                state.save(checkpoint_path.expect("checkpoint path accompanies state"))?;
            }
            let _ = std::fs::remove_dir_all(&root);
            if !continue_after_agent_error {
                return Err(anyhow::anyhow!(
                    "evaluation stopped after agent/provider error in {}; pass \
                     --continue-after-agent-error only after diagnosing it",
                    case.id
                ));
            }
            continue;
        }
        let outcome = result.expect("handled agent error above");
        let agent_response = talos
            .messages
            .iter()
            .rev()
            .find(|message| message.role == knossos::engine::Role::Assistant)
            .map(|message| message.text());
        let tampered = knossos::eval::restore_tests(case, &root)?;
        let mut grade =
            knossos::eval::grade(case, &root, tampered, agent_response.as_deref()).await?;
        let agent_completed = knossos::eval::completion_verdict(
            &case.expected_action,
            outcome.halt,
            grade.action_passed,
        );
        grade.passed &= agent_completed;
        let failure_kind = if !grade.action_passed {
            "action_violation"
        } else if agent_completed {
            grade.failure_kind()
        } else {
            outcome.halt.label()
        };
        let verifier_pass = knossos::eval::verdict(
            grade.fixed,
            grade.fixed_total,
            grade.kept,
            grade.kept_total,
            grade.held,
            grade.held_total,
        );
        let infrastructure_status = if grade.unverifiable > 0 {
            "unverifiable"
        } else if grade.timed_out > 0 {
            "timeout"
        } else {
            "ok"
        };
        talos.session.log(&TraceEvent::EvaluationFinished {
            case_id: case.id.clone(),
            grader_pass: grade.passed,
            verifier_pass,
            halt_reason: if grade.passed { "done" } else { failure_kind }.into(),
            provider_status: "ok".into(),
            infrastructure_status: infrastructure_status.into(),
            tamper: grade.tamper,
        });
        if grade.passed {
            passed += 1;
            println!(
                "PASS {}  fixed {}/{} kept {}/{} held {}/{} tamper={} trace={}",
                case.id,
                grade.fixed,
                grade.fixed_total,
                grade.kept,
                grade.kept_total,
                grade.held,
                grade.held_total,
                grade.tamper,
                trace_path.display()
            );
        } else {
            println!(
                "FAIL {}  fixed {}/{} kept {}/{} held {}/{} tamper={} reason={} timeout={} unverifiable={} trace={}",
                case.id,
                grade.fixed,
                grade.fixed_total,
                grade.kept,
                grade.kept_total,
                grade.held,
                grade.held_total,
                grade.tamper,
                failure_kind,
                grade.timed_out,
                grade.unverifiable,
                trace_path.display()
            );
        }
        if let Some(state) = &mut checkpoint {
            state.completed.insert(case.id.clone());
            if grade.passed {
                state.passed.insert(case.id.clone());
            } else {
                state.passed.remove(&case.id);
            }
            let spent = eval_quota.snapshot();
            state.requests_spent = spent.requests;
            state.tokens_spent = spent.tokens;
            state.save(checkpoint_path.expect("checkpoint path accompanies state"))?;
        }
        let _ = std::fs::remove_dir_all(&root);
    }
    println!("{passed}/{selected_total} passed");
    if passed != selected_total {
        std::process::exit(1);
    }
    Ok(())
}

fn harness_revision() -> String {
    if let Ok(value) = std::env::var("KNOSSOS_HARNESS_COMMIT") {
        if !value.trim().is_empty() {
            return value;
        }
    }
    let root = env!("CARGO_MANIFEST_DIR");
    let head = std::process::Command::new("git")
        .args(["-C", root, "rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".into());
    let dirty = std::process::Command::new("git")
        .args(["-C", root, "status", "--porcelain"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .is_some_and(|output| !output.stdout.is_empty());
    if dirty {
        format!("{head}+dirty")
    } else {
        head
    }
}

async fn run_task(
    cfg: &Config,
    task: &str,
    opts: &LoopArgs,
    recovery: &RecoveryArgs,
) -> Result<()> {
    let cfg = prefer_cameo(cfg.clone()).await;
    let root = cfg.workspace_root()?;
    if recovery_requested(recovery) {
        ensure_recovery_supported(&root, opts)?;
    }
    let (mut talos, trace_path) = build_talos(&cfg, opts, false)?;
    if recovery_requested(recovery) {
        talos.enable_conversation_checkpoints(recovery_identity(&cfg, opts))?;
    }
    // This occurs before Metis consumes any prompt budget. The record says
    // plainly that this task runs on the host; it is not a sandbox claim.
    let environment = EnvironmentFingerprint::discover(talos.ctx.root())?;
    if let Some(path) = &opts.environment_export {
        environment.export(path)?;
        eprintln!("Environment recipe exported to {}", path.display());
    }
    talos.set_environment(environment.into_record());

    talos.session.log(&TraceEvent::TaskStarted {
        task: task.to_string(),
        engine: talos.session.engine_name.clone(),
        workspace: talos.oracle.root().display().to_string(),
        max_steps: opts.max_steps,
        target_steps: opts.target_steps,
    });

    let outcome = if let Some(mission_id) = recovery.resume_mission.as_deref() {
        talos.restore_conversation(mission_id)?;
        talos.resume(task).await?
    } else {
        let plan = metis::plan(
            talos.engine.as_ref(),
            &talos.themis,
            &talos.scribe,
            task,
            cfg.max_tokens,
        )
        .await?;
        eprintln!("Plan:\n{}\n", plan.render());

        talos.run(task, &plan).await?
    };
    if recovery_requested(recovery) {
        eprintln!(
            "Conversation checkpoint retained for mission {}",
            talos.mission_id().unwrap_or("unknown")
        );
    }

    println!("\n{}", outcome.summary);

    if outcome.dry_run {
        println!("\n{}", diff::render(&talos.diffs()));
        println!(
            "Nothing was written. Re-run without --dry-run to apply, or use `repl` to \
                  review and /apply interactively."
        );
    } else if !outcome.changed.is_empty() {
        println!("\nChanged {} file(s):", outcome.changed.len());
        for p in &outcome.changed {
            println!("  {}", talos.ctx.display(p));
        }
    }

    eprintln!(
        "\n[{} in {} step(s); trace: {}]",
        outcome.halt.label(),
        outcome.steps_used,
        trace_path.display()
    );

    if !outcome.succeeded() {
        std::process::exit(1);
    }
    Ok(())
}

async fn run_serve(cfg: &Config, opts: &LoopArgs, recovery: &RecoveryArgs) -> Result<()> {
    let cfg = prefer_cameo(cfg.clone()).await;
    let root = cfg.workspace_root()?;
    if recovery_requested(recovery) {
        ensure_recovery_supported(&root, opts)?;
    }
    let (mut talos, _) = build_talos(&cfg, opts, true)?;
    if recovery_requested(recovery) {
        talos.enable_conversation_checkpoints(recovery_identity(&cfg, opts))?;
    }

    // The real streams live out here, and only here. `serve::run` takes
    // channels so the protocol can be driven from a test — which is what makes
    // the permission handshake provably non-deadlocking rather than hoped
    // about; see `tests/serve_loop.rs`.
    let (line_tx, line_rx) = tokio::sync::mpsc::unbounded_channel();
    let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();

    // A dedicated OS thread, not `spawn_blocking`: this read blocks for the
    // whole life of the process, and a blocking-pool slot held that long is a
    // slot the rest of the runtime never gets back.
    std::thread::spawn(move || {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        let mut line = String::new();
        loop {
            line.clear();
            match stdin.lock().read_line(&mut line) {
                Ok(0) | Err(_) => break, // front end closed the pipe
                Ok(_) => {
                    if line_tx.send(std::mem::take(&mut line)).is_err() {
                        break; // server has stopped
                    }
                }
            }
        }
    });

    let writer = tokio::spawn(knossos::serve::write_events(event_rx, std::io::stdout()));

    let result = knossos::serve::run_with_restore(
        talos,
        cfg.max_tokens,
        line_rx,
        knossos::serve::Emitter::new(event_tx),
        recovery.resume_mission.as_deref(),
    )
    .await;

    // Dropping the emitter closes the channel, so this drains what is left and
    // returns rather than hanging — the last events of a session still reach
    // the front end.
    let _ = writer.await;
    result
}

async fn run_repl(
    cfg: &Config,
    task: Option<String>,
    opts: &LoopArgs,
    recovery: &RecoveryArgs,
) -> Result<()> {
    let cfg = prefer_cameo(cfg.clone()).await;
    let root = cfg.workspace_root()?;
    if recovery_requested(recovery) {
        ensure_recovery_supported(&root, opts)?;
    }
    let (mut talos, _) = build_talos(&cfg, opts, false)?;
    if recovery_requested(recovery) {
        talos.enable_conversation_checkpoints(recovery_identity(&cfg, opts))?;
    }

    talos.session.log(&TraceEvent::TaskStarted {
        task: task.clone().unwrap_or_else(|| "(interactive)".to_string()),
        engine: talos.session.engine_name.clone(),
        workspace: talos.oracle.root().display().to_string(),
        max_steps: opts.max_steps,
        target_steps: opts.target_steps,
    });

    repl::run(
        talos,
        task,
        cfg.max_tokens,
        recovery.resume_mission.as_deref(),
    )
    .await
}

fn init_tracing(verbose: bool) {
    let filter = if verbose {
        "knossos=debug"
    } else {
        "knossos=info"
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()),
        )
        .with_target(false)
        .without_time()
        .with_writer(std::io::stderr)
        .init();
}

#[cfg(test)]
mod field_tests {
    use super::*;

    fn fixture() -> tempfile::TempDir {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(temp.path().join("server/src")).unwrap();
        std::fs::write(
            temp.path().join("package.json"),
            r#"{"name":"knossos-field"}"#,
        )
        .unwrap();
        std::fs::write(temp.path().join("server/src/index.js"), "").unwrap();
        temp
    }

    #[test]
    fn explicit_field_root_requires_the_product_marker_and_entry_point() {
        let valid = fixture();
        assert_eq!(
            field_root(Some(valid.path())).unwrap(),
            valid.path().canonicalize().unwrap()
        );

        let invalid = tempfile::tempdir().unwrap();
        std::fs::write(
            invalid.path().join("package.json"),
            r#"{"name":"someone-elses-field"}"#,
        )
        .unwrap();
        assert!(field_root(Some(invalid.path())).is_err());
    }

    #[test]
    fn field_subcommand_accepts_an_explicit_bundle() {
        let cli = Cli::try_parse_from(["knossos", "field", "--dir", "bundle"]).unwrap();
        match cli.command {
            Command::Field { dir } => assert_eq!(dir, Some(PathBuf::from("bundle"))),
            _ => panic!("field subcommand parsed as a different command"),
        }
    }

    #[test]
    fn durable_recovery_flags_are_available_on_every_interactive_interface() {
        for args in [
            vec!["knossos", "task", "work", "--persist-conversation"],
            vec!["knossos", "repl", "--persist-conversation"],
            vec!["knossos", "serve", "--persist-conversation"],
            vec!["knossos", "acp", "--persist-conversation"],
        ] {
            let cli = Cli::try_parse_from(args).expect("recovery flags should parse");
            let recovery = match cli.command {
                Command::Task { recovery, .. }
                | Command::Repl { recovery, .. }
                | Command::Serve { recovery, .. }
                | Command::Acp { recovery, .. } => recovery,
                _ => panic!("interactive command parsed as a different command"),
            };
            assert!(recovery.persist_conversation);
        }

        let cli = Cli::try_parse_from(["knossos", "serve", "--resume-mission", "mission-123"])
            .expect("serve restore flag should parse");
        let Command::Serve { recovery, .. } = cli.command else {
            panic!("serve parsed as a different command")
        };
        assert_eq!(recovery.resume_mission.as_deref(), Some("mission-123"));
    }
}
