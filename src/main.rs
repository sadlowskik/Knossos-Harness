use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use daedalus_harness::ariadne::Ariadne;
use daedalus_harness::config::{Config, EngineKind};
use daedalus_harness::engine::{self, Message, Request};
use daedalus_harness::metis;
use daedalus_harness::oracle::Oracle;
use daedalus_harness::scribe::SymbolIndex;
use daedalus_harness::session::{Session, TraceEvent};
use daedalus_harness::talos::Talos;
use daedalus_harness::themis::Themis;
use daedalus_harness::tools::{ToolCtx, ToolRegistry};

#[derive(Parser)]
#[command(
    name = "daedalus",
    version,
    about = "An agentic coding harness shaped by the Daedalus architecture",
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

    #[arg(long, short = 'e', global = true, value_enum, default_value = "anthropic")]
    engine: EngineKind,

    /// Model id. Defaults per engine; also read from DAEDALUS_MODEL.
    #[arg(long, short = 'm', global = true)]
    model: Option<String>,

    #[arg(long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Command {
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
        /// Hard ceiling on engine turns.
        #[arg(long, default_value = "12")]
        max_steps: usize,
        /// Where budget pressure begins.
        #[arg(long, default_value = "6")]
        target_steps: usize,
        /// JSONL trajectory log. Defaults to .daedalus/trace-<pid>.jsonl.
        #[arg(long)]
        trace: Option<PathBuf>,
        /// Skip Oracle tier 4 (model judgement against the constitution).
        #[arg(long)]
        no_judge: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    let cfg = Config {
        engine: cli.engine,
        model: cli.model.clone(),
        workspace: cli.workspace.clone(),
        ..Config::default()
    };

    match cli.command {
        Command::Chat { ref prompt } => chat(&cfg, prompt).await,
        Command::Index { full, ref lookup } => index(&cfg, full, lookup.as_deref()),
        Command::Verify => verify(&cfg).await,
        Command::Plan { ref task } => plan_only(&cfg, task).await,
        Command::Task { ref task, max_steps, target_steps, ref trace, no_judge } => {
            run_task(&cfg, task, max_steps, target_steps, trace.clone(), !no_judge).await
        }
    }
}

async fn chat(cfg: &Config, prompt: &str) -> Result<()> {
    let eng = cfg.build_engine()?;
    let req = Request::new(
        "You are Daedalus, a precise coding assistant.",
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
            println!("{}:{}: {} {}", s.file.display(), s.line, s.kind.label(), s.signature);
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

async fn run_task(
    cfg: &Config,
    task: &str,
    max_steps: usize,
    target_steps: usize,
    trace: Option<PathBuf>,
    judge: bool,
) -> Result<()> {
    let root = cfg.workspace_root()?;
    let eng = cfg.build_engine()?;
    let engine_name = eng.name().to_string();

    let idx = SymbolIndex::build(&root)?;
    let themis = Themis::load(&root);
    let ariadne = Ariadne::new(max_steps, target_steps);

    let trace_path = trace.unwrap_or_else(|| {
        root.join(".daedalus")
            .join(format!("trace-{}.jsonl", std::process::id()))
    });
    let session = Session::new(&root, &engine_name).with_trace(&trace_path)?;

    session.log(&TraceEvent::TaskStarted {
        task: task.to_string(),
        engine: engine_name.clone(),
        workspace: root.display().to_string(),
        max_steps,
        target_steps,
    });

    eprintln!("engine      {engine_name}");
    eprintln!("workspace   {}", root.display());
    eprintln!("constitution {}", themis.source());
    eprintln!("symbols     {} across {} files", idx.symbol_count(), idx.file_count());
    eprintln!("budget      {target_steps} target / {max_steps} max");
    eprintln!("trace       {}\n", trace_path.display());

    let plan = metis::plan(eng.as_ref(), &themis, &idx, task, cfg.max_tokens).await?;
    eprintln!("Plan:\n{}\n", plan.render());

    let mut talos = Talos {
        engine: eng,
        tools: ToolRegistry::standard(),
        ctx: ToolCtx::new(&root),
        oracle: Oracle::new(&root),
        scribe: idx,
        themis,
        ariadne,
        session,
        max_tokens: cfg.max_tokens,
        judge,
    };

    let outcome = talos.run(task, &plan).await?;

    println!("\n{}", outcome.summary);
    if !outcome.changed.is_empty() {
        println!("\nChanged {} file(s):", outcome.changed.len());
        for p in &outcome.changed {
            println!("  {}", p.display());
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

fn init_tracing(verbose: bool) {
    let filter = if verbose { "daedalus_harness=debug" } else { "daedalus_harness=info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| filter.into()),
        )
        .with_target(false)
        .without_time()
        .with_writer(std::io::stderr)
        .init();
}
