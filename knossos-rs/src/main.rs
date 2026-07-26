use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use knossos::ariadne::Ariadne;
use knossos::config::{Config, EngineKind};
use knossos::engine::{self, Message, Request};
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

/// Options shared by `task` and `repl`.
#[derive(clap::Args, Clone)]
struct LoopArgs {
    /// Hard ceiling on engine turns.
    #[arg(long, default_value = "12")]
    max_steps: usize,
    /// Where budget pressure begins.
    #[arg(long, default_value = "6")]
    target_steps: usize,
    /// JSONL trajectory log. Defaults to .daedalus/trace-<pid>.jsonl.
    #[arg(long)]
    trace: Option<PathBuf>,
    /// Stage edits in memory and show diffs instead of writing to disk.
    #[arg(long)]
    dry_run: bool,
    /// Skip Oracle tier 4 (model judgement against the constitution).
    #[arg(long)]
    no_judge: bool,
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
        #[command(flatten)]
        opts: LoopArgs,
    },

    /// Interactive session that keeps context between turns.
    Repl {
        /// Optional first task. Omit it to start at the prompt.
        task: Option<String>,
        #[command(flatten)]
        opts: LoopArgs,
    },

    /// Long-lived NDJSON server for editor front ends.
    ///
    /// Commands in on stdin, events out on stdout, one JSON object per line.
    /// Not meant to be driven by hand — see the VS Code extension.
    Serve {
        #[command(flatten)]
        opts: LoopArgs,
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
        Command::Task { ref task, ref opts } => run_task(&cfg, task, opts).await,
        Command::Repl { ref task, ref opts } => run_repl(&cfg, task.clone(), opts).await,
        Command::Serve { ref opts } => run_serve(&cfg, opts).await,
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

/// Assemble a Talos and announce the configuration.
///
/// `stream` makes the session mirror every trace event to stdout, which is how
/// `serve` gives a front end live progress. Announcements stay on stderr in
/// every mode, so enabling it cannot corrupt the protocol channel.
fn build_talos(cfg: &Config, opts: &LoopArgs, stream: bool) -> Result<(Talos, PathBuf)> {
    let root = cfg.workspace_root()?;
    let eng = cfg.build_engine()?;
    let engine_name = eng.name().to_string();

    let idx = SymbolIndex::build(&root)?;
    let themis = Themis::load(&root);
    // Built before `idx` is moved into Talos; both read the same adapter.
    let retrieval = std::sync::Arc::new(Mnemosyne::build(&root, idx.adapter())?);

    let trace_path = opts.trace.clone().unwrap_or_else(|| {
        root.join(".daedalus")
            .join(format!("trace-{}.jsonl", std::process::id()))
    });
    let mut session = Session::new(&root, &engine_name).with_trace(&trace_path)?;
    if stream {
        session = session.streaming();
    }

    eprintln!("engine       {engine_name}");
    eprintln!("workspace    {}", root.display());
    eprintln!("constitution {}", themis.source());
    eprintln!("symbols      {} across {} files", idx.symbol_count(), idx.file_count());
    eprintln!("retrieval    {} chunks indexed", retrieval.chunk_count());
    eprintln!("budget       {} target / {} max", opts.target_steps, opts.max_steps);
    if opts.dry_run {
        eprintln!("mode         DRY RUN — nothing is written to disk");
    }
    eprintln!("trace        {}\n", trace_path.display());

    let mut ctx = ToolCtx::new(&root);
    if opts.dry_run {
        ctx = ctx.dry_run();
    }

    let talos = Talos::new(
        eng,
        ToolRegistry::with_retrieval(retrieval),
        ctx,
        Oracle::new(&root),
        idx,
        themis,
        Ariadne::new(opts.max_steps, opts.target_steps),
        session,
        cfg.max_tokens,
        !opts.no_judge,
    );

    Ok((talos, trace_path))
}

async fn run_task(cfg: &Config, task: &str, opts: &LoopArgs) -> Result<()> {
    let (mut talos, trace_path) = build_talos(cfg, opts, false)?;

    talos.session.log(&TraceEvent::TaskStarted {
        task: task.to_string(),
        engine: talos.session.engine_name.clone(),
        workspace: talos.oracle.root().display().to_string(),
        max_steps: opts.max_steps,
        target_steps: opts.target_steps,
    });

    let plan = metis::plan(
        talos.engine.as_ref(),
        &talos.themis,
        &talos.scribe,
        task,
        cfg.max_tokens,
    )
    .await?;
    eprintln!("Plan:\n{}\n", plan.render());

    let outcome = talos.run(task, &plan).await?;

    println!("\n{}", outcome.summary);

    if outcome.dry_run {
        println!("\n{}", diff::render(&talos.diffs()));
        println!("Nothing was written. Re-run without --dry-run to apply, or use `repl` to \
                  review and /apply interactively.");
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

async fn run_serve(cfg: &Config, opts: &LoopArgs) -> Result<()> {
    let (talos, _) = build_talos(cfg, opts, true)?;
    knossos::serve::run(talos, cfg.max_tokens).await
}

async fn run_repl(cfg: &Config, task: Option<String>, opts: &LoopArgs) -> Result<()> {
    let (talos, _) = build_talos(cfg, opts, false)?;

    talos.session.log(&TraceEvent::TaskStarted {
        task: task.clone().unwrap_or_else(|| "(interactive)".to_string()),
        engine: talos.session.engine_name.clone(),
        workspace: talos.oracle.root().display().to_string(),
        max_steps: opts.max_steps,
        target_steps: opts.target_steps,
    });

    repl::run(talos, task, cfg.max_tokens).await
}

fn init_tracing(verbose: bool) {
    let filter = if verbose { "knossos=debug" } else { "knossos=info" };
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
