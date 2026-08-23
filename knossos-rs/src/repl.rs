//! Interactive session.
//!
//! The difference between a script you run carefully and a tool you use daily
//! is being able to say "no, do it differently" without losing everything the
//! agent already worked out. [`Talos::resume`] keeps the conversation, the
//! symbol index and the staged changes; this module is the front end for it.

use std::io::{BufRead, Write};

use anyhow::Result;

use crate::diff;
use crate::metis;
use crate::talos::Talos;

const HELP: &str = "\
Commands:
  /help              show this
  /diff              show staged changes (dry-run only)
  /apply             write staged changes to disk
  /discard           throw staged changes away
  /verify            run the verification ladder now
  /index [name]      symbol counts, or look one up
  /plan <task>       plan without executing
  /reset             clear the conversation, keep the workspace
  /steps <n>         change the step ceiling
  /quit              leave

Anything else is sent to the agent as an instruction.";

/// An approval waiting for the next line the user types.
type Pending = std::sync::Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Sender<bool>>>>;

/// Asks on the terminal before a consequential call.
///
/// This used to read stdin directly, and could, because the REPL read a command
/// and *then* ran it: stdin was idle for the whole turn with nobody to race for
/// it. That is exactly what stopped the REPL from accepting interjections — a
/// second reader would have raced this one, and a `y` meant for a permission
/// prompt could have been swallowed as a message to the agent, or the reverse.
///
/// So it no longer reads anything. One reader owns stdin (see [`route`]) and
/// hands the answer over, the same shape `serve` has always used.
///
/// Two deliberate choices, both failing closed:
///
/// * **EOF denies.** Piped or redirected input that runs out must not read as
///   approval. It also means a scripted REPL session that never anticipated the
///   prompt refuses the write rather than performing it unattended.
/// * **Anything that is not an explicit yes denies.** No default-accept on a
///   bare newline, because a user pressing enter to get their prompt back is not
///   consenting to a write.
struct PromptApprover {
    pending: Pending,
}

#[async_trait::async_trait]
impl crate::talos::Approver for PromptApprover {
    async fn approve(&self, tool: &str, input: &serde_json::Value) -> bool {
        let (tx, rx) = tokio::sync::oneshot::channel();

        {
            let mut slot = self.pending.lock().unwrap();
            if slot.is_some() {
                // Tool calls are dispatched one at a time, so this should not
                // happen. If it ever does, overwriting would strand the first
                // request forever; refusing the second is recoverable.
                return false;
            }
            *slot = Some(tx);
        }

        println!("\n  {}", describe(tool, input));
        print!("  allow? [y/N] ");
        let _ = std::io::stdout().flush();

        // The router drops or answers the sender. A dropped one — stdin closed,
        // router gone — resolves to an error, which is a refusal.
        rx.await.unwrap_or(false)
    }
}

/// Refuse anything still waiting. Called when stdin goes away.
///
/// Without it the approver awaits a sender nobody holds any more, `talos.run`
/// never returns, and the session hangs at exactly the moment the user pressed
/// Ctrl-D to leave.
fn deny_outstanding(pending: &Pending) {
    if let Some(tx) = pending.lock().unwrap().take() {
        let _ = tx.send(false);
    }
}

/// Read lines and decide what each one is for.
///
/// Three destinations, in this order, and the order is the whole design:
///
/// 1. **A waiting approval.** If one is outstanding, the next line answers it —
///    including a line that looks like a command. Someone who types `/quit` at
///    `allow? [y/N]` has not approved the write, and treating it as a command
///    would leave the agent waiting for an answer that already went elsewhere.
/// 2. **The running agent.** Anything typed while a turn is in flight is a
///    message to it, delivered at the next step boundary rather than ending the
///    run. See [`interject`](crate::interject).
/// 3. **The command loop**, when nothing else is going on.
async fn route(
    mut lines: tokio::sync::mpsc::UnboundedReceiver<String>,
    commands: tokio::sync::mpsc::UnboundedSender<String>,
    pending: Pending,
    busy: std::sync::Arc<std::sync::atomic::AtomicBool>,
    interjections: crate::interject::Interjections,
) {
    while let Some(raw) = lines.recv().await {
        let line = clean(&raw);

        if let Some(tx) = pending.lock().unwrap().take() {
            let _ = tx.send(matches!(line.as_str(), "y" | "Y" | "yes" | "Yes"));
            continue;
        }

        if line.is_empty() {
            continue;
        }

        if busy.load(std::sync::atomic::Ordering::Relaxed) {
            if interjections.push(line.clone()) {
                println!("  (queued — the agent will see it at the next step)");
            } else {
                println!("  (not queued — too many are already waiting)");
            }
            continue;
        }

        if commands.send(line).is_err() {
            return; // the command loop is gone
        }
    }

    deny_outstanding(&pending);
}

/// One line describing what is about to happen, for the prompt.
///
/// The path matters more than the arguments blob: "allow?" over a raw JSON dump
/// is a question nobody reads carefully, and a prompt people stop reading is
/// worse than no prompt at all.
fn describe(tool: &str, input: &serde_json::Value) -> String {
    match tool {
        "write_file" | "edit_file" => format!(
            "{tool} {}",
            input.get("path").and_then(|v| v.as_str()).unwrap_or("?")
        ),
        "run" => format!(
            "run {}",
            input.get("command").and_then(|v| v.as_str()).unwrap_or("?")
        ),
        _ => format!("{tool} {input}"),
    }
}

pub async fn run(mut talos: Talos, initial: Option<String>, max_tokens: u32) -> Result<()> {
    let pending: Pending = Default::default();
    let busy = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    talos.approver = Some(std::sync::Arc::new(PromptApprover {
        pending: pending.clone(),
    }));

    // One reader owns stdin for the whole session. On its own thread because
    // `read_line` blocks for as long as the user takes to type, which is not
    // something to do to an async worker.
    let (line_tx, line_rx) = tokio::sync::mpsc::unbounded_channel();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        loop {
            let mut buf = String::new();
            match stdin.lock().read_line(&mut buf) {
                // EOF (Ctrl-D, or piped input ending) or a broken terminal.
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if line_tx.send(buf).is_err() {
                        break; // the router is gone
                    }
                }
            }
        }
    });

    let (cmd_tx, mut cmd_rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(route(
        line_rx,
        cmd_tx,
        pending.clone(),
        busy.clone(),
        talos.interjections(),
    ));

    println!("Daedalus interactive session. /help for commands, /quit to leave.");
    println!("Type while it is working to steer it without stopping it.");
    if talos.ctx.is_dry_run() {
        println!("DRY RUN — nothing will be written until you /apply.");
    }
    println!();

    if let Some(task) = initial {
        busy.store(true, std::sync::atomic::Ordering::Relaxed);
        let started = first_task(&mut talos, &task, max_tokens).await;
        busy.store(false, std::sync::atomic::Ordering::Relaxed);
        started?;
    }

    loop {
        print!("daedalus> ");
        std::io::stdout().flush()?;

        // Already cleaned and non-empty: the router does both, because it has
        // to make the same judgement about a line it routes elsewhere. A UTF-8
        // BOM is not whitespace, so `trim` leaves it in place; piped input on
        // Windows routinely carries one, and left alone it turns `/quit` into
        // an instruction for the engine.
        let Some(line) = cmd_rx.recv().await else {
            println!();
            break; // stdin closed
        };
        let line = line.as_str();

        if let Some(rest) = line.strip_prefix('/') {
            let (cmd, arg) = split_command(rest);
            match cmd {
                "quit" | "exit" | "q" => break,
                "help" | "h" | "?" => println!("{HELP}"),
                "diff" => show_diff(&talos),
                "apply" => apply(&mut talos)?,
                "discard" => {
                    talos.discard();
                    println!("Staged changes discarded.");
                }
                "verify" => verify(&mut talos).await?,
                "index" => index(&talos, arg),
                "plan" => plan_only(&talos, arg, max_tokens).await?,
                "reset" => {
                    talos.messages.clear();
                    talos.changed.clear();
                    println!("Conversation cleared.");
                }
                "steps" => set_steps(&mut talos, arg),
                other => println!("Unknown command `/{other}`. /help for the list."),
            }
            continue;
        }

        // An engine failure must not end the session: staged changes and the
        // whole conversation would go with it. Report and keep the prompt.
        //
        // `busy` is what tells the router that a line typed from here on is for
        // the agent rather than for the command loop. Set around the whole
        // turn, including the planning call, since that is time the user spends
        // watching too.
        busy.store(true, std::sync::atomic::Ordering::Relaxed);
        let result = if talos.messages.is_empty() {
            first_task(&mut talos, line, max_tokens).await
        } else {
            match talos.resume(line).await {
                Ok(outcome) => {
                    report(&talos, &outcome);
                    Ok(())
                }
                Err(e) => Err(e),
            }
        };
        busy.store(false, std::sync::atomic::Ordering::Relaxed);

        if let Err(e) = result {
            eprintln!("\nEngine error: {e:#}\n");
        }
    }

    if !talos.diffs().is_empty() {
        println!(
            "\nWarning: {} staged change(s) were never applied.",
            talos.diffs().len()
        );
    }
    Ok(())
}

/// Where each line the user types ends up.
#[cfg(test)]
mod routing {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    struct Harness {
        lines: tokio::sync::mpsc::UnboundedSender<String>,
        commands: tokio::sync::mpsc::UnboundedReceiver<String>,
        pending: Pending,
        busy: Arc<AtomicBool>,
        interjections: crate::interject::Interjections,
    }

    fn start() -> Harness {
        let (line_tx, line_rx) = tokio::sync::mpsc::unbounded_channel();
        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel();
        let pending: Pending = Default::default();
        let busy = Arc::new(AtomicBool::new(false));
        let interjections = crate::interject::Interjections::new();

        tokio::spawn(route(
            line_rx,
            cmd_tx,
            pending.clone(),
            busy.clone(),
            interjections.clone(),
        ));

        Harness {
            lines: line_tx,
            commands: cmd_rx,
            pending,
            busy,
            interjections,
        }
    }

    impl Harness {
        fn send(&self, line: &str) {
            self.lines.send(format!("{line}\n")).unwrap();
        }

        /// Register an approval the way `PromptApprover` does.
        fn ask_permission(&self) -> tokio::sync::oneshot::Receiver<bool> {
            let (tx, rx) = tokio::sync::oneshot::channel();
            *self.pending.lock().unwrap() = Some(tx);
            rx
        }

        /// Bounded wait, so a routing bug fails instead of hanging the suite.
        async fn eventually(&self, mut done: impl FnMut(&Harness) -> bool) -> bool {
            for _ in 0..200 {
                if done(self) {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            false
        }
    }

    #[tokio::test]
    async fn an_idle_line_reaches_the_command_loop() {
        let mut h = start();
        h.send("/diff");
        assert_eq!(h.commands.recv().await.unwrap(), "/diff");
    }

    #[tokio::test]
    async fn a_line_typed_while_the_agent_works_becomes_an_interjection() {
        let h = start();
        h.busy.store(true, Ordering::Relaxed);
        h.send("actually use the existing helper");

        assert!(
            h.eventually(|h| h.interjections.len() == 1).await,
            "a line typed during a run should reach the agent"
        );
    }

    #[tokio::test]
    async fn a_waiting_approval_takes_the_next_line_before_anything_else() {
        let h = start();
        let rx = h.ask_permission();
        h.send("y");
        assert!(rx.await.unwrap(), "`y` should approve");
    }

    /// The reason the approval check runs first. Someone typing `/quit` at
    /// `allow? [y/N]` has not approved the write, and routing it to the command
    /// loop would leave the agent waiting for an answer that went elsewhere.
    #[tokio::test]
    async fn a_command_typed_at_the_permission_prompt_answers_it_and_denies() {
        let mut h = start();
        let rx = h.ask_permission();
        h.send("/quit");

        assert!(!rx.await.unwrap(), "anything that is not yes must deny");
        assert!(
            h.commands.try_recv().is_err(),
            "the line answered the prompt; it must not also run as a command"
        );
    }

    #[tokio::test]
    async fn only_an_explicit_yes_approves() {
        for answer in ["", "n", "no", "sure", "Y E S"] {
            let h = start();
            let rx = h.ask_permission();
            h.send(answer);
            assert!(!rx.await.unwrap(), "{answer:?} must not approve");
        }
        for answer in ["y", "Y", "yes", "Yes"] {
            let h = start();
            let rx = h.ask_permission();
            h.send(answer);
            assert!(rx.await.unwrap(), "{answer:?} should approve");
        }
    }

    /// Fail closed. Without this the approver waits on a sender nobody holds,
    /// and the session hangs at the moment the user pressed Ctrl-D to leave.
    #[tokio::test]
    async fn stdin_closing_denies_an_outstanding_approval() {
        let h = start();
        let rx = h.ask_permission();
        drop(h.lines); // EOF

        assert!(!rx.await.unwrap(), "EOF is not approval");
    }

    #[tokio::test]
    async fn an_approval_does_not_swallow_the_line_after_it() {
        let mut h = start();
        let rx = h.ask_permission();
        h.send("y");
        assert!(rx.await.unwrap());

        h.send("/diff");
        assert_eq!(h.commands.recv().await.unwrap(), "/diff");
    }

    #[tokio::test]
    async fn blank_lines_are_not_commands_and_not_interjections() {
        let mut h = start();
        h.send("   ");
        h.send("/diff");

        // The blank was dropped rather than forwarded, so the next command is
        // the first thing the loop sees.
        assert_eq!(h.commands.recv().await.unwrap(), "/diff");
        assert!(h.interjections.is_empty());
    }
}

async fn first_task(talos: &mut Talos, task: &str, max_tokens: u32) -> Result<()> {
    let plan = metis::plan(
        talos.engine.as_ref(),
        &talos.themis,
        &talos.scribe,
        task,
        max_tokens,
    )
    .await?;

    println!("Plan:\n{}\n", plan.render());
    let outcome = talos.run(task, &plan).await?;
    report(talos, &outcome);
    Ok(())
}

fn report(talos: &Talos, outcome: &crate::talos::Outcome) {
    println!("\n{}", outcome.summary);

    if !outcome.changed.is_empty() {
        println!("\n{} file(s) touched:", outcome.changed.len());
        for p in &outcome.changed {
            println!("  {}", talos.ctx.display(p));
        }
    }
    if outcome.dry_run && !talos.diffs().is_empty() {
        println!("\n/diff to review, /apply to write, /discard to throw away.");
    }
    println!(
        "[{} in {} step(s)]\n",
        outcome.halt.label(),
        outcome.steps_used
    );
}

fn show_diff(talos: &Talos) {
    if !talos.ctx.is_dry_run() {
        println!("Not a dry run — changes are already on disk. Use `git diff`.");
        return;
    }
    println!("{}", diff::render(&talos.diffs()));
}

fn apply(talos: &mut Talos) -> Result<()> {
    if !talos.ctx.is_dry_run() {
        println!("Not a dry run — changes were already written.");
        return Ok(());
    }
    let written = talos.apply()?;
    if written.is_empty() {
        println!("Nothing staged.");
    } else {
        println!("Wrote {} file(s):", written.len());
        for p in &written {
            println!("  {}", talos.ctx.display(p));
        }
        println!("\nRun /verify — the ladder could only check syntax while unwritten.");
    }
    Ok(())
}

async fn verify(talos: &mut Talos) -> Result<()> {
    let files: Vec<std::path::PathBuf> = talos.changed.iter().cloned().collect();
    let verdict = if talos.ctx.is_dry_run() {
        talos
            .oracle
            .verify_staged(talos.scribe.adapter(), &talos.ctx.staged_contents())
    } else {
        talos.oracle.verify(talos.scribe.adapter(), &files).await?
    };

    for tier in &verdict.tiers {
        let mark = if tier.passed { "PASS" } else { "FAIL" };
        println!("[{mark}] tier {} — {}", tier.tier, tier.label);
        if !tier.passed {
            println!("\n{}\n", tier.detail);
        }
    }
    println!("{}", verdict.summary());
    Ok(())
}

fn index(talos: &Talos, arg: &str) {
    if arg.is_empty() {
        println!(
            "{} symbols across {} files",
            talos.scribe.symbol_count(),
            talos.scribe.file_count()
        );
        return;
    }
    let hits = talos.scribe.lookup(arg);
    if hits.is_empty() {
        println!("`{arg}` is not declared in this workspace.");
        return;
    }
    for s in hits {
        println!("{}:{}: {}", s.file.display(), s.line, s.signature);
    }
}

async fn plan_only(talos: &Talos, task: &str, max_tokens: u32) -> Result<()> {
    if task.is_empty() {
        println!("Usage: /plan <task>");
        return Ok(());
    }
    let plan = metis::plan(
        talos.engine.as_ref(),
        &talos.themis,
        &talos.scribe,
        task,
        max_tokens,
    )
    .await?;
    println!("{}", plan.render());
    Ok(())
}

fn set_steps(talos: &mut Talos, arg: &str) {
    match arg.trim().parse::<usize>() {
        Ok(n) if n > 0 => {
            talos.ariadne.max_steps = n;
            talos.ariadne.target_steps = talos.ariadne.target_steps.min(n);
            println!(
                "Step ceiling is now {} (pressure begins at {}).",
                talos.ariadne.max_steps, talos.ariadne.target_steps
            );
        }
        _ => println!("Usage: /steps <positive integer>"),
    }
}

fn split_command(rest: &str) -> (&str, &str) {
    match rest.find(char::is_whitespace) {
        Some(i) => (&rest[..i], rest[i..].trim()),
        None => (rest, ""),
    }
}

/// Trim whitespace and strip a leading byte-order mark.
fn clean(raw: &str) -> String {
    raw.trim_start_matches('\u{feff}').trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_a_bare_command() {
        assert_eq!(split_command("diff"), ("diff", ""));
    }

    #[test]
    fn splits_a_command_with_an_argument() {
        assert_eq!(split_command("plan add a flag"), ("plan", "add a flag"));
        assert_eq!(split_command("steps   9"), ("steps", "9"));
    }

    #[test]
    fn a_byte_order_mark_does_not_hide_a_command() {
        // Piped input on Windows carries one, and a BOM is not whitespace.
        assert_eq!(clean("\u{feff}/quit\r\n"), "/quit");
        assert!(clean("\u{feff}/quit\r\n").starts_with('/'));
    }

    #[test]
    fn clean_trims_ordinary_input_too() {
        assert_eq!(clean("  add a flag \n"), "add a flag");
        assert_eq!(clean("\r\n"), "");
    }

    #[test]
    fn help_lists_every_command_the_loop_handles() {
        for cmd in [
            "/help", "/diff", "/apply", "/discard", "/verify", "/index", "/plan", "/reset",
            "/steps", "/quit",
        ] {
            assert!(HELP.contains(cmd), "help text is missing {cmd}");
        }
    }
}
