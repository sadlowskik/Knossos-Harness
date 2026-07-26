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

pub async fn run(mut talos: Talos, initial: Option<String>, max_tokens: u32) -> Result<()> {
    println!("Daedalus interactive session. /help for commands, /quit to leave.");
    if talos.ctx.is_dry_run() {
        println!("DRY RUN — nothing will be written until you /apply.");
    }
    println!();

    if let Some(task) = initial {
        first_task(&mut talos, &task, max_tokens).await?;
    }

    let stdin = std::io::stdin();
    loop {
        print!("daedalus> ");
        std::io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break; // EOF (Ctrl-D / piped input ended)
        }
        // A UTF-8 BOM is not whitespace, so `trim` leaves it in place. Piped
        // input on Windows routinely carries one, and left alone it turns
        // `/quit` into an instruction for the engine.
        let line = clean(&line);
        let line = line.as_str();
        if line.is_empty() {
            continue;
        }

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
    println!("[{} in {} step(s)]\n", outcome.halt.label(), outcome.steps_used);
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
        for cmd in ["/help", "/diff", "/apply", "/discard", "/verify", "/index", "/plan", "/reset", "/steps", "/quit"] {
            assert!(HELP.contains(cmd), "help text is missing {cmd}");
        }
    }
}
