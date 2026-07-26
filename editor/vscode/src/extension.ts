// Daedalus Harness — VS Code front end.
//
// Deliberately thin. All the behaviour lives in the Rust binary; this spawns
// it, streams its output, and uses VS Code for the parts VS Code is good at:
// input prompts, an output channel, diff syntax highlighting, and cancellation.
//
// Nothing here reimplements harness logic, so the editor cannot drift from
// what the CLI does.

import * as vscode from "vscode";
import { spawn } from "child_process";

let channel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext): void {
  channel = vscode.window.createOutputChannel("Daedalus");
  context.subscriptions.push(channel);

  const register = (id: string, fn: () => Promise<void>) =>
    context.subscriptions.push(
      vscode.commands.registerCommand(id, () => fn().catch(reportError))
    );

  register("daedalus.task", runTask);
  register("daedalus.preview", previewTask);
  register("daedalus.verify", verifyWorkspace);
  register("daedalus.index", showIndex);
  register("daedalus.lookup", lookupSymbol);
}

export function deactivate(): void {
  // The output channel is disposed via context.subscriptions.
}

// ---- configuration ----

function config(): vscode.WorkspaceConfiguration {
  return vscode.workspace.getConfiguration("daedalus");
}

function binaryPath(): string {
  return config().get<string>("binaryPath")?.trim() || "daedalus";
}

/** Flags that apply to every subcommand. */
function globalArgs(root: string): string[] {
  const args = ["--workspace", root];
  const engine = config().get<string>("engine");
  if (engine) {
    args.push("--engine", engine);
  }
  const model = config().get<string>("model")?.trim();
  if (model) {
    args.push("--model", model);
  }
  return args;
}

/** Budget and judgement flags, shared by `task` and its dry-run variant. */
function loopArgs(): string[] {
  const args = [
    "--max-steps",
    String(config().get<number>("maxSteps") ?? 12),
    "--target-steps",
    String(config().get<number>("targetSteps") ?? 6),
  ];
  if (config().get<boolean>("judge") === false) {
    args.push("--no-judge");
  }
  return args;
}

function workspaceRoot(): string | undefined {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    return undefined;
  }
  return folders[0].uri.fsPath;
}

async function requireRoot(): Promise<string | undefined> {
  const root = workspaceRoot();
  if (!root) {
    await vscode.window.showErrorMessage(
      "Daedalus needs an open folder to use as its workspace."
    );
    return undefined;
  }
  return root;
}

// ---- process plumbing ----

interface RunResult {
  code: number;
  stdout: string;
  stderr: string;
  cancelled: boolean;
}

/**
 * Spawn the binary and stream both streams into the output channel as they
 * arrive, so a long task shows progress rather than appearing hung.
 */
function run(
  args: string[],
  cwd: string,
  token?: vscode.CancellationToken
): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    channel.appendLine(`$ ${binaryPath()} ${args.join(" ")}`);

    const child = spawn(binaryPath(), args, { cwd });
    let stdout = "";
    let stderr = "";
    let cancelled = false;

    child.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString();
      stdout += text;
      channel.append(text);
    });

    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString();
      stderr += text;
      channel.append(text);
    });

    token?.onCancellationRequested(() => {
      cancelled = true;
      child.kill();
      channel.appendLine("\n[cancelled]");
    });

    child.on("error", (err: NodeJS.ErrnoException) => {
      if (err.code === "ENOENT") {
        reject(
          new Error(
            `Could not find '${binaryPath()}'. Set daedalus.binaryPath to the ` +
              `full path of the executable, or add it to your PATH.`
          )
        );
        return;
      }
      reject(err);
    });

    child.on("close", (code: number | null) => {
      channel.appendLine("");
      resolve({ code: code ?? -1, stdout, stderr, cancelled });
    });
  });
}

/** Run with a cancellable progress notification. */
function runWithProgress(
  title: string,
  args: string[],
  cwd: string
): Thenable<RunResult> {
  return vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title,
      cancellable: true,
    },
    (_progress, token) => run(args, cwd, token)
  );
}

function reportError(err: unknown): void {
  const message = err instanceof Error ? err.message : String(err);
  channel.appendLine(`\nError: ${message}`);
  void vscode.window.showErrorMessage(`Daedalus: ${message}`);
}

// ---- commands ----

async function askForTask(prompt: string): Promise<string | undefined> {
  return vscode.window.showInputBox({
    prompt,
    placeHolder: "e.g. add a --json flag to the CLI",
    ignoreFocusOut: true,
  });
}

async function runTask(): Promise<void> {
  const root = await requireRoot();
  if (!root) {
    return;
  }

  const task = await askForTask("What should Daedalus do?");
  if (!task) {
    return;
  }

  const confirmed = await vscode.window.showWarningMessage(
    "This will edit files in your workspace. Commit or stash first if you want an easy undo.",
    { modal: true },
    "Run",
    "Preview instead"
  );
  if (confirmed === "Preview instead") {
    await preview(root, task);
    return;
  }
  if (confirmed !== "Run") {
    return;
  }

  channel.show(true);
  const result = await runWithProgress(
    "Daedalus: running task",
    ["task", task, ...globalArgs(root), ...loopArgs()],
    root
  );

  if (result.cancelled) {
    return;
  }
  if (result.code === 0) {
    void vscode.window.showInformationMessage("Daedalus: task completed and verified.");
  } else {
    void vscode.window.showWarningMessage(
      "Daedalus: task did not verify. See the Daedalus output channel."
    );
  }
}

async function previewTask(): Promise<void> {
  const root = await requireRoot();
  if (!root) {
    return;
  }
  const task = await askForTask("What should Daedalus propose?");
  if (!task) {
    return;
  }
  await preview(root, task);
}

/**
 * Dry run: the harness stages edits in memory and prints unified diffs.
 * Those open in an editor tab as a `diff` document, which gets VS Code's own
 * syntax highlighting for free.
 */
async function preview(root: string, task: string): Promise<void> {
  channel.show(true);
  const result = await runWithProgress(
    "Daedalus: previewing task",
    ["task", task, "--dry-run", ...globalArgs(root), ...loopArgs()],
    root
  );

  if (result.cancelled) {
    return;
  }

  const diffText = extractDiff(result.stdout);
  if (!diffText) {
    void vscode.window.showInformationMessage(
      "Daedalus: no changes were proposed. See the output channel."
    );
    return;
  }

  const doc = await vscode.workspace.openTextDocument({
    content: diffText,
    language: "diff",
  });
  await vscode.window.showTextDocument(doc, { preview: false });

  void vscode.window.showInformationMessage(
    "Daedalus: preview only — nothing was written. Re-run without preview to apply."
  );
}

/**
 * Pull the diff section out of the CLI's stdout.
 *
 * The summary line the harness prints before the diffs is a stable marker;
 * if it is missing we show the whole of stdout rather than silently nothing.
 */
export function extractDiff(stdout: string): string | undefined {
  const marker = stdout.indexOf(" file(s) changed");
  if (marker === -1) {
    return stdout.includes("No changes proposed") ? undefined : stdout.trim() || undefined;
  }
  const lineStart = stdout.lastIndexOf("\n", marker) + 1;
  return stdout.slice(lineStart).trim();
}

async function verifyWorkspace(): Promise<void> {
  const root = await requireRoot();
  if (!root) {
    return;
  }

  channel.show(true);
  const result = await runWithProgress(
    "Daedalus: verifying",
    ["verify", ...globalArgs(root)],
    root
  );

  if (result.cancelled) {
    return;
  }
  if (result.code === 0) {
    void vscode.window.showInformationMessage("Daedalus: all verification tiers passed.");
  } else {
    void vscode.window.showErrorMessage(
      "Daedalus: verification failed. See the Daedalus output channel."
    );
  }
}

async function showIndex(): Promise<void> {
  const root = await requireRoot();
  if (!root) {
    return;
  }
  channel.show(true);
  await run(["index", "--full", ...globalArgs(root)], root);
}

async function lookupSymbol(): Promise<void> {
  const root = await requireRoot();
  if (!root) {
    return;
  }

  const editor = vscode.window.activeTextEditor;
  const selected = editor?.document.getText(editor.selection).trim();
  const name =
    selected ||
    (await vscode.window.showInputBox({
      prompt: "Symbol to look up in the exact index",
      ignoreFocusOut: true,
    }));

  if (!name) {
    return;
  }

  const result = await run(["index", "--lookup", name, ...globalArgs(root)], root);
  if (result.code !== 0) {
    void vscode.window.showWarningMessage(
      `Daedalus: '${name}' is not declared in this workspace.`
    );
    return;
  }

  channel.show(true);
}
