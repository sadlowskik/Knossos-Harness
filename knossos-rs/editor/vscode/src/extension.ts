// Knossos Harness — VS Code front end.
//
// Two surfaces over the same binary:
//   * a sidebar session panel backed by `knossos serve` (see panel.ts)
//   * one-shot command-palette entries for index / verify / lookup
//
// Deliberately thin. No harness logic is reimplemented here, so the editor
// cannot drift from what the CLI does.

import * as vscode from "vscode";
import { spawn } from "child_process";

import {
  API_KEY_SECRET,
  childEnv,
  globalArgs,
  loopArgs,
  promptForBinary,
  requireRoot,
  resolveBinary,
  workspaceRoot,
} from "./config";
import { ProposedContentProvider, SessionPanel, SCHEME } from "./panel";

let channel: vscode.OutputChannel;
let secrets: vscode.SecretStorage;

export function activate(context: vscode.ExtensionContext): void {
  channel = vscode.window.createOutputChannel("Knossos");
  context.subscriptions.push(channel);
  secrets = context.secrets;

  const proposed = new ProposedContentProvider();
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(SCHEME, proposed)
  );

  const panel = new SessionPanel(context.extensionUri, proposed, channel, context.secrets);
  context.subscriptions.push(panel);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(SessionPanel.viewType, panel, {
      // Keep the conversation when the panel scrolls out of view.
      webviewOptions: { retainContextWhenHidden: true },
    })
  );

  const register = (id: string, fn: () => Promise<void>) =>
    context.subscriptions.push(
      vscode.commands.registerCommand(id, () => fn().catch(reportError))
    );

  register("knossos.openPanel", async () => {
    await vscode.commands.executeCommand("knossos.session.focus");
  });
  register("knossos.task", () => runTask(panel));
  register("knossos.preview", previewTask);
  register("knossos.verify", verifyWorkspace);
  register("knossos.index", showIndex);
  register("knossos.lookup", lookupSymbol);
  register("knossos.setApiKey", setApiKey);
  register("knossos.clearApiKey", clearApiKey);
  register("knossos.locateBinary", async () => {
    const chosen = await promptForBinary();
    if (chosen) {
      void vscode.window.showInformationMessage(`Knossos: using ${chosen}`);
    }
  });
}

/**
 * Store the API key in the OS keychain via SecretStorage.
 *
 * Not a setting: settings.json syncs between machines and gets committed by
 * accident. `password: true` also keeps it off the screen while typing.
 */
async function setApiKey(): Promise<void> {
  const key = await vscode.window.showInputBox({
    prompt: "Anthropic API key — stored in your OS keychain, never in settings",
    password: true,
    ignoreFocusOut: true,
    placeHolder: "sk-ant-…",
  });
  if (key === undefined) {
    return;
  }
  const trimmed = key.trim();
  if (!trimmed) {
    void vscode.window.showWarningMessage("Knossos: no key entered.");
    return;
  }

  await secrets.store(API_KEY_SECRET, trimmed);
  void vscode.window.showInformationMessage(
    "Knossos: API key saved. Reload the session panel to use it."
  );
}

async function clearApiKey(): Promise<void> {
  await secrets.delete(API_KEY_SECRET);
  void vscode.window.showInformationMessage("Knossos: stored API key removed.");
}

export function deactivate(): void {
  // Everything is disposed through context.subscriptions.
}

// ---- process plumbing (one-shot commands) ----

interface RunResult {
  code: number;
  stdout: string;
  stderr: string;
  cancelled: boolean;
}

/**
 * Spawn the binary and stream both streams into the output channel as they
 * arrive, so a long command shows progress rather than appearing hung.
 */
async function run(
  args: string[],
  cwd: string,
  token?: vscode.CancellationToken
): Promise<RunResult> {
  const binary = resolveBinary(workspaceRoot()) ?? "knossos";
  const env = await childEnv(secrets);

  return new Promise((resolve, reject) => {
    channel.appendLine(`$ ${binary} ${args.join(" ")}`);

    const child = spawn(binary, args, { cwd, env });
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
            `Could not find '${binary}'. Build it with \`cargo build --release\`, ` +
              `then run "Knossos: Locate Executable".`
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

function runWithProgress(
  title: string,
  args: string[],
  cwd: string
): Thenable<RunResult> {
  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title, cancellable: true },
    (_progress, token) => run(args, cwd, token)
  );
}

function reportError(err: unknown): void {
  const message = err instanceof Error ? err.message : String(err);
  channel.appendLine(`\nError: ${message}`);
  void vscode.window.showErrorMessage(`Knossos: ${message}`);
}

// ---- commands ----

async function askForTask(prompt: string): Promise<string | undefined> {
  return vscode.window.showInputBox({
    prompt,
    placeHolder: "e.g. add a --json flag to the CLI",
    ignoreFocusOut: true,
  });
}

/** Route a task into the session panel, so it lands in the conversation. */
async function runTask(panel: SessionPanel): Promise<void> {
  const root = await requireRoot();
  if (!root) {
    return;
  }
  const task = await askForTask("What should Knossos do?");
  if (!task) {
    return;
  }
  await vscode.commands.executeCommand("knossos.session.focus");
  panel.runTask(task);
}

async function previewTask(): Promise<void> {
  const root = await requireRoot();
  if (!root) {
    return;
  }
  const task = await askForTask("What should Knossos propose?");
  if (!task) {
    return;
  }

  channel.show(true);
  const result = await runWithProgress(
    "Knossos: previewing task",
    ["task", task, "--dry-run", ...globalArgs(root), ...loopArgs()],
    root
  );
  if (result.cancelled) {
    return;
  }

  const diffText = extractDiff(result.stdout);
  if (!diffText) {
    void vscode.window.showInformationMessage(
      "Knossos: no changes were proposed. See the output channel."
    );
    return;
  }

  const doc = await vscode.workspace.openTextDocument({
    content: diffText,
    language: "diff",
  });
  await vscode.window.showTextDocument(doc, { preview: false });
  void vscode.window.showInformationMessage(
    "Knossos: preview only — nothing was written."
  );
}

/**
 * Pull the diff section out of the CLI's stdout.
 *
 * The summary line the harness prints before the diffs is a stable marker; if
 * it is missing we show the whole of stdout rather than silently nothing.
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
    "Knossos: verifying",
    ["verify", ...globalArgs(root)],
    root
  );
  if (result.cancelled) {
    return;
  }

  if (result.code === 0) {
    void vscode.window.showInformationMessage("Knossos: all verification tiers passed.");
  } else {
    void vscode.window.showErrorMessage(
      "Knossos: verification failed. See the Knossos output channel."
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
      `Knossos: '${name}' is not declared in this workspace.`
    );
    return;
  }
  channel.show(true);
}
