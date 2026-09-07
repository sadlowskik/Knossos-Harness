// The sidebar session panel.
//
// Owns one long-lived `knossos serve` process and relays between it and the
// webview. The panel does no harness work of its own — it renders events and
// forwards intent — so it cannot drift from what the CLI does.

import * as path from "path";
import * as vscode from "vscode";

import { KnossosClient, KnossosEvent } from "./client";
import {
  childEnv,
  globalArgs,
  loopArgs,
  panelPreviewsByDefault,
  promptForBinary,
  resolveBinary,
  workspaceRoot,
} from "./config";

/** Virtual documents backing the diff viewer. */
export const SCHEME = "knossos";

export class ProposedContentProvider implements vscode.TextDocumentContentProvider {
  private readonly contents = new Map<string, string>();
  private readonly changed = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this.changed.event;

  set(uri: vscode.Uri, content: string): void {
    this.contents.set(uri.toString(), content);
    this.changed.fire(uri);
  }

  clear(): void {
    this.contents.clear();
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.contents.get(uri.toString()) ?? "";
  }
}

interface DiffFile {
  path: string;
  content: string;
  existed: boolean;
}

interface HunkSelection {
  path: string;
  hunks: number[];
}

/** How much of a mentioned file to inline before truncating. */
const MENTION_BUDGET = 20_000;
/** Upper bound on the @-mention file list. */
const MAX_FILES = 5000;

export class SessionPanel implements vscode.WebviewViewProvider, vscode.Disposable {
  static readonly viewType = "knossos.session";

  private view: vscode.WebviewView | undefined;
  private readonly client = new KnossosClient();
  /** Latest proposed content, keyed by workspace-relative path. */
  private diffs = new Map<string, DiffFile>();
  /** Guards against re-reporting a failed start on every webview message. */
  private startFailed = false;
  private handlersWired = false;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly proposed: ProposedContentProvider,
    private readonly channel: vscode.OutputChannel,
    private readonly secrets: vscode.SecretStorage
  ) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;

    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, "media")],
    };
    view.webview.html = this.html(view.webview);

    view.webview.onDidReceiveMessage((msg) => void this.onMessage(msg));
    view.onDidDispose(() => this.client.stop());

    void this.startServer();
    void this.sendFileList();
  }

  /** Supply the webview with workspace paths for the @-mention picker. */
  private async sendFileList(): Promise<void> {
    const root = workspaceRoot();
    if (!root) {
      return;
    }
    const found = await vscode.workspace.findFiles(
      "**/*",
      "**/{node_modules,target,.git,dist,out}/**",
      MAX_FILES
    );
    const files = found
      .map((uri) => path.relative(root, uri.fsPath).replace(/\\/g, "/"))
      .sort();
    void this.view?.webview.postMessage({ type: "files", files });
  }

  private async startServer(): Promise<void> {
    const root = workspaceRoot();
    if (!root) {
      this.post({
        event: "error",
        message: "Open a folder (File → Open Folder) to start a Knossos session.",
      });
      this.startFailed = true;
      return;
    }

    const binary = resolveBinary(root);
    if (!binary) {
      await this.reportMissingBinary(undefined);
      return;
    }

    const args = ["serve", ...globalArgs(root), ...loopArgs()];
    if (panelPreviewsByDefault()) {
      args.push("--dry-run");
    }

    this.channel.appendLine(`$ ${binary} ${args.join(" ")}`);

    // Handlers persist across restarts; registering them once avoids each
    // retry multiplying every event.
    if (!this.handlersWired) {
      this.handlersWired = true;
      this.client.onEvent((event) => void this.onServerEvent(event));
      this.client.onStderr((text) => this.channel.append(text));
      this.client.onExit((code) => {
        if (!this.startFailed) {
          this.post({ event: "exited", code });
        }
      });
    }

    this.startFailed = false;
    this.client.start(binary, args, root, await childEnv(this.secrets));
  }

  /**
   * Report a missing executable once, with a way to fix it.
   *
   * Repeating "set knossos.binaryPath" on every retry is noise, and telling
   * someone to go find a setting is a worse answer than opening a picker.
   */
  private async reportMissingBinary(binary: string | undefined): Promise<void> {
    if (this.startFailed) {
      return;
    }
    this.startFailed = true;

    this.post({
      event: "error",
      message:
        "Could not find the Knossos executable. Build it with `cargo build --release`, " +
        "then use Locate to point at it.",
    });

    const choice = await vscode.window.showErrorMessage(
      binary
        ? `Knossos: could not run '${binary}'.`
        : "Knossos: could not find the Knossos executable.",
      "Locate…",
      "Open Output"
    );

    if (choice === "Locate…") {
      const chosen = await promptForBinary();
      if (chosen) {
        this.startFailed = false;
        await this.startServer();
      }
    } else if (choice === "Open Output") {
      this.channel.show(true);
    }
  }

  private async onServerEvent(event: KnossosEvent): Promise<void> {
    if (event.event === "missing_binary") {
      await this.reportMissingBinary(event.binary as string);
      return;
    }
    if (event.event === "diffs") {
      this.diffs.clear();
      const files = (event.files as DiffFile[]) ?? [];
      for (const file of files) {
        this.diffs.set(file.path, file);
      }
    }
    if (event.event === "applied" || event.event === "discarded") {
      this.diffs.clear();
      this.proposed.clear();
    }
    this.post(event);
  }

  private async onMessage(msg: {
    type: string;
    text?: string;
    path?: string;
    selection?: HunkSelection[];
  }): Promise<void> {
    switch (msg.type) {
      case "ready":
        // The webview reloaded; the server may already be running. Do not
        // retry a start that already failed — that is what produced a stack
        // of identical errors.
        if (!this.client.running && !this.startFailed) {
          await this.startServer();
        }
        void this.sendFileList();
        break;
      case "restart":
        this.startFailed = false;
        await this.startServer();
        break;
      case "task":
        this.client.send({ cmd: "task", text: await this.expandMentions(msg.text ?? "") });
        break;
      case "resume":
        this.client.send({ cmd: "resume", text: await this.expandMentions(msg.text ?? "") });
        break;
      case "applyHunks":
        this.client.send({ cmd: "apply_hunks", selection: msg.selection ?? [] });
        break;
      case "reset":
        this.client.send({ cmd: "reset" });
        break;
      case "verify":
        this.client.send({ cmd: "verify" });
        break;
      case "apply":
        this.client.send({ cmd: "apply" });
        break;
      case "discard":
        this.client.send({ cmd: "discard" });
        break;
      case "openDiff":
        void this.openDiff(msg.path ?? "");
        break;
      default:
        break;
    }
  }

  /**
   * Inline the contents of any `@path` the user mentioned.
   *
   * Done here rather than in the webview because the extension host is the
   * side with filesystem access. A mention that does not resolve to a real
   * file is left alone as prose — the agent can still read it as a hint, and
   * silently dropping it would be worse than passing it through.
   */
  private async expandMentions(text: string): Promise<string> {
    const root = workspaceRoot();
    if (!root) {
      return text;
    }

    const mentioned = [...text.matchAll(/@([^\s]+)/g)].map((m) => m[1]);
    const unique = [...new Set(mentioned)];
    if (unique.length === 0) {
      return text;
    }

    let context = "";
    for (const relative of unique) {
      try {
        const uri = vscode.Uri.file(path.join(root, relative));
        const bytes = await vscode.workspace.fs.readFile(uri);
        let content = Buffer.from(bytes).toString("utf8");
        if (content.length > MENTION_BUDGET) {
          content = `${content.slice(0, MENTION_BUDGET)}\n[truncated]`;
        }
        context += `\n### ${relative}\n\`\`\`\n${content}\n\`\`\`\n`;
      } catch {
        // Not a readable path — leave the mention as written.
      }
    }

    return context ? `${text}\n\n# Files referenced\n${context}` : text;
  }

  /**
   * Open VS Code's own side-by-side diff.
   *
   * The proposed side is a virtual document holding the staged content — which
   * is why the protocol sends whole files rather than only unified text. A new
   * file gets an empty virtual document on the left, since there is nothing on
   * disk to compare against.
   */
  private async openDiff(relative: string): Promise<void> {
    const file = this.diffs.get(relative);
    const root = workspaceRoot();
    if (!file || !root) {
      return;
    }

    const proposedUri = vscode.Uri.from({
      scheme: SCHEME,
      path: `/proposed/${relative.replace(/\\/g, "/")}`,
    });
    this.proposed.set(proposedUri, file.content);

    let originalUri: vscode.Uri;
    if (file.existed) {
      originalUri = vscode.Uri.file(path.join(root, relative));
    } else {
      originalUri = vscode.Uri.from({
        scheme: SCHEME,
        path: `/empty/${relative.replace(/\\/g, "/")}`,
      });
      this.proposed.set(originalUri, "");
    }

    await vscode.commands.executeCommand(
      "vscode.diff",
      originalUri,
      proposedUri,
      `${relative} — proposed`,
      { preview: true }
    );
  }

  private post(event: KnossosEvent): void {
    void this.view?.webview.postMessage({ type: "knossos", event });
  }

  /** Ask the panel to run a task, e.g. from a command-palette entry. */
  runTask(text: string): void {
    this.client.send({ cmd: "task", text });
  }

  dispose(): void {
    this.client.stop();
  }

  private html(webview: vscode.Webview): string {
    const asset = (name: string) =>
      webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, "media", name));

    // A nonce keeps the CSP strict: only our own script may run.
    const nonce = Array.from({ length: 32 }, () =>
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".charAt(
        Math.floor(Math.random() * 62)
      )
    ).join("");

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource}; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="${asset("panel.css")}" rel="stylesheet">
<title>Knossos</title>
</head>
<body>
  <header id="harness-head">
    <span class="harness-seal" aria-hidden="true">K</span>
    <span class="harness-title"><strong>KNOSSOS</strong><small>AGENT SESSION</small></span>
    <span class="harness-live"><i></i> READY</span>
  </header>
  <div id="status">
    <span id="status-left">starting…</span>
    <span id="status-right"></span>
  </div>

  <div id="log"></div>

  <div id="composer">
    <div id="mentions" hidden></div>
    <textarea id="input" placeholder="Describe a task…  (@ to reference a file, Enter to send, Shift+Enter for a newline)"></textarea>
    <div class="row">
      <button id="send">Send</button>
      <button id="verify" class="secondary">Verify</button>
      <button id="reset" class="secondary">New session</button>
      <span class="spacer"></span>
      <span id="busy"></span>
    </div>
  </div>

  <script nonce="${nonce}" src="${asset("panel.js")}"></script>
</body>
</html>`;
  }
}
