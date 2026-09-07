// Settings lookup, shared by the command palette and the panel so both drive
// the binary with identical flags.

import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";

export function config(): vscode.WorkspaceConfiguration {
  return vscode.workspace.getConfiguration("knossos");
}

const EXE = process.platform === "win32" ? "knossos.exe" : "knossos";
const LEGACY_EXE = process.platform === "win32" ? "daedalus.exe" : "daedalus";

/**
 * Find the harness executable without making the user configure anything.
 *
 * Order: an explicit setting, then a build inside the workspace, then a
 * sibling `Knossos-Harness` checkout, then PATH. Requiring a path up front
 * was the single worst thing about first-run — most of the time it is
 * discoverable, so it should be discovered.
 *
 * Returns `undefined` when nothing was found, so callers can offer a picker
 * instead of spawning something that will not exist.
 */
export function resolveBinary(root: string | undefined): string | undefined {
  const configured = config().get<string>("binaryPath")?.trim();
  if (configured && configured !== "knossos") {
    return fs.existsSync(configured) ? configured : undefined;
  }

  for (const candidate of candidates(root)) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  // Fall back to PATH and let the spawn decide.
  return "knossos";
}

function candidates(root: string | undefined): string[] {
  const out: string[] = [];
  if (!root) {
    return out;
  }

  const roots = [
    root,
    path.join(path.dirname(root), "Knossos-Harness"),
    path.join(path.dirname(root), "knossos-harness"),
    path.join(path.dirname(root), "daedalus-harness"),
  ];
  for (const base of roots) {
    out.push(path.join(base, "target", "release", EXE));
    out.push(path.join(base, "target", "debug", EXE));
    out.push(path.join(base, "knossos-rs", "target", "release", EXE));
    out.push(path.join(base, "knossos-rs", "target", "debug", EXE));
    // v0.1 shipped the binary as `daedalus`; keep discovering it during the
    // Knossos rename so existing local builds continue to work.
    out.push(path.join(base, "target", "release", LEGACY_EXE));
    out.push(path.join(base, "target", "debug", LEGACY_EXE));
    out.push(path.join(base, "knossos-rs", "target", "release", LEGACY_EXE));
    out.push(path.join(base, "knossos-rs", "target", "debug", LEGACY_EXE));
  }
  return out;
}

/** Whether the user pinned a path explicitly. */
export function binaryIsConfigured(): boolean {
  const configured = config().get<string>("binaryPath")?.trim();
  return !!configured && configured !== "knossos";
}

/**
 * Ask the user to point at the executable, and remember it.
 *
 * A file picker beats "go edit this setting" — the setting is still there for
 * anyone who wants it, but nobody should have to find it to get started.
 */
export async function promptForBinary(): Promise<string | undefined> {
  const picked = await vscode.window.showOpenDialog({
    canSelectMany: false,
    openLabel: "Use this Knossos executable",
    title: "Locate the Knossos executable",
    filters: process.platform === "win32" ? { Executable: ["exe"] } : undefined,
  });

  const chosen = picked?.[0]?.fsPath;
  if (!chosen) {
    return undefined;
  }

  await config().update("binaryPath", chosen, vscode.ConfigurationTarget.Global);
  return chosen;
}

/** Flags that apply to every subcommand. */
export function globalArgs(root: string): string[] {
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

/** Budget and judgement flags, shared by `task`, `repl` and `serve`. */
export function loopArgs(): string[] {
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

/** Whether the panel stages changes for review instead of writing them. */
export function panelPreviewsByDefault(): boolean {
  return config().get<boolean>("previewByDefault") !== false;
}

export function workspaceRoot(): string | undefined {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    return undefined;
  }
  return folders[0].uri.fsPath;
}

export async function requireRoot(): Promise<string | undefined> {
  const root = workspaceRoot();
  if (!root) {
    const choice = await vscode.window.showErrorMessage(
      "Knossos needs an open folder to use as its workspace.",
      "Open Folder"
    );
    if (choice === "Open Folder") {
      await vscode.commands.executeCommand("vscode.openFolder");
    }
    return undefined;
  }
  return root;
}

export const API_KEY_SECRET = "knossos.anthropicApiKey";
const LEGACY_API_KEY_SECRET = "daedalus.anthropicApiKey";

/**
 * Environment for the child process.
 *
 * The API key comes from VS Code's SecretStorage — the OS keychain — rather
 * than from settings.json, which syncs across machines and gets committed by
 * accident. An inherited environment variable still wins if one is set.
 */
export async function childEnv(
  secrets: vscode.SecretStorage
): Promise<NodeJS.ProcessEnv> {
  const env = { ...process.env };
  if (!env.ANTHROPIC_API_KEY) {
    const stored =
      (await secrets.get(API_KEY_SECRET)) ??
      (await secrets.get(LEGACY_API_KEY_SECRET));
    if (stored) {
      env.ANTHROPIC_API_KEY = stored;
    }
  }
  return env;
}
