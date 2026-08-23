/**
 * Knossos against the Agent Client Protocol's own implementation.
 *
 * Every other test in this repository checks Knossos against code written by
 * the same hand that wrote Knossos -- including `test_acp.py`'s FakeClient and
 * the Lapce client's `against_real_agent.rs`. Both drive real pipes, which is
 * better than a mock, but neither can catch a misreading of the specification:
 * if the agent and the client agree on the wrong shape, they agree.
 *
 * This suite removes that. The client is `@agentclientprotocol/sdk`, published
 * by the protocol's authors, and every message the agent sends is additionally
 * validated against `schema/schema.json` -- the machine-readable schema shipped
 * with that package. A field renamed upstream fails here and nowhere else.
 *
 *     npm install && npm test
 *
 * Set KNOSSOS_PYTHON if `python` is not the interpreter you want.
 */
import { spawn, spawnSync } from "node:child_process";
import { Readable, Writable, Transform } from "node:stream";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join, isAbsolute } from "node:path";
import { mkdtempSync, writeFileSync, readFileSync, existsSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";

import * as acp from "@agentclientprotocol/sdk";
import Ajv2020 from "ajv/dist/2020.js";

const require = createRequire(import.meta.url);
const SCHEMA = require("@agentclientprotocol/sdk/schema/schema.json");

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = dirname(HERE);
const MODEL = join(REPO, "model");
const PYTHON = process.env.KNOSSOS_PYTHON || "python";
const RUST_BIN = process.env.KNOSSOS_ACP_BIN || join(
  REPO,
  "knossos-rs",
  "target",
  "debug",
  process.platform === "win32" ? "daedalus.exe" : "daedalus",
);

// -------------------------------------------------------------- schema checks

// `logger: false` because the schema annotates integers with Rust-derived
// formats (`uint32`, `int64`) that ajv does not know and correctly ignores --
// hundreds of lines of noise about something that is not a finding.
const ajv = new Ajv2020({ strict: false, allErrors: true, logger: false });

// The schema's root is one big `anyOf` over every message envelope. Carrying
// that root into a sub-validator would additionally require each payload to be
// a complete JSON-RPC message, so only `$defs` is kept: the point is to check a
// payload against the specific type it claims to be, and get a usable error
// rather than "did not match any of 262 branches".
const validators = new Map();
function validatorFor(def) {
  if (!validators.has(def)) {
    validators.set(def, ajv.compile({
      $schema: SCHEMA.$schema,
      $defs: SCHEMA.$defs,
      $ref: `#/$defs/${def}`,
    }));
  }
  return validators.get(def);
}

function checkSchema(def, value) {
  const validate = validatorFor(def);
  if (validate(value)) return null;
  return ajv.errorsText(validate.errors, { dataVar: def });
}

// ------------------------------------------------------------------- fixtures

function makeWorkspace() {
  const dir = mkdtempSync(join(tmpdir(), "knossos-conf-"));
  mkdirSync(join(dir, "src"));
  writeFileSync(
    join(dir, "src", "halting.py"),
    "def halting_probability(state):\n" +
      "    \"\"\"Compute how much compute this step deserves.\"\"\"\n" +
      "    return 1.0 / (1.0 + state)\n",
    "utf8",
  );
  return dir;
}

function toolCall(tool, args) {
  return "```json\n" + JSON.stringify({ tool, args }) + "\n```";
}

// -------------------------------------------------------------- agent harness

/**
 * Spawns an agent and connects the official client to it.
 *
 * `onPermission` receives the raw request so a check can both answer it and
 * assert on its shape.
 */
function connect({ args = [], env = {}, onPermission, onElicit,
                   buffers: seedBuffers } = {}) {
  // Product agent is Rust. Python remains only if KNOSSOS_ACP=python.
  const usePython = process.env.KNOSSOS_ACP === "python";
  const proc = usePython
    ? spawn(PYTHON, args, {
        cwd: MODEL,
        stdio: ["pipe", "pipe", "pipe"],
        env: { ...process.env, PYTHONPATH: MODEL, PYTHONUNBUFFERED: "1", ...env },
      })
    // Tier 4 would consume another scripted engine reply to judge work that
    // these protocol checks already verify directly. Keep the deterministic
    // wire suite focused on ACP; live checks below exercise a real model.
    : spawn(RUST_BIN, ["acp", "--no-judge"], {
        cwd: REPO,
        stdio: ["pipe", "pipe", "pipe"],
        env: { ...process.env, ...env },
      });

  const stderr = [];
  proc.stderr.on("data", (d) => stderr.push(d.toString()));

  // Every agent->client line, recorded before the SDK sees it, so the suite can
  // validate traffic the SDK might tolerate.
  const raw = [];
  let pending = "";
  const tap = new Transform({
    transform(chunk, _enc, cb) {
      pending += chunk.toString("utf8");
      const lines = pending.split("\n");
      pending = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          raw.push(JSON.parse(line));
        } catch {
          raw.push({ __unparseable: line });
        }
      }
      cb(null, chunk);
    },
  });
  proc.stdout.pipe(tap);

  const updates = [];
  const permissions = [];
  const fsCalls = [];
  const terminalCalls = [];
  const terminals = new Map();
  // Unsaved buffers, keyed by absolute path -- an editor's view of the tree,
  // which is not the same as the tree.
  const buffers = new Map(Object.entries(seedBuffers || {}));

  const client = {
    async sessionUpdate(params) {
      updates.push(params);
    },
    async requestPermission(params) {
      permissions.push(params);
      if (onPermission) return onPermission(params);
      return { outcome: { outcome: "cancelled" } };
    },
    // A real editor owns the file. Writing it here is what the agent is
    // entitled to assume when it delegates.
    async writeTextFile(params) {
      fsCalls.push(["write", params.path]);
      buffers.set(params.path, params.content);
      writeFileSync(params.path, params.content, "utf8");
      return {};
    },
    async readTextFile(params) {
      fsCalls.push(["read", params.path]);
      if (buffers.has(params.path)) return { content: buffers.get(params.path) };
      return { content: readFileSync(params.path, "utf8") };
    },
    // Stable in ACP SDK 1.4. Keep the former name below so the suite can still
    // be run against the lower end of package.json's compatible range.
    async createElicitation(params) {
      if (onElicit) return onElicit(params);
      return { action: "decline" };
    },
    async unstable_createElicitation(params) {
      if (onElicit) return onElicit(params);
      return { action: "decline" };
    },
    // A real editor owns the terminal. Actually running the command keeps this
    // honest: a stub that returned canned output would not prove the agent
    // sends a usable argv.
    async createTerminal(params) {
      const id = `term_${terminals.size + 1}`;
      const done = spawnSync(params.command, params.args ?? [], {
        cwd: params.cwd ?? undefined,
        encoding: "utf8",
      });
      terminals.set(id, {
        output: [done.stdout, done.stderr].filter(Boolean).join(""),
        exitCode: done.status,
        request: params,
      });
      terminalCalls.push(["create", params.command]);
      return { terminalId: id };
    },
    async waitForTerminalExit(params) {
      terminalCalls.push(["wait", params.terminalId]);
      return { exitCode: terminals.get(params.terminalId)?.exitCode ?? null };
    },
    async terminalOutput(params) {
      terminalCalls.push(["output", params.terminalId]);
      return { output: terminals.get(params.terminalId)?.output ?? "", truncated: false };
    },
    async releaseTerminal(params) {
      terminalCalls.push(["release", params.terminalId]);
      terminals.delete(params.terminalId);
      return {};
    },
  };

  const stream = acp.ndJsonStream(
    Writable.toWeb(proc.stdin),
    Readable.toWeb(tap),
  );
  const conn = new acp.ClientSideConnection(() => client, stream);

  return {
    conn,
    updates,
    permissions,
    fsCalls,
    terminalCalls,
    terminals,
    buffers,
    raw,
    stderr,
    kill() {
      try {
        proc.kill();
      } catch {
        /* already gone */
      }
    },
  };
}

const CAPS = {
  protocolVersion: acp.PROTOCOL_VERSION,
  clientCapabilities: {
    fs: { readTextFile: true, writeTextFile: true },
    terminal: true,
    // `{}` is how the schema spells "supported" -- which is falsy in some
    // languages, and is exactly the shape an agent must handle correctly.
    elicitation: { form: {} },
  },
};

function retrievalAgent(opts) {
  if (process.env.KNOSSOS_ACP === "python") {
    return connect({ args: ["-m", "knossos", "--engine", "retrieval"], ...opts });
  }
  // Rust has no RetrievalOnlyEngine: a scripted read of the fixture is the
  // same observable (a tool_call with an absolute path) without a model.
  return scriptedAgent(
    [toolCall("read_file", { path: "src/halting.py" }), "halting_probability is in src/halting.py"],
    opts,
  );
}

function scriptedAgent(script, { execute = true, write = false, ...rest } = {}) {
  // `execute: false` starts the agent in ask mode, which is what makes
  // session/set_mode observable.
  return connect({
    args: [join(HERE, "scripted_agent.py")],
    env: {
      KNOSSOS_SCRIPT: JSON.stringify(script),
      KNOSSOS_EXECUTE: execute ? "1" : "0",
      KNOSSOS_WRITE: write ? "1" : "0",
    },
    ...rest,
  });
}

function refusedWriteScript(path) {
  return [
    toolCall("write_file", { path, content: "x = 1\n" }),
    ...Array(10).fill("The requested write was not permitted, so I stopped."),
  ];
}

function updatesOfKind(updates, kind) {
  return updates
    .map((u) => u.update)
    .filter((u) => u && u.sessionUpdate === kind);
}

// --------------------------------------------------------------------- checks

const checks = [];
function check(name, fn) {
  checks.push({ name, fn });
}

/**
 * A check that needs a real model. Skipped, not failed, when unconfigured --
 * unless KNOSSOS_STRICT is set, which CI does. A cross-harness test that
 * silently does not run is the failure mode that kept the Lapce client's
 * against-a-real-agent suite unexecuted for its whole existence.
 */
const STRICT = process.env.KNOSSOS_STRICT === "1";
const LIVE = process.env.KNOSSOS_LIVE === "1";
const LIVE_PROVIDER = process.env.KNOSSOS_LIVE_PROVIDER || "ollama";
const LIVE_MODEL = process.env.KNOSSOS_LIVE_MODEL || "";
function liveCheck(name, fn) {
  checks.push({
    name,
    live: true,
    fn: LIVE ? fn : async () => "skipped: set KNOSSOS_LIVE=1",
    skipped: !LIVE,
  });
}

function liveAgent(opts) {
  if (process.env.KNOSSOS_ACP === "python") {
    const args = ["-m", "knossos", "--engine", "api", "--provider", LIVE_PROVIDER];
    if (LIVE_MODEL) args.push("--model", LIVE_MODEL);
    if (opts?.execute) args.push("--execute");
    return connect({ args, ...opts });
  }
  return connect({
    env: {
      ...(LIVE_MODEL ? { DAEDALUS_MODEL: LIVE_MODEL } : {}),
    },
    ...opts,
  });
}

check("handshake reports a valid InitializeResponse", async () => {
  const a = retrievalAgent();
  try {
    const init = await a.conn.initialize(CAPS);
    const err = checkSchema("InitializeResponse", init);
    if (err) throw new Error(err);
    if (init.protocolVersion !== acp.PROTOCOL_VERSION) {
      throw new Error(
        `negotiated v${init.protocolVersion}, client offered v${acp.PROTOCOL_VERSION}`,
      );
    }
    return `protocol v${init.protocolVersion}, agent ${init.agentInfo?.name}`;
  } finally {
    a.kill();
  }
});

check("session/new returns a session id", async () => {
  const a = retrievalAgent();
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const s = await a.conn.newSession({ cwd, mcpServers: [] });
    const err = checkSchema("NewSessionResponse", s);
    if (err) throw new Error(err);
    if (!s.sessionId) throw new Error("empty sessionId");
    return s.sessionId;
  } finally {
    a.kill();
  }
});

check("a retrieval turn streams valid updates and ends cleanly", async () => {
  const a = retrievalAgent();
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    const res = await a.conn.prompt({
      sessionId,
      prompt: [{ type: "text", text: "where is the halting probability computed?" }],
    });
    const err = checkSchema("PromptResponse", res);
    if (err) throw new Error(err);
    if (res.stopReason !== "end_turn") {
      throw new Error(`stopReason was ${res.stopReason}`);
    }
    const calls = updatesOfKind(a.updates, "tool_call");
    if (calls.length < 1) throw new Error("retrieval did not surface as a tool call");
    return `${a.updates.length} updates, ${calls.length} tool calls`;
  } finally {
    a.kill();
  }
});

check("tool call locations are absolute paths", async () => {
  const a = retrievalAgent();
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    await a.conn.prompt({
      sessionId,
      prompt: [{ type: "text", text: "halting probability" }],
    });
    const located = updatesOfKind(a.updates, "tool_call_update")
      .concat(updatesOfKind(a.updates, "tool_call"))
      .flatMap((u) => u.locations || []);
    if (located.length < 1) throw new Error("no locations reported");
    const relative = located.filter((l) => !isAbsolute(l.path));
    if (relative.length) {
      throw new Error(
        `spec requires absolute paths; got ${relative.map((l) => l.path).join(", ")}`,
      );
    }
    return `${located.length} locations, all absolute`;
  } finally {
    a.kill();
  }
});

check("an unknown session is an error the connection survives", async () => {
  const a = retrievalAgent();
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    let threw = false;
    try {
      await a.conn.prompt({
        sessionId: "sess_does_not_exist",
        prompt: [{ type: "text", text: "hello" }],
      });
    } catch {
      threw = true;
    }
    if (!threw) throw new Error("unknown session was accepted");
    const s = await a.conn.newSession({ cwd, mcpServers: [] });
    if (!s.sessionId) throw new Error("connection did not survive the error");
    return "rejected, and the connection still works";
  } finally {
    a.kill();
  }
});

check("session/load replays the conversation", async () => {
  const a = retrievalAgent();
  const cwd = makeWorkspace();
  try {
    const init = await a.conn.initialize(CAPS);
    if (!init.agentCapabilities?.loadSession) {
      throw new Error("agent does not advertise loadSession");
    }
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    await a.conn.prompt({
      sessionId,
      prompt: [{ type: "text", text: "halting probability" }],
    });
    const before = a.updates.length;
    await a.conn.loadSession({ sessionId, cwd, mcpServers: [] });
    const replayed = a.updates.length - before;
    if (replayed < 1) throw new Error("load replayed nothing");
    return `${replayed} updates replayed`;
  } finally {
    a.kill();
  }
});

check("a write asks permission, and the request validates", async () => {
  let seen = null;
  const a = scriptedAgent([toolCall("write_file", { path: "new.py", content: "x = 1\n" }), "Done."], {
    onPermission(params) {
      seen = params;
      return { outcome: { outcome: "selected", optionId: params.options[0].optionId } };
    },
  });
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    await a.conn.prompt({ sessionId, prompt: [{ type: "text", text: "add a file" }] });
    if (!seen) throw new Error("no permission was requested before writing");
    const err = checkSchema("RequestPermissionRequest", seen);
    if (err) throw new Error(err);
    if (!seen.options?.length) throw new Error("no options offered");
    return `asked: ${seen.toolCall?.title}`;
  } finally {
    a.kill();
  }
});

check("rejecting a write prevents it", async () => {
  const a = scriptedAgent(refusedWriteScript("nope.py"), {
    write: true,
    onPermission(params) {
      const reject = params.options.find((o) => o.kind === "reject_once") || params.options[0];
      return { outcome: { outcome: "selected", optionId: reject.optionId } };
    },
  });
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    await a.conn.prompt({ sessionId, prompt: [{ type: "text", text: "write a file" }] });
    if (existsSync(join(cwd, "nope.py"))) {
      throw new Error("the file was written despite a rejection");
    }
    return "no file on disk";
  } finally {
    a.kill();
  }
});

check("cancelling a permission prompt is not consent", async () => {
  const a = scriptedAgent(refusedWriteScript("nope.py"), {
    write: true,
    onPermission() {
      return { outcome: { outcome: "cancelled" } };
    },
  });
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    await a.conn.prompt({ sessionId, prompt: [{ type: "text", text: "write a file" }] });
    if (existsSync(join(cwd, "nope.py"))) {
      throw new Error("a dismissed dialog was treated as approval");
    }
    return "no file on disk";
  } finally {
    a.kill();
  }
});

check("allowing a write performs it", async () => {
  const a = scriptedAgent([toolCall("write_file", { path: "yes.py", content: "x = 1\n" }), "Done."], {
    write: true,
    onPermission(params) {
      const allow = params.options.find((o) => o.kind === "allow_once") || params.options[0];
      return { outcome: { outcome: "selected", optionId: allow.optionId } };
    },
  });
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    await a.conn.prompt({ sessionId, prompt: [{ type: "text", text: "write a file" }] });
    if (!existsSync(join(cwd, "yes.py"))) {
      throw new Error("an approved write did not happen");
    }
    return "file written after approval";
  } finally {
    a.kill();
  }
});

check("an approved write is routed through the client's fs capability", async () => {
  const a = scriptedAgent([toolCall("write_file", { path: "routed.py", content: "x = 1\n" }), "Done."], {
    write: true,
    onPermission: (p) => ({ outcome: { outcome: "selected", optionId: p.options[0].optionId } }),
  });
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    await a.conn.prompt({ sessionId, prompt: [{ type: "text", text: "write a file" }] });

    const writes = a.fsCalls.filter(([kind]) => kind === "write");
    if (!writes.length) {
      throw new Error(
        "the client advertised fs.writeTextFile and the agent wrote to disk " +
        "behind it -- the change never reaches the editor's undo stack",
      );
    }
    return `${writes.length} write(s) via fs/write_text_file`;
  } finally {
    a.kill();
  }
});

check("a read sees the client's unsaved buffer, not disk", async () => {
  const cwd = makeWorkspace();
  const target = join(cwd, "src", "halting.py");
  const a = scriptedAgent([toolCall("read_file", { path: "src/halting.py" }), "Read it."], {
    buffers: { [target]: "UNSAVED_MARKER = True\n" },
    onPermission: () => ({ outcome: { outcome: "cancelled" } }),
  });
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    await a.conn.prompt({ sessionId, prompt: [{ type: "text", text: "read it" }] });

    const shown = a.updates
      .map((u) => JSON.stringify(u.update))
      .join("\n");
    if (!shown.includes("UNSAVED_MARKER")) {
      throw new Error("the agent read through to disk and missed the open buffer");
    }
    return "unsaved buffer was used";
  } finally {
    a.kill();
  }
});

check("session/new advertises a valid mode state", async () => {
  const a = retrievalAgent();
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const s = await a.conn.newSession({ cwd, mcpServers: [] });
    const err = checkSchema("SessionModeState", s.modes);
    if (err) throw new Error(err);
    const ids = s.modes.availableModes.map((m) => m.id);
    if (!ids.includes(s.modes.currentModeId)) {
      throw new Error(`current mode ${s.modes.currentModeId} is not in ${ids}`);
    }
    return `${s.modes.currentModeId} of [${ids.join(", ")}]`;
  } finally {
    a.kill();
  }
});

check("session/set_mode switches what a prompt does", async () => {
  // Starts in ask: the agent was given no --execute. Switching to write must
  // make the same prompt act.
  const a = scriptedAgent([toolCall("write_file", { path: "moded.py", content: "x = 1\n" }), "Done."], {
    execute: false,
    write: true,
    onPermission: (p) => ({ outcome: { outcome: "selected", optionId: p.options[0].optionId } }),
  });
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const s = await a.conn.newSession({ cwd, mcpServers: [] });
    if (s.modes.currentModeId !== "ask") {
      throw new Error(`expected to start in ask, got ${s.modes.currentModeId}`);
    }
    await a.conn.setSessionMode({ sessionId: s.sessionId, modeId: "write" });
    await a.conn.prompt({
      sessionId: s.sessionId,
      prompt: [{ type: "text", text: "add a file" }],
    });
    const wrote = a.fsCalls.some(([kind]) => kind === "write");
    if (!wrote) throw new Error("switching to write did not make the agent act");
    return "ask -> write took effect";
  } finally {
    a.kill();
  }
});

check("an unknown mode is rejected", async () => {
  const a = retrievalAgent();
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    let threw = false;
    try {
      await a.conn.setSessionMode({ sessionId, modeId: "definitely-not-a-mode" });
    } catch {
      threw = true;
    }
    if (!threw) throw new Error("an unknown mode was accepted");
    return "rejected";
  } finally {
    a.kill();
  }
});

check("session/fork branches a session", async () => {
  const a = retrievalAgent();
  const cwd = makeWorkspace();
  try {
    const init = await a.conn.initialize(CAPS);
    if (!init.agentCapabilities?.unstable_forkSession) {
      throw new Error("fork is implemented but not advertised");
    }
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    const forked = await a.conn.unstable_forkSession({ sessionId, cwd, mcpServers: [] });
    const err = checkSchema("ForkSessionResponse", forked);
    if (err) throw new Error(err);
    if (forked.sessionId === sessionId) throw new Error("fork reused the session id");

    // Both halves must still work independently.
    const res = await a.conn.prompt({
      sessionId: forked.sessionId,
      prompt: [{ type: "text", text: "halting probability" }],
    });
    if (res.stopReason !== "end_turn") throw new Error(`fork turn: ${res.stopReason}`);
    return forked.sessionId;
  } finally {
    a.kill();
  }
});

check("session/list returns valid, titled sessions", async () => {
  const a = retrievalAgent();
  const cwd = makeWorkspace();
  try {
    const init = await a.conn.initialize(CAPS);
    if (!init.agentCapabilities?.sessionCapabilities?.list) {
      throw new Error("session/list is implemented but not advertised");
    }
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    await a.conn.prompt({
      sessionId,
      prompt: [{ type: "text", text: "where is the halting probability computed?" }],
    });

    const listed = await a.conn.listSessions({});
    const err = checkSchema("ListSessionsResponse", listed);
    if (err) throw new Error(err);
    const mine = listed.sessions.find((s) => s.sessionId === sessionId);
    if (!mine) throw new Error("the session we just used is not listed");
    if (!mine.title) throw new Error("no title, so a picker shows only ids");
    return `${listed.sessions.length} session(s), titled`;
  } finally {
    a.kill();
  }
});

check("session/list filters by cwd", async () => {
  const a = retrievalAgent();
  const one = makeWorkspace();
  const two = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const first = await a.conn.newSession({ cwd: one, mcpServers: [] });
    await a.conn.newSession({ cwd: two, mcpServers: [] });

    const listed = await a.conn.listSessions({ cwd: one });
    const ids = listed.sessions.map((s) => s.sessionId);
    if (ids.length !== 1 || ids[0] !== first.sessionId) {
      throw new Error(`filter returned ${JSON.stringify(ids)}`);
    }
    return "filtered to one";
  } finally {
    a.kill();
  }
});

check("the agent can ask the user a structured question", async () => {
  let asked = null;
  const a = scriptedAgent(
    [toolCall("ask_user", { question: "Which module did you mean?",
                            choices: ["router", "gate"] }), "Understood."],
    {
      onElicit(params) {
        asked = params;
        return { action: "accept", content: { answer: "gate" } };
      },
    },
  );
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    await a.conn.prompt({ sessionId, prompt: [{ type: "text", text: "fix it" }] });

    if (!asked) {
      throw new Error("the agent never asked. stderr:\n" + a.stderr.join("").slice(-600));
    }
    const err = checkSchema("CreateElicitationRequest", asked);
    if (err) throw new Error(err);
    return `asked: ${asked.message}`;
  } finally {
    a.kill();
  }
});

check("a command runs in the client's terminal", async () => {
  let captured = null;
  const a = scriptedAgent([toolCall("run_command", { command: "python --version" }), "Done."], {
    onPermission: (p) => ({ outcome: { outcome: "selected", optionId: p.options[0].optionId } }),
  });
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    await a.conn.prompt({ sessionId, prompt: [{ type: "text", text: "check python" }] });

    const kinds = a.terminalCalls.map(([k]) => k);
    if (!kinds.includes("create")) {
      throw new Error("the client advertised terminal and the agent used a subprocess");
    }
    if (!kinds.includes("release")) {
      throw new Error("the terminal was never released; the editor keeps it open");
    }
    captured = a.terminals.size;
    if (captured !== 0) throw new Error(`${captured} terminal(s) left open`);
    return kinds.join(" -> ");
  } finally {
    a.kill();
  }
});

check("every session/update validates against the published schema", async () => {
  const a = scriptedAgent([toolCall("read_file", { path: "src/halting.py" }), "Read it."], {
    onPermission: () => ({ outcome: { outcome: "cancelled" } }),
  });
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    await a.conn.prompt({ sessionId, prompt: [{ type: "text", text: "read it" }] });

    const notifications = a.raw.filter((m) => m.method === "session/update");
    if (!notifications.length) throw new Error("no session/update traffic recorded");
    const failures = [];
    for (const n of notifications) {
      const err = checkSchema("SessionNotification", n.params);
      if (err) failures.push(`${n.params?.update?.sessionUpdate}: ${err}`);
    }
    if (failures.length) {
      throw new Error(`${failures.length}/${notifications.length} invalid\n    ` +
        failures.slice(0, 5).join("\n    "));
    }
    return `${notifications.length} notifications, all valid`;
  } finally {
    a.kill();
  }
});

check("no unparseable lines on stdout", async () => {
  const a = retrievalAgent();
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    await a.conn.prompt({
      sessionId,
      prompt: [{ type: "text", text: "halting probability" }],
    });
    const junk = a.raw.filter((m) => m.__unparseable);
    if (junk.length) {
      throw new Error(
        `stdout is the protocol channel; found ${junk.length} non-JSON lines: ` +
          junk.slice(0, 2).map((j) => JSON.stringify(j.__unparseable.slice(0, 80))).join(", "),
      );
    }
    return `${a.raw.length} messages, all well-formed JSON`;
  } finally {
    a.kill();
  }
});

// ------------------------------------------------------------ live provider

liveCheck("a real provider answers a retrieval-grounded question", async () => {
  const a = liveAgent();
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    const res = await a.conn.prompt({
      sessionId,
      prompt: [{ type: "text", text: "What does halting_probability do? Answer in one sentence." }],
    });
    if (res.stopReason !== "end_turn") throw new Error(`stopReason ${res.stopReason}`);
    const text = updatesOfKind(a.updates, "agent_message_chunk")
      .map((u) => (u.content?.type === "text" ? u.content.text : ""))
      .join("");
    if (!text.trim()) {
      throw new Error(`no reply text. stderr: ${a.stderr.join("").slice(-400)}`);
    }
    return `${text.trim().length} chars from ${LIVE_PROVIDER}/${LIVE_MODEL || "default"}`;
  } finally {
    a.kill();
  }
});

liveCheck("a real model's tool calls still go through the permission gate", async () => {
  let asked = 0;
  const a = liveAgent({
    execute: true,
    onPermission(params) {
      asked += 1;
      const reject = params.options.find((o) => o.kind === "reject_once") || params.options[0];
      return { outcome: { outcome: "selected", optionId: reject.optionId } };
    },
  });
  const cwd = makeWorkspace();
  try {
    await a.conn.initialize(CAPS);
    const { sessionId } = await a.conn.newSession({ cwd, mcpServers: [] });
    const res = await a.conn.prompt({
      sessionId,
      prompt: [{ type: "text", text: "Create a file called generated.py containing a add(a, b) function." }],
    });
    // The model may decline to call a tool at all -- that is its choice, not a
    // conformance failure. What must hold is the conditional: if it did try to
    // write, it was asked, and the rejection was honoured.
    if (existsSync(join(cwd, "generated.py"))) {
      throw new Error("a file appeared despite every permission being rejected");
    }
    if (asked) return `${asked} permission requests, all rejected, nothing written`;

    // A turn that did nothing must at least say why. Silence here is the
    // failure mode that made an empty run look like a successful one.
    const log = a.stderr.join("");
    if (/native tool-call/.test(log)) {
      throw new Error(
        "model answered with native OpenAI tool calls, which Knossos does not " +
        "consume -- execute mode cannot act with this model. See the report.",
      );
    }
    if (res.stopReason === "end_turn") {
      throw new Error("no tool call was made, yet the turn reported end_turn");
    }
    return `model made no tool call; turn honestly reported ${res.stopReason}`;
  } finally {
    a.kill();
  }
});

// ----------------------------------------------------------------------- main

const TIMEOUT_MS = 60000;

function withTimeout(promise, name) {
  return Promise.race([
    promise,
    new Promise((_, rej) =>
      setTimeout(() => rej(new Error(`timed out after ${TIMEOUT_MS}ms`)), TIMEOUT_MS),
    ),
  ]).catch((e) => {
    throw new Error(`${name}: ${e.message}`);
  });
}

async function main() {
  const only = process.argv[2];
  let passed = 0;
  let skipped = 0;
  const failed = [];

  console.log(`\nKnossos ACP conformance`);
  console.log(`  client   @agentclientprotocol/sdk (protocol v${acp.PROTOCOL_VERSION})`);
  console.log(`  agent    ${process.env.KNOSSOS_ACP === "python" ? `${PYTHON} -m knossos` : RUST_BIN}`);
  console.log(`  live     ${LIVE ? `${LIVE_PROVIDER}/${LIVE_MODEL || "default"}` : "off"}\n`);

  for (const c of checks) {
    if (only && !c.name.includes(only)) continue;
    process.stdout.write(`  ${c.name} ... `);
    if (c.skipped) {
      if (STRICT) {
        failed.push({ name: c.name, message: "skipped, but KNOSSOS_STRICT is set" });
        console.log("FAILED (skipped under KNOSSOS_STRICT)");
        continue;
      }
      skipped += 1;
      console.log("skipped (set KNOSSOS_LIVE=1)");
      continue;
    }
    try {
      const detail = await withTimeout(c.fn(), c.name);
      passed += 1;
      console.log(`ok${detail ? `  (${detail})` : ""}`);
    } catch (e) {
      failed.push({ name: c.name, message: e.message });
      console.log(`FAILED`);
      console.log(`      ${e.message.split("\n").join("\n      ")}`);
    }
  }

  const tail = skipped ? `, ${skipped} skipped` : "";
  console.log(`\n  ${passed} passed, ${failed.length} failed${tail}`);
  if (skipped) {
    // Loud on purpose. A skipped cross-harness check reads as a pass in every
    // summary that does not say this.
    console.log(
      `\n  !!! ${skipped} check(s) DID NOT RUN and proved nothing.` +
      `\n      Set KNOSSOS_LIVE=1 (with OLLAMA_HOST / KNOSSOS_LIVE_MODEL) to run them,` +
      `\n      or KNOSSOS_STRICT=1 to make an unconfigured skip a failure.`,
    );
  }
  console.log("");
  process.exit(failed.length ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
