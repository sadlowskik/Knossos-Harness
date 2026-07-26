// Webview front end. Renders events streamed from `daedalus serve` and sends
// user intent back to the extension host.
//
// No framework: the DOM here is small enough that one would be overhead, and
// a CSP-restricted webview is the wrong place for a bundler.

(function () {
  "use strict";

  const vscode = acquireVsCodeApi();

  const log = document.getElementById("log");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const resetBtn = document.getElementById("reset");
  const verifyBtn = document.getElementById("verify");
  const busy = document.getElementById("busy");
  const statusLeft = document.getElementById("status-left");
  const statusRight = document.getElementById("status-right");
  const mentions = document.getElementById("mentions");

  let idle = false;
  let started = false;

  /** Workspace files, for @-mentions. Sent by the extension on startup. */
  let fileList = [];
  let matches = [];
  let active = -1;

  function scroll() {
    log.scrollTop = log.scrollHeight;
  }

  function add(className, build) {
    const el = document.createElement("div");
    el.className = className;
    build(el);
    log.appendChild(el);
    scroll();
    return el;
  }

  function text(className, content) {
    return add(className, (el) => {
      el.textContent = content;
    });
  }

  function heading(el, label) {
    const h = document.createElement("h4");
    h.textContent = label;
    el.appendChild(h);
  }

  function setBusy(isBusy) {
    idle = !isBusy;
    sendBtn.disabled = isBusy;
    verifyBtn.disabled = isBusy;
    resetBtn.disabled = isBusy;
    busy.textContent = isBusy ? "working…" : "";
  }

  function send() {
    const value = input.value.trim();
    if (!value || !idle) {
      return;
    }
    text("msg user", value);
    input.value = "";
    setBusy(true);
    vscode.postMessage({ type: started ? "resume" : "task", text: value });
    started = true;
  }

  sendBtn.addEventListener("click", send);

  // ---- @-mention picker ----

  /** The `@partial` immediately before the cursor, if any. */
  function mentionQuery() {
    const upto = input.value.slice(0, input.selectionStart);
    const at = upto.lastIndexOf("@");
    if (at === -1) {
      return null;
    }
    // Must start a word, and must not span whitespace.
    if (at > 0 && !/\s/.test(upto[at - 1])) {
      return null;
    }
    const fragment = upto.slice(at + 1);
    if (/\s/.test(fragment)) {
      return null;
    }
    return { at, fragment };
  }

  /** Rank by basename prefix, then basename, then full path. */
  function rank(query) {
    if (!query) {
      return fileList.slice(0, 20);
    }
    const q = query.toLowerCase();
    const scored = [];
    for (const path of fileList) {
      const lower = path.toLowerCase();
      const base = lower.slice(lower.lastIndexOf("/") + 1);
      let score = -1;
      if (base.startsWith(q)) {
        score = 0;
      } else if (base.includes(q)) {
        score = 1;
      } else if (lower.includes(q)) {
        score = 2;
      }
      if (score >= 0) {
        scored.push({ path, score });
      }
    }
    scored.sort((a, b) => a.score - b.score || a.path.length - b.path.length);
    return scored.slice(0, 20).map((s) => s.path);
  }

  function hideMentions() {
    mentions.hidden = true;
    matches = [];
    active = -1;
  }

  function showMentions() {
    const query = mentionQuery();
    if (!query || fileList.length === 0) {
      hideMentions();
      return;
    }

    matches = rank(query.fragment);
    if (matches.length === 0) {
      hideMentions();
      return;
    }

    active = 0;
    mentions.replaceChildren();
    matches.forEach((path, i) => {
      const item = document.createElement("div");
      item.className = i === 0 ? "item active" : "item";
      item.textContent = path;
      item.addEventListener("mousedown", (e) => {
        // mousedown, not click: the textarea must not lose focus first.
        e.preventDefault();
        choose(i);
      });
      mentions.appendChild(item);
    });
    mentions.hidden = false;
  }

  function highlight(next) {
    if (matches.length === 0) {
      return;
    }
    active = (next + matches.length) % matches.length;
    Array.from(mentions.children).forEach((el, i) => {
      el.className = i === active ? "item active" : "item";
    });
    mentions.children[active]?.scrollIntoView({ block: "nearest" });
  }

  function choose(index) {
    const query = mentionQuery();
    if (!query || !matches[index]) {
      return;
    }
    const before = input.value.slice(0, query.at);
    const after = input.value.slice(input.selectionStart);
    const inserted = `@${matches[index]} `;
    input.value = before + inserted + after;

    const caret = before.length + inserted.length;
    input.setSelectionRange(caret, caret);
    hideMentions();
    input.focus();
  }

  input.addEventListener("input", showMentions);
  input.addEventListener("blur", hideMentions);

  input.addEventListener("keydown", (e) => {
    if (!mentions.hidden) {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          highlight(active + 1);
          return;
        case "ArrowUp":
          e.preventDefault();
          highlight(active - 1);
          return;
        case "Enter":
        case "Tab":
          e.preventDefault();
          choose(active);
          return;
        case "Escape":
          e.preventDefault();
          hideMentions();
          return;
        default:
          break;
      }
    }

    // Enter sends; Shift+Enter is a newline.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  resetBtn.addEventListener("click", () => {
    if (!idle) {
      return;
    }
    setBusy(true);
    vscode.postMessage({ type: "reset" });
  });

  verifyBtn.addEventListener("click", () => {
    if (!idle) {
      return;
    }
    setBusy(true);
    vscode.postMessage({ type: "verify" });
  });

  function renderPlan(steps) {
    add("msg agent", (el) => {
      heading(el, "Plan");
      const ol = document.createElement("ol");
      for (const step of steps) {
        const li = document.createElement("li");
        li.textContent = step;
        ol.appendChild(li);
      }
      el.appendChild(ol);
    });
  }

  function stat(added, removed) {
    const el = document.createElement("span");
    el.className = "stat";
    const plus = document.createElement("span");
    plus.className = "add";
    plus.textContent = `+${added}`;
    const minus = document.createElement("span");
    minus.className = "del";
    minus.textContent = ` -${removed}`;
    el.appendChild(plus);
    el.appendChild(minus);
    return el;
  }

  /** Colourize a hunk body by its +/-/space prefixes. */
  function hunkBody(body) {
    const pre = document.createElement("pre");
    pre.hidden = true;
    for (const line of body.split("\n")) {
      const span = document.createElement("span");
      if (line.startsWith("+")) {
        span.className = "add";
      } else if (line.startsWith("-")) {
        span.className = "del";
      }
      span.textContent = line + "\n";
      pre.appendChild(span);
    }
    return pre;
  }

  function renderDiffs(files) {
    if (!files || files.length === 0) {
      text("msg note", "No changes proposed.");
      return;
    }

    // Checkbox state lives on the elements; this collects it at click time.
    const boxes = [];

    add("msg diffs", (el) => {
      const total = files.reduce((n, f) => n + (f.hunks ? f.hunks.length : 0), 0);
      heading(el, `${files.length} file(s), ${total} hunk(s) proposed`);

      for (const file of files) {
        const row = document.createElement("div");
        row.className = "file";

        const name = document.createElement("span");
        name.className = "name";
        name.textContent = file.path + (file.existed ? "" : "  (new)");
        name.title = "Open side-by-side diff";
        name.addEventListener("click", () => {
          vscode.postMessage({ type: "openDiff", path: file.path });
        });

        row.appendChild(name);
        row.appendChild(stat(file.added, file.removed));
        el.appendChild(row);

        const hunks = document.createElement("div");
        hunks.className = "hunks";

        for (const hunk of file.hunks || []) {
          const item = document.createElement("div");
          item.className = "hunk";

          const head = document.createElement("div");
          head.className = "hunk-head";

          const label = document.createElement("label");
          const box = document.createElement("input");
          box.type = "checkbox";
          box.checked = true;
          boxes.push({ path: file.path, id: hunk.id, box });

          const caption = document.createElement("span");
          caption.textContent = hunk.header;
          label.appendChild(box);
          label.appendChild(caption);
          label.appendChild(stat(hunk.added, hunk.removed));

          const toggle = document.createElement("span");
          toggle.className = "toggle";
          toggle.textContent = "show";

          head.appendChild(label);
          head.appendChild(toggle);
          item.appendChild(head);

          const body = hunkBody(hunk.body);
          item.appendChild(body);
          toggle.addEventListener("click", () => {
            body.hidden = !body.hidden;
            toggle.textContent = body.hidden ? "show" : "hide";
            scroll();
          });

          hunks.appendChild(item);
        }

        if (hunks.childElementCount > 0) {
          el.appendChild(hunks);
        }
      }

      const actions = document.createElement("div");
      actions.className = "actions";

      const selected = document.createElement("button");
      selected.textContent = "Accept selected";
      selected.addEventListener("click", () => {
        // Group the ticked hunks by file.
        const byFile = new Map();
        for (const entry of boxes) {
          if (!entry.box.checked) {
            continue;
          }
          if (!byFile.has(entry.path)) {
            byFile.set(entry.path, []);
          }
          byFile.get(entry.path).push(entry.id);
        }
        if (byFile.size === 0) {
          return;
        }
        setBusy(true);
        vscode.postMessage({
          type: "applyHunks",
          selection: Array.from(byFile, ([path, hunks]) => ({ path, hunks })),
        });
      });

      const acceptAll = document.createElement("button");
      acceptAll.className = "secondary";
      acceptAll.textContent = "Accept all";
      acceptAll.addEventListener("click", () => {
        setBusy(true);
        vscode.postMessage({ type: "apply" });
      });

      const reject = document.createElement("button");
      reject.className = "secondary";
      reject.textContent = "Reject all";
      reject.addEventListener("click", () => {
        setBusy(true);
        vscode.postMessage({ type: "discard" });
      });

      actions.appendChild(selected);
      actions.appendChild(acceptAll);
      actions.appendChild(reject);
      el.appendChild(actions);
    });
  }

  function renderVerdict(event) {
    add("msg agent", (el) => {
      heading(el, event.passed ? "Verification passed" : "Verification failed");
      for (const tier of event.tiers || []) {
        const line = document.createElement("div");
        line.className = "trace";
        const mark = document.createElement("span");
        mark.className = tier.passed ? "ok" : "bad";
        mark.textContent = tier.passed ? "PASS" : "FAIL";
        line.appendChild(mark);
        line.appendChild(document.createTextNode(`  tier ${tier.tier} — ${tier.label}`));
        el.appendChild(line);

        if (!tier.passed && tier.detail) {
          const detail = document.createElement("div");
          detail.className = "trace bad";
          detail.textContent = tier.detail;
          el.appendChild(detail);
        }
      }
      if (event.dry_run) {
        const note = document.createElement("div");
        note.className = "trace";
        note.textContent =
          "Preview only — nothing is on disk, so the compiler and tests could not run.";
        el.appendChild(note);
      }
    });
  }

  function trace(content, cls) {
    add(`trace${cls ? " " + cls : ""}`, (el) => {
      el.textContent = content;
    });
  }

  window.addEventListener("message", (e) => {
    const msg = e.data;
    if (msg.type === "files") {
      fileList = msg.files || [];
      return;
    }
    if (msg.type !== "daedalus") {
      return;
    }
    const ev = msg.event;

    switch (ev.event) {
      case "ready":
        statusLeft.textContent = `${ev.engine} · ${ev.symbols} symbols`;
        statusRight.textContent = ev.dry_run ? "PREVIEW" : "WRITES ENABLED";
        statusRight.className = ev.dry_run ? "mode" : "";
        text(
          "msg note",
          ev.dry_run
            ? "Preview mode — changes are staged, nothing is written until you accept."
            : "Writes are enabled — changes go straight to disk."
        );
        break;

      case "idle":
        setBusy(false);
        break;

      case "plan":
        renderPlan(ev.steps || []);
        break;

      case "step_started":
        trace(`— step ${ev.index}`);
        break;

      case "tool_call":
        trace(`  ${ev.tool}${ev.is_error ? " (failed)" : ""}`, ev.is_error ? "bad" : undefined);
        break;

      case "oracle_verdict":
        trace(`  ${ev.passed ? "PASS" : "FAIL"} ${ev.tier}`, ev.passed ? "ok" : "bad");
        break;

      case "outcome":
        add("msg agent", (el) => {
          heading(el, ev.succeeded ? "Done" : ev.halt.replace(/_/g, " "));
          el.appendChild(document.createTextNode(ev.summary));
        });
        break;

      case "diffs":
        renderDiffs(ev.files);
        break;

      case "applied":
        text(
          "msg note",
          ev.files.length
            ? `Wrote ${ev.files.length} file(s). Run Verify — the ladder could only check syntax while unwritten.`
            : "Nothing staged."
        );
        break;

      case "discarded":
        text("msg note", "Proposed changes discarded.");
        break;

      case "verdict":
        renderVerdict(ev);
        break;

      case "index":
        text("msg note", `${ev.symbols} symbols across ${ev.files} files`);
        break;

      case "reset":
        log.replaceChildren();
        started = false;
        text("msg note", "Conversation cleared.");
        break;

      case "error":
        text("msg error", ev.message);
        break;

      case "exited":
        text("msg error", `The harness process exited (${ev.code}).`);
        setBusy(true);
        sendBtn.disabled = true;
        break;

      default:
        break;
    }
  });

  setBusy(true);
  vscode.postMessage({ type: "ready" });
})();
