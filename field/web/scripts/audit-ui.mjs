#!/usr/bin/env node
/* audit-ui — the design audit, as a program.

   The findings this enforces were each measured by hand once, in the running app, and
   each of them had already survived a pass that claimed to fix it: the border token was
   raised to a value that was still invisible, the type floor was declared at eleven
   pixels and shipped at ten, the surface ramp stepped by six per cent. A number nobody
   re-measures drifts back. So the measurements live here.

   Plain Node, no dependencies, and deliberately a lexer rather than a parser: it reads
   the stylesheets and the JSX as text, which is enough for every rule below and cannot
   break on a syntax it has not met.

   Usage:  node scripts/audit-ui.mjs [--json]
   Exit:   0 when nothing is over baseline, 1 otherwise.  */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(HERE, '..');
const SRC = path.join(WEB, 'src');
const BASELINE_FILE = path.join(HERE, 'audit-ui.baseline.json');
const JSON_OUT = process.argv.includes('--json');

/* The minimum type size that ships. Ornament may be quiet; nothing may be unreadable. */
const TYPE_FLOOR = 11;
/* An element with no children of its own and more than this many runs of text is a
   console habit: every number that is available, printed at once. */
const MAX_ADJACENT_RUNS = 3;

/* A screen is what renders with no further interaction: the bar, the map or the board,
   and whatever is open beside them. A modal, an overlay or a settings sheet is a tap
   deeper and is its own group, because it is never read at the same time as another
   one. A string repeated inside a group is the same fact said twice in one glance; the
   same string in two groups is two screens each saying it once, which is correct. */
const SCREENS = {
  rome: [
    'App.jsx', 'theater/TheaterMode.jsx', 'theater/FolderDetail.jsx',
    'theater/TimeControl.jsx', 'theater/IslandPlate.jsx',
    'ui/ReplayRail.jsx', 'ui/Conversation.jsx', 'ui/WorkCard.jsx', 'ui/EmptyState.jsx',
  ],
  atlas: [
    'App.jsx', 'atlas/AtlasMode.jsx', 'hud/PermissionRequests.jsx',
    'ui/Conversation.jsx', 'ui/WorkCard.jsx', 'ui/EmptyState.jsx',
  ],
  settings: ['theater/FieldSettings.jsx', 'routines/RoutinesPanel.jsx'],
  plans: ['campaigns/PlansOverlay.jsx'],
  starter: ['hud/ContextMenu.jsx'],
  models: ['setup/PowerSources.jsx'],
  folder: ['theater/FolderWorkspace.jsx', 'workspace/panels.jsx', 'ui/FilePicker.jsx', 'ui/Transcript.jsx'],
};

/* The contrast pairs, by name, with the threshold each has to clear and the backdrop
   each is composited over. Colours are resolved from the token table; an alpha that
   lives in a rule rather than a token is read from that rule, so changing the rule
   changes the measurement. WCAG ratio throughout. */
const CONTRAST_PAIRS = [
  { name: 'border-subtle on surface-1', fg: '--border-subtle', bg: '--surface-1', min: 1.6 },
  { name: 'border on surface-1', fg: '--border', bg: '--surface-1', min: 1.6 },
  { name: 'border-strong on surface-2', fg: '--border-strong', bg: '--surface-2', min: 1.6 },
  { name: 'surface-1 over surface-0', fg: '--surface-1', bg: '--surface-0', min: 1.15 },
  { name: 'surface-2 over surface-1', fg: '--surface-2', bg: '--surface-1', min: 1.15 },
  { name: 'surface-3 over surface-2', fg: '--surface-3', bg: '--surface-2', min: 1.15 },
  { name: 'text-1 on surface-1', fg: '--text-1', bg: '--surface-1', min: 4.5 },
  { name: 'text-2 on surface-1', fg: '--text-2', bg: '--surface-1', min: 4.5 },
  { name: 'text-3 on surface-1', fg: '--text-3', bg: '--surface-1', min: 4.5 },
  {
    name: 'land against sea',
    fg: '--isle-land',
    bg: '--isle-sea',
    bgAlpha: { selector: '.island-sea', prop: 'opacity' },
    bgOver: '--isle-paper',
    min: 1.6,
  },
  {
    name: 'shelf band against sea',
    fg: '--isle-sea-deep',
    fgAlpha: { selector: '.island-depth.band-shelf path', prop: 'opacity' },
    bg: '--isle-sea',
    bgAlpha: { selector: '.island-sea', prop: 'opacity' },
    bgOver: '--isle-paper',
    min: 1.5,
  },
  {
    name: 'outer shore against sea',
    fg: '#FBF6E8',
    fgAlpha: { selector: '.island-shore-outer path', prop: 'opacity' },
    bg: '--isle-sea',
    bgAlpha: { selector: '.island-sea', prop: 'opacity' },
    bgOver: '--isle-paper',
    min: 1.2,
  },
  { name: 'map label on land', fg: '--isle-label', bg: '--isle-land', min: 4.5 },
];

// ------------------------------------------------------------------ reading the tree

function walk(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

const files = walk(SRC).map((full) => ({
  full,
  rel: path.relative(SRC, full).replaceAll('\\', '/'),
  ext: path.extname(full),
  text: fs.readFileSync(full, 'utf8'),
}));
const cssFiles = files.filter((f) => f.ext === '.css');
const jsxFiles = files.filter((f) => f.ext === '.jsx');

/* Comments are prose about the code, not code. Every rule below reads the stripped
   text and reports line numbers from the original, so a sentence in a comment can
   mention `#262B31` or a 10px type step without being counted as one. */
const stripCssComments = (text) => text.replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, ' '));
const stripJsComments = (text) => text
  .replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, ' '))
  .replace(/(^|[^:"'`\\])\/\/[^\n]*/g, (m, lead) => lead + ' '.repeat(m.length - lead.length));

const lineOf = (text, index) => text.slice(0, index).split('\n').length;

for (const file of cssFiles) file.code = stripCssComments(file.text);
for (const file of jsxFiles) file.code = stripJsComments(file.text);

// ------------------------------------------------------------------ colour arithmetic

function parseColour(value) {
  const v = String(value).trim();
  let m = /^#([0-9a-f]{3})$/i.exec(v);
  if (m) return [...m[1]].map((c) => parseInt(c + c, 16)).concat(1);
  m = /^#([0-9a-f]{6})$/i.exec(v);
  if (m) return [1, 3, 5].map((i) => parseInt(m[1].slice(i - 1, i + 1), 16)).concat(1);
  m = /^#([0-9a-f]{8})$/i.exec(v);
  if (m) return [1, 3, 5].map((i) => parseInt(m[1].slice(i - 1, i + 1), 16)).concat(parseInt(m[1].slice(6, 8), 16) / 255);
  m = /^rgba?\(([^)]+)\)$/i.exec(v);
  if (m) {
    const parts = m[1].split(/[,\s/]+/).filter(Boolean).map(Number);
    if (parts.length >= 3 && parts.slice(0, 3).every(Number.isFinite)) {
      return [parts[0], parts[1], parts[2], Number.isFinite(parts[3]) ? parts[3] : 1];
    }
  }
  m = /^hsla?\(([^)]+)\)$/i.exec(v);
  if (m) {
    const parts = m[1].split(/[,\s/]+/).filter(Boolean);
    const h = parseFloat(parts[0]); const s = parseFloat(parts[1]) / 100; const l = parseFloat(parts[2]) / 100;
    if ([h, s, l].every(Number.isFinite)) {
      const c = (1 - Math.abs(2 * l - 1)) * s;
      const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
      const mm = l - c / 2;
      const seg = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]][Math.floor(((h % 360) + 360) % 360 / 60)];
      const a = parts[3] === undefined ? 1 : parseFloat(parts[3]);
      return [...seg.map((n) => Math.round((n + mm) * 255)), Number.isFinite(a) ? a : 1];
    }
  }
  return null;
}

const over = (fg, bg) => [0, 1, 2].map((i) => fg[i] * fg[3] + bg[i] * (1 - fg[3])).concat(1);

function luminance([r, g, b]) {
  const f = (v) => { const n = v / 255; return n <= 0.03928 ? n / 12.92 : ((n + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}

function ratio(a, b) {
  const [l1, l2] = [luminance(a), luminance(b)];
  return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
}

// ------------------------------------------------------------------ the token table

/* Every `--name: value` declared anywhere, then `var()` resolved to a literal. A token
   defined in more than one place keeps the last definition, which is what the cascade
   does for the single `:root` block this project has. */
const tokens = new Map();
for (const file of cssFiles) {
  for (const m of file.code.matchAll(/(--[\w-]+)\s*:\s*([^;{}]+);/g)) {
    tokens.set(m[1], m[2].trim());
  }
}

function resolve(value, depth = 0) {
  if (depth > 12 || typeof value !== 'string') return value;
  const m = /var\(\s*(--[\w-]+)\s*(?:,\s*([^)]+))?\)/.exec(value);
  if (!m) return value.trim();
  const replacement = tokens.has(m[1]) ? tokens.get(m[1]) : (m[2] ?? '');
  return resolve(value.slice(0, m.index) + replacement + value.slice(m.index + m[0].length), depth + 1);
}

const tokenColour = (name) => {
  const raw = name.startsWith('--') ? tokens.get(name) : name;
  return raw === undefined ? null : parseColour(resolve(raw));
};

/* Every literal colour any token resolves to. A raw literal in a normal declaration is
   allowed when it is a value the token system already names — that is the same colour,
   spelled out — and a finding when it is a colour nothing names. */
const tokenPalette = new Set();
for (const [name, raw] of tokens) {
  if (!/color|ink|paper|sea|land|accent|surface|border|text|st-|isle|world|edge|line|bg/.test(name)) continue;
  const parsed = parseColour(resolve(raw));
  if (parsed) tokenPalette.add(parsed.join(','));
}

/* One declared value inside one rule, by selector. Used by the contrast pairs to read
   an alpha that lives in a stylesheet rather than in a token. */
function declared(selector, prop) {
  for (const file of cssFiles) {
    const at = file.code.indexOf(`${selector} {`);
    if (at < 0) continue;
    const end = file.code.indexOf('}', at);
    const m = new RegExp(`(?:^|[;{\\s])${prop}\\s*:\\s*([^;}]+)`).exec(file.code.slice(at, end));
    if (m) return resolve(m[1].trim());
  }
  return null;
}

// ------------------------------------------------------------------ the rules

const findings = [];
const add = (rule, where, message) => findings.push({ rule, where, message });

// ---- 1. type floor ---------------------------------------------------------------
{
  const seen = new Set();
  for (const file of cssFiles) {
    // font-size, the `font:` shorthand, and the type-scale tokens themselves.
    const patterns = [
      /font-size\s*:\s*([^;}]+)/g,
      /(?:^|[;{\s])font\s*:\s*([^;}]+)/g,
      /(--fs-[\w-]+)\s*:\s*([^;}]+)/g,
    ];
    for (const [index, pattern] of patterns.entries()) {
      for (const m of file.code.matchAll(pattern)) {
        const value = resolve(index === 2 ? m[2] : m[1]);
        for (const px of value.matchAll(/(\d+(?:\.\d+)?)px/g)) {
          const size = Number(px[1]);
          // The `font:` shorthand also carries line heights and weights; only a value
          // in the size slot counts, which is the one followed by `/` or a family.
          if (index === 1 && !/\/|\s/.test(value.slice(px.index + px[0].length, px.index + px[0].length + 2))) continue;
          if (size >= TYPE_FLOOR || size < 4) continue;
          const key = `${file.rel}:${m[0].trim()}`;
          if (seen.has(key)) continue;
          seen.add(key);
          add('type-floor', `${file.rel}:${lineOf(file.code, m.index)}`,
            `${size}px is under the ${TYPE_FLOOR}px floor — ${m[0].trim().slice(0, 70)}`);
        }
      }
    }
  }
}

// ---- 2. raw colour literals ------------------------------------------------------
/* Counted, not blocked: the stylesheet has a history and a hard failure here would be
   a wall nobody climbs. The count is recorded in the repo and may only fall. */
const rawColours = [];
{
  const literal = /#[0-9a-fA-F]{3,8}\b|\brgba?\([^)]*\)|\bhsla?\([^)]*\)/g;
  for (const file of cssFiles) {
    for (const m of file.code.matchAll(literal)) {
      // Inside a custom-property declaration is where a literal belongs.
      const lineStart = file.code.lastIndexOf('\n', m.index) + 1;
      const line = file.code.slice(lineStart, file.code.indexOf('\n', m.index));
      const decl = /(--[\w-]+)\s*:/.exec(file.code.slice(lineStart, m.index));
      if (decl) continue;
      // A url() payload is an embedded image, not a colour choice.
      if (/url\(/.test(line.slice(0, m.index - lineStart)) && /\)/.test(line.slice(m.index - lineStart))) continue;
      const parsed = parseColour(m[0]);
      if (parsed && tokenPalette.has(parsed.join(','))) continue;
      rawColours.push({ where: `${file.rel}:${lineOf(file.code, m.index)}`, value: m[0] });
    }
  }
  for (const file of jsxFiles) {
    for (const m of file.code.matchAll(/#[0-9a-fA-F]{6}\b/g)) {
      rawColours.push({ where: `${file.rel}:${lineOf(file.code, m.index)}`, value: m[0] });
    }
  }
}

// ---- 3. native controls without an appearance rule -------------------------------
{
  // Every selector that declares `appearance`, and the class names inside it.
  const appearanceRules = [];
  for (const file of cssFiles) {
    for (const m of file.code.matchAll(/([^{}]+)\{([^{}]*appearance\s*:[^{}]*)\}/g)) {
      for (const selector of m[1].split(',')) {
        appearanceRules.push({
          selector: selector.trim(),
          classes: [...selector.matchAll(/\.([\w-]+)/g)].map((c) => c[1]),
        });
      }
    }
  }
  const NATIVE = [
    { re: /<select\b/g, name: 'select', match: /(^|[\s>])select\b/ },
    { re: /<input\b[^>]*type=(["'])range\1/g, name: 'input[type=range]', match: /input\[type=["']?range/ },
    { re: /<input\b[^>]*type=(["'])checkbox\1/g, name: 'input[type=checkbox]', match: /input\[type=["']?checkbox/ },
    { re: /<input\b[^>]*type=(["'])radio\1/g, name: 'input[type=radio]', match: /input\[type=["']?radio/ },
  ];
  for (const file of jsxFiles) {
    const fileClasses = new Set([...file.code.matchAll(/className=["']([^"']+)["']/g)]
      .flatMap((m) => m[1].split(/\s+/))
      .concat([...file.code.matchAll(/className=\{`([^`]*)`/g)].flatMap((m) => m[1].split(/[\s${}]+/)))
      .filter(Boolean));
    for (const kind of NATIVE) {
      for (const m of file.code.matchAll(kind.re)) {
        // The element's own classes, if it has any, within its open tag.
        const tagEnd = file.code.indexOf('>', m.index);
        const tag = file.code.slice(m.index, tagEnd < 0 ? m.index + 400 : tagEnd);
        const own = [...tag.matchAll(/className=["'{`]([^"'`}]*)/g)].flatMap((c) => c[1].split(/\s+/)).filter(Boolean);
        const covered = appearanceRules.some((rule) => {
          if (!kind.match.test(rule.selector)) return false;
          if (rule.classes.length === 0) return true;                       // a bare element rule
          if (own.some((c) => rule.classes.includes(c))) return true;       // its own class
          return rule.classes.every((c) => fileClasses.has(c));             // an ancestor's, in this file
        });
        if (!covered) {
          add('native-control', `${file.rel}:${lineOf(file.code, m.index)}`,
            `${kind.name} renders the platform's control — no rule sets \`appearance\``);
        }
      }
    }
  }
}

// ---- 4. adjacent text runs -------------------------------------------------------
{
  // Leaf elements only: an open tag, content with no further tags, the close tag.
  const leaf = /<([A-Za-z][\w.]*)((?:[^<>{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)>((?:[^<>{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)<\/\1>/g;
  for (const file of jsxFiles) {
    for (const m of file.code.matchAll(leaf)) {
      const inner = m[3];
      if (inner.includes('<')) continue;
      let runs = 0; let depth = 0; let text = '';
      for (const ch of inner) {
        if (ch === '{') { if (depth === 0) { if (text.trim()) runs += 1; text = ''; runs += 1; } depth += 1; continue; }
        if (ch === '}') { depth = Math.max(0, depth - 1); continue; }
        if (depth === 0) text += ch;
      }
      if (text.trim()) runs += 1;
      if (runs > MAX_ADJACENT_RUNS) {
        add('text-runs', `${file.rel}:${lineOf(file.code, m.index)}`,
          `<${m[1]}> prints ${runs} runs of text with no child element — the budget is ${MAX_ADJACENT_RUNS}`);
      }
    }
  }
}

// ---- 5. a string rendered at more than one site in a screen ----------------------
{
  // JSX text: what sits between the `>` that closes a tag and the next `<`. The `>` has
  // to belong to a tag, or `=>` in ordinary code reads as one; and the text has to be
  // prose, or an expression's operators read as words.
  const sites = new Map();
  for (const file of jsxFiles) {
    for (const m of file.code.matchAll(/>([^<>{}]*[A-Za-z]{3}[^<>{}]*)</g)) {
      const open = file.code.lastIndexOf('<', m.index);
      if (open < 0 || file.code.slice(open + 1, m.index).includes('>')) continue;
      if (!/^\/?[A-Za-z]/.test(file.code.slice(open + 1, open + 2 + 1))) continue;
      const value = m[1].replace(/\s+/g, ' ').trim();
      if (value.length < 5) continue;
      if (/[(){};=|&`]/.test(value)) continue;
      if (!/[A-Za-z]{3}/.test(value)) continue;
      const key = value.toLowerCase();
      if (!sites.has(key)) sites.set(key, []);
      sites.get(key).push({ file: file.rel, line: lineOf(file.code, m.index), value });
    }
  }
  for (const [, uses] of sites) {
    if (uses.length < 2) continue;
    for (const [screen, globs] of Object.entries(SCREENS)) {
      const inScreen = uses.filter((u) => globs.some((g) => u.file === g || u.file.startsWith(g)));
      if (inScreen.length < 2) continue;
      add('duplicate-string', inScreen.map((u) => `${u.file}:${u.line}`).join(' and '),
        `"${inScreen[0].value.slice(0, 52)}" is rendered at ${inScreen.length} sites on ${screen}`);
      break;
    }
  }
}

// ---- 6. the contrast pairs -------------------------------------------------------
const contrast = [];
{
  const alphaOf = (spec) => {
    if (spec === undefined) return 1;
    const value = declared(spec.selector, spec.prop);
    return value === null ? 1 : Number(value);
  };
  for (const pair of CONTRAST_PAIRS) {
    let bg = tokenColour(pair.bg);
    let fg = tokenColour(pair.fg);
    if (!bg || !fg) { add('contrast', pair.name, 'a colour in this pair no longer resolves to a value'); continue; }
    if (pair.bgOver) {
      const under = tokenColour(pair.bgOver);
      if (under) bg = over([bg[0], bg[1], bg[2], alphaOf(pair.bgAlpha)], under);
    }
    if (pair.fgAlpha) fg = over([fg[0], fg[1], fg[2], alphaOf(pair.fgAlpha)], bg);
    if (fg[3] < 1) fg = over(fg, bg);
    const value = ratio(fg, bg);
    contrast.push({ name: pair.name, value: Number(value.toFixed(3)), min: pair.min, pass: value >= pair.min });
    if (value < pair.min) {
      add('contrast', pair.name, `${value.toFixed(2)}:1 against a ${pair.min}:1 threshold`);
    }
  }
}

// ------------------------------------------------------------------ baseline & report

/* The ratchet. Contrast and the type floor are absolute: they are the measurements this
   pass was run to fix and there is no debt to carry. The three population rules — raw
   colour literals, text runs, repeated strings — each carry a recorded count from the
   panels this pass did not touch, so the audit is green today and can only get greener.
   Raising a number in the baseline file is the deliberate act it should be. */
const counts = {
  rawColours: rawColours.length,
  'text-runs': findings.filter((f) => f.rule === 'text-runs').length,
  'duplicate-string': findings.filter((f) => f.rule === 'duplicate-string').length,
};
const RATCHETED = new Set(['text-runs', 'duplicate-string']);
const baseline = fs.existsSync(BASELINE_FILE)
  ? JSON.parse(fs.readFileSync(BASELINE_FILE, 'utf8'))
  : counts;

const overBaseline = Object.entries(counts).filter(([k, v]) => v > (baseline[k] ?? 0));
const hard = findings.filter((f) => !RATCHETED.has(f.rule));
const failed = hard.length > 0 || overBaseline.length > 0;

if (JSON_OUT) {
  process.stdout.write(`${JSON.stringify({
    ok: !failed, findings, contrast, counts, baseline,
    rawColours: { sample: rawColours.slice(0, 20) },
    files: { css: cssFiles.length, jsx: jsxFiles.length, tokens: tokens.size },
  }, null, 2)}\n`);
  process.exit(failed ? 1 : 0);
}

const pad = (s, n) => String(s).padEnd(n);
const out = [];
out.push('');
out.push(`Field UI audit — ${cssFiles.length} stylesheets, ${jsxFiles.length} components, ${tokens.size} tokens`);
out.push('');
out.push('  contrast');
for (const row of contrast) {
  out.push(`    ${row.pass ? 'ok  ' : 'FAIL'}  ${pad(row.name, 30)} ${pad(`${row.value.toFixed(2)}:1`, 9)} needs ${row.min}:1`);
}
out.push('');
out.push('  raw colour literals outside the token set');
out.push(`    ${counts.rawColours > (baseline.rawColours ?? 0) ? 'FAIL' : 'ok  '}  ${counts.rawColours} found, baseline ${baseline.rawColours ?? 0}`);
out.push('');

const byRule = new Map();
for (const f of findings) {
  if (f.rule === 'contrast') continue;
  if (!byRule.has(f.rule)) byRule.set(f.rule, []);
  byRule.get(f.rule).push(f);
}
const RULES = ['type-floor', 'native-control', 'text-runs', 'duplicate-string'];
for (const rule of RULES) {
  const rows = byRule.get(rule) ?? [];
  const mark = RATCHETED.has(rule)
    ? `${rows.length} found, baseline ${baseline[rule] ?? 0}${rows.length > (baseline[rule] ?? 0) ? ' — over' : ''}`
    : (rows.length === 0 ? 'ok' : `${rows.length} finding${rows.length === 1 ? '' : 's'}`);
  out.push(`  ${rule}: ${mark}`);
  for (const row of rows.slice(0, 25)) out.push(`    ${row.where}  ${row.message}`);
  if (rows.length > 25) out.push(`    … and ${rows.length - 25} more`);
}
out.push('');
out.push(failed
  ? `  FAIL — ${hard.length} finding(s)${overBaseline.length ? `, over baseline: ${overBaseline.map(([k]) => k).join(', ')}` : ''}`
  : '  PASS');
out.push('');
process.stdout.write(`${out.join('\n')}\n`);
process.exit(failed ? 1 : 0);
