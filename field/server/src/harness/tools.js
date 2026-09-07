// Translates a real harness tool call into Field's spatial vocabulary.
// Nothing here invents activity: every field returned is derived from the actual
// tool name and the actual arguments the agent passed.
import fs from 'node:fs';
import path from 'node:path';

const parentOf = (rel) => (rel.includes('/') ? rel.slice(0, rel.lastIndexOf('/')) : '');

/**
 * Which mounted workspace does this absolute path belong to, and where inside it?
 *
 * `dir` is always a directory: the path itself when it is one, otherwise its parent.
 * Tools like Grep and Glob accept either, and treating a file as a directory would
 * put a file on the Field as though it were a folder.
 */
export function locate(absPath, workspaces) {
  if (!absPath) return null;
  let candidate = path.resolve(absPath);
  try { candidate = (fs.realpathSync.native ?? fs.realpathSync)(candidate); } catch { /* report non-existent targets lexically */ }
  const norm = candidate.replace(/\\/g, '/');
  let best = null;
  for (const w of workspaces) {
    const root = (w.canonicalPath ?? path.resolve(w.path)).replace(/\\/g, '/');
    if (norm === root || norm.startsWith(root + '/')) {
      if (!best || root.length > best.rootLen) best = { w, rootLen: root.length, root };
    }
  }
  if (!best) return null;

  const rel = norm.slice(best.root.length).replace(/^\//, '');
  let isDir;
  try { isDir = fs.statSync(norm).isDirectory(); }
  catch { isDir = path.extname(rel) === ''; }   // gone already: fall back to the shape

  return { workspaceId: best.w.id, rel, dir: isDir ? rel : parentOf(rel), isDir };
}

function domainOf(url) {
  try { return new URL(url).hostname; } catch { return null; }
}

function short(p, n = 48) {
  if (!p) return '';
  return p.length <= n ? p : '…' + p.slice(-(n - 1));
}

const EDIT_TOOLS = new Set(['Edit', 'Write', 'NotebookEdit']);

/**
 * @returns {{summary:string, workspaceId?:string, dir?:string, path?:string,
 *            browser?:{url:string,domain:string}, delegation?:{description:string,type:string},
 *            command?:string, isEdit:boolean}}
 */
export function describeTool(name, input = {}, ctx = {}) {
  const workspaces = ctx.workspaces ?? [];
  const cwd = ctx.cwd;
  const out = { summary: name, isEdit: EDIT_TOOLS.has(name) };

  const filePath = input.file_path ?? input.filePath ?? input.notebook_path;
  const searchPath = input.path;

  if (filePath) {
    const loc = locate(filePath, workspaces);
    if (loc) { out.workspaceId = loc.workspaceId; out.dir = loc.dir; out.path = loc.rel; }
    out.summary = `${name} ${short(loc ? loc.rel : filePath)}`;
    return out;
  }

  switch (name) {
    case 'Grep': {
      const loc = locate(searchPath ?? cwd, workspaces);
      if (loc) { out.workspaceId = loc.workspaceId; out.dir = loc.dir; }
      out.summary = `Grep /${String(input.pattern ?? '').slice(0, 32)}/`;
      return out;
    }
    case 'Glob': {
      const loc = locate(searchPath ?? cwd, workspaces);
      if (loc) { out.workspaceId = loc.workspaceId; out.dir = loc.dir; }
      out.summary = `Glob ${String(input.pattern ?? '').slice(0, 40)}`;
      return out;
    }
    case 'Bash':
    case 'PowerShell': {
      const loc = locate(cwd, workspaces);
      if (loc) { out.workspaceId = loc.workspaceId; out.dir = loc.dir; }
      const cmd = String(input.command ?? '').replace(/\s+/g, ' ').trim();
      out.command = cmd;
      out.summary = `$ ${cmd.slice(0, 56)}`;
      return out;
    }
    case 'WebFetch': {
      const url = input.url;
      const domain = domainOf(url);
      if (domain) out.browser = { url, domain };
      out.summary = `Fetch ${domain ?? url ?? ''}`;
      return out;
    }
    case 'WebSearch': {
      out.summary = `Search "${String(input.query ?? '').slice(0, 40)}"`;
      return out;
    }
    case 'Task':
    case 'Agent': {
      out.delegation = {
        description: input.description ?? input.prompt?.slice?.(0, 80) ?? 'subagent',
        type: input.subagent_type ?? 'general-purpose',
      };
      out.summary = `Delegate → ${out.delegation.type}`;
      return out;
    }
    case 'TodoWrite': {
      const todos = Array.isArray(input.todos) ? input.todos : [];
      const done = todos.filter((t) => t.status === 'completed').length;
      out.progress = { done, total: todos.length };
      out.summary = `Plan ${done}/${todos.length}`;
      return out;
    }
    default: {
      // Browser-driving MCP tools carry a url; treat them as real browser routes.
      if (typeof input.url === 'string') {
        const domain = domainOf(input.url);
        if (domain) out.browser = { url: input.url, domain };
        out.summary = `${name} ${domain ?? ''}`.trim();
        return out;
      }
      const loc = locate(cwd, workspaces);
      if (loc) { out.workspaceId = loc.workspaceId; out.dir = loc.dir; }
      return out;
    }
  }
}
