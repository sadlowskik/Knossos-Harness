// Loads the git-backed civilization configuration from field/.
// Everything here is durable and human-readable. Nothing operational is written back.
import fs from 'node:fs';
import path from 'node:path';
import YAML from 'yaml';
import { canonicalizeWorkspace } from './workspace-path.js';

/** Split `---\nyaml\n---\nbody` into { data, body }. */
export function frontmatter(text) {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/.exec(text);
  if (!m) return { data: {}, body: text };
  let data = {};
  try { data = YAML.parse(m[1]) ?? {}; } catch { data = {}; }
  return { data, body: m[2] };
}

/** Update frontmatter fields while preserving the record body verbatim. */
export function updateFrontmatterFile(file, updates) {
  const text = fs.readFileSync(file, 'utf8');
  const { data, body } = frontmatter(text);
  const next = { ...data, ...updates };
  const yaml = YAML.stringify(next).trimEnd();
  const separator = body.startsWith('\n') || body === '' ? '' : '\n';
  fs.writeFileSync(file, `---\n${yaml}\n---\n${separator}${body}`, 'utf8');
  return next;
}

function readDir(dir, ext) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter((f) => f.endsWith(ext))
    .map((f) => {
      const full = path.join(dir, f);
      const text = fs.readFileSync(full, 'utf8');
      if (ext === '.md') {
        const { data, body } = frontmatter(text);
        return { ...data, id: data.id ?? data.name ?? path.basename(f, ext), body, file: full };
      }
      let data = {};
      try { data = YAML.parse(text) ?? {}; } catch { data = {}; }
      return { ...data, id: data.id ?? path.basename(f, ext), file: full };
    });
}

export function loadConfig(fieldDir) {
  const rootFile = path.join(fieldDir, 'field.yaml');
  if (!fs.existsSync(rootFile)) throw new Error(`field.yaml not found at ${rootFile}`);
  const root = YAML.parse(fs.readFileSync(rootFile, 'utf8'));

  const roles = readDir(path.join(fieldDir, 'roles'), '.md');
  const agents = readDir(path.join(fieldDir, 'agents'), '.md');
  const missions = readDir(path.join(fieldDir, 'missions'), '.md');
  const constitutions = readDir(path.join(fieldDir, 'constitutions'), '.md');
  const skills = readDir(path.join(fieldDir, 'skills'), '.md');
  const routines = readDir(path.join(fieldDir, 'routines'), '.yaml');
  const memory = readDir(path.join(fieldDir, 'memory'), '.md');

  // Resolve every workspace against the real filesystem. A workspace that does not
  // exist is reported as unmounted rather than silently invented.
  const workspaces = (root.workspaces ?? []).map((workspace) => canonicalizeWorkspace(
    workspace,
    fieldDir,
    root.field?.sensitive_names ?? [],
  ));

  return {
    fieldDir,
    field: root.field ?? {},
    defaults: root.defaults ?? {},
    workspaces,
    endpoints: root.endpoints ?? [],
    websites: root.websites ?? [],
    roles, agents, missions, constitutions, skills, routines, memory,
  };
}

/** Compose the system prompt an agent actually runs under: constitution + role + mission. */
export function composePrompt(cfg, { agentId, roleId, missionId, orders }) {
  const agent = cfg.agents.find((a) => a.id === agentId);
  const role = cfg.roles.find((r) => r.id === (roleId ?? agent?.role));
  const con = cfg.constitutions.find((c) => c.id === (agent?.constitution ?? 'core'));
  const mission = missionId ? cfg.missions.find((m) => m.id === missionId) : null;

  const parts = [];
  if (con) parts.push(con.body.trim());
  if (role) parts.push(role.body.trim());
  if (agent?.body?.trim()) parts.push(agent.body.trim());
  if (mission) parts.push(`# Active mission: ${mission.name}\n\n${mission.body.trim()}`);
  if (orders) parts.push(`# Orders from the operator\n\n${orders.trim()}`);
  parts.push(
    '# Field reporting protocol\n\n' +
    'You are running as a unit inside Field, an operator-controlled multi-agent environment.\n' +
    'The operator watches your tool calls live. Keep prose short; the work is the output.\n' +
    'When you finish, state plainly what changed, what you verified, and what you did not.'
  );
  return parts.join('\n\n---\n\n');
}
