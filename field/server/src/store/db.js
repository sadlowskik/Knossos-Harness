// The event log is the single source of truth for operational state.
// Nothing operational is ever UPDATEd; state is a fold over this log.
// Preferred backend is node:sqlite. If unavailable, a durable JSONL log is used with
// identical semantics, so the Field behaves the same either way.
import fs from 'node:fs';
import path from 'node:path';

let DatabaseSync = null;
try { ({ DatabaseSync } = await import('node:sqlite')); } catch { DatabaseSync = null; }

export const EVENT_LOG_SCHEMA_VERSION = 2;

export const EVENT_SOURCES = Object.freeze(['observed', 'derived', 'manual', 'synthetic']);
const EVENT_SOURCE_SET = new Set(EVENT_SOURCES);

const SCHEMA = `
CREATE TABLE IF NOT EXISTS events (
  seq     INTEGER PRIMARY KEY AUTOINCREMENT,
  ts      INTEGER NOT NULL,
  kind    TEXT    NOT NULL,
  actor   TEXT,
  subject TEXT,
  source  TEXT    NOT NULL DEFAULT 'observed',
  data    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_kind    ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_subject ON events(subject);
CREATE INDEX IF NOT EXISTS idx_events_ts      ON events(ts);
`;

export class EventLog {
  constructor(dir, { secrets = [], backend = null } = {}) {
    fs.mkdirSync(dir, { recursive: true });
    this.dir = dir;
    this.listeners = new Set();
    this.secrets = [...new Set(secrets.map(String).filter((value) => value.length >= 8))];
    if (backend != null && !['sqlite', 'jsonl'].includes(backend)) {
      throw new Error(`unsupported event log backend: ${backend}`);
    }
    if (backend === 'sqlite' && !DatabaseSync) {
      throw new Error('SQLite event log requested but node:sqlite is unavailable');
    }
    this.backend = backend ?? (DatabaseSync ? 'sqlite' : 'jsonl');
    this.schemaVersion = EVENT_LOG_SCHEMA_VERSION;
    this.integrity = 'ok';

    if (this.backend === 'sqlite') {
      this.db = new DatabaseSync(path.join(dir, 'field.db'));
      this.db.exec('PRAGMA journal_mode = WAL;');
      const checks = this.db.prepare('PRAGMA quick_check').all();
      if (!checks.length || checks.some((row) => row.quick_check !== 'ok')) {
        this.integrity = 'corrupt';
        this.db.close();
        throw new Error(`Field event database failed integrity check: ${JSON.stringify(checks)}`);
      }
      const storedVersion = Number(this.db.prepare('PRAGMA user_version').get().user_version);
      if (!Number.isSafeInteger(storedVersion) || storedVersion < 0) {
        this.db.close();
        throw new Error(`Field event database has invalid schema version: ${storedVersion}`);
      }
      if (storedVersion > EVENT_LOG_SCHEMA_VERSION) {
        this.db.close();
        throw new Error(
          `Field event database schema ${storedVersion} is newer than supported schema ${EVENT_LOG_SCHEMA_VERSION}`
        );
      }
      if (storedVersion < EVENT_LOG_SCHEMA_VERSION) {
        this.db.exec('BEGIN IMMEDIATE;');
        try {
          this.db.exec(SCHEMA);
          const columns = this.db.prepare('PRAGMA table_info(events)').all();
          if (!columns.some((column) => column.name === 'source')) {
            this.db.exec("ALTER TABLE events ADD COLUMN source TEXT NOT NULL DEFAULT 'observed';");
          }
          this.db.exec(`PRAGMA user_version = ${EVENT_LOG_SCHEMA_VERSION}; COMMIT;`);
        } catch (error) {
          try { this.db.exec('ROLLBACK;'); } catch { /* original migration error wins */ }
          this.db.close();
          throw new Error('Field event database migration failed', { cause: error });
        }
      } else {
        this.db.exec(SCHEMA);
      }
      this.insert = this.db.prepare(
        'INSERT INTO events (ts, kind, actor, subject, source, data) VALUES (?, ?, ?, ?, ?, ?)'
      );
      this.selectFrom = this.db.prepare(
        'SELECT * FROM events WHERE seq > ? ORDER BY seq ASC LIMIT ?'
      );
      this.selectSubject = this.db.prepare(
        'SELECT * FROM events WHERE subject = ? ORDER BY seq ASC'
      );
      const row = this.db.prepare('SELECT COALESCE(MAX(seq),0) AS s FROM events').get();
      this.lastSeq = Number(row.s);
    } else {
      this.file = path.join(dir, 'events.jsonl');
      this.rows = [];
      if (fs.existsSync(this.file)) {
        const lines = fs.readFileSync(this.file, 'utf8')
          .split('\n')
          .map((line, index) => ({ line, index }))
          .filter(({ line }) => line.trim());
        for (let position = 0; position < lines.length; position += 1) {
          const { line, index } = lines[position];
          let row;
          try {
            row = JSON.parse(line);
          } catch (error) {
            // A process can stop between write and fsync. Only one malformed tail
            // record is recoverable; corruption inside committed history is fatal.
            if (position === lines.length - 1) break;
            this.integrity = 'corrupt';
            throw new Error(`Field JSONL event log is malformed at line ${index + 1}`, { cause: error });
          }
          const expected = this.rows.length ? this.rows.at(-1).seq + 1 : 1;
          validateJsonlRow(row, expected, index + 1);
          row.source = normalizePersistedSource(row);
          this.rows.push(row);
        }
      }
      this.lastSeq = this.rows.length ? this.rows[this.rows.length - 1].seq : 0;
    }
  }

  /** Append one event. Returns the stored event with its assigned seq. */
  append(kind, data = {}, { actor = null, subject = null, source = null, simulated = false } = {}) {
    const ts = Date.now();
    const safeData = sanitizeEventData(data, { secrets: this.secrets });
    const eventSource = source ?? (simulated || safeData?.simulated ? 'synthetic' : 'observed');
    if (!EVENT_SOURCE_SET.has(eventSource)) throw new Error(`unsupported event source: ${eventSource}`);
    let evt;
    if (this.backend === 'sqlite') {
      const info = this.insert.run(ts, kind, actor, subject, eventSource, JSON.stringify(safeData));
      this.lastSeq = Number(info.lastInsertRowid);
      evt = { seq: this.lastSeq, ts, kind, actor, subject, source: eventSource, data: safeData };
    } else {
      evt = { seq: ++this.lastSeq, ts, kind, actor, subject, source: eventSource, data: safeData };
      this.rows.push(evt);
      const handle = fs.openSync(this.file, 'a', 0o600);
      try {
        fs.writeFileSync(handle, JSON.stringify(evt) + '\n');
        fs.fsyncSync(handle);
      } finally {
        fs.closeSync(handle);
      }
    }
    for (const fn of this.listeners) {
      try { fn(evt); } catch (e) { console.error('[eventlog] listener failed:', e.message); }
    }
    return evt;
  }

  /** Read events after `fromSeq`, oldest first. */
  read(fromSeq = 0, limit = 100000) {
    if (this.backend === 'sqlite') {
      return this.selectFrom.all(fromSeq, limit).map(hydrate);
    }
    return this.rows.filter((r) => r.seq > fromSeq).slice(0, limit);
  }

  /** Every event about one subject, in order. Optional cursor/limit keep trace reads bounded. */
  bySubject(subject, { fromSeq = 0, limit = 100000 } = {}) {
    if (this.backend === 'sqlite') {
      return this.db.prepare(
        'SELECT * FROM events WHERE subject = ? AND seq > ? ORDER BY seq ASC LIMIT ?'
      ).all(subject, fromSeq, limit).map(hydrate);
    }
    return this.rows.filter((r) => r.subject === subject && r.seq > fromSeq).slice(0, limit);
  }

  subscribe(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  addSecret(value) {
    const secret = String(value ?? '');
    if (secret.length >= 8 && !this.secrets.includes(secret)) this.secrets.push(secret);
  }
  get size() { return this.lastSeq; }
  health() {
    return {
      backend: this.backend,
      schemaVersion: this.schemaVersion,
      integrity: this.integrity,
      events: this.lastSeq,
    };
  }

  close() {
    this.listeners.clear();
    if (this.backend === 'sqlite') this.db.close();
  }
}

function validateJsonlRow(row, expectedSequence, lineNumber) {
  if (!row || typeof row !== 'object' || Array.isArray(row)) {
    throw new Error(`Field JSONL event log has a non-object record at line ${lineNumber}`);
  }
  if (!Number.isSafeInteger(row.seq) || row.seq !== expectedSequence) {
    throw new Error(
      `Field JSONL event log sequence is invalid at line ${lineNumber}: expected ${expectedSequence}`
    );
  }
  if (!Number.isFinite(row.ts) || typeof row.kind !== 'string' || !row.kind) {
    throw new Error(`Field JSONL event log record is invalid at line ${lineNumber}`);
  }
}

function normalizePersistedSource(row) {
  const source = row.source ?? (row.data?.simulated ? 'synthetic' : 'observed');
  if (!EVENT_SOURCE_SET.has(source)) throw new Error(`Field JSONL event log has invalid source at sequence ${row.seq}`);
  return source;
}

const SECRET_KEY = /(^|[_-])(password|passwd|secret|token|authorization|cookie|api[_-]?key|private[_-]?key|bearer)($|[_-])/i;
const MAX_STRING_CHARS = 16_000;
const MAX_TOTAL_CHARS = 256_000;
const MAX_ITEMS = 300;
const MAX_DEPTH = 12;

/** Redact credential-shaped keys and bound persisted payloads before they reach disk. */
export function sanitizeEventData(value, { secrets = [] } = {}) {
  const budget = {
    chars: 0, items: 0, truncated: false,
    secrets: [...new Set(secrets.map(String).filter((secret) => secret.length >= 8))],
  };
  return sanitize(value, budget, 0);
}

function sanitize(value, budget, depth) {
  if (value == null || typeof value === 'boolean' || typeof value === 'number') return value;
  if (depth > MAX_DEPTH || budget.items++ >= MAX_ITEMS || budget.chars >= MAX_TOTAL_CHARS) {
    budget.truncated = true;
    return '[TRUNCATED]';
  }
  if (typeof value === 'string') {
    let safe = value;
    for (const secret of budget.secrets) safe = safe.replaceAll(secret, '[REDACTED]');
    safe = safe
      .replace(/\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b/gi, 'Bearer [REDACTED]')
      .replace(/\bsk-[A-Za-z0-9_-]{8,}\b/g, '[REDACTED]');
    const room = Math.max(0, Math.min(MAX_STRING_CHARS, MAX_TOTAL_CHARS - budget.chars));
    budget.chars += Math.min(safe.length, room);
    if (safe.length > room) { budget.truncated = true; return safe.slice(0, room) + '…[TRUNCATED]'; }
    return safe;
  }
  if (Array.isArray(value)) {
    const out = [];
    for (const item of value) {
      if (budget.items >= MAX_ITEMS || budget.chars >= MAX_TOTAL_CHARS) { out.push('[TRUNCATED]'); budget.truncated = true; break; }
      out.push(sanitize(item, budget, depth + 1));
    }
    return out;
  }
  if (typeof value === 'object') {
    const out = {};
    for (const [key, item] of Object.entries(value)) {
      if (budget.items >= MAX_ITEMS || budget.chars >= MAX_TOTAL_CHARS) { out._truncated = true; budget.truncated = true; break; }
      out[key] = SECRET_KEY.test(key) ? '[REDACTED]' : sanitize(item, budget, depth + 1);
    }
    return out;
  }
  return String(value);
}

function hydrate(row) {
  let data = {};
  try { data = JSON.parse(row.data); } catch { data = { _unparsed: row.data }; }
  return {
    seq: Number(row.seq), ts: Number(row.ts), kind: row.kind,
    actor: row.actor, subject: row.subject,
    source: EVENT_SOURCE_SET.has(row.source) ? row.source : (data?.simulated ? 'synthetic' : 'observed'),
    data,
  };
}
