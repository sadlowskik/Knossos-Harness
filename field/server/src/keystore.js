// Local key store. Values live ONLY here (under the gitignored .field-state), are registered
// for event-log redaction, and are never written to field.yaml, the snapshot, or any API
// response. The API exposes key *ids* (refs), never values.
import fs from 'node:fs';
import path from 'node:path';

export class KeyStore {
  constructor(stateDir) {
    this.file = path.join(stateDir, 'keys.json');
    this.map = this._load();
  }

  _load() {
    try {
      const obj = JSON.parse(fs.readFileSync(this.file, 'utf8'));
      return new Map(Object.entries(obj && typeof obj === 'object' ? obj : {}));
    } catch { return new Map(); }
  }

  _save() {
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    fs.writeFileSync(this.file, JSON.stringify(Object.fromEntries(this.map)), { encoding: 'utf8', mode: 0o600 });
    try { fs.chmodSync(this.file, 0o600); } catch { /* Windows: no-op */ }
  }

  set(ref, value) { this.map.set(String(ref), String(value)); this._save(); }
  get(ref) { return this.map.get(String(ref)); }
  delete(ref) { if (this.map.delete(String(ref))) this._save(); }
  ids() { return [...this.map.keys()]; }
  values() { return [...this.map.values()]; }
}
