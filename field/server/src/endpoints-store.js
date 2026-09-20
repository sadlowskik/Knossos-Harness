// Runtime endpoint descriptors added through the UI. Non-secret only:
// { id, name, kind, model, base_url, secretRef? }. The key itself lives in the SecretStore
// and is referenced by secretRef. Merged over field.yaml endpoints at boot; persisted under
// the gitignored .field-state so a downloader's models survive restarts without touching git.
import fs from 'node:fs';
import path from 'node:path';

export class EndpointsStore {
  constructor(stateDir) {
    this.file = path.join(stateDir, 'endpoints.json');
    this.list = this._load();
  }

  _load() {
    try {
      const obj = JSON.parse(fs.readFileSync(this.file, 'utf8'));
      return Array.isArray(obj?.endpoints) ? obj.endpoints : [];
    } catch { return []; }
  }

  _save() {
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    fs.writeFileSync(this.file, JSON.stringify({ schema: 'field-endpoints/v1', endpoints: this.list }, null, 2));
  }

  all() { return this.list.slice(); }
  get(id) { return this.list.find((e) => e.id === id); }
  add(desc) { this.list = this.list.filter((e) => e.id !== desc.id); this.list.push(desc); this._save(); return desc; }
  remove(id) {
    const before = this.list.length;
    this.list = this.list.filter((e) => e.id !== id);
    if (this.list.length !== before) this._save();
    return this.list.length !== before;
  }
}
