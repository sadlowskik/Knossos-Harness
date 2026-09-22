/* Choose the files a conversation is working on. One directory at a time, read from
   `/api/fs/tree`, with the already-attached files ticked. It takes a starting directory
   so a screen that knows where you are — the Map's folder detail — can open it already
   scoped to the folder you clicked. */

import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, ChevronLeft, Folder, FileText } from 'lucide-react';
import { api } from '../net/client.js';
import { MAX_SESSION_FILES } from '../theater/fieldPreferences.js';

const parentOf = (dir) => (dir.includes('/') ? dir.slice(0, dir.lastIndexOf('/')) : '');

export default function FilePicker({ wsId, startDir = '', attached = [], onToggle, onClose, label = 'Files in scope' }) {
  const [dir, setDir] = useState(startDir ?? '');
  const [entries, setEntries] = useState(null);
  const [error, setError] = useState(null);
  const ref = useRef(null);

  useEffect(() => {
    const onDown = (e) => { if (!ref.current?.contains(e.target)) onClose?.(); };
    const onKey = (e) => { if (e.key === 'Escape') { e.stopPropagation(); onClose?.(); } };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [onClose]);

  const load = useCallback(() => {
    if (!wsId) { setEntries([]); setError('This agent has no project folder.'); return undefined; }
    let alive = true;
    setEntries(null); setError(null);
    api.tree(wsId, dir)
      .then((r) => { if (alive) setEntries(r.entries ?? []); })
      .catch((e) => { if (alive) { setEntries([]); setError(e.message); } });
    return () => { alive = false; };
  }, [wsId, dir]);
  useEffect(() => load(), [load]);

  const full = attached.length >= MAX_SESSION_FILES;

  return (
    <div className="convo-picker" ref={ref} role="dialog" aria-label={label}>
      <div className="convo-picker-head">
        <button
          type="button"
          className="btn sm ghost icon"
          aria-label="Up one folder"
          disabled={!dir}
          onClick={() => setDir(parentOf(dir))}
        ><ChevronLeft aria-hidden="true" /></button>
        <span className="convo-picker-path mono" title={dir || 'project root'}>{dir || 'project root'}</span>
        <button type="button" className="btn sm ghost" onClick={onClose}>Done</button>
      </div>
      <div className="convo-picker-body">
        {error && <p className="convo-picker-note bad" role="alert">{error}</p>}
        {entries === null && !error && <p className="convo-picker-note">Reading the folder…</p>}
        {entries?.length === 0 && !error && <p className="convo-picker-note">This folder is empty.</p>}
        {(entries ?? []).map((entry) => {
          const on = attached.includes(entry.path);
          return (
            <button
              key={entry.path}
              type="button"
              className={`convo-picker-row${on ? ' on' : ''}`}
              disabled={!entry.dir && !on && full}
              title={!entry.dir && !on && full ? `At most ${MAX_SESSION_FILES} files per conversation` : entry.path}
              aria-pressed={entry.dir ? undefined : on}
              onClick={() => (entry.dir ? setDir(entry.path) : onToggle(entry.path))}
            >
              {entry.dir
                ? <Folder aria-hidden="true" />
                : on ? <Check aria-hidden="true" /> : <FileText aria-hidden="true" />}
              <span className="convo-picker-name">{entry.name}</span>
              {entry.dir && <span className="convo-picker-more" aria-hidden="true">›</span>}
            </button>
          );
        })}
      </div>
      {full && <p className="convo-picker-note">{MAX_SESSION_FILES} files is the limit; remove one to add another.</p>}
    </div>
  );
}
