/* A folder, opened.

   This is the whole point of Rome: click a district and you get the place, not a
   statistics card. The place is its conversations — the same `Conversation` panel the
   Board mounts, so you can read and talk to whoever is working here with the same
   controls and the same file strip — above them the folder's path, weight and
   staleness, and below them its subfolders, so you can keep going down.

   It replaced three things that each showed a slice of this: the city command hub, the
   senate roster and the region maturity inspector. */

import { useEffect, useRef } from 'react';
import { ChevronRight, Folder, Plus, X } from 'lucide-react';
import { useField } from '../state/store.js';
import Conversation, { conversationNeedsYou, sortConversations, TERMINAL_STATES } from '../ui/Conversation.jsx';
import { crumbsFor, dirName, staleness, weightLabel } from './districts.js';

export default function FolderDetail({
  workspace,
  dir,
  weight,
  subfolders,
  sessions,
  selectedId = null,
  primary = true,
  onNavigate,
  onStart,
  onClose,
}) {
  const st = useField();
  const scrollRef = useRef(null);
  const now = st.snap.now ?? Date.now();
  const permissions = st.snap.permissions ?? [];
  const here = sortConversations(sessions, permissions);
  const live = here.filter((session) => !TERMINAL_STATES.has(session.state));
  const heat = staleness(weight?.lastTs ?? 0, now, live.length);
  const crumbs = crumbsFor(dir);
  const needing = here.filter((session) => conversationNeedsYou(session, permissions)).length;

  // A new folder starts at its own top, not wherever the last one was left.
  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = 0; }, [dir]);

  // Clicking an agent on the map opens this folder; put that conversation in view.
  useEffect(() => {
    if (!selectedId) return;
    const node = scrollRef.current?.querySelector('.convo.map-selected');
    node?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [selectedId, dir]);

  return (
    <aside className="folder-sheet" onClick={(event) => event.stopPropagation()} aria-label={`Folder ${dir || workspace.name}`}>
      <header className="folder-head">
        <nav className="folder-crumbs" aria-label="Folder path">
          <button type="button" className="folder-crumb" onClick={() => onNavigate('')}>{workspace.name}</button>
          {crumbs.slice(1).map((crumb, index) => (
            <span key={crumb.dir}>
              <ChevronRight aria-hidden="true" />
              {index === crumbs.length - 2
                ? <b className="mono">{crumb.name}</b>
                : <button type="button" className="folder-crumb mono" onClick={() => onNavigate(crumb.dir)}>{crumb.name}</button>}
            </span>
          ))}
          <span className="grow" />
          <button type="button" className="btn sm ghost icon" aria-label="Close this folder" onClick={onClose}>
            <X aria-hidden="true" />
          </button>
        </nav>

        <h2>{dirName(dir)}</h2>
        <p className={`folder-facts heat-${heat.id}`}>
          <span className="mono">{weightLabel(weight ?? { files: null })}</span>
          <i aria-hidden="true" />
          <span>{heat.label}</span>
        </p>

        <div className="folder-head-actions">
          <button
            type="button"
            className={`btn${primary ? ' primary' : ''}`}
            onClick={(event) => onStart(dir, event)}
          ><Plus aria-hidden="true" />Start an agent here</button>
          {needing > 0 && (
            <span className="folder-needs" role="status">
              {needing === 1 ? 'One conversation needs you' : `${needing} conversations need you`}
            </span>
          )}
        </div>
      </header>

      <div className="folder-sheet-scroll" ref={scrollRef}>
        <section className="folder-convos" role="list" aria-label="Conversations in this folder">
          {here.map((session) => (
            <Conversation
              key={session.id}
              session={session}
              workspace={workspace}
              startDir={dir}
              showProject={false}
              className={session.id === selectedId ? 'map-selected' : ''}
              onStartSimilar={(ended) => onStart(dir, null, ended.agentId ?? null)}
            />
          ))}
          {!here.length && (
            <p className="folder-quiet">
              No one is working in <code>{dir || workspace.name}</code>. Start an agent here, or open a
              subfolder below to see who is further down.
            </p>
          )}
        </section>

        <section className="folder-subs" aria-label="Subfolders">
          <h3 className="label">Inside this folder</h3>
          {subfolders === null && <p className="folder-quiet">Reading the folder…</p>}
          {subfolders?.length === 0 && <p className="folder-quiet">No subfolders — this is the bottom of the map here.</p>}
          {(subfolders ?? []).map((sub) => {
            const subHeat = staleness(sub.lastTs, now, sub.agents ?? 0);
            return (
              <div key={sub.dir} className={`folder-sub heat-${subHeat.id}`}>
                <button type="button" className="folder-sub-open" onClick={() => onNavigate(sub.dir)}>
                  <Folder aria-hidden="true" />
                  <span className="folder-sub-text">
                    <b>{sub.name}</b>
                    <small className="mono">{weightLabel(sub)}</small>
                  </span>
                  <small className="folder-sub-heat">{subHeat.label}</small>
                </button>
                <button
                  type="button"
                  className="btn sm ghost"
                  title={`Start an agent scoped to ${sub.dir}`}
                  onClick={(event) => onStart(sub.dir, event)}
                ><Plus aria-hidden="true" />agent</button>
              </div>
            );
          })}
        </section>
      </div>
    </aside>
  );
}
