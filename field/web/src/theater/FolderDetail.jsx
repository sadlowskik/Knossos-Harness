/* A folder, opened.

   This is the whole point of Rome: click a district and you get the place, not a
   statistics card. The place is its conversations — the same `Conversation` panel Atlas
   mounts, so you can read and talk to whoever is working here with the same controls and
   the same file strip — above them the folder's path, weight and staleness, and below
   them its subfolders, so you can keep going down.

   It opens compact, as a sheet beside the map. Expand it and the same folder fills the
   screen with its files, its diffs, the project's changes and a shell: that is what the
   Project destination was, and it was never a place of its own — it was this folder at a
   larger size. Expanding does not navigate, so the map is one Escape away.

   It replaced four things that each showed a slice of this: the city command hub, the
   senate roster, the region maturity inspector and the Project screen. */

import { useEffect, useRef, useState } from 'react';
import { ChevronRight, Folder, Maximize2, Minimize2, Plus, X } from 'lucide-react';
import { useField } from '../state/store.js';
import Conversation, { conversationNeedsYou, sortConversations, TERMINAL_STATES } from '../ui/Conversation.jsx';
import FolderWorkspace from './FolderWorkspace.jsx';
import { crumbsFor, dirName, staleness, weightLabel } from './districts.js';

export default function FolderDetail({
  workspace,
  dir,
  weight,
  subfolders,
  sessions,
  selectedId = null,
  primary = true,
  expanded = false,
  request = null,
  onNavigate,
  onStart,
  onToggleExpand,
  onClose,
}) {
  const st = useField();
  const scrollRef = useRef(null);
  const [weightOpen, setWeightOpen] = useState(false);
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
    <aside
      className={`folder-sheet${expanded ? ' expanded' : ''}`}
      onClick={(event) => event.stopPropagation()}
      aria-label={`Folder ${dir || workspace.name}`}
    >
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
          <button
            type="button"
            className="btn sm ghost icon"
            aria-pressed={expanded}
            aria-label={expanded ? 'Collapse this folder back beside the map' : 'Expand this folder to its files, changes and terminal'}
            title={expanded ? 'Back to the map' : 'Files, changes and terminal'}
            onClick={onToggleExpand}
          >{expanded ? <Minimize2 aria-hidden="true" /> : <Maximize2 aria-hidden="true" />}</button>
          <button type="button" className="btn sm ghost icon" aria-label="Close this folder" onClick={onClose}>
            <X aria-hidden="true" />
          </button>
        </nav>

        <h2>
          {dirName(dir)}
          {/* The one thing about this folder you must read first: whether it wants you. */}
          {needing > 0 && (
            <span className="folder-badge mono" role="status" aria-label={
              needing === 1 ? 'One conversation needs you' : `${needing} conversations need you`
            }>{needing}</span>
          )}
        </h2>

        {/* "128 files · 6 folders · 3 changed today" was a second sentence of arithmetic
            above the conversations. What is left is how alive the place is; the weight
            is one tap behind it. */}
        <button
          type="button"
          className={`folder-facts heat-${heat.id}`}
          aria-expanded={weightOpen}
          onClick={() => setWeightOpen((open) => !open)}
        >
          <i aria-hidden="true" />
          <span>{heat.label}</span>
        </button>
        {weightOpen && <p className="folder-weight mono">{weightLabel(weight ?? { files: null })}</p>}

        <div className="folder-head-actions">
          <button
            type="button"
            className={`btn${primary ? ' primary' : ''}`}
            onClick={(event) => onStart(dir, event)}
          ><Plus aria-hidden="true" />Start an agent here</button>
        </div>
      </header>

      {expanded ? (
        <FolderWorkspace
          workspace={workspace}
          dir={dir}
          initialPath={request?.path ?? null}
          initialView={request?.view ?? null}
          initialPane={request?.pane ?? null}
          initialUrl={request?.url ?? null}
        />
      ) : (
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
              /* Each row used to carry its own file count, folder count and staleness
                 sentence — forty of them under every folder. The row is now a name, a
                 heat dot, and a count when someone is in there; the rest is the
                 tooltip, and all of it is on the folder itself once you walk in. */
              <div key={sub.dir} className={`folder-sub heat-${subHeat.id}`}>
                <button
                  type="button"
                  className="folder-sub-open"
                  onClick={() => onNavigate(sub.dir)}
                  title={`${sub.dir} — ${weightLabel(sub)} · ${subHeat.label}`}
                  aria-label={`Open ${sub.name}: ${weightLabel(sub)}, ${subHeat.label}`}
                >
                  <Folder aria-hidden="true" />
                  <span className="folder-sub-text"><b>{sub.name}</b></span>
                  {(sub.agents ?? 0) > 0 && <small className="folder-sub-count mono">{sub.agents}</small>}
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
      )}
    </aside>
  );
}
