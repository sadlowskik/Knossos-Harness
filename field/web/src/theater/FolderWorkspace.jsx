/* The folder, expanded.

   A folder's detail opens as a sheet beside the map. Expand it and it fills the screen
   with what the Project destination used to be, scoped to the folder you opened: the
   files in it, the file you picked in Source, Rendered or Diff, the project's Changes
   with Accept and Revert, and a real shell. It is the same folder — same crumbs, same
   heading, same "start an agent here" — at a different size, so nothing navigates. */

import { useEffect, useState } from 'react';
import { Changes, Diff, FileTree, FileView, Terminal, BrowserPane } from '../workspace/panels.jsx';

export default function FolderWorkspace({
  workspace,
  dir = '',
  initialPath = null,
  initialView = null,
  initialPane = null,
  initialUrl = null,
}) {
  const [leftTab, setLeftTab] = useState(initialView === 'changes' ? 'changes' : 'files');
  const [openPath, setOpenPath] = useState(initialPath ?? null);
  const [fileView, setFileView] = useState(
    initialView === 'changes' ? 'diff' : initialPath?.toLowerCase().endsWith('.md') ? 'rendered' : 'source',
  );
  const [rightTab, setRightTab] = useState(initialPane === 'browser' ? 'browser' : 'terminal');

  // A request from elsewhere ("open the changes on this project") arrives as new props.
  useEffect(() => { if (initialView === 'changes') { setLeftTab('changes'); setFileView('diff'); } }, [initialView]);
  useEffect(() => { if (initialPath) setOpenPath(initialPath); }, [initialPath]);
  useEffect(() => { if (initialPane === 'browser') setRightTab('browser'); }, [initialPane]);

  const git = workspace?.git;
  const changedCount = git?.files?.length ?? 0;
  // Rendered is offered for Markdown; Diff only for a path git actually reports changed.
  const canRender = Boolean(openPath && openPath.toLowerCase().endsWith('.md'));
  const canDiff = Boolean(openPath && (git?.files ?? []).some((f) => f.path === openPath));
  const view = (fileView === 'rendered' && !canRender) || (fileView === 'diff' && !canDiff) ? 'source' : fileView;
  const VIEWS = [
    { id: 'source', label: 'Source', on: true },
    { id: 'rendered', label: 'Rendered', on: canRender },
    { id: 'diff', label: 'Diff', on: canDiff },
  ].filter((item) => item.on);

  return (
    <div className="folder-workspace panes">
      <div className="pane pane-files">
        <div className="pane-head">
          <div className="tabs">
            {[['files', 'files'], ['changes', `changes${changedCount ? ` · ${changedCount}` : ''}`]].map(([t, label]) => (
              <button
                key={t}
                className={`tab${leftTab === t ? ' on' : ''}`}
                onClick={() => setLeftTab(t)}
                type="button"
              >{label}</button>
            ))}
          </div>
          <span className="grow" />
          {git?.branch && <span className="label">{git.branch}</span>}
        </div>
        <div className="pane-body">
          {leftTab === 'changes'
            ? (
              <Changes
                wsId={workspace?.id}
                scopeDir={dir}
                openPath={openPath}
                onOpen={(p) => { setOpenPath(p); setFileView('diff'); }}
              />
            )
            : (
              <FileTree
                wsId={workspace?.id}
                root={dir}
                git={git}
                openPath={openPath}
                onOpen={(p) => {
                  setOpenPath(p);
                  setFileView(p.toLowerCase().endsWith('.md') ? 'rendered' : 'source');
                }}
              />
            )}
        </div>
      </div>

      <div className="pane pane-file">
        <div className="pane-head">
          <span className="grow pane-head-path label" title={openPath ?? undefined}>{openPath ?? 'no file open'}</span>
          {VIEWS.length > 1 && (
            <div className="viewctl" role="group" aria-label="File view">
              {VIEWS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`viewctl-item${view === item.id ? ' on' : ''}`}
                  aria-pressed={view === item.id}
                  onClick={() => setFileView(item.id)}
                >{item.label}</button>
              ))}
            </div>
          )}
        </div>
        <div className="pane-body">
          {view === 'diff'
            ? <Diff wsId={workspace?.id} path={openPath} />
            : <FileView wsId={workspace?.id} path={openPath} markdown={view === 'rendered'} />}
        </div>
      </div>

      <div className="pane pane-shell">
        <div className="pane-head">
          <div className="tabs">
            {['terminal', 'browser'].map((t) => (
              <button
                key={t}
                className={`tab${rightTab === t ? ' on' : ''}`}
                onClick={() => setRightTab(t)}
                type="button"
              >{t}</button>
            ))}
          </div>
        </div>
        <div className="pane-body pane-body-fixed">
          {rightTab === 'terminal'
            ? <Terminal wsId={workspace?.id} />
            : <BrowserPane initialUrl={initialUrl} />}
        </div>
      </div>
    </div>
  );
}
