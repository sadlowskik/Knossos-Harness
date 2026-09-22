import { BookOpen, Container, GitBranch, Globe2, TerminalSquare, Wrench } from 'lucide-react';

// One glyph per family of tool, shared by the Rome inspector and the settings panel.
export default function ToolIcon({ name }) {
  const lower = String(name).toLowerCase();
  const Icon = /git/.test(lower) ? GitBranch
    : /docker|container|podman/.test(lower) ? Container
      : /web|browser|fetch|search/.test(lower) ? Globe2
        : /read|grep|glob/.test(lower) ? BookOpen
          : /edit|write/.test(lower) ? Wrench
            : TerminalSquare;
  return <Icon aria-hidden="true" />;
}
