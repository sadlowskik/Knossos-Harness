"""Policy that runs around every tool call.

The guardrails in `tools` and `workspace` are structural: they hold for every
workspace because they are properties of the code. A hook is the other kind of
rule -- the one that is true for *this* repository, or this run, and that the
harness has no business hard-coding. "Do not touch the migrations directory" is
a real constraint and a bad constant.

Hooks live inside `ToolRegistry.dispatch`, so every caller gets them. Talos is
not the only thing that dispatches tools, and a policy that applies on one path
is not a policy.

Why `after` cannot change a result
----------------------------------

`before` can deny or rewrite; `after` can only observe. The asymmetry is
deliberate. A denied call is visible to the engine as an error and it adapts --
an ordinary failure it already knows how to handle. A *rewritten result* is a
lie: the engine is told a file contains something it does not, and every later
step reasons from it. Refusing to build that road is cheaper than policing who
walks down it.

Mirrors `knossos-rs/src/hooks.rs`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Decision", "ALLOW", "deny", "rewrite", "Hook", "ProtectPaths",
           "before_chain"]


@dataclass(frozen=True)
class Decision:
    """What a hook decided about a call it was shown.

    Exactly one of the fields is set, or neither, which is `ALLOW`.
    """

    #: Replacement arguments. The call goes ahead with these instead.
    rewritten: Optional[Dict[str, Any]] = None
    #: Why the call must not happen. Reaches the engine as the tool's error.
    denial: Optional[str] = None


#: Run it unchanged.
ALLOW = Decision()


def rewrite(args: Dict[str, Any]) -> Decision:
    return Decision(rewritten=dict(args))


def deny(reason: str) -> Decision:
    return Decision(denial=reason)


class Hook:
    """Base class, with both halves defaulting to doing nothing."""

    #: Named so a denial can say what stopped it. An agent told only "denied"
    #: will retry the same call; one told which rule denied it will not.
    name = "hook"

    def before(self, tool: str, args: Dict[str, Any]) -> Decision:
        """Runs before the tool, in registration order.

        The first denial wins and the hooks after it do not run.
        """
        return ALLOW

    def after(self, tool: str, args: Dict[str, Any], result) -> None:
        """Runs after **every** dispatched call, in registration order.

        Every call means every call: one that ran, one a hook denied, one whose
        tool name does not exist, and one whose tool returned an error. A hook
        that sees only successful calls cannot audit anything, since the
        interesting cases are exactly the ones that went wrong.

        `args` is the call as the `before` chain left it -- rewritten if a hook
        rewrote it. On a denial it is still the rewritten value, because the
        rewritten call is the one that was attempted.
        """


class ProtectPaths(Hook):
    """Refuse changes to paths inside the workspace that are not the work.

    The path jail stops at the workspace boundary, which is correct and is not
    the same as saying everything inside is fair game. `.git` is the sharpest
    case: it sits under the root, so `Workspace.resolve` admits it and
    `write_file` to `.git/config` would be allowed -- while `RunCommand` goes to
    the trouble of restricting `git` to read-only subcommands. The intent there
    is plain, and the filesystem tools route straight around it.

    Matching is on path components, not substrings, so `.git` protects `.git/`
    and everything under it without also catching a file called `mygit.py`.
    """

    name = "protect-paths"

    #: Reads are not the concern: knowing what is in `.git/config` is
    #: occasionally useful and never destructive.
    WRITING_TOOLS = ("write_file", "edit_file", "rename_symbol")

    def __init__(self, components: Iterable[str] = (".git",)) -> None:
        self.components = tuple(components)

    def violated(self, path: str) -> Optional[str]:
        """The protected component this path falls under, if any."""
        parts = {part.lower() for part in PurePath(path).parts}
        for component in self.components:
            if component.lower() in parts:
                return component
        return None

    def before(self, tool: str, args: Dict[str, Any]) -> Decision:
        if tool not in self.WRITING_TOOLS:
            return ALLOW
        path = args.get("path")
        if not isinstance(path, str):
            return ALLOW
        component = self.violated(path)
        if component is None:
            return ALLOW
        return deny(f"`{component}` is protected; {path} is not part of the "
                    f"working tree")


def before_chain(hooks: Sequence[Hook], tool: str,
                 args: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Run the `before` chain. Returns `(args to use, denial or None)`.

    The args come back even on a denial, and that is the point: if one hook
    rewrites a path and a later one refuses it, the call that was *attempted* is
    the rewritten one. Reporting the original to `Hook.after` would make an
    audit log describe something nobody tried to do.

    Separated from `dispatch` so the ordering rules -- first denial wins, a
    rewrite is visible to the hooks after it -- can be tested without a
    filesystem or a tool.
    """
    current = args
    for hook in hooks:
        decision = hook.before(tool, current)
        if decision.denial is not None:
            return current, (f"{tool} was refused by `{hook.name}`: "
                             f"{decision.denial}")
        if decision.rewritten is not None:
            current = decision.rewritten
    return current, None
