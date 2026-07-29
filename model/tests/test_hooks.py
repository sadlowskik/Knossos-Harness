"""Policy around a tool call, and the hole it closes.

The path jail stops at the workspace boundary, which is correct and is not the
same as saying everything inside it is fair game. `.git` sits under the root, so
`Workspace.resolve` admits it -- while `RunCommand` restricts `git` to read-only
subcommands. The intent is plain and the filesystem tools routed around it.

No torch, no network.

    pytest -q tests/test_hooks.py
"""
from knossos.hooks import ALLOW, Decision, Hook, ProtectPaths, before_chain, deny, rewrite
from knossos.tools import ToolCall, ToolRegistry, WriteFile
from knossos.workspace import Workspace


def test_writes_into_the_git_directory_are_refused():
    h = ProtectPaths()
    assert h.before("write_file", {"path": ".git/config"}).denial is not None


def test_the_working_tree_is_untouched_by_the_rule():
    h = ProtectPaths()
    assert h.before("write_file", {"path": "src/main.py"}) == ALLOW
    # A substring match would have caught these two.
    assert h.before("write_file", {"path": "src/mygit.py"}) == ALLOW
    assert h.before("write_file", {"path": "gitignore.py"}) == ALLOW


def test_reads_are_never_blocked():
    h = ProtectPaths()
    assert h.before("read_file", {"path": ".git/config"}) == ALLOW
    assert h.before("list_dir", {"path": ".git"}) == ALLOW


def test_nested_paths_under_a_protected_component_are_caught():
    h = ProtectPaths()
    d = h.before("edit_file", {"path": "a/b/.git/hooks/pre-commit"})
    assert d.denial is not None


def test_the_protected_set_is_configurable():
    h = ProtectPaths(["migrations", "vendor"])
    assert h.before("write_file", {"path": "migrations/003.sql"}).denial
    assert h.before("write_file", {"path": ".git/config"}) == ALLOW


class Rewriter(Hook):
    name = "rewriter"

    def before(self, tool, args):
        return rewrite({"path": ".git/config", "content": "x"})


class Blocker(Hook):
    name = "blocker"

    def before(self, tool, args):
        return deny("no")


def test_a_rewrite_is_visible_to_the_hooks_that_follow_it():
    # Otherwise a rewrite could smuggle a path past a later policy hook, which
    # would make hook order a security boundary.
    args, denial = before_chain([Rewriter(), ProtectPaths()],
                                "write_file", {"path": "src/ok.py"})
    assert denial is not None


def test_a_denial_reports_the_call_that_was_actually_attempted():
    args, denial = before_chain([Rewriter(), ProtectPaths()],
                                "write_file", {"path": "src/harmless.py"})
    assert denial is not None
    assert args["path"] == ".git/config", (
        "reporting the original would describe an attempt nobody made")


def test_the_first_denial_wins_and_names_itself():
    _, denial = before_chain([Blocker(), Rewriter()], "write_file", {})
    assert "blocker" in denial


def test_an_empty_chain_passes_the_input_through_untouched():
    args = {"path": "a.py"}
    out, denial = before_chain([], "write_file", args)
    assert denial is None
    assert out == args


# ------------------------------------------------------------ through dispatch


class Counting(Hook):
    name = "counting"

    def __init__(self):
        self.seen = []

    def after(self, tool, args, result):
        self.seen.append((tool, result.is_error))


def test_the_default_registry_refuses_to_write_into_dot_git(tmp_path):
    # Checked through dispatch rather than against the hook alone: the path jail
    # admits `.git` because it is under the root, so nothing else stops this.
    (tmp_path / ".git").mkdir()
    w = Workspace(tmp_path)
    registry = ToolRegistry([WriteFile()])

    result = registry.dispatch(
        ToolCall(name="write_file",
                 args={"path": ".git/config", "content": "[remote]\n"}), w)

    assert result.is_error
    assert "protect-paths" in result.content
    assert not (tmp_path / ".git" / "config").exists()


def test_ordinary_writes_are_unaffected_by_the_default_hook(tmp_path):
    w = Workspace(tmp_path)
    registry = ToolRegistry([WriteFile()])
    result = registry.dispatch(
        ToolCall(name="write_file",
                 args={"path": "src/main.py", "content": "x = 1\n"}), w)
    assert not result.is_error, result.content


def test_after_hooks_see_every_call_including_the_ones_that_failed(tmp_path):
    w = Workspace(tmp_path)
    counter = Counting()
    registry = ToolRegistry([WriteFile()]).with_hook(counter)

    # 1: ordinary success.
    registry.dispatch(ToolCall(name="write_file",
                               args={"path": "ok.py", "content": "x\n"}), w)
    # 2: denied by the default hook before the tool ran.
    registry.dispatch(ToolCall(name="write_file",
                               args={"path": ".git/config", "content": "x\n"}), w)
    # 3: a name no tool answers to.
    registry.dispatch(ToolCall(name="no_such_tool", args={}), w)

    assert counter.seen == [
        ("write_file", False),
        ("write_file", True),
        ("no_such_tool", True),
    ], f"an audit hook that misses a call is not an audit hook: {counter.seen}"


def test_a_child_registry_keeps_the_policy_its_parent_ran_under(tmp_path):
    # Delegation must not be a way to gain a capability.
    (tmp_path / ".git").mkdir()
    w = Workspace(tmp_path)
    child = ToolRegistry([WriteFile()]).without(["read_file"])

    result = child.dispatch(
        ToolCall(name="write_file",
                 args={"path": ".git/config", "content": "x\n"}), w)
    assert result.is_error
    assert "protect-paths" in result.content
