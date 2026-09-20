"""What a command inherits once it is allowed to run.

`tools.Run` decides *which* programs may run, and that question has a ceiling:
the allowlist admits `pytest`, and `pytest` executes whatever is in the
workspace. The Oracle's tiers run it too, unprompted, on every verification.
`codeval._pytest` runs it against code an agent wrote seconds earlier, as the
core of the eval loop. So the harness deliberately, repeatedly, executes
agent-authored code -- no tightening of the allowlist changes that, because
running the tests is the point.

What is left to control is the environment that code lands in. A child process
inherits its parent's entire environment by default, and `engine.py` reads the
provider key out of exactly that environment (`os.environ.get(self.key_env)`).
A test that prints `os.environ["ANTHROPIC_API_KEY"]` puts the key in the
command's stdout, which the harness folds into the transcript and sends to the
model provider on the next turn. That is a credential leak reachable with no
path-jail escape and no allowlist bypass, using only what the agent is supposed
to have.

So the child gets an explicitly named environment rather than an inherited one,
and there is one function that starts a subprocess so the five call sites cannot
drift apart.

What this does not do
---------------------

This is environment isolation, not process isolation. A sandboxed child can
still write anywhere the account can write -- `Workspace.resolve` jails the
harness's own tools, not a subprocess those tools start -- and can still open
sockets. Closing those needs OS-level containment, which is a different and much
larger piece of work; it is left undone rather than approximated, so that the
guarantee this module *does* make stays believable.

Mirrors `knossos-rs/src/sandbox.rs`. The two lists are allowed to differ, and
do: this one has to keep a Python toolchain working.
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, Iterable, Mapping, Optional, Sequence

__all__ = ["Sandbox", "DEFAULT"]

#: Without this nothing runs at all.
_BASE = ("PATH",)

#: Platform variables the interpreter and the linker need to function.
_PLATFORM_WINDOWS = (
    # Win32 fails in obscure ways without the first few.
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT",
    "TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "PROGRAMDATA",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    # The MSVC linker reads its search paths from the environment; a native
    # extension building without these fails at the link step.
    "LIB", "INCLUDE", "VCINSTALLDIR", "VCTOOLSINSTALLDIR",
    "WINDOWSSDKDIR", "WINDOWSSDKVERSION", "UNIVERSALCRTSDKDIR", "UCRTVERSION",
)

_PLATFORM_POSIX = ("HOME", "TMPDIR", "LANG", "LC_ALL", "TERM", "SHLVL")

#: Toolchain variables that carry no useful prefix.
_TOOLCHAIN = (
    "VIRTUAL_ENV", "CONDA_PREFIX", "RUSTC", "RUSTDOC", "RUSTFLAGS", "TERM",
)

#: Families admitted wholesale. Enumerating each member would break on the next
#: release of any of these tools, and the layer below still refuses secrets.
_PREFIXES = ("PYTHON", "PY_", "PIP_", "PYTEST_", "MYPY", "RUFF_",
             "CARGO_", "RUSTUP_", "RUST_")

#: Substrings that disqualify a name even when a rule above would admit it.
#:
#: This exists because `_PREFIXES` is generous, and among the things it would
#: otherwise admit are `PYPI_TOKEN` and `PIP_INDEX_URL` credentials. A prefix
#: rule wide enough to be maintainable is wide enough to leak, so the two rules
#: are layered rather than merged.
_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL",
                   "APIKEY", "_KEY")


class Sandbox:
    """The environment policy applied to every command the harness runs."""

    def __init__(self, extra: Iterable[str] = (), offline: bool = True) -> None:
        #: Names the operator added, for toolchains this module did not
        #: anticipate. The secret check still applies to them.
        self.extra = tuple(name.upper() for name in extra)
        #: Whether package managers may reach the network. Off by default: a
        #: run that quietly installs a dependency has changed the environment
        #: in a way the diff does not show.
        self.offline = offline

    # ------------------------------------------------------------- policy

    def admits(self, name: str) -> bool:
        """Whether a variable of this name reaches the child."""
        upper = name.upper()

        # Checked first, so no rule below can be used to reach a secret.
        if any(marker in upper for marker in _SECRET_MARKERS):
            return False

        platform = _PLATFORM_WINDOWS if os.name == "nt" else _PLATFORM_POSIX
        return (upper in _BASE
                or upper in platform
                or upper in _TOOLCHAIN
                or upper in self.extra
                or any(upper.startswith(p) for p in _PREFIXES))

    def environ(self, source: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        """The environment a child receives, given the one this process holds.

        Takes the source mapping as an argument rather than always reading the
        real one, so the policy can be tested against a constructed environment
        holding secrets that are not present on the machine.
        """
        source = os.environ if source is None else source
        env = {k: v for k, v in source.items() if self.admits(k)}

        # Colour codes are noise in a transcript a model has to read, and they
        # cost tokens on every line of output.
        env["PY_COLORS"] = "0"
        env["NO_COLOR"] = "1"
        env["CARGO_TERM_COLOR"] = "never"
        # Deterministic hashing, so a failure reproduces. Without it a
        # set-ordering bug passes and fails at random across eval runs.
        env["PYTHONHASHSEED"] = "0"

        if self.offline:
            env["PIP_NO_INPUT"] = "1"
            env["CARGO_NET_OFFLINE"] = "true"

        return env

    def withheld(self, source: Optional[Mapping[str, str]] = None) -> list:
        """The names that would be dropped. For explaining the policy."""
        source = os.environ if source is None else source
        return sorted(k for k in source if not self.admits(k))

    # -------------------------------------------------------------- spawn

    def run(self, argv: Sequence[str], *, cwd, timeout: float,
            **kwargs) -> subprocess.CompletedProcess:
        """Run a command under this policy.

        The only place the harness starts a subprocess. Five call sites -- the
        `run` tool, three in the Oracle, and `codeval._pytest` -- each need the
        same four things beyond the environment, and five call sites each
        remembering four settings is five that can drift. One function cannot.

        Beyond the scrubbed environment those settings are:

        * **No shell.** `shell=False` so `&&`, `|` and backticks are inert;
          argv is passed to the OS as a program and a list of arguments.
        * **No stdin.** There is nobody to type at it, and a child that blocks
          reading stdin hangs until the timeout, which then reports it as slow
          rather than stuck.
        * **Captured, decoded output**, because every caller wants text.
        """
        return subprocess.run(
            list(argv),
            cwd=cwd,
            timeout=timeout,
            env=self.environ(),
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            shell=False,
            **kwargs,
        )


#: The policy every call site uses unless it was handed another one.
DEFAULT = Sandbox()
