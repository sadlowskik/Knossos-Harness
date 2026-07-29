"""What a subprocess started by the harness can see.

Two claims:

  1. No credential in this process's environment reaches a child. The harness
     reads the provider key from `os.environ`, and it runs agent-authored tests
     on purpose, so those two facts together are a leak unless something stops
     it.
  2. The environment that survives is still enough to run the toolchain. An
     allowlist one variable too narrow does not fail loudly -- it fails as a
     broken import in the middle of an eval run, on whichever machine happens
     to need the variable nobody listed.

The first is checked against a constructed environment, so the test does not
depend on which secrets the machine actually holds. The second runs a real
interpreter.

No torch, no network.

    pytest -q tests/test_sandbox.py
"""
import os
import sys
import textwrap

from knossos.sandbox import DEFAULT, Sandbox


def realistic():
    """A developer machine that has been used for real work."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin"),
        "PYTHONHASHSEED": "0",
        "PYTEST_ADDOPTS": "-q",
        "VIRTUAL_ENV": "/home/k/.venv",
        "ANTHROPIC_API_KEY": "sk-ant-should-never-appear",
        "OPENAI_API_KEY": "sk-should-never-appear",
        "GEMINI_API_KEY": "should-never-appear",
        "AWS_SECRET_ACCESS_KEY": "should-never-appear",
        "GITHUB_TOKEN": "ghp_should-never-appear",
        "PYPI_TOKEN": "pypi-should-never-appear",
        "DATABASE_PASSWORD": "hunter2",
        "SSH_AUTH_SOCK": "/tmp/ssh-agent",
    }


def test_the_harnesss_own_provider_key_does_not_reach_a_child():
    # The leak this module exists for: engine.py reads this variable, and the
    # Oracle and codeval both run pytest over code the agent wrote.
    assert "ANTHROPIC_API_KEY" not in DEFAULT.environ(realistic())


def test_no_value_marked_secret_survives():
    env = DEFAULT.environ(realistic())
    leaked = [name for name, value in env.items()
              if "should-never-appear" in value or value == "hunter2"]
    assert leaked == [], f"credentials reached the child: {leaked}"


def test_a_prefix_rule_cannot_be_used_to_reach_a_token():
    # PYPI_TOKEN matches the PYTHON/PY_ family closely enough to be a real
    # risk, and PIP_ credentials match outright. The secret layer has to win.
    assert DEFAULT.admits("PYTEST_ADDOPTS")
    assert not DEFAULT.admits("PYPI_TOKEN")
    assert not DEFAULT.admits("PIP_INDEX_PASSWORD")


def test_the_toolchain_still_gets_what_it_needs():
    env = DEFAULT.environ(realistic())
    assert env["PATH"]
    assert env["VIRTUAL_ENV"] == "/home/k/.venv"
    assert env["PYTEST_ADDOPTS"] == "-q"


def test_unrecognised_variables_are_dropped_rather_than_passed():
    # The default is deny: a name nobody thought about does not get through
    # just because it looks harmless.
    assert not DEFAULT.admits("SSH_AUTH_SOCK")
    assert not DEFAULT.admits("KUBECONFIG")
    assert not DEFAULT.admits("SOME_INTERNAL_ENDPOINT")


def test_an_operator_can_widen_the_list_but_not_past_the_secret_check():
    s = Sandbox(extra=["KUBECONFIG", "MY_TOKEN"])
    assert s.admits("KUBECONFIG")
    assert not s.admits("MY_TOKEN"), "extra must not override the secret layer"


def test_matching_is_case_insensitive():
    # Windows environment names are case-insensitive, and `Path` is how it is
    # actually spelled there.
    assert DEFAULT.admits("Path")
    assert not DEFAULT.admits("anthropic_api_key")


def test_hashing_is_pinned_so_a_failure_reproduces():
    # Without this a set-ordering bug passes and fails at random between eval
    # runs, which is worse than either outcome consistently.
    assert DEFAULT.environ(realistic())["PYTHONHASHSEED"] == "0"


def test_withheld_names_the_things_it_dropped():
    # A sandbox nobody can inspect is one nobody trusts.
    withheld = DEFAULT.withheld(realistic())
    assert "ANTHROPIC_API_KEY" in withheld
    assert "PATH" not in withheld


# --------------------------------------------------------------- end to end


def test_run_actually_detaches_the_child_from_this_environment(tmp_path, monkeypatch):
    """`environ` could be perfect and `run` could still forget to pass it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-canary")

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os\n"
        "print('KEY=%r' % os.environ.get('ANTHROPIC_API_KEY'))\n",
        encoding="utf-8")

    proc = DEFAULT.run([sys.executable, str(probe)], cwd=tmp_path, timeout=60)

    # Guards against passing vacuously: if the probe never ran, the absence of
    # the canary would prove nothing.
    assert "KEY=" in proc.stdout, f"the probe did not run: {proc.stderr}"
    assert "sk-ant-canary" not in proc.stdout


def test_a_test_the_agent_wrote_cannot_read_the_key(tmp_path, monkeypatch):
    """The eval-loop shape of the same leak, through a real pytest run."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-eval-canary")

    (tmp_path / "test_written_by_the_agent.py").write_text(textwrap.dedent("""
        import os

        def test_exfiltrate():
            print("KEY=%r" % os.environ.get("ANTHROPIC_API_KEY"))
    """), encoding="utf-8")

    proc = DEFAULT.run(
        [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
         str(tmp_path)],
        cwd=tmp_path, timeout=300)

    output = proc.stdout + proc.stderr
    assert "KEY=" in output, f"the probe test did not run:\n{output}"
    assert "sk-ant-eval-canary" not in output


def test_the_interpreter_still_works_under_the_sandbox(tmp_path):
    """The allowlist has to leave a usable toolchain, not just a safe one."""
    proc = DEFAULT.run(
        [sys.executable, "-c",
         "import json, pathlib, subprocess, sys; print('ok')"],
        cwd=tmp_path, timeout=60)

    assert proc.returncode == 0, (
        f"the allowlist is too narrow on this machine.\n"
        f"stderr: {proc.stderr}\nwithheld: {DEFAULT.withheld()}")
    assert proc.stdout.strip() == "ok"


def test_pytest_itself_still_runs_under_the_sandbox(tmp_path):
    # The Oracle and codeval both shell out to pytest; if the sandbox broke
    # that, every verification tier would report a skip and nothing would fail.
    (tmp_path / "test_trivial.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")

    proc = DEFAULT.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         str(tmp_path)],
        cwd=tmp_path, timeout=300)

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


def test_no_shell_is_involved(tmp_path):
    # argv reaches the OS as a program and its arguments; operators are inert.
    proc = DEFAULT.run(
        [sys.executable, "-c", "import sys; print(sys.argv[1:])",
         "&&", "echo", "pwned"],
        cwd=tmp_path, timeout=60)

    assert "&&" in proc.stdout, "the operator should arrive as a literal"
    assert "pwned" not in proc.stdout.replace("'pwned'", "")


def test_a_child_cannot_block_on_stdin(tmp_path):
    # Reading stdin must hit EOF rather than wait for a person who is not there.
    proc = DEFAULT.run(
        [sys.executable, "-c", "import sys; print('read %r' % sys.stdin.read())"],
        cwd=tmp_path, timeout=30)

    assert proc.returncode == 0
    assert "read ''" in proc.stdout
