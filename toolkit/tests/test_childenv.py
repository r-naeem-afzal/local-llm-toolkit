"""The environment `lms` is started with must not carry Electron's "run as a script
engine" switch.

Why this has tests of its own: the fault it guards against is silent. With the switch
inherited from an editor, LM Studio's app exits instead of starting and `lms` only
reports a timeout, so a regression here would look like "the model server is slow
today" and be chased in the wrong place. See `local_llm/childenv.py` for the
measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_llm import server_start
from local_llm.childenv import lms_environment
from local_llm.server_start import LmsServerCommand

SWITCH = "ELECTRON_RUN_AS_NODE"


def test_the_switch_is_removed_and_everything_else_is_kept() -> None:
    base = {"PATH": "C:/bin", SWITCH: "1", "USERPROFILE": "C:/Users/someone"}

    assert lms_environment(base) == {"PATH": "C:/bin", "USERPROFILE": "C:/Users/someone"}


def test_an_environment_without_the_switch_comes_back_unchanged() -> None:
    base = {"PATH": "C:/bin"}

    assert lms_environment(base) == {"PATH": "C:/bin"}


def test_the_callers_own_environment_is_not_changed() -> None:
    # The editor that set the switch still needs it for its own helper programs, so the
    # function must hand back a copy and leave what it was given alone.
    base = {"PATH": "C:/bin", SWITCH: "1"}

    lms_environment(base)

    assert base == {"PATH": "C:/bin", SWITCH: "1"}


def test_with_no_argument_the_live_environment_is_read_at_call_time(monkeypatch: Any) -> None:
    # `monkeypatch.setenv` sets a real environment variable for this one test and puts
    # the old value back afterwards. Setting it after the module was imported shows the
    # environment is read when the function is called, not once at import.
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setenv("LOCAL_LLM_TEST_MARKER", "kept")

    result = lms_environment()

    assert SWITCH not in result
    assert result["LOCAL_LLM_TEST_MARKER"] == "kept"


class _FinishedProcess:
    """Stands in for the process object `subprocess.Popen` returns: one that has
    already ended with exit code 0."""

    returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        return None


def test_server_start_runs_lms_without_the_switch(tmp_path: Path, monkeypatch: Any) -> None:
    # This is the command that wakes LM Studio's service, so it is the one place where
    # the inherited switch did the damage. The test starts no program: `Popen` is
    # replaced by a function that records what it was asked to run and with which
    # environment.
    lms = tmp_path / "lms.exe"
    lms.write_bytes(b"")   # only has to exist; it is never executed
    seen: dict[str, Any] = {}

    def fake_popen(command: list[str], **options: Any) -> _FinishedProcess:
        seen["command"] = command
        seen["env"] = options.get("env")
        return _FinishedProcess()

    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setattr(server_start.subprocess, "Popen", fake_popen)

    outcome = LmsServerCommand(lambda: lms).start(timeout_s=5)

    assert outcome.ran is True
    assert seen["command"] == [str(lms), "server", "start"]
    assert seen["env"] is not None, "lms inherited the whole environment, switch included"
    assert SWITCH not in seen["env"]
