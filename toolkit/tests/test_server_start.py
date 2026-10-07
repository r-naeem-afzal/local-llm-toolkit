"""Tests for the unattended server starter. Everything is a fake: no process is started, no
model is loaded, no real clock is waited on. Each fake is a few lines so a reader can see
exactly what scenario a test sets up."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_llm.config import Settings
from local_llm.server_start import (
    CommandOutcome,
    LockFile,
    ModelServerStarter,
    StartResult,
    main,
)

MODEL = "qwen/qwen3-14b"


class FakeClock:
    """Time that only moves when told to. `sleep` moves it, so a 30 second wait costs
    nothing and the test can still assert how long the code thought it waited."""

    def __init__(self) -> None:
        self.start = 1_000_000.0
        self.now = self.start

    def monotonic(self) -> float:
        return self.now

    def wall(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeServer:
    """The state of the pretend server, shared by the fake probe and the fake command."""

    def __init__(self, up: bool = False) -> None:
        self.up = up


class FakeProbe:
    def __init__(self, server: FakeServer, raises: bool = False) -> None:
        self._server = server
        self._raises = raises

    def is_up_now(self) -> bool:
        if self._raises:
            raise RuntimeError("probe exploded")
        return self._server.up


class FakeCommand:
    """`lms server start`. `ups` says, per call, whether that call brings the server up:
    [False, True] is the cold-start quirk (first call fails, second works)."""

    def __init__(self, server: FakeServer, ups: list[bool], ran: bool = True,
                 timed_out: bool = False) -> None:
        self._server = server
        self._ups = list(ups)
        self._ran = ran
        self._timed_out = timed_out
        self.calls = 0

    def start(self, timeout_s: float) -> CommandOutcome:
        self.calls += 1
        if not self._ran:
            return CommandOutcome(False, False, None, "lms not found at X")
        if self._ups and self._ups.pop(0):
            self._server.up = True
        return CommandOutcome(True, self._timed_out, None if self._timed_out else 1,
                              "Timed out waiting for LM Studio daemon to start")


class FakeLoader:
    def __init__(self, succeeds: bool = True, last_error: str = "") -> None:
        self._succeeds = succeeds
        self.last_error = last_error
        self.asked: list[str] = []

    def ensure(self, key: str) -> bool:
        self.asked.append(key)
        return self._succeeds


class FakeBudget:
    def __init__(self, free: int | None = 14000) -> None:
        self._free = free

    def free_mib(self) -> int | None:
        return self._free


class FakeMemory:
    def __init__(self, free: int | None = 20000) -> None:
        self._free = free

    def free_ram_mib(self) -> int | None:
        return self._free


class Rig:
    """Everything wired together, with the pieces exposed so a test can inspect them."""

    def __init__(self, tmp_path: Path, *, up: bool = False, ups: list[bool] | None = None,
                 ran: bool = True, timed_out: bool = False, loader_ok: bool = True,
                 loader_error: str = "", vram: int | None = 14000, ram: int | None = 20000,
                 probe_raises: bool = False) -> None:
        self.clock = FakeClock()
        self.server = FakeServer(up)
        self.command = FakeCommand(self.server, ups if ups is not None else [True], ran,
                                   timed_out)
        self.loader = FakeLoader(loader_ok, loader_error)
        self.lock_path = tmp_path / "start.lock"
        self.lock = LockFile(self.lock_path, self.clock, stale_after_s=600)
        self.starter = ModelServerStarter(
            # _env_file=None so a developer's real .env cannot change what a test sees.
            settings=Settings(model=MODEL, min_free_ram_mib=6144,
                              server_start_wait_s=30, _env_file=None),
            probe=FakeProbe(self.server, probe_raises),
            command=self.command,
            loader=self.loader,
            budget=FakeBudget(vram),
            memory=FakeMemory(ram),
            lock=self.lock,
            clock=self.clock,
        )

    def age_lock(self, seconds: float) -> None:
        """Write a lock file whose modification time is `seconds` before fake-now."""
        self.lock_path.write_text("pid 1")
        stamp = self.clock.now - seconds
        os.utime(self.lock_path, (stamp, stamp))


# ─────────────────────────── server start ───────────────────────────


def test_server_already_up_runs_no_command_and_loads(tmp_path):
    rig = Rig(tmp_path, up=True)
    result = rig.starter.ensure_ready()
    assert rig.command.calls == 0
    assert result.server_up and not result.server_started_by_us
    assert result.model_loaded and result.model == MODEL
    assert rig.loader.asked == [MODEL]
    assert result.vram_free_mib == 14000 and result.ram_free_mib == 20000


def test_cold_start_needs_the_second_call(tmp_path):
    rig = Rig(tmp_path, ups=[False, True])
    result = rig.starter.ensure_ready()
    assert rig.command.calls == 2
    assert result.server_up and result.server_started_by_us and result.model_loaded


def test_first_call_enough_means_no_second_call(tmp_path):
    rig = Rig(tmp_path, ups=[True])
    rig.starter.ensure_ready()
    assert rig.command.calls == 1


def test_server_never_answers_is_bounded_and_reported(tmp_path):
    rig = Rig(tmp_path, ups=[False, False])
    result = rig.starter.ensure_ready()
    assert rig.command.calls == 2          # exactly one retry, never a loop
    assert not result.server_up and not result.model_loaded
    assert "did not answer within 30 seconds" in result.error
    assert rig.loader.asked == []           # nothing loaded onto a server that is down
    assert 30 <= result.seconds <= 40       # the wait was bounded by the 30 second limit
    assert not rig.lock_path.exists()


def test_server_answers_late_within_the_wait(tmp_path):
    rig = Rig(tmp_path, ups=[False, False])
    real_sleep = rig.clock.sleep

    def sleep_then_maybe_up(seconds: float) -> None:
        # The commands never bring it up, but it answers 9 seconds into the wait.
        real_sleep(seconds)
        if rig.clock.now - rig.clock.start >= 9:
            rig.server.up = True

    rig.clock.sleep = sleep_then_maybe_up  # type: ignore[method-assign]
    result = rig.starter.ensure_ready()
    assert result.server_up and result.model_loaded


def test_lms_missing_is_a_result_not_an_exception(tmp_path):
    rig = Rig(tmp_path, ran=False)
    result = rig.starter.ensure_ready()
    assert rig.command.calls == 1           # a missing file stays missing; no retry
    assert not result.server_up
    assert "could not run lms" in result.error
    assert not rig.lock_path.exists()


def test_timed_out_command_is_named_in_the_error(tmp_path):
    rig = Rig(tmp_path, ups=[False, False], timed_out=True)
    result = rig.starter.ensure_ready()
    assert "timed out" in result.error


# ─────────────────────────── load guards ───────────────────────────


def test_low_ram_refuses_load_but_server_stays_up(tmp_path):
    rig = Rig(tmp_path, up=True, ram=3100)
    result = rig.starter.ensure_ready()
    assert result.server_up and not result.model_loaded
    assert "3100" in result.refused and "6144" in result.refused
    assert rig.loader.asked == []
    assert result.ram_free_mib == 3100


def test_ram_exactly_at_floor_is_allowed(tmp_path):
    rig = Rig(tmp_path, up=True, ram=6144)
    assert rig.starter.ensure_ready().model_loaded


def test_unreadable_gpu_refuses_load(tmp_path):
    rig = Rig(tmp_path, up=True, vram=None)
    result = rig.starter.ensure_ready()
    assert result.server_up and not result.model_loaded
    assert "GPU" in result.refused
    assert rig.loader.asked == []


def test_loader_refusal_is_passed_through(tmp_path):
    rig = Rig(tmp_path, up=True, loader_ok=False,
              loader_error="qwen/qwen3-14b needs about 14000 MiB but only 9000 MiB free")
    result = rig.starter.ensure_ready()
    assert result.server_up and not result.model_loaded
    assert result.error == "qwen/qwen3-14b needs about 14000 MiB but only 9000 MiB free"
    assert result.vram_free_mib == 14000


def test_no_load_skips_every_load_guard(tmp_path):
    rig = Rig(tmp_path, up=True, ram=10, vram=None)
    result = rig.starter.ensure_ready(load_model=False)
    assert result.server_up and not result.model_loaded
    assert result.refused == "" and result.error == ""
    assert rig.loader.asked == []


def test_model_argument_overrides_the_default(tmp_path):
    rig = Rig(tmp_path, up=True)
    result = rig.starter.ensure_ready(model_key="qwen/qwen3-4b")
    assert rig.loader.asked == ["qwen/qwen3-4b"] and result.model == "qwen/qwen3-4b"


# ─────────────────────────── lock ───────────────────────────


def test_fresh_lock_held_by_someone_else_starts_nothing(tmp_path):
    rig = Rig(tmp_path, up=False)
    rig.age_lock(30)
    result = rig.starter.ensure_ready()
    assert rig.command.calls == 0 and rig.loader.asked == []
    assert "another starter" in result.refused
    assert rig.lock_path.exists()            # not ours, so not removed


def test_stale_lock_is_taken_over(tmp_path):
    rig = Rig(tmp_path, up=True)
    rig.age_lock(7200)
    result = rig.starter.ensure_ready()
    assert result.model_loaded
    assert not rig.lock_path.exists()        # taken over, then released


def test_lock_is_removed_after_a_normal_run(tmp_path):
    rig = Rig(tmp_path, up=True)
    rig.starter.ensure_ready()
    assert not rig.lock_path.exists()


def test_lock_is_removed_after_an_exception(tmp_path):
    rig = Rig(tmp_path, up=True, probe_raises=True)
    with pytest.raises(RuntimeError):
        rig.starter.ensure_ready()
    assert not rig.lock_path.exists()


def test_second_acquire_fails_while_first_holds(tmp_path):
    clock = FakeClock()
    path = tmp_path / "x.lock"
    first = LockFile(path, clock, 600)
    second = LockFile(path, clock, 600)
    assert first.acquire() is True
    os.utime(path, (clock.now, clock.now))   # make the file's age match the fake clock
    assert second.acquire() is False
    second.release()                          # must not delete first's lock
    assert path.exists()
    first.release()
    assert not path.exists()


# ─────────────────────────── command line ───────────────────────────


class StubStarter:
    def __init__(self, result: StartResult | Exception) -> None:
        self._result = result
        self.asked: tuple[str | None, bool] | None = None

    def ensure_ready(self, model_key=None, load_model=True):
        self.asked = (model_key, load_model)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def make_result(**changes) -> StartResult:
    base = {"server_up": True, "server_started_by_us": False, "model": MODEL,
            "model_loaded": True, "refused": "", "error": "", "vram_free_mib": 14000,
            "ram_free_mib": 20000, "seconds": 1.5}
    base.update(changes)
    return StartResult(**base)


def run_main(argv, result):
    lines: list[str] = []
    stub = StubStarter(result)
    code = main(argv, starter=stub, out=lines.append)  # type: ignore[arg-type]
    return code, lines, stub


def test_json_output_is_one_line_with_every_field():
    code, lines, _ = run_main(["--json"], make_result())
    assert code == 0 and len(lines) == 1
    data = json.loads(lines[0])
    assert set(data) == {
        "server_up", "server_started_by_us", "model", "model_loaded", "refused", "error",
        "vram_free_mib", "ram_free_mib", "seconds",
    }
    assert data["model"] == MODEL and data["seconds"] == 1.5


def test_plain_output_is_sentences_without_em_dashes():
    code, lines, _ = run_main([], make_result())
    assert code == 0
    assert lines[0] == ("The model server is up (it was already running). "
                        f"{MODEL} is loaded. Took 1.5 seconds.")
    assert chr(0x2014) not in lines[0]


def test_exit_one_when_model_not_loaded():
    code, lines, _ = run_main([], make_result(model_loaded=False, refused="free RAM is low"))
    assert code == 1 and "free RAM is low" in lines[0]


def test_exit_one_when_server_down_even_with_no_load():
    code, _, _ = run_main(["--no-load"], make_result(server_up=False, model_loaded=False))
    assert code == 1


def test_no_load_succeeds_with_server_up_and_passes_flags_through():
    code, _, stub = run_main(["--no-load", "--model", "m/x"], make_result(model_loaded=False))
    assert code == 0 and stub.asked == ("m/x", False)


def test_unexpected_exception_becomes_an_error_line_and_exit_one():
    code, lines, _ = run_main(["--json"], RuntimeError("boom"))
    assert code == 1
    assert json.loads(lines[0])["error"] == "RuntimeError: boom"
