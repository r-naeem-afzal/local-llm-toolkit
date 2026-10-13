"""One model load at a time, across processes.

Each agent runs its own copy of the MCP server, so the lock inside the gate (a
`threading.Lock`) does not stop two of them loading at once. The loader therefore takes a
lock FILE around the whole "look again, then unload and load". These tests stand two
loaders side by side, each with its own `LockFile` object on the same path, which is what
two processes are to each other: separate objects that share only the disk.

Nothing real is started. `lms` is replaced by a recorder, the lock file is in pytest's
temporary folder, and the clock only moves when `sleep` is called.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fakes import FakeLoader, FakeRegistry, installed_entry

from local_llm.client import LocalLLMError
from local_llm.gate import ResidencyGate
from local_llm.loader import ModelLoader
from local_llm.lockfile import LockFile

NINE_B = "qwen/qwen3.5-9b"
FOURTEEN_B = "qwen/qwen3-14b"


class FakeClock:
    """Time that only moves when told to: `sleep` moves it, so a three minute wait costs
    nothing and a test can still say how long the code thought it waited."""

    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def monotonic(self) -> float:
        return self.now

    def wall(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class RecordingLoader(ModelLoader):
    """The real loader with its one way of starting `lms` replaced, and nothing else.

    A load or unload updates the shared fake registry, as the real command changes the
    one real server, so the other "process" sees it on its next look.
    """

    def __init__(self, registry: FakeRegistry, raises: Exception | None = None,
                 **options) -> None:
        super().__init__(runner=None, registry=registry, sleep=lambda seconds: None,
                         **options)
        self._raises = raises
        self.commands: list[list[str]] = []

    def _run(self, args: list[str]) -> bool:
        self.commands.append(list(args))
        if self._raises is not None:
            raise self._raises
        if args[0] == "load":
            self._registry.mark_loaded(args[1])
        elif args[0] == "unload":
            self._registry.clear()
        return True

    def _settle(self) -> None:
        return None

    def loads(self) -> list[str]:
        return [command[1] for command in self.commands if command[0] == "load"]


class Machine:
    """One pretend computer: one model server (the registry), one lock path, one clock.
    `process()` gives a loader with a lock object of its own, as a new process would have."""

    def __init__(self, tmp_path: Path, alive: set[int] | None = None) -> None:
        self.clock = FakeClock()
        self.lock_path = tmp_path / "model-load.lock"
        self.registry = FakeRegistry(installed=[
            installed_entry(NINE_B, size_mib=6245),
            installed_entry(FOURTEEN_B, size_mib=8584),
        ])
        # The process numbers that count as running. None means "do not check".
        self._alive = alive

    def lock(self) -> LockFile:
        is_alive = None if self._alive is None else (lambda pid: pid in self._alive)
        return LockFile(self.lock_path, self.clock, stale_after_s=600, is_alive=is_alive)

    def process(self, wait_s: float = 180, **options) -> RecordingLoader:
        return RecordingLoader(self.registry, load_lock=self.lock(),
                               load_lock_wait_s=wait_s, **options)

    def hold(self, note: str) -> LockFile:
        """Another process takes the lock now and keeps it. Returns its lock object.

        The file's modification time is set to the fake clock's "now". The real file
        system stamps it with the real time, which is years away from the fake clock and
        would make a lock taken this instant look stale.
        """
        holder = self.lock()
        assert holder.acquire(note)
        os.utime(self.lock_path, (self.clock.now, self.clock.now))
        return holder

    def write_lock(self, text: str, age_s: float) -> None:
        """Leave a lock file behind, as another process would, `age_s` seconds old."""
        self.lock_path.write_text(text, encoding="utf-8")
        stamp = self.clock.now - age_s
        os.utime(self.lock_path, (stamp, stamp))


# ─────────────────────────── taking turns ───────────────────────────


def test_the_second_process_finds_the_model_the_first_loaded_and_loads_nothing(tmp_path):
    machine = Machine(tmp_path)
    first, second = machine.process(), machine.process()
    assert first.ensure(NINE_B)
    assert second.ensure(NINE_B)
    assert first.loads() == [NINE_B]
    assert second.commands == []                 # no unload, no load: it only looked
    assert not machine.lock_path.exists()        # and both gave the lock back


def test_a_fresh_look_at_what_is_loaded_is_taken_after_the_lock_is_won(tmp_path):
    # What a process read before it waited says "nothing is loaded". Acting on that would
    # load the model a second time, so the cached answer is dropped first.
    machine = Machine(tmp_path)
    loader = machine.process()
    loader.ensure(NINE_B)
    assert machine.registry.refreshes == 1


def test_the_lock_is_held_for_the_whole_of_the_load(tmp_path):
    machine = Machine(tmp_path)
    seen: list[str] = []

    class Watching(RecordingLoader):
        def _run(self, args: list[str]) -> bool:
            # What a second process would be told if it looked at this moment.
            seen.append(machine.lock().describe_holder())
            return super()._run(args)

    loader = Watching(machine.registry, load_lock=machine.lock(), load_lock_wait_s=180)
    assert loader.ensure(NINE_B)
    assert len(seen) == 2                        # during the unload and during the load
    for told in seen:
        assert f"process {os.getpid()}, loading {NINE_B}, since " in told


def test_a_process_that_waits_gets_its_turn_when_the_holder_finishes(tmp_path):
    machine = Machine(tmp_path)
    holder = machine.hold("loading qwen/qwen3-14b")
    real_sleep = machine.clock.sleep

    def sleep_and_finish_at_14_seconds(seconds: float) -> None:
        real_sleep(seconds)
        if machine.clock.now - 1_800_000_000.0 >= 14:
            # The other process finishes its load and lets go.
            machine.registry.mark_loaded(FOURTEEN_B)
            holder.release()

    machine.clock.sleep = sleep_and_finish_at_14_seconds  # type: ignore[method-assign]
    waiting = machine.process()
    assert waiting.ensure(FOURTEEN_B)
    assert waiting.commands == []                # the model was there when its turn came
    assert machine.clock.now - 1_800_000_000.0 == 14.0


# ─────────────────────────── a lock that is not given back ───────────────────────────


def test_a_lock_held_too_long_is_refused_with_who_holds_it_and_since_when(tmp_path):
    machine = Machine(tmp_path)
    holder = machine.hold("loading qwen/qwen3-14b")
    waiting = machine.process(wait_s=180)
    assert not waiting.ensure(NINE_B)
    assert waiting.commands == []                # nothing unloaded, nothing loaded
    assert machine.clock.now - 1_800_000_000.0 == 180.0     # waited the limit, no longer
    error = waiting.last_error
    assert error.startswith(f"{NINE_B} was not loaded: another process holds the model "
                            f"load lock (process {os.getpid()}, loading qwen/qwen3-14b, since ")
    assert "(180 seconds ago)" in error
    assert " am (" in error or " pm (" in error  # a time of day a person reads
    assert error.endswith("did not release it within 180 seconds")
    assert machine.lock_path.exists()            # not ours, so not removed


def test_the_gate_turns_that_into_a_refusal_and_nothing_is_sent(tmp_path):
    machine = Machine(tmp_path)
    holder = machine.hold("loading qwen/qwen3-14b")
    gate = ResidencyGate(machine.process(wait_s=30), machine.registry)
    with pytest.raises(LocalLLMError, match="holds the model load lock"):
        gate.ensure_resident(NINE_B)


def test_a_stale_lock_is_taken_over(tmp_path):
    machine = Machine(tmp_path)
    machine.write_lock("pid 4321\nsince 1.0\nnote loading qwen/qwen3-14b\n", age_s=7200)
    loader = machine.process()
    assert loader.ensure(NINE_B)
    assert loader.loads() == [NINE_B]
    assert machine.clock.now == 1_800_000_000.0  # taken over at once, no waiting
    assert not machine.lock_path.exists()        # and released afterwards


def test_a_lock_whose_holder_has_died_is_taken_over_without_waiting_for_it_to_age(tmp_path):
    machine = Machine(tmp_path, alive={111})     # process 4321 is not running
    machine.write_lock("pid 4321\nsince 1.0\n", age_s=20)
    loader = machine.process()
    assert loader.ensure(NINE_B)
    assert machine.clock.now == 1_800_000_000.0


def test_a_young_lock_whose_holder_is_running_is_respected(tmp_path):
    machine = Machine(tmp_path, alive={4321})
    machine.write_lock("pid 4321\nsince 1.0\n", age_s=20)
    loader = machine.process(wait_s=5)
    assert not loader.ensure(NINE_B)
    assert "process 4321" in loader.last_error


def test_a_lock_file_still_being_written_is_not_taken_for_a_dead_holder(tmp_path):
    machine = Machine(tmp_path, alive=set())
    machine.write_lock("", age_s=0)              # created, process number not yet in it
    assert not machine.lock().acquire()


def test_the_lock_is_released_when_the_load_raises(tmp_path):
    machine = Machine(tmp_path)
    loader = machine.process(raises=RuntimeError("lms exploded"))
    with pytest.raises(RuntimeError, match="lms exploded"):
        loader.ensure(NINE_B)
    assert not machine.lock_path.exists()
    # The next process gets its turn at once.
    assert machine.process().ensure(NINE_B)


def test_the_lock_is_released_when_the_load_is_refused(tmp_path):
    machine = Machine(tmp_path)
    loader = machine.process()
    machine.registry.mark_loaded(FOURTEEN_B)
    # Not in the registry's list of installed models, so the loader refuses to evict the
    # 14B for it and returns False from the middle of the sequence.
    assert not loader.ensure("some/model-that-is-not-installed")
    assert "not in the model registry" in loader.last_error
    assert not machine.lock_path.exists()


def test_a_loader_built_without_a_lock_still_loads(tmp_path):
    registry = FakeRegistry(installed=[installed_entry(NINE_B)])
    loader = RecordingLoader(registry)
    assert loader.ensure(NINE_B) and registry.refreshes == 0


def test_the_shared_gate_fake_is_unchanged_by_all_this():
    # `fakes.FakeLoader` stands in for the loader in the gate's own tests and has no lock;
    # the gate must keep working with a loader that takes turns some other way, or not at all.
    registry = FakeRegistry()
    ResidencyGate(FakeLoader(registry), registry).ensure_resident(NINE_B)
    assert [model.key for model in registry.loaded()] == [NINE_B]
