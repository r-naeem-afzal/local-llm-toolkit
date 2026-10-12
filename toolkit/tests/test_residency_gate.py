"""The residency gate: a request may name a model only once that model is loaded.

Everything is a fake: no `lms`, no GPU, no model server. A "load" is a method on a fake that
adds the model to a fake registry, so the tests can say exactly how many loads happened and
in what order things were looked at.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import types

import pytest
from fakes import FakeLoader, FakeRegistry, loaded_model

from local_llm.client import LocalLLMError, ModelGate
from local_llm.config import Settings
from local_llm.gate import ResidencyGate
from local_llm.loader import ModelLoader, VramBudget
from local_llm.monitor import GpuInfo, LmsCommandRunner, ModelRegistry

NINE_B = "qwen/qwen3.5-9b"
FOURTEEN_B = "qwen/qwen3-14b"


def make_gate(*, loaded=(), **loader_options):
    """A gate over a registry that already holds `loaded`, and a loader for the rest."""
    registry = FakeRegistry(loaded=[loaded_model(key) for key in loaded])
    loader = FakeLoader(registry, **loader_options)
    return ResidencyGate(loader, registry), loader, registry


def test_a_gate_can_stand_wherever_the_client_expects_a_model_gate():
    gate, _, _ = make_gate()
    assert isinstance(gate, ModelGate)


# ─────────────────────────── a model that is loaded ───────────────────────────


def test_a_loaded_model_costs_one_cached_lookup_and_no_load():
    gate, loader, registry = make_gate(loaded=[NINE_B])
    gate.ensure_resident(NINE_B)
    assert loader.asked == []
    assert registry.loaded_lookups == 1


def test_every_call_for_a_loaded_model_is_one_lookup_and_never_a_load():
    gate, loader, registry = make_gate(loaded=[NINE_B])
    for _ in range(5):
        gate.ensure_resident(NINE_B)
    assert loader.asked == []
    assert registry.loaded_lookups == 5


# ─────────────────────────── a model that is missing ───────────────────────────


def test_a_missing_model_is_loaded_once_and_found_on_the_next_call():
    gate, loader, _ = make_gate()
    gate.ensure_resident(NINE_B)
    gate.ensure_resident(NINE_B)
    assert loader.asked == [NINE_B]


def test_a_different_model_loaded_is_not_enough():
    gate, loader, _ = make_gate(loaded=[NINE_B])
    gate.ensure_resident(FOURTEEN_B)
    assert loader.asked == [FOURTEEN_B]


def test_a_refused_load_raises_with_the_loaders_own_reason():
    reason = ("qwen/qwen3-14b needs about 14227 MiB but only 9000 MiB would be free "
              "with nothing else loaded")
    gate, loader, _ = make_gate(succeeds=False, last_error=reason)
    with pytest.raises(LocalLLMError) as raised:
        gate.ensure_resident(FOURTEEN_B)
    assert reason in str(raised.value)
    assert "nothing was sent" in str(raised.value)
    assert loader.asked == [FOURTEEN_B]


def test_a_refusal_is_not_remembered_so_the_next_call_asks_the_loader_again():
    # A card that was full a minute ago may have room now. Remembering "no" would turn
    # one bad moment into a permanent refusal.
    gate, loader, _ = make_gate(succeeds=False, last_error="no room")
    for _ in range(2):
        with pytest.raises(LocalLLMError):
            gate.ensure_resident(FOURTEEN_B)
    assert loader.asked == [FOURTEEN_B, FOURTEEN_B]


def test_a_refusal_that_gives_no_reason_still_raises():
    gate, _, _ = make_gate(succeeds=False, last_error="")
    with pytest.raises(LocalLLMError, match="the loader gave no reason"):
        gate.ensure_resident(FOURTEEN_B)


def test_a_loader_that_raises_is_reported_as_a_refusal_and_keeps_its_cause():
    boom = RuntimeError("lms fell over")
    gate, _, _ = make_gate(raises=boom)
    with pytest.raises(LocalLLMError, match="RuntimeError: lms fell over") as raised:
        gate.ensure_resident(FOURTEEN_B)
    assert raised.value.__cause__ is boom


def test_an_unreadable_registry_is_never_taken_as_a_loaded_model():
    # The safe reading of "cannot tell" is "not loaded": the loader is asked, and it reads
    # the registry for itself. Taking it as loaded would wave a request through to a
    # server that may have nothing loaded, which is the unguarded load.
    registry = FakeRegistry(raises=True)
    loader = FakeLoader(registry)
    ResidencyGate(loader, registry).ensure_resident(NINE_B)
    assert loader.asked == [NINE_B]


# ─────────────────────────── two callers at once ───────────────────────────


def test_two_callers_that_both_find_the_model_missing_load_it_once():
    log: list[tuple[str, str]] = []

    class LoggingRegistry(FakeRegistry):
        def loaded(self):
            log.append(("lookup", threading.current_thread().name))
            return super().loaded()

    registry = LoggingRegistry()
    loader = FakeLoader(registry, hold=True, log=log)
    gate = ResidencyGate(loader, registry)
    errors: list[Exception] = []

    def call() -> None:
        try:
            gate.ensure_resident(NINE_B)
        except Exception as exc:  # noqa: BLE001 - reported below, never lost in a thread
            errors.append(exc)

    first = threading.Thread(target=call, name="first")
    second = threading.Thread(target=call, name="second")

    first.start()
    assert loader.inside.wait(timeout=5), "the first caller never reached the load"
    second.start()
    # Time for the second caller to arrive at the gate. A gate without the lock would let
    # it look at the registry now, see nothing loaded, and start a second load while the
    # first is still going. With the lock it can only wait.
    time.sleep(0.3)
    loader.release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert loader.asked == [NINE_B]
    # The second caller looked again only after the first load had finished.
    assert log == [
        ("lookup", "first"), ("load-start", "first"), ("load-done", "first"),
        ("lookup", "second"),
    ]


# ─────────────────────── with the real registry cache in the middle ───────────────────────


class FakeLms:
    """Behaves like the `lms` command for the four sub-commands the toolkit runs.

    Installed in place of `subprocess.run`, the operating-system call that would start the
    real program, so the real runner, registry, budget, loader and gate can all be used
    together and nothing is ever started. `load` takes a moment, as a real load does.
    """

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.commands: list[list[str]] = []

    def run(self, argv, **_options):
        args = [str(part) for part in argv[1:]]
        self.commands.append(args)
        output = ""
        if args[0] == "ps":
            output = json.dumps([
                {"modelKey": key, "identifier": key, "sizeBytes": 7_031_000_000,
                 "contextLength": 32768, "status": "idle"}
                for key in self.loaded
            ])
        elif args[0] == "ls":
            output = json.dumps([
                {"modelKey": NINE_B, "type": "llm", "sizeBytes": 7_031_000_000,
                 "architecture": "qwen3"}
            ])
        elif args[0] == "load":
            time.sleep(0.5)
            self.loaded = [args[1]]
        elif args[0] == "unload":
            self.loaded = []
        return types.SimpleNamespace(returncode=0, stdout=output, stderr="")


class RoomyGpu:
    """A card with plenty of free memory, so the loader's fit and headroom checks pass."""

    def read(self) -> GpuInfo:
        return GpuInfo(available=True, name="fake", total_mib=16303, used_mib=1800)


def test_the_real_registry_cache_does_not_hide_a_finished_load_from_the_second_caller(
        tmp_path, monkeypatch):
    # The registry keeps each `lms ps` answer for 3 seconds. The first caller's answer from
    # before its load is still cached while the load runs, so the loader has to wait for a
    # fresh one; this proves the second caller then sees the model, and that the one load
    # it makes carries the context length. It takes about 3.5 seconds because the loader's
    # waits are real: they are what the cache is exercised against.
    fake = FakeLms()
    monkeypatch.setattr(subprocess, "run", fake.run)
    lms_path = tmp_path / "lms.exe"
    lms_path.write_text("")                    # it only has to exist
    settings = Settings(lms_path=lms_path, data_dir=tmp_path, _env_file=None)
    runner = LmsCommandRunner(settings)
    registry = ModelRegistry(runner)
    loader = ModelLoader(runner, registry, VramBudget(RoomyGpu()),
                         context_length=settings.load_context_length)
    gate = ResidencyGate(loader, registry)
    errors: list[Exception] = []

    def call() -> None:
        try:
            gate.ensure_resident(NINE_B)
        except Exception as exc:  # noqa: BLE001 - reported below, never lost in a thread
            errors.append(exc)

    callers = [threading.Thread(target=call, name=name) for name in ("first", "second")]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=60)

    assert errors == []
    loads = [command for command in fake.commands if command[0] == "load"]
    assert len(loads) == 1
    assert loads[0][loads[0].index("--context-length") + 1] == "32768"
    # A later call finds the model and loads nothing.
    gate.ensure_resident(NINE_B)
    assert len([command for command in fake.commands if command[0] == "load"]) == 1
