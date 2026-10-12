"""The composition root wires the gate and the context length from the settings.

`Toolkit` is the one place that decides which classes are used, so a setting that is read
nowhere else (`guard_model_loads`, `load_context_length`) is only real if it is read here.
The real `Toolkit` is used with just its two stores replaced, so nothing opens a database,
runs `lms` or touches the GPU: building the objects does none of those.
"""

from __future__ import annotations

from fakes import MemoryLiveStore, MemoryRepository

from local_llm.config import Settings
from local_llm.container import Toolkit
from local_llm.gate import ResidencyGate


class OfflineToolkit(Toolkit):
    """The real composition root, with the repository and live store swapped for memory.

    A class attribute with the same name shadows the base class's `cached_property`
    (computed once, then remembered), so the base never builds the database-backed one.
    """

    repository = MemoryRepository()
    live_store = MemoryLiveStore()


def make_toolkit(tmp_path, **settings) -> OfflineToolkit:
    # _env_file=None so a developer's real .env cannot change what a test sees.
    return OfflineToolkit(Settings(data_dir=tmp_path, _env_file=None, **settings))


def test_the_client_is_gated_by_default(tmp_path):
    toolkit = make_toolkit(tmp_path)
    assert isinstance(toolkit.client.gate, ResidencyGate)
    assert toolkit.client.gate is toolkit.residency_gate


def test_guard_model_loads_off_builds_a_client_with_no_gate(tmp_path):
    toolkit = make_toolkit(tmp_path, guard_model_loads=False)
    assert toolkit.client.gate is None


def test_the_loader_uses_the_default_context_length(tmp_path):
    assert make_toolkit(tmp_path).loader.context_length == 32768


def test_the_loader_uses_the_configured_context_length(tmp_path):
    assert make_toolkit(tmp_path, load_context_length=16384).loader.context_length == 16384


def test_there_is_one_gate_per_toolkit(tmp_path):
    # Its lock is what stops two concurrent calls loading one model twice; a second gate
    # would be a second lock that knows nothing about the first.
    toolkit = make_toolkit(tmp_path)
    assert toolkit.residency_gate is toolkit.residency_gate
