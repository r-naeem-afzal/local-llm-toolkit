"""Fakes for LM Studio's view of the machine, shared by the tests that need one.

Everything here is plain Python: no `lms`, no GPU, no model server, no clock. Each fake is a
few lines so a reader of a test can see exactly what scenario it sets up. This file is not a
test module (pytest only collects `test_*.py`), it is imported by them, which works because
pytest puts the `tests` folder on the import path.
"""

from __future__ import annotations

import threading
from typing import Any

from local_llm.monitor import LoadedModel
from local_llm.store import CallRepository, LiveProgressStore


def loaded_model(key: str, *, size_mib: int = 6706, identifier: str | None = None,
                 context_length: int = 32768, parallel: int = 1) -> LoadedModel:
    """A model as `lms ps` reports it, with the fields the toolkit reads.

        loaded_model("qwen/qwen3.5-9b")
          ->  LoadedModel(key="qwen/qwen3.5-9b", identifier="qwen/qwen3.5-9b", size_mib=6706,
                          context_length=32768, parallel=1, ...)
        loaded_model("qwen/qwen3.5-9b", context_length=243968, parallel=4)
          ->  the same model as the server loads it by itself: its own context, four slots

    The identifier defaults to the key, which is what LM Studio does for the first copy of
    a model. A second copy would be `loaded_model(key, identifier=key + ":2")`.
    """
    return LoadedModel(
        key=key, identifier=identifier or key, display_name=key, size_mib=size_mib,
        context_length=context_length, max_context_length=32768, status="idle",
        quantisation="Q4_K_M", ttl_remaining_s=3600, queued=0, parallel=parallel,
    )


def installed_entry(key: str, *, size_mib: int = 6706, kind: str = "llm",
                    architecture: str = "qwen3") -> dict[str, Any]:
    """A model as `ModelRegistry.installed()` reports it (`lms ls`).

        installed_entry("qwen/qwen3.5-9b")
          ->  {"key": "qwen/qwen3.5-9b", "type": "llm", "size_mib": 6706, ...}
        installed_entry("text-embedding-nomic-embed-text-v1.5", kind="embedding")
          ->  {"key": "text-embedding-nomic-embed-text-v1.5", "type": "embedding", ...}
    """
    return {
        "key": key, "type": kind, "display_name": key, "size_mib": size_mib, "params": "",
        "architecture": architecture, "max_context_length": 32768,
    }


class FakeRegistry:
    """Stands in for `ModelRegistry`: what is loaded now and what is installed.

    Counts how often it is asked, because the gate is allowed exactly one cached lookup
    per request and a test needs to be able to say so.
    """

    def __init__(self, loaded: list[LoadedModel] | None = None,
                 installed: list[dict[str, Any]] | None = None, raises: bool = False,
                 load_context: int = 32768, load_parallel: int = 1) -> None:
        self._loaded = list(loaded or [])
        self._installed = list(installed or [])
        self._raises = raises
        # What the pretend server reports for a model after `mark_loaded`. The defaults
        # are a server that does as it is asked. `load_context=243968` is the real app,
        # which picks its own context; `load_parallel=4` is a load that came back with
        # four prediction slots; 0 stands for an `lms` that does not report the field.
        self._load_context = load_context
        self._load_parallel = load_parallel
        self.loaded_lookups = 0
        self.refreshes = 0

    def refresh(self) -> None:
        """The real registry drops its cached `lms` answers here. The fake has no cache,
        so it only counts, which lets a test say "a fresh look was asked for"."""
        self.refreshes += 1

    def loaded(self) -> list[LoadedModel]:
        self.loaded_lookups += 1
        if self._raises:
            raise RuntimeError("registry exploded")
        return list(self._loaded)

    def installed(self) -> list[dict[str, Any]]:
        return list(self._installed)

    def mark_loaded(self, key: str) -> None:
        """Pretend a load worked: afterwards `key` is the only model resident."""
        self._loaded = [loaded_model(key, context_length=self._load_context,
                                     parallel=self._load_parallel)]

    def clear(self) -> None:
        """Pretend an unload worked: nothing is resident."""
        self._loaded = []


class FakeLoader:
    """Stands in for `ModelLoader` as the gate sees it: `ensure(model)` and `last_error`.

    A load that succeeds updates the fake registry, as a real one would, so the next
    look at the registry finds the model. Every model asked for is recorded in `asked`.

    `hold=True` makes `ensure` wait until the test sets `release`, and sets `inside` the
    moment it has started. That is how a test keeps one load open while a second caller
    arrives, which is the only way to make "two callers at once" a repeatable scenario
    instead of a matter of luck.
    """

    def __init__(self, registry: FakeRegistry, *, succeeds: bool = True, last_error: str = "",
                 raises: Exception | None = None, hold: bool = False,
                 log: list[tuple[str, str]] | None = None) -> None:
        self._registry = registry
        self._succeeds = succeeds
        self._raises = raises
        self._hold = hold
        self._log = log
        self.last_error = last_error
        self.asked: list[str] = []
        self.inside = threading.Event()
        self.release = threading.Event()

    def ensure(self, model_key: str) -> bool:
        self.asked.append(model_key)
        self._note("load-start")
        if self._raises is not None:
            raise self._raises
        if self._hold:
            self.inside.set()
            # A timeout so a test that forgets to release fails instead of hanging.
            self.release.wait(timeout=5)
        if self._succeeds:
            self._registry.mark_loaded(model_key)
        self._note("load-done")
        return self._succeeds

    def _note(self, what: str) -> None:
        if self._log is not None:
            self._log.append((what, threading.current_thread().name))


class MemoryRepository(CallRepository):
    """Keeps every call the client records in lists, so a test can read them back."""

    def __init__(self) -> None:
        self.started: list[dict[str, Any]] = []
        self.finished: list[dict[str, Any]] = []

    def start_call(self, *, ts: str, tool: str, model: str, meta: dict,
                   prompt: list[dict[str, str]], call_id: str | None = None) -> str:
        call_id = call_id or f"call-{len(self.started) + 1}"
        self.started.append({"id": call_id, "tool": tool, "model": model})
        return call_id

    def finish_call(self, call_id: str, *, status: str, duration_ms: int | None = None,
                    usage: dict | None = None, error: str | None = None,
                    response: str | None = None, reasoning: str | None = None,
                    stage: str | None = None, ts: str | None = None,
                    tool: str | None = None, model: str | None = None,
                    meta: dict | None = None) -> None:
        self.finished.append({
            "id": call_id, "status": status, "error": error, "response": response,
            "duration_ms": duration_ms,
        })

    def list_calls(self, **_options: Any) -> list:
        return []

    def get_payload(self, call_id: str) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {}

    def size_bytes(self) -> int:
        return 0


class MemoryLiveStore(LiveProgressStore):
    """Remembers which calls had their live entry cleared; writes nothing anywhere."""

    def __init__(self) -> None:
        self.cleared: list[str] = []

    def write(self, entry: dict[str, Any]) -> None:
        return None

    def clear(self, call_id: str) -> None:
        self.cleared.append(call_id)

    def read_all(self) -> list[dict[str, Any]]:
        return []
