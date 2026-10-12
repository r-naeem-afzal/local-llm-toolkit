"""Making sure a model is loaded, through the guarded loader, before a request names it.

## The fault this closes

LM Studio loads a model **by itself** when a request names one that is not loaded. This is
called just-in-time loading ("JIT"). It is convenient and, on this machine, dangerous,
because the automatic load uses LM Studio's own defaults and none of the checks in
`loader.py`. Two measurements from the night of 7 to 8 October 2026:

* LM Studio's default context for qwen/qwen3.5-9b is 234,752 tokens. Loaded with every
  layer on the GPU at that default, the model fills the 16 GB card. The guarded loader
  caught this and unloaded it again ("loaded but left only 170 MiB free").
* A request then reached the running server while no model was loaded. LM Studio loaded
  the model by itself, with the same defaults and no guard: 15.8 GB used, 190 MiB free,
  and the answer crawled for eight minutes.

So any tool call made while the server was up and the model was not loaded could fill the
card with nobody home. `ResidencyGate` closes that. The client asks it before every
request, and it either confirms the model is loaded, loads it through `ModelLoader` (which
carries every VRAM check and an explicit context length), or refuses. A refused request is
never sent.

## Why one lock

The research pipeline makes calls concurrently. Two calls that both find the model missing
must not both load it: loading a model that is already loading creates a **second
instance** of it, each taking its full share of memory. That fault was measured, with the
pair `qwen/qwen2.5-coder-14b` and `qwen/qwen2.5-coder-14b:2`, and is described in
`loader.py`. So the check and the load happen together under one lock, and a caller that
had to wait for the lock looks again afterwards.

## Known limit

The gate checks that the model is loaded, not how it was loaded. A model that something
else loaded earlier at an oversized context still passes, because it is resident.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

from .client import LocalLLMError, ModelGate

# A Protocol is a structural interface: any class with the listed methods qualifies without
# inheriting from anything. `ModelRegistry` and `ModelLoader` fit these two as they are,
# and a test fake fits them with a few lines. They are kept this narrow so the gate cannot
# be handed, or come to depend on, more of the loader than it uses.


class LoadedModels(Protocol):
    """The one thing the gate needs from the model registry: what is in VRAM now."""

    def loaded(self) -> list[Any]:
        """Every loaded model instance; each has a `.key` such as "qwen/qwen3.5-9b"."""


class GuardedLoader(Protocol):
    """The part of `ModelLoader` the gate uses: load a model safely, and say why not."""

    last_error: str

    def ensure(self, model_key: str) -> bool:
        """True once `model_key` is loaded; False, with `last_error` set, when it is not."""


class ResidencyGate(ModelGate):
    """Loads a missing model through the guarded loader before a request may name it.

    Takes the loader and the registry through the constructor, so a test hands it fakes and
    never needs `lms`, a GPU or a model server.
    """

    def __init__(self, loader: GuardedLoader, registry: LoadedModels) -> None:
        self._loader = loader
        self._registry = registry
        # A lock lets one thread at a time into the block it guards; every other thread
        # waits at the `with` line until the holder leaves. Under the hood it is an
        # operating-system primitive shared by all the threads of this process, which is
        # the right scope: the client calls the gate from helper threads
        # (`asyncio.to_thread`), so an `asyncio.Lock` would not see across them.
        self._lock = threading.Lock()

    def ensure_resident(self, model: str) -> None:
        """Return once `model` is loaded, loading it first if it is not; or raise.

            model loaded                      ->  returns, after one cached registry lookup
            model missing, loader succeeds    ->  returns, the model loaded exactly once
            model missing, loader refuses     ->  LocalLLMError("qwen/qwen3-14b is not
                loaded and could not be loaded safely, so nothing was sent to the model
                server: qwen/qwen3-14b needs about 14227 MiB but only 9000 MiB would be
                free with nothing else loaded")

        The residency check and the load are both inside the lock, so the sequence for two
        callers that arrive together is:

            caller A takes the lock, finds the model missing, loads it, leaves
            caller B was waiting; takes the lock, looks again, finds it loaded, leaves

        The second look is the point. What breaks without it: B, having found the model
        missing a moment earlier, loads it a second time, and the card holds two copies
        of a model that fits once.

        The lock is taken even when the model is already loaded. An uncontended lock costs
        microseconds, and it keeps the rule simple and safe: the check is never separated
        from the load it may lead to. Skipping the lock on the quick path would let a
        caller read "loaded" for a model that another caller is, at that moment, about to
        unload to make room for a different one.
        """
        with self._lock:
            if self._is_resident(model):
                return
            self._load(model)

    # ── internals ──

    def _is_resident(self, model: str) -> bool:
        """Whether the registry shows `model` loaded.

            registry shows ["qwen/qwen3.5-9b"], asked about "qwen/qwen3.5-9b"  ->  True
            registry shows ["qwen/qwen3.5-9b"], asked about "qwen/qwen3-14b"   ->  False
            registry cannot be read                                            ->  False

        Under the hood the registry does not run `lms ps` on every call: the command runner
        keeps each answer for about 3 seconds. So the usual case, a model that is simply
        there, costs a cached lookup and no subprocess, which is all this check is allowed
        to cost. The price of the cache is that a model unloaded by someone else in the
        last 3 seconds can still look loaded. That window is accepted; closing it would
        mean spawning `lms ps` before every request.

        "Cannot be read" counts as "not loaded", and that is a decision. Why: the gate
        exists to stop a request reaching a server that has no model, and a guess in the
        other direction is the unguarded load. What happens as a result: the loader is
        asked, and it reads the registry itself and loads only if the model is truly
        missing. What breaks if it counted as "loaded": a failing `lms` would wave every
        request through to a server with nothing loaded. (In practice
        `ModelRegistry.loaded()` returns an empty list when `lms ps` fails, so an unreadable
        registry already arrives here as "nothing loaded" and takes the same path.)
        """
        try:
            loaded = self._registry.loaded()
        except Exception:  # noqa: BLE001 - whatever went wrong, the answer is "cannot confirm"
            return False
        return any(entry.key == model for entry in loaded)

    def _load(self, model: str) -> None:
        """Load `model` through the guarded loader, or raise `LocalLLMError`.

        The loader's own `last_error` is carried into the exception, because "the model
        does not fit" and "the CLI is missing" want completely different responses from
        whoever reads the message, and a bare "could not load" tells them neither.

        An exception from the loader is turned into the same `LocalLLMError`. `ensure` is
        written to return False rather than raise, so this is a guard against that promise
        being broken one day. What breaks without it: the exception would leave
        `complete()` as some unrelated error type, the callers that catch only
        `LocalLLMError` (the MCP tools, the ranker) would not catch it, and the call's row
        in the history would stay "running" for ever. Nothing is sent in either case,
        because the exception leaves before the request is built.
        """
        try:
            loaded = self._loader.ensure(model)
        except Exception as exc:
            # Deliberately broad (see the docstring): any failure at all becomes a refusal.
            # It is re-raised as `LocalLLMError` below, never swallowed.
            reason = f"the loader failed with {type(exc).__name__}: {exc}"
            raise LocalLLMError(self._refusal(model, reason)) from exc
        if not loaded:
            # Read straight after `ensure`, still inside the lock, so it is the reason for
            # this load and not for another caller's.
            raise LocalLLMError(
                self._refusal(model, self._loader.last_error or "the loader gave no reason")
            )

    @staticmethod
    def _refusal(model: str, reason: str) -> str:
        """The sentence a caller sees when a request is refused.

            ("qwen/qwen3.5-9b", "lms load qwen/qwen3.5-9b failed")
              ->  "qwen/qwen3.5-9b is not loaded and could not be loaded safely, so nothing
                   was sent to the model server: lms load qwen/qwen3.5-9b failed"

        It says what was not done ("nothing was sent") as well as why, because the reader
        is often an unattended caller deciding whether to retry, and "was my request
        half-processed?" is the first thing it needs answered.
        """
        return (
            f"{model} is not loaded and could not be loaded safely, so nothing was sent to "
            f"the model server: {reason}"
        )
