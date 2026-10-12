"""The client asks the gate before it sends, and sends nothing when the gate refuses.

The client under test is the real `LocalLLMClient` with one change: `_stream`, the step
that opens a connection to the model server, is replaced by a fake that only records the
model name it was asked to send. So every other line of `complete()` runs for real, and no
network, `lms` or GPU is involved.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fakes import FakeLoader, FakeRegistry, MemoryLiveStore, MemoryRepository
from pydantic import BaseModel

from local_llm.client import LocalLLMClient, LocalLLMError, ModelGate
from local_llm.config import Settings
from local_llm.gate import ResidencyGate

NINE_B = "qwen/qwen3.5-9b"
FOURTEEN_B = "qwen/qwen3-14b"
MESSAGES = [{"role": "user", "content": "Reply with the word ok."}]


class Label(BaseModel):
    label: str


class StubStreamClient(LocalLLMClient):
    """The real client with only the network step replaced.

    The fake "server" answers every request with `{"label": "ok"}`, which is valid as a
    plain-text answer and as a `Label`, so one fake serves both kinds of call.
    """

    def __init__(self, *, events: list[tuple[str, str]] | None = None,
                 gate: ModelGate | None = None, default_model: str = FOURTEEN_B) -> None:
        self.repository = MemoryRepository()
        self.live = MemoryLiveStore()
        # _env_file=None so a developer's real .env cannot change what a test sees.
        settings = Settings(model=default_model, _env_file=None)
        super().__init__(settings, self.repository, self.live, gate=gate)
        self.sent: list[str] = []
        self.events = events if events is not None else []

    async def _stream(self, body, accumulator, reporter, elapsed_ms) -> None:
        self.sent.append(body["model"])
        self.events.append(("send", body["model"]))
        accumulator.consume({"choices": [{"delta": {"content": '{"label": "ok"}'}}]})


class RecordingGate(ModelGate):
    """A gate that notes when and on which thread it was asked, and can refuse or dawdle."""

    def __init__(self, events: list[tuple[str, str]], *, refuse_with: str | None = None,
                 delay_s: float = 0.0) -> None:
        self._events = events
        self._refuse_with = refuse_with
        self._delay_s = delay_s
        self.thread_ids: list[int] = []

    def ensure_resident(self, model: str) -> None:
        self.thread_ids.append(threading.get_ident())
        self._events.append(("gate", model))
        if self._delay_s:
            time.sleep(self._delay_s)
        if self._refuse_with is not None:
            raise LocalLLMError(self._refuse_with)


def make_client(*, refuse_with: str | None = None, delay_s: float = 0.0,
                default_model: str = FOURTEEN_B):
    events: list[tuple[str, str]] = []
    gate = RecordingGate(events, refuse_with=refuse_with, delay_s=delay_s)
    client = StubStreamClient(events=events, gate=gate, default_model=default_model)
    return client, gate, events


# ─────────────────────────── order and coverage ───────────────────────────


def test_the_gate_runs_before_the_request_and_is_given_the_model_being_sent():
    client, _, events = make_client()
    asyncio.run(client.complete(MESSAGES, model=NINE_B))
    assert events == [("gate", NINE_B), ("send", NINE_B)]


def test_the_default_model_is_gated_too_when_none_is_named():
    client, _, events = make_client(default_model=FOURTEEN_B)
    asyncio.run(client.complete(MESSAGES))
    assert events == [("gate", FOURTEEN_B), ("send", FOURTEEN_B)]


def test_plain_and_structured_calls_are_both_gated():
    client, _, events = make_client()

    async def both():
        plain = await client.complete(MESSAGES, model=NINE_B)
        structured = await client.complete(MESSAGES, schema=Label, model=NINE_B)
        return plain, structured

    plain, structured = asyncio.run(both())
    assert plain == '{"label": "ok"}'
    assert structured == Label(label="ok")
    assert events == [("gate", NINE_B), ("send", NINE_B)] * 2


# ─────────────────────────── a refusal ───────────────────────────


def test_a_refusing_gate_sends_nothing_and_the_error_reaches_the_caller():
    reason = "qwen/qwen3-14b is not loaded and could not be loaded safely, so nothing was sent"
    client, _, events = make_client(refuse_with=reason)
    with pytest.raises(LocalLLMError, match="could not be loaded safely"):
        asyncio.run(client.complete(MESSAGES, model=FOURTEEN_B))
    assert client.sent == []
    assert events == [("gate", FOURTEEN_B)]


def test_a_refused_call_still_leaves_an_error_row_and_clears_its_live_entry():
    client, _, _ = make_client(refuse_with="no room on the card")
    with pytest.raises(LocalLLMError):
        asyncio.run(client.complete(MESSAGES, model=FOURTEEN_B, tool="mcp_complete"))
    (started,) = client.repository.started
    (finished,) = client.repository.finished
    assert started["tool"] == "mcp_complete" and started["model"] == FOURTEEN_B
    assert finished["id"] == started["id"]
    assert finished["status"] == "error" and finished["error"] == "no room on the card"
    assert client.live.cleared == [started["id"]]


# ─────────────────────────── threads and timing ───────────────────────────


def test_the_gate_runs_on_a_helper_thread_so_the_event_loop_is_not_frozen():
    client, gate, _ = make_client()

    async def run() -> int:
        await client.complete(MESSAGES, model=NINE_B)
        return threading.get_ident()

    loop_thread = asyncio.run(run())
    assert gate.thread_ids and gate.thread_ids[0] != loop_thread


def test_the_event_loop_keeps_running_while_the_gate_is_busy():
    client, _, _ = make_client(delay_s=0.4)

    async def run() -> int:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        watcher = asyncio.create_task(ticker())
        await client.complete(MESSAGES, model=NINE_B)
        watcher.cancel()
        return ticks

    # A gate that blocked the loop for its 0.4 seconds would leave the ticker at zero or
    # one. Run on a helper thread, it gets about twenty turns.
    assert asyncio.run(run()) >= 5


def test_time_spent_in_the_gate_is_not_counted_in_the_calls_duration():
    # A load takes 18 seconds or more. Counted as part of the call, the first call of every
    # shift would record a duration and time to first token that say nothing about how
    # fast the model generates.
    client, _, _ = make_client(delay_s=0.4)
    asyncio.run(client.complete(MESSAGES, model=NINE_B))
    (finished,) = client.repository.finished
    assert finished["status"] == "ok"
    assert finished["duration_ms"] < 300


# ─────────────────────────── no gate, and two callers ───────────────────────────


def test_a_client_built_without_a_gate_sends_as_it_always_did():
    client = StubStreamClient(gate=None)
    asyncio.run(client.complete(MESSAGES, model=NINE_B))
    assert client.sent == [NINE_B]
    assert client.gate is None


def test_two_concurrent_calls_for_a_missing_model_load_it_once():
    registry = FakeRegistry()
    loader = FakeLoader(registry, hold=True)
    client = StubStreamClient(gate=ResidencyGate(loader, registry))

    async def run() -> None:
        first = asyncio.create_task(client.complete(MESSAGES, model=NINE_B))
        second = asyncio.create_task(client.complete(MESSAGES, model=NINE_B))
        # Long enough for both calls to reach the gate. Without the gate's lock, the second
        # would reach the loader too, while the first load is still held open.
        await asyncio.sleep(0.3)
        loader.release.set()
        await asyncio.gather(first, second)

    asyncio.run(run())
    assert loader.asked == [NINE_B]
    assert client.sent == [NINE_B, NINE_B]


def test_a_real_gate_that_cannot_load_stops_the_request_and_names_the_reason():
    registry = FakeRegistry()
    reason = "qwen/qwen3.5-9b loaded but left only 170 MiB free; unloaded it"
    loader = FakeLoader(registry, succeeds=False, last_error=reason)
    client = StubStreamClient(gate=ResidencyGate(loader, registry))
    with pytest.raises(LocalLLMError) as raised:
        asyncio.run(client.complete(MESSAGES, model=NINE_B))
    assert reason in str(raised.value)
    assert client.sent == []
