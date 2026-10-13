"""What the loader decides before a load and after one.

Before: a model is refused without loading only when its weights alone, plus a small
margin, do not fit. After: the model is kept when it has one prediction slot, the card is
not truly exhausted and a short timed answer is fast enough; otherwise it is unloaded and
the reason carries the measured numbers.

Everything is a fake. `lms` is never started (the loader's `_run` is replaced by one that
records the command), the card is a list of numbers, the speed is a number the test picks
and the waiting function returns at once.
"""

from __future__ import annotations

from fakes import FakeRegistry, installed_entry, loaded_model

from local_llm.config import Settings
from local_llm.loader import LoadReport, ModelLoader, VramBudget
from local_llm.monitor import GpuInfo
from local_llm.speed import CompletionSpeedProbe, SpeedReading

NINE_B = "qwen/qwen3.5-9b"
FOURTEEN_B = "qwen/qwen3-14b"
THIRTY_B = "qwen/qwen3-30b-a3b"
CARD_MIB = 16303


class FakeCard:
    """A graphics card whose free memory is scripted: each reading is the next number in
    the list, and the last one repeats. One number means a card that never changes."""

    def __init__(self, free_readings: list[int]) -> None:
        self._free = list(free_readings)

    def read(self) -> GpuInfo:
        free = self._free.pop(0) if len(self._free) > 1 else self._free[0]
        return GpuInfo(available=True, name="fake", total_mib=CARD_MIB,
                       used_mib=CARD_MIB - free)


class FakeSpeed:
    """Stands in for the speed probe: answers with the reading the test chose and records
    which models it was asked to time."""

    def __init__(self, tokens_per_second: float = 96.4, problem: str = "") -> None:
        self._reading = SpeedReading(
            tokens_per_second=0.0 if problem else tokens_per_second,
            tokens=0 if problem else 64, seconds=1.0, problem=problem,
        )
        self.asked: list[str] = []

    def measure(self, model_key: str) -> SpeedReading:
        self.asked.append(model_key)
        return self._reading


class RecordingLoader(ModelLoader):
    """The real loader with its one way of starting `lms` replaced, and nothing else.

    A load or unload updates the fake registry the way the real command would, so the
    loader's own "is it resident" checks see the result at once.
    """

    def __init__(self, registry: FakeRegistry, **options) -> None:
        super().__init__(runner=None, registry=registry, sleep=lambda seconds: None,
                         **options)
        self.commands: list[list[str]] = []

    def _run(self, args: list[str]) -> bool:
        self.commands.append(list(args))
        if args[0] == "load":
            self._registry.mark_loaded(args[1])
        elif args[0] == "unload":
            self._registry.clear()
        return True

    def _settle(self) -> None:
        return None

    def count(self, verb: str) -> int:
        return len([command for command in self.commands if command[0] == verb])


def make_loader(*, free: list[int], speed: FakeSpeed | None = None,
                registry: FakeRegistry | None = None, **options) -> RecordingLoader:
    registry = registry or FakeRegistry(load_context=243968)
    registry._installed = [
        installed_entry(NINE_B, size_mib=6245),
        installed_entry(FOURTEEN_B, size_mib=8584),
        installed_entry(THIRTY_B, size_mib=19072),
    ]
    return RecordingLoader(registry, budget=VramBudget(FakeCard(free)),
                           speed_probe=speed, **options)


# ─────────────────────────── before the load ───────────────────────────


def test_the_fit_check_counts_the_weights_and_a_small_margin_only():
    budget = VramBudget(FakeCard([13796]))
    assert budget.fits(8584)                 # 8584 + 512 = 9096 fits in 13796
    assert not budget.fits(19072)            # the weights alone are larger than the card
    assert budget.fits(13284)                # exactly weights + margin
    assert not budget.fits(13285)


def test_the_14b_is_no_longer_refused_on_an_estimate_for_a_context_the_app_ignores():
    # 13 October 2026: 13,796 MiB free, and the 14B was refused as needing "about 14249".
    speed = FakeSpeed(61.0)
    loader = make_loader(free=[13796, 13796, 500], speed=speed)
    assert loader.ensure(FOURTEEN_B)
    assert loader.count("load") == 1 and speed.asked == [FOURTEEN_B]


def test_weights_too_large_are_refused_before_any_load_or_unload():
    speed = FakeSpeed()
    loader = make_loader(free=[14900], speed=speed)
    assert not loader.ensure(THIRTY_B)
    assert loader.commands == []             # nothing unloaded, nothing loaded
    assert speed.asked == []
    assert "19072 MiB of weights" in loader.last_error
    assert "14900 MiB" in loader.last_error and "not loaded" in loader.last_error
    assert loader.last_load is None


def test_a_refusal_before_loading_keeps_the_model_that_is_resident():
    registry = FakeRegistry(loaded=[loaded_model(NINE_B, size_mib=6245)])
    loader = make_loader(free=[600], registry=registry)
    assert not loader.ensure(THIRTY_B)
    assert loader.commands == []
    assert [model.key for model in registry.loaded()] == [NINE_B]


# ─────────────────────────── after the load ───────────────────────────


def test_a_fast_model_is_kept_although_little_memory_is_free():
    # The starter's own run on 13 October 2026 left 148 MiB free and was unloaded for it.
    speed = FakeSpeed(96.4)
    loader = make_loader(free=[14900, 14900, 148], speed=speed)
    assert loader.ensure(NINE_B)
    assert loader.last_error == ""
    assert loader.count("unload") == 1       # only the clearing unload before the load
    assert [model.key for model in loader.resident()] == [NINE_B]
    assert loader.last_load == LoadReport(
        model=NINE_B, context_asked=32768, context_loaded=243968, parallel_loaded=1,
        tokens_per_second=96.4, vram_free_mib=148,
    )


def test_a_slow_model_is_unloaded_and_the_reason_says_its_speed():
    loader = make_loader(free=[14900, 14900, 190], speed=FakeSpeed(6.5))
    assert not loader.ensure(NINE_B)
    assert "answered at 6.5 tokens a second" in loader.last_error
    assert "under the 20 floor" in loader.last_error
    assert "190 MiB" in loader.last_error and "243968" in loader.last_error
    assert loader.resident() == []           # unloaded again
    assert loader.last_load is not None and loader.last_load.tokens_per_second == 6.5


def test_the_speed_floor_comes_from_the_loader_s_settings():
    kept = make_loader(free=[14900], speed=FakeSpeed(12.0), min_tokens_per_second=10.0)
    assert kept.ensure(NINE_B)
    refused = make_loader(free=[14900], speed=FakeSpeed(12.0), min_tokens_per_second=30.0)
    assert not refused.ensure(NINE_B)
    assert "under the 30 floor" in refused.last_error


def test_a_speed_exactly_at_the_floor_is_kept():
    assert make_loader(free=[14900], speed=FakeSpeed(20.0)).ensure(NINE_B)


def test_a_model_that_does_not_answer_the_speed_check_is_unloaded():
    loader = make_loader(free=[14900],
                         speed=FakeSpeed(problem="no answer within 60 seconds"))
    assert not loader.ensure(NINE_B)
    assert "no answer within 60 seconds" in loader.last_error
    assert loader.resident() == []
    assert loader.last_load is not None and loader.last_load.tokens_per_second is None


def test_a_load_that_comes_back_with_four_slots_is_unloaded_without_being_timed():
    speed = FakeSpeed(96.4)
    registry = FakeRegistry(load_context=234752, load_parallel=4)
    loader = make_loader(free=[14900], speed=speed, registry=registry)
    assert not loader.ensure(NINE_B)
    assert "4 prediction slots" in loader.last_error and "1 was asked for" in loader.last_error
    assert loader.resident() == []
    assert speed.asked == []                 # wrong on its face, so not asked to generate
    assert loader.last_load is not None and loader.last_load.parallel_loaded == 4


def test_a_slot_count_that_cannot_be_read_does_not_fail_the_load():
    registry = FakeRegistry(load_parallel=0)   # an `lms` that does not report the field
    loader = make_loader(free=[14900], speed=FakeSpeed(96.4), registry=registry)
    assert loader.ensure(NINE_B)
    assert loader.last_load is not None and loader.last_load.parallel_loaded is None


def test_a_card_that_is_truly_exhausted_is_unloaded_whatever_the_speed():
    speed = FakeSpeed(96.4)
    loader = make_loader(free=[14900, 14900, 40], speed=speed)
    assert not loader.ensure(NINE_B)
    assert "only 40 MiB" in loader.last_error and "100 MiB floor" in loader.last_error
    assert loader.resident() == [] and speed.asked == []


def test_free_memory_that_recovers_within_the_wait_passes_the_floor():
    # A swap has both models resident for a moment; the reading comes back a beat later.
    loader = make_loader(free=[14900, 14900, 40, 40, 700], speed=FakeSpeed(96.4))
    assert loader.ensure(NINE_B)


def test_the_memory_floor_comes_from_the_loader_s_settings():
    loader = make_loader(free=[14900, 14900, 250], speed=FakeSpeed(96.4),
                         min_free_vram_mib=300)
    assert not loader.ensure(NINE_B)
    assert "300 MiB floor" in loader.last_error


def test_without_a_speed_probe_the_load_is_kept_and_the_speed_is_unknown():
    loader = make_loader(free=[14900, 14900, 600])
    assert loader.ensure(NINE_B)
    assert loader.last_load is not None and loader.last_load.tokens_per_second is None
    assert loader.last_load.context_loaded == 243968


# ─────────────────────────── a model that is already loaded ───────────────────────────


def test_a_model_already_loaded_with_one_slot_is_reported_and_left_alone():
    registry = FakeRegistry(loaded=[loaded_model(NINE_B, context_length=243968)])
    speed = FakeSpeed()
    loader = make_loader(free=[700], speed=speed, registry=registry)
    assert loader.ensure(NINE_B)
    assert loader.commands == [] and speed.asked == []
    assert loader.last_load == LoadReport(
        model=NINE_B, context_asked=32768, context_loaded=243968, parallel_loaded=1,
        tokens_per_second=None, vram_free_mib=700,
    )


def test_a_model_the_server_loaded_by_itself_with_four_slots_is_loaded_again():
    registry = FakeRegistry(
        loaded=[loaded_model(NINE_B, size_mib=6245, context_length=234752, parallel=4)],
        load_context=243968,
    )
    speed = FakeSpeed(96.4)
    loader = make_loader(free=[190, 14900, 14900, 700], speed=speed, registry=registry)
    assert loader.ensure(NINE_B)
    assert loader.commands[0] == ["unload", NINE_B]
    assert loader.count("load") == 1
    assert loader.last_load is not None and loader.last_load.parallel_loaded == 1
    assert speed.asked == [NINE_B]


# ─────────────────────────── the real speed probe, with a fake server ───────────────────


class ScriptedServer:
    """Stands in for the HTTP call and the clock together: each request takes the next
    number of seconds from `durations` and answers with the next count from `tokens`."""

    def __init__(self, durations: list[float], tokens: list[int],
                 raises: Exception | None = None) -> None:
        self._durations = list(durations)
        self._tokens = list(tokens)
        self._raises = raises
        self.now = 500.0
        self.requests: list[dict] = []

    def monotonic(self) -> float:
        return self.now

    def post(self, url: str, body: dict, headers: dict, timeout_s: float) -> dict:
        self.requests.append({"url": url, "body": body, "timeout_s": timeout_s})
        self.now += self._durations.pop(0)
        if self._raises is not None:
            raise self._raises
        return {"usage": {"completion_tokens": self._tokens.pop(0)}}


def make_probe(server: ScriptedServer) -> CompletionSpeedProbe:
    settings = Settings(url="http://127.0.0.1:1234/v1", _env_file=None)
    return CompletionSpeedProbe(settings, post=server.post, monotonic=server.monotonic)


def test_the_probe_judges_the_second_call_so_the_warm_up_does_not_decide():
    # Measured: the first call after a load is the slow one.
    server = ScriptedServer(durations=[4.0, 0.64], tokens=[64, 64])
    reading = make_probe(server).measure(NINE_B)
    assert reading.problem == ""
    assert reading.tokens_per_second == 100.0      # 64 / 0.64, not 64 / 4.0
    assert len(server.requests) == 2


def test_the_probe_asks_for_a_short_fixed_completion_at_temperature_zero():
    server = ScriptedServer(durations=[1.0, 1.0], tokens=[64, 64])
    make_probe(server).measure(NINE_B)
    first, second = (request["body"] for request in server.requests)
    assert first == second
    assert first["model"] == NINE_B and first["max_tokens"] == 64
    assert first["temperature"] == 0 and first["stream"] is False
    assert server.requests[0]["url"] == "http://127.0.0.1:1234/v1/chat/completions"


def test_an_answer_too_short_to_time_is_a_problem_not_a_speed():
    server = ScriptedServer(durations=[1.0, 0.15], tokens=[64, 3])
    reading = make_probe(server).measure(NINE_B)
    assert "only 3 tokens" in reading.problem and reading.tokens_per_second == 0.0


def test_a_failed_request_is_a_problem_and_the_second_call_is_not_made():
    server = ScriptedServer(durations=[0.1], tokens=[], raises=RuntimeError("refused"))
    reading = make_probe(server).measure(NINE_B)
    assert "RuntimeError: refused" in reading.problem
    assert len(server.requests) == 1
