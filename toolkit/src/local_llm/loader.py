"""Making sure the model a stage needs is resident, and that loading it cannot overfill VRAM.

Naming a different model on a request makes the server load it just in time. That is
convenient and, on this machine, dangerous: LM Studio loads what is asked for and leaves
what is already there, so a request that names a second model **adds** it rather than
swapping. On a 16 GB card that holds one 14B at a time, the result is two models sharing
memory that fits one.

That failure has now happened twice, in two different ways, and both were invisible while
they were happening:

* a router sent ranking to a small model while a 14B was resident, which loaded both and
  turned a 12 second extraction into 115 seconds; and
* a single model was loaded **twice**, once by just-in-time loading and once explicitly,
  producing instances `qwen/qwen2.5-coder-14b` and `qwen/qwen2.5-coder-14b:2`. Nothing
  noticed, because both report the same *model key* and only the *instance identifier*
  differs, so any check that deduplicates by key sees exactly one model.

So the rule this module enforces is: **swap deliberately, between stages, or not at all**,
and never load anything without first establishing that it fits.

## The two-part safeguard, and why one part is not enough

`VramBudget` checks before loading, so an impossible load is refused rather than
attempted. It can only reject the impossible: the weights themselves not fitting on the
card. Everything else about a load is decided by the server and can only be read
afterwards. `ModelLoader` therefore also **verifies afterwards**, and unloads the model
again when any of these is false:

* exactly one instance is resident;
* it was loaded with one prediction slot (`parallel` 1), read back from `lms ps`;
* the card is not truly exhausted (a small floor of free memory, 100 MiB by default);
* the model **answers at a usable speed**: one short fixed completion, timed, against a
  floor of 20 tokens a second by default.

The last one is the real test, and it replaced a fixed "at least 300 MiB free" rule. The
next section says why free memory stopped meaning anything.

## The context length that is asked for is not the one that is loaded

Measured on 13 October 2026 (Bionic 1.1.1, engine llama.cpp 2.56.0, one 16 GB card):
`lms load qwen/qwen3.5-9b --context-length 8192`, the same with 16384 and with 32768, all
loaded the model with a context of 243,968 tokens. The app sizes the context to the free
graphics memory by itself (its settings call this `autoFit`), so **every load fills the
card on purpose**: 638 to 908 MiB were left free in five loads by hand, 148 MiB in one
more. In that state the model was fast, not crawling: 109 tokens a second.

Two things follow, and both are in the code below:

1. `--context-length` does **not** bound the memory a load takes on this version. It is
   still sent on every load, from `Settings.load_context_length`, because a later version
   may honour it and a load with no number at all has nothing to honour. What was really
   loaded is read back from `lms ps` and reported (`LoadReport`).
2. Free memory after a load says nothing about whether the load is healthy, because a
   healthy load and an overfilled one both leave the card nearly full. Speed tells them
   apart, so speed is what is measured.

`--parallel 1` **is** honoured, and it matters: the one crawling load on record (8 October
2026) was a load the server made by itself with four slots.

## Why overfilling matters more than it sounds

Asking for more VRAM than the card has does not fail. Windows places the overflow in
shared system memory and the model keeps working, roughly fifteen times slower, with no
error anywhere: `nvidia-smi` reports the same memory used and the same 100% utilisation
either way. The only place the difference shows is in how fast the model answers, which
is why the check after a load asks it a question and times the answer.

## When it is worth swapping at all

A research run extracts from dozens of pages, and extraction measurably yields more
verified claims on the coder model than on the reasoning one at similar latency, so one
load repaid across forty calls is obviously worth it. A single ad-hoc call is not.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from .config import (
    DEFAULT_LOAD_CONTEXT_LENGTH,
    DEFAULT_LOAD_LOCK_WAIT_S,
    DEFAULT_MIN_FREE_VRAM_MIB,
    DEFAULT_MIN_TOKENS_PER_SECOND,
)
from .childenv import lms_environment
from .speed import SpeedProbe


@dataclass(frozen=True)
class LoadReport:
    """What a load really produced, read back from the server after it.

        asked for 32768, the app chose its own context, answered at 96.4 tokens a second
          ->  LoadReport(model="qwen/qwen3.5-9b", context_asked=32768,
                         context_loaded=243968, parallel_loaded=1,
                         tokens_per_second=96.4, vram_free_mib=712)

    Kept because the command that was sent and the load that resulted are not the same
    thing: the app ignores the context length it is given (module docstring). Without
    this record a caller would report the number it asked for as if it were the truth.

    "Frozen" means the fields cannot be changed after creation. `None` in a field means
    that value could not be read (an older `lms` that does not report it, no speed probe
    configured, a card that cannot be read), never zero.
    """

    model: str
    context_asked: int
    context_loaded: int | None
    parallel_loaded: int | None
    tokens_per_second: float | None
    vram_free_mib: int | None


class VramBudget:
    """Answers "will this model fit?" before anything tries to load it.

    Separate from `ModelLoader` because it is a different question. The loader knows how
    to drive the CLI; this knows what the card can hold. Splitting them means the budget
    can be consulted by anything that wants to decide *whether* to swap, the router, the
    dashboard, without that caller also acquiring the ability to load models.
    """

    # Room asked for on top of the weights before a load is allowed to start, in MiB.
    #
    # Why this small: the app fits the context to whatever memory is left (module
    # docstring), so the only load that cannot work is one whose weights do not fit. The
    # margin covers the compute buffers every load needs beside its weights. What happens
    # as a result: a model is refused before loading only when it is plainly too large,
    # and every other question is answered after the load, by measurement.
    #
    # What breaks with a larger figure: the 14B was refused on 13 October 2026 as needing
    # "about 14249 MiB" against 13,796 MiB free. That figure was weights times 1.66, an
    # estimate of the whole footprint, set against a low guess of the free memory. The
    # model was never tried. Loaded for real the same evening it left 907 MiB free and
    # answered at 69.9 tokens a second.
    _WEIGHTS_MARGIN_MIB = 512

    # What a loaded model gives back when it is unloaded, as a multiple of its weights.
    #
    # Used only by `ModelLoader._free_if_emptied`, to guess how much memory an unload will
    # return without unloading. One calibration point:
    #
    #     qwen2.5-coder-14b   weights 8,571 MiB   resident total 14,193 MiB   = 1.66x
    #
    # On this version of the app, which fills the card on every load, a resident model
    # usually returns more than this. Too low is the safe direction: it under-states the
    # free memory, which can only make the check before a load more careful.
    _OVERHEAD_FACTOR = 1.66

    def __init__(self, gpu_probe: Any) -> None:
        # Takes the probe rather than calling nvidia-smi itself, so this shares whatever
        # caching and failure handling the monitor already has, and so a test can hand it
        # a fake card.
        self._gpu_probe = gpu_probe

    def _gpu(self) -> Any | None:
        """The card's current state, or None when it cannot be read.

        `GpuInfo.available` is False when pynvml is missing or the driver call failed. A
        card that cannot be read is treated as absent rather than as full, because every
        caller here interprets None as "do not block on these grounds".
        """
        try:
            info = self._gpu_probe.read()
        except Exception:
            return None
        return info if getattr(info, "available", False) else None

    def total_mib(self) -> int | None:
        """The card's total VRAM, or None when it cannot be read.

        Needed as a ceiling on estimates. Without it, "how much would be free if
        everything were unloaded" can exceed the physical card, which is not an
        arithmetic slip but the signature of the very problem being guarded against: it
        happens exactly when the resident models already total more than the card holds.
        """
        gpu = self._gpu()
        return None if gpu is None else int(gpu.total_mib)

    def free_mib(self) -> int | None:
        """VRAM the card reports as unused, or None when the card cannot be read.

        None is distinct from zero and must stay that way: zero means "no room", while
        None means "unknown", and the safe response to unknown is to carry on rather than
        to refuse every load because a monitoring tool is missing.
        """
        gpu = self._gpu()
        return None if gpu is None else int(gpu.free_mib)

    def required_mib(self, weights_mib: int) -> int:
        """A low guess at what a resident model of this weight gives back when unloaded.

            8,571 MiB of weights  ->  14,227 MiB
                                         (measured once: 14,193 MiB resident)

        This does **not** decide whether a model may load: `fits` does. It is used to
        guess how much memory unloading a resident model will return.
        """
        return int(weights_mib * self._OVERHEAD_FACTOR)

    def weights_margin_mib(self) -> int:
        """The room asked for beside the weights, so an error message can name it."""
        return self._WEIGHTS_MARGIN_MIB

    def fits(self, weights_mib: int, *, already_free_mib: int | None = None) -> bool:
        """Whether the weights alone, plus a small margin, fit in the free memory.

            free 13,796 MiB, weights  8,584  ->  needs  9,096  ->  True
            free 13,796 MiB, weights 19,072  ->  needs 19,584  ->  False
            free  9,000 MiB, weights  8,584  ->  needs  9,096  ->  False

        Deliberately not an estimate of the whole footprint. The context, and with it most
        of the memory beyond the weights, is chosen by the app at load time to fit what is
        free, so there is nothing to estimate in advance. True here means "worth trying",
        not "will run well": `ModelLoader` measures that after the load.

        Returns True when the card cannot be read at all. Refusing every load because
        `nvidia-smi` is missing would break a working setup to protect against a problem
        that might not exist.
        """
        free = already_free_mib if already_free_mib is not None else self.free_mib()
        if free is None:
            return True
        return weights_mib + self._WEIGHTS_MARGIN_MIB <= free


class OffloadPlanner:
    """Works out how much of a too-large model to keep on the card.

    Separate from both `VramBudget` (which answers "does it fit") and `ModelLoader` (which
    drives the CLI) because this is a third question, "if it does not fit, what share
    should go on the GPU?", and it is a *policy* answer, tuned by measurement, while the
    other two are facts.

    ## The arithmetic, and why the reserve is a guess

    VRAM has to hold three things: the weights placed on the card, the KV cache, and the
    compute buffers. Only the first is being divided here, so the other two have to be
    reserved before the division:

        ratio = (free_when_empty - reserve) / weights

        free 15,900 MiB, reserve 4,000, weights 19,072  ->  (15900-4000)/19072 = 0.62

    The reserve is genuinely an estimate. The KV cache depends on the model's attention
    geometry and the context length the server chooses, neither of which `lms` reports
    before loading. Qwen3-30B-A3B is unusually cheap here: 48 layers with 4 key/value
    heads works out near 96 KiB per token, about 3 GiB at a 32K context, *less* than the
    14B it replaces despite being twice the size, but that is a per-model fact and this
    class cannot know it.

    So the number below is a deliberately generous starting point, and the honest workflow
    is: plan a ratio, load, measure, adjust. `ModelLoader.ensure` times a short answer
    afterwards and unloads a model that is too slow, so a wrong guess costs a failed load
    rather than a silently crawling one.

    What breaks without the reserve: the ratio comes out near 1.0, every layer is placed on
    a card with no room left for the cache, and the driver quietly moves the overflow into
    system memory. That state does not raise an error and does not look different in
    `nvidia-smi`: it just runs about fifteen times slower, which is the failure this whole
    module exists to prevent.
    """

    #: VRAM held back for the KV cache and compute buffers, in MiB.
    #:
    #: Chosen as roughly a quarter of a 16 GB card. Calibration point: a 14B at a 30K
    #: context needed about 4.7 GiB for the cache alone, so this is on the low side for a
    #: dense model with many key/value heads and comfortable for a grouped-query
    #: mixture-of-experts model. Raise it if a load succeeds and then runs slowly.
    _RESERVE_MIB = 4000

    def __init__(self, budget: VramBudget) -> None:
        self._budget = budget

    def ratio_for(self, weights_mib: int, *, reserve_mib: int | None = None,
                  free_mib: int | None = None) -> float | None:
        """The GPU share to request, or None when the whole model should fit anyway.

        Returning None rather than 1.0 for the fits-entirely case is deliberate: None is
        what `ModelLoader.ensure` interprets as "keep the normal whole-model path,
        including its pre-load fit check". Passing an explicit 1.0 would skip that check
        for no reason.
        """
        if not weights_mib:
            return None
        if self._budget.fits(weights_mib, already_free_mib=free_mib):
            return None

        available = free_mib if free_mib is not None else self._budget.total_mib()
        if available is None:
            # The card cannot be read. Refuse to invent a ratio: a wrong one is worse
            # than none, because it would place layers on a card whose size is unknown.
            return None

        reserve = self._RESERVE_MIB if reserve_mib is None else reserve_mib
        usable = available - reserve
        if usable <= 0:
            return 0.0
        return min(usable / weights_mib, 1.0)


class ModelLoader:
    """Ensures a named model is the one and only resident model, using the `lms` CLI.

    Takes the runner, registry and budget rather than reaching for them, so a test can
    drive this with fakes and so it shares the registry's short cache instead of spawning
    its own subprocesses.
    """

    # How long to wait for a load before giving up. A cold 14B takes about 18 seconds
    # here; a slow disk or a model being read for the first time can take longer, and
    # failing early would turn a slow load into a spurious error.
    _LOAD_TIMEOUT_S = 300

    # How long to wait for VRAM to actually be released after an unload. The unload
    # command returns as soon as the request is *accepted*, not when the memory is free,
    # and issuing the next load into memory that has not been released yet re-creates the
    # overlap this class exists to prevent.
    _UNLOAD_SETTLE_S = 15

    # How long the card is given to show its floor of free memory after a load. A swap has
    # two models resident for a moment, and the released memory comes back a beat later,
    # so anything shorter reports an exhausted card that is not one.
    _FREE_FLOOR_SETTLE_S = 10

    # The number of prediction slots every load asks for, and the number it must come back
    # with. A slot is one conversation the server can generate for at a time. One, because
    # there is one pipeline and one card, and each extra slot keeps its own share of the
    # attention cache. The one crawling load on record (8 October 2026) had four.
    _PARALLEL = 1

    # Options every load carries.
    #
    # `--gpu max` asks for every layer on the card. On its own it has not been shown to
    # change throughput, but it is what is wanted whenever the model fits, and the
    # alternative is letting the server decide silently.
    _LOAD_OPTIONS = ("--gpu", "max", "--parallel", str(_PARALLEL))

    def _load_flags(self, gpu_ratio: float | None) -> list[str]:
        """The `lms load` flags for this load: GPU share from the caller, context from us.

            gpu_ratio=None  ->  ["--gpu", "max",  "--parallel", "1", "--context-length", "32768"]
                                 (every layer on the GPU)
            gpu_ratio=0.6   ->  ["--gpu", "0.60", "--parallel", "1", "--context-length", "32768"]
                                 (60% of the layers on the GPU)

        32768 is the default `Settings.load_context_length`.

        A ratio below 1 means part of the model is read from system RAM over PCIe during
        generation. For a **dense** model that is close to useless: every layer is needed
        for every token, so the slowest link sets the pace. It exists for **mixture-of-
        experts** models, where only a small share of the weights participates in any one
        token, and the idle remainder can sit in RAM costing nothing but address space.

        The context flag is added on **both** paths. `--context-length` is how many tokens
        of conversation the loaded model should hold. On the app version measured on
        13 October 2026 the number is accepted and then ignored: the app sizes the context
        to the free graphics memory by itself (module docstring). It is sent anyway, so
        that a version that does honour it gets a sensible number without a code change,
        and `_read_back` reports what was really loaded so nobody has to take the flag on
        trust. What breaks if a reader assumes the flag works: they conclude a load is
        bounded at 32,768 tokens of cache when it has in fact filled the card.
        """
        context = ["--context-length", str(self._context_length)]
        if gpu_ratio is None:
            return [*self._LOAD_OPTIONS, *context]
        # Clamped because `lms` rejects anything outside 0-1 and a caller computing a
        # ratio from measured VRAM can land marginally outside by rounding, which would
        # fail the load for a reason that has nothing to do with the model.
        ratio = min(max(gpu_ratio, 0.0), 1.0)
        return ["--gpu", f"{ratio:.2f}", "--parallel", str(self._PARALLEL), *context]

    def __init__(self, runner: Any, registry: Any, budget: VramBudget | None = None,
                 context_length: int = DEFAULT_LOAD_CONTEXT_LENGTH,
                 speed_probe: SpeedProbe | None = None,
                 min_tokens_per_second: float = DEFAULT_MIN_TOKENS_PER_SECOND,
                 min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
                 sleep: Callable[[float], None] = time.sleep,
                 load_lock: Any = None,
                 load_lock_wait_s: float = DEFAULT_LOAD_LOCK_WAIT_S) -> None:
        self._runner = runner
        self._registry = registry
        # The lock shared with every other process that may load a model (lockfile.py):
        # needs wait_for(seconds, note) -> bool, release() and describe_holder() -> str.
        # Passed in, so a test uses a temporary folder or a fake. Optional like the
        # budget: a loader built by hand for a one-off script works without it and is
        # then only safe while nothing else loads models.
        self._load_lock = load_lock
        self._load_lock_wait_s = load_lock_wait_s
        # Optional so existing callers that build a loader with two arguments keep
        # working; without it the class does the swapping and none of the memory checks.
        self._budget = budget
        # A sensible number is insisted on here, at construction, rather than passed on to
        # `lms`. The app ignores the number today, but a version that honours it would
        # take a zero as "use my own default", and failing loudly when the loader is built
        # is better than failing quietly on the first morning it is used with nobody home.
        if context_length <= 0:
            raise ValueError(
                f"context_length must be a positive number of tokens, got {context_length}"
            )
        self._context_length = context_length
        # Asks the loaded model one short question and times the answer. Passed in, so a
        # test hands over a fake and needs no server. Optional for the same reason the
        # budget is: a loader built by hand for a script still swaps models, it only does
        # not judge their speed.
        self._speed_probe = speed_probe
        self._min_tokens_per_second = min_tokens_per_second
        self._min_free_vram_mib = min_free_vram_mib
        # The waiting function. Real time by default; a test passes one that returns at
        # once, so the ten second wait for a card that never recovers costs nothing.
        self._sleep = sleep
        # Why the last `ensure` failed, in words a person can act on. Kept because
        # returning a bare False tells a caller that something went wrong and nothing
        # about what, and "the model does not fit" and "the CLI is missing" want
        # completely different responses.
        self.last_error: str = ""
        # What the last `ensure` found loaded, read back from the server: see `LoadReport`.
        # None until a model has been read back. Set on a failed check too, so the reason
        # in `last_error` and the numbers behind it can be read together.
        self.last_load: LoadReport | None = None

    @property
    def context_length(self) -> int:
        """Tokens of context every load asks for. Read-only: chosen once, at construction.

            ModelLoader(runner, registry, context_length=32768).context_length  ->  32768

        Public so a caller can report what a load will ask for (a status line, a test)
        without reading a private field.
        """
        return self._context_length

    # ── inspection ──

    def resident(self) -> list[Any]:
        """Every model instance currently in VRAM, duplicates included."""
        try:
            return list(self._registry.loaded())
        except Exception:
            # Unknown is treated as "nothing resident", which at worst causes one
            # redundant load attempt. Guessing the other way would skip a needed load.
            return []

    def resident_keys(self) -> set[str]:
        """Distinct model keys in VRAM.

        Deliberately *not* the residency test to use before loading. Two instances of one
        model collapse to a single key here, which is exactly the blindness that let a
        duplicate load go unnoticed: use `duplicates()` for that question.
        """
        return {model.key for model in self.resident()}

    def duplicates(self) -> list[Any]:
        """Instances that are a second (or third) copy of a model already loaded.

            [coder(id="qwen/coder-14b"), coder(id="qwen/coder-14b:2")]
              ->  [coder(id="qwen/coder-14b:2")]

        The first instance of each key is kept and the rest are reported, because the
        extra copies are pure waste: they serve no request the first cannot, and each one
        takes its full share of weights and KV cache.
        """
        seen: set[str] = set()
        extras = []
        for model in self.resident():
            if model.key in seen:
                extras.append(model)
            else:
                seen.add(model.key)
        return extras

    def unload_duplicates(self) -> int:
        """Unload every duplicate instance, returning how many were removed.

        Worth doing even when nothing is about to be loaded, because a duplicate arrives
        without anything asking for it: just-in-time loading creates one whenever a
        request names a model whose loaded instance carries a different identifier.
        """
        extras = self.duplicates()
        for model in extras:
            # By identifier, not by key: unloading by key would be ambiguous between the
            # copy being removed and the one being kept.
            self._run(["unload", model.identifier])
        return len(extras)

    # ── loading ──

    def ensure(self, model_key: str, *, evict_others: bool = True,
               gpu_ratio: float | None = None) -> bool:
        """Make `model_key` the resident model. Returns whether it is, as far as we know.

        `gpu_ratio` places only that share of the model's layers on the card, leaving the
        rest in system RAM. Default None means "all of it", which is what any model that
        fits should use. See `_load_flags` for when a ratio is worth setting; in short,
        for a mixture-of-experts model and essentially never for a dense one.

        The whole sequence runs inside the load lock shared by every process (see the
        first lines of the method), so the look at what is loaded and the unload and load
        that may follow are never interleaved with another process doing the same.

        The sequence, and why each step is there:

        1. **Clear duplicates** even when the right model is already loaded: a second
           copy of the right model is as damaging as the wrong model.
        2. **Refuse an impossible load** before touching anything, so a model too large
           for the card fails fast and leaves the working one alone. Without this the
           unload happens first and the run is left with *no* model.
        3. **Unload everything and wait for the memory to actually come back**, rather
           than assuming the command's return means the card is free.
        4. **Load**, then confirm by asking what is resident rather than trusting the exit
           code, because `lms load` accepts options it then ignores (the context length
           is one, see `_load_flags`).
        5. **Read back what was really loaded** and check it: one prediction slot, a
           floor of free memory, and an answer at a usable speed. These are the checks
           that cannot be made in advance. A model that fails one is unloaded.

        Returns False rather than raising when anything fails. A caller that cannot get
        its preferred model should carry on with whatever is loaded: a slower or slightly
        worse model is a far better outcome than aborting a run that has already paid for
        searching and ranking.
        """
        self.last_error = ""
        self.last_load = None

        if self._load_lock is None:
            return self._ensure_in_turn(model_key, evict_others, gpu_ratio)

        # One process at a time, across every process on the machine. Each agent runs
        # its own copy of the MCP server, and they share one card. What breaks without
        # this: two of them find the model missing at the same moment and each runs
        # `unload --all` and `load`, which asks two models onto a card that holds one, or
        # loads one model twice.
        if not self._load_lock.wait_for(self._load_lock_wait_s, f"loading {model_key}"):
            self.last_error = (
                f"{model_key} was not loaded: another process holds the model load lock "
                f"({self._load_lock.describe_holder()}) and did not release it within "
                f"{self._load_lock_wait_s:.0f} seconds"
            )
            return False
        # `finally` so the lock is given back even when something below raises. A lock
        # left behind would make every other process wait out its full limit.
        try:
            # Look again, with a fresh answer. What this process read before it waited
            # is out of date by exactly the thing that matters: the process it waited
            # for has probably just loaded this very model. `_ensure_in_turn` starts by
            # asking what is loaded, finds it, and loads nothing.
            self._registry.refresh()
            return self._ensure_in_turn(model_key, evict_others, gpu_ratio)
        finally:
            self._load_lock.release()

    def _ensure_in_turn(self, model_key: str, evict_others: bool,
                        gpu_ratio: float | None) -> bool:
        """The body of `ensure`, run while this process holds the load lock.

        Split out so the lock is taken and given back in exactly one place, and nothing
        in the long sequence below can return without releasing it.
        """
        if self.unload_duplicates():
            self._settle()

        # "The right model is loaded" is not the same as "the card is in a good state".
        # Returning early on the first condition alone was a real bug: with gpt-oss-20b
        # and the coder both resident, asking for the coder returned True and left 199 MiB
        # free, because the coder *was* there, so the guard reported success on exactly
        # the overfilled card it exists to prevent. When eviction is wanted, the model must
        # be resident **and alone**.
        others = [key for key in self.resident_keys() if key != model_key]
        if model_key in self.resident_keys() and not (evict_others and others):
            report = self._read_back(model_key)
            if report.parallel_loaded in (None, self._PARALLEL):
                self.last_load = report
                return True
            # Resident, but not as this class loads it: something else loaded it with
            # several prediction slots. That is what the server does by itself when a
            # request names a model that is not loaded, and it is the one load on record
            # that crawled. So it is removed here and loaded again below with one slot.
            # What breaks if it is accepted as it stands: a day's work runs on the slow
            # load because "the model is loaded" was true.
            for model in self.resident():
                if model.key == model_key:
                    self._run(["unload", model.identifier])
            self._settle()

        # Only now, with a load genuinely about to happen, is "does it fit" worth asking.
        # Asking earlier was a bug: it refused a model that was *already loaded and
        # running at full speed*, because the estimate of how much memory an unload would
        # return is deliberately conservative and came out below what the model needs. An
        # estimate has no business overruling a model that is demonstrably working.
        weights_mib = self._weights_mib(model_key)

        # A model the registry has never heard of cannot be sized, so it cannot be checked
        # and an unsizeable model must not be allowed to cost a working one. Refusing
        # here, *before* the unload, keeps whatever is currently resident.
        #
        # This is not hypothetical. `lms ls` stopped listing openai/gpt-oss-20b partway
        # through a session, so the size lookup returned 0, the "unknown means do not
        # block" rule skipped the check, and the loader unloaded a healthy coder model for
        # a load that then failed, leaving the card with nothing on it. "Unknown" is a
        # reason to be careful, not a reason to proceed.
        if not weights_mib and self.resident():
            self.last_error = (
                f"{model_key} is not in the model registry, so its memory cost cannot be "
                f"estimated; keeping the resident model rather than evicting it for an "
                f"unknown quantity"
            )
            return False

        # A caller asking for partial offload has *already decided* the weights do not all
        # fit, that is what the ratio is for, so the whole-model fit check would refuse
        # every such load by definition. Skipping it here is not dropping the safeguard:
        # the checks after the load below still run, and the timed answer is the one that
        # actually catches an overfilled card. The check before a load can only reject the
        # impossible, and "impossible" is the wrong word for a load that is deliberately
        # placing some layers in system RAM.
        #
        # What breaks if this is left out: a mixture-of-experts model can never be loaded
        # at all. Its 18.6 GiB of weights do not fit a 16 GiB card, so `fits` returns
        # False and `ensure` keeps the smaller model, silently from the caller's point of
        # view, because the return value is a bare False.
        if self._budget is not None and weights_mib and gpu_ratio is None:
            # Measured against an *empty* card rather than current free memory: everything
            # else is about to be unloaded, so the question is whether this model fits on
            # its own, not whether it fits alongside what happens to be loaded now.
            free_when_empty = self._free_if_emptied()
            if not self._budget.fits(weights_mib, already_free_mib=free_when_empty):
                self.last_error = (
                    f"{model_key} has {weights_mib} MiB of weights, which with a "
                    f"{self._budget.weights_margin_mib()} MiB margin do not fit in the "
                    f"{free_when_empty} MiB that would be free with nothing else loaded; "
                    f"it was not loaded"
                )
                return False

        if evict_others:
            # Ignore the result: if nothing was loaded this is a no-op, and if it fails
            # the load below surfaces the real problem.
            self._run(["unload", "--all"])
            self._settle()

        if not self._run(["load", model_key, *self._load_flags(gpu_ratio), "-y"]):
            self.last_error = f"lms load {model_key} failed"
            return False

        if not self._await_resident(model_key):
            self.last_error = f"{model_key} did not become resident after loading"
            return False

        return self._verify_loaded(model_key)

    def _verify_loaded(self, model_key: str) -> bool:
        """Check a model that has just loaded, and unload it if a check fails.

        A load can succeed and still leave the machine in the state this class exists to
        avoid. Three checks, cheapest first, so a load that is wrong on its face is not
        also asked to generate text:

            parallel 4 read back              ->  unloaded, "loaded with 4 prediction slots"
            40 MiB free, floor 100            ->  unloaded, "left only 40 MiB of graphics ..."
            6.5 tokens a second, floor 20     ->  unloaded, "answered at 6.5 tokens a second"
            148 MiB free, 96 tokens a second  ->  kept: a full card that answers fast is
                                                  the normal state on this version

        Unloading is the right response to a failure: running on an overfilled card is
        slower than running on the CPU would be, and unlike a failure it reports nothing.
        """
        report = self._read_back(model_key)
        self.last_load = report

        # `None` (an `lms` that does not report the field) is let through: it is "cannot
        # tell", and failing every load on a version that says less would break a working
        # setup. A number that is not 1 is a fact, and is refused.
        if report.parallel_loaded not in (None, self._PARALLEL):
            return self._reject(
                f"{model_key} loaded with {report.parallel_loaded} prediction slots "
                f"although {self._PARALLEL} was asked for; unloaded it, because each slot "
                f"keeps its own share of the context memory"
            )

        if not self._await_free_floor():
            free = None if self._budget is None else self._budget.free_mib()
            self.last_load = replace(report, vram_free_mib=free)
            return self._reject(
                f"{model_key} loaded but left only {free} MiB of graphics memory free, "
                f"under the {self._min_free_vram_mib} MiB floor; unloaded it, because a "
                f"card with nothing left spills the next allocation to system memory"
            )

        if self._speed_probe is not None:
            reading = self._speed_probe.measure(model_key)
            # `replace` makes a copy of the frozen record with some fields changed.
            report = replace(
                report,
                tokens_per_second=None if reading.problem else reading.tokens_per_second,
                # Read again after the answer: generating takes a little more memory than
                # sitting idle, and this is the figure a caller wants to see.
                vram_free_mib=None if self._budget is None else self._budget.free_mib(),
            )
            self.last_load = report
            if reading.problem:
                return self._reject(
                    f"{model_key} loaded but did not pass the speed check "
                    f"({reading.problem}); unloaded it"
                )
            if reading.tokens_per_second < self._min_tokens_per_second:
                return self._reject(
                    f"{model_key} loaded but answered at {reading.tokens_per_second:.1f} "
                    f"tokens a second, under the {self._min_tokens_per_second:g} floor "
                    f"(context {report.context_loaded}, {report.vram_free_mib} MiB of "
                    f"graphics memory free); unloaded it rather than crawl through the day"
                )

        return True

    def _reject(self, reason: str) -> bool:
        """Record why a loaded model is not being kept, unload it, and return False.

        Returns False so the caller can write `return self._reject(...)`.
        """
        self.last_error = reason
        self._run(["unload", "--all"])
        return False

    def _read_back(self, model_key: str) -> LoadReport:
        """What the server says it really loaded, from `lms ps`.

            {"modelKey": "qwen/qwen3.5-9b", "contextLength": 243968, "parallel": 1}
              ->  LoadReport(model="qwen/qwen3.5-9b", context_asked=32768,
                             context_loaded=243968, parallel_loaded=1,
                             tokens_per_second=None, vram_free_mib=712)

        The registry turns a missing field into 0, and no real load has a context or a
        slot count of 0, so 0 is reported as None ("could not be read").

        Under the hood: the registry keeps each `lms ps` answer for three seconds. The
        answer used here is the one `_await_resident` has just fetched to see the model
        arrive, so it already describes the new load.
        """
        loaded = next((model for model in self.resident() if model.key == model_key), None)
        return LoadReport(
            model=model_key,
            context_asked=self._context_length,
            context_loaded=(getattr(loaded, "context_length", 0) or None),
            parallel_loaded=(getattr(loaded, "parallel", 0) or None),
            tokens_per_second=None,
            vram_free_mib=None if self._budget is None else self._budget.free_mib(),
        )

    def _await_free_floor(self) -> bool:
        """Whether the card has its floor of free memory, allowing time for it to settle.

            floor 100:  148 MiB free  ->  True       (a full card by design: normal)
                         40 MiB free for ten seconds  ->  False
                        card cannot be read           ->  True  (unknown is not "empty")

        This floor is small on purpose. On this version of the app every load fills the
        card, so a floor of a few hundred MiB refuses loads that run at full speed (148
        MiB free at 96 tokens a second was measured). It is kept only for a card that is
        truly exhausted; the timed answer is what judges everything above that.

        Checking once was wrong, and produced a false alarm on the first real run: the
        previous model's memory had not finished being released when the new one finished
        loading, so a low momentary reading was taken as proof of an overfilled card. So
        the check waits for the reading to recover and gives up only if the card is still
        short after several seconds: persistent, not transient.
        """
        if self._budget is None:
            return True
        for _ in range(int(self._FREE_FLOOR_SETTLE_S * 2)):
            free = self._budget.free_mib()
            if free is None or free >= self._min_free_vram_mib:
                return True
            self._sleep(0.5)
        return False

    # ── internals ──

    def _weights_mib(self, model_key: str) -> int:
        """The on-disk size of a model in MiB, or 0 when it cannot be determined.

            {"key": "qwen/qwen3.5-9b", "size_mib": 6706}  ->  6706

        Zero means "unknown", and every caller treats unknown as "do not refuse on these
        grounds": a missing size must not become a reason to block a load that would have
        worked.
        """
        try:
            for entry in self._registry.installed():
                if entry.get("key") == model_key:
                    return int(entry.get("size_mib") or 0)
        except Exception:
            pass
        return 0

    def _free_if_emptied(self) -> int | None:
        """How much VRAM would be free if every loaded model were unloaded.

            free now 600 MiB, resident models 8,571 + 6,706  ->  15,877 MiB

        Computed rather than measured, because measuring it would mean unloading first and
        the point of the check is to decide *whether* to unload. The models' own weights
        are a floor rather than the true figure, their KV caches come back too, so this
        under-estimates the free memory, which errs toward refusing a marginal load.
        """
        if self._budget is None:
            return None
        free = self._budget.free_mib()
        if free is None:
            return None
        # Each resident model returns its *whole* footprint, not just its weights. Summing
        # weights alone under-counted badly: with the coder resident and 398 MiB free it
        # estimated 8,969 MiB would be recovered, when the true figure is about 14,600,
        # because the KV cache is nearly as large as the weights and comes back too. That
        # under-count made the guard refuse the one model known to work.
        recovered = free + sum(
            self._budget.required_mib(model.size_mib) for model in self.resident()
        )

        # Capped at the physical card. Two models whose weights total more than the card
        # produced an estimate of 20,332 MiB free on a 16,303 MiB card, which is not a
        # slip in the arithmetic but the fingerprint of the overflow being guarded
        # against, since it can only happen when what is loaded already exceeds what fits.
        # Left uncapped, the guard would read that as abundant room and wave through the
        # next load.
        total = self._budget.total_mib()
        return recovered if total is None else min(recovered, total)

    def _settle(self) -> None:
        """Wait until the card's free memory stops rising, or the timeout expires.

        `lms unload` returns when the request is accepted, not when the driver has
        released the memory. Polling until the figure stops moving is more reliable than
        any fixed sleep: the release is usually under a second and occasionally several,
        so a fixed sleep is either wasteful or wrong.
        """
        if self._budget is None:
            self._sleep(2.0)
            return
        previous = -1
        deadline = time.monotonic() + self._UNLOAD_SETTLE_S
        while time.monotonic() < deadline:
            self._sleep(0.5)
            free = self._budget.free_mib()
            if free is None:
                return
            if free <= previous:
                # Stopped rising, so the release is done.
                return
            previous = free

    def _await_resident(self, model_key: str) -> bool:
        """Poll until the model shows up as loaded, rather than trusting the exit code."""
        for _ in range(20):
            if model_key in self.resident_keys():
                return True
            self._sleep(1.0)
        return False

    def _run(self, args: list[str]) -> bool:
        """Run one `lms` command, returning whether it succeeded."""
        executable = getattr(self._runner, "_executable", None)
        if executable is None:
            return False
        try:
            path = executable()
        except Exception:
            return False
        if not path or not path.exists():
            return False

        try:
            completed = subprocess.run(
                [str(path), *args],
                capture_output=True, text=True,
                timeout=self._LOAD_TIMEOUT_S, check=False,
                # The CLI prints progress spinners containing bytes the Windows console
                # codepage cannot decode, which raised UnicodeDecodeError inside
                # subprocess's reader thread and made a successful load look like a crash.
                encoding="utf-8", errors="replace",
                # Without Electron's "run as a script engine" switch (childenv.py). A
                # load may have to wake LM Studio's service first, and with the switch
                # inherited from an editor the app exits instead of starting.
                env=lms_environment(),
            )
            return completed.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False
