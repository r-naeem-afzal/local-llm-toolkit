"""Making sure the model server is up and a model is loaded, with nobody watching.

An unattended tool (dayshift) calls this at the start of a working day. That caller cannot
answer a prompt, cannot notice a frozen window and cannot press a key, so the contract is
stricter than for a person at a keyboard:

* **It never hangs.** Every step has a time limit, and nothing loops without a bound.
* **It never overloads the machine.** It refuses to load a model when free RAM is low or
  when the GPU cannot be read, and leaves the rest to `ModelLoader`, which keeps a model
  only if it answers at a usable speed once loaded.
* **It always says plainly what happened.** The result is data, not an exception, so the
  caller can log "RAM too low (3,100 MiB free, needs 6,144)" instead of a stack trace.

Run it by hand with:

    python -m local_llm.server_start [--model KEY] [--no-load] [--json]

## The sequence

    1. server answering?            yes -> skip to 3
    2. `lms server start`, a short wait for the server to answer, and again, until it
       answers or two minutes have gone by (see below)
    3. unless --no-load, for each model to try, in order:
           RAM above the floor?  GPU readable?  ModelLoader.ensure(model)
       and stop at the first model that loads and answers at a usable speed
    4. all of it under one lock file, so two starters cannot race

The JSON answer says what was really loaded, not what was asked for: `context_asked`
beside `context_loaded`, `parallel_loaded`, `tokens_per_second` and
`vram_free_after_load_mib`. They differ in practice. The app ignores the context length
it is given and sizes the context to the free graphics memory itself (`loader.py`).

## Which models, and what happens when the first does not load

With no `--model`, the starter tries `Settings.unattended_models` in order, by default the
14B and then the 9B. A day on which other programs hold part of the card can refuse the
14B and still have room for the 9B, and a shift with some model loaded is worth more than
a shift with none. The result says which model loaded (`model`); when none did, `error`
gives the reason for each.

`--model KEY` means exactly that model and never falls back. A person who names a model is
testing it or needs it, and a run that loaded a different one while reporting success
would leave them believing the one they named was resident.

The RAM floor and the unreadable-GPU refusal are checked again before every attempt, not
only the first. A failed attempt changes the machine (a model was mapped into RAM, loaded
and unloaded), so a reading taken before attempt one can be wrong by attempt two.

## Why `lms server start` is tried for up to two minutes

Measured on 13 October 2026: from a cold start the first call prints "Timed out waiting
for LM Studio daemon to start" although nothing is wrong. The daemon (the background
service that `lms` talks to) is starting, and needs about a minute. A call made a little
later succeeds. So a failed call is not the server being unavailable, and the starter
keeps going: run the command, watch for the server to answer for fifteen seconds, run it
again, until the server answers or `Settings.server_start_wait_s` (two minutes) is used up.

An earlier version retried exactly once and then waited 30 seconds. That gave up on most
cold mornings, which is the exact situation this tool exists for. What keeps the loop
safe with nobody home is the deadline, worked out once before the first call: no path
through the loop can make the total wait longer, and when it runs out the error carries
the number of calls made and the last thing `lms` printed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .childenv import lms_environment
from .config import Settings
# `Clock`, `SystemClock` and `LockFile` used to be defined in this module. They moved to
# lockfile.py when the model loader needed the same lock, and are imported here so the
# names this module has always offered still resolve.
from .lockfile import Clock, LockFile, SystemClock  # noqa: F401

# ─────────────────────────── result ───────────────────────────


@dataclass(frozen=True)
class StartResult:
    """What one run did and found. "Frozen" means it cannot be changed after creation, so
    a caller that logs it and passes it on cannot accidentally log a different story.

        server up, model loaded, nothing started by us
          -> StartResult(server_up=True, server_started_by_us=False, model_loaded=True,
                         refused="", error="", ...)
        RAM too low
          -> server_up=True, model_loaded=False, refused="free RAM 3100 MiB is below ..."
        the 14B would not load but the 9B did
          -> model="qwen/qwen3.5-9b", model_loaded=True, error=""
        neither would load
          -> model="qwen/qwen3-14b", model_loaded=False,
             error="qwen/qwen3-14b has 8584 MiB of weights, which ...; qwen/qwen3.5-9b
                    loaded but answered at 6.5 tokens a second ..."
    """

    server_up: bool
    server_started_by_us: bool
    # The model that loaded. When none did, the first model the run would have tried, so
    # a refusal or an error still names the model that was wanted.
    model: str
    model_loaded: bool
    # A deliberate decision not to load ("RAM too low"). Not a failure of the tool. Only
    # set when the run was stopped before any model had been tried and failed: once an
    # attempt has failed, every later reason goes into `error` instead, so a caller that
    # reads only one of the two fields sees all of them.
    refused: str
    # Something went wrong ("lms.exe not found", "server never answered", the loader's
    # last_error). When several models were tried and none loaded, one reason per model,
    # joined with "; ". Empty when everything went to plan, including when a later model
    # loaded after an earlier one did not.
    error: str
    # Free graphics memory and free RAM as the guards read them BEFORE the load.
    vram_free_mib: int | None
    ram_free_mib: int | None
    seconds: float
    # What the load really produced, read back from the server afterwards. All None when
    # no model was loaded by this run. They exist because what is asked for and what is
    # loaded differ: the app ignores the context length it is given (`loader.py`).
    #
    #     asked 32768, the app chose its own context
    #       -> context_asked=32768, context_loaded=243968, parallel_loaded=1,
    #          tokens_per_second=96.4, vram_free_after_load_mib=712
    #
    # Defaults of None so the many places that build a result for a run that loaded
    # nothing (server down, refused, --no-load) do not each have to spell out five Nones.
    context_asked: int | None = None
    context_loaded: int | None = None
    parallel_loaded: int | None = None
    tokens_per_second: float | None = None
    vram_free_after_load_mib: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def succeeded(self, load_requested: bool) -> bool:
        """The one definition of "all good", shared by the command line's exit code.

            server up, model loaded            -> True
            server up, --no-load               -> True
            server up, model refused           -> False
        """
        if not self.server_up:
            return False
        return self.model_loaded or not load_requested


# ─────────────────────────── small collaborators ───────────────────────────
#
# Each of these is a narrow interface with one real implementation, so a test hands the
# starter fakes and never has to patch a module attribute, start a process or wait.


@dataclass(frozen=True)
class CommandOutcome:
    """How one `lms` call ended. Only the facts the starter acts on."""

    ran: bool          # False when the CLI could not be started at all (missing file)
    timed_out: bool
    exit_code: int | None
    text: str          # the tail of what it printed, for error messages


class ServerCommand(Protocol):
    def start(self, timeout_s: float) -> CommandOutcome: ...


class MemoryReader(Protocol):
    def free_ram_mib(self) -> int | None: ...


class PsutilMemory:
    """Free system RAM, via psutil (already a dependency of the monitor)."""

    def free_ram_mib(self) -> int | None:
        """RAM the OS says programs can use right now, or None when it cannot be read.

            virtual_memory().available = 9,663,676,416 bytes  ->  9216 MiB

        `available` rather than `free`: on Windows "free" excludes cache that the OS hands
        back the moment a program asks, so it understates what a model load can really
        get and would refuse loads that would have worked.
        """
        try:
            import psutil

            return int(psutil.virtual_memory().available // 1024 // 1024)
        except Exception:
            return None


class LmsServerCommand:
    """Runs `lms server start` with a hard time limit.

    Takes a function that returns the lms path rather than the path itself, so the "where
    is lms" rule stays in one place (`LmsCommandRunner`) and an override in settings is
    honoured.
    """

    # How much of the CLI's output to keep for error messages.
    _TAIL_CHARS = 300

    def __init__(self, locate: Callable[[], Path]) -> None:
        self._locate = locate

    def start(self, timeout_s: float) -> CommandOutcome:
        path = self._locate()
        if not path or not Path(path).exists():
            return CommandOutcome(ran=False, timed_out=False, exit_code=None,
                                  text=f"lms not found at {path}")

        # Output goes to a temporary FILE, not a pipe. `lms server start` leaves the
        # server daemon running as a child that inherits the command's output handles. With
        # a pipe, Python waits for the pipe to close, which only happens when that daemon
        # exits, i.e. never. The call would then hang past its timeout, which is the one
        # thing this tool must not do unattended. A file has no "wait for the other end".
        with tempfile.TemporaryFile() as output:
            try:
                process = subprocess.Popen(
                    [str(path), "server", "start"],
                    stdin=subprocess.DEVNULL,   # never wait for a keypress
                    stdout=output,
                    stderr=subprocess.STDOUT,   # same file, so errors are kept too
                    # Without Electron's "run as a script engine" switch. This command
                    # is the one that wakes LM Studio's service by starting the app's
                    # own program file. Started from inside an editor it inherits the
                    # switch, the app then exits at once, and all that shows here is
                    # "Timed out waiting for LM Studio daemon to start" however long
                    # we wait. Measured: 60 seconds and failure with the switch, 7
                    # seconds and success without it. See childenv.py.
                    env=lms_environment(),
                )
            except OSError as exc:
                return CommandOutcome(ran=False, timed_out=False, exit_code=None,
                                      text=f"could not start lms: {exc}")

            timed_out = False
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass  # cannot do more; we stop waiting for it and report the timeout

            output.seek(0)
            # errors="replace": the CLI prints spinner characters the console codepage
            # cannot decode, which once turned a good run into a UnicodeDecodeError.
            text = output.read().decode("utf-8", errors="replace").strip()
            return CommandOutcome(
                ran=True, timed_out=timed_out, exit_code=process.returncode,
                text=text[-self._TAIL_CHARS:],
            )


# ─────────────────────────── the starter ───────────────────────────


@dataclass(frozen=True)
class _GuardReading:
    """What the two load guards saw just before one load attempt.

        RAM 20000 MiB free, GPU 14000 MiB free
          ->  _GuardReading(ram_mib=20000, vram_mib=14000, refusal="")
        RAM 3100 MiB free, floor 6144
          ->  _GuardReading(ram_mib=3100, vram_mib=None,
                            refusal="free RAM is 3100 MiB, below the 6144 MiB floor; ...")

    Frozen (cannot be changed after it is made) because it is a record of a moment: a
    caller that logs it and passes it on cannot accidentally log a different story.
    """

    ram_mib: int | None
    vram_mib: int | None     # None when not read (RAM already refused) or unreadable
    refusal: str             # "" means the attempt may go ahead


class ModelServerStarter:
    """Brings the model server up and a model in, safely. See the module docstring."""

    # Gap between "is it answering yet" checks. Longer than the probe's 2 s cache so every
    # check is a fresh answer, and long enough not to hammer a server that is booting.
    _POLL_INTERVAL_S = 3.0

    # How long to watch for the server after one `lms server start` before running the
    # command again. Long enough for a daemon that was nearly up to finish, short enough
    # that a two minute limit holds several attempts.
    _RETRY_GAP_S = 15.0

    def __init__(self, settings: Settings, probe: Any, command: ServerCommand, loader: Any,
                 budget: Any, memory: MemoryReader, lock: LockFile, clock: Clock) -> None:
        self._settings = settings
        self._probe = probe      # needs is_up_now() -> bool
        self._command = command
        # Needs ensure(key) -> bool, resident_keys() -> the model keys loaded now, and
        # .last_error. May also have .last_load, the record
        # of what was really loaded (`loader.LoadReport`); a loader without one is fine.
        self._loader = loader
        self._budget = budget    # needs free_mib() -> int | None  (None = unreadable)
        self._memory = memory
        self._lock = lock
        self._clock = clock

    def ensure_ready(self, model_key: str | None = None, load_model: bool = True) -> StartResult:
        """Run the whole sequence once and return what happened. Does not raise for
        expected failures; those come back as `refused` or `error`.

            ensure_ready()                  ->  tries Settings.unattended_models in order
            ensure_ready(model_key="a/b")   ->  tries "a/b" and nothing else
        """
        candidates = self._candidates(model_key)
        began = self._clock.monotonic()

        if not self._lock.acquire():
            return self._finish(
                began, candidates[0], server_up=self._probe.is_up_now(),
                refused=f"another starter holds the lock at {self._lock.path}; "
                        f"started nothing",
            )
        # `finally` so the lock is removed even when something below raises. A leaked lock
        # would block every later run until it aged out ten minutes later.
        try:
            return self._run(began, candidates, load_model)
        finally:
            self._lock.release()

    def _candidates(self, model_key: str | None) -> list[str]:
        """The models to try, in order.

            model_key="qwen/qwen3-4b"  ->  ["qwen/qwen3-4b"]
            model_key=None             ->  ["qwen/qwen3-14b", "qwen/qwen3.5-9b"]   (the setting)

        A named model means exactly that model, with no fallback. Why: someone who names a
        model is testing it or needs it, not asking for "it, or whatever else loads". What
        happens as a result: if it will not load, the run says so and stops. What breaks if
        a named model fell back: a run that asked for the 14B could report success having
        loaded the 9B, and the caller would go on believing the 14B was resident.
        """
        if model_key:
            return [model_key]
        return list(self._settings.unattended_models)

    # ── steps ──

    def _run(self, began: float, candidates: list[str], load_model: bool) -> StartResult:
        started_by_us = False
        if not self._probe.is_up_now():
            error = self._start_server()
            if error:
                return self._finish(began, candidates[0], server_up=False, error=error)
            # It was down when we arrived and answers now, so our commands did it.
            started_by_us = True

        if not load_model:
            return self._finish(
                began, candidates[0], server_up=True, started_by_us=started_by_us
            )

        return self._load(began, candidates, started_by_us)

    def _start_server(self) -> str:
        """Start the server. Returns "" once it answers, or a plain error sentence.

            first call "daemon timed out", server answers 40 s later  ->  ""
            lms.exe is missing                                        ->  "could not run lms: ..."
            two minutes of calls and the server never answers         ->  "the model server
                did not answer within 120 seconds of `lms server start` (6 calls, ...)"

        Every round is the same two steps: run the command, then watch for the server for
        `_RETRY_GAP_S`. A round is not told apart by what `lms` printed. The known reason
        for a failed call is the daemon still starting, and for any other reason trying
        again costs at most the same two minutes and ends with the last output in the
        error, so one rule covers both and there is no text of `lms` to keep in step with.
        """
        wait_s = self._settings.server_start_wait_s
        # The deadline is computed once, before anything runs, so the total cannot grow no
        # matter how the loop behaves. Without it a server that never comes up would block
        # the whole working day's first step.
        deadline = self._clock.monotonic() + wait_s
        calls = 0
        last: CommandOutcome | None = None

        while True:
            remaining = deadline - self._clock.monotonic()
            if remaining <= 0:
                break
            # A call is never given longer than what is left, so the last call cannot
            # carry the run past the deadline by its own time limit.
            last = self._command.start(
                min(self._settings.server_start_cmd_timeout_s, remaining)
            )
            calls += 1
            if not last.ran:
                # Nothing to retry: a missing file stays missing.
                return f"could not run lms: {last.text}"
            remaining = deadline - self._clock.monotonic()
            if self._answers_within(min(self._RETRY_GAP_S, max(remaining, 0.0))):
                return ""

        detail = "no call was made"
        output = ""
        if last is not None:
            detail = "timed out" if last.timed_out else f"exit code {last.exit_code}"
            output = f" Last output: {last.text}" if last.text else ""
        return (f"the model server did not answer within {wait_s:.0f} seconds of "
                f"`lms server start` ({calls} calls, last call: {detail}).{output}")

    def _answers_within(self, seconds: float) -> bool:
        """Watch for the server to answer for up to `seconds`. True as soon as it does.

            answers on the third look, 6 seconds in  ->  True   (after two sleeps of 3 s)
            never answers, seconds=15                ->  False  (after 15 s, six looks)

        Always looks at least once, so a zero wait is still one fresh check.
        """
        end = self._clock.monotonic() + seconds
        while True:
            if self._probe.is_up_now():
                return True
            left = end - self._clock.monotonic()
            if left <= 0:
                return False
            self._clock.sleep(min(self._POLL_INTERVAL_S, left))

    def _load(self, began: float, candidates: list[str], started_by_us: bool) -> StartResult:
        """Try each model in turn and stop at the first that loads, guarding every attempt.

        The ways this ends, and which field carries the story:

            the first model loads                        ->  model=14B, loaded, no error
            the 14B fails, the 9B loads                  ->  model=9B, loaded, no error
            every model fails                            ->  model=14B (first choice), not
                                                             loaded, error = one reason each
            a guard refuses, a wanted model is loaded    ->  that model, loaded, no error
                                                             (see `_keep_loaded`)
            a guard refuses before any model was tried   ->  model=14B, not loaded, refused
            a guard refuses after one model had failed   ->  model=14B, not loaded, error =
                                                             that failure plus "the 9B was
                                                             not tried: <the guard's words>"

        The last case goes into `error`, not `refused`, so that a caller reading only one
        of the two fields (dayshift reads `refused` first, then `error`) still sees the
        failed attempt as well as the refusal.
        """
        failures: list[str] = []
        reading = _GuardReading(ram_mib=None, vram_mib=None, refusal="")

        for model in candidates:
            # Read again before every attempt, not once before the first. A failed attempt
            # changes the machine: a model was mapped into RAM, loaded, found too slow
            # and unloaded. What breaks with one early reading: attempt two goes ahead on
            # a RAM figure from before attempt one, on a machine that may no longer pass
            # the guard that attempt one passed.
            reading = self._read_guards()
            if reading.refusal:
                # The guards protect the machine from a NEW load. A model that is already
                # loaded needs none, so before reporting a refusal, see whether one of
                # the wanted models is there already and keep it.
                #
                # Measured on 13 October 2026: with the 14B loaded and answering at 70
                # tokens a second, free RAM was 4,426 MiB, because the loaded model's
                # file is itself held in RAM. A second run of the starter then said "free
                # RAM is below the floor; the model was not loaded" and exited 1, with
                # the model loaded. What breaks without this: every run after the first
                # tells the caller there is no model, and the day's work is not handed
                # to a model that is sitting there ready.
                kept = self._keep_loaded(candidates)
                if kept:
                    return self._finish(
                        began, kept, server_up=True, started_by_us=started_by_us,
                        loaded=True, ram=reading.ram_mib, vram=reading.vram_mib,
                        load=getattr(self._loader, "last_load", None),
                    )
                if not failures:
                    return self._finish(
                        began, candidates[0], server_up=True, started_by_us=started_by_us,
                        ram=reading.ram_mib, vram=reading.vram_mib, refused=reading.refusal,
                    )
                failures.append(f"{model} was not tried: {reading.refusal}")
                break

            if self._loader.ensure(model):
                return self._finish(
                    began, model, server_up=True, started_by_us=started_by_us,
                    loaded=True, ram=reading.ram_mib, vram=reading.vram_mib,
                    # `getattr` with a default: a loader that keeps no such record (a
                    # test fake, an older loader) gives None and the fields stay empty.
                    load=getattr(self._loader, "last_load", None),
                )
            failures.append(self._loader.last_error or f"could not load {model}")

        return self._finish(
            began, candidates[0], server_up=True, started_by_us=started_by_us,
            error=self._join_reasons(failures), ram=reading.ram_mib, vram=reading.vram_mib,
        )

    def _keep_loaded(self, candidates: list[str]) -> str:
        """The first wanted model that is already loaded and passes the loader, or "".

            candidates [14B, 9B], the 9B is loaded          ->  "qwen/qwen3.5-9b"
            candidates [14B, 9B], nothing is loaded         ->  ""
            candidates [14B], the 9B is loaded              ->  ""   (not a wanted model)

        `ensure` is still called on it. For a model that is loaded alone with one
        prediction slot that starts no load: it reads back what is loaded and returns.
        Any reload it does make (a model the server loaded by itself with four slots, see
        `loader.py`) puts back a model whose memory was already in use.
        """
        try:
            loaded = set(self._loader.resident_keys())
        except Exception:
            # Cannot tell what is loaded, so nothing is claimed: the refusal stands.
            return ""
        for model in candidates:
            if model in loaded and self._loader.ensure(model):
                return model
        return ""

    def _read_guards(self) -> _GuardReading:
        """Check free RAM, then GPU readability, and say whether a load may go ahead.

            RAM 20000, floor 6144, GPU 14000 MiB free  ->  may go ahead, refusal ""
            RAM 3100, floor 6144                       ->  refusal "free RAM is 3100 MiB ..."
            RAM fine, GPU cannot be read               ->  refusal "the GPU memory could ..."
        """
        ram = self._memory.free_ram_mib()
        floor = self._settings.min_free_ram_mib
        # Unknown RAM is allowed through, like the monitor: psutil is a declared
        # dependency, so None means a freak read failure, and RAM is a coarse guard. The
        # GPU check below is the strict one.
        if ram is not None and ram < floor:
            return _GuardReading(
                ram_mib=ram, vram_mib=None,
                refusal=f"free RAM is {ram} MiB, below the {floor} MiB floor; "
                        f"the model was not loaded",
            )

        vram = self._budget.free_mib()
        if vram is None:
            # `VramBudget.fits()` returns True when the GPU cannot be read, so as not to
            # break a working setup because a monitoring library is missing. That is right
            # for an interactive run and wrong here. With nobody home, an unreadable GPU
            # means we cannot tell whether the card is already full, and a load onto a
            # full card does not fail: Windows spills it into system RAM and everything
            # runs about fifteen times slower, silently, all day. Refusing costs one
            # visible "not loaded" message that the owner sees and fixes in a minute.
            return _GuardReading(
                ram_mib=ram, vram_mib=None,
                refusal="the GPU memory could not be read, so the model was not loaded "
                        "(unattended runs do not load blind)",
            )
        return _GuardReading(ram_mib=ram, vram_mib=vram, refusal="")

    @staticmethod
    def _join_reasons(reasons: list[str]) -> str:
        """One sentence per distinct reason, in the order the attempts happened.

            ["A has 19072 MiB of weights", "B answered at 6.5 tokens a second"]
              ->  "A has 19072 MiB of weights; B answered at 6.5 tokens a second"
            ["the card is full", "the card is full"]
              ->  "the card is full"

        `dict.fromkeys` is the repeat-remover here: a dictionary holds each key once and
        remembers the order keys first arrived in, so listing its keys gives the reasons
        with repeats dropped and the order kept.

        Repeats are dropped so that a run in which every attempt fails in the same words
        reads exactly as it did when only one model was ever tried. The loader's own
        messages each name their model, so in practice they differ; this is for a reason
        that does not, which would otherwise be printed once per model.
        """
        return "; ".join(dict.fromkeys(reasons))

    def _finish(self, began: float, model: str, *, server_up: bool,
                started_by_us: bool = False, loaded: bool = False, refused: str = "",
                error: str = "", ram: int | None = None,
                vram: int | None = None, load: Any = None) -> StartResult:
        """Build the result. `load` is the loader's record of what was really loaded.

            load=LoadReport(context_asked=32768, context_loaded=243968, parallel_loaded=1,
                            tokens_per_second=96.4, vram_free_mib=712)
              ->  StartResult(..., context_asked=32768, context_loaded=243968,
                              parallel_loaded=1, tokens_per_second=96.4,
                              vram_free_after_load_mib=712)
            load=None  ->  those five fields are None
        """
        return StartResult(
            server_up=server_up, server_started_by_us=started_by_us, model=model,
            model_loaded=loaded, refused=refused, error=error,
            vram_free_mib=vram, ram_free_mib=ram,
            seconds=round(self._clock.monotonic() - began, 1),
            context_asked=getattr(load, "context_asked", None),
            context_loaded=getattr(load, "context_loaded", None),
            parallel_loaded=getattr(load, "parallel_loaded", None),
            tokens_per_second=getattr(load, "tokens_per_second", None),
            vram_free_after_load_mib=getattr(load, "vram_free_mib", None),
        )


# ─────────────────────────── command line ───────────────────────────


def describe(result: StartResult, load_requested: bool) -> str:
    """Two or three plain sentences saying what happened.

        StartResult(server_up=True, model_loaded=True, model="qwen/qwen3-14b", seconds=21.4)
          -> "The model server is up (it was already running). qwen/qwen3-14b is loaded.
              Took 21.4 seconds."
    """
    if result.server_up:
        how = "started by this run" if result.server_started_by_us else "it was already running"
        first = f"The model server is up ({how})."
    else:
        first = "The model server is not up."

    if result.error:
        second = f"Problem: {result.error}"
    elif result.refused:
        second = f"Not done: {result.refused}."
    elif not load_requested:
        second = "No model was loaded because loading was switched off."
    elif result.model_loaded:
        second = f"{result.model} is loaded.{_load_details(result)}"
    else:
        second = f"{result.model} is not loaded."
    return f"{first} {second} Took {result.seconds} seconds."


def _load_details(result: StartResult) -> str:
    """One more sentence with what the load really produced, or "" when nothing is known.

        context_loaded=243968, parallel_loaded=1, tokens_per_second=96.4,
        vram_free_after_load_mib=712
          -> " Context 243968 tokens (32768 asked for), 1 prediction slot, 96.4 tokens a
              second, 712 MiB of graphics memory free."
        every field None  -> ""
    """
    parts: list[str] = []
    if result.context_loaded is not None:
        parts.append(f"context {result.context_loaded} tokens "
                     f"({result.context_asked} asked for)")
    if result.parallel_loaded is not None:
        slots = "slot" if result.parallel_loaded == 1 else "slots"
        parts.append(f"{result.parallel_loaded} prediction {slots}")
    if result.tokens_per_second is not None:
        parts.append(f"{result.tokens_per_second} tokens a second")
    if result.vram_free_after_load_mib is not None:
        parts.append(f"{result.vram_free_after_load_mib} MiB of graphics memory free")
    if not parts:
        return ""
    sentence = ", ".join(parts)
    # Capital first letter without touching the rest ("context ..." -> "Context ...").
    return f" {sentence[0].upper()}{sentence[1:]}."


def main(argv: Sequence[str] | None = None, starter: ModelServerStarter | None = None,
         out: Callable[[str], None] = print) -> int:
    """Entry point. Returns the exit code: 0 only if the server is up and (unless
    --no-load) the model is loaded, otherwise 1.

    `starter` is injectable so a test runs this with a fake and never builds the real
    object graph.
    """
    parser = argparse.ArgumentParser(
        prog="python -m local_llm.server_start",
        description="Make sure the local model server is up and a model is loaded.",
    )
    parser.add_argument(
        "--model", default=None,
        help="load exactly this model, with no fallback (default: try each model in "
             "LOCAL_LLM_UNATTENDED_MODELS in order and keep the first that loads)",
    )
    parser.add_argument("--no-load", action="store_true",
                        help="only make sure the server is up; do not load a model")
    parser.add_argument("--json", action="store_true", help="print one line of JSON")
    args = parser.parse_args(argv)

    load_requested = not args.no_load
    try:
        if starter is None:
            # Imported here so importing this module for its classes does not pull in the
            # whole composition root.
            from .container import Toolkit

            starter = Toolkit().server_starter
        result = starter.ensure_ready(model_key=args.model, load_model=load_requested)
    except Exception as exc:  # last resort: unattended callers need a line, not a trace
        result = StartResult(
            server_up=False, server_started_by_us=False, model=args.model or "",
            model_loaded=False, refused="", error=f"{type(exc).__name__}: {exc}",
            vram_free_mib=None, ram_free_mib=None, seconds=0.0,
        )

    if args.json:
        out(json.dumps(result.as_dict()))
    else:
        out(describe(result, load_requested))
    return 0 if result.succeeded(load_requested) else 1


if __name__ == "__main__":
    sys.exit(main())
