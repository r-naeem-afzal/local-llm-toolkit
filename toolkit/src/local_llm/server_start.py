"""Making sure the model server is up and its default model is loaded, with nobody watching.

An unattended tool (dayshift) calls this at the start of a working day. That caller cannot
answer a prompt, cannot notice a frozen window and cannot press a key, so the contract is
stricter than for a person at a keyboard:

* **It never hangs.** Every step has a time limit, and nothing loops without a bound.
* **It never overloads the machine.** It refuses to load a model when free RAM is low or
  when the GPU cannot be read, and leaves the final VRAM decision to `ModelLoader`.
* **It always says plainly what happened.** The result is data, not an exception, so the
  caller can log "RAM too low (3,100 MiB free, needs 6,144)" instead of a stack trace.

Run it by hand with:

    python -m local_llm.server_start [--model KEY] [--no-load] [--json]

## The sequence

    1. server answering?            yes -> skip to 3
    2. `lms server start`           (twice if the first did not work, see below)
       then wait a bounded time for the server to answer
    3. unless --no-load:  RAM above the floor?  GPU readable?  ModelLoader.ensure(model)
    4. all of it under one lock file, so two starters cannot race

## Why `lms server start` may run twice

Measured, and written down in STATUS.md: from a cold start the first call prints "Timed out
waiting for LM Studio daemon to start" even when nothing is wrong, and the second call
succeeds at once. So one failed call is not the server being unavailable. Without the retry
this tool would report "server down" on most cold mornings, which is the exact situation it
exists for. The retry is exactly one, not a loop: a server that fails twice has a real
problem, and a loop with nobody home would only hide it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import Settings

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
    """

    server_up: bool
    server_started_by_us: bool
    model: str
    model_loaded: bool
    # A deliberate decision not to load ("RAM too low"). Not a failure of the tool.
    refused: str
    # Something went wrong ("lms.exe not found", "server never answered", the loader's
    # last_error). Empty when everything went to plan.
    error: str
    vram_free_mib: int | None
    ram_free_mib: int | None
    seconds: float

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


class Clock(Protocol):
    def monotonic(self) -> float: ...
    def wall(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class MemoryReader(Protocol):
    def free_ram_mib(self) -> int | None: ...


class SystemClock:
    """Real time. A class so a test can substitute a clock whose `sleep` just moves a
    counter forward, making a 30 second wait take no time at all."""

    def monotonic(self) -> float:
        # For measuring durations: never jumps backwards when the PC's clock is adjusted.
        return time.monotonic()

    def wall(self) -> float:
        # For comparing with file modification times, which are wall-clock stamps.
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


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


class LockFile:
    """A lock made of a file: whoever creates it owns the run.

    Creating a file with "fail if it already exists" is a single step the operating system
    performs atomically (all-or-nothing, with no gap another process can slip into), which
    is what makes it a safe lock between two separate processes.

    A lock older than `stale_after_s` is taken over. Without that, one starter killed by a
    power cut would leave the file behind and every later morning would report "another
    starter is running" forever, with nobody home to delete it.
    """

    def __init__(self, path: Path, clock: Clock, stale_after_s: float) -> None:
        self._path = path
        self._clock = clock
        self._stale_after_s = stale_after_s
        self._held = False

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self) -> bool:
        """Take the lock. False means a fresh lock is held by someone else.

            no file                  -> create it            -> True
            file, 30 s old           -> leave it             -> False
            file, 2 hours old        -> delete, create anew  -> True
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Two attempts: the second only happens after clearing a stale file. Two and not a
        # loop, so two starters both clearing the same stale file cannot spin forever; the
        # loser of the second race simply reports the lock as held.
        for _ in range(2):
            if self._create():
                self._held = True
                return True
            if not self._is_stale():
                return False
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass  # someone else cleared it first; just try to create again
            except OSError:
                return False
        return False

    def release(self) -> None:
        """Remove the lock if this object took it. Safe to call twice.

        Only removes a lock we hold: removing someone else's would let a third starter in
        while theirs is still running, which is the race the lock exists to prevent.
        """
        if not self._held:
            return
        self._held = False
        try:
            self._path.unlink()
        except OSError:
            pass  # already gone; the goal (no lock left) is met

    def _create(self) -> bool:
        try:
            # O_EXCL = "fail if it exists", the atomic part.
            fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        except OSError:
            # Unwritable folder etc. Treated as "cannot take the lock", which refuses the
            # run: starting without the lock would risk the race it prevents.
            return False
        with os.fdopen(fd, "w") as handle:
            handle.write(f"pid {os.getpid()}\n")
        return True

    def _is_stale(self) -> bool:
        try:
            age_s = self._clock.wall() - self._path.stat().st_mtime
        except OSError:
            return True  # vanished while we looked: nothing holds it
        return age_s > self._stale_after_s


# ─────────────────────────── the starter ───────────────────────────


class ModelServerStarter:
    """Brings the model server up and the model in, safely. See the module docstring."""

    # Gap between "is it answering yet" checks. Longer than the probe's 2 s cache so every
    # check is a fresh answer, and long enough not to hammer a server that is booting.
    _POLL_INTERVAL_S = 3.0

    def __init__(self, settings: Settings, probe: Any, command: ServerCommand, loader: Any,
                 budget: Any, memory: MemoryReader, lock: LockFile, clock: Clock) -> None:
        self._settings = settings
        self._probe = probe      # needs is_up_now() -> bool
        self._command = command
        self._loader = loader    # needs ensure(key) -> bool and .last_error
        self._budget = budget    # needs free_mib() -> int | None  (None = unreadable)
        self._memory = memory
        self._lock = lock
        self._clock = clock

    def ensure_ready(self, model_key: str | None = None, load_model: bool = True) -> StartResult:
        """Run the whole sequence once and return what happened. Does not raise for
        expected failures; those come back as `refused` or `error`."""
        model = model_key or self._settings.model
        began = self._clock.monotonic()

        if not self._lock.acquire():
            return self._finish(
                began, model, server_up=self._probe.is_up_now(),
                refused=f"another starter holds the lock at {self._lock.path}; "
                        f"started nothing",
            )
        # `finally` so the lock is removed even when something below raises. A leaked lock
        # would block every later run until it aged out ten minutes later.
        try:
            return self._run(began, model, load_model)
        finally:
            self._lock.release()

    # ── steps ──

    def _run(self, began: float, model: str, load_model: bool) -> StartResult:
        started_by_us = False
        if not self._probe.is_up_now():
            error = self._start_server()
            if error:
                return self._finish(began, model, server_up=False, error=error)
            # It was down when we arrived and answers now, so our commands did it.
            started_by_us = True

        if not load_model:
            return self._finish(began, model, server_up=True, started_by_us=started_by_us)

        return self._load(began, model, started_by_us)

    def _start_server(self) -> str:
        """Start the server. Returns "" once it answers, or a plain error sentence."""
        timeout = self._settings.server_start_cmd_timeout_s
        last = self._command.start(timeout)
        if not last.ran:
            # Nothing to retry: a missing file stays missing.
            return f"could not run lms: {last.text}"

        if not self._probe.is_up_now():
            # The measured cold-start quirk (module docstring): try exactly once more.
            last = self._command.start(timeout)
            if not last.ran:
                return f"could not run lms: {last.text}"

        # Bounded wait. The deadline is computed once, so the total wait cannot grow no
        # matter how the loop behaves; without it a server that never comes up would
        # block the whole working day's first step.
        wait_s = self._settings.server_start_wait_s
        deadline = self._clock.monotonic() + wait_s
        while True:
            if self._probe.is_up_now():
                return ""
            if self._clock.monotonic() >= deadline:
                break
            self._clock.sleep(self._POLL_INTERVAL_S)

        detail = "timed out" if last.timed_out else f"exit code {last.exit_code}"
        output = f" Last output: {last.text}" if last.text else ""
        return (f"the model server did not answer within {wait_s:.0f} seconds of "
                f"`lms server start` (last call: {detail}).{output}")

    def _load(self, began: float, model: str, started_by_us: bool) -> StartResult:
        ram = self._memory.free_ram_mib()
        floor = self._settings.min_free_ram_mib
        # Unknown RAM is allowed through, like the monitor: psutil is a declared
        # dependency, so None means a freak read failure, and RAM is a coarse guard. The
        # GPU check below is the strict one.
        if ram is not None and ram < floor:
            return self._finish(
                began, model, server_up=True, started_by_us=started_by_us, ram=ram,
                refused=f"free RAM is {ram} MiB, below the {floor} MiB floor; "
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
            return self._finish(
                began, model, server_up=True, started_by_us=started_by_us, ram=ram,
                refused="the GPU memory could not be read, so the model was not loaded "
                        "(unattended runs do not load blind)",
            )

        loaded = self._loader.ensure(model)
        error = "" if loaded else (self._loader.last_error or f"could not load {model}")
        return self._finish(
            began, model, server_up=True, started_by_us=started_by_us,
            loaded=loaded, error=error, ram=ram, vram=vram,
        )

    def _finish(self, began: float, model: str, *, server_up: bool,
                started_by_us: bool = False, loaded: bool = False, refused: str = "",
                error: str = "", ram: int | None = None,
                vram: int | None = None) -> StartResult:
        return StartResult(
            server_up=server_up, server_started_by_us=started_by_us, model=model,
            model_loaded=loaded, refused=refused, error=error,
            vram_free_mib=vram, ram_free_mib=ram,
            seconds=round(self._clock.monotonic() - began, 1),
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
        second = f"{result.model} is loaded."
    else:
        second = f"{result.model} is not loaded."
    return f"{first} {second} Took {result.seconds} seconds."


def main(argv: Sequence[str] | None = None, starter: ModelServerStarter | None = None,
         out: Callable[[str], None] = print) -> int:
    """Entry point. Returns the exit code: 0 only if the server is up and (unless
    --no-load) the model is loaded, otherwise 1.

    `starter` is injectable so a test runs this with a fake and never builds the real
    object graph.
    """
    parser = argparse.ArgumentParser(
        prog="python -m local_llm.server_start",
        description="Make sure the local model server is up and its model is loaded.",
    )
    parser.add_argument("--model", default=None, help="model key (default: the configured one)")
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
