"""A lock made of a file, so that separate processes can take turns.

## Why a file

A lock inside a program (`threading.Lock`) is seen only by the threads of that one
process. This toolkit is run as several processes at once: each agent gets its own copy of
the MCP server, the unattended starter is another, and they all share one model server and
one graphics card. Two of them deciding at the same moment to unload and load a model is
two models asked onto a card that holds one. The only thing all of them can see is the
disk, so the lock is a file there.

Creating a file with "fail if it already exists" is a single step the operating system
performs atomically (all or nothing, with no gap another process can slip into). Whoever
creates the file holds the lock; everyone else finds it there and waits or gives up.

## The two ways a lock is left behind, and what is done about each

A process that dies while holding the lock (a power cut, a killed window) never removes
the file. Without an answer to that, every later caller would wait for a holder that no
longer exists.

* **Age.** A lock older than `stale_after_s` is taken over, whoever wrote it.
* **A dead holder.** The file records the holder's process number. When a function that
  can tell whether a process is running is passed in (`is_alive`), a lock whose holder is
  gone is taken over at once, without waiting for it to age.

## One class, two users

`server_start.py` uses it so two starters cannot race, and `loader.py` uses it around
every model load. It lives here so there is one copy of the takeover rules.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Clock(Protocol):
    """The three things about time the lock and the starter need. A Protocol is an
    interface by shape: any class with these methods fits, so a test passes a clock whose
    `sleep` only moves a counter and a three minute wait costs nothing."""

    def monotonic(self) -> float: ...
    def wall(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """Real time."""

    def monotonic(self) -> float:
        # For measuring durations: never jumps backwards when the PC's clock is adjusted.
        return time.monotonic()

    def wall(self) -> float:
        # For comparing with file modification times, which are wall-clock stamps.
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def process_is_running(pid: int) -> bool:
    """Whether a process with this number exists now, via psutil.

        4321 is running            ->  True
        4321 ended an hour ago     ->  False
        psutil cannot be imported  ->  True   ("cannot tell" must not break a held lock)

    The answer True on doubt is the safe one: a lock wrongly thought held costs a wait,
    a lock wrongly taken over costs two loads at once.
    """
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:
        return True


@dataclass(frozen=True)
class LockHolder:
    """What a lock file says about who took it. "Frozen": it cannot be changed once made.

        "pid 4321\\nsince 1791921130.0\\nnote loading qwen/qwen3-14b\\n"
          ->  LockHolder(pid=4321, since=1791921130.0, note="loading qwen/qwen3-14b")

    Any part that cannot be read is None or "", never a guess.
    """

    pid: int | None
    since: float | None
    note: str


class LockFile:
    """A lock made of a file: whoever creates it owns the run. See the module docstring."""

    # Gap between attempts while waiting for a held lock. Short, because the wait ends the
    # moment the holder finishes and nobody should sit idle for long after that.
    _POLL_INTERVAL_S = 0.5

    def __init__(self, path: Path, clock: Clock, stale_after_s: float,
                 is_alive: Callable[[int], bool] | None = None) -> None:
        self._path = path
        self._clock = clock
        self._stale_after_s = stale_after_s
        # Optional. Without it only age makes a lock stale.
        self._is_alive = is_alive
        self._held = False

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self, note: str = "") -> bool:
        """Take the lock. False means a fresh lock is held by someone else.

            no file                           -> create it            -> True
            file, 30 s old, holder running    -> leave it             -> False
            file, 2 hours old                 -> delete, create anew  -> True
            file, 30 s old, holder is gone    -> delete, create anew  -> True

        `note` is written into the file ("loading qwen/qwen3-14b") so that a process
        that has to give up can say what the holder was doing.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Two attempts: the second only happens after clearing a stale file. Two and not a
        # loop, so two processes both clearing the same stale file cannot spin forever; the
        # loser of the second race simply reports the lock as held.
        for _ in range(2):
            if self._create(note):
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

    def wait_for(self, timeout_s: float, note: str = "") -> bool:
        """Keep trying to take the lock for up to `timeout_s`. True once it is ours.

            free at once                              ->  True, no waiting
            held, released 14 seconds in              ->  True after about 14 seconds
            held for longer than timeout_s            ->  False after timeout_s

        The deadline is worked out once, so the wait cannot grow whatever the loop does.
        What breaks without a bound: a holder that hangs would hang every other process
        with it, and with nobody home that is a day lost in silence.
        """
        deadline = self._clock.monotonic() + timeout_s
        while True:
            if self.acquire(note):
                return True
            left = deadline - self._clock.monotonic()
            if left <= 0:
                return False
            self._clock.sleep(min(self._POLL_INTERVAL_S, left))

    def release(self) -> None:
        """Remove the lock if this object took it. Safe to call twice.

        Only removes a lock we hold: removing someone else's would let a third process in
        while theirs is still running, which is the race the lock exists to prevent.
        """
        if not self._held:
            return
        self._held = False
        try:
            self._path.unlink()
        except OSError:
            pass  # already gone; the goal (no lock left) is met

    def holder(self) -> LockHolder:
        """Who the file says holds the lock. Every field empty when it cannot be read."""
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError:
            return LockHolder(pid=None, since=None, note="")
        # "pid 4321\nsince 1791921130.0\nnote loading x"
        #   ->  {"pid": "4321", "since": "1791921130.0", "note": "loading x"}
        fields: dict[str, str] = {}
        for line in text.splitlines():
            name, _, value = line.partition(" ")
            fields[name] = value.strip()
        try:
            pid = int(fields["pid"])
        except (KeyError, ValueError):
            pid = None
        try:
            since = float(fields["since"])
        except (KeyError, ValueError):
            since = None
        return LockHolder(pid=pid, since=since, note=fields.get("note", ""))

    def describe_holder(self) -> str:
        """Who holds the lock and since when, in words for a refusal message.

            LockHolder(pid=4321, since=<8:52:10 pm>, note="loading qwen/qwen3-14b"),
            read 41 seconds later
              ->  "process 4321, loading qwen/qwen3-14b, since 8:52:10 pm (41 seconds ago)"
            a file that cannot be read  ->  "an unknown process"
        """
        found = self.holder()
        parts = ["an unknown process" if found.pid is None else f"process {found.pid}"]
        if found.note:
            parts.append(found.note)
        if found.since is not None:
            # %I is the hour on a 12-hour clock with a leading zero ("08"), %p is AM or PM.
            #   "08:52:10 PM"  ->  "8:52:10 pm"
            stamp = time.strftime("%I:%M:%S %p", time.localtime(found.since))
            stamp = stamp.lstrip("0").lower()
            ago = max(0, int(self._clock.wall() - found.since))
            parts.append(f"since {stamp} ({ago} seconds ago)")
        return ", ".join(parts)

    def _create(self, note: str) -> bool:
        try:
            # O_EXCL = "fail if it exists", the atomic part.
            fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        except OSError:
            # Unwritable folder, or on Windows a file another process is deleting at this
            # instant. Treated as "cannot take the lock": going ahead without it would
            # risk the race it prevents.
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"pid {os.getpid()}\nsince {self._clock.wall()}\n")
            if note:
                # One line only, so a note cannot be mistaken for another field.
                handle.write(f"note {' '.join(note.split())}\n")
        return True

    def _is_stale(self) -> bool:
        """Whether the lock on disk was left behind and may be taken over.

            10 minutes is the limit, the file is 2 hours old      ->  True
            the file is 30 seconds old, its holder is not running  ->  True
            the file is 30 seconds old, its holder is running      ->  False
            the file is 30 seconds old, no process number in it    ->  False
        """
        try:
            age_s = self._clock.wall() - self._path.stat().st_mtime
        except OSError:
            return True  # vanished while we looked: nothing holds it
        if age_s > self._stale_after_s:
            return True
        if self._is_alive is None:
            return False
        pid = self.holder().pid
        # No number yet means the holder has created the file and is still writing it,
        # which is a live holder, not a dead one.
        return pid is not None and not self._is_alive(pid)
