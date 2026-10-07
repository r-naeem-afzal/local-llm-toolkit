"""Host and usage monitoring — what the GPU is doing, which models are resident, and how
much Claude plan usage has been spent in the current 5-hour window.

This answers the question the whole toolkit exists to serve: *is the local machine
actually doing the work, and what is the paid plan being spent on instead?* Without it
the delegation story is unverifiable — you would be trusting that work moved off the
plan rather than seeing it.

Five independent probe classes, composed by `SystemMonitor`:

    GpuProbe            VRAM, utilisation, temperature via NVML
    HostProbe           CPU, RAM, and RAM held by the inference processes
    ModelRegistry       resident and installed models via the `lms` CLI
    ModelServerProbe    is the OpenAI-compatible endpoint answering
    ClaudeUsageReader   plan spend, parsed from local session transcripts

Each is separate because each can fail on its own and for its own reason — a missing
driver library, an absent CLI, a sleeping daemon. Composing them means one failure
degrades one panel instead of blanking the dashboard.

Everything here is read-only and best-effort by design. A monitoring module that raised
because the GPU driver hiccuped would take down the dashboard it exists to feed, so every
probe reports "unavailable" instead of throwing.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import Settings

# The plan allowance resets on a rolling 5-hour window, so this is the only window size
# that answers "how much have I got left right now".
USAGE_WINDOW_HOURS = 5


# ─────────────────────────── shared transcript access ───────────────────────────


def parse_iso_timestamp(value: Any) -> datetime | None:
    """Parse the transcript's ISO timestamps into timezone-aware datetimes.

    Transcripts use a trailing 'Z' for UTC, which `fromisoformat` did not accept before
    Python 3.11, so it is rewritten to the '+00:00' form that always works.

        "2026-09-04T19:22:31.863Z"  ->  datetime(2026,9,4,19,22,31,863000, tz=utc)

    Every caller compares the result against an aware "now", and comparing an aware to a
    naive datetime raises TypeError — so a timestamp that cannot be made aware is returned
    as None and dropped by the caller, rather than being allowed to crash a window
    calculation with a type error far from the bad data.

    A module-level function rather than a method because two unrelated readers — plan
    usage and agent activity — both need it, and neither should have to reach into the
    other's class to get it.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class TranscriptLocator:
    """Finds the Claude Code transcript files worth reading.

    Claude Code appends one JSON object per line to
    `~/.claude/projects/<project-dir>/<session-id>.jsonl`. Everything this toolkit knows
    about plan spend and agent activity comes from those files, so *where they are* and
    *which ones are recent* is a concern two different readers share.

    It is its own class for exactly that reason: plan usage and live agent activity both
    need this and nothing else about each other. Without it, the second reader would
    duplicate the home-directory resolution and the modification-time pre-filter, and the
    two copies would drift — the usual outcome being one of them silently reading the
    wrong directory after a settings change.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def projects_dir(self) -> Path:
        """The directory holding one subdirectory per project.

        Overridable through settings so a test can point at a fixture tree; otherwise the
        fixed location Claude Code writes to.
        """
        if self._settings.claude_projects_dir:
            return self._settings.claude_projects_dir
        return Path(os.path.expanduser("~")) / ".claude" / "projects"

    def recent_transcripts(self, since: datetime,
                           project: str | None = None) -> Iterator[tuple[str, Path]]:
        """Yield (project directory name, transcript path) for files touched since `since`.

            since = now - 5h
              ->  ("c--Work-Personal-Ideas-local-llm", .../ded195c6-….jsonl)

        The modification-time check is a deliberate cheap pre-filter: a file untouched
        since before the window opened cannot hold a record inside it, so it is skipped
        without reading a single byte. That matters because transcripts are megabytes each
        and there can be dozens of them — without this the dashboard's own polling becomes
        the most expensive process on the machine.

        Yields nothing at all if the projects directory is absent; callers report that as
        an error state, since "no transcripts" and "no spend" must not look the same.
        """
        root = self.projects_dir()
        if not root.exists():
            return

        if project:
            directories = [root / project]
        else:
            # Every project by default, because the plan allowance is per account, not per
            # project. Filtering to one would under-report what has been spent, which is
            # the opposite of useful when the question is "how much is left".
            try:
                directories = [entry for entry in root.iterdir() if entry.is_dir()]
            except OSError:
                # `exists()` passing does not mean the path can be listed: a configured
                # `claude_projects_dir` pointing at a *file* raises NotADirectoryError
                # here, and an unreadable directory raises PermissionError. Unguarded,
                # that exception escapes the reader and turns the endpoint into a 500 —
                # so a monitoring feature would take down the dashboard it feeds.
                return

        for directory in directories:
            if not directory.is_dir():
                continue
            try:
                transcripts = list(directory.glob("*.jsonl"))
            except OSError:
                # Same reasoning as above, per directory: one unreadable project must not
                # hide every other project's sessions.
                continue
            for transcript in transcripts:
                try:
                    modified = datetime.fromtimestamp(transcript.stat().st_mtime, timezone.utc)
                except OSError:
                    # The file vanished between the glob and the stat — a session ending,
                    # or a cleanup pass. Not an error worth surfacing.
                    continue
                if modified < since:
                    continue
                yield directory.name, transcript


# ─────────────────────────── GPU ───────────────────────────


@dataclass
class GpuInfo:
    available: bool
    name: str = ""
    total_mib: int = 0
    used_mib: int = 0
    utilisation_pct: int = 0
    temperature_c: int | None = None
    error: str = ""

    @property
    def free_mib(self) -> int:
        return max(0, self.total_mib - self.used_mib)

    @property
    def used_pct(self) -> float:
        # Guards against dividing by zero when the probe failed and total is still 0.
        # Without it a failed GPU read would raise here instead of rendering an empty
        # gauge, turning a degraded panel into a broken page.
        return round(self.used_mib / self.total_mib * 100, 1) if self.total_mib else 0.0

    def as_dict(self) -> dict[str, Any]:
        # asdict does not include @property values, so the two derived numbers are added
        # explicitly rather than being re-derived in TypeScript, where they could drift
        # from the definitions used here.
        return {**asdict(self), "free_mib": self.free_mib, "used_pct": self.used_pct}


class GpuProbe:
    """Reads VRAM, utilisation and temperature from the NVIDIA driver.

    Uses NVML (the NVIDIA Management Library) — the same C library `nvidia-smi` is a
    front-end for. Calling the library directly avoids spawning a process and parsing its
    text output on every dashboard poll.

    ## Why there is no per-process VRAM here

    It was attempted and it does not work on this hardware. NVML exposes
    `nvmlDeviceGetComputeRunningProcesses`, which lists the running compute processes, but
    on a consumer GeForce card under Windows' WDDM display driver model every process
    comes back with `usedGpuMemory = None`. WDDM, not the NVIDIA driver, owns video memory
    allocation, so the driver genuinely cannot attribute it per process. This is a
    platform limitation, not a missing call or a permissions problem, and
    `nvidia-smi --query-compute-apps=used_memory` returns `[N/A]` for the same reason.

    Per-model memory is therefore attributed from `lms ps` instead — see `ModelRegistry`,
    which reports each resident model's size directly. That is better information anyway:
    it names the model rather than a process id.
    """

    def read(self) -> GpuInfo:
        try:
            import pynvml
        except ImportError:
            return GpuInfo(
                available=False, error="pynvml not installed (pip install nvidia-ml-py)"
            )

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

            return GpuInfo(
                available=True,
                name=self._device_name(pynvml, handle),
                # NVML reports bytes; everything above this layer thinks in MiB, because
                # that is the unit model sizes and VRAM budgets are discussed in.
                #   17091915776 bytes  ->  16303 MiB
                total_mib=memory.total // 1024 // 1024,
                used_mib=memory.used // 1024 // 1024,
                utilisation_pct=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                temperature_c=self._temperature(pynvml, handle),
            )
        except Exception as exc:
            return GpuInfo(available=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            # Always hand the NVML handle back. Leaking it across thousands of dashboard
            # polls would slowly exhaust driver-side resources.
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    @staticmethod
    def _device_name(pynvml: Any, handle: Any) -> str:
        name = pynvml.nvmlDeviceGetName(handle)
        # Older pynvml releases return bytes here, newer ones return str. Normalising
        # means the dashboard never renders a stray b'...' prefix.
        return name.decode("utf-8", "replace") if isinstance(name, bytes) else name

    @staticmethod
    def _temperature(pynvml: Any, handle: Any) -> int | None:
        try:
            return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            # Temperature is a nice-to-have. Some cards and driver versions refuse it,
            # and that must not cost us the memory numbers, which are the point.
            return None


# ─────────────────────────── host ───────────────────────────


@dataclass
class HostInfo:
    available: bool
    cpu_pct: float = 0.0
    ram_total_mib: int = 0
    ram_used_mib: int = 0
    inference_rss_mib: int = 0
    inference_processes: int = 0
    error: str = ""


class HostProbe:
    """CPU and RAM for the machine, plus the system RAM held by inference processes.

    System RAM matters even though the model runs on the GPU: the runtime memory-maps the
    model file, so a 9 GB model shows up in host RAM as well. A machine that starts
    swapping presents as the model having mysteriously slowed down, and this is the panel
    that explains it.
    """

    # Name fragments of the processes that actually do inference. Bionic is LM Studio
    # renamed, so both names are matched; `llama` catches the llama.cpp server process the
    # app spawns per loaded model.
    _PROCESS_NAMES = ("bionic", "lm studio", "lmstudio", "llama")

    # Walking every process on the machine costs about 260 ms — by far the most expensive
    # part of a host reading, and far too slow to repeat on a 1-second telemetry poll.
    # The number it produces barely moves: a loaded model's resident memory is stable for
    # minutes at a time. So the scan is cached and CPU/RAM, which are nearly free and do
    # change continuously, are read fresh every time.
    _PROCESS_SCAN_TTL_S = 10.0

    def __init__(self) -> None:
        self._process_cache: tuple[float, int, int] | None = None

    def read(self) -> HostInfo:
        try:
            import psutil
        except ImportError:
            return HostInfo(available=False, error="psutil not installed")

        try:
            virtual = psutil.virtual_memory()
            rss_bytes, count = self._cached_inference_memory(psutil)

            return HostInfo(
                available=True,
                # interval=None returns the CPU percentage since the *previous* call
                # rather than blocking to sample. Blocking for even 0.1 s would stall
                # every dashboard poll; the trade is that the first reading after startup
                # is meaningless.
                cpu_pct=psutil.cpu_percent(interval=None),
                ram_total_mib=virtual.total // 1024 // 1024,
                ram_used_mib=virtual.used // 1024 // 1024,
                inference_rss_mib=rss_bytes // 1024 // 1024,
                inference_processes=count,
            )
        except Exception as exc:
            return HostInfo(available=False, error=f"{type(exc).__name__}: {exc}")

    def _cached_inference_memory(self, psutil: Any) -> tuple[int, int]:
        """The process scan, re-run at most every `_PROCESS_SCAN_TTL_S`.

            first call            -> full scan, ~260 ms
            calls within 10 s     -> cached tuple, ~0 ms
        """
        now = time.monotonic()
        if self._process_cache is not None and now - self._process_cache[0] < self._PROCESS_SCAN_TTL_S:
            return self._process_cache[1], self._process_cache[2]
        rss_bytes, count = self._inference_memory(psutil)
        self._process_cache = (now, rss_bytes, count)
        return rss_bytes, count

    def _inference_memory(self, psutil: Any) -> tuple[int, int]:
        """Total RSS and process count for the inference processes.

        RSS ("resident set size") is the physical RAM a process currently holds, as
        opposed to virtual address space it has merely reserved. RSS is the number that
        competes with other programs for real memory, which is why it is the one reported.
        """
        rss_bytes = 0
        count = 0
        # Passing an attribute list makes psutil fetch only those fields in one pass.
        # Touching each Process object afterwards instead would cost a syscall per
        # attribute per process, which on a Windows box with hundreds of processes is the
        # difference between a fast poll and a visibly slow one.
        for process in psutil.process_iter(["name", "memory_info"]):
            try:
                name = (process.info["name"] or "").lower()
                if not any(fragment in name for fragment in self._PROCESS_NAMES):
                    continue
                memory = process.info["memory_info"]
                if memory:
                    rss_bytes += memory.rss
                    count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # A process can exit between being listed and being read, and Windows
                # denies access to some system processes. Both are normal during a scan,
                # so skip the process rather than failing the whole probe.
                continue
        return rss_bytes, count


# ─────────────────────────── models ───────────────────────────


@dataclass
class LoadedModel:
    key: str
    # The server's handle for *this instance*, which is not the same as the model key.
    # Loading a model twice gives the second copy an identifier like
    # "qwen/qwen2.5-coder-14b:2" while both report the same `key` — so anything that
    # deduplicates by key cannot see a duplicate load at all. That is exactly how two 14B
    # models ended up sharing a card that fits one, unnoticed.
    identifier: str
    display_name: str
    size_mib: int
    context_length: int
    max_context_length: int
    status: str
    quantisation: str
    ttl_remaining_s: int | None
    queued: int
    parallel: int


class LmsCommandRunner:
    """Runs the `lms` CLI and parses its JSON output, with a short cache.

    Its own class because "shell out to a CLI and survive the ways that goes wrong" is a
    distinct concern from interpreting the results, and both `ModelRegistry` and any
    future model-control code need it.
    """

    # Calling out to `lms` spawns a process and talks to the Bionic daemon, taking a
    # noticeable fraction of a second. The dashboard polls, so without a cache every poll
    # would pay that cost to learn nothing new, and the UI would feel sluggish.
    _CACHE_TTL_S = 3.0

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: dict[str, tuple[float, Any]] = {}

    def _executable(self) -> Path:
        if self._settings.lms_path:
            return self._settings.lms_path
        # Bionic keeps the LM Studio directory layout unchanged, so the CLI is where LM
        # Studio always put it. Documented because looking for a "Bionic" folder under the
        # home directory finds nothing and wastes real time.
        return Path(os.path.expanduser("~")) / ".lmstudio" / "bin" / "lms.exe"

    def executable_path(self) -> Path:
        """Where the `lms` CLI is expected to be (it may not exist there).

        Public so other classes that must run `lms` themselves (the server starter runs
        `lms server start`, which has no JSON form) find it by the same rule instead of
        copying the home-directory logic and drifting from it.
        """
        return self._executable()

    def run(self, *args: str) -> Any:
        """Run `lms <args> --json` and return the parsed result, or None on any failure.

        None rather than an exception because the server being down is a *normal state
        that we want displayed*. The dashboard shows "unknown" and keeps working.
        """
        key = " ".join(args)
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached and now - cached[0] < self._CACHE_TTL_S:
            return cached[1]

        executable = self._executable()
        if not executable.exists():
            return None

        try:
            completed = subprocess.run(
                [str(executable), *args, "--json"],
                capture_output=True,
                text=True,
                # A hung daemon must not hang the dashboard. `lms` talks to the Bionic app
                # over a socket, and if the app is starting that handshake can stall;
                # without this timeout the whole API request would block indefinitely.
                timeout=15,
                check=False,
            )
            if completed.returncode != 0:
                return None
            parsed = self._parse(completed.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
            return None

        if parsed is not None:
            self._cache[key] = (now, parsed)
        return parsed

    @staticmethod
    def _parse(text: str) -> Any:
        """Find the JSON payload in output that may be preceded by status chatter.

        The CLI prints progress messages before the payload, so the whole of stdout is not
        valid JSON. Slicing from the first bracket is what makes this robust.

            'Waking up LM Studio service...\\n[{"type":"llm"}]'  ->  [{"type": "llm"}]
        """
        candidates = [index for index in (text.find("["), text.find("{")) if index != -1]
        if not candidates:
            return None
        return json.loads(text[min(candidates):])


class ModelRegistry:
    """What is resident in VRAM now, and what is available on disk."""

    def __init__(self, runner: LmsCommandRunner) -> None:
        self._runner = runner

    def loaded(self) -> list[LoadedModel]:
        """Models currently resident in VRAM, from `lms ps`.

        This is the per-model memory attribution NVML could not provide. Summing
        `size_mib` gives how much VRAM the models account for; the difference against
        `GpuInfo.used_mib` is everything else — desktop compositor, browser, and the KV
        cache for the loaded context.

            {"modelKey":"qwen/qwen2.5-coder-14b","sizeBytes":8988288458,"status":"idle",
             "ttlMs":3600000,"lastUsedTime":1788550228289,"contextLength":30464}
              ->  LoadedModel(key="qwen/qwen2.5-coder-14b", size_mib=8571,
                              status="idle", ttl_remaining_s=3480, context_length=30464)
        """
        raw = self._runner.run("ps")
        if not isinstance(raw, list):
            return []
        return [self._to_model(entry) for entry in raw if isinstance(entry, dict)]

    def _to_model(self, entry: dict[str, Any]) -> LoadedModel:
        quantisation = entry.get("quantization") or {}
        return LoadedModel(
            key=entry.get("modelKey") or entry.get("identifier") or "?",
            identifier=entry.get("identifier") or entry.get("modelKey") or "?",
            display_name=entry.get("displayName") or "",
            size_mib=int(entry.get("sizeBytes") or 0) // 1024 // 1024,
            context_length=int(entry.get("contextLength") or 0),
            max_context_length=int(entry.get("maxContextLength") or 0),
            status=entry.get("status") or "unknown",
            quantisation=quantisation.get("name") if isinstance(quantisation, dict) else "",
            ttl_remaining_s=self._ttl_remaining(entry),
            queued=int(entry.get("queued") or 0),
            parallel=int(entry.get("parallel") or 0),
        )

    @staticmethod
    def _ttl_remaining(entry: dict[str, Any]) -> int | None:
        """Seconds before an idle model is unloaded to free VRAM.

        Surfacing this explains an otherwise baffling observation: a model that was loaded
        a moment ago is suddenly gone and the next call pays the ~18 s load again. Only
        one 14B model fits in 16 GB at a time here, so TTL eviction is the mechanism by
        which models swap — it is normal behaviour, not a fault.

            ttlMs 3600000, lastUsedTime 40s ago  ->  3560
        """
        ttl_ms = entry.get("ttlMs")
        last_used_ms = entry.get("lastUsedTime")
        if not isinstance(ttl_ms, (int, float)) or not isinstance(last_used_ms, (int, float)):
            return None
        # lastUsedTime is a Unix timestamp in milliseconds; time.time() is in seconds.
        elapsed_s = time.time() - last_used_ms / 1000
        return max(0, int(ttl_ms / 1000 - elapsed_s))

    def installed(self) -> list[dict[str, Any]]:
        """Every model on disk, from `lms ls` — the menu of what could be loaded.

        Left as plain dicts rather than a dataclass: this feeds a "what is available" list
        and nothing branches on its fields, so imposing a schema would be maintenance cost
        with no payoff.
        """
        raw = self._runner.run("ls")
        if not isinstance(raw, list):
            return []
        return [
            {
                "key": entry.get("modelKey", "?"),
                "type": entry.get("type", "llm"),
                "display_name": entry.get("displayName", ""),
                "size_mib": int(entry.get("sizeBytes") or 0) // 1024 // 1024,
                "params": entry.get("paramsString", ""),
                "architecture": entry.get("architecture", ""),
                "max_context_length": entry.get("maxContextLength"),
            }
            for entry in raw
            if isinstance(entry, dict)
        ]


class ModelServerProbe:
    """Whether the local OpenAI-compatible endpoint is answering.

    The answer is cached for a couple of seconds. This probe used to be the single most
    expensive thing in a dashboard poll — 2 seconds of a 2.5 second `snapshot()` — because
    a URL of "localhost" resolves to IPv6 ::1 first and the model server binds only IPv4,
    so every check waited for a connection attempt to time out before falling back. That
    specific cause is fixed (see `Settings.url`), but the caching stays: whether a server
    is up cannot meaningfully change several times a second, and re-asking on every poll
    made the dashboard's own monitoring the slowest thing on the machine.
    """

    # Short enough that starting the model server is reflected almost immediately, long
    # enough that a 1-second telemetry poll does not open a socket every time.
    _CACHE_TTL_S = 2.0

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cached: tuple[float, bool] | None = None

    def is_up(self) -> bool:
        now = time.monotonic()
        if self._cached is not None and now - self._cached[0] < self._CACHE_TTL_S:
            return self._cached[1]
        result = self._probe()
        self._cached = (now, result)
        return result

    def is_up_now(self) -> bool:
        """Ask the server right now, ignoring the cache.

        `is_up` may return an answer up to 2 seconds old. A caller that has just run a
        command to change the answer (starting the server) must not be told the old one:
        it would conclude the command failed and run it again, or give up, when the server
        came up a moment after the cached "no".
        """
        result = self._probe()
        self._cached = (time.monotonic(), result)
        return result

    def _probe(self) -> bool:
        # A plain blocking request rather than the async client: this is called from
        # synchronous contexts (a CLI status command, a startup check) where dragging in
        # an event loop to ask one yes/no question is not worth it.
        try:
            # 3 s so a down server is reported *quickly*. The expected failure is a
            # refused connection, which returns immediately; the timeout only matters for
            # the rarer case of a socket that accepts and then never replies.
            response = httpx.get(f"{self._settings.url}/models", timeout=3.0)
            return response.status_code == 200
        except Exception:
            return False


# ─────────────────────── Claude plan usage ───────────────────────


@dataclass
class AgentUsage:
    """Usage attributed to one model, within one bucket (main loop or subagent)."""

    model: str
    messages: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens + self.output_tokens
            + self.cache_read_tokens + self.cache_write_tokens
        )

    def add(self, record: UsageRecord) -> None:
        self.messages += 1
        self.input_tokens += record.input_tokens
        self.output_tokens += record.output_tokens
        self.cache_read_tokens += record.cache_read_tokens
        self.cache_write_tokens += record.cache_write_tokens

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "total_tokens": self.total_tokens}


@dataclass(frozen=True)
class UsageRecord:
    """One billable assistant message, extracted from a session transcript."""

    ts: datetime
    model: str
    # True when the message was produced by a subagent rather than the main conversation
    # loop.
    #
    # **Measured 2026-09-05: on this machine this flag is never set.** Claude Code
    # 2.1.260 writes no `isSidechain` record and no transcript of its own for a subagent —
    # across every transcript in every project here, the count of sidechain records is
    # zero, including sessions that demonstrably ran subagents. Subagent tokens are billed
    # to the 5-hour window but appear in no local file.
    #
    # The flag is kept rather than deleted because other Claude Code versions and
    # entrypoints do emit it, and reading it costs nothing. What must not happen is
    # someone treating an empty `subagents` bucket as proof that no subagents ran, or as
    # proof they were free. See `agents.py`, which answers "what is running now" from the
    # `Agent` tool-call records that *are* written.
    is_subagent: bool
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


@dataclass
class ClaudeUsage:
    window_hours: int
    window_start: str
    messages: int = 0
    main_loop: dict[str, AgentUsage] = field(default_factory=dict)
    # Empty on this machine, and not because subagents are cheap — see the note on
    # `UsageRecord.is_subagent`. Subagent spend is real but absent from the transcripts,
    # so these totals are a floor on plan usage whenever agents have run, never a
    # complete figure. The dashboard says so rather than letting the number be read as
    # exact.
    subagents: dict[str, AgentUsage] = field(default_factory=dict)
    sessions: int = 0
    error: str = ""

    @property
    def subagent_messages(self) -> int:
        return sum(usage.messages for usage in self.subagents.values())

    @property
    def total_tokens(self) -> int:
        buckets = list(self.main_loop.values()) + list(self.subagents.values())
        return sum(usage.total_tokens for usage in buckets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_hours": self.window_hours,
            "window_start": self.window_start,
            "sessions": self.sessions,
            "messages": self.messages,
            "subagent_messages": self.subagent_messages,
            "total_tokens": self.total_tokens,
            "main_loop": [usage.as_dict() for usage in self.main_loop.values()],
            "subagents": [usage.as_dict() for usage in self.subagents.values()],
            "error": self.error,
        }


class TranscriptParser:
    """Extracts billable facts from Claude Code session transcripts, caching by file.

    A transcript is JSONL — one JSON object per line — mixing many record types (`user`,
    `assistant`, `attachment`, file-history snapshots and more). Only `assistant` records
    carry a `usage` block, so everything else is skipped.
    """

    def __init__(self) -> None:
        # Parsing means reading every line of a file that is already ~2 MB and grows
        # during a live session. The dashboard polls, so re-parsing on every poll would
        # burn CPU re-deriving an unchanged answer — the dashboard would become the most
        # expensive thing on the machine. Keyed by path, invalidated when size or mtime
        # moves, both of which change on any append.
        self._cache: dict[str, tuple[float, int, list[UsageRecord]]] = {}

    def parse(self, path: Path) -> list[UsageRecord]:
        stat = path.stat()
        cached = self._cache.get(str(path))
        if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            return cached[2]

        records: list[UsageRecord] = []
        # errors="replace" because a live transcript can be caught mid-write with a
        # partial multi-byte character at the end; refusing to decode would lose the
        # entire file over one broken character.
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record = self._parse_line(line)
                if record is not None:
                    records.append(record)

        self._cache[str(path)] = (stat.st_mtime, stat.st_size, records)
        return records

    def _parse_line(self, line: str) -> UsageRecord | None:
        """Turn one transcript line into a UsageRecord, or None if it is not billable.

            {"type":"assistant","timestamp":"2026-09-04T19:22:31.863Z","isSidechain":false,
             "message":{"model":"claude-opus-5","usage":{"input_tokens":2,
               "output_tokens":158,"cache_read_input_tokens":30325,
               "cache_creation_input_tokens":11024}}}
              ->  UsageRecord(ts=…, model="claude-opus-5", is_subagent=False,
                              input_tokens=2, output_tokens=158,
                              cache_read_tokens=30325, cache_write_tokens=11024)
        """
        line = line.strip()
        if not line:
            return None
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            # The final line of a live session is often half-written. Skipping it is
            # correct; the next poll will see it complete.
            return None

        if not isinstance(raw, dict):
            # Valid JSON but not an object — a bare number, string or array. `.get` on
            # those raises AttributeError, which the JSONDecodeError guard above does not
            # catch, so one odd line would break the whole file's parse.
            return None

        if raw.get("type") != "assistant":
            return None
        message = raw.get("message") or {}
        usage = message.get("usage") or {}
        if not usage:
            return None

        timestamp = self._parse_iso(raw.get("timestamp"))
        if timestamp is None:
            return None

        return UsageRecord(
            ts=timestamp,
            model=message.get("model") or "unknown",
            is_subagent=bool(raw.get("sidechain") or raw.get("isSidechain")),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            # Cache reads are billed at a large discount and cache writes at a premium, so
            # they are tracked separately rather than folded into input. The mix is
            # extreme in practice — one measured 5-hour window held 73.5M tokens of which
            # 72.2M were cache reads and only 854 were fresh input — so combining them
            # would overstate real spend by roughly fifty times.
            cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        )

    @staticmethod
    def _parse_iso(value: Any) -> datetime | None:
        """Kept as a method so existing callers keep working; the logic now lives in
        `parse_iso_timestamp`, because the agent-activity reader needs the same rule."""
        return parse_iso_timestamp(value)


class ClaudeUsageReader:
    """Sums Claude plan usage over a trailing window, split main loop vs subagent.

    Reads the transcripts Claude Code writes to
    `~/.claude/projects/<project>/<session>.jsonl`. This is why plan spend is observable
    at all without any API access: the client records every assistant message's usage
    block locally, as it happens.
    """

    def __init__(self, settings: Settings, locator: TranscriptLocator | None = None,
                 parser: TranscriptParser | None = None) -> None:
        self._settings = settings
        self._locator = locator or TranscriptLocator(settings)
        self._parser = parser or TranscriptParser()

    def read(self, window_hours: int = USAGE_WINDOW_HOURS,
             project: str | None = None) -> ClaudeUsage:
        """Usage inside the trailing window.

        `project` filters to one project directory. The default sums every project, which
        is the correct default because the 5-hour allowance is per account, not per
        project — restricting to one would under-report what has been spent.
        """
        started = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        usage = ClaudeUsage(window_hours=window_hours, window_start=started.isoformat())

        root = self._locator.projects_dir()
        if not root.exists():
            # Reported rather than returned as a zero, because "no transcripts here" and
            # "nothing has been spent" are very different answers and must not look
            # identical on the dashboard.
            usage.error = f"no Claude transcripts at {root}"
            return usage

        for _project, transcript in self._locator.recent_transcripts(started, project):
            try:
                records = self._parser.parse(transcript)
            except OSError:
                continue
            if self._accumulate(records, started, usage):
                usage.sessions += 1

        return usage

    def _accumulate(self, records: list[UsageRecord], started: datetime,
                    usage: ClaudeUsage) -> bool:
        """Fold one transcript's records into the totals. Returns whether any counted."""
        counted = 0
        for record in records:
            if record.ts < started:
                continue
            bucket = usage.subagents if record.is_subagent else usage.main_loop
            bucket.setdefault(record.model, AgentUsage(model=record.model)).add(record)
            usage.messages += 1
            counted += 1
        return counted > 0


# ─────────────────────────── composition ───────────────────────────


class SystemMonitor:
    """Composes the probes into one snapshot for the dashboard.

    A facade rather than a god object: it holds no monitoring logic of its own, only the
    decision to gather these particular readings together. Each probe is still usable on
    its own — a CLI status command needs only `ModelServerProbe`, and should not have to
    construct an NVML handle to ask one question.
    """

    def __init__(self, gpu: GpuProbe, host: HostProbe, models: ModelRegistry,
                 server: ModelServerProbe, usage: ClaudeUsageReader) -> None:
        self._gpu = gpu
        self._host = host
        self._models = models
        self._server = server
        self._usage = usage

    @property
    def usage_reader(self) -> ClaudeUsageReader:
        return self._usage

    def telemetry(self) -> dict[str, Any]:
        """The fast-moving numbers only, cheap enough to poll once a second.

        ## Why this exists separately from `snapshot`

        GPU load, VRAM and temperature are *sampled telemetry*: they change continuously
        whether or not anything else happens. Everything else in `snapshot` is
        *event-driven state* — which models are resident, whether the server is up — that
        changes only when something acts on it.

        The dashboard originally refreshed the whole snapshot only when the change-stream
        fired, and that stream fires on model-call activity. So while a machine sat idle,
        or during one long generation with no call boundary, the GPU gauges froze at
        whatever they read when the last call started. They were live only by coincidence.

        Splitting them lets the frontend poll this every second for genuinely live gauges
        while leaving the expensive inventory on the event path.

        ## What is deliberately excluded

        The two expensive probes, because including either would defeat the purpose:

        - the `lms` subprocess that lists resident models,
        - the process-table walk that attributes RAM to the inference processes (~260 ms).

        `server_up` *is* included, but only because it is cached — see `ModelServerProbe`.
        Measured cost of this method is a few milliseconds, against 2.5 seconds for a full
        snapshot before these fixes.
        """
        gpu = self._gpu.read()
        host = self._host.read()

        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "server_up": self._server.is_up(),
            "gpu": gpu.as_dict(),
            # Only the continuously-varying host fields. The inference-process figures come
            # from the cached scan and belong with the slower snapshot.
            "cpu_pct": host.cpu_pct,
            "ram_used_mib": host.ram_used_mib,
            "ram_total_mib": host.ram_total_mib,
        }

    def snapshot(self) -> dict[str, Any]:
        """One serialisable dict with everything the dashboard needs in a single poll.

        Gathered in one call rather than one endpoint per probe because the pieces are
        read together and shown together. Split apart, they could disagree: VRAM sampled
        before a model was evicted, next to a model list sampled after, would show free
        memory the display still attributes to a resident model.
        """
        gpu = self._gpu.read()
        loaded = self._models.loaded()

        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "server_up": self._server.is_up(),
            "gpu": gpu.as_dict(),
            "host": asdict(self._host.read()),
            "loaded_models": [asdict(model) for model in loaded],
            # The models' on-disk size. Note this is NOT their runtime VRAM footprint —
            # the gap between this and `gpu.used_mib` is large and worth understanding,
            # because it is what explains why a 9 GB model leaves far less than 7 GB free
            # on a 16 GB card.
            #
            # Measured on this machine with Qwen3 14B resident at 32K context:
            #
            #   desktop + browser + Bionic (no model loaded)   2,240 MiB
            #   model weights                                  8,584 MiB
            #   KV cache                                      ~5,100 MiB
            #   ------------------------------------------------------
            #   total used                                    ~15,940 MiB  (98%)
            #
            # The KV cache is the surprising one, and it is the KV cache rather than the
            # desktop that dominates: Qwen3 14B has 40 layers with 8 key/value heads of
            # 128 dimensions, so at fp16 it stores 2 x 40 x 8 x 128 x 2 bytes = 160 KiB
            # **per token** of context. At 32,768 tokens that is 5,120 MiB, which matches
            # the measurement to within 0.4%.
            #
            # It scales linearly with context, so halving the context frees half the cache
            # — the practical lever if a second model needs to fit alongside.
            "model_vram_mib": sum(model.size_mib for model in loaded),
            "installed_models": self._models.installed(),
        }
