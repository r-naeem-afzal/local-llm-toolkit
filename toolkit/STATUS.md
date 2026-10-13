# Local LLM Toolkit: engineering log

**What this file is.** A running record of what has been built, what has been *measured*,
and what turned out to be wrong. It is kept because nearly every non-obvious decision in
this codebase was driven by a number rather than a preference, and those numbers are
invisible in the source. If you are wondering why some piece of this is shaped oddly, the
answer is probably here.

It is written as working notes rather than polished documentation, and it contains
machine-specific measurements (a 16 GB RTX 5070 Ti on Windows 11) that will not all
transfer. Read it as evidence, not as instructions. For getting started, see the
[README](../README.md); for how to contribute, see
[CONTRIBUTING.md](../CONTRIBUTING.md).

Last substantive update 2026-10-13.

## Goal

Delegate mechanical work to a local model instead of spending metered API or plan usage on
it, with an interface that shows exactly what the local models are doing, and ship it as a
reusable, `.env`-configured package.

Constraints this design has to respect:

- Claude effort stays at `high` at most. Subagent fan-out is now permitted, because
  parallel agents cost about the same total usage as sequential ones and finish sooner,
  provided activity is watched live and a batch cannot exhaust the 5-hour window.
- Plan usage resets on a 5-hour rolling window, so consumption must be visible.
- 16 GB VRAM total. One 14B model at a time, plus a small model at most.
- Model downloads are the user's job via the Bionic GUI, not the CLI.
- Python for logic, React + TypeScript (Next.js) for the interface.

## Environment as it stands

| Thing | State |
|---|---|
| Bionic (LM Studio renamed) | installed at `%LOCALAPPDATA%\Programs\Bionic`, home is `~/.lmstudio` |
| Local server | ON, `http://localhost:1234/v1` (`lms server start` if not) |
| `lms` CLI | `~/.lmstudio/bin/lms.exe`, on PATH |
| Qwen3 14B Q4_K_M | installed, loads in ~18s, 32K context (its maximum), ~49 tok/s |
| `nomic-embed-text-v1.5` | installed |
| GPU | RTX 5070 Ti, 16303 MiB. **With Qwen3 14B resident at 32K, VRAM sits at ~94%.** |
| Python | 3.14.7, pip available, no `uv` |
| Node | 24.19.0, npm 11.17.0 |

Models still wanted (user installs these via the GUI):

- **Qwen2.5-Coder 14B Instruct Q4_K_M** (~9 GB): the coding lane.
- **Qwen3 4B Q4_K_M** (~2.6 GB): cheap triage. Note it will *not* fit beside the
  14B at 32K context; either lower the 14B's context or rely on TTL eviction.

## What works (Python package, verified live)

Installed editable: `pip install -e .` from `local-llm/toolkit/`.
Extras installed so far: `trafilatura`, `psutil`, `nvidia-ml-py`, `fastapi`, `uvicorn`,
`PyMySQL`.

The package is **classes, following SOLID**, with collaborators injected through
constructors. `Toolkit` in `container.py` is the composition root and the only place that
names concrete implementations. Comments follow the house standard in the project
`CLAUDE.md`: the why, the failure mode a guard defends against, jargon glossed plainly,
and a worked example at every data transformation.

| Module | Status | Classes |
|---|---|---|
| `config.py` | OK | `Settings`, `DatabaseConfig`: data only |
| `database.py` | OK | `DatabaseBackend` -> `SqliteBackend`, `MySqlBackend` |
| `store.py` | OK | `ConnectionProvider`, `SchemaMigrator`, `SqlCallRepository`, `FileLiveProgressStore`, `RetentionService` |
| `client.py` | OK | `StreamAccumulator`, `ProgressReporter`, `ThinkingStripper`, `JsonResponseParser`, `LocalLLMClient` |
| `extract.py` | OK | `TrafilaturaExtractor`, `RegexExtractor`, `ExtractorChain`, `PageFetcher`, `PromptLibrary`, `ClaimExtractor`, `ResultRanker` |
| `monitor.py` | OK | `GpuProbe`, `HostProbe`, `LmsCommandRunner`, `ModelRegistry`, `ModelServerProbe`, `TranscriptLocator`, `TranscriptParser`, `ClaudeUsageReader`, `SystemMonitor` |
| `agents.py` | OK | `AgentInvocation`, `AgentTranscriptScanner`, `ActiveAgent`, `AgentActivity`, `AgentActivityReader` |
| `api.py` | OK | `SystemRoutes`, `AgentRoutes`, `HistoryRoutes`, `MaintenanceRoutes`, `DashboardApi` |
| `container.py` | OK | `Toolkit` |
| `scripts/smoke_test.py` | OK | `SmokeTest`: run this first on resume |
| `scripts/agent_probe.py` | OK | `AgentProbe`: the live agent view without the UI |

Usage is now via the composition root:

```python
from local_llm import Toolkit
toolkit = Toolkit()
extraction = await toolkit.claim_extractor.extract(url, question)
```

## Database: choose the engine in one config block

Everything about the database lives in `DatabaseConfig`. Nothing else in the package
mentions it, so switching engines is configuration, not code.

```dotenv
# SQLite: the default, works with nothing set
LOCAL_LLM_DATABASE__ENGINE=sqlite
LOCAL_LLM_DATABASE__NAME=C:/data/calls.db

# MySQL / MariaDB
LOCAL_LLM_DATABASE__ENGINE=mysql
LOCAL_LLM_DATABASE__NAME=local_llm
LOCAL_LLM_DATABASE__HOST=127.0.0.1
LOCAL_LLM_DATABASE__PORT=3306
LOCAL_LLM_DATABASE__USER=llm
LOCAL_LLM_DATABASE__PASSWORD=secret
```

`pip install -e ".[mysql]"` for the driver (PyMySQL: pure Python, so no compiler needed).

**MySQL is verified against a live MySQL 8.0.46** (already running here; credentials are
in `.env`). Confirmed: schema creation, idempotent re-apply, insert, upsert without row
duplication, the coalesce upsert preserving reasoning, utf8mb4 emoji/CJK round-trip,
stats, size, count- and age-based retention, `OPTIMIZE TABLE`, a real model call end to
end, and all six API endpoints.

Two bugs that only the live server exposed, both now fixed:

- **`autocommit=False` made a read park an open transaction**, so `OPTIMIZE TABLE` hung
  178s behind an idle connection holding a 498s transaction, and the dashboard would
  have shown a permanently stale snapshot. Now autocommit on, with explicit `begin()`
  for multi-statement writes via `backend.begin_transaction`.
- **MySQL returns `Decimal` for `SUM()`**, which `json.dumps` refuses, so `/stats` was
  200 on SQLite and **500 on MySQL**. Normalised in the MySQL backend's `query`.

The dialect differences handled, each a runtime failure rather than a syntax error:

| Concern | SQLite | MySQL |
|---|---|---|
| Parameters | `?` | `%s` |
| Upsert | `on conflict … do update set x = excluded.x` | `on duplicate key update x = values(x)` |
| Retention by count | `not in (select … limit ?)` | **rejected**: LIMIT in an IN subquery needs a derived-table wrapper |
| Size / reclaim | `stat()` / `VACUUM` | `information_schema` / `OPTIMIZE TABLE` |
| Idle connections | never expire | closed past `wait_timeout`: needs `ping(reconnect=True)` |
| Primary key type | `text` | `text` is illegal; `varchar(191)` with utf8mb4 |
| Index DDL | `create index if not exists` | no such form; "Duplicate key name" must be ignored |
| Transactions | implicit, commit on success | autocommit **on** required, else reads block maintenance |
| Aggregate types | `int` | `Decimal`: not JSON-serialisable |

Measured on the live model:

- Plain completion: 3.3s: small asks are effectively free.
- Claim extraction with trafilatura: **13.6s**, page fits in 18,738 chars untruncated.
  With the regex fallback: 17.5s and truncated at the 24,000 char cap.
- `claims[0]` really is a `Claim` instance, so schema validation is doing its job.
- All API endpoints answer 200 after the rewrite, verified on a clean process.

### Storage question, answered

The original JSONL log wrote the full page prompt **twice per call** (~55 KB/call) and
was parsed in full on every dashboard load. The SQLite store keeps metadata at about
**200 bytes per call** and stores each prompt once in a side table that can be pruned
independently. Same six calls: 115 KB of JSONL versus a 32 KB database with payloads
pruned. Live progress is not persisted at all: it goes to one overwritten file per
in-flight call under `.local-llm-data/live/`.

## What is broken or incomplete

1. ~~`pyproject.toml` declares a console script that does not exist~~: **fixed.** The
   `[project.scripts]` block is removed; better to ship no entry point than a broken one.
2. ~~The old JavaScript dashboard is broken~~: **replaced.** The Next.js dashboard
   at `../dashboard` is built and verified, and `Start-Local-LLM-Dashboard.cmd` is
   rewritten to launch the model server, the API and the UI together.
3. ~~The JS files are superseded but kept until the Python MCP server is proven~~:
   **done.** `mcp_server.py` was written and verified over real stdio first (handshake,
   tool listing, all four tools), then the four `.mjs` files were deleted and `.mcp.json`
   repointed at `python -m local_llm.mcp_server`.

   Note the SDK moved: `mcp` is 2.1.1, where `FastMCP` became `MCPServer`
   (`from mcp.server.mcpserver import MCPServer`). v1 examples will not import.
4. **Foreground subagent cost cannot be accounted for, and the metric that was meant to
   account for it is dead.** `subagent_messages` in the usage panel is derived from
   `isSidechain`, which is never set, so it reads zero whether ten agents ran or none,
   and a zero reads as reassurance rather than as "not visible". Background agents report
   real usage and `agents.py` now reads it; foreground agents are unavailable at any
   price. This one is not fixable here, only labelled, and the dashboard now labels it.

## Next steps, in order

1. ~~`monitor.py`~~: **done and verified live.** See the correction below: NVML
   cannot do per-process VRAM either.
2. ~~`api.py`~~: **done and verified live.** `uvicorn local_llm.api:app --port 7878`.
3. ~~Next.js + React + TypeScript dashboard~~: **done.** At `../dashboard`. Type-checks
   clean, builds to 111 KB first-load JS, verified against the live API with real model
   activity. See its README for how it updates and why the panels are ordered as they are.
4. ~~`search.py`~~: **done.** `SearchService` over `BraveSearchProvider`,
   `SearxngSearchProvider` and `DuckDuckGoProvider` (itself two backends: the `ddgs`
   package, falling back to HTML scraping). 25/25 live calls succeeded.

   **Only DuckDuckGo is actually configured**: Brave has no key and SearXNG no URL, so
   `fallbacks: []` on every call. One rate-limit stops a run. Adding a Playwright-backed
   provider was discussed as the option needing neither a key nor a service.
5. ~~`pipeline.py`~~: **done and verified on a real question.** `QueryPlanner`,
   `ExtractionCache`, `QuoteVerifier`, `ReportWriter`, `ResearchPipeline`.

       6 queries -> 43 distinct pages -> ranked -> 5 read (1 cached)
       20 claims, 19 quote-verified, 149s, no plan usage

   Two caveats found by running it. Verification is **string matching only**: it catches
   an invented quote, which is the failure that matters most, but not a real quote used to
   support a claim it does not support, and there is no contradiction detection at all.
   And the report answered a *neighbouring* question: asked about merchant-of-record
   platforms, DuckDuckGo returned freelancer-payment listicles and nothing re-queries when
   results drift off-topic.

6. `routing.py`: **done.** `ModelRouter` picks a model per task from what is installed,
   classified by architecture and name rather than a hard-coded list so newly downloaded
   models are routed to without a restart. Coarse by necessity: only one 14B fits in
   16 GB, so a swap costs an ~18s load and `prefer_loaded` avoids marginal ones.
7. ~~`mcp_server.py`~~: **done and verified.** Four tools: `local_extract_claims`,
   `local_rank_results`, `local_complete`, `local_status`.
8. ~~A live view of currently-active Claude agents in the dashboard~~: **done and
   verified live.** `agents.py`, `GET /agents`, and the dashboard's Claude agents panel
   with start/finish/fail notifications. The planned implementation had to be thrown
   away: "a sidechain with a recent message is an active agent" cannot work, because no
   sidechain record is ever written, so it would have reported zero agents for ever while
   looking like working code.

   What works instead is the `Agent` tool call itself. An `assistant` record carries a
   `tool_use` block named `Agent` with the `subagent_type`, `description` and `model`;
   the agent is running until its ending arrives. **Foreground and background agents end
   differently, and this is the trap:** a background launch gets a `tool_result` within
   about two seconds that is only an acknowledgement it started, not its report. Taking
   that at face value marked two agents finished after 1.6 s and 2.2 s when they in fact
   ran for four minutes and two, so `running` read 0 while agents were running, which is
   the single number the panel exists to provide. A background agent's real ending is its
   task notification; a foreground agent's is its `tool_result`.

   Verified live with real agents: `running 1` with elapsed climbing 5.6 s → 15.0 s →
   24.4 s → 2m 26s, two parallel agents as two distinct rows, measured tokens on the
   background pair, and the degraded paths (projects directory missing, pointing at a
   file, or holding malformed JSONL) all returning empty rather than a 500.

### Claude plan usage is only partly measurable locally: corrected 2026-09-05

`~/.claude/projects/<project>/<session>.jsonl` records every assistant message with a
`usage` block, so **main-loop** consumption over a 5-hour window is summable, and that
part works.

The rest of the earlier note was wrong, and wrong in the direction that matters.
`isSidechain` does not do what it says: measured across every transcript in every project
on this machine (Claude Code 2.1.260), **it is never true**. Subagents write no sidechain
records and no transcript of their own. So the plan to "split by `isSidechain`" produces
one bucket holding everything and a second permanently empty, which is why the
dashboard's `subagent_messages` metric has always read zero, and why that zero must not
be read as "no agents ran".

The consequence for budgeting: a **foreground** subagent's tokens are billed to the
5-hour window and recorded nowhere at all, so any total computed here is a *floor*
whenever agents have run.

**Background agents are the exception, and the only usable route.** An agent launched
with `run_in_background: true` reports its real usage when it completes, in a
task-notification record:

```xml
<task-notification>
  <tool-use-id>toolu_011hr…</tool-use-id>
  <status>completed</status>
  <usage><subagent_tokens>55343</subagent_tokens><tool_uses>7</tool_uses>
         <duration_ms>113246</duration_ms></usage>
</task-notification>
```

It arrives as an `attachment` record whose `prompt` holds that block (also seen on a
`queue-operation` record, so `agents.py` checks both shapes). Measured on two real runs:
**55,343 and 83,526 tokens**.

Parsed transcripts should still be cached by mtime and size: the current session file is
already over 1 MB. `AgentTranscriptScanner` goes further and tails by byte offset, because
the agent panel polls every two seconds and re-reading a megabyte each time would make
the monitoring more expensive than the work it monitors.

## Fixed: the reasoning channel was being discarded

`client.py` read only `delta.content`. Bionic/LM Studio streams a reasoning model's
thinking in a **separate `reasoning_content` field**, and `enable_thinking: false` is
accepted and **ignored** (28 reasoning tokens out of a 30-token budget). So every
Qwen3 call threw away the thinking, and any call whose budget was consumed by it
failed with the unhelpful `local model returned empty content`.

The `_THINK_RE` strip for inline `<think>` tags was therefore dead code on this
runtime. It is kept for llama.cpp / vLLM / Ollama, which do inline it.

Now: reasoning is accumulated separately, stored in a new `payloads.reasoning` column
(with a `_migrate` step, since `create table if not exists` never adds a column to an
existing database), and shown live as a distinct `thinking` status with a character
count, verified at `status: thinking, chunks: 244, reasoning_chars: 1197`. Previously
that window showed `chars: 0` and was indistinguishable from a stalled call.

Budget implications, measured: Qwen3 14B spends **203 output tokens and ~1,000
characters of reasoning to say "OK"**. Use a Coder (non-reasoning) model for extraction
and ranking, or set `max_tokens` well clear of the expected answer length.

Note the original smoke test passed throughout, because its 2048-token default left
room to answer after thinking. Test near the limits, not in the comfortable middle.

## Corrections to earlier notes (measured 2026-09-05)

**Per-process VRAM is not obtainable on this machine at all.** The earlier note said
NVML via `pynvml` was "the route that can attribute memory to a process". It is not.
`nvmlDeviceGetComputeRunningProcesses` returns every process with
`usedGpuMemory = None` on this consumer GeForce card, because under Windows' WDDM
driver model the OS owns video memory allocation, not the NVIDIA driver. This is a
platform limitation with no workaround, not a missing call or a permissions issue.

The replacement is better anyway: `lms ps --json` reports each resident model's
`sizeBytes` directly, so memory is attributed to a *named model* instead of a process
id. `monitor.py` does that and exposes the gap between the two numbers.

**Most of the used VRAM is not the model.** Measured with Qwen2.5-Coder 14B resident:
16,195 of 16,303 MiB used (99.3%), of which the model accounts for only 8,571 MiB. The
other ~7.4 GB is the desktop compositor, the browser, and the KV cache for the 30,464
loaded context. This is the concrete reason a 9 GB model does not leave 7 GB free, and
why the second 14B will not fit.

**The `lms` CLI needs `server start` called more than once from cold.** The first
invocation returns "Timed out waiting for LM Studio daemon to start" even with Bionic
already running; a later one succeeds. Budget for retries rather than treating the first
failure as the server being unavailable. Corrected on 13 October 2026: "the second
succeeds immediately" was too hopeful. The service needs about a minute, so the unattended
starter now keeps trying for up to two minutes (see the entry for that date below).

**Two more models are installed** beyond what the earlier note recorded as "wanted":
Qwen2.5-Coder 14B (8,571 MiB) and Qwen3 4B (2,381 MiB). Both are on disk; only one 14B
is ever resident.

**Claude usage is dominated by cache reads.** Over one 5-hour window: 73.5M total
tokens, of which 72.2M were cache *reads* and only 854 were fresh input. Any usage
display that sums tokens without separating cache reads from fresh input will overstate
real spend by roughly fifty times. `monitor.py` keeps the four categories apart for
exactly this reason.

**The error runs both ways, and the second direction is new.** Separating cache reads
stops a fiftyfold *overstatement*. Nothing corrects the opposite gap: whatever a
foreground subagent spent is missing from all four categories, so during any session in
which agents ran the headline is a lower bound. Both facts belong together, because
knowing only one of them leaves a reader confidently wrong.

**Estimating an agent's cost from character counts understates it by about thirteen
times.** Sizing an agent from the characters of its prompt plus the characters of its
returned report is the only option for a foreground agent, and it is a poor one: one
background agent reported 55,343 tokens where prompt and report together came to roughly
16,700 characters, which that method would have put at about 4,200. The reason is
structural: a report is a summary of work whose intermediate file reads, tool results and
reasoning appear at neither end. `agents.py` shows such figures with a `~` and never sums
them with measured ones. Treat any budgeting done this way as a lower bound, and prefer
launching agents in the background where there is a choice, since the cost then stops
being guesswork.

## Model benchmark: measured baseline, and three timing corrections (2026-09-08)

`scripts/benchmark_models.py` replays recorded `extract_claims` calls from the call log
against any model, so both models see byte-identical prompts and a difference cannot be
explained by one having had easier work. Quality is scored by checking that each claim's
verbatim `quote` actually occurs in the prompt - deterministic, offline, and not a model
grading a model. See `src/local_llm/benchmark.py` for why that choice is not merely
cheaper but more trustworthy.

**Baseline: `qwen/qwen3-14b`, all layers on GPU, 3 replayed extractions (median prompt
20,233 chars):**

| metric | value |
|---|---|
| schema-valid | 3/3 |
| decode speed (median) | **61.1 tok/s** |
| time to first *answer* token | 8,551 ms: this is thinking time, not latency to first output |
| total per call | 12,432 ms |
| quote groundedness | **92%** (11/12; 22/24 over a separate 6-task run, so the figure is stable) |

Three things were wrong before this was trustworthy, and all three were invisible:

1. **`decode_tps` overstated speed by ~2.7×**, reporting 141-211 tok/s. Caught only because
   this card's 896 GB/s over ~8.4 GiB of weights caps sequential decode near 106 tok/s, so
   the number was above a hardware limit. Cause: `completion_tokens` counts reasoning *and*
   answer tokens, while the decode window started at the first **answer** token, crediting
   every token the model thought over only the seconds it spent speaking. The window now
   opens at the first token of any kind. The corrected 61.1 tok/s agrees with the 69 tok/s
   recorded elsewhere here for the coder 14B, which is independent corroboration.
2. **`upsert_call` never updated `meta_json`**, in either backend. Anything recorded when a
   call *finished* was computed, passed, accepted and dropped, with an empty column as the
   only symptom. Both backends now use `coalesce(excluded.meta_json, calls.meta_json)`, so
   a finishing write adds timing without erasing the URL and question stored at the start.
3. **`lms load --estimate-only` excludes the KV cache and cannot be used as a fit check.**
   It reports 8.38 GiB for qwen3-14b, self-reports "Confidence: LOW", and the model
   actually occupies ~14,193 MiB resident. `VramBudget`'s own 1.66x factor was the more
   honest estimate at the time. Since 13 October 2026 neither is used to refuse a load:
   the app sizes the context to the free memory, so only the weights are checked in
   advance and the load is judged afterwards by its speed (see the entry for that date).

A trap worth writing down: **the live database is MySQL** (`.env` sets
`LOCAL_LLM_DATABASE__ENGINE=mysql`). `toolkit\.local-llm-data\calls.db` is a stale SQLite
leftover holding 20 old rows that nothing reads or writes, and under the MySQL
configuration `settings.db_path` returns the meaningless value `local_llm`. Half an hour
went into "why are my rows not being recorded" before this was spotted.

Also new: `ModelLoader.ensure(..., gpu_ratio=)` and `OffloadPlanner`, which allow a model
larger than the card to be loaded with some layers deliberately left in system RAM. The
pre-load fit check is skipped **only** when a ratio is passed, because a caller passing one
has already decided the weights do not all fit; the checks after the load still run and
remain what actually catches an overfilled card (since 13 October 2026 that is a timed
answer, not a free-memory figure).

## Unattended server start

`python -m local_llm.server_start [--model KEY] [--no-load] [--json]` makes sure the model
server is up and a model is loaded, for a caller with nobody at the keyboard (dayshift). It
exits 0 only when the server answers and, unless `--no-load`, a model is loaded; `--json`
prints one line with the result's fields.

What it checks, in order, each step with a time limit: whether the server already answers;
if not, `lms server start`, run again every fifteen seconds until the server answers or two
minutes have gone by (the cold start above); then, for each model it may load and before
every attempt, free system RAM against a floor (`LOCAL_LLM_MIN_FREE_RAM_MIB`, default
6144), that the GPU can be read at all (an unreadable GPU means "do not load" here, the
opposite of `VramBudget.fits`), and `ModelLoader.ensure`, which loads the model and keeps
it only if it then answers at a usable speed. One lock file in the data directory
(`server-start.lock`, taken over after 10 minutes) keeps two starters from racing.

The sections below are in date order. The entry of 13 October 2026 corrects the one before
it in several places, and where they disagree the later one is right.

### Two measurements, and the four changes they led to (8 October 2026)

Measured on the night of 7 to 8 October 2026:

1. LM Studio's own default context for `qwen/qwen3.5-9b` is 234,752 tokens. Loaded with
   every layer on the GPU at that default, it fills the 16 GB card. `ModelLoader.ensure`
   caught it and unloaded the model again ("loaded but left only 170 MiB free").
2. A request then reached the running server with no model loaded. LM Studio loaded the
   model by itself, with the same defaults and no guard at all: 15.8 GB used, 190 MiB
   free, and the answer crawled for eight minutes. So any tool call made while the server
   was up and the model was not loaded could fill the card with nobody home.

What changed:

1. **A context length on every load.** `ModelLoader` passes `--context-length <n>` on every
   `lms load` it runs, the whole-model path and the partial-offload path alike. The number
   is the setting `load_context_length` (`LOCAL_LLM_LOAD_CONTEXT_LENGTH`), default 32768,
   which is the 14B model's own maximum, so one number serves both models. **Corrected on
   13 October 2026: the flag does not bound the memory a load takes.** The app accepts the
   number and loads the 9B with a context of 243,968 tokens all the same. The flag is
   still sent; what was really loaded is read back and reported.
2. **A gate in front of every request.** `ResidencyGate` (`gate.py`), wired in
   `container.py`, is asked by `LocalLLMClient.complete` before anything is sent, for plain
   and structured calls alike. If the registry shows the model is not loaded it calls
   `ModelLoader.ensure`; if that returns False the request is refused with `LocalLLMError`,
   carrying the loader's `last_error`, and nothing is sent. A model that is already loaded
   costs one lookup in the registry's existing 3 second cache. One lock covers the check
   and the load, so two concurrent calls load a missing model once; the second looks again
   after the first has finished. The gate runs on a helper thread so a minute-long load does
   not freeze the event loop, and a refused call still leaves an error row in the call
   history. The setting `guard_model_loads` (`LOCAL_LLM_GUARD_MODEL_LOADS`, default true)
   switches it off, for a server that is not LM Studio or a script that loads models itself.
   The benchmark script can leave it on: it loads each model before replaying.
3. **The router prefers what is loaded.** A task with no entry in the router's role map
   used to get `settings.model` whatever was loaded. Now, if `settings.model` is not loaded
   and another language model is (never an embedding model), that one is returned. The
   plain `local_complete` MCP tool did not ask the router at all, so it now does; the other
   two tools already did.
4. **The starter tries more than one model.** With no model named it tries the setting
   `unattended_models` (default `qwen/qwen3-14b`, then `qwen/qwen3.5-9b`) in order and
   stops at the first that loads inside the guards. The result's `model` says which one
   loaded; if none did, `error` gives the reason for each. `--model KEY` still means
   exactly that model, with no fallback. The RAM floor and the unreadable-GPU refusal run
   before every attempt, not only the first. The starter no longer reads `settings.model`
   (`LOCAL_LLM_MODEL`); change `unattended_models` to change what it loads.

**Status of this entry.** It was written before any of it had run against the real `lms`.
The first real runs were made on 13 October 2026 and are recorded in the next section.
They confirmed some of the list this entry ended with and overturned the rest:

- `lms load` accepts `--context-length <n>` next to `--gpu`, `--parallel` and `-y`.
  Confirmed. `lms ps` reports 32768 for the 14B (its maximum) and 243,968 for the 9B, so
  the number is accepted and not applied.
- "Both models load at 32768 and pass the loader's headroom check." Overturned. The 9B
  does not load at 32768, and the headroom check refused loads that ran at full speed. It
  has been replaced.
- The `modelKey` in `lms ps --json` equals the name sent in requests. Confirmed for both
  models.
- With the gate on, a tool call for a model that is loaded goes straight through.
  Confirmed with one real call. A call with no model loaded was not tried for real.

Known limit: the gate checks that the model is loaded, not how it was loaded. A model that
something else loaded earlier with several prediction slots still passes the gate, because
it is loaded. The starter does catch it (next section).

### What a load really does, and the changes that followed (13 October 2026)

Measured on the evening of 13 October 2026 on the one 16 GB card (16,303 MiB, with other
programs holding about 1,400 to 1,750 MiB). The app is Bionic 1.1.1, engine llama.cpp
2.56.0.

1. **From cold, `lms` needs about a minute.** The first call reports "Timed out waiting
   for LM Studio daemon to start" although the service is starting. A call a little later
   succeeds. (Found later the same evening: the long failures were not slowness. They
   came from an inherited environment variable, see change 6 below. Without it the
   service is up in about 7 seconds.)
2. **The app does not use the context length it is asked for.** These all loaded
   `qwen/qwen3.5-9b` with a context of 243,968 tokens: `lms load qwen/qwen3.5-9b --gpu max
   --parallel 1 --context-length 8192 -y`, the same with 16384, and with 32768. `POST
   /api/v1/models/load` with `"context_length": 16384` and with 32768 loaded it with
   234,752 and four prediction slots. The app sizes the context to the free graphics
   memory by itself (its own settings call this `autoFit`, with
   `autoFitMinContextLength` 16384). `lms ps --json` reports what was really loaded, in
   the fields `contextLength` and `parallel`. `--parallel 1` on the command line **is**
   honoured.
3. **So every load fills the card on purpose.** Graphics memory in use after a load was
   15,395 to 15,665 MiB, which left 638 to 908 MiB free in five loads by hand, and 148 MiB
   in one run of the starter.
4. **In that state the model is fast.** Loaded with `--gpu max --parallel 1` and asked
   for 260 tokens through `POST /v1/chat/completions`: 4.1 seconds the first time (63
   tokens a second, with the warm-up) and 2.4 seconds the second (109 a second), with 765
   MiB free during and after.
5. **The crawl of 8 October was a load the server made by itself** on a first request:
   four prediction slots, 190 MiB free.
6. **The starter therefore refused both models and left nothing loaded.** Its answer:
   "qwen/qwen3-14b needs about 14249 MiB but only 13796 MiB would be free with nothing
   else loaded; qwen/qwen3.5-9b loaded but left only 148 MiB free; unloaded it rather than
   run in a state where allocations spill to system memory". The first refusal rested on
   an estimate (the weights times 1.66) set against a low guess of the memory an empty
   card would have. Loaded for real later the same evening, the 14B took its full 32,768
   context, left 907 MiB free and answered at 69.9 tokens a second. The second refusal
   rested on free memory, which says nothing once the app fills the card by design.
7. **A loaded model holds system RAM too.** Free RAM read 13,368 MiB before the 14B was
   loaded and 4,426 MiB with it loaded, under the starter's 6,144 MiB floor.

What changed:

1. **The check after a load is a timed answer** (`speed.py`, `ModelLoader`). After a load
   the loader asks the server for one short fixed completion (64 tokens, temperature 0)
   twice, throws the first away as warm-up, and works out tokens a second from the
   second. Under the floor `min_tokens_per_second` (`LOCAL_LLM_MIN_TOKENS_PER_SECOND`,
   default 20) the model is unloaded and the reason says the measured speed. A small
   floor on free graphics memory stays for a card that is truly exhausted:
   `min_free_vram_mib` (`LOCAL_LLM_MIN_FREE_VRAM_MIB`, default 100, was a fixed 300).
2. **The check before a load counts the weights only.** A model is refused without
   loading only when its weights plus 512 MiB do not fit in the memory that would be free
   with nothing loaded. The 1.66x estimate no longer refuses anything.
3. **What was really loaded is read back and reported.** `--parallel 1` and
   `--context-length` are still sent. After the load the loader reads `lms ps` and the
   starter's JSON answer carries `context_asked`, `context_loaded`, `parallel_loaded`,
   `tokens_per_second` and `vram_free_after_load_mib` (`vram_free_mib` is still the reading
   before the load). A load that comes back with more than one prediction slot is
   unloaded. A model found already loaded with more than one slot, which is how the
   server loads by itself, is loaded again with one.
4. **The cold start is waited for.** `lms server start` is run again every fifteen seconds
   until the server answers, for up to `server_start_wait_s`
   (`LOCAL_LLM_SERVER_START_WAIT_S`, now 120 seconds, was 30 with a single retry).
5. **A model that is already loaded is not refused by the guards.** The RAM floor and the
   unreadable-GPU refusal protect against starting a load. When one of them refuses and a
   wanted model is already loaded, the starter keeps that model and reports it. Before
   this, a second run of the starter said "the model was not loaded" and exited 1 with
   the model loaded (measurement 7).

6. **`lms` is started without Electron's "run as a script engine" switch**
   (`childenv.py`). This turned out to be the real reason the service could not be woken
   from cold, and measurement 1 above was a symptom of it. LM Studio's app is built on
   Electron. Editors built on Electron set the environment variable
   `ELECTRON_RUN_AS_NODE=1` for the helper programs they start, and every terminal or
   tool started inside such an editor inherits it. When `lms` then starts the app to wake
   its service, the app inherits the variable too, runs as a bare script engine, rejects
   its own service flag (`bad option: --run-as-service`) and exits. `lms` reports only
   "Timed out waiting for LM Studio daemon to start". Measured: with the variable set,
   `lms server start` from a stopped service failed after 60 seconds on five tries out
   of five; with it removed the same command succeeded in 7 seconds. A service that is
   already running hides the fault, which is why it seemed to come and go. Every place
   the toolkit starts `lms` (the loader, the registry's command runner, the starter) now
   passes an environment with that one variable removed. Outside the toolkit, the same
   applies to anything that starts `lms` from inside such an editor: clear the variable
   for that command.
7. **One model load at a time across processes** (`lockfile.py`, `ModelLoader.ensure`).
   The gate's lock is a `threading.Lock` and is seen only inside one process, but the
   toolkit runs as several: each agent has its own copy of the MCP server and they share
   the one card. Two processes could each find a model missing and each run `unload
   --all` and `load` at the same moment. `ensure` now takes the lock file
   `model-load.lock` in the data directory around the whole of "look again, then unload
   and load", speed check included. A process that waits looks again with a fresh `lms
   ps` once it has the lock, finds the model the other one loaded and loads nothing. The
   wait is bounded by `model_load_lock_wait_s` (`LOCAL_LLM_MODEL_LOAD_LOCK_WAIT_S`,
   default 180 seconds); past it the load is refused with a sentence naming the holder,
   for example "another process holds the model load lock (process 4321, loading
   qwen/qwen3-14b, since 8:52:10 pm (180 seconds ago))". A lock older than
   `model_load_lock_stale_s` (default 600 seconds), or one whose holder is no longer
   running, is taken over. `LockFile` moved out of `server_start.py` so the starter's
   lock and this one share one set of rules.

Real runs with the changed code, same evening, server already up:

| run | model | context asked | context loaded | slots | tokens a second | free after load | time |
|---|---|---|---|---|---|---|---|
| default models | `qwen/qwen3-14b` | 32768 | 32768 | 1 | 69.9 | 907 MiB | 14.0 s |
| `--model qwen/qwen3.5-9b` | `qwen/qwen3.5-9b` | 32768 | 243968 | 1 | 102.1 | 448 MiB | 14.2 s |
| default models, 14B already loaded, RAM 4,422 MiB | `qwen/qwen3-14b` | 32768 | 32768 | 1 | not timed | 877 MiB | 0.3 s |

One reading call through `LocalLLMClient.complete` with the 14B loaded this way (12,000
characters of this file in, three bullet points out) took 6.6 seconds.

The cold start, with change 6 in place: from a stopped service, started from inside an
Electron editor, the starter brought the server up and a model in within 15 seconds.

Two processes started at the same instant from an empty card, each asking
`LocalLLMClient.complete` for an answer from `qwen/qwen3-14b` (change 7):

| process | `lms` commands it ran | time to its answer |
|---|---|---|
| A | `unload --all`, `load qwen/qwen3-14b` | 13.6 s |
| B | none | 17.8 s |

Afterwards `lms ps` showed one instance (identifier equal to the model key, so no second
copy), context 32768, one slot, and no lock file was left behind.

A swap from one loaded model to another through `ModelLoader.ensure`, speed check
included: 16.1 seconds from the 14B to the 9B and 12.5 seconds back.

Not run for real: a load that comes back slow or with four slots. Neither can be produced
on demand, so both are proven with fakes (`tests/test_loader_checks.py`), as is a lock
that is held too long (`tests/test_load_lock.py`).

**Two agents that want different models.** The router avoids this in the usual case: with
`prefer_loaded` (the default) every task is given the language model that is already
loaded, because every language model is classed as able to do both structured and
reasoning work. So once a model is loaded, routed calls do not swap at all. Two openings
remain, and nothing closes them yet:

- From an empty card, two different tasks arriving together choose different models
  before either is loaded (extraction prefers the coder model, a plain completion the
  configured default). The first loads its model, the second gets the lock, evicts it and
  loads its own. The first caller's request has already passed its gate and now names a
  model that is not loaded, which the server answers by loading it by itself, with its
  own slot count and none of the checks here.
- A caller that names a model outright (`model=`, or `prefer_loaded=False`) is obeyed
  every time. Two such callers naming different models swap on every call, at 12 to 16
  seconds a swap, and nothing limits how often.

The rule that would close both is "a language model that is loaded and was used in the
last few minutes answers in place of the one that was named". It is not built. The gate
can only say yes or raise today; to substitute, it would have to hand back a different
model name, and the client records the model in the call history before it asks the
gate, so the history, the live progress file and the request body would all have to
follow the substitution. Until then: have the starter load the model before agents begin
(the unattended shift does), and do not name models outright from two agents at once.

Known limits. A model that is already loaded is not timed again, so a later run of the
starter reports its speed as unknown. The RAM floor still refuses a swap from one loaded
model to a different one, because the loaded model is what holds the RAM; unload first
(`lms unload --all`) to change models.

## Resume checklist

```powershell
cd toolkit
py -m local_llm.server_start            # server up and a model loaded; waits out a cold start
lms ps                                  # confirm which model is loaded
py scripts\smoke_test.py
py scripts\agent_probe.py --watch       # the live Claude agent view, no UI needed
```

`py` is the launcher a python.org install puts on Windows. Where the bare word `python`
works it can be used instead; on Windows it is sometimes a stub that only offers to install
Python from the Microsoft Store.

If the smoke test passes, pick up at step 4 above: `search.py`.
