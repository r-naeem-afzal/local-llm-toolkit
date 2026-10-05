"""Comparing local models on the work this toolkit actually does.

The question this exists to answer is narrow and practical: **is a bigger model worth
loading?** On a 16 GB card that is never a free choice — a bigger model means either a
harsher quantisation, a shorter context, or weights spilling into system RAM — so the
answer has to come from measurement rather than from parameter counts.

## Why it replays recorded calls instead of using a standard benchmark

Public benchmarks measure general ability. What matters here is whether a model does
*these* jobs — extract claims from a fetched page, rate a pool of search results — on
*these* prompts, which are long, schema-constrained, and unlike benchmark questions.
Every such call ever made is already in the call log with its full prompt, so the honest
test set is sitting in the database.

That has a second benefit worth naming: a replay is a *paired* comparison. Both models see
byte-identical prompts, so a difference in the result cannot be explained away by one
model having been given easier work.

## The two things measured, and why quality is not judged by a model

**Speed** is split into prefill and decode (see `StreamAccumulator.ttft_ms`), because a
partially-offloaded model is penalised almost entirely on decode and a combined
tokens-per-second figure would hide exactly the effect being investigated.

**Quality** is measured as *quote groundedness*: the extraction schema asks for a verbatim
supporting quote for every claim, so a quote that does not appear in the prompt was
fabricated. That is checkable with string matching — no second model needed.

Using a model to grade a model was rejected deliberately. The known failure rate of a 14B
on this exact task is roughly one quoted claim in five not appearing in the page it is
attributed to; a grader with its own error rate of that order tells you very little about a
candidate, and the two error rates are not independent when grader and candidate share a
family. String matching cannot be charitable, cannot be flattered, and costs nothing.

What breaks without a deterministic metric: the benchmark becomes a vote between two models
that are wrong in correlated ways, and a fabricating model that fabricates *plausibly*
scores well — which is the precise failure this whole toolkit is built to catch.

## Terms, in plain words

* **prefill** — the model reading the prompt before it writes anything. One-off per call.
* **decode** — the model emitting tokens, one at a time. This is where generation speed
  lives, and it is limited by how fast weights can be read from memory.
* **time to first token (TTFT)** — how long prefill took, from the caller's point of view.
* **offload** — keeping part of a model's weights in system RAM because they do not fit in
  VRAM. Reading them crosses the PCIe bus, roughly fifteen times slower than VRAM.
* **groundedness** — whether a quoted span really occurs in the source text.
"""

from __future__ import annotations

import json
import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel

from .client import CompletionClient, LocalLLMError
from .extract import Extraction, IndexRanking
from .store import CallRepository


# ─────────────────────────── the test set ───────────────────────────


#: Which pydantic schema each recorded tool was called with.
#:
#: Needed because the call log stores the prompt and the model but not the schema, and a
#: replay without the schema is not the same call: `response_format` constrains generation,
#: so a schema-less replay measures a different and easier task.
#:
#: A tool absent from this map replays as free text, which is correct for the smoke-test
#: calls in the log ("Reply with exactly three words describing SQLite").
_TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "extract_claims": Extraction,
    "rank_results": IndexRanking,
}


@dataclass(frozen=True)
class ReplayTask:
    """One recorded call, ready to be run again against any model.

    Frozen because a task is the fixed input of a comparison. If a runner could mutate it
    between models the comparison would silently stop being paired, which is the one
    property that makes these numbers meaningful.
    """

    task_id: str
    tool: str
    messages: list[dict[str, str]]
    schema: type[BaseModel] | None
    #: Every character the model was shown, concatenated. This is the *only* text it could
    #: legitimately quote from, which makes it the reference for groundedness scoring.
    prompt_text: str

    @property
    def prompt_chars(self) -> int:
        return len(self.prompt_text)


class TaskSource(ABC):
    """Where replay tasks come from.

    An abstraction rather than a function because the obvious second implementation — a
    hand-written set of tasks covering work the log happens not to contain yet — should
    drop in without the runner knowing the difference.
    """

    @abstractmethod
    def tasks(self, *, limit: int = 50) -> list[ReplayTask]: ...


class CallLogTaskSource(TaskSource):
    """Replay tasks recovered from previously recorded calls.

    Takes the repository through the constructor rather than opening the database itself,
    so a test can supply a fake with three known rows and assert on the selection rules
    below without a database file existing.
    """

    def __init__(self, repository: CallRepository,
                 tools: Sequence[str] | None = None) -> None:
        self._repository = repository
        # Default to the tools that do the toolkit's real work. The log also contains
        # smoke tests and probes, which are not worth a model swap to measure.
        self._tools = tuple(tools) if tools else ("extract_claims", "rank_results")

    def tasks(self, *, limit: int = 50) -> list[ReplayTask]:
        """Recorded calls worth replaying, newest first.

        Only `status == "ok"` rows with a recoverable prompt qualify. An errored row is
        excluded not because it is uninteresting but because many of them failed for
        reasons that have since been fixed — a token budget too small for a reasoning
        model, for instance — so replaying them would measure a bug that no longer exists.
        """
        found: list[ReplayTask] = []
        seen_prompts: set[str] = set()

        for tool in self._tools:
            for record in self._repository.list_calls(limit=limit, tool=tool, status="ok"):
                payload = self._repository.get_payload(record.id)
                if payload is None or not payload.prompt:
                    continue

                messages = self._as_messages(payload.prompt)
                if not messages:
                    continue

                prompt_text = "\n".join(m.get("content", "") for m in messages)

                # Deduplicate by prompt content, not by call id. The same page gets
                # extracted more than once across runs, and replaying an identical prompt
                # twice does not add information — it just doubles the cost of the
                # benchmark and makes the median look more precise than it is.
                fingerprint = str(hash(prompt_text))
                if fingerprint in seen_prompts:
                    continue
                seen_prompts.add(fingerprint)

                found.append(ReplayTask(
                    task_id=record.id,
                    tool=tool,
                    messages=messages,
                    schema=_TOOL_SCHEMAS.get(tool),
                    prompt_text=prompt_text,
                ))

        return found[:limit]

    @staticmethod
    def _as_messages(stored: Any) -> list[dict[str, str]]:
        """Normalise a stored prompt back into the message list the client expects.

            '[{"role": "user", "content": "hi"}]'  ->  [{"role": "user", "content": "hi"}]
            [{"role": "user", "content": "hi"}]    ->  unchanged

        Both shapes occur: the repository decodes the JSON column for some callers and
        hands back the raw string for others. Handling one shape only would drop half the
        available test set, and silently — the rows would simply not appear.
        """
        if isinstance(stored, str):
            try:
                stored = json.loads(stored)
            except json.JSONDecodeError:
                return []
        if not isinstance(stored, list):
            return []
        return [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in stored
            if isinstance(m, dict)
        ]


# ─────────────────────────── quality scoring ───────────────────────────


@dataclass(frozen=True)
class GroundingScore:
    """How many of a model's quotes really occur in what it was shown."""

    quotes: int
    grounded: int

    @property
    def rate(self) -> float | None:
        """Fraction verifiable, or None when the model quoted nothing.

        None rather than 1.0 for the empty case, and the distinction matters: a model that
        returns no claims has not achieved perfect groundedness, it has abstained. Scoring
        that as 100% would rank a model that says nothing above every model that says
        something, which would make the metric actively misleading.
        """
        return None if self.quotes == 0 else self.grounded / self.quotes


class GroundednessScorer:
    """Checks that quoted spans appear in the text the model was given.

    ## Why the comparison is fuzzy rather than exact

    A model reproducing a quote from a web page routinely differs from the source in ways
    that are not fabrication: a non-breaking space becomes a normal one, smart quotes
    become straight ones, a line break inside a sentence disappears. Scoring those as
    invented would put the false-negative rate well above the fabrication rate being
    measured, and the metric would be dominated by typography.

    So both sides are normalised — whitespace collapsed, quote and dash characters
    unified, case folded — and then compared as substrings.

        source: "costs \\u00a340 per\\nmonth"   quote: "costs £40 per month"   -> grounded
        source: "costs £40 per month"        quote: "costs £50 per month"   -> NOT grounded

    Note what is deliberately *not* forgiven: any difference in the actual words or
    numbers. Normalisation touches only characters that carry no meaning. A changed figure
    is the single most damaging kind of fabrication in a research pipeline, so it must
    never be normalised away.

    ## The limit of this metric, stated plainly

    A verbatim quote proves the text exists in the document. It does **not** prove the
    claim attached to it is supported by that text — a model can quote accurately and then
    draw a conclusion the quote does not license. So this is a fabrication detector, not a
    correctness oracle, and a high score is a floor on quality rather than a guarantee of
    it.
    """

    _WHITESPACE = re.compile(r"\s+")

    # Characters that differ between a page and a model's reproduction of it without any
    # change in meaning. Mapped to a single canonical form on both sides.
    _EQUIVALENTS = {
        "‘": "'", "’": "'", "‛": "'",       # curly single quotes
        "“": '"', "”": '"',                        # curly double quotes
        "–": "-", "—": "-", "−": "-",       # en/em dash, minus
        " ": " ", " ": " ", " ": " ",       # non-breaking/thin spaces
        "…": "...",                                     # ellipsis
    }

    #: Quotes shorter than this are not scored. A three-character fragment such as "40"
    #: occurs by chance in almost any long document, so counting it as grounded would
    #: inflate the score for exactly the models that quote least usefully.
    _MIN_QUOTE_CHARS = 12

    def normalise(self, text: str) -> str:
        for original, replacement in self._EQUIVALENTS.items():
            text = text.replace(original, replacement)
        return self._WHITESPACE.sub(" ", text).strip().casefold()

    def score(self, task: ReplayTask, answer: Any) -> GroundingScore:
        """Compare every quote in `answer` against the prompt of `task`."""
        quotes = [q for q in self._quotes(answer) if len(q) >= self._MIN_QUOTE_CHARS]
        if not quotes:
            return GroundingScore(quotes=0, grounded=0)

        # Normalised once, not once per quote: the prompt is the large side of this
        # comparison — extraction prompts run to tens of thousands of characters — and
        # normalising it inside the loop turned out to dominate the scoring time.
        haystack = self.normalise(task.prompt_text)
        grounded = sum(1 for q in quotes if self.normalise(q) in haystack)
        return GroundingScore(quotes=len(quotes), grounded=grounded)

    @staticmethod
    def _quotes(answer: Any) -> list[str]:
        """Pull the quoted spans out of whatever shape the answer has.

            Extraction(claims=[Claim(quote="x"), Claim(quote="y")])  ->  ["x", "y"]
            IndexRanking(...)                                        ->  []   (no quotes)
            "some free text"                                         ->  []

        Duck-typed on the presence of a `quote` attribute rather than switched on the
        schema class, so a new schema with quoted evidence is scored without editing this.
        """
        claims = getattr(answer, "claims", None)
        if not isinstance(claims, Iterable):
            return []
        return [
            str(getattr(c, "quote", "")) for c in claims if getattr(c, "quote", "")
        ]


# ─────────────────────────── running the comparison ───────────────────────────


@dataclass(frozen=True)
class TrialResult:
    """One model's attempt at one task."""

    model: str
    task_id: str
    tool: str
    ok: bool
    prompt_chars: int
    error: str = ""
    duration_ms: int | None = None
    ttft_ms: int | None = None
    decode_tps: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    reasoning_tokens: int | None = None
    grounding: GroundingScore | None = None


class TrialMetricsReader:
    """Recovers a finished call's timings from the log.

    Exists because `CompletionClient.complete` deliberately returns the *answer* and
    nothing else — every caller in the toolkit wants the result, not instrumentation — and
    widening that return type to carry metrics would change an interface used everywhere
    to serve one consumer.

    The log is the right place to read them from anyway: it is where the project's claim
    about local work being auditable actually rests, so a benchmark that reads its own
    numbers from the same rows the dashboard shows is checking the log as well as the
    model.

    Lookup is by the unique tool name the runner assigns each trial, not by "the most
    recent row", because the latter is a race the moment anything else is running.
    """

    def __init__(self, repository: CallRepository) -> None:
        self._repository = repository

    def read(self, trial_tool: str) -> dict[str, Any]:
        records = self._repository.list_calls(limit=1, tool=trial_tool)
        if not records:
            return {}
        record = records[0]
        meta = record.meta or {}
        return {
            "duration_ms": record.duration_ms,
            "tokens_in": record.tokens_in,
            "tokens_out": record.tokens_out,
            "ttft_ms": meta.get("ttft_ms"),
            "decode_tps": meta.get("decode_tps"),
        }


class ModelBenchmark:
    """Runs a set of replay tasks against one model and scores the results.

    Every collaborator arrives through the constructor. That is what makes this testable
    without a model server: hand it a fake client returning canned `Extraction` objects and
    the scoring and aggregation can be asserted on directly.
    """

    def __init__(self, client: CompletionClient, repository: CallRepository,
                 scorer: GroundednessScorer | None = None,
                 metrics: TrialMetricsReader | None = None) -> None:
        self._client = client
        self._repository = repository
        self._scorer = scorer or GroundednessScorer()
        self._metrics = metrics or TrialMetricsReader(repository)

    async def run(self, model: str, tasks: Sequence[ReplayTask], *,
                  max_tokens: int | None = None,
                  run_tag: str = "bench",
                  on_trial: Callable[[int, ReplayTask, TrialResult], None] | None = None,
                  ) -> list[TrialResult]:
        """Run every task against `model`, in order, recording each attempt.

        The model is **not** loaded here. Choosing and loading a model is `ModelLoader`'s
        job, and doing it per call is the documented way to end up with two models sharing
        a card that fits one. The caller loads once, then calls this.

        `on_trial` is called after each task with its 1-based position, the task and the
        result. It exists so a command-line caller can print progress without driving this
        loop one task at a time from outside — which is how the trial-naming bug below was
        introduced, and is the kind of thing a caller should not have to get right.
        """
        results: list[TrialResult] = []
        for position, task in enumerate(tasks, start=1):
            result = await self._run_one(
                model, task, max_tokens=max_tokens, run_tag=run_tag,
            )
            results.append(result)
            if on_trial is not None:
                on_trial(position, task, result)
        return results

    async def _run_one(self, model: str, task: ReplayTask, *,
                       max_tokens: int | None, run_tag: str) -> TrialResult:
        # A tool name unique to this trial. Two purposes: it makes the metrics lookup
        # exact rather than "newest row", and it keeps benchmark traffic out of the real
        # per-tool statistics — without it, replaying forty extractions would move the
        # dashboard's average extraction time and nobody would know why.
        #
        # Keyed on the *task id*, not on a loop counter. A counter is only unique if this
        # method sees the whole task list, and the first version of the command-line
        # caller called `run` once per task — so every trial was numbered 0, every trial
        # shared one tool name, and the metrics lookup returned whichever row happened to
        # be newest. Six trials silently reported one trial's timings.
        trial_tool = f"{run_tag}:{task.task_id}:{self._short(model)}"

        try:
            answer = await self._client.complete(
                task.messages,
                schema=task.schema,
                tool=trial_tool,
                meta={"bench_task": task.task_id, "bench_of": task.tool, "model": model},
                max_tokens=max_tokens,
                model=model,
            )
        except LocalLLMError as exc:
            # A failure is a result, not an interruption. A model that cannot produce
            # valid JSON for a schema has told us something decisive about its fitness for
            # this pipeline, and aborting the run would throw that finding away along with
            # every task after it.
            metrics = self._metrics.read(trial_tool)
            return TrialResult(
                model=model, task_id=task.task_id, tool=task.tool, ok=False,
                prompt_chars=task.prompt_chars, error=str(exc)[:300], **metrics,
            )

        metrics = self._metrics.read(trial_tool)
        return TrialResult(
            model=model, task_id=task.task_id, tool=task.tool, ok=True,
            prompt_chars=task.prompt_chars,
            grounding=self._scorer.score(task, answer), **metrics,
        )

    @staticmethod
    def _short(model: str) -> str:
        """`qwen/qwen3-30b-a3b` -> `qwen3-30b-a3b`, for a readable tool name."""
        return model.split("/")[-1]


# ─────────────────────────── reporting ───────────────────────────


@dataclass
class ModelSummary:
    """Aggregated results for one model. Mutable — it is built up as trials arrive."""

    model: str
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return len(self.trials)

    @property
    def succeeded(self) -> int:
        return sum(1 for t in self.trials if t.ok)

    @property
    def success_rate(self) -> float | None:
        return None if not self.trials else self.succeeded / len(self.trials)

    @property
    def median_decode_tps(self) -> float | None:
        """Median generation speed, medians throughout rather than means.

        A single call that hit a model swap, a Windows memory spill or a browser stealing
        VRAM can be an order of magnitude slower than the rest. With the handful of tasks
        the log currently holds, one such outlier moves a mean far enough to reverse the
        comparison; it moves a median hardly at all.
        """
        return self._median(t.decode_tps for t in self.trials if t.ok)

    @property
    def median_ttft_ms(self) -> float | None:
        return self._median(t.ttft_ms for t in self.trials if t.ok)

    @property
    def median_duration_ms(self) -> float | None:
        return self._median(t.duration_ms for t in self.trials if t.ok)

    @property
    def grounding_rate(self) -> float | None:
        """Quotes verifiable in the prompt, pooled across every trial.

        Pooled — total grounded over total quoted — rather than averaged per trial. A
        per-trial average weights a call that made one claim the same as a call that made
        five, so a model could improve its score by being terse on the hard pages.
        """
        quotes = sum(t.grounding.quotes for t in self.trials if t.grounding)
        grounded = sum(t.grounding.grounded for t in self.trials if t.grounding)
        return None if quotes == 0 else grounded / quotes

    @property
    def total_quotes(self) -> int:
        return sum(t.grounding.quotes for t in self.trials if t.grounding)

    @staticmethod
    def _median(values: Iterable[Any]) -> float | None:
        present = [float(v) for v in values if v is not None]
        return statistics.median(present) if present else None


class BenchmarkReport:
    """Turns trial results into something a person can read and act on.

    Its own class because presentation changes for entirely different reasons than
    measurement does, and because the runner should not care whether output goes to a
    terminal, the dashboard or a note in the vault.
    """

    def __init__(self, summaries: Sequence[ModelSummary]) -> None:
        self._summaries = list(summaries)

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "model": s.model,
                    "attempted": s.attempted,
                    "succeeded": s.succeeded,
                    "success_rate": s.success_rate,
                    "median_decode_tps": s.median_decode_tps,
                    "median_ttft_ms": s.median_ttft_ms,
                    "median_duration_ms": s.median_duration_ms,
                    "grounding_rate": s.grounding_rate,
                    "total_quotes": s.total_quotes,
                }
                for s in self._summaries
            ]
        }

    def render(self) -> str:
        lines = [
            "",
            f"{'model':28s} {'ok':>7s} {'decode':>9s} {'ttft':>8s} {'total':>8s} "
            f"{'quotes':>7s} {'grounded':>9s}",
            f"{'':28s} {'':>7s} {'tok/s':>9s} {'ms':>8s} {'ms':>8s} {'':>7s} {'':>9s}",
            "-" * 84,
        ]
        for s in self._summaries:
            lines.append(
                f"{s.model[:28]:28s} "
                f"{s.succeeded:>3d}/{s.attempted:<3d} "
                f"{self._num(s.median_decode_tps, 1):>9s} "
                f"{self._num(s.median_ttft_ms, 0):>8s} "
                f"{self._num(s.median_duration_ms, 0):>8s} "
                f"{s.total_quotes:>7d} "
                f"{self._pct(s.grounding_rate):>9s}"
            )
        lines.append("")
        lines.extend(self._caveats())
        return "\n".join(lines)

    def _caveats(self) -> list[str]:
        """Print what the numbers cannot support, next to the numbers themselves.

        Deliberately part of the output rather than documentation. A table of figures gets
        copied into a note and quoted back weeks later; a caveat that lives only in a
        docstring does not travel with it, and the figure then carries more authority than
        it earned.
        """
        notes = ["Read with care:"]
        attempted = max((s.attempted for s in self._summaries), default=0)
        if attempted < 20:
            notes.append(
                f"  * only {attempted} task(s) per model - enough to catch a large "
                f"difference, not enough to resolve a small one."
            )
        if any(s.total_quotes == 0 for s in self._summaries):
            notes.append(
                "  * at least one model produced no scorable quotes, so its groundedness "
                "column is absent rather than perfect."
            )
        notes.append(
            "  * groundedness proves a quote exists in the prompt, not that the claim "
            "attached to it follows from the quote."
        )
        return notes

    @staticmethod
    def _num(value: float | None, places: int) -> str:
        return "-" if value is None else f"{value:.{places}f}"

    @staticmethod
    def _pct(value: float | None) -> str:
        return "-" if value is None else f"{value * 100:.0f}%"
