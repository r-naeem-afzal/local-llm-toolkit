"""Streaming client for an OpenAI-compatible local model server.

Four classes, split by responsibility:

* `StreamAccumulator` — understands the wire format. Given each streamed event it
  separates the answer from the model's private reasoning and tracks totals. It touches
  no network and no storage, so the trickiest logic here is testable with plain dicts.
* `ProgressReporter` — decides when a progress frame is worth writing, and writes it.
* `JsonResponseParser` / `ThinkingStripper` — small, single-purpose text handling.
* `LocalLLMClient` — performs the request and orchestrates the rest.

What this buys over calling the endpoint directly:

* **Typed structured output.** Pass a pydantic model and get an instance of it back. The
  JSON schema is generated from that model and sent as `response_format`, so the server
  constrains generation to match it, and the reply is validated on the way home. No
  hand-written schema dicts, and a malformed reply is an exception rather than a subtly
  wrong dict.
* **Observability by default.** Every call records a start row, throttled live progress
  while tokens stream, and a final row with usage and timing. Nothing is needed at the
  call site.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings
from .store import CallRepository, LiveProgressStore

T = TypeVar("T", bound=BaseModel)


class LocalLLMError(RuntimeError):
    """Raised when the local model cannot be reached or returns unusable output."""


# ─────────────────────────── text handling ───────────────────────────


class ThinkingStripper:
    """Removes inline `<think>…</think>` blocks from a model's answer.

    Some runtimes — llama.cpp, vLLM, Ollama — inline a reasoning model's private
    monologue into the answer text. Left in place it would be handed to whatever consumes
    the result, including the JSON parser for structured calls, where it guarantees a
    validation failure.

    Bionic/LM Studio does *not* do this: it sends thinking in a separate
    `reasoning_content` field, so on that runtime this class never matches anything. It
    is kept because the same client points at all of them, and its absence would only be
    noticed as corrupted output on the ones that inline.

        "<think>Let me see…</think>The answer is 4."  ->  "The answer is 4."
    """

    _PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

    def strip(self, text: str) -> str:
        return self._PATTERN.sub("", text).strip()


class JsonResponseParser:
    """Extracts a JSON object from model output that may be wrapped in prose.

    Schema-constrained generation should make this unnecessary, but not every runtime
    enforces the schema strictly, and a model that prefixes "Here is the JSON:" would
    otherwise fail the whole call over formatting.

        'Here is the result:\\n{"claims": []}'  ->  {"claims": []}
    """

    def parse(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Greedy match from the first '{' to the last '}', so a nested object is
            # captured whole rather than truncated at its first closing brace.
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        raise LocalLLMError(f"local model did not return valid JSON: {text[:300]}")


# ─────────────────────────── stream parsing ───────────────────────────


@dataclass
class StreamAccumulator:
    """Collects an answer and its reasoning from the streamed events, keeping them apart.

    ## Why two channels

    A hybrid-reasoning model such as Qwen3 produces two different kinds of output, and
    Bionic/LM Studio puts them in different fields of the same stream:

        {"choices":[{"delta":{"reasoning_content":"Okay"}}]}   -> private thinking
        {"choices":[{"delta":{"content":"OK"}}]}               -> the actual answer

    Reading only `content` — which this client originally did — makes a thinking model
    look as though it returned nothing at all, and the call fails with a message blaming
    the model for a field-name mistake. Merging the two channels is equally wrong: the
    monologue would end up inside the answer and would break schema validation.

    Note also that `chat_template_kwargs: {"enable_thinking": false}` is **accepted and
    ignored** by this runtime — measured at 28 reasoning tokens out of a 30-token
    budget — so nothing here may assume thinking was switched off.
    """

    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    chunks: int = 0
    usage: dict[str, Any] | None = None

    #: Milliseconds from request to the first *answer* token — "time to first token".
    #: Written by the client rather than computed here, because the client already owns
    #: the monotonic clock for the call and a second clock could disagree with the first.
    #:
    #: Kept separate from the total duration because the two halves of a completion have
    #: completely different costs, and a single tokens-per-second figure hides that:
    #:
    #:   * **prefill** — reading the prompt — is compute-bound and happens once;
    #:   * **decode** — emitting tokens — is memory-bandwidth-bound and happens per token.
    #:
    #: What breaks without this: comparing a model whose weights are entirely in VRAM
    #: against one partly offloaded to system RAM becomes meaningless. Offload penalises
    #: decode far more than prefill, so `tokens_out / total_duration` can look similar for
    #: the two while the actual generation speed differs several-fold. Any conclusion drawn
    #: from the combined figure about whether offloading is viable would be wrong.
    ttft_ms: int | None = None

    #: Set on the first *reasoning* token, for the same reason. A hybrid-reasoning model
    #: may think for many seconds before the first answer token, so `ttft_ms` alone cannot
    #: distinguish "slow to start generating" from "spent ten seconds thinking first".
    first_token_ms: int | None = None

    def consume(self, event: dict[str, Any]) -> str:
        """Fold one streamed event in, returning which channel it carried.

        Returns "content", "reasoning" or "none", so the caller can report the correct
        phase without having to re-inspect the event itself.
        """
        # The usage block arrives as its own final event — requested via
        # stream_options.include_usage — and carries no choices at all.
        if event.get("usage"):
            self.usage = event["usage"]

        choices = event.get("choices") or []
        delta = choices[0].get("delta", {}) if choices else {}

        thought = delta.get("reasoning_content")
        answer = delta.get("content")

        if answer:
            self.content_parts.append(answer)
            self.chunks += 1
            return "content"
        if thought:
            self.reasoning_parts.append(thought)
            # Counted as a chunk too, so the live view shows movement during thinking.
            # Without this a reasoning model appears frozen at zero for the whole time it
            # is working — the exact ambiguity the live view exists to remove.
            self.chunks += 1
            return "reasoning"
        return "none"

    @property
    def content(self) -> str:
        return "".join(self.content_parts)

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_parts)

    @property
    def reasoning_tokens(self) -> int | None:
        """The server's own count of reasoning tokens, when it reports one.

            {"completion_tokens_details": {"reasoning_tokens": 28}}  ->  28
        """
        details = (self.usage or {}).get("completion_tokens_details") or {}
        return details.get("reasoning_tokens")


class ProgressReporter:
    """Writes live progress frames, no more often than the throttle interval allows.

    The throttle is the point of the class. Tokens arrive dozens of times a second;
    writing a frame for each would mean thousands of file writes per call, producing
    updates far faster than anyone can read them. 0.7 s is frequent enough to look live
    and slow enough to cost nothing.
    """

    _INTERVAL_S = 0.7
    # How much of the tail to show: enough to confirm the model is on topic, small enough
    # that a frame stays one small write however long the output grows.
    _TAIL_CHARS = 400

    def __init__(self, live_store: LiveProgressStore, call_id: str, tool: str,
                 model: str, subject: str = "") -> None:
        self._live = live_store
        self._call_id = call_id
        self._tool = tool
        self._model = model
        # What this particular call is working on — the URL for an extraction, the question
        # for a ranking. Carried because the pipeline runs two extractions at once, and two
        # rows reading "extract_claims / qwen3-14b" are indistinguishable on screen: they
        # look like the same call rendered twice rather than two different pages being read.
        self._subject = subject
        # Starts at 0 so the very first frame always passes the throttle: time.monotonic
        # returns a large number, so that first comparison is guaranteed true. Without
        # it, a call shorter than one interval would never report progress at all.
        self._last_emit = 0.0

    def maybe_report(self, accumulator: StreamAccumulator, phase: str,
                     elapsed_ms: int) -> None:
        now = time.monotonic()
        if now - self._last_emit < self._INTERVAL_S:
            return
        self._last_emit = now

        # During thinking there is no answer text yet, so the tail shows reasoning
        # instead. An empty tail would be indistinguishable from a stalled call.
        answering = phase == "content"
        text = accumulator.content if answering else accumulator.reasoning

        self._live.write({
            "id": self._call_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool": self._tool,
            "model": self._model,
            # A distinct status per phase, so the dashboard can label it honestly: a
            # model that is thinking is busy, not stuck.
            "status": "running" if answering else "thinking",
            "chunks": accumulator.chunks,
            "chars": len(accumulator.content),
            "reasoning_chars": len(accumulator.reasoning),
            "elapsed_ms": elapsed_ms,
            "subject": self._subject,
            "tail": text[-self._TAIL_CHARS:],
        })


# ─────────────────────────── the client ───────────────────────────


class CompletionClient(ABC):
    """The one capability everything else in the toolkit needs from a model.

    Extractors, rankers and the pipeline depend on this abstraction rather than on
    `LocalLLMClient` directly, so each can be driven by a fake in tests, or later by a
    different backend, without any of them changing.
    """

    @abstractmethod
    async def complete(self, messages: list[dict[str, str]], *,
                       schema: type[T] | None = None, tool: str = "complete",
                       meta: dict | None = None, max_tokens: int | None = None,
                       model: str | None = None) -> Any:
        """Run one chat completion, returning a `schema` instance or the answer text."""


class LocalLLMClient(CompletionClient):
    """Talks to an OpenAI-compatible endpoint and records everything it does.

    Collaborators arrive through the constructor rather than being reached for inside the
    methods, which is what makes this class usable against a different database or a
    different model server without editing it.
    """

    def __init__(self, settings: Settings, repository: CallRepository,
                 live_store: LiveProgressStore) -> None:
        self._settings = settings
        self._repository = repository
        self._live = live_store
        self._stripper = ThinkingStripper()
        self._json_parser = JsonResponseParser()

    async def complete(self, messages: list[dict[str, str]], *,
                       schema: type[T] | None = None, tool: str = "complete",
                       meta: dict | None = None, max_tokens: int | None = None,
                       model: str | None = None) -> Any:
        model = model or self._settings.model
        token_budget = max_tokens or self._settings.max_tokens

        call_id = self._repository.start_call(
            ts=self._now(), tool=tool, model=model, meta=meta or {}, prompt=messages,
        )
        started = time.monotonic()

        def elapsed_ms() -> int:
            # monotonic, not wall-clock: it cannot jump backwards if the system clock is
            # adjusted mid-call, which would otherwise record a negative duration.
            return int((time.monotonic() - started) * 1000)

        accumulator = StreamAccumulator()
        # The subject comes from the call's metadata, which is where each caller already
        # records what it is working on — so no caller has to pass it twice.
        subject = str((meta or {}).get("url") or (meta or {}).get("question") or "")
        reporter = ProgressReporter(self._live, call_id, tool, model, subject)
        body = self._build_body(messages, model, token_budget, schema)

        try:
            await self._stream(body, accumulator, reporter, elapsed_ms)
        except LocalLLMError as exc:
            self._fail(call_id, str(exc), elapsed_ms(), accumulator, meta)
            raise
        except httpx.TimeoutException as exc:
            message = f"local model timed out after {self._settings.request_timeout_s}s"
            self._fail(call_id, message, elapsed_ms(), accumulator, meta)
            raise LocalLLMError(message) from exc
        except httpx.HTTPError as exc:
            # Deliberately names the fix. A refused connection almost always means the
            # server is simply not running, and this message saves the reader from
            # debugging their code when they need to start a service.
            message = (
                f"cannot reach local model at {self._settings.url} — is the server "
                f"running? (lms server start): {exc}"
            )
            self._fail(call_id, message, elapsed_ms(), accumulator, meta)
            raise LocalLLMError(message) from exc

        return self._finish(call_id, accumulator, schema, elapsed_ms(), token_budget, meta)

    # ── request construction ──

    def _build_body(self, messages: list[dict[str, str]], model: str, max_tokens: int,
                    schema: type[T] | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self._settings.temperature,
            "max_tokens": max_tokens,
            # Asks a hybrid-reasoning model to skip thinking. Measured on Bionic/LM
            # Studio with Qwen3 14B: accepted and IGNORED. Sent anyway because runtimes
            # that do honour it save real time, but see StreamAccumulator — nothing here
            # assumes it worked.
            "chat_template_kwargs": {"enable_thinking": False},
            # Streaming is what makes live progress possible at all. include_usage adds a
            # final event carrying token counts, which a non-streaming call would have
            # returned in the response body instead.
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if schema is not None:
            # `model_json_schema()` turns the pydantic class into the JSON Schema the
            # server uses to constrain generation, so invalid output is prevented while
            # sampling rather than rejected after the fact.
            #   class Claim(BaseModel): claim: str
            #     ->  {"properties": {"claim": {"type": "string"}}, "required": ["claim"]}
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            }
        return body

    # ── the network stream ──

    async def _stream(self, body: dict[str, Any], accumulator: StreamAccumulator,
                      reporter: ProgressReporter, elapsed_ms) -> None:
        """Read the server-sent event stream, feeding each event to the accumulator.

        The response is a long-lived HTTP body delivering one `data: {...}` line per token
        group. `http.stream` keeps it open and yields lines as they arrive instead of
        waiting for the whole body, which is what allows progress to be reported while
        the model is still generating.
        """
        # The timeout covers the entire stream, not just the response headers. Applied to
        # headers alone, a server that accepted the request and then stalled mid-body
        # would hang this call forever.
        timeout = httpx.Timeout(self._settings.request_timeout_s, connect=10.0)

        async with httpx.AsyncClient(timeout=timeout) as http:
            async with http.stream(
                "POST",
                f"{self._settings.url}/chat/completions",
                json=body,
                headers={"authorization": f"Bearer {self._settings.api_key}"},
            ) as response:
                if response.status_code != 200:
                    # On a streaming response the body has not been read yet, so it must
                    # be pulled explicitly. Without aread() the error would carry no
                    # detail about what the server actually objected to.
                    detail = (await response.aread()).decode("utf-8", "replace")
                    raise LocalLLMError(
                        f"local model returned {response.status_code}: {detail[:400]}"
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        # Blank lines separate events and ':' lines are keepalives;
                        # neither carries data.
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        # The end-of-stream sentinel is literal text, not JSON, so it has
                        # to be filtered before parsing or every call would end on a
                        # decode error.
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    phase = accumulator.consume(event)
                    if phase != "none":
                        # Stamp the first token of each kind as it arrives. Done here
                        # rather than inside the accumulator because `elapsed_ms` is the
                        # call's own monotonic clock, started before the request was
                        # sent — so these figures share an origin with the recorded
                        # duration and can be subtracted from it meaningfully.
                        if accumulator.first_token_ms is None:
                            accumulator.first_token_ms = elapsed_ms()
                        if phase == "content" and accumulator.ttft_ms is None:
                            accumulator.ttft_ms = elapsed_ms()
                        reporter.maybe_report(accumulator, phase, elapsed_ms())

    # ── completion bookkeeping ──

    @staticmethod
    def _timing_meta(accumulator: StreamAccumulator, duration_ms: int,
                     caller_meta: dict | None) -> dict:
        """The caller's metadata with prefill/decode timing folded in.

        Timing goes into `meta_json` rather than into new columns on the `calls` table
        because meta is exactly the place for facts *about* a call that not every caller
        produces, and adding a column means editing the schema statements, the upsert SQL
        and the migration list for a figure only some rows will carry.

            caller_meta={"url": "https://x"}, first token at 240ms, first *answer* token
            at 7,800ms, 589 completion tokens, 12,900ms total
              ->  {"url": "https://x", "first_token_ms": 240, "ttft_ms": 7800,
                   "decode_ms": 12660, "decode_tps": 46.5}

        Three timings because they answer three different questions:

        * `first_token_ms` — how long prefill took. When decode begins.
        * `ttft_ms` — how long until the *answer* began. On a reasoning model the gap
          between these two is time spent thinking, which is the caller's felt latency.
        * `decode_tps` — generation speed: every token emitted, reasoning included,
          divided by the time spent emitting them. This is the figure worth comparing
          between models, because it is the one the hardware determines.

        What breaks without the subtraction: a long prompt makes a fast model look slow.
        Extraction calls here run 6,000-token prompts, where prefill is seconds — so
        `tokens_out / duration_ms` would rank models mostly by how much prompt they were
        handed rather than by how fast they generate.
        """
        meta = dict(caller_meta or {})
        ttft = accumulator.ttft_ms
        emitted = (accumulator.usage or {}).get("completion_tokens")

        if ttft is not None:
            meta["ttft_ms"] = ttft
        if accumulator.first_token_ms is not None:
            meta["first_token_ms"] = accumulator.first_token_ms

        # The decode window opens at the first token of **any** kind, not at the first
        # answer token, and this distinction was measured rather than reasoned about.
        #
        # `completion_tokens` counts reasoning tokens together with answer tokens — on a
        # hybrid-reasoning model most of them are reasoning. Dividing that total by the
        # time after the *answer* began credits the model with every token it thought,
        # over only the seconds it spent speaking. The first version of this did exactly
        # that and reported 141-211 tok/s for a 14B on a card whose memory bandwidth caps
        # sequential decode near 106 tok/s: a figure above the hardware limit, which is
        # how the error announced itself.
        #
        # Emitting a reasoning token costs the same work as emitting an answer token, so
        # the honest denominator is everything after generation started.
        start = accumulator.first_token_ms
        if start is not None and emitted and duration_ms > start:
            decode_ms = duration_ms - start
            meta["decode_ms"] = decode_ms
            meta["decode_tps"] = round(emitted * 1000.0 / decode_ms, 1)

        return meta

    def _fail(self, call_id: str, message: str, duration_ms: int,
              accumulator: StreamAccumulator, caller_meta: dict | None = None) -> None:
        self._repository.finish_call(
            call_id, status="error", error=message, duration_ms=duration_ms,
            usage=accumulator.usage, reasoning=accumulator.reasoning or None,
            meta=self._timing_meta(accumulator, duration_ms, caller_meta),
        )
        # Always clear the live entry, or a failed call would linger in the dashboard as
        # permanently "running" and the display would show phantom activity forever.
        self._live.clear(call_id)

    def _finish(self, call_id: str, accumulator: StreamAccumulator,
                schema: type[T] | None, duration_ms: int, token_budget: int,
                caller_meta: dict | None = None) -> Any:
        text = self._stripper.strip(accumulator.content)
        reasoning = accumulator.reasoning or None
        # Computed once and reused by all four exit paths below, so a call that fails
        # schema validation still records how fast the model produced the bad output.
        # Without that, the log would carry timings only for successes and any average
        # drawn from it would be biased towards whatever the model finds easy.
        meta = self._timing_meta(accumulator, duration_ms, caller_meta)

        if not text:
            message = self._describe_empty_output(accumulator, token_budget)
            self._repository.finish_call(
                call_id, status="error", error=message, duration_ms=duration_ms,
                usage=accumulator.usage, reasoning=reasoning, meta=meta,
            )
            self._live.clear(call_id)
            raise LocalLLMError(message)

        if schema is None:
            self._repository.finish_call(
                call_id, status="ok", duration_ms=duration_ms, usage=accumulator.usage,
                response=text, reasoning=reasoning, meta=meta,
            )
            self._live.clear(call_id)
            return text

        try:
            validated = schema.model_validate(self._json_parser.parse(text))
        except (ValidationError, LocalLLMError) as exc:
            # The raw text is stored even on failure. Without it, diagnosing why output
            # did not match the schema would mean re-running the call and hoping it fails
            # the same way — and at temperature 0 that is likely but not guaranteed.
            self._repository.finish_call(
                call_id, status="error", error=f"schema validation failed: {exc}",
                duration_ms=duration_ms, usage=accumulator.usage, response=text,
                reasoning=reasoning, meta=meta,
            )
            self._live.clear(call_id)
            raise LocalLLMError(
                f"local model output did not match {schema.__name__}: {exc}"
            ) from exc

        self._repository.finish_call(
            call_id, status="ok", duration_ms=duration_ms, usage=accumulator.usage,
            response=text, reasoning=reasoning, meta=meta,
        )
        self._live.clear(call_id)
        return validated

    @staticmethod
    def _describe_empty_output(accumulator: StreamAccumulator, token_budget: int) -> str:
        """Explain *why* there is no answer, because the two causes need different fixes.

        The generic "empty content" message this replaced sent the reader looking at the
        model when the actual problem was the token budget.
        """
        if not accumulator.reasoning:
            return "local model returned empty content (no answer and no reasoning)"

        # The model spent its whole output budget thinking and never reached an answer.
        # This is the common failure with a reasoning model and a small max_tokens:
        # reasoning tokens are billed against the SAME budget as the answer, so a
        # 30-token cap can be consumed entirely by the monologue.
        tokens = accumulator.reasoning_tokens
        detail = f"{len(accumulator.reasoning)} chars"
        if tokens:
            detail += f", {tokens} reasoning tokens"
        return (
            f"local model produced only reasoning and no answer ({detail}). "
            f"max_tokens={token_budget} was consumed by thinking — raise it, or use a "
            f"non-reasoning model such as a Coder variant for this call."
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
