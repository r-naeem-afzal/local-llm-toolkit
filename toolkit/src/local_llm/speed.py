"""Timing one short answer from a loaded model, to tell a healthy load from a crawling one.

## Why speed is the thing measured

A model that does not fit on the graphics card still loads. Windows places the overflow in
ordinary system memory and the model keeps answering, about fifteen times slower, with no
error anywhere. Free graphics memory does not show the difference either: on the app
version measured on 13 October 2026 every load fills the card on purpose, so a healthy
load and an overfilled one both leave a few hundred MiB free. The only reading that
separates them is how fast the model answers. Two measurements of the same model:

    loaded with one prediction slot, 765 MiB free   ->  109 tokens a second
    loaded by the server itself, four slots, 190 MiB free
                                                    ->  one answer took eight minutes

So `ModelLoader` asks this module for one number after every load and unloads a model
that is under the floor (`Settings.min_tokens_per_second`).

## Why two calls

The first request after a load is slower than every later one: the runtime finishes
setting itself up on first use. Measured: 4.1 seconds for the first 260 tokens, 2.4 for
the second. Judging the first would fail a healthy model on a slow disk, so the first call
is made and thrown away and only the second is timed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .config import Settings


@dataclass(frozen=True)
class SpeedReading:
    """One measurement. "Frozen" means it cannot be changed after it is made.

        64 tokens in 0.66 seconds   ->  SpeedReading(tokens_per_second=97.0, tokens=64,
                                                     seconds=0.66, problem="")
        the server never answered   ->  SpeedReading(tokens_per_second=0.0, tokens=0,
                                                     seconds=60.0,
                                                     problem="no answer within 60 seconds")

    `problem` is empty when the speed is a real measurement. When it is not empty the
    speed is 0.0 and must not be read as "zero tokens a second": the caller reports the
    problem in words instead.
    """

    tokens_per_second: float
    tokens: int
    seconds: float
    problem: str = ""


class SpeedProbe(Protocol):
    """Anything that can time a model. A Protocol is an interface by shape: any class with
    this method fits, without inheriting from anything, so a test passes a three-line fake."""

    def measure(self, model_key: str) -> SpeedReading: ...


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str],
               timeout_s: float) -> dict[str, Any]:
    """Send one JSON request and return the JSON answer. Raises on any failure.

    A plain blocking request, not the toolkit's streaming client: the loader runs in
    ordinary synchronous code, and this call must not be recorded in the call history or
    pass through the residency gate (the gate calls the loader, which calls this).
    """
    response = httpx.post(url, json=body, headers=headers, timeout=timeout_s)
    response.raise_for_status()
    return response.json()


class CompletionSpeedProbe:
    """Times a short fixed completion through the server's OpenAI-compatible endpoint.

    The clock and the function that sends the request are passed in, so a test supplies
    both and nothing waits or touches the network.
    """

    # How many tokens the model is asked to produce. Enough that the time is mostly
    # generation and not the fixed cost of one request; few enough that a healthy model is
    # done in about a second.
    _TOKENS = 64

    # Fewer tokens than this in the answer and the time is mostly the fixed cost of the
    # request, so dividing by it would give a speed that means nothing.
    #
    #     3 tokens in 0.15 seconds  ->  "20 tokens a second" from a model that does 100
    _MIN_TOKENS = 16

    # The prompt is fixed so every measurement times the same work, and it asks for far
    # more text than `_TOKENS` so the answer is cut off by the limit and never ends early.
    _PROMPT = "Count from 1 to 200. Write one number on each line and nothing else."

    # Longest one call may take. A healthy model needs about a second and the warm-up call
    # a few more. A crawling one needs minutes, and waiting those out with nobody home is
    # the failure this check exists to prevent, so the call is abandoned and reported.
    _CALL_TIMEOUT_S = 60.0

    def __init__(self, settings: Settings,
                 post: Callable[[str, dict[str, Any], dict[str, str], float],
                                dict[str, Any]] = _post_json,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self._settings = settings
        self._post = post
        # `monotonic` is a clock that only moves forward, which is the right one for
        # measuring a duration: the wall clock can jump when the PC corrects its time.
        self._monotonic = monotonic

    def measure(self, model_key: str) -> SpeedReading:
        """Make the warm-up call, then time a second one. Never raises.

            warm-up 4.1 s, second call 64 tokens in 0.66 s  ->  97.0 tokens a second
            second call returns 3 tokens                    ->  problem "... only 3 tokens"
            either call fails or runs past 60 seconds       ->  problem saying which
        """
        warm_up = self._one_call(model_key)
        if warm_up.problem:
            return warm_up
        return self._one_call(model_key)

    def _one_call(self, model_key: str) -> SpeedReading:
        """Ask for the fixed completion once and work out tokens a second.

            {"usage": {"completion_tokens": 64}} after 0.66 seconds
              ->  SpeedReading(tokens_per_second=97.0, tokens=64, seconds=0.66)

        The time is the whole request, so it includes reading the short prompt as well as
        writing the answer. That makes the figure slightly lower than the model's true
        generation speed, which is the safe direction for a floor.

        `completion_tokens` counts everything the model wrote, including the "thinking"
        text a reasoning model produces before its answer. That is what is wanted here:
        the question is how fast tokens come out, not what they say.
        """
        body = {
            "model": model_key,
            "messages": [{"role": "user", "content": self._PROMPT}],
            "max_tokens": self._TOKENS,
            # 0 makes the model pick the likeliest token every time, so two measurements
            # do the same work.
            "temperature": 0,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self._settings.api_key}"}
        began = self._monotonic()
        try:
            answer = self._post(f"{self._settings.url}/chat/completions", body, headers,
                                self._CALL_TIMEOUT_S)
        except httpx.TimeoutException:
            return SpeedReading(
                tokens_per_second=0.0, tokens=0,
                seconds=round(self._monotonic() - began, 2),
                problem=f"no answer within {self._CALL_TIMEOUT_S:.0f} seconds",
            )
        except Exception as exc:
            return SpeedReading(
                tokens_per_second=0.0, tokens=0,
                seconds=round(self._monotonic() - began, 2),
                problem=f"the request failed: {type(exc).__name__}: {exc}"[:200],
            )
        seconds = self._monotonic() - began

        usage = answer.get("usage") if isinstance(answer, dict) else None
        tokens = int((usage or {}).get("completion_tokens") or 0)
        if tokens < self._MIN_TOKENS or seconds <= 0:
            return SpeedReading(
                tokens_per_second=0.0, tokens=tokens, seconds=round(seconds, 2),
                problem=f"the answer had only {tokens} tokens, too few to time",
            )
        return SpeedReading(
            tokens_per_second=round(tokens / seconds, 1), tokens=tokens,
            seconds=round(seconds, 2),
        )
