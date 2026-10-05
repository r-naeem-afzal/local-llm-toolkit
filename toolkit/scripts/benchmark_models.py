"""Compare local models on the toolkit's own recorded work.

    python scripts/benchmark_models.py                          # resident model vs the 30B
    python scripts/benchmark_models.py --models a,b --limit 8    # explicit pair, 8 tasks
    python scripts/benchmark_models.py --gpu-ratio 0.55          # force the offload share
    python scripts/benchmark_models.py --plan                    # print the plan, load nothing

The question being answered is whether a larger, partly offloaded model earns the swap.
See `local_llm.benchmark` for why the test set is replayed from the call log and why
quality is scored by string matching rather than by a second model.

## What a run costs

Each model is loaded once, then every task runs against it. Loading is the expensive part
- roughly eighteen seconds, plus the unload that precedes it - which is why the loop is
ordered model-then-tasks and not the other way round. Reversing it would swap models on
every single call and spend more time loading than generating, and on a card that holds
one large model at a time it also risks two models sharing memory that fits one.

Nothing here uses metered API usage: every call goes to the local server, and every call
is recorded in the call log like any other.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from local_llm import Toolkit
from local_llm.benchmark import (
    BenchmarkReport,
    CallLogTaskSource,
    ModelBenchmark,
    ModelSummary,
    ReplayTask,
)
from local_llm.loader import OffloadPlanner


class BenchmarkCommand:
    """Plans the comparison, runs it model by model, and prints the table.

    A class, and one that takes its `Toolkit` through the constructor, for the reason the
    other scripts here do: the composition root stays the only place that knows how the
    pieces fit together, and a test can drive this with a toolkit whose loader and client
    are fakes.
    """

    def __init__(self, toolkit: Toolkit | None = None) -> None:
        self._toolkit = toolkit or Toolkit()
        self._planner = OffloadPlanner(self._toolkit.vram_budget)

    # ── planning ──

    def _weights_mib(self, model: str) -> int:
        """On-disk size of a model, or 0 when the registry has not heard of it."""
        try:
            for entry in self._toolkit.model_registry.installed():
                if entry.get("key") == model:
                    return int(entry.get("size_mib") or 0)
        except Exception:
            # A missing or broken `lms` CLI must not stop a benchmark that could still run
            # against whatever is already loaded. Zero means unknown everywhere here.
            return 0
        return 0

    def _plan(self, models: list[str], forced_ratio: float | None) -> list[tuple[str, float | None]]:
        """Pair each model with the GPU share to load it at.

            [("qwen/qwen3-14b", None), ("qwen/qwen3-30b-a3b", 0.62)]

        None means "all layers on the card", which is what any model that fits should get.
        A ratio is only planned for a model that does not fit, because partial offload is
        a cost to be paid when necessary rather than a setting to prefer.
        """
        free_when_empty = self._toolkit.vram_budget.total_mib()
        plan: list[tuple[str, float | None]] = []
        for model in models:
            if forced_ratio is not None:
                plan.append((model, forced_ratio))
                continue
            weights = self._weights_mib(model)
            plan.append((model, self._planner.ratio_for(weights, free_mib=free_when_empty)))
        return plan

    def _describe(self, plan: list[tuple[str, float | None]], tasks: list[ReplayTask]) -> str:
        lines = [
            "",
            f"Test set: {len(tasks)} recorded call(s) replayed against {len(plan)} model(s).",
        ]
        by_tool: dict[str, int] = {}
        for task in tasks:
            by_tool[task.tool] = by_tool.get(task.tool, 0) + 1
        for tool, count in sorted(by_tool.items()):
            median_chars = sorted(t.prompt_chars for t in tasks if t.tool == tool)
            lines.append(
                f"  * {tool}: {count} task(s), "
                f"median prompt {median_chars[len(median_chars) // 2]:,} chars"
            )
        lines.append("")
        for model, ratio in plan:
            weights = self._weights_mib(model)
            size = f"{weights:,} MiB" if weights else "size unknown"
            if ratio is None:
                lines.append(f"  {model} - {size}, all layers on GPU")
            else:
                on_gpu = int(weights * ratio) if weights else 0
                lines.append(
                    f"  {model} - {size}, {ratio:.0%} on GPU "
                    f"(~{on_gpu:,} MiB VRAM, ~{weights - on_gpu:,} MiB in system RAM)"
                )
        return "\n".join(lines)

    # ── running ──

    async def run(self, models: list[str], *, limit: int, max_tokens: int | None,
                  forced_ratio: float | None, plan_only: bool) -> int:
        source = CallLogTaskSource(self._toolkit.repository)
        tasks = source.tasks(limit=limit)

        if not tasks:
            print(
                "No replayable calls in the log. Run the research pipeline once first -\n"
                "the benchmark deliberately has no synthetic test set, because the point\n"
                "is to measure this machine on its own work.",
                file=sys.stderr,
            )
            return 1

        plan = self._plan(models, forced_ratio)
        print(self._describe(plan, tasks), flush=True)

        if plan_only:
            print("\n--plan given: nothing loaded, nothing run.", flush=True)
            return 0

        benchmark = ModelBenchmark(self._toolkit.client, self._toolkit.repository)
        summaries: list[ModelSummary] = []

        for model, ratio in plan:
            print(f"\n[{model}] loading{'' if ratio is None else f' at {ratio:.0%} GPU'}...",
                  flush=True)

            # One deliberate load per model, before any of its tasks. `ensure` evicts
            # whatever else is resident, which is the whole reason this is not done per
            # call.
            if not self._toolkit.loader.ensure(model, gpu_ratio=ratio):
                print(f"[{model}] SKIPPED: {self._toolkit.loader.last_error}", flush=True)
                continue

            summary = ModelSummary(model=model)

            def report(position: int, task, trial,
                       total: int = len(tasks), model: str = model) -> None:
                # Printed per task because a single extraction can take two minutes on an
                # offloaded model, and a silent process for twenty minutes is
                # indistinguishable from a hung one.
                #
                # `total` and `model` are default arguments rather than closed-over
                # variables. The enclosing loop rebinds `model` on every pass, and a
                # closure reads a variable's value when it *runs*, not when it was
                # defined — so a callback that outlived one iteration would label its
                # output with the last model in the list. Binding at definition time
                # makes each callback carry the model it was made for.
                print(f"[{model}] task {position}/{total} ({task.tool})... "
                      f"{self._trial_line(trial)}", flush=True)

            summary.trials.extend(await benchmark.run(
                model, tasks, max_tokens=max_tokens, on_trial=report,
            ))
            summaries.append(summary)

        if not summaries:
            print("\nNo model could be loaded; nothing to compare.", file=sys.stderr)
            return 1

        print(BenchmarkReport(summaries).render(), flush=True)
        return 0

    @staticmethod
    def _trial_line(trial) -> str:
        if not trial.ok:
            return f"FAILED ({trial.error[:60]})"
        speed = "-" if trial.decode_tps is None else f"{trial.decode_tps:.1f} tok/s"
        ground = ""
        if trial.grounding and trial.grounding.quotes:
            ground = (f", {trial.grounding.grounded}/{trial.grounding.quotes} "
                      f"quotes grounded")
        return f"ok ({speed}{ground})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        default="qwen/qwen3-14b,qwen/qwen3-30b-a3b",
        help="Comma-separated model keys, in the order they should be loaded.",
    )
    parser.add_argument(
        "--limit", type=int, default=6,
        help="How many recorded calls to replay per model (default 6).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Output budget per call. Defaults to the configured value.",
    )
    parser.add_argument(
        "--gpu-ratio", type=float, default=None,
        help="Force the GPU layer share for every model, overriding the planner. "
             "Useful for sweeping the offload ratio to find where throughput peaks.",
    )
    parser.add_argument(
        "--plan", action="store_true",
        help="Print the plan and exit without loading a model or making a call.",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    command = BenchmarkCommand()
    return asyncio.run(command.run(
        models,
        limit=args.limit,
        max_tokens=args.max_tokens,
        forced_ratio=args.gpu_ratio,
        plan_only=args.plan,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
