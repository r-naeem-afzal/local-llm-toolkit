"""MCP server — lets Claude call the local model directly.

This is the Claude-facing side of the whole arrangement. Claude decides *what* needs
doing; these tools do the reading, extracting and ranking on the local GPU, so that work
costs no plan usage at all.

Replaces a hand-rolled JavaScript bridge that spoke the protocol by hand over stdin. That
worked, but it duplicated the client, the store and the page fetcher in a second language,
so every fix had to be made twice. This version is a thin wrapper over the same `Toolkit`
everything else uses, which means MCP calls land in the same database and show up in the
same dashboard as any other call.

Run it directly, or register it in `.mcp.json`:

    {
      "mcpServers": {
        "local-llm": {
          "command": "python",
          "args": ["-m", "local_llm.mcp_server"]
        }
      }
    }

## Why stdio, and what that means for printing

The transport is stdio: the client launches this as a subprocess and speaks JSON-RPC over
its standard input and output. That has one consequence worth stating plainly, because it
is the most common way to break an MCP server: **anything written to stdout is protocol
traffic.** A stray `print()` corrupts the stream and the client drops the connection with
a parse error that names no cause. Diagnostics must go to stderr, which the client shows
as server logs.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from .client import LocalLLMError
from .container import Toolkit


class LocalLlmMcpServer:
    """Exposes the toolkit's local-model capabilities as MCP tools.

    A class taking its `Toolkit` through the constructor, so the tools can be pointed at a
    different model server or database — in a test, or a second instance — without editing
    anything here.

    The tool set is deliberately small. Each one is a job where the answer is *contained in
    text the model is handed*: read this page and pull out claims, rank these results,
    answer this self-contained question. That boundary was arrived at by measurement rather
    than taste — a local model asked to explain undocumented intent produces confident
    filler, so tasks of that shape are not offered here.
    """

    def __init__(self, toolkit: Toolkit | None = None) -> None:
        self._toolkit = toolkit or Toolkit()
        self._server = MCPServer(
            name="local-llm",
            version="0.1.0",
            instructions=(
                "Runs a local model on this machine at no API cost. Use it for the "
                "mechanical share of the work: reading web pages and extracting claims, "
                "ranking search results, and self-contained generation. Every call is "
                "recorded locally with its full prompt and response."
            ),
        )
        self._register_tools()

    @property
    def server(self) -> MCPServer:
        return self._server

    def run(self) -> None:
        """Serve over stdio until the client disconnects."""
        self._server.run(transport="stdio")

    def _model_for(self, task: str) -> str | None:
        """The model a task should name, or None to let the client use the configured one.

            "mcp_complete", default qwen3-14b loaded                 ->  "qwen/qwen3-14b"
            "mcp_complete", default not loaded, qwen3.5-9b loaded    ->  "qwen/qwen3.5-9b"
            "mcp_complete", the router raises                        ->  None

        `local_complete` used to name no model, so the client always asked for the
        configured one whatever was loaded. On a day when only the 9B fits on the card that
        asked for a model that had to be loaded, which either evicts the working 9B or is
        refused by the request gate while the 9B sits idle. The router now answers with the
        model that is already loaded when the configured one is not, so this tool uses
        what is there. The other two tools already ask the router; this one did not.

        None on a routing failure is the old behaviour exactly, and a model that is not
        loaded is then dealt with by the gate in the client. Routing must never be the
        reason a call fails.
        """
        try:
            return self._toolkit.router.choose(task)
        except Exception:  # noqa: BLE001 - a routing failure falls back to the configured model
            return None

    # ── tools ──

    def _register_tools(self) -> None:
        """Attach the tool functions to the server.

        Registered inside a method rather than at module level so each closes over
        `self._toolkit`. That is what keeps the dependency injected: a module-level
        function would have to reach for a global toolkit, which is the shape this package
        was rewritten to remove.
        """
        toolkit = self._toolkit

        @self._server.tool(
            name="local_extract_claims",
            description=(
                "Fetch a web page and extract falsifiable claims from it that bear on a "
                "question, each with a verbatim supporting quote and an importance rating. "
                "Runs on the local GPU at no API cost. Prefer this over fetching a page "
                "yourself when you only need the claims, not the whole document."
            ),
        )
        async def local_extract_claims(url: str, question: str) -> dict[str, Any]:
            """Read one page and pull out what it actually asserts.

            The heaviest-value tool here: this is the job that made the whole toolkit
            worth building, because it is pure mechanical reading and it dominated the
            cost of the research harness this replaces.

            Returns a dict rather than the pydantic model because MCP serialises the
            result to JSON for the client.

                url="https://en.wikipedia.org/wiki/Payoneer", question="How do PK
                freelancers receive USD?"
                  ->  {"url": ..., "source_quality": "secondary",
                       "claims": [{"claim": ..., "quote": ..., "importance": "supporting"}],
                       "extractor": "trafilatura", "truncated": false}
            """
            try:
                result = await toolkit.claim_extractor.extract(url, question)
            except LocalLLMError as exc:
                # Returned as data, not raised. A tool that throws gives the caller a
                # protocol-level error with no room to explain; returning the message
                # lets Claude read what went wrong — "the server is not running", say —
                # and act on it rather than just seeing a failure.
                return {"error": str(exc), "url": url}
            except Exception as exc:
                # Page fetching fails in many ordinary ways: DNS, 403, a non-HTML body.
                # None of them should look like a broken tool.
                return {"error": f"{type(exc).__name__}: {exc}", "url": url}

            return result.model_dump()

        @self._server.tool(
            name="local_rank_results",
            description=(
                "Rank search results by relevance to a question, marking SEO spam and "
                "content farms as low. Runs locally at no API cost. Use this to triage a "
                "long result list before deciding which pages are worth fetching."
            ),
        )
        async def local_rank_results(
            question: str, results: list[dict[str, Any]]
        ) -> dict[str, Any]:
            """Triage a result list before anything expensive happens to it.

            The cheap gate in front of the expensive step: ranking twenty results in one
            call costs far less than fetching and extracting from twenty pages.

                question="...", results=[{"title": ..., "url": ..., "snippet": ...}]
                  ->  {"results": [{"url": ..., "title": ..., "relevance": "high",
                                    "snippet": "why it is relevant"}]}
            """
            try:
                ranking = await toolkit.result_ranker.rank(question, results)
            except LocalLLMError as exc:
                return {"error": str(exc)}
            return ranking.model_dump()

        @self._server.tool(
            name="local_complete",
            description=(
                "Answer a self-contained prompt with the local model, at no API cost. Good "
                "for summarising or rewriting text you supply, and for classification. Not "
                "good for questions whose answer is not in the prompt — a local model asked "
                "to explain something undocumented produces plausible-sounding invention."
            ),
        )
        async def local_complete(
            prompt: str, system: str | None = None, max_tokens: int = 2048
        ) -> dict[str, Any]:
            """One completion, with the caveat about ungrounded questions in the docstring.

            `max_tokens` defaults to 2048 rather than something smaller because the local
            models are hybrid-reasoning: they spend this budget on their own thinking
            *before* answering. Measured, Qwen3 14B used 203 output tokens simply to say
            "OK", and a 30-token budget produced no answer at all. A tight default would
            hand back empty results that look like model failures.
            """
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            try:
                text = await toolkit.client.complete(
                    messages, tool="mcp_complete", max_tokens=max_tokens,
                    model=self._model_for("mcp_complete"),
                )
            except LocalLLMError as exc:
                return {"error": str(exc)}
            return {"text": text}

        @self._server.tool(
            name="local_status",
            description=(
                "Report whether the local model server is running, which models are "
                "resident in VRAM, and how much VRAM is free. Check this first if a local "
                "tool has just failed."
            ),
        )
        def local_status() -> dict[str, Any]:
            """Enough machine state to explain why a local call failed or was slow.

            Synchronous, because every probe behind it is a blocking read — an NVML call
            and a subprocess. Declaring it `async` would be a lie that yields no
            concurrency.

            Deliberately narrower than the dashboard's `/system`: the installed-model
            inventory and host RAM are noise in a tool result, and a tool that returns
            more than the caller needs costs context on every invocation.
            """
            snapshot = toolkit.monitor.snapshot()
            gpu = snapshot["gpu"]
            return {
                "server_up": snapshot["server_up"],
                "endpoint": toolkit.settings.url,
                "default_model": toolkit.settings.model,
                "loaded_models": [
                    {
                        "key": model["key"],
                        "status": model["status"],
                        "context_length": model["context_length"],
                        # Surfaced because it explains an otherwise baffling slow call: an
                        # idle model is evicted to free VRAM, and the next request pays the
                        # load again — about 18 seconds on this machine.
                        "evicts_in_s": model["ttl_remaining_s"],
                    }
                    for model in snapshot["loaded_models"]
                ],
                "vram_used_mib": gpu.get("used_mib"),
                "vram_total_mib": gpu.get("total_mib"),
                "vram_free_mib": gpu.get("free_mib"),
            }


def main() -> None:
    """Entry point for `python -m local_llm.mcp_server`."""
    LocalLlmMcpServer().run()


if __name__ == "__main__":
    main()
