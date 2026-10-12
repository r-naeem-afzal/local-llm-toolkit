"""`local_complete` names the model the router chooses, not always the configured default.

The tool lives in `mcp_server.py`, which imports the third-party `mcp` package. That
package is only needed to serve the tool to a client, not to test what the tool asks of the
toolkit, so these tests import the module against a small stand-in for `mcp`. The stand-in
is installed for each test only and removed afterwards, and it replaces the real package
even when that is installed, so the result does not depend on the machine.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types

import pytest

NINE_B = "qwen/qwen3.5-9b"


class StubMcpServer:
    """Collects the tool functions registered with `@server.tool(...)`, and nothing else."""

    def __init__(self, **_options) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, name: str, description: str):
        def register(function):
            self.tools[name] = function
            return function
        return register

    def run(self, transport: str) -> None:
        raise AssertionError("the tests never serve")


@pytest.fixture
def mcp_server_module(monkeypatch):
    """`local_llm.mcp_server`, imported with the stand-in in place of `mcp`."""
    package = types.ModuleType("mcp")
    server = types.ModuleType("mcp.server")
    mcpserver = types.ModuleType("mcp.server.mcpserver")
    mcpserver.MCPServer = StubMcpServer
    monkeypatch.setitem(sys.modules, "mcp", package)
    monkeypatch.setitem(sys.modules, "mcp.server", server)
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", mcpserver)
    # Imported fresh, and dropped afterwards, so the module that holds the stand-in does
    # not stay in `sys.modules` for other tests to find.
    monkeypatch.delitem(sys.modules, "local_llm.mcp_server", raising=False)
    module = importlib.import_module("local_llm.mcp_server")
    yield module
    sys.modules.pop("local_llm.mcp_server", None)


class FakeClient:
    """Records the model each completion was asked to use."""

    def __init__(self) -> None:
        self.models: list[str | None] = []

    async def complete(self, messages, *, tool, max_tokens, model=None):
        self.models.append(model)
        return "OK"


class FakeRouter:
    def __init__(self, answer: str | None = None, raises: bool = False) -> None:
        self._answer = answer
        self._raises = raises
        self.tasks: list[str] = []

    def choose(self, task: str) -> str:
        self.tasks.append(task)
        if self._raises:
            raise RuntimeError("router exploded")
        return self._answer


class FakeToolkit:
    def __init__(self, router: FakeRouter) -> None:
        self.client = FakeClient()
        self.router = router


def call_local_complete(module, toolkit: FakeToolkit):
    server = module.LocalLlmMcpServer(toolkit=toolkit)
    return asyncio.run(server.server.tools["local_complete"]("Reply with ok."))


def test_local_complete_asks_the_router_and_names_the_model_it_returns(mcp_server_module):
    toolkit = FakeToolkit(FakeRouter(answer=NINE_B))
    result = call_local_complete(mcp_server_module, toolkit)
    assert result == {"text": "OK"}
    assert toolkit.router.tasks == ["mcp_complete"]
    assert toolkit.client.models == [NINE_B]


def test_local_complete_falls_back_to_the_clients_default_if_routing_fails(mcp_server_module):
    # None means "name no model", which makes the client use the configured one, and the
    # gate then decides whether it needs loading. Routing never fails the call itself.
    toolkit = FakeToolkit(FakeRouter(raises=True))
    result = call_local_complete(mcp_server_module, toolkit)
    assert result == {"text": "OK"}
    assert toolkit.client.models == [None]
