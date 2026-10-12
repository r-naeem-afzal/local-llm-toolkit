"""Every `lms load` the loader runs names a context length.

Without one, LM Studio picks its own: 234,752 tokens for qwen/qwen3.5-9b, which fills a
16 GB card. These tests prove the flag is on the command for both loads the loader makes,
whole-model and partial offload, using a loader whose only change is that it records the
command instead of starting `lms`. No process is started and no model is loaded.
"""

from __future__ import annotations

import pytest
from fakes import FakeRegistry, installed_entry

from local_llm.config import DEFAULT_LOAD_CONTEXT_LENGTH, Settings
from local_llm.loader import ModelLoader

NINE_B = "qwen/qwen3.5-9b"
FOURTEEN_B = "qwen/qwen3-14b"


class RecordingLoader(ModelLoader):
    """The real loader with its two slow side effects replaced and nothing else.

    `_run` is the single place `lms` is ever started, so replacing it records every command
    the loader would have run. It also updates the fake registry the way a successful
    command would, so the loader's "did the model become resident" check passes at once.
    `_settle` waits for memory to be released after an unload; there is no real memory.
    """

    def __init__(self, registry: FakeRegistry, **options) -> None:
        super().__init__(runner=None, registry=registry, **options)
        self.commands: list[list[str]] = []

    def _run(self, args: list[str]) -> bool:
        self.commands.append(list(args))
        if args[0] == "load":
            self._registry.mark_loaded(args[1])
        elif args[:2] == ["unload", "--all"]:
            self._registry.clear()
        return True

    def _settle(self) -> None:
        return None

    def load_commands(self) -> list[list[str]]:
        return [command for command in self.commands if command[0] == "load"]


def make_loader(**options) -> RecordingLoader:
    registry = FakeRegistry(installed=[
        installed_entry(NINE_B, size_mib=6706),
        installed_entry(FOURTEEN_B, size_mib=8584),
    ])
    return RecordingLoader(registry, **options)


def test_whole_model_load_carries_the_context_length():
    loader = make_loader()
    assert loader.ensure(NINE_B)
    assert loader.load_commands() == [
        ["load", NINE_B, "--gpu", "max", "--parallel", "1", "--context-length", "32768", "-y"]
    ]


def test_partial_offload_load_carries_the_context_length():
    loader = make_loader()
    assert loader.ensure(NINE_B, gpu_ratio=0.62)
    assert loader.load_commands() == [
        ["load", NINE_B, "--gpu", "0.62", "--parallel", "1", "--context-length", "32768", "-y"]
    ]


def test_out_of_range_ratio_is_clamped_but_the_context_is_still_there():
    loader = make_loader()
    assert loader.ensure(NINE_B, gpu_ratio=1.4)
    (command,) = loader.load_commands()
    assert command[command.index("--gpu") + 1] == "1.00"
    assert command[command.index("--context-length") + 1] == "32768"


def test_a_configured_context_length_is_used_on_both_paths():
    whole = make_loader(context_length=16384)
    whole.ensure(NINE_B)
    partial = make_loader(context_length=16384)
    partial.ensure(NINE_B, gpu_ratio=0.5)
    for loader in (whole, partial):
        (command,) = loader.load_commands()
        assert command[command.index("--context-length") + 1] == "16384"


def test_every_load_in_a_swap_carries_the_context_length():
    # Two loads in a row, the second evicting the first: the flag must be on both, not
    # only on the first load of the process.
    loader = make_loader()
    assert loader.ensure(NINE_B)
    assert loader.ensure(FOURTEEN_B)
    commands = loader.load_commands()
    assert [command[1] for command in commands] == [NINE_B, FOURTEEN_B]
    for command in commands:
        assert command[command.index("--context-length") + 1] == "32768"


def test_a_model_that_is_already_loaded_runs_no_load_at_all():
    loader = make_loader()
    loader.ensure(NINE_B)
    loader.commands.clear()
    assert loader.ensure(NINE_B)
    assert loader.commands == []


def test_the_default_context_length_is_32768_everywhere():
    assert DEFAULT_LOAD_CONTEXT_LENGTH == 32768
    assert Settings(_env_file=None).load_context_length == 32768
    assert ModelLoader(runner=None, registry=FakeRegistry()).context_length == 32768


@pytest.mark.parametrize("bad", [0, -1, -32768])
def test_a_non_positive_context_length_is_refused_when_the_loader_is_built(bad):
    # Left to reach `lms`, a zero may be read as "use the default", which is the load
    # this argument exists to prevent.
    with pytest.raises(ValueError, match="positive"):
        ModelLoader(runner=None, registry=FakeRegistry(), context_length=bad)
