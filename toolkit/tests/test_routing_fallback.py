"""A task with no entry in the role map uses the model that is loaded when the default is not.

Today such a task (the plain `local_complete` tool is one) gets `settings.model`
whatever is loaded. These tests cover the one path that changes: the default is not loaded
and a language model is. The fake registry stands in for `lms`; nothing is loaded.
"""

from __future__ import annotations

from fakes import FakeRegistry, installed_entry, loaded_model

from local_llm.config import Settings
from local_llm.routing import ModelRouter

FOURTEEN_B = "qwen/qwen3-14b"
NINE_B = "qwen/qwen3.5-9b"
CODER = "qwen/qwen2.5-coder-14b"
EMBEDDING = "text-embedding-nomic-embed-text-v1.5"

# A task that is not in the router's role map, as `local_complete` is not.
UNMAPPED = "mcp_complete"

INSTALLED = [
    installed_entry(FOURTEEN_B, size_mib=8584),
    installed_entry(NINE_B, size_mib=6706),
    installed_entry(CODER, size_mib=8571, architecture="qwen2"),
    installed_entry(EMBEDDING, size_mib=80, kind="embedding", architecture="nomic-bert"),
]


def make_router(*, loaded=(), installed=INSTALLED, raises=False, default=FOURTEEN_B):
    registry = FakeRegistry(
        loaded=[loaded_model(key) for key in loaded], installed=list(installed), raises=raises,
    )
    # _env_file=None so a developer's real .env cannot change what a test sees.
    return ModelRouter(Settings(model=default, _env_file=None), registry)


# ─────────────────────────── the new fallback ───────────────────────────


def test_the_default_is_used_when_it_is_loaded():
    assert make_router(loaded=[FOURTEEN_B]).choose(UNMAPPED) == FOURTEEN_B


def test_another_language_model_is_used_when_the_default_is_not_loaded():
    assert make_router(loaded=[NINE_B]).choose(UNMAPPED) == NINE_B


def test_the_default_is_used_when_nothing_is_loaded():
    assert make_router(loaded=[]).choose(UNMAPPED) == FOURTEEN_B


def test_the_default_is_used_when_only_an_embedding_model_is_loaded():
    assert make_router(loaded=[EMBEDDING]).choose(UNMAPPED) == FOURTEEN_B


def test_an_embedding_model_is_passed_over_for_a_language_model():
    assert make_router(loaded=[EMBEDDING, NINE_B]).choose(UNMAPPED) == NINE_B


def test_the_default_wins_when_it_is_loaded_alongside_another_model():
    assert make_router(loaded=[NINE_B, FOURTEEN_B]).choose(UNMAPPED) == FOURTEEN_B


def test_the_first_listed_is_used_when_several_other_models_are_loaded():
    first = make_router(loaded=[CODER, NINE_B]).choose(UNMAPPED)
    second = make_router(loaded=[NINE_B, CODER]).choose(UNMAPPED)
    assert (first, second) == (CODER, NINE_B)


def test_a_loaded_model_missing_from_the_installed_list_is_still_used():
    # `lms ls` once stopped listing a model that was loaded and running.
    assert make_router(loaded=[NINE_B], installed=[]).choose(UNMAPPED) == NINE_B


def test_a_loaded_embedding_model_missing_from_the_installed_list_is_still_skipped():
    # Classified from its name when the installed list cannot say what it is.
    assert make_router(loaded=[EMBEDDING], installed=[]).choose(UNMAPPED) == FOURTEEN_B


def test_an_unreadable_registry_gives_the_default():
    assert make_router(loaded=[NINE_B], raises=True).choose(UNMAPPED) == FOURTEEN_B


def test_prefer_loaded_false_ignores_what_is_loaded_on_this_path_too():
    router = make_router(loaded=[NINE_B])
    assert router.choose(UNMAPPED, prefer_loaded=False) == FOURTEEN_B


def test_a_different_configured_default_is_the_one_looked_for():
    router = make_router(loaded=[NINE_B], default=NINE_B)
    assert router.choose(UNMAPPED) == NINE_B
    router = make_router(loaded=[FOURTEEN_B], default=NINE_B)
    assert router.choose(UNMAPPED) == FOURTEEN_B


# ─────────────────────────── what did not change ───────────────────────────


def test_a_task_with_roles_still_prefers_a_loaded_model():
    # The mapped path already preferred what is loaded; it must still.
    assert make_router(loaded=[NINE_B]).choose("extract_claims") == NINE_B


def test_a_task_with_roles_and_nothing_loaded_still_picks_by_role():
    # Structured work goes to the coder model when nothing is loaded to prefer.
    assert make_router(loaded=[]).choose("extract_claims") == CODER


def test_no_inventory_at_all_still_gives_the_default_for_a_mapped_task():
    router = make_router(loaded=[NINE_B], installed=[])
    assert router.choose("extract_claims") == FOURTEEN_B
