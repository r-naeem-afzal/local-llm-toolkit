"""The three settings that guard model loading: their defaults, their environment form, and
the values they refuse. `_env_file=None` throughout so a developer's real `.env` cannot
change what a test sees."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_llm.config import Settings


def test_the_defaults():
    settings = Settings(_env_file=None)
    assert settings.load_context_length == 32768
    assert settings.guard_model_loads is True
    assert settings.unattended_models == ("qwen/qwen3-14b", "qwen/qwen3.5-9b")


def test_each_setting_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_LOAD_CONTEXT_LENGTH", "16384")
    monkeypatch.setenv("LOCAL_LLM_GUARD_MODEL_LOADS", "false")
    # A list-valued setting is a JSON list in the environment: that is how
    # pydantic-settings reads any list.
    monkeypatch.setenv("LOCAL_LLM_UNATTENDED_MODELS", '["a/one", "b/two"]')
    settings = Settings(_env_file=None)
    assert settings.load_context_length == 16384
    assert settings.guard_model_loads is False
    assert settings.unattended_models == ("a/one", "b/two")


def test_the_guard_can_be_switched_off_in_code():
    assert Settings(guard_model_loads=False, _env_file=None).guard_model_loads is False


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_context_length_is_refused(bad):
    with pytest.raises(ValidationError):
        Settings(load_context_length=bad, _env_file=None)


def test_an_empty_model_list_is_refused():
    with pytest.raises(ValidationError):
        Settings(unattended_models=(), _env_file=None)
