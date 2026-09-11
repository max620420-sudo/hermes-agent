"""Fail-closed contract for automatic background review.

Automatic review spawns ONLY when ``auxiliary.background_review.enabled`` is
explicitly truthy. Missing keys, malformed blocks, invalid values, loader
exceptions, and dispatch-time reload failures all stay OFF. Explicit
``/refine`` (``focus`` set) keeps working while automatic review is disabled.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import run_agent as run_agent_module
from agent import background_review as br
from agent.review_idle_queue import ReviewIdleQueue
from hermes_cli import config_defaults as config_defaults
from run_agent import AIAgent


def _cfg(enabled_block):
    return {"auxiliary": {"background_review": enabled_block}}


def _loaded(cfg):
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        return br.load_background_review_settings()


# ── loader truth table ───────────────────────────────────────────


def test_enabled_true_stays_on():
    assert _loaded(_cfg({"enabled": True}))[0] is True
    assert br.is_background_review_enabled({"enabled": True}) is True


@pytest.mark.parametrize("good", [1, "true", "True", "1", "yes", "on"])
def test_truthy_spellings_stay_on(good):
    assert _loaded(_cfg({"enabled": good}))[0] is True


def test_enabled_false_is_off():
    assert _loaded(_cfg({"enabled": False}))[0] is False


def test_missing_block_is_off():
    assert _loaded({})[0] is False
    assert _loaded({"auxiliary": {}})[0] is False


def test_missing_enabled_key_is_off():
    assert _loaded(_cfg({"model": "x"}))[0] is False
    assert br.is_background_review_enabled({}) is False


@pytest.mark.parametrize("bad", ["yes-please", ["enabled"], 42, None])
def test_malformed_block_is_off(bad):
    assert _loaded({"auxiliary": {"background_review": bad}})[0] is False


@pytest.mark.parametrize("bad", ["maybe", "", 0, [], {}])
def test_invalid_enabled_value_is_off(bad):
    assert _loaded(_cfg({"enabled": bad}))[0] is False


def test_loader_exception_is_off():
    with patch(
        "hermes_cli.config.load_config_readonly",
        side_effect=RuntimeError("config store on fire"),
    ):
        assert br.load_background_review_settings() == (False, {})


def test_default_config_ships_disabled():
    assert config_defaults.DEFAULT_CONFIG["auxiliary"]["background_review"]["enabled"] is False


# ── dispatch-time gate ───────────────────────────────────────────


def test_still_enabled_exception_drops_queued_review():
    with patch.object(
        br, "load_background_review_settings", side_effect=RuntimeError("reload blew up")
    ):
        assert ReviewIdleQueue._still_enabled(object()) is False


def test_still_enabled_honors_loader_result():
    with patch.object(br, "load_background_review_settings", return_value=(False, {})):
        assert ReviewIdleQueue._still_enabled(object()) is False
    with patch.object(br, "load_background_review_settings", return_value=(True, {"enabled": True})):
        assert ReviewIdleQueue._still_enabled(object()) is True


# ── spawn gate: missing key spawns nothing; /refine still works ──


class _ImmediateThread:
    def __init__(self, *, target, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "telegram"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.session_id = "test-session"
    agent._parent_session_id = ""
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._cached_system_prompt = "test-cached-system-prompt"
    import datetime as _dt
    import threading as _threading

    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.background_review_callback = None
    agent.status_callback = None
    agent._safe_print = lambda *_args, **_kwargs: None
    agent._background_review_agent = None
    agent._background_review_run = None
    agent._background_review_lock = _threading.Lock()
    agent._active_children = []
    agent._active_children_lock = _threading.Lock()
    return agent


def test_missing_enabled_key_spawns_nothing_but_refine_works(monkeypatch):
    """No ``enabled`` key: automatic fork skipped, explicit ``/refine`` runs."""
    forks = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            forks.append(kwargs)

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", _ImmediateThread)

    agent = _bare_agent()
    agent._delegate_depth = 0
    cfg = {"auxiliary": {"background_review": {"model": "x"}}}  # no "enabled" key

    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
        )
        assert forks == [], "automatic review must not spawn without enabled=true"

        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
            focus="save the deploy workflow",
        )
        assert len(forks) == 1, "/refine must still run while automatic review is off"
