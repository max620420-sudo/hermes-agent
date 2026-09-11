"""Unit tests for the subagent-only tool-result budget (Task 4 semantic port).

Covers ``agent.tool_executor._is_subagent_session`` (platform /
_delegate_depth / is_subagent detection, safe False fallback) and
``agent.tool_executor._budget_for_agent`` (subagent branch dispatch,
parent/main behavior unchanged, exception -> DEFAULT_BUDGET).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.tool_executor import _budget_for_agent, _is_subagent_session
from tools.budget_config import (
    DEFAULT_BUDGET,
    SUBAGENT_RESULT_SIZE_CHARS,
    budget_for_context_window,
    budget_for_subagent,
)


def _agent(**kwargs):
    """Minimal stand-in agent with only the attributes these functions read."""
    defaults = dict(platform="cli", _delegate_depth=0, is_subagent=False, context_compressor=None)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ── _is_subagent_session: detection matrix ─────────────────────────────

class TestIsSubagentSessionDetection:
    def test_platform_subagent_is_true(self):
        assert _is_subagent_session(_agent(platform="subagent")) is True

    def test_platform_subagent_case_and_whitespace_insensitive(self):
        assert _is_subagent_session(_agent(platform="  SubAgent  ")) is True

    def test_delegate_depth_positive_is_true(self):
        assert _is_subagent_session(_agent(_delegate_depth=1)) is True
        assert _is_subagent_session(_agent(_delegate_depth=3)) is True

    def test_delegate_depth_zero_is_false(self):
        assert _is_subagent_session(_agent(_delegate_depth=0)) is False

    def test_is_subagent_flag_alone_is_true(self):
        # Forward-compat signal: not set anywhere in production today, but
        # honored if a future caller sets it directly.
        assert _is_subagent_session(_agent(is_subagent=True)) is True

    def test_ordinary_main_session_is_false(self):
        assert _is_subagent_session(_agent()) is False
        assert _is_subagent_session(_agent(platform="cli")) is False
        assert _is_subagent_session(_agent(platform="gateway")) is False

    def test_missing_attributes_default_to_false(self):
        assert _is_subagent_session(SimpleNamespace()) is False

    def test_exception_falls_back_to_false(self):
        class _Weird:
            @property
            def platform(self):
                raise RuntimeError("boom")

        assert _is_subagent_session(_Weird()) is False

    def test_non_string_delegate_depth_does_not_raise(self):
        # int("not-a-number") raises ValueError inside the try/except; must
        # still resolve to a safe False, not propagate.
        assert _is_subagent_session(_agent(_delegate_depth="not-a-number")) is False


# ── _budget_for_agent: subagent dispatch + parent/main invariance ──────

class TestBudgetForAgentDispatch:
    def test_subagent_gets_subagent_budget(self):
        agent = _agent(platform="subagent")
        budget = _budget_for_agent(agent)
        assert budget.default_result_size == SUBAGENT_RESULT_SIZE_CHARS
        assert budget.preview_tail_size > 0

    def test_subagent_via_delegate_depth_gets_subagent_budget(self):
        agent = _agent(_delegate_depth=2)
        budget = _budget_for_agent(agent)
        assert budget.default_result_size == SUBAGENT_RESULT_SIZE_CHARS

    def test_parent_main_result_is_byte_identical_to_pre_port(self):
        """The exact call `_budget_for_agent` made before this port existed
        (ctx-window scaling only, no subagent branch) must still be produced
        for an ordinary (non-subagent) agent."""
        cc = SimpleNamespace(context_length=131_072)
        agent = _agent(context_compressor=cc)
        assert _budget_for_agent(agent) == budget_for_context_window(131_072)

    def test_parent_main_unknown_context_length_matches_pre_port(self):
        agent = _agent(context_compressor=None)
        assert _budget_for_agent(agent) == budget_for_context_window(None)

    def test_exception_falls_back_to_default_budget(self):
        class _Explodes:
            @property
            def context_compressor(self):
                raise RuntimeError("boom")

            platform = "cli"
            _delegate_depth = 0
            is_subagent = False

        assert _budget_for_agent(_Explodes()) == DEFAULT_BUDGET

    def test_exception_in_subagent_path_also_falls_back_to_default(self):
        class _ExplodesSubagent:
            @property
            def context_compressor(self):
                raise RuntimeError("boom")

            platform = "subagent"

        assert _budget_for_agent(_ExplodesSubagent()) == DEFAULT_BUDGET


# ── read_file stays pinned to inf under the subagent budget ────────────

class TestSubagentBudgetPreservesInvariants:
    def test_read_file_still_inf_under_subagent_budget(self):
        budget = budget_for_subagent(131_072)
        assert budget.resolve_threshold("read_file") == float("inf")

    def test_subagent_budget_never_mutates_parent_default_budget(self):
        _budget_for_agent(_agent(platform="subagent"))
        # DEFAULT_BUDGET / budget_for_context_window output must be untouched
        # by a subagent call happening first (frozen dataclass, no shared
        # mutable state to worry about, but verify the invariant directly).
        assert budget_for_context_window(None) == DEFAULT_BUDGET
