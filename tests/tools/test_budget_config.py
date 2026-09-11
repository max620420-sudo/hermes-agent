"""Unit tests for tools/budget_config.py.

Covers default values, resolve_threshold() priority chain
(pinned > tool_overrides > registry > default), immutability,
and the PINNED_THRESHOLDS escape-hatch for read_file.
"""

import dataclasses
import math
from unittest.mock import patch

import pytest

from tools.budget_config import (
    DEFAULT_BUDGET,
    DEFAULT_PREVIEW_SIZE_CHARS,
    DEFAULT_RESULT_SIZE_CHARS,
    DEFAULT_TURN_BUDGET_CHARS,
    PINNED_THRESHOLDS,
    SUBAGENT_PREVIEW_HEAD_CHARS,
    SUBAGENT_PREVIEW_TAIL_CHARS,
    SUBAGENT_RESULT_SIZE_CHARS,
    BudgetConfig,
    budget_for_context_window,
    budget_for_subagent,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Verify documented default values haven't drifted."""

    def test_default_result_size(self):
        assert DEFAULT_RESULT_SIZE_CHARS == 100_000


    def test_default_preview_size(self):
        assert DEFAULT_PREVIEW_SIZE_CHARS == 1_500


class TestPinnedThresholds:
    """PINNED_THRESHOLDS – tools whose values must never be overridden."""

    def test_read_file_is_inf(self):
        assert PINNED_THRESHOLDS["read_file"] == float("inf")
        assert math.isinf(PINNED_THRESHOLDS["read_file"])

    def test_pinned_is_not_empty(self):
        assert len(PINNED_THRESHOLDS) >= 1


# ---------------------------------------------------------------------------
# BudgetConfig defaults
# ---------------------------------------------------------------------------


class TestBudgetConfigDefaults:
    """BudgetConfig() should match the module-level defaults exactly."""

    def test_default_result_size(self):
        cfg = BudgetConfig()
        assert cfg.default_result_size == DEFAULT_RESULT_SIZE_CHARS


    def test_default_budget_singleton_matches(self):
        """DEFAULT_BUDGET should equal a freshly constructed BudgetConfig."""
        assert DEFAULT_BUDGET == BudgetConfig()


# ---------------------------------------------------------------------------
# Immutability (frozen=True)
# ---------------------------------------------------------------------------


class TestBudgetConfigFrozen:
    """Frozen dataclass must reject attribute mutation."""

    def test_cannot_set_default_result_size(self):
        cfg = BudgetConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.default_result_size = 999


    def test_cannot_set_tool_overrides(self):
        cfg = BudgetConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.tool_overrides = {"foo": 1}


# ---------------------------------------------------------------------------
# Custom construction
# ---------------------------------------------------------------------------


class TestBudgetConfigCustom:
    """BudgetConfig can be created with non-default values."""

    def test_custom_values(self):
        cfg = BudgetConfig(
            default_result_size=50_000,
            turn_budget=100_000,
            preview_size=500,
            tool_overrides={"my_tool": 42},
        )
        assert cfg.default_result_size == 50_000
        assert cfg.turn_budget == 100_000
        assert cfg.preview_size == 500
        assert cfg.tool_overrides == {"my_tool": 42}


# ---------------------------------------------------------------------------
# resolve_threshold() priority chain
# ---------------------------------------------------------------------------


class TestResolveThreshold:
    """Priority: pinned > tool_overrides > registry > default."""

    def test_pinned_wins_over_override(self):
        """Even if tool_overrides contains read_file, pinned value wins."""
        cfg = BudgetConfig(tool_overrides={"read_file": 1})
        result = cfg.resolve_threshold("read_file")
        assert result == float("inf")

    def test_tool_override_wins_over_default(self):
        """tool_overrides should be returned before falling back to registry."""
        cfg = BudgetConfig(tool_overrides={"my_tool": 42})
        result = cfg.resolve_threshold("my_tool")
        assert result == 42


    @patch("tools.registry.registry")
    def test_registry_value_capped_at_default(self, mock_registry):
        """A scaled-down budget caps an oversized registry value (#23767).

        web/terminal/x_search register max_result_size_chars=100_000; a small
        model's scaled budget must not be re-inflated by that.
        """
        mock_registry.get_max_result_size.return_value = 100_000
        cfg = BudgetConfig(default_result_size=30_000)
        assert cfg.resolve_threshold("web_search") == 30_000


    @patch("tools.registry.registry")
    def test_default_budget_unchanged_for_100k_tool(self, mock_registry):
        """Default budget keeps 100K registry tools at 100K (no behavior change)."""
        mock_registry.get_max_result_size.return_value = 100_000
        cfg = BudgetConfig()  # default_result_size == 100_000
        assert cfg.resolve_threshold("web_search") == 100_000


# ---------------------------------------------------------------------------
# budget_for_context_window() — context-aware scaling (#23767)
# ---------------------------------------------------------------------------


class TestBudgetForContextWindow:
    """Scaling the tool-output budget to the active model's context window."""

    def test_none_returns_default(self):
        assert budget_for_context_window(None) is DEFAULT_BUDGET

    def test_zero_or_negative_returns_default(self):
        assert budget_for_context_window(0) is DEFAULT_BUDGET
        assert budget_for_context_window(-5) is DEFAULT_BUDGET


    def test_scaled_budget_constrains_oversized_result(self):
        """A 279K-char result against a 65K model exceeds the scaled per-result
        threshold, so it will be persisted/truncated rather than sent whole."""
        cfg = budget_for_context_window(65_536)
        huge_len = 279_549
        threshold = cfg.resolve_threshold("mcp_firecrawl_firecrawl_search")
        assert threshold < huge_len
        assert cfg.default_result_size < huge_len


# ---------------------------------------------------------------------------
# MCP-prefix threshold (mcp_result_size)
# ---------------------------------------------------------------------------


class TestMcpPrefixThreshold:
    """mcp_* tools get the tighter 50K default, config-overridable."""

    def test_default_mcp_threshold_is_50k(self):
        from tools.budget_config import DEFAULT_MCP_RESULT_SIZE_CHARS
        assert DEFAULT_MCP_RESULT_SIZE_CHARS == 50_000
        assert DEFAULT_BUDGET.resolve_threshold("mcp_composio_search_tools") == 50_000

    def test_non_mcp_tools_keep_generic_default(self):
        assert DEFAULT_BUDGET.resolve_threshold("some_random_tool") == DEFAULT_RESULT_SIZE_CHARS

    def test_pinned_wins_over_mcp_prefix(self):
        with patch.dict(PINNED_THRESHOLDS, {"mcp_pinned_tool": float("inf")}):
            assert DEFAULT_BUDGET.resolve_threshold("mcp_pinned_tool") == float("inf")

    def test_tool_override_wins_over_mcp_prefix(self):
        cfg = BudgetConfig(tool_overrides={"mcp_special": 75_000})
        assert cfg.resolve_threshold("mcp_special") == 75_000

    def test_mcp_threshold_capped_by_scaled_default(self):
        """On a small model the scaled default_result_size caps the MCP value."""
        cfg = BudgetConfig(default_result_size=20_000, mcp_result_size=50_000)
        assert cfg.resolve_threshold("mcp_anything") == 20_000

    def test_mcp_threshold_never_exceeds_default_result_size(self):
        cfg = BudgetConfig(default_result_size=100_000, mcp_result_size=999_999)
        assert cfg.resolve_threshold("mcp_anything") == 100_000


# ---------------------------------------------------------------------------
# budget_for_subagent() -- Task 4 semantic port
# ---------------------------------------------------------------------------


class TestBudgetForSubagent:
    """Effective threshold clamps to SUBAGENT_RESULT_SIZE_CHARS (24K); the
    head/tail preview scales proportionally to the EFFECTIVE threshold, not
    the fixed ceiling, so preview never approaches the spill threshold on a
    context-window-shrunk budget."""

    def test_large_window_hits_the_24k_ceiling(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # no config.yaml
        cfg = budget_for_subagent(None)  # base default_result_size == 100_000
        assert cfg.default_result_size == 24_000 == SUBAGENT_RESULT_SIZE_CHARS
        assert cfg.preview_size == 6_000 == SUBAGENT_PREVIEW_HEAD_CHARS
        assert cfg.preview_tail_size == 3_000 == SUBAGENT_PREVIEW_TAIL_CHARS

    def test_16k_effective_threshold_scales_preview_to_4k_2k(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        base = budget_for_context_window(26_667)
        assert base.default_result_size == 16_000  # sanity-check the fixture
        cfg = budget_for_subagent(26_667)
        assert cfg.default_result_size == 16_000
        assert cfg.preview_size == 4_000
        assert cfg.preview_tail_size == 2_000

    def test_8k_effective_threshold_scales_preview_to_2k_1k(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        base = budget_for_context_window(10_000)
        assert base.default_result_size == 8_000  # sanity-check the fixture (floor)
        cfg = budget_for_subagent(10_000)
        assert cfg.default_result_size == 8_000
        assert cfg.preview_size == 2_000
        assert cfg.preview_tail_size == 1_000

    def test_preview_never_approaches_the_spill_threshold(self, tmp_path, monkeypatch):
        """head + tail must stay well under the threshold at every scale
        tested above -- the whole point of scaling instead of using a fixed
        6K/3K preview on a shrunk budget."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        for ctx_len in (None, 26_667, 10_000):
            cfg = budget_for_subagent(ctx_len)
            assert cfg.preview_size + cfg.preview_tail_size <= cfg.default_result_size

    def test_read_file_stays_pinned_to_inf(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_subagent(None)
        assert cfg.resolve_threshold("read_file") == float("inf")
        assert math.isinf(cfg.resolve_threshold("read_file"))

    def test_mcp_result_size_capped_by_effective_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_subagent(10_000)  # effective_threshold == 8_000
        assert cfg.mcp_result_size == 8_000
        assert cfg.resolve_threshold("mcp_anything") == 8_000

    def test_turn_budget_and_tool_overrides_carried_through_from_base(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        base = budget_for_context_window(26_667)
        cfg = budget_for_subagent(26_667)
        assert cfg.turn_budget == base.turn_budget
        assert cfg.tool_overrides == base.tool_overrides

    def test_result_is_frozen_like_any_other_budget_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_subagent(None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.default_result_size = 1

    def test_config_override_via_hermes_home(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  mcp_result_size_chars: 30000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(None)
        assert cfg.resolve_threshold("mcp_composio_multi_execute") == 30_000
        # Generic tools are untouched by the MCP knob.
        assert cfg.default_result_size == DEFAULT_RESULT_SIZE_CHARS

    def test_config_override_survives_window_scaling(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text(
            "tool_budget:\n  mcp_result_size_chars: 30000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(200_000)
        assert cfg.mcp_result_size == 30_000

    def test_malformed_config_falls_back_to_default(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text("tool_budget: not-a-mapping\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = budget_for_context_window(None)
        assert cfg.resolve_threshold("mcp_x_y") == 50_000

    def test_scaled_small_window_caps_mcp_threshold(self, tmp_path, monkeypatch):
        """A tiny model's scaled default_result_size caps even the MCP value."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # no config.yaml
        cfg = budget_for_context_window(16_384)  # scaled default < 50K
        assert cfg.default_result_size < 50_000
        assert cfg.resolve_threshold("mcp_tool") == cfg.default_result_size
