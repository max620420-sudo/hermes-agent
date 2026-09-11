"""Configurable budget constants for tool result persistence.
Per-tool resolution: pinned > config overrides > registry > default."""

from dataclasses import dataclass, field
from typing import Dict

# Never overridden; read_file=inf prevents infinite persist->read->persist loops.
PINNED_THRESHOLDS: Dict[str, float] = {"read_file": float("inf")}

# Single source of truth for the defaults; tool_result_storage.py imports these.
DEFAULT_RESULT_SIZE_CHARS: int = 100_000
DEFAULT_TURN_BUDGET_CHARS: int = 200_000
DEFAULT_PREVIEW_SIZE_CHARS: int = 1_500

# Tighter per-result default for ``mcp_`` tools: MCP servers routinely return
# un-paginated 20-50K payloads that sail under the generic 100K threshold; spillover
# keeps the full payload on disk. Config: ``tool_budget.mcp_result_size_chars``.
DEFAULT_MCP_RESULT_SIZE_CHARS: int = 50_000
# Same prefix the untrusted-content wrapper keys on (agent/tool_dispatch_helpers.py).
MCP_TOOL_PREFIX: str = "mcp_"

# Tighter per-result cap for SUBAGENT sessions (see budget_for_subagent below).
# A subagent runs a bounded task loop whose whole history is re-sent on every
# model call -- no /compact, no user in the loop to trim the thread, and no
# rollover before the task ends. A single 30-60K-char tool result sails under
# the generic 100K per-result threshold and the 200K per-turn budget, stays
# verbatim in the active conversation, and is re-sent on every subsequent
# call for the rest of the task.
SUBAGENT_RESULT_SIZE_CHARS: int = 24_000
# Head+tail preview kept for a spilled subagent result: a subagent has no
# user to re-ask, so the spilled result must stay usable from the preview
# alone (start of the call + exit status / totals / traceback tail). These
# are CEILINGS -- budget_for_subagent scales them down proportionally to the
# effective (possibly context-window-shrunk) threshold so the preview never
# approaches the spill threshold itself.
SUBAGENT_PREVIEW_HEAD_CHARS: int = 6_000
SUBAGENT_PREVIEW_TAIL_CHARS: int = 3_000


def _configured_mcp_result_size() -> int:
    """Read ``tool_budget.mcp_result_size_chars`` via ``load_config_readonly`` (the
    sanctioned path; raw config.yaml parsing outside owner modules is test-guarded).
    Any error, missing key or non-positive value returns the built-in default.

    The ``tool_budget:`` block name is shared with the wider configurable-caps proposal (#80508) so the two
    can merge without a key rename.
    """
    try:
        from hermes_cli.config import load_config_readonly
        data = load_config_readonly()
        block = data.get("tool_budget") if isinstance(data, dict) else None
        raw = block.get("mcp_result_size_chars") if isinstance(block, dict) else None
        if raw is not None and int(raw) > 0:
            return int(raw)
    except Exception:
        pass
    return DEFAULT_MCP_RESULT_SIZE_CHARS


@dataclass(frozen=True)
class BudgetConfig:
    """Immutable budget constants: per-result threshold (``resolve_threshold``),
    per-turn aggregate (``turn_budget``) and inline snippet size (``preview_size``).

    ``preview_tail_size`` (0 by default) opts a config into a head+TAIL
    preview instead of the historical head-only one -- see
    ``budget_for_subagent`` and ``tools.tool_result_storage.generate_preview``.
    """

    default_result_size: int = DEFAULT_RESULT_SIZE_CHARS
    turn_budget: int = DEFAULT_TURN_BUDGET_CHARS
    preview_size: int = DEFAULT_PREVIEW_SIZE_CHARS
    preview_tail_size: int = 0
    mcp_result_size: int = DEFAULT_MCP_RESULT_SIZE_CHARS
    tool_overrides: Dict[str, int] = field(default_factory=dict)

    def resolve_threshold(self, tool_name: str) -> int | float:
        """Priority: pinned -> tool_overrides -> mcp_ prefix -> registry per-tool -> default.
        MCP tools get ``mcp_result_size`` (no registry entry). MCP and registry values
        are capped at ``default_result_size`` so a context-scaled budget for a small
        model still constrains tools registering a fixed 100K ``max_result_size_chars``.

        For the default budget this is a no-op because both equal 100K; for a scaled-down budget it prevents
        a per-tool registry value from re-inflating the cap past the model's window (#23767).
        """
        if tool_name in PINNED_THRESHOLDS:
            return PINNED_THRESHOLDS[tool_name]
        if tool_name in self.tool_overrides:
            return self.tool_overrides[tool_name]
        if tool_name.startswith(MCP_TOOL_PREFIX):
            return min(self.mcp_result_size, self.default_result_size)
        from tools.registry import registry
        registry_value = registry.get_max_result_size(tool_name, default=self.default_result_size)
        if registry_value == float("inf"):
            return registry_value
        return min(registry_value, self.default_result_size)


# Default config -- matches the historical hardcoded behavior exactly.
DEFAULT_BUDGET = BudgetConfig()

# Same rough 4-chars-per-token the estimator uses (agent/model_metadata.py);
# a smaller divisor would UNDER-protect small models.
_CHARS_PER_TOKEN: int = 4
# Window fraction ONE result / the WHOLE turn's tool output may occupy — well
# under 1.0 since system prompt, schemas, history and the reply all compete.
_PER_RESULT_WINDOW_FRACTION: float = 0.15
_PER_TURN_WINDOW_FRACTION: float = 0.30
# Floors so a tiny model still gets a usable result, never a 0-char budget.
_MIN_RESULT_SIZE_CHARS: int = 8_000
_MIN_TURN_BUDGET_CHARS: int = 16_000


def budget_for_context_window(context_length: int | None) -> BudgetConfig:
    """Return a BudgetConfig scaled to the model's context window: the fixed
    defaults suit 200K+ models but on 65K one result/turn can fill the window.
    The proportional value is clamped to the defaults as a CAP (large models
    stay byte-identical) and floored so a usable preview always survives.

    The fixed defaults (100K result / 200K turn chars) are correct for large (200K+ token) models but blind
    to small ones: on a 65K-token model a single tool result persisted at the 100K-char threshold, or a
    200K-char turn budget (~50K tokens), can by itself approach or exceed the whole window and force an
    oversized request (#23767).
    """
    mcp_result_size = _configured_mcp_result_size()
    if not context_length or context_length <= 0:
        if mcp_result_size == DEFAULT_MCP_RESULT_SIZE_CHARS:
            return DEFAULT_BUDGET
        return BudgetConfig(mcp_result_size=mcp_result_size)
    window_chars = context_length * _CHARS_PER_TOKEN
    return BudgetConfig(
        default_result_size=max(_MIN_RESULT_SIZE_CHARS, min(int(window_chars * _PER_RESULT_WINDOW_FRACTION), DEFAULT_RESULT_SIZE_CHARS)),
        turn_budget=max(_MIN_TURN_BUDGET_CHARS, min(int(window_chars * _PER_TURN_WINDOW_FRACTION), DEFAULT_TURN_BUDGET_CHARS)),
        preview_size=DEFAULT_PREVIEW_SIZE_CHARS,
        mcp_result_size=mcp_result_size,
    )


def budget_for_subagent(context_length: int | None) -> BudgetConfig:
    """Return the tool-result budget for a SUBAGENT session.

    Starts from :func:`budget_for_context_window` (so a small-model subagent
    still gets the context-scaled values), then tightens the per-result
    threshold to at most ``SUBAGENT_RESULT_SIZE_CHARS`` (24K) -- the fix for
    large results living verbatim in a subagent's active conversation and
    being re-sent on every subsequent model call.

    The preview stays proportional to the EFFECTIVE threshold (not the fixed
    24K ceiling): on a context-window-shrunk budget where the threshold
    itself lands below 24K, a fixed 6K/3K preview would approach or exceed
    the spill threshold, defeating the point of spilling. Scaling keeps the
    preview a small fraction of whatever the actual threshold ends up being
    (head <= threshold/4, tail <= threshold/8), e.g.:

        threshold 24K -> head 6K / tail 3K   (ceiling, unscaled)
        threshold 16K -> head 4K / tail 2K
        threshold  8K -> head 2K / tail 1K

    ``resolve_threshold`` caps per-tool registry values at
    ``default_result_size``, so tightening that one field also tightens
    every tool that registers a larger ``max_result_size_chars``.
    ``read_file`` stays pinned to ``inf`` (PINNED_THRESHOLDS) regardless, so
    paging back through a spilled file can never re-spill.

    Parent/main sessions never call this -- their budget is unchanged.
    """
    base = budget_for_context_window(context_length)
    effective_threshold = min(base.default_result_size, SUBAGENT_RESULT_SIZE_CHARS)
    effective_head = min(SUBAGENT_PREVIEW_HEAD_CHARS, max(1, effective_threshold // 4))
    effective_tail = min(SUBAGENT_PREVIEW_TAIL_CHARS, max(0, effective_threshold // 8))
    return BudgetConfig(
        default_result_size=effective_threshold,
        turn_budget=base.turn_budget,
        preview_size=effective_head,
        preview_tail_size=effective_tail,
        mcp_result_size=min(base.mcp_result_size, effective_threshold),
        tool_overrides=base.tool_overrides,
    )
