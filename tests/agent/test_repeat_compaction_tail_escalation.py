"""Repeated-compaction tail-budget escalation (Task 6 semantic port).

A session that keeps re-compacting (landing well above its compaction floor
every pass, so it re-compacts a near-floor transcript again a few turns
later) shrinks its verbatim tail budget further on each ADDITIONAL
same-session completed compaction, on top of the existing
LEAN_TAIL_FLOOR/CAP calculation. Round 0 (no completed compaction yet) is
byte-identical to the pre-escalation behavior.

Covers:
  * ``ContextCompressor._escalated_tail_token_budget`` -- pure escalation math
  * ``ContextCompressor.tail_token_budget`` -- escalation applied at access
    time, never baked into the cached base
  * ``ContextCompressor.record_completed_compaction`` -- round increment,
    including the feasibility-skip path
  * ``ContextCompressor._reset_session_compaction_state`` /
    ``bind_session_state`` -- durable round-count reset/reload, including
    across compressor object recreation (gateway rebind) and multi-session
    isolation
  * ``hermes_state_compression`` get/set_compression_round_count`` +
    schema auto-reconciliation for the new column
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.context_compressor import (
    LEAN_TAIL_CAP_TOKENS,
    LEAN_TAIL_FLOOR_TOKENS,
    _REPEAT_TAIL_MIN_TOKENS,
    ContextCompressor,
)
from hermes_state import SessionDB


def _compressor(context_length: int, db: SessionDB | None = None, session_id: str = "") -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=context_length):
        cc = ContextCompressor(
            model="test/model", threshold_percent=0.85, protect_first_n=2, protect_last_n=2,
            quiet_mode=True,
        )
        # context_length resolution is deferred to first access (no sync probe
        # during __init__) and then cached forever -- force that resolution
        # NOW, while the patch is still active, so every later access outside
        # this function (the whole point of returning cc) sees the fixed
        # context_length instead of falling through to the real resolver.
        assert cc.context_length == context_length
    if db is not None:
        cc.bind_session_state(db, session_id)
    return cc


def _db(tmp_path: Path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


# ── 1: round 0 byte-identical to pre-escalation base ────────────────────

class TestRoundZeroUnchanged:
    def test_large_window_round_zero_equals_cap(self):
        cc = _compressor(context_length=1_000_000)  # base -> LEAN_TAIL_CAP_TOKENS (25K)
        assert cc.tail_token_budget == LEAN_TAIL_CAP_TOKENS == 25_000

    def test_small_window_round_zero_equals_floor(self):
        cc = _compressor(context_length=100_000)  # base -> LEAN_TAIL_FLOOR_TOKENS (10K)
        assert cc.tail_token_budget == LEAN_TAIL_FLOOR_TOKENS == 10_000

    def test_escalation_helper_is_noop_at_round_zero(self):
        cc = _compressor(context_length=1_000_000)
        assert cc._session_compaction_rounds == 0
        assert cc._escalated_tail_token_budget(25_000) == 25_000
        assert cc._escalated_tail_token_budget(10_000) == 10_000


# ── 2-3: escalation math at both window sizes ───────────────────────────

class TestEscalationMath:
    def test_base_25k_escalates_12500_then_6250_then_floors(self):
        cc = _compressor(context_length=1_000_000)
        assert cc.tail_token_budget == 25_000
        cc._session_compaction_rounds = 1
        assert cc.tail_token_budget == 12_500
        cc._session_compaction_rounds = 2
        assert cc.tail_token_budget == 6_250
        cc._session_compaction_rounds = 3
        assert cc.tail_token_budget == 6_250  # capped at _REPEAT_TAIL_MAX_ROUNDS=2
        cc._session_compaction_rounds = 10
        assert cc.tail_token_budget == 6_250

    def test_base_10k_escalates_to_floor_and_stays(self):
        cc = _compressor(context_length=100_000)
        assert cc.tail_token_budget == 10_000
        cc._session_compaction_rounds = 1
        assert cc.tail_token_budget == 6_000  # 10_000 * 0.5 = 5000, floored to 6000
        cc._session_compaction_rounds = 2
        assert cc.tail_token_budget == 6_000
        cc._session_compaction_rounds = 5
        assert cc.tail_token_budget == 6_000


# ── 4: base already at/below the floor -> no-op ─────────────────────────

class TestBelowFloorNoop:
    def test_base_at_floor_unaffected_by_rounds(self):
        cc = _compressor(context_length=1_000_000)
        cc._session_compaction_rounds = 5
        assert cc._escalated_tail_token_budget(_REPEAT_TAIL_MIN_TOKENS) == _REPEAT_TAIL_MIN_TOKENS

    def test_base_below_floor_unaffected_by_rounds(self):
        cc = _compressor(context_length=1_000_000)
        cc._session_compaction_rounds = 5
        assert cc._escalated_tail_token_budget(3_000) == 3_000


# ── 5-6: record_completed_compaction increments the round ──────────────

class TestRecordCompletedCompactionIncrementsRound:
    def test_normal_completion_increments_round(self):
        cc = _compressor(context_length=1_000_000)
        assert cc._session_compaction_rounds == 0
        cc.record_completed_compaction()
        assert cc._session_compaction_rounds == 1
        cc.record_completed_compaction()
        assert cc._session_compaction_rounds == 2

    def test_used_fallback_completion_increments_round(self):
        cc = _compressor(context_length=1_000_000)
        cc.record_completed_compaction(used_fallback=True)
        assert cc._session_compaction_rounds == 1

    def test_feasibility_skip_completion_still_increments_round(self):
        """A feasibility-skip boundary is streak-neutral for the fallback
        streak but IS a completed boundary for repeated-tail purposes -- the
        next compaction on this session is still a repeat."""
        cc = _compressor(context_length=1_000_000)
        cc.record_completed_compaction(feasibility_skip=True)
        assert cc._session_compaction_rounds == 1
        # And the streak-neutral contract is untouched.
        assert cc._fallback_compression_streak == 0


# ── 7: in-memory reset ───────────────────────────────────────────────────

class TestInMemoryReset:
    def test_reset_zeroes_in_memory_round_count(self):
        cc = _compressor(context_length=1_000_000)
        cc.record_completed_compaction()
        cc.record_completed_compaction()
        assert cc._session_compaction_rounds == 2
        cc._reset_session_compaction_state()
        assert cc._session_compaction_rounds == 0


# ── 8: durable zero survives same-session_id rebind (the critical guarantee) ──

class TestDurableResetPreventsRevival:
    def test_reset_then_rebind_same_session_id_durable_round_is_zero(self, tmp_path):
        db = _db(tmp_path)
        db.create_session("s1", source="cli")
        cc = _compressor(context_length=1_000_000, db=db, session_id="s1")
        cc.record_completed_compaction()
        cc.record_completed_compaction()
        assert db.get_compression_round_count("s1") == 2

        # /reset reusing the SAME session_id.
        cc._reset_session_compaction_state()
        assert db.get_compression_round_count("s1") == 0  # durable zero, not just in-memory

        # A brand-new compressor object rebinding to that same id (gateway
        # rebuild) must NOT resurrect the old round count.
        cc2 = _compressor(context_length=1_000_000, db=db, session_id="s1")
        assert cc2._session_compaction_rounds == 0
        assert cc2.tail_token_budget == 25_000  # round-0 behavior restored


# ── 9: two sessions round-trip independently ────────────────────────────

class TestMultiSessionIsolation:
    def test_two_sessions_independent_round_counts(self, tmp_path):
        db = _db(tmp_path)
        db.create_session("sess-a", source="cli")
        db.create_session("sess-b", source="cli")

        cc_a = _compressor(context_length=1_000_000, db=db, session_id="sess-a")
        cc_a.record_completed_compaction()
        cc_a.record_completed_compaction()

        cc_b = _compressor(context_length=1_000_000, db=db, session_id="sess-b")
        # sess-b never compacted.

        assert db.get_compression_round_count("sess-a") == 2
        assert db.get_compression_round_count("sess-b") == 0

        # Rebind fresh compressor objects to each and confirm independent reload.
        fresh_a = _compressor(context_length=1_000_000, db=db, session_id="sess-a")
        fresh_b = _compressor(context_length=1_000_000, db=db, session_id="sess-b")
        assert fresh_a._session_compaction_rounds == 2
        assert fresh_b._session_compaction_rounds == 0
        assert fresh_a.tail_token_budget == 6_250
        assert fresh_b.tail_token_budget == 25_000


# ── 10: durable round survives compressor object recreation ────────────

class TestGatewayRebindRecovery:
    def test_new_compressor_object_recovers_round_from_db(self, tmp_path):
        """The gateway rebuilds the compressor object every turn / cache
        eviction; the round count must survive that, unlike a naive
        in-memory-only counter."""
        db = _db(tmp_path)
        db.create_session("sess-gw", source="cli")

        cc1 = _compressor(context_length=1_000_000, db=db, session_id="sess-gw")
        cc1.record_completed_compaction()
        cc1.record_completed_compaction()
        cc1.record_completed_compaction()
        del cc1  # simulate the gateway discarding the compressor object

        cc2 = _compressor(context_length=1_000_000, db=db, session_id="sess-gw")
        assert cc2._session_compaction_rounds == 3
        assert cc2.tail_token_budget == 6_250  # round-3 == round-2 (capped)


# ── 11: no session_db -> safe in-memory-only operation ──────────────────

class TestNoSessionDb:
    def test_unbound_compressor_works_in_memory_only(self):
        cc = _compressor(context_length=1_000_000)  # never bound to a DB
        cc.record_completed_compaction()
        cc.record_completed_compaction()
        assert cc._session_compaction_rounds == 2
        assert cc.tail_token_budget == 6_250
        cc._reset_session_compaction_state()  # must not raise without a session_db
        assert cc._session_compaction_rounds == 0


# ── 12: cache does not stick -- round change reflected immediately ─────

class TestCacheDoesNotStick:
    def test_tail_budget_reflects_round_increase_without_re_resolving_base(self):
        cc = _compressor(context_length=1_000_000)
        first = cc.tail_token_budget  # forces the base cache to populate
        assert first == 25_000
        assert cc._tail_token_budget == 25_000  # base cache: unescalated round-0 value

        cc.record_completed_compaction()
        second = cc.tail_token_budget
        assert second == 12_500
        # The CACHED base must remain the round-0 value -- only the returned,
        # escalated number changes.
        assert cc._tail_token_budget == 25_000


# ── 13: 3 consecutive completed compactions -> monotonic decrease ──────

class TestThreeConsecutiveCompactions:
    def test_tail_budget_monotonically_decreases_across_three_rounds(self, tmp_path):
        db = _db(tmp_path)
        db.create_session("sess-mono", source="cli")
        cc = _compressor(context_length=1_000_000, db=db, session_id="sess-mono")

        landings = []
        for _ in range(3):
            landings.append(cc.tail_token_budget)
            cc.record_completed_compaction()
        landings.append(cc.tail_token_budget)  # after the 3rd completed round

        assert landings == [25_000, 12_500, 6_250, 6_250]
        assert landings[0] > landings[1] > landings[2] == landings[3]


# ── DB layer: get/set_compression_round_count ───────────────────────────

class TestDurableRoundCountApi:
    def test_default_is_zero(self, tmp_path):
        db = _db(tmp_path)
        db.create_session("s1", source="cli")
        assert db.get_compression_round_count("s1") == 0

    def test_set_then_get_round_trips(self, tmp_path):
        db = _db(tmp_path)
        db.create_session("s1", source="cli")
        db.set_compression_round_count("s1", 4)
        assert db.get_compression_round_count("s1") == 4

    def test_negative_is_clamped_to_zero(self, tmp_path):
        db = _db(tmp_path)
        db.create_session("s1", source="cli")
        db.set_compression_round_count("s1", -5)
        assert db.get_compression_round_count("s1") == 0

    def test_empty_session_id_is_a_noop(self, tmp_path):
        db = _db(tmp_path)
        db.set_compression_round_count("", 3)  # must not raise
        assert db.get_compression_round_count("") == 0


# ── 19: legacy DB missing the column auto-reconciles on open ───────────

class TestSchemaAutoReconciliation:
    def test_legacy_db_missing_column_gets_it_added_with_default_zero(self, tmp_path):
        db_path = tmp_path / "legacy.db"
        db = SessionDB(db_path=db_path)
        db.create_session("s1", source="cli")
        db.set_compression_round_count("s1", 3)

        # Simulate a pre-migration DB file by dropping the column outright.
        with db._lock:
            db._conn.execute("ALTER TABLE sessions DROP COLUMN compression_round_count")
            db._conn.commit()
        db.close()

        # Reopening (read_only=False) must reconcile the schema: ADD COLUMN
        # back with its DEFAULT 0, no exception, no data loss elsewhere.
        db2 = SessionDB(db_path=db_path)
        assert db2.get_compression_round_count("s1") == 0  # column recreated at its default
        db2.set_compression_round_count("s1", 7)
        assert db2.get_compression_round_count("s1") == 7


# ── 14-16: rolling summary / latest user message / protect_last_n intact ──

class TestProtectedRegionsUnaffectedByEscalation:
    def test_escalation_does_not_touch_rolling_summary_state(self):
        cc = _compressor(context_length=1_000_000)
        cc._previous_summary = "## Goal\nship the port\n"
        cc.record_completed_compaction()
        cc.record_completed_compaction()
        # Escalation only touches tail_token_budget / round bookkeeping.
        assert cc._previous_summary == "## Goal\nship the port\n"

    def test_protect_last_n_and_protect_first_n_unaffected(self):
        cc = _compressor(context_length=1_000_000)
        assert cc.protect_first_n == 2 and cc.protect_last_n == 2
        cc.record_completed_compaction()
        cc.record_completed_compaction()
        cc.record_completed_compaction()
        assert cc.protect_first_n == 2 and cc.protect_last_n == 2

    def test_tail_mode_lean_constants_untouched(self):
        # LEAN_TAIL_FLOOR/CAP themselves must never change value -- the
        # explicit "do not touch" invariant from the design review.
        assert LEAN_TAIL_FLOOR_TOKENS == 10_000
        assert LEAN_TAIL_CAP_TOKENS == 25_000


# ── 18: round 0 behavior identical for every session kind ──────────────

class TestRoundZeroIdenticalAcrossSessionKinds:
    @pytest.mark.parametrize("platform", ["cli", "gateway", "subagent"])
    def test_round_zero_tail_budget_same_regardless_of_platform_label(self, platform):
        # tail_token_budget/escalation has no platform branch at all -- this
        # pins that a platform string (parent/main vs subagent) never enters
        # the calculation; round-0 output is identical either way.
        cc = _compressor(context_length=1_000_000)
        cc.platform = platform  # attribute is inert here; compressor doesn't read it
        assert cc.tail_token_budget == 25_000
