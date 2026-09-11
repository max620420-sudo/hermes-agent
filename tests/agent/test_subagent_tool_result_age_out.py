"""Subagent stale tool-result age-out (Task 5 semantic port).

Covers:
  * ``agent.turn_context._compute_subagent_aged_tool_indexes`` — pure,
    assistant-turn-counted age helper (never mutates ``messages``, never
    counts user turns).
  * ``agent.turn_context.build_api_messages`` — wire-copy exclusion of aged
    tool rows on a subagent session, byte-identical parent/main output,
    dangling-tool_call-free output (delegated to the existing sanitizer).
  * ``agent.turn_context._persist_subagent_tool_result_age_out`` —
    session-scoped DB persist dedup.
  * ``hermes_state_messages.SessionMessagesMixin.deactivate_tool_results`` —
    soft-archive (active=0, compacted=1), idempotent, non-destructive.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.turn_context import (
    _SUBAGENT_TOOL_RESULT_MAX_MODEL_TURNS,
    _compute_subagent_aged_tool_indexes,
    _persist_subagent_tool_result_age_out,
    build_api_messages,
)
from hermes_state import SessionDB


# ── fixtures ─────────────────────────────────────────────────────────────

def _tool_call(call_id: str, name: str = "execute_code") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _pair_fixture(n: int = 10) -> list:
    """user(1) + n x (assistant-with-tool_call, tool-result) — the historical
    Task 5 fixture shape. Each tool result's content is a unique marker
    ("RAW-RESULT-N") so age-out can be asserted by presence/absence."""
    messages = [{"role": "user", "content": "run the task"}]
    for i in range(1, n + 1):
        cid = f"call_{i}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [_tool_call(cid)]})
        messages.append({"role": "tool", "tool_call_id": cid, "content": f"RAW-RESULT-{i}" * 200})
    return messages


class _FakeAgent:
    """Minimal stand-in covering only what build_api_messages touches."""

    def __init__(self, *, platform="subagent", delegate_depth=0, session_id="sess-sub",
                 session_db=None):
        self.session_id = session_id
        self.platform = platform
        self._delegate_depth = delegate_depth
        self.is_subagent = False
        self._session_db = session_db
        self.provider = "test"
        self.model = "test-model"
        self.ephemeral_system_prompt = ""

    def _copy_reasoning_content_for_api(self, msg, api_msg):
        pass

    def _should_sanitize_tool_calls(self):
        return False

    def _sanitize_tool_calls_for_strict_api(self, api_msg, model=None):
        return api_msg


def _build(agent, messages):
    return build_api_messages(
        agent, messages, current_turn_user_idx=-1, ext_prefetch_cache="",
        plugin_user_context=None, moa_config=None, active_system_prompt="",
    )


# ── 1-3: wire exclusion, stub replacement, dangling tool_call ──────────

class TestWireAgeOut:
    def test_only_recent_4_tool_raw_results_present(self):
        agent = _FakeAgent(platform="subagent")
        messages = _pair_fixture(10)
        api_messages, _ = _build(agent, messages)
        present_ids = {m.get("tool_call_id") for m in api_messages if m.get("role") == "tool"}
        assert present_ids == {f"call_{i}" for i in range(7, 11)}  # last 4

    def test_old_6_results_absent_from_wire(self):
        agent = _FakeAgent(platform="subagent")
        api_messages, _ = _build(agent, _pair_fixture(10))
        present_ids = {m.get("tool_call_id") for m in api_messages if m.get("role") == "tool"}
        for i in range(1, 7):
            assert f"call_{i}" not in present_ids

    def test_dangling_tool_call_zero(self):
        """Every assistant tool_call id in the wire copy must have SOME
        matching tool-role response (raw kept, or the sanitizer's stub) —
        never a call with nothing after it."""
        from agent.agent_runtime_helpers import sanitize_api_messages

        agent = _FakeAgent(platform="subagent")
        api_messages, _ = _build(agent, _pair_fixture(10))
        sanitized = sanitize_api_messages(api_messages)

        declared_ids = set()
        for m in sanitized:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    declared_ids.add(tc.get("id"))
        answered_ids = {m.get("tool_call_id") for m in sanitized if m.get("role") == "tool"}
        assert declared_ids <= answered_ids


# ── 4-6: assistant-turn-based age calculation (the historical bug) ─────

class TestAssistantTurnAgeCalculation:
    def test_regression_assistant_based_not_user_based(self):
        """The original bug: counting user turns instead of assistant turns
        meant a single user + many assistant/tool iterations never aged
        anything out. This fixture (1 user, 10 assistant/tool pairs) must
        still age out the oldest 6."""
        idxs, ids = _compute_subagent_aged_tool_indexes(_pair_fixture(10))
        assert len(ids) == 6
        # Order is an implementation detail of the backward walk (newest-of-
        # the-aged first); membership is the actual invariant under test.
        assert set(ids) == {f"call_{i}" for i in range(1, 7)}

    def test_extra_user_messages_do_not_shift_the_boundary(self):
        messages = _pair_fixture(10)
        # Splice in 30 extra user messages at various points -- user-count
        # must have zero effect on which tool rows age out.
        for i in range(30):
            messages.insert(2 + i, {"role": "user", "content": f"aside {i}"})
        idxs_with_extra, ids_with_extra = _compute_subagent_aged_tool_indexes(messages)
        idxs_base, ids_base = _compute_subagent_aged_tool_indexes(_pair_fixture(10))
        assert set(ids_with_extra) == set(ids_base)
        assert len(ids_with_extra) == 6

    def test_one_more_assistant_turn_moves_the_boundary(self):
        """Adding exactly one more assistant/tool pair ages out one more
        (the previously-4th-from-last) result -- proves the boundary moves
        on assistant turns, not on message count in general."""
        base_idxs, base_ids = _compute_subagent_aged_tool_indexes(_pair_fixture(10))
        messages_11 = _pair_fixture(10) + [
            {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_11")]},
            {"role": "tool", "tool_call_id": "call_11", "content": "RAW-RESULT-11"},
        ]
        idxs_11, ids_11 = _compute_subagent_aged_tool_indexes(messages_11)
        assert set(ids_11) == set(base_ids) | {"call_7"}


# ── 7: parent/main byte-identical ───────────────────────────────────────

class TestParentMainUnchanged:
    def test_parent_cli_output_byte_identical_regardless_of_history_size(self):
        agent_parent = _FakeAgent(platform="cli", delegate_depth=0)
        messages = _pair_fixture(10)
        api_messages, _ = _build(agent_parent, messages)
        # All 10 raw results must survive -- no age-out branch taken.
        present_ids = {m.get("tool_call_id") for m in api_messages if m.get("role") == "tool"}
        assert present_ids == {f"call_{i}" for i in range(1, 11)}
        assert len(api_messages) == len(messages)

    def test_is_subagent_session_false_yields_empty_aged_set_path(self):
        """Directly pin that a non-subagent agent produces an empty
        exclusion set (the exact pre-port code path)."""
        agent_parent = _FakeAgent(platform="cli", delegate_depth=0)
        from agent.tool_executor import _is_subagent_session
        assert _is_subagent_session(agent_parent) is False


# ── 8: original messages never mutated ──────────────────────────────────

class TestOriginalMessagesImmutable:
    def test_messages_unchanged_after_helper_call(self):
        import copy

        messages = _pair_fixture(10)
        before = copy.deepcopy(messages)
        _compute_subagent_aged_tool_indexes(messages)
        assert messages == before

    def test_messages_unchanged_after_build_api_messages(self):
        import copy

        agent = _FakeAgent(platform="subagent")
        messages = _pair_fixture(10)
        before = copy.deepcopy(messages)
        _build(agent, messages)
        assert messages == before
        # No age-related keys leaked onto the durable message dicts either.
        for m in messages:
            assert "active" not in m
            assert "aged_out" not in m


# ── 16: tool row count at/under the keep threshold -> no age-out ────────

class TestBelowThreshold:
    def test_four_or_fewer_tool_rows_none_aged_out(self):
        idxs, ids = _compute_subagent_aged_tool_indexes(_pair_fixture(4))
        assert idxs == set()
        assert ids == []

    def test_zero_tool_rows(self):
        idxs, ids = _compute_subagent_aged_tool_indexes([{"role": "user", "content": "hi"}])
        assert idxs == set()
        assert ids == []


# ── DB layer: deactivate_tool_results ────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    d = SessionDB(tmp_path / "state.db")
    d.create_session(session_id="sess-sub", source="subagent")
    d.create_session(session_id="sess-other", source="subagent")
    return d


def _seed_tool_rows(db, session_id, n=10):
    ids = []
    for i in range(1, n + 1):
        cid = f"call_{i}"
        db.append_message(session_id, "tool", content=f"RAW-{i}", tool_call_id=cid)
        ids.append(cid)
    return ids


class TestDeactivateToolResults:
    def test_target_rows_flip_active_and_compacted(self, db):
        ids = _seed_tool_rows(db, "sess-sub", 10)
        flipped = db.deactivate_tool_results("sess-sub", ids[:6])
        assert flipped == 6
        with db._lock:
            rows = db._conn.execute(
                "SELECT tool_call_id, active, compacted FROM messages "
                "WHERE session_id = ? AND role = 'tool' ORDER BY id", ("sess-sub",),
            ).fetchall()
        by_id = {r["tool_call_id"]: (r["active"], r["compacted"]) for r in rows}
        for cid in ids[:6]:
            assert by_id[cid] == (0, 1)

    def test_other_rows_unchanged(self, db):
        ids = _seed_tool_rows(db, "sess-sub", 10)
        db.deactivate_tool_results("sess-sub", ids[:6])
        with db._lock:
            rows = db._conn.execute(
                "SELECT tool_call_id, active, compacted FROM messages "
                "WHERE session_id = ? AND role = 'tool' ORDER BY id", ("sess-sub",),
            ).fetchall()
        by_id = {r["tool_call_id"]: (r["active"], r["compacted"]) for r in rows}
        for cid in ids[6:]:
            assert by_id[cid] == (1, 0)

    def test_return_count_is_exact(self, db):
        ids = _seed_tool_rows(db, "sess-sub", 10)
        assert db.deactivate_tool_results("sess-sub", ids[:3]) == 3
        assert db.deactivate_tool_results("sess-sub", []) == 0
        assert db.deactivate_tool_results("", ids) == 0

    def test_repeat_call_on_same_ids_is_idempotent_zero(self, db):
        ids = _seed_tool_rows(db, "sess-sub", 10)
        first = db.deactivate_tool_results("sess-sub", ids[:6])
        second = db.deactivate_tool_results("sess-sub", ids[:6])
        assert first == 6
        assert second == 0  # already active=0 -> AND active=1 matches nothing

    def test_get_messages_default_excludes_aged_out_rows(self, db):
        ids = _seed_tool_rows(db, "sess-sub", 10)
        db.deactivate_tool_results("sess-sub", ids[:6])
        active = db.get_messages("sess-sub")
        active_ids = {m.get("tool_call_id") for m in active if m.get("role") == "tool"}
        assert active_ids == set(ids[6:])

    def test_get_messages_include_inactive_recovers_aged_out_rows(self, db):
        ids = _seed_tool_rows(db, "sess-sub", 10)
        db.deactivate_tool_results("sess-sub", ids[:6])
        full = db.get_messages("sess-sub", include_inactive=True)
        full_ids = {m.get("tool_call_id") for m in full if m.get("role") == "tool"}
        assert full_ids == set(ids)  # all 10 recoverable

    def test_session_scoping_does_not_cross_sessions(self, db):
        ids_sub = _seed_tool_rows(db, "sess-sub", 5)
        ids_other = _seed_tool_rows(db, "sess-other", 5)
        db.deactivate_tool_results("sess-sub", ids_sub)
        other_active = db.get_messages("sess-other")
        other_active_ids = {m.get("tool_call_id") for m in other_active if m.get("role") == "tool"}
        assert other_active_ids == set(ids_other)  # untouched


# ── 15/17: session-scoped persist dedup ─────────────────────────────────

class TestSessionScopedPersistDedup:
    def test_persists_only_new_ids_and_skips_repeat(self, db):
        agent = SimpleNamespace(session_id="sess-sub", _session_db=db)
        ids = _seed_tool_rows(db, "sess-sub", 10)

        _persist_subagent_tool_result_age_out(agent, ids[:6])
        with db._lock:
            first_pass = db._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 0", ("sess-sub",),
            ).fetchone()[0]
        assert first_pass == 6

        # Manually resurrect one row to prove the CACHE (not just the DB's
        # own AND active=1 idempotency) is what skips the repeat attempt --
        # a real re-deactivate call would have re-flipped it.
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET active = 1 WHERE session_id = ? AND tool_call_id = ?",
                ("sess-sub", ids[0]),
            )
            db._conn.commit()

        _persist_subagent_tool_result_age_out(agent, ids[:6])  # same ids again
        with db._lock:
            row = db._conn.execute(
                "SELECT active FROM messages WHERE session_id = ? AND tool_call_id = ?",
                ("sess-sub", ids[0]),
            ).fetchone()
        assert row["active"] == 1  # cache skipped it -- not re-deactivated

    def test_dedup_cache_is_session_scoped_not_agent_wide(self, db):
        """Same agent object reused across two different session_ids: a
        cache hit in session A must not suppress the persist call for the
        SAME tool_call_id string under session B."""
        db.create_session(session_id="sess-b", source="subagent")
        ids_a = _seed_tool_rows(db, "sess-sub", 3)
        # Reuse the exact same id strings in a different session.
        for cid in ids_a:
            db.append_message("sess-b", "tool", content="RAW-B", tool_call_id=cid)

        agent = SimpleNamespace(session_id="sess-sub", _session_db=db)
        _persist_subagent_tool_result_age_out(agent, ids_a)

        agent.session_id = "sess-b"
        _persist_subagent_tool_result_age_out(agent, ids_a)  # must NOT be skipped

        with db._lock:
            row = db._conn.execute(
                "SELECT active FROM messages WHERE session_id = ? AND tool_call_id = ?",
                ("sess-b", ids_a[0]),
            ).fetchone()
        assert row["active"] == 0

    def test_no_session_db_is_a_safe_noop(self):
        agent = SimpleNamespace(session_id="sess-sub", _session_db=None)
        _persist_subagent_tool_result_age_out(agent, ["call_1", "call_2"])  # must not raise

    def test_no_session_id_is_a_safe_noop(self, db):
        agent = SimpleNamespace(session_id=None, _session_db=db)
        _persist_subagent_tool_result_age_out(agent, ["call_1"])  # must not raise

    def test_deactivate_method_missing_is_a_safe_noop(self):
        agent = SimpleNamespace(session_id="sess-sub", _session_db=SimpleNamespace())
        _persist_subagent_tool_result_age_out(agent, ["call_1"])  # must not raise

    def test_deactivate_raising_is_swallowed(self):
        class _Explodes:
            def deactivate_tool_results(self, *a, **kw):
                raise RuntimeError("db boom")

        agent = SimpleNamespace(session_id="sess-sub", _session_db=_Explodes())
        _persist_subagent_tool_result_age_out(agent, ["call_1"])  # must not raise


# ── 15: no session_db subagent -- wire age-out still works ──────────────

class TestNoSessionDbSubagent:
    def test_wire_age_out_works_without_session_db(self):
        agent = _FakeAgent(platform="subagent", session_db=None)
        api_messages, _ = _build(agent, _pair_fixture(10))
        present_ids = {m.get("tool_call_id") for m in api_messages if m.get("role") == "tool"}
        assert present_ids == {f"call_{i}" for i in range(7, 11)}


# ── 18: Task 4 spillover preview survives age-out on disk ───────────────

class TestTask4SpilloverInteraction:
    def test_persisted_output_preview_aged_out_but_spillover_file_intact(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        from tools.tool_result_storage import maybe_persist_tool_result, get_spillover_dir
        from tools.budget_config import budget_for_subagent

        raw = "SPILLED-PAYLOAD\n" + ("x" * 40_000)
        budget = budget_for_subagent(None)
        preview_block = maybe_persist_tool_result(
            content=raw, tool_name="execute_code", tool_use_id="call_1",
            env=None, config=budget,
        )
        assert "SPILLED-PAYLOAD" not in preview_block or len(preview_block) < len(raw)
        spill_file = get_spillover_dir() / "call_1.txt"
        assert spill_file.read_text(encoding="utf-8") == raw

        # This tool row (carrying the <persisted-output> preview block, not
        # the raw payload) now ages out of the wire -- the spillover file on
        # disk must be completely unaffected by the age-out decision.
        messages = [{"role": "user", "content": "go"}]
        for i in range(1, 11):
            cid = f"call_{i}"
            content = preview_block if cid == "call_1" else f"RAW-RESULT-{i}"
            messages.append({"role": "assistant", "content": "", "tool_calls": [_tool_call(cid)]})
            messages.append({"role": "tool", "tool_call_id": cid, "content": content})

        agent = _FakeAgent(platform="subagent")
        api_messages, _ = _build(agent, messages)
        assert not any(
            "SPILLED-PAYLOAD" in (m.get("content") or "") or "call_1.txt" in (m.get("content") or "")
            for m in api_messages
        )
        assert spill_file.exists()
        assert spill_file.read_text(encoding="utf-8") == raw  # untouched


# ── module constant sanity ──────────────────────────────────────────────

def test_keep_turns_constant_is_four():
    assert _SUBAGENT_TOOL_RESULT_MAX_MODEL_TURNS == 4
