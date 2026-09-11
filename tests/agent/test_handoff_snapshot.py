"""Unit tests for the explicit session-handoff snapshot (Task 8-lite).

Scope: pure trigger-detection / section-parsing / storage helpers in
``agent.handoff_snapshot``. No network, no LLM calls — the module makes
none, which these tests also implicitly guard (any accidental added call
would need a fixture/mock this file doesn't provide).
"""

from __future__ import annotations

import json

import pytest

from agent import handoff_snapshot as hs


# ── Trigger detection ───────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "새 세션 시작할게",
    "새 세션으로 갈게",
    "다른 세션에서 이어갈게",
    "세션을 새로 시작해줘",
    "let's start a new session",
    "starting a new session now",
])
def test_detect_handoff_trigger_matches(text):
    assert hs.detect_handoff_trigger(text) is True


@pytest.mark.parametrize("text", [
    "작업 끝",
    "종료할게",
    "/exit",
    "고마워, 여기까지 할게",
    "이 파일 좀 고쳐줘",
    "",
    None,
])
def test_detect_handoff_trigger_does_not_match_ordinary_exit(text):
    assert hs.detect_handoff_trigger(text) is False


@pytest.mark.parametrize("text", [
    "이어서 하자",
    "이전 세션에서 이어가자",
    "let's resume the previous session",
    "continue where we left off",
])
def test_detect_resume_request_matches(text):
    assert hs.detect_resume_request(text) is True


def test_detect_resume_request_does_not_match_new_session_trigger():
    # The two intents must stay mutually exclusive for the same phrase set.
    assert hs.detect_resume_request("새 세션 시작할게") is False
    assert hs.detect_handoff_trigger("이어서 하자") is False


# ── Section extraction from the existing rolling summary ───────────────

_SAMPLE_SUMMARY = """\
## Goal
Ship the handoff snapshot feature.

## Constraints & Preferences
Keep it lite; reuse existing state.

## Completed Actions
1. Investigated existing rolling summary schema.
2. Wrote agent/handoff_snapshot.py.

## Active State
Wiring the trigger into run_conversation().

## Blocked
Nothing currently blocking.

## Key Decisions
Reuse SessionDB.set_meta instead of a new table.

## Relevant Files
agent/handoff_snapshot.py, agent/conversation_loop.py
"""


def test_extract_snapshot_sections_maps_existing_headings():
    fields = hs.extract_snapshot_sections(_SAMPLE_SUMMARY)
    assert "Ship the handoff snapshot feature." in fields["CURRENT_TASK"]
    assert "Wrote agent/handoff_snapshot.py" in fields["DONE"]
    assert "SessionDB.set_meta" in fields["DECISIONS"]
    assert "Nothing currently blocking." in fields["OPEN_ISSUES"]
    assert "Wiring the trigger" in fields["NEXT_STEP"]
    assert "conversation_loop.py" in fields["EVIDENCE_REFS"]


def test_extract_snapshot_sections_empty_summary_degrades_gracefully():
    fields = hs.extract_snapshot_sections("")
    assert all(fields[k] == "" for k in hs.SNAPSHOT_FIELDS)


# ── Size bound (~5k tokens) ─────────────────────────────────────────────

def test_build_handoff_snapshot_stays_within_token_budget():
    huge_summary = "## Goal\n" + ("x" * 50000)

    class _FakeCompressor:
        _previous_summary = huge_summary

    class _FakeAgent:
        context_compressor = _FakeCompressor()
        session_id = "sess-1"
        conversation_history = []

    snapshot = hs.build_handoff_snapshot(_FakeAgent())
    assert hs.estimate_snapshot_tokens(snapshot) <= hs.MAX_SNAPSHOT_TOKENS


def test_build_handoff_snapshot_falls_back_without_prior_summary():
    class _FakeAgent:
        context_compressor = None
        session_id = "sess-2"
        conversation_history = [
            {"role": "user", "content": "please fix the bug in foo.py"},
        ]

    snapshot = hs.build_handoff_snapshot(_FakeAgent())
    assert "fix the bug in foo.py" in snapshot["CURRENT_TASK"]
    assert snapshot["_session_id"] == "sess-2"


# ── Storage: reuse of SessionDB.set_meta / list_meta_prefix ────────────

class _FakeSessionDB:
    def __init__(self):
        self._store: dict[str, str] = {}

    def set_meta(self, key, value):
        self._store[key] = value

    def get_meta(self, key):
        return self._store.get(key)

    def list_meta_prefix(self, prefix):
        return [(k, v) for k, v in self._store.items() if k.startswith(prefix)]


def test_save_and_load_latest_handoff_snapshot_roundtrip():
    db = _FakeSessionDB()

    class _FakeAgent:
        _session_db = db
        cwd = "/repo/hermes-agent"
        session_id = "sess-3"
        context_compressor = None
        conversation_history = [{"role": "user", "content": "task A"}]

    agent = _FakeAgent()
    snap1 = hs.build_handoff_snapshot(agent)
    hs.save_handoff_snapshot(agent, snap1)

    agent.conversation_history = [{"role": "user", "content": "task B (newer)"}]
    snap2 = hs.build_handoff_snapshot(agent)
    hs.save_handoff_snapshot(agent, snap2)

    # Looked up from a DIFFERENT session — sess-3's own snapshots are never
    # resume candidates for sess-3 itself (see the self-exclusion tests
    # below), so use a distinct querying session here.
    class _QueryingAgent:
        _session_db = db
        cwd = "/repo/hermes-agent"
        session_id = "sess-3-querier"

    latest = hs.load_latest_handoff_snapshot(_QueryingAgent())
    assert latest is not None
    assert "task B (newer)" in latest["CURRENT_TASK"]


def test_load_latest_handoff_snapshot_returns_none_without_session_db():
    class _FakeAgent:
        _session_db = None
        cwd = "/repo/hermes-agent"

    assert hs.load_latest_handoff_snapshot(_FakeAgent()) is None


# ── maybe_handle_handoff_intent: end-to-end best-effort wiring ─────────

def test_maybe_handle_handoff_intent_saves_on_trigger_and_never_raises():
    db = _FakeSessionDB()

    class _FakeAgent:
        _session_db = db
        cwd = "/repo/x"
        session_id = "sess-4"
        context_compressor = None
        conversation_history = []

        def _buffer_status(self, *_a, **_kw):
            pass

    agent = _FakeAgent()
    result = hs.maybe_handle_handoff_intent(agent, "새 세션 시작할게")
    assert result == "새 세션 시작할게"  # message itself is unchanged
    rows = db.list_meta_prefix("handoff_snapshot::")
    assert len(rows) == 1


def test_maybe_handle_handoff_intent_prepends_snapshot_on_resume():
    # Resume must come from a DIFFERENT session than the one that saved the
    # snapshot — same-session self-resume is covered separately below by
    # the Task 8-lite-fix regression tests.
    db = _FakeSessionDB()

    class _SourceAgent:
        _session_db = db
        cwd = "/repo/x"
        session_id = "sess-5"
        context_compressor = None
        conversation_history = [{"role": "user", "content": "earlier task"}]

    class _TargetAgent:
        _session_db = db
        cwd = "/repo/x"
        session_id = "sess-6"
        context_compressor = None
        conversation_history = []

    hs.maybe_handle_handoff_intent(_SourceAgent(), "새 세션 시작할게")

    result = hs.maybe_handle_handoff_intent(_TargetAgent(), "이어서 하자")
    assert "handoff snapshot" in result
    assert result.endswith("이어서 하자")


def test_maybe_handle_handoff_intent_ignores_ordinary_messages():
    class _FakeAgent:
        _session_db = None

    agent = _FakeAgent()
    assert hs.maybe_handle_handoff_intent(agent, "그냥 코드 리뷰해줘") == "그냥 코드 리뷰해줘"
    assert hs.maybe_handle_handoff_intent(agent, "작업 끝, 고마워") == "작업 끝, 고마워"


# ── Task 8-lite-fix regressions: self-session and duplicate-target guards ──
# See "[Task 8-lite Fix] handoff snapshot 자기 세션/중복 재주입 방지".

class _Agent:
    """Minimal stand-in agent for the fix-verification scenarios below."""

    def __init__(self, session_db, session_id, cwd="/repo/x", conversation_history=None):
        self._session_db = session_db
        self.session_id = session_id
        self.cwd = cwd
        self.context_compressor = None
        self.conversation_history = conversation_history or []

    def _buffer_status(self, *_a, **_kw):
        pass


def test_fix_scenario1_same_session_never_resumes_its_own_snapshot():
    """sess-A saves, sess-A resumes -> 0 injections."""
    db = _FakeSessionDB()
    sess_a = _Agent(db, "sess-A", conversation_history=[{"role": "user", "content": "task A"}])

    hs.maybe_handle_handoff_intent(sess_a, "새 세션 시작할게")
    result = hs.maybe_handle_handoff_intent(sess_a, "이어서 하자")

    assert result == "이어서 하자"  # unchanged: no snapshot text prepended


def test_fix_scenario2_target_session_resumes_once_then_dedups():
    """sess-A saves; sess-B resumes exactly once; a second sess-B resume of
    the SAME source snapshot injects nothing more."""
    db = _FakeSessionDB()
    sess_a = _Agent(db, "sess-A", conversation_history=[{"role": "user", "content": "task A"}])
    sess_b = _Agent(db, "sess-B")

    hs.maybe_handle_handoff_intent(sess_a, "새 세션 시작할게")

    first = hs.maybe_handle_handoff_intent(sess_b, "이어서 하자")
    assert "handoff snapshot" in first
    assert first.count("[Resumed from a previous session's handoff snapshot]") == 1

    second = hs.maybe_handle_handoff_intent(sess_b, "이어서 하자")
    assert second == "이어서 하자"  # unchanged: duplicate source snapshot suppressed


def test_fix_scenario3_a_different_new_session_can_still_resume():
    """sess-A saves; sess-C (never resumed before) can resume sess-A's
    snapshot independently of sess-B's consumption state."""
    db = _FakeSessionDB()
    sess_a = _Agent(db, "sess-A", conversation_history=[{"role": "user", "content": "task A"}])
    sess_b = _Agent(db, "sess-B")
    sess_c = _Agent(db, "sess-C")

    hs.maybe_handle_handoff_intent(sess_a, "새 세션 시작할게")
    hs.maybe_handle_handoff_intent(sess_b, "이어서 하자")  # sess-B consumes it

    result = hs.maybe_handle_handoff_intent(sess_c, "이어서 하자")
    assert "handoff snapshot" in result
    assert "task A" in result


def test_fix_scenario4_current_session_excluded_falls_back_to_older_snapshot():
    """workspace has an OLD snapshot from sess-X and a NEWER one from
    sess-A; when sess-A itself asks to resume, sess-A's own (newer)
    snapshot is skipped and sess-X's older snapshot is selected instead."""
    db = _FakeSessionDB()
    sess_x = _Agent(db, "sess-X", conversation_history=[{"role": "user", "content": "old task X"}])
    sess_a = _Agent(db, "sess-A", conversation_history=[{"role": "user", "content": "newer task A"}])

    # Build + save directly with explicit, distinct timestamps: two "새
    # 세션 시작할게" calls back-to-back can land in the same epoch-millis
    # bucket (meta-key collision, a separate/unrelated pre-existing
    # timestamp-granularity issue) which would make this test flaky rather
    # than exercising the self-exclusion fallback it targets.
    snap_x = hs.build_handoff_snapshot(sess_x)
    snap_x["_saved_at"] = 1000.0
    hs.save_handoff_snapshot(sess_x, snap_x)

    snap_a = hs.build_handoff_snapshot(sess_a)
    snap_a["_saved_at"] = 2000.0
    hs.save_handoff_snapshot(sess_a, snap_a)

    snapshot = hs.load_latest_handoff_snapshot(sess_a)
    assert snapshot is not None
    assert "old task X" in snapshot["CURRENT_TASK"]
    assert snapshot["_session_id"] == "sess-X"


def test_fix_scenario5_ordinary_messages_still_unaffected():
    db = _FakeSessionDB()
    sess_a = _Agent(db, "sess-A")
    assert hs.maybe_handle_handoff_intent(sess_a, "버그 하나만 고쳐줘") == "버그 하나만 고쳐줘"
    assert hs.maybe_handle_handoff_intent(sess_a, "작업 끝") == "작업 끝"
    assert db.list_meta_prefix("handoff_snapshot::") == []
    assert db.list_meta_prefix("handoff_consumed::") == []


def test_fix_scenario6_persisted_transcript_has_no_synthetic_prefix():
    """The conversation_loop.py hook keeps persist_user_message as the
    clean original text; this test locks the pure building block it relies
    on (maybe_handle_handoff_intent's return value) so the annotated,
    snapshot-carrying text is never mistaken for the text to persist."""
    db = _FakeSessionDB()
    sess_a = _Agent(db, "sess-A", conversation_history=[{"role": "user", "content": "task A"}])
    sess_b = _Agent(db, "sess-B")

    hs.maybe_handle_handoff_intent(sess_a, "새 세션 시작할게")
    annotated = hs.maybe_handle_handoff_intent(sess_b, "이어서 하자")

    original_text = "이어서 하자"
    assert annotated != original_text
    assert "[Resumed from a previous session's handoff snapshot]" in annotated
    # The clean original text a caller would persist is recoverable exactly
    # (conversation_loop.py stores THIS, not `annotated`, as
    # persist_user_message) and contains no snapshot fields.
    assert "CURRENT_TASK" not in original_text
    assert "EVIDENCE_REFS" not in original_text


# ── Task 8-lite-final-edge-fix: same-millisecond meta-key collision ─────
# See "[Task 8-lite Final Edge Fix]".

def test_edge_fix_same_millisecond_saves_from_different_sessions_do_not_overwrite():
    """Two different sessions saving in the same workspace at the exact
    same epoch-millis must both persist as distinct rows, not collide on
    the same meta key."""
    db = _FakeSessionDB()
    sess_x = _Agent(db, "sess-X", conversation_history=[{"role": "user", "content": "task X"}])
    sess_y = _Agent(db, "sess-Y", conversation_history=[{"role": "user", "content": "task Y"}])

    snap_x = hs.build_handoff_snapshot(sess_x)
    snap_x["_saved_at"] = 5000.0
    key_x = hs.save_handoff_snapshot(sess_x, snap_x)

    snap_y = hs.build_handoff_snapshot(sess_y)
    snap_y["_saved_at"] = 5000.0  # identical millisecond on purpose
    key_y = hs.save_handoff_snapshot(sess_y, snap_y)

    assert key_x is not None and key_y is not None
    assert key_x != key_y  # distinct keys despite the identical timestamp

    rows = db.list_meta_prefix("handoff_snapshot::")
    assert len(rows) == 2  # neither row was overwritten

    stored_session_ids = {json.loads(v)["_session_id"] for _k, v in rows}
    assert stored_session_ids == {"sess-X", "sess-Y"}

    # A third session can still resolve the latest same-millisecond row
    # deterministically (either of the two is an acceptable tiebreak; the
    # point is it's a real snapshot, not data loss).
    sess_z = _Agent(db, "sess-Z")
    resolved = hs.load_latest_handoff_snapshot(sess_z)
    assert resolved is not None
    assert resolved["_session_id"] in {"sess-X", "sess-Y"}


def test_edge_fix_timestamp_ordering_still_wins_over_suffix():
    """A later timestamp must still sort as "latest" regardless of the
    random per-key suffix."""
    db = _FakeSessionDB()
    sess_old = _Agent(db, "sess-OLD", conversation_history=[{"role": "user", "content": "older task"}])
    sess_new = _Agent(db, "sess-NEW", conversation_history=[{"role": "user", "content": "newer task"}])

    snap_old = hs.build_handoff_snapshot(sess_old)
    snap_old["_saved_at"] = 1000.0
    hs.save_handoff_snapshot(sess_old, snap_old)

    snap_new = hs.build_handoff_snapshot(sess_new)
    snap_new["_saved_at"] = 9000.0
    hs.save_handoff_snapshot(sess_new, snap_new)

    querier = _Agent(db, "sess-QUERY")
    latest = hs.load_latest_handoff_snapshot(querier)
    assert latest is not None
    assert latest["_session_id"] == "sess-NEW"


def test_edge_fix_legacy_key_without_suffix_still_resolves():
    """A row saved under the pre-fix key format (bare 16-digit millis, no
    "::<suffix>") must still be discoverable and selectable as latest."""
    db = _FakeSessionDB()
    workspace = "/repo/legacy"

    legacy_agent = _Agent(db, "sess-LEGACY", cwd=workspace)
    legacy_snapshot = hs.build_handoff_snapshot(legacy_agent)
    legacy_snapshot["_saved_at"] = 4000.0
    legacy_key = f"handoff_snapshot::{workspace}::{int(4000.0 * 1000):016d}"  # no suffix
    db.set_meta(legacy_key, json.dumps(legacy_snapshot, ensure_ascii=False))

    querier = _Agent(db, "sess-QUERY2", cwd=workspace)
    resolved = hs.load_latest_handoff_snapshot(querier)
    assert resolved is not None
    assert resolved["_session_id"] == "sess-LEGACY"
