"""Tests for agent/shared_evidence_cache.py — 20 contract cases."""
from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ── minimal SessionDB stub ────────────────────────────────────────────────────

class _FakeDB:
    """In-memory SessionDB stub that implements only the meta interface."""

    def __init__(self):
        self._store: Dict[str, str] = {}

    def get_meta(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def set_meta(self, key: str, value: str, *, cursor=None) -> None:
        self._store[key] = value

    def list_meta_prefix(self, prefix: str) -> List[tuple]:
        return [(k, v) for k, v in self._store.items() if k.startswith(prefix)]

    # Expose _conn-like interface for _delete_meta
    class _FakeConn:
        def __init__(self, store):
            self._store = store
        def execute(self, sql, params=()):
            if sql.startswith("DELETE"):
                key = params[0]
                self._store.pop(key, None)
        def commit(self): pass

    @property
    def _lock(self):
        import contextlib
        return contextlib.nullcontext()

    @property
    def _conn(self):
        return self._FakeConn(self._store)


def _make_agent(db=None, workspace="/tmp/ws"):
    agent = MagicMock()
    agent._session_db = db
    # _resolve_workspace_hint reads environment / attrs
    agent.terminal_cwd = workspace
    return agent


# ── import target ─────────────────────────────────────────────────────────────

from agent import shared_evidence_cache as sec


# ─────────────────────────────────────────────────────────────────────────────
# 1. Empty cache → no injection, behavior unchanged
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_cache_no_injection():
    db = _FakeDB()
    block = sec.maybe_inject_evidence(db, workspace="/ws", task="Fix bug in auth module")
    assert block == ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. Child A saves evidence → Child B can retrieve it
# ─────────────────────────────────────────────────────────────────────────────

def test_save_then_lookup_same_workspace(tmp_path):
    db = _FakeDB()
    ev_id = sec.save_evidence(
        db,
        workspace=str(tmp_path),
        task="audit auth module login path",
        summary="Found bug in login: token not validated. File: auth/login.py:42.",
        source_refs=[],
    )
    assert ev_id is not None
    assert ev_id.startswith("EV-")

    entries = sec.lookup_relevant_evidence(
        db, workspace=str(tmp_path), task="fix login validation in auth module"
    )
    assert len(entries) == 1
    assert entries[0]["evidence_id"] == ev_id


# ─────────────────────────────────────────────────────────────────────────────
# 3. Different workspace → no cross-contamination
# ─────────────────────────────────────────────────────────────────────────────

def test_different_workspace_no_sharing(tmp_path):
    db = _FakeDB()
    sec.save_evidence(
        db,
        workspace=str(tmp_path / "proj_a"),
        task="analyse auth module",
        summary="auth module has sql injection at login.py:22",
        source_refs=[],
    )
    entries = sec.lookup_relevant_evidence(
        db, workspace=str(tmp_path / "proj_b"), task="analyse auth module"
    )
    assert entries == []


# ─────────────────────────────────────────────────────────────────────────────
# 4. Relevant task → evidence injected into block
# ─────────────────────────────────────────────────────────────────────────────

def test_relevant_task_injected(tmp_path):
    db = _FakeDB()
    sec.save_evidence(
        db,
        workspace=str(tmp_path),
        task="analyse background_review fail-open path",
        summary="background_review.py line 194 has default=True; should be False.",
        source_refs=[],
    )
    block = sec.maybe_inject_evidence(
        db, workspace=str(tmp_path), task="fix fail-open in background_review.py"
    )
    assert "SHARED EVIDENCE" in block
    assert "background_review" in block


# ─────────────────────────────────────────────────────────────────────────────
# 5. Unrelated task → no injection
# ─────────────────────────────────────────────────────────────────────────────

def test_unrelated_task_not_injected(tmp_path):
    db = _FakeDB()
    sec.save_evidence(
        db,
        workspace=str(tmp_path),
        task="analyse background_review module",
        summary="background_review default is True, need fix.",
        source_refs=[],
    )
    block = sec.maybe_inject_evidence(
        db, workspace=str(tmp_path), task="translate README to Spanish"
    )
    assert block == ""


# ─────────────────────────────────────────────────────────────────────────────
# 6. Max 3 entries injected even if more exist
# ─────────────────────────────────────────────────────────────────────────────

def test_max_three_entries_injected(tmp_path):
    db = _FakeDB()
    for i in range(6):
        sec.save_evidence(
            db,
            workspace=str(tmp_path),
            task=f"analyse module_{i} background_review auth login token",
            summary=f"finding_{i}: module_{i} background_review auth token login issue",
            source_refs=[],
        )
    entries = sec.lookup_relevant_evidence(
        db, workspace=str(tmp_path),
        task="fix background_review auth login token module"
    )
    assert len(entries) <= sec._MAX_INJECT_ENTRIES


# ─────────────────────────────────────────────────────────────────────────────
# 7. Injection char budget respected
# ─────────────────────────────────────────────────────────────────────────────

def test_injection_char_budget(tmp_path):
    db = _FakeDB()
    for i in range(3):
        sec.save_evidence(
            db,
            workspace=str(tmp_path),
            task=f"task_{i} auth login module token background",
            summary="auth " * 500,  # large summary
            source_refs=[],
        )
    entries = sec.lookup_relevant_evidence(
        db, workspace=str(tmp_path), task="auth login token background module"
    )
    block = sec.build_evidence_block(entries)
    assert len(block) <= sec._MAX_INJECTION_CHARS + 50  # small tolerance for policy footer


# ─────────────────────────────────────────────────────────────────────────────
# 8. Duplicate fingerprint → no duplicate stored
# ─────────────────────────────────────────────────────────────────────────────

def test_dedup_same_evidence(tmp_path):
    db = _FakeDB()
    id1 = sec.save_evidence(
        db, workspace=str(tmp_path), task="fix auth bug", summary="Found auth issue at login.py",
        source_refs=[],
    )
    id2 = sec.save_evidence(
        db, workspace=str(tmp_path), task="fix auth bug", summary="Found auth issue at login.py",
        source_refs=[],
    )
    assert id1 == id2  # same fingerprint → same id returned, not a new entry
    entries = db.list_meta_prefix(sec._workspace_prefix(str(tmp_path)))
    assert len(entries) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 9. Unchanged source file → reusable (fresh)
# ─────────────────────────────────────────────────────────────────────────────

def test_unchanged_source_file_fresh(tmp_path):
    src = tmp_path / "agent" / "background_review.py"
    src.parent.mkdir(parents=True)
    src.write_text("# content")
    st = src.stat()
    ref = {"path": str(src), "mtime_ns": st.st_mtime_ns, "size": st.st_size}
    assert sec._ref_is_fresh(ref) is True


# ─────────────────────────────────────────────────────────────────────────────
# 10. Modified source file (mtime changed) → stale
# ─────────────────────────────────────────────────────────────────────────────

def test_modified_source_file_stale(tmp_path):
    src = tmp_path / "myfile.py"
    src.write_text("original")
    st = src.stat()
    ref = {"path": str(src), "mtime_ns": st.st_mtime_ns - 1, "size": st.st_size}
    assert sec._ref_is_fresh(ref) is False


# ─────────────────────────────────────────────────────────────────────────────
# 11. Deleted source file → stale
# ─────────────────────────────────────────────────────────────────────────────

def test_deleted_source_file_stale(tmp_path):
    ref = {"path": str(tmp_path / "gone.py"), "mtime_ns": 12345, "size": 100}
    assert sec._ref_is_fresh(ref) is False


# ─────────────────────────────────────────────────────────────────────────────
# 12. DB read exception → child execution unaffected (fail-open)
# ─────────────────────────────────────────────────────────────────────────────

def test_db_read_exception_fail_open(tmp_path):
    db = MagicMock()
    db.list_meta_prefix.side_effect = RuntimeError("disk error")
    result = sec.maybe_inject_evidence(db, workspace=str(tmp_path), task="any task here")
    assert result == ""  # fail-open: returns empty string, no exception


# ─────────────────────────────────────────────────────────────────────────────
# 13. DB write exception → child completion unaffected (fail-open)
# ─────────────────────────────────────────────────────────────────────────────

def test_db_write_exception_fail_open(tmp_path):
    db = MagicMock()
    db.list_meta_prefix.return_value = []
    db.set_meta.side_effect = RuntimeError("write error")
    # Should not raise
    ev_id = sec.save_evidence(
        db, workspace=str(tmp_path), task="some task", summary="some finding", source_refs=[]
    )
    assert ev_id is None  # graceful failure


# ─────────────────────────────────────────────────────────────────────────────
# 14. Failed/timeout child → evidence NOT saved
# ─────────────────────────────────────────────────────────────────────────────

def test_failed_child_not_saved(tmp_path):
    db = _FakeDB()
    for status in ("failed", "error", "timeout"):
        ev_id = sec.maybe_save_evidence_from_result(
            db,
            workspace=str(tmp_path),
            goal="analyse auth module",
            result={"status": status, "summary": "auth bug found"},
        )
        assert ev_id is None
    entries = db.list_meta_prefix(sec._workspace_prefix(str(tmp_path)))
    assert len(entries) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 15. Successful child → evidence ID saved and returned
# ─────────────────────────────────────────────────────────────────────────────

def test_successful_child_evidence_saved(tmp_path):
    db = _FakeDB()
    ev_id = sec.maybe_save_evidence_from_result(
        db,
        workspace=str(tmp_path),
        goal="analyse auth module login token",
        result={"status": "success", "summary": "Found auth bug at login.py:42. Fixed token validation."},
    )
    assert ev_id is not None
    assert ev_id.startswith("EV-")
    entries = db.list_meta_prefix(sec._workspace_prefix(str(tmp_path)))
    assert len(entries) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 16. Nested child (grandchild) same workspace → can look up parent evidence
# ─────────────────────────────────────────────────────────────────────────────

def test_nested_child_same_workspace(tmp_path):
    """Parent, Child A and Grandchild all share same DB file → grandchild sees evidence."""
    db = _FakeDB()
    # Parent saves (via Child A result)
    sec.save_evidence(
        db,
        workspace=str(tmp_path),
        task="audit auth login token validation",
        summary="auth token not validated at login.py:42",
        source_refs=[],
    )
    # Grandchild lookup
    entries = sec.lookup_relevant_evidence(
        db, workspace=str(tmp_path), task="fix auth login token at login module"
    )
    assert len(entries) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 17. Raw tool output not stored (summary is bounded)
# ─────────────────────────────────────────────────────────────────────────────

def test_raw_tool_output_not_fully_stored(tmp_path):
    db = _FakeDB()
    huge_summary = "auth issue " * 2000  # >> _MAX_SUMMARY_CHARS
    ev_id = sec.save_evidence(
        db, workspace=str(tmp_path), task="auth task", summary=huge_summary, source_refs=[]
    )
    assert ev_id is not None
    key = db.list_meta_prefix(sec._workspace_prefix(str(tmp_path)))[0][0]
    raw = db.get_meta(key)
    assert raw is not None
    entry = json.loads(raw)
    assert len(entry["summary"]) <= sec._MAX_SUMMARY_CHARS


# ─────────────────────────────────────────────────────────────────────────────
# 18. No new LLM calls added by evidence feature
# ─────────────────────────────────────────────────────────────────────────────

def test_no_llm_calls_in_evidence_module(tmp_path):
    """Verify the module contains no imports of LLM client / provider code."""
    import ast
    import inspect
    src = inspect.getsource(sec)
    tree = ast.parse(src)
    llm_imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if any(kw in (name or "") for kw in ("anthropic", "openai", "provider", "client", "model_router")):
                    llm_imports.append(name)
    assert llm_imports == [], f"LLM imports found: {llm_imports}"


# ─────────────────────────────────────────────────────────────────────────────
# 19. Handoff snapshot tests — regression guard (import only)
# ─────────────────────────────────────────────────────────────────────────────

def test_handoff_snapshot_import_regression():
    """shared_evidence_cache must not break handoff_snapshot importability."""
    import agent.handoff_snapshot  # noqa: F401  — just check it still imports clean


# ─────────────────────────────────────────────────────────────────────────────
# 20. Tool-result budget / age-out imports unaffected
# ─────────────────────────────────────────────────────────────────────────────

def test_tool_result_imports_regression():
    """Task 4/5 modules must still import cleanly alongside shared_evidence_cache."""
    import tools.budget_config  # noqa: F401
    import tools.tool_result_storage  # noqa: F401
    import agent.turn_context  # noqa: F401
    assert True
