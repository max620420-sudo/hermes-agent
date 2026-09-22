"""Explicit session handoff snapshots (Task 8-lite).

Saves a short, structured snapshot of the CURRENT session's state, but only
when the user explicitly signals they are moving to a new/different session
(e.g. "새 세션 시작할게" / "새 세션으로 갈게" / "다른 세션에서 이어갈게").
Ordinary turn completion or session exit ("작업 끝", "종료할게", "/exit")
never triggers a save — the trigger patterns below simply don't match those
phrases.

Explicitly OUT of scope here (see Task 8-lite spec):
  * automatic context rollover — nothing here starts a new session; it only
    stores a snapshot and, on a later explicit "resume" request, retrieves it.
  * a new LLM summarization call — the snapshot is built by parsing the
    EXISTING rolling-compaction summary already maintained by
    ``ContextCompressor`` (see ``agent/context_compressor.py``'s
    "## Goal" / "## Active State" / "## Completed Actions" / "## Key
    Decisions" / "## Blocked" / "## Relevant Files" template), not by asking
    an LLM to write a new one.
  * raw transcript / GBrain / SessionDB deletion — nothing here deletes
    anything; the snapshot is a small pointer-bearing extract, and full
    history stays recoverable via ``session_search`` as before.

Storage reuses SessionDB's existing generic key/value store
(``set_meta`` / ``get_meta`` / ``list_meta_prefix`` in ``hermes_state.py``)
instead of a new table, namespaced by workspace so a same-workspace resume
finds the most recent snapshot.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "detect_handoff_trigger",
    "detect_resume_request",
    "extract_snapshot_sections",
    "build_handoff_snapshot",
    "save_handoff_snapshot",
    "load_latest_handoff_snapshot",
    "format_snapshot_for_prompt",
    "estimate_snapshot_tokens",
    "maybe_handle_handoff_intent",
    "SNAPSHOT_FIELDS",
    "MAX_SNAPSHOT_TOKENS",
]

SNAPSHOT_FIELDS = (
    "CURRENT_TASK",
    "DONE",
    "DECISIONS",
    "OPEN_ISSUES",
    "NEXT_STEP",
    "EVIDENCE_REFS",
)

# ---------------------------------------------------------------------------
# Trigger phrase detection.
#
# Deliberately a narrow lexical/regex matcher, NOT a semantic classifier —
# routing every user message through an LLM intent call would add latency
# and cost to every single turn and conflicts with the "no new large LLM
# call" principle in spirit. A missed unusual phrasing is safe (the user
# just asks again plainly); the thing to avoid is a FALSE positive, so the
# patterns are kept specific to "new/other session" semantics and never
# overlap with ordinary end-of-work phrases ("작업 끝", "종료할게", "/exit"),
# which are not matched by design (no need for an explicit exclusion list).
# ---------------------------------------------------------------------------
_NEW_SESSION_PATTERNS: List[re.Pattern] = [
    re.compile(p) for p in [
        r"새\s*세션\s*(?:(?:으)?로\s*)?(시작|열|만들|갈게|가자|넘어갈게)",
        r"다른\s*세션(?:에서|으로)\s*(이어가|이어갈|계속|넘어가)",
        r"세션\s*(?:을\s*)?(?:새로|새롭게)\s*(시작|열)",
        r"\bstart(?:ing)?\s+(?:a\s+)?new\s+session\b",
        r"\b(?:let'?s\s+)?(?:go|move|switch)\s+to\s+a\s+new\s+session\b",
        r"\bcontinue\s+(?:this\s+)?in\s+a\s+new\s+session\b",
    ]
]

_RESUME_PATTERNS: List[re.Pattern] = [
    re.compile(p) for p in [
        r"이어서\s*(하자|할게|가자|진행)",
        r"이전\s*(세션|작업)\s*(?:에서\s*)?(이어|계속)",
        r"\bcontinue\s+(?:from\s+)?(?:where\s+we\s+left\s+off|the\s+last\s+session)\b",
        r"\bresume\s+(?:the\s+)?(?:previous|last)\s+session\b",
    ]
]


def detect_handoff_trigger(text: Any) -> bool:
    """Whether ``text`` expresses "I'm starting/moving to a new session"."""
    if not text or not isinstance(text, str):
        return False
    return any(p.search(text) for p in _NEW_SESSION_PATTERNS)


def detect_resume_request(text: Any) -> bool:
    """Whether ``text`` expresses "let's continue [from the last session]"."""
    if not text or not isinstance(text, str):
        return False
    return any(p.search(text) for p in _RESUME_PATTERNS)


# ---------------------------------------------------------------------------
# Snapshot construction: parse the EXISTING rolling summary, no LLM call.
# ---------------------------------------------------------------------------

# Maps our 6 snapshot fields to the rolling-summary heading(s) that already
# carry the equivalent information (see the summary template in
# agent/context_compressor.py, e.g. lines ~4986-5015 / ~5460-5500).
_SECTION_MAP: Dict[str, List[str]] = {
    "CURRENT_TASK": ["## Goal", "## Active Task", "## Active State"],
    "DONE": ["## Completed Actions"],
    "DECISIONS": ["## Key Decisions"],
    "OPEN_ISSUES": ["## Blocked"],
    "NEXT_STEP": ["## Active State", "## Active Task"],
    "EVIDENCE_REFS": ["## Relevant Files"],
}

_HEADING_LINE_RE = re.compile(r"(?m)^(##\s+.+)$")


def _split_sections(summary_text: str) -> Dict[str, str]:
    """Split a '## Heading\\n body ...' markdown summary into {heading: body}."""
    sections: Dict[str, str] = {}
    if not summary_text:
        return sections
    matches = list(_HEADING_LINE_RE.finditer(summary_text))
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(summary_text)
        sections[heading] = summary_text[start:end].strip()
    return sections


def extract_snapshot_sections(summary_text: str) -> Dict[str, str]:
    """Derive the 6 snapshot fields from an existing rolling summary.

    Pure text parsing of a summary the compressor already produced — no LLM
    call happens here. Sections absent from the summary degrade to "".
    """
    sections = _split_sections(summary_text or "")
    out: Dict[str, str] = {}
    for field, candidate_headings in _SECTION_MAP.items():
        value = ""
        for heading in candidate_headings:
            for key, body in sections.items():
                if key.startswith(heading) and body:
                    value = body
                    break
            if value:
                break
        out[field] = value
    return out


# ---------------------------------------------------------------------------
# Size bound: target <= 5k tokens total, split evenly across the 6 fields.
# Uses the same rough chars-per-token heuristic already used elsewhere in
# this codebase for cheap estimates (no tokenizer call).
# ---------------------------------------------------------------------------
_CHARS_PER_TOKEN = 4
MAX_SNAPSHOT_TOKENS = 5000
_MAX_FIELD_CHARS = (MAX_SNAPSHOT_TOKENS * _CHARS_PER_TOKEN) // len(SNAPSHOT_FIELDS)


def _truncate_field(value: str, max_chars: int = _MAX_FIELD_CHARS) -> str:
    if not value:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n…[truncated — recover full detail via session_search]"


def estimate_snapshot_tokens(snapshot: Dict[str, Any]) -> int:
    text = "\n".join(str(v) for k, v in snapshot.items() if not str(k).startswith("_"))
    return len(text) // _CHARS_PER_TOKEN


def _last_user_message_text(agent: Any) -> str:
    history = getattr(agent, "conversation_history", None) or []
    for msg in reversed(history):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                if parts:
                    return "\n".join(parts)
    return ""


def _default_evidence_ref(agent: Any) -> str:
    session_id = getattr(agent, "session_id", None) or "unknown-session"
    return f"session_search(session_id='{session_id}') for the full transcript"


def build_handoff_snapshot(agent: Any) -> Dict[str, Any]:
    """Build a short handoff snapshot, reusing existing structured state.

    Source of truth is the bound ``ContextCompressor``'s rolling summary
    (``_previous_summary``) when one exists — maintained across compactions
    with no fresh LLM call made here. When no summary exists yet (session
    never compacted), falls back to a bare deterministic snapshot built from
    the last user message only — still no LLM call.
    """
    compressor = getattr(agent, "context_compressor", None)
    summary_text = getattr(compressor, "_previous_summary", None) if compressor else None

    if summary_text:
        fields = extract_snapshot_sections(summary_text)
    else:
        fields = {k: "" for k in SNAPSHOT_FIELDS}
        last_user = _last_user_message_text(agent)
        if last_user:
            fields["CURRENT_TASK"] = last_user
            fields["NEXT_STEP"] = (
                "No prior compaction summary yet — see raw session transcript."
            )

    fields = {k: _truncate_field(v) for k, v in fields.items()}
    fields["EVIDENCE_REFS"] = fields.get("EVIDENCE_REFS") or _default_evidence_ref(agent)

    snapshot: Dict[str, Any] = {field: fields.get(field, "") for field in SNAPSHOT_FIELDS}
    snapshot["_session_id"] = getattr(agent, "session_id", None)
    snapshot["_saved_at"] = time.time()
    return snapshot


# ---------------------------------------------------------------------------
# Storage — reuses SessionDB.set_meta / get_meta / list_meta_prefix
# (hermes_state.py), the existing generic key/value store already used by
# other per-session feature rows (e.g. ``loop:<session_id>``). No new table.
# ---------------------------------------------------------------------------
_META_PREFIX = "handoff_snapshot::"


def _resolve_workspace(agent: Any) -> str:
    for attr in ("cwd", "working_directory", "workspace_id", "repo_root"):
        value = getattr(agent, attr, None)
        if value:
            return str(value)
    try:
        return os.getcwd()
    except OSError:
        return "unknown-workspace"


def _unique_suffix() -> str:
    """A cheap, local, non-deterministic tiebreaker — not an ordering key.

    Only needs to avoid same-millisecond collisions between snapshots saved
    by different sessions in the same workspace; it is never compared for
    "latest" purposes (the timestamp prefix alone still decides that).
    """
    return uuid.uuid4().hex[:8]


def _meta_key(workspace: str, ts: float) -> str:
    # Zero-padded epoch-millis prefix so lexicographic sort == chronological
    # (this part alone decided uniqueness pre-fix). A random local suffix is
    # appended so two different sessions saving in the same workspace within
    # the same millisecond get distinct keys instead of one overwriting the
    # other (#8-lite-final-edge-fix). Legacy rows saved before this suffix
    # existed (bare timestamp, no trailing "::<suffix>") keep matching the
    # same prefix and keep sorting correctly relative to new-style keys: for
    # equal timestamps a legacy (shorter) key just sorts immediately before
    # its same-millisecond, suffixed neighbor — an arbitrary but harmless
    # tiebreak for what was already a same-millisecond collision.
    return f"{_META_PREFIX}{workspace}::{int(ts * 1000):016d}::{_unique_suffix()}"


def save_handoff_snapshot(agent: Any, snapshot: Dict[str, Any]) -> Optional[str]:
    """Persist ``snapshot`` via the existing SessionDB meta store.

    Returns the storage key on success, None if there is no session_db to
    persist to. Never raises — a snapshot save must not break the turn.
    """
    session_db = getattr(agent, "_session_db", None)
    if session_db is None or not hasattr(session_db, "set_meta"):
        return None
    workspace = _resolve_workspace(agent)
    key = _meta_key(workspace, snapshot.get("_saved_at") or time.time())
    try:
        session_db.set_meta(key, json.dumps(snapshot, ensure_ascii=False))
        return key
    except Exception:
        logger.warning("Failed to persist handoff snapshot", exc_info=True)
        return None


def load_latest_handoff_snapshot(agent: Any) -> Optional[Dict[str, Any]]:
    """Look up the most recent handoff snapshot for this agent's workspace.

    Excludes any snapshot whose ``_session_id`` equals the CURRENT session
    (``agent.session_id``) — a session must never resume the snapshot it
    just saved itself. If the newest row for this workspace happens to be
    the current session's own snapshot, this keeps walking backward to the
    newest snapshot that belongs to a DIFFERENT session, rather than
    stopping at the first (self) match and returning None (#8-lite-fix).
    """
    session_db = getattr(agent, "_session_db", None)
    if session_db is None or not hasattr(session_db, "list_meta_prefix"):
        return None
    workspace = _resolve_workspace(agent)
    current_session_id = getattr(agent, "session_id", None)
    prefix = f"{_META_PREFIX}{workspace}::"
    try:
        rows = session_db.list_meta_prefix(prefix)
    except Exception:
        logger.warning("Failed to list handoff snapshots", exc_info=True)
        return None
    if not rows:
        return None
    # Keys preserve only milliseconds; the random collision suffix is not
    # chronological. Prefer the full saved timestamp for same-ms snapshots.
    candidates = []
    for key, value in rows:
        try:
            snapshot = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(snapshot, dict):
            continue
        try:
            saved_at = float(snapshot["_saved_at"])
            if not math.isfinite(saved_at):
                raise ValueError("non-finite snapshot timestamp")
        except (KeyError, TypeError, ValueError):
            # Older rows may lack the timestamp in the payload; retain their
            # previous ordering using the timestamp encoded in the key.
            try:
                saved_at = int(key[len(prefix):].split("::", 1)[0]) / 1000
            except (TypeError, ValueError):
                continue
        candidates.append((saved_at, key, snapshot))
    for _saved_at, _key, snapshot in sorted(candidates, reverse=True):
        if current_session_id and snapshot.get("_session_id") == current_session_id:
            continue  # skip our own snapshot; keep looking further back
        return snapshot
    return None


# ---------------------------------------------------------------------------
# Duplicate-injection guard: once a TARGET session has resumed a given
# SOURCE snapshot, the same source must not be re-injected on a later
# "이어서 하자" in that same target session (a newer source snapshot from a
# different session is still allowed through). Reuses the same SessionDB
# meta store as the snapshot itself — no new table — keyed
# ``handoff_consumed::<workspace>::<target_session_id>``, valued with an
# identifier for the consumed source snapshot.
# ---------------------------------------------------------------------------
_CONSUMED_META_PREFIX = "handoff_consumed::"


def _consumed_meta_key(workspace: str, target_session_id: str) -> str:
    return f"{_CONSUMED_META_PREFIX}{workspace}::{target_session_id}"


def _snapshot_source_id(snapshot: Dict[str, Any]) -> str:
    """Stable identity for a source snapshot (its origin session + save time)."""
    return f"{snapshot.get('_session_id')}::{snapshot.get('_saved_at')}"


def _get_consumed_source_id(
    session_db: Any, workspace: str, target_session_id: str
) -> Optional[str]:
    if session_db is None or not hasattr(session_db, "get_meta"):
        return None
    try:
        return session_db.get_meta(_consumed_meta_key(workspace, target_session_id))
    except Exception:
        logger.debug("Failed to read handoff-consumption marker", exc_info=True)
        return None


def _mark_source_consumed(
    session_db: Any, workspace: str, target_session_id: str, source_id: str
) -> None:
    if session_db is None or not hasattr(session_db, "set_meta"):
        return
    try:
        session_db.set_meta(_consumed_meta_key(workspace, target_session_id), source_id)
    except Exception:
        logger.warning("Failed to record handoff-snapshot consumption", exc_info=True)


def format_snapshot_for_prompt(snapshot: Dict[str, Any]) -> str:
    """Render the snapshot as a compact plain-text block for the model."""
    lines = ["[Resumed from a previous session's handoff snapshot]"]
    for field in SNAPSHOT_FIELDS:
        value = snapshot.get(field) or "(none)"
        lines.append(f"{field}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Turn-level entry point. Called once near the top of run_conversation()
# with the raw user_message text; best-effort — any failure here must never
# break the turn it's attached to. Returns the (possibly annotated)
# user_message unchanged unless a resume request found a snapshot.
# ---------------------------------------------------------------------------
def maybe_handle_handoff_intent(agent: Any, user_message: Any) -> Any:
    text = user_message if isinstance(user_message, str) else None
    if not text:
        return user_message

    try:
        if detect_handoff_trigger(text):
            snapshot = build_handoff_snapshot(agent)
            key = save_handoff_snapshot(agent, snapshot)
            if key:
                buffer_status = getattr(agent, "_buffer_status", None)
                if callable(buffer_status):
                    try:
                        buffer_status("📌 Saved a handoff snapshot for the next session.")
                    except Exception:
                        pass
            return user_message

        if detect_resume_request(text):
            snapshot = load_latest_handoff_snapshot(agent)
            if snapshot:
                session_db = getattr(agent, "_session_db", None)
                workspace = _resolve_workspace(agent)
                target_session_id = str(getattr(agent, "session_id", None) or "")
                source_id = _snapshot_source_id(snapshot)
                already_consumed = (
                    _get_consumed_source_id(session_db, workspace, target_session_id)
                    == source_id
                )
                if not already_consumed:
                    _mark_source_consumed(session_db, workspace, target_session_id, source_id)
                    note = format_snapshot_for_prompt(snapshot)
                    return f"{note}\n\n---\n\n{text}"
    except Exception:
        logger.warning("handoff snapshot intent handling failed", exc_info=True)

    return user_message
