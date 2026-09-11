"""Shared Evidence Cache — reuse previously collected workspace evidence across subagents.

When a subagent completes successfully, its final summary and the source file references it
touched are stored in the SessionDB generic meta store (same DB file as the parent, so all
children and grandchildren share it automatically without IPC or global singletons).

On the next delegation the parent queries the cache, filters stale entries (source file
changed/deleted), scores remaining entries against the new task with a cheap lexical
overlap, and injects the top-3 most relevant entries as a bounded evidence block into the
child's system prompt.

Design constraints honoured here
─────────────────────────────────
• No new LLM calls.
• No new DB tables — reuses SessionDB.set_meta / get_meta / list_meta_prefix.
• No raw tool output / raw transcript storage.
• Evidence write / read failures are FAIL-OPEN for the caller (agent execution continues).
• Max 50 entries per workspace; oldest evicted when the cap is hit.
• Source-ref staleness checked with os.stat (mtime_ns + size).
• No-source-ref summaries get a 45-minute TTL instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────

_MAX_ENTRIES_PER_WORKSPACE: int = 50
"""Evict the oldest entries when this cap is exceeded."""

_MAX_SUMMARY_CHARS: int = 4_000
"""Hard ceiling on the summary text stored per evidence entry."""

_MAX_INJECTION_CHARS: int = 6_000
"""Total char budget for the [SHARED EVIDENCE] block injected into a child prompt."""

_MAX_INJECT_ENTRIES: int = 3
"""Maximum number of evidence entries injected per child."""

_NO_REFS_TTL_SECONDS: int = 45 * 60
"""TTL for entries that have no source file refs (45 minutes)."""

_RELEVANCE_MIN_TOKENS: int = 2
"""Minimum shared token count to consider an entry relevant at all."""

_STOPWORDS: frozenset = frozenset(
    "a an the and or not is in on at to of for with by from this that it be as do "
    "was were are has have had its he she they we you i did can will would could should "
    "may might must shall".split()
)

# ── namespace helpers ─────────────────────────────────────────────────────────

_META_PREFIX = "shared_evidence::"


def _workspace_key(workspace: str, ev_id: str, created_ts: float) -> str:
    """Stable state_meta key for one evidence entry."""
    ts_int = int(created_ts * 1000)  # ms precision, sortable lexicographically when zero-padded
    return f"{_META_PREFIX}{workspace}::{ts_int:020d}::{ev_id}"


def _workspace_prefix(workspace: str) -> str:
    return f"{_META_PREFIX}{workspace}::"


# ── fingerprinting / dedup ────────────────────────────────────────────────────

def _fingerprint(workspace: str, task: str, summary: str) -> str:
    """Deterministic hex fingerprint for dedup.  Uses workspace + normalised task + first
    512 chars of normalised summary so that trivially different whitespace / punctuation
    doesn't create duplicate entries."""
    norm = re.sub(r"[^a-z0-9 ]", " ", (task + " " + summary[:512]).lower())
    norm = " ".join(norm.split())
    payload = f"{workspace}\x00{norm}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── lexical relevance scoring ─────────────────────────────────────────────────

def _tokenise(text: str) -> frozenset:
    """Lowercase alphanumeric tokens, length >= 3, not in stopwords."""
    tokens = re.findall(r"[a-z0-9_/.-]{3,}", text.lower())
    return frozenset(t for t in tokens if t not in _STOPWORDS)


def _relevance_score(query_tokens: frozenset, entry: Dict[str, Any]) -> int:
    """Count of shared tokens between the query and the entry's task + summary + source paths."""
    candidate_text = " ".join(filter(None, [
        entry.get("task", ""),
        entry.get("summary", ""),
        " ".join(r.get("path", "") for r in (entry.get("source_refs") or [])),
    ]))
    candidate_tokens = _tokenise(candidate_text)
    return len(query_tokens & candidate_tokens)


# ── staleness check ───────────────────────────────────────────────────────────

def _ref_is_fresh(ref: Dict[str, Any]) -> bool:
    """Return True iff the source file still exists with the same mtime_ns and size."""
    path = ref.get("path", "")
    if not path:
        return True  # no path recorded — treat as fresh (general summary)
    try:
        st = os.stat(path)
        return st.st_mtime_ns == ref.get("mtime_ns") and st.st_size == ref.get("size")
    except OSError:
        return False  # deleted or inaccessible → stale


def _entry_is_fresh(entry: Dict[str, Any]) -> bool:
    """Return True iff the entry is still usable (not stale, not TTL-expired)."""
    refs = entry.get("source_refs") or []
    if refs:
        return all(_ref_is_fresh(r) for r in refs)
    # No source refs → TTL check
    age = time.time() - float(entry.get("created_at", 0))
    return age < _NO_REFS_TTL_SECONDS


# ── public API ────────────────────────────────────────────────────────────────

def save_evidence(
    session_db: Any,
    *,
    workspace: str,
    task: str,
    summary: str,
    producer_session_id: str = "",
    producer_subagent_id: str = "",
    source_refs: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Persist one evidence entry to the SessionDB meta store.

    Returns the evidence_id string on success, None on any failure (fail-open).
    Deduplicates via fingerprint; silently skips if an identical entry already exists.
    Evicts the oldest entries when the per-workspace cap is hit.
    """
    if not workspace or not summary or not task:
        return None
    try:
        fp = _fingerprint(workspace, task, summary)
        prefix = _workspace_prefix(workspace)
        existing = session_db.list_meta_prefix(prefix)

        # Dedup check
        for _key, raw in existing:
            try:
                e = json.loads(raw)
                if e.get("fingerprint") == fp:
                    return e.get("evidence_id")
            except Exception:
                continue

        # Evict oldest if over cap
        if len(existing) >= _MAX_ENTRIES_PER_WORKSPACE:
            # Keys are timestamp-sortable; smallest = oldest
            oldest_key = min(existing, key=lambda kv: kv[0])[0]
            _delete_meta(session_db, oldest_key)

        ev_id = "EV-" + uuid.uuid4().hex[:8].upper()
        now = time.time()
        entry: Dict[str, Any] = {
            "evidence_id": ev_id,
            "workspace": workspace,
            "created_at": now,
            "fingerprint": fp,
            "producer_session_id": producer_session_id,
            "producer_subagent_id": producer_subagent_id,
            "task": task[:500],
            "summary": summary[:_MAX_SUMMARY_CHARS],
            "source_refs": source_refs or [],
        }
        key = _workspace_key(workspace, ev_id, now)
        session_db.set_meta(key, json.dumps(entry, ensure_ascii=False))
        logger.debug("SharedEvidence: saved %s (workspace=%s)", ev_id, workspace)
        return ev_id
    except Exception:
        logger.debug("SharedEvidence: save failed (fail-open)", exc_info=True)
        return None


def _delete_meta(session_db: Any, key: str) -> None:
    """Best-effort deletion via raw SQL (SessionDB has no public delete_meta)."""
    try:
        db_obj = getattr(session_db, "_db", session_db)  # unwrap AsyncSessionDB if needed
        with getattr(db_obj, "_lock", _noop_cm()):
            conn = getattr(db_obj, "_conn", None)
            if conn is not None:
                conn.execute("DELETE FROM state_meta WHERE key = ?", (key,))
                conn.commit()
    except Exception:
        logger.debug("SharedEvidence: delete_meta failed (non-fatal)", exc_info=True)


class _noop_cm:
    def __enter__(self): return self
    def __exit__(self, *a): pass


def lookup_relevant_evidence(
    session_db: Any,
    *,
    workspace: str,
    task: str,
    context: str = "",
) -> List[Dict[str, Any]]:
    """Return up to _MAX_INJECT_ENTRIES fresh, relevant evidence entries for a new task.

    Returns [] on any failure (fail-open).
    """
    if not workspace or not task:
        return []
    try:
        prefix = _workspace_prefix(workspace)
        all_pairs = session_db.list_meta_prefix(prefix)
        if not all_pairs:
            return []

        query_tokens = _tokenise(task + " " + context)
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for _key, raw in all_pairs:
            try:
                entry = json.loads(raw)
            except Exception:
                continue
            if not _entry_is_fresh(entry):
                continue
            score = _relevance_score(query_tokens, entry)
            if score >= _RELEVANCE_MIN_TOKENS:
                scored.append((score, entry))

        # Highest score first; ties broken by recency (created_at desc)
        scored.sort(key=lambda x: (x[0], x[1].get("created_at", 0)), reverse=True)
        return [e for _, e in scored[:_MAX_INJECT_ENTRIES]]
    except Exception:
        logger.debug("SharedEvidence: lookup failed (fail-open)", exc_info=True)
        return []


# ── prompt block builder ──────────────────────────────────────────────────────

_EVIDENCE_BLOCK_HEADER = "[SHARED EVIDENCE — REUSE BEFORE REREADING]\n\n"
_EVIDENCE_BLOCK_POLICY = (
    "\nPolicy:\n"
    "- Treat these as previously collected workspace evidence.\n"
    "- Reuse them when sufficient.\n"
    "- Do NOT reread the same unchanged source merely to reconfirm it.\n"
    "- Reread only when: (1) the source changed, (2) the evidence is incomplete for the\n"
    "  current task, (3) evidence conflicts with current state, or (4) the task is\n"
    "  safety/critical enough to require fresh verification.\n"
)


def build_evidence_block(entries: List[Dict[str, Any]]) -> str:
    """Render the evidence entries as a bounded prompt block.

    Returns "" if entries is empty.  The result is hard-capped at _MAX_INJECTION_CHARS.
    """
    if not entries:
        return ""

    parts: List[str] = [_EVIDENCE_BLOCK_HEADER]
    chars_used = len(_EVIDENCE_BLOCK_HEADER) + len(_EVIDENCE_BLOCK_POLICY)
    budget = _MAX_INJECTION_CHARS - chars_used

    for entry in entries:
        ev_id = entry.get("evidence_id", "EV-?")
        task_line = entry.get("task", "")[:200]
        summary = entry.get("summary", "")
        refs = entry.get("source_refs") or []

        ref_lines = "\n".join(
            f"- {r.get('path', '?')}" + (f":{r.get('line_or_range', '')}" if r.get("line_or_range") else "")
            for r in refs[:10]
        )
        block = f"{ev_id}\nTask: {task_line}\n"
        if ref_lines:
            block += f"Sources:\n{ref_lines}\n"
        block += f"Finding:\n{summary}\n\n"

        if len(block) > budget:
            # Truncate summary to fit
            overhead = len(block) - len(summary)
            allowed = max(0, budget - overhead - 3)
            if allowed < 50:
                break  # not enough room for even a snippet
            block = block.replace(summary, summary[:allowed] + "...")
            block = block[: budget + 3]

        parts.append(block)
        budget -= len(block)
        if budget <= 0:
            break

    if len(parts) == 1:  # only header, no entries fit
        return ""

    parts.append(_EVIDENCE_BLOCK_POLICY)
    return "".join(parts)


# ── source ref helpers ────────────────────────────────────────────────────────

def stat_source_ref(path: str, line_or_range: str = "") -> Optional[Dict[str, Any]]:
    """Build a source_ref dict for ``path`` with current stat, or None on error."""
    try:
        st = os.stat(path)
        return {
            "path": os.path.abspath(path),
            "line_or_range": line_or_range,
            "mtime_ns": st.st_mtime_ns,
            "size": st.st_size,
        }
    except OSError:
        return None


def parse_evidence_refs_from_summary(summary: str) -> List[Dict[str, Any]]:
    """Best-effort parse of an ``EVIDENCE REFS:`` block from a subagent summary.

    Looks for lines like:
        EVIDENCE REFS:
        - path/to/file.py:120-160
        - path/to/other.py:42

    Returns a list of source_ref dicts (with stat if file exists, else path only).
    """
    refs: List[Dict[str, Any]] = []
    in_block = False
    for line in summary.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("EVIDENCE REFS"):
            in_block = True
            continue
        if in_block:
            if not stripped or (not stripped.startswith("-") and not stripped[0].isalpha() and not stripped[0] == "/"):
                break  # end of block
            # Remove leading "- "
            entry_text = stripped.lstrip("- ").strip()
            # Split off line range
            if ":" in entry_text:
                path_part, _, line_part = entry_text.rpartition(":")
                # Heuristic: if line_part looks like a line spec keep it
                if re.match(r"^\d+(-\d+)?$", line_part.strip()):
                    ref = stat_source_ref(path_part.strip(), line_part.strip())
                    if ref:
                        refs.append(ref)
                    else:
                        refs.append({"path": path_part.strip(), "line_or_range": line_part.strip(), "mtime_ns": None, "size": None})
                    continue
            ref = stat_source_ref(entry_text)
            if ref:
                refs.append(ref)
            else:
                refs.append({"path": entry_text, "line_or_range": "", "mtime_ns": None, "size": None})
    return refs


# ── delegate_tool integration helpers ────────────────────────────────────────

def maybe_save_evidence_from_result(
    session_db: Any,
    *,
    workspace: str,
    goal: str,
    result: Dict[str, Any],
    producer_session_id: str = "",
    producer_subagent_id: str = "",
) -> Optional[str]:
    """Called after a successful child completion to persist evidence.

    Safe to call with any result dict; fails silently (fail-open).
    Only saves when status is not in SUBAGENT_FAILURE_STATUSES and summary is present.
    Returns evidence_id or None.
    """
    try:
        from tools.delegate_tool_progress import SUBAGENT_FAILURE_STATUSES
        status = str(result.get("status") or "").strip().lower()
        if status in SUBAGENT_FAILURE_STATUSES:
            return None
        summary = ""
        for key in ("summary", "output", "result"):
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                summary = v.strip()
                break
        if not summary:
            return None
        source_refs = parse_evidence_refs_from_summary(summary)
        ev_id = save_evidence(
            session_db,
            workspace=workspace,
            task=goal,
            summary=summary,
            producer_session_id=producer_session_id,
            producer_subagent_id=producer_subagent_id,
            source_refs=source_refs,
        )
        return ev_id
    except Exception:
        logger.debug("SharedEvidence: maybe_save failed (fail-open)", exc_info=True)
        return None


def maybe_inject_evidence(
    session_db: Any,
    *,
    workspace: str,
    task: str,
    context: str = "",
) -> str:
    """Return a bounded evidence block to prepend to a child prompt, or "" if none relevant.

    Safe to call even when session_db is None; fails silently (fail-open).
    """
    if session_db is None or not workspace:
        return ""
    try:
        entries = lookup_relevant_evidence(session_db, workspace=workspace, task=task, context=context)
        return build_evidence_block(entries)
    except Exception:
        logger.debug("SharedEvidence: inject failed (fail-open)", exc_info=True)
        return ""


def evidence_id_footer(ev_id: Optional[str]) -> str:
    """Tiny footer appended to a child result so the parent knows the cache entry id."""
    if not ev_id:
        return ""
    return f"\n\n[Shared Evidence]\nSaved: {ev_id}\nReuse this evidence before rereading unchanged sources.\n"
