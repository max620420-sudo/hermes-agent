"""Subagent tool-result age-out: model-turn ageing, pair safety, persistence.

Behavior contracts for the age-out that ``run_conversation`` applies while
building ``api_messages`` for a delegated child session:

- age is counted in MODEL turns (assistant messages), because a subagent run
  is one user message followed by many assistant<->tool loop iterations;
- only the last 4 model turns' tool results stay in the outgoing request;
- excluding a tool result never leaves its assistant ``tool_call`` dangling;
- the flip is persisted (``messages.active = 0``) so a reload cannot
  resurrect an aged-out result, while the raw row stays readable;
- non-subagent (main/parent) sessions are untouched.

The age-out and pair-safety statements live inline inside ``run_conversation``
(a function that needs a live provider to call), so the two blocks are
extracted FROM THE SHIPPED SOURCE by their anchors and executed here against a
fixture. That keeps these tests bound to the real code rather than to a
re-typed copy of it.
"""

import textwrap
from pathlib import Path

import pytest

from agent.agent_runtime_helpers import _classify_tool_call_orphans
from hermes_state import SessionDB

LOOP_SRC = Path(__file__).resolve().parents[2] / "agent" / "conversation_loop.py"

AGE_OUT_START = "_subagent_aged_out_idxs = set()"
AGE_OUT_END = "api_messages = []"
PAIR_START = "# Pair safety for the subagent tool-result age-out above:"
PAIR_END = "if not _has_text:"


def _slice_source(start_anchor: str, end_anchor: str, *, include_end: int = 0) -> str:
    """Return the dedented source between two anchors in conversation_loop.py."""
    lines = LOOP_SRC.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if start_anchor in ln)
    end = next(i for i, ln in enumerate(lines[start:], start) if end_anchor in ln)
    return textwrap.dedent("\n".join(lines[start:end + include_end]))


class _FakeLogger:
    def debug(self, *a, **k):
        pass


def _build_api_messages(messages, platform, session_db=None, session_id="sess"):
    """Run the SHIPPED age-out + pair-safety statements over *messages*."""
    age_out_src = _slice_source(AGE_OUT_START, AGE_OUT_END)
    # The pair block ends with `if not _has_text:` + its `continue`.
    pair_src = _slice_source(PAIR_START, PAIR_END, include_end=2)

    agent = type(
        "FakeAgent",
        (),
        {"platform": platform, "session_id": session_id, "_session_db": session_db},
    )()

    loop_src = (
        "api_messages = []\n"
        "for idx, msg in enumerate(messages):\n"
        "    if idx in _subagent_aged_out_idxs:\n"
        "        continue\n"
        "    api_msg = {k: v for k, v in msg.items()}\n"
        "    if isinstance(api_msg.get('tool_calls'), list):\n"
        "        api_msg['tool_calls'] = [dict(tc) for tc in api_msg['tool_calls']]\n"
        + textwrap.indent(pair_src, "    ")
        + "\n    api_messages.append(api_msg)\n"
    )

    ns = {"agent": agent, "messages": messages, "request_logger": _FakeLogger()}
    exec(compile(age_out_src, "<age_out_block>", "exec"), ns)
    exec(compile(loop_src, "<api_messages_loop>", "exec"), ns)
    return ns["api_messages"], ns["_subagent_aged_out_idxs"]


def subagent_fixture(turns: int = 10):
    """user x1 -> (assistant tool_call -> tool result) x turns."""
    messages = [{"role": "user", "content": "delegate: audit the module"}]
    for t in range(turns):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{t}",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{t}",
                "tool_name": "read_file",
                "content": f"RESULT-{t}" + ("x" * 500),
            }
        )
    return messages


def test_only_last_four_model_turns_of_tool_results_stay_active():
    """Age is counted in assistant/model turns, not user turns."""
    messages = subagent_fixture(10)
    api_messages, aged_idxs = _build_api_messages(messages, platform="subagent")

    sent_results = [m["tool_call_id"] for m in api_messages if m["role"] == "tool"]
    assert sent_results == ["call_6", "call_7", "call_8", "call_9"]

    # ...and the older six were excluded, not deleted.
    assert len(aged_idxs) == 6
    aged_ids = {messages[i]["tool_call_id"] for i in aged_idxs}
    assert aged_ids == {f"call_{t}" for t in range(6)}
    for i in aged_idxs:
        assert messages[i]["active"] is False
        assert messages[i]["content"].startswith("RESULT-"), "raw content preserved"


def test_excluded_results_leave_no_dangling_tool_calls():
    """The codebase's own orphan classifier must find nothing to repair."""
    messages = subagent_fixture(10)
    api_messages, _ = _build_api_messages(messages, platform="subagent")

    _, _, orphaned_results, missing_tool_calls = _classify_tool_call_orphans(
        api_messages
    )
    assert missing_tool_calls == [], "assistant tool_call with no result reached the wire"
    assert orphaned_results == [], "tool result with no assistant call reached the wire"

    # The durable transcript still carries every original call.
    assert sum(len(m.get("tool_calls") or []) for m in messages) == 10


def test_assistant_text_survives_when_only_its_tool_call_ages_out():
    """A scaffolding-only assistant turn is dropped; one with text is kept."""
    messages = subagent_fixture(10)
    messages[1]["content"] = "Reading the first file now."

    api_messages, _ = _build_api_messages(messages, platform="subagent")

    kept_text = [
        m for m in api_messages
        if m["role"] == "assistant" and m.get("content") == "Reading the first file now."
    ]
    assert len(kept_text) == 1
    assert not kept_text[0].get("tool_calls"), "aged call must be stripped from the copy"
    assert messages[1]["tool_calls"], "durable message keeps its tool_calls"


def test_main_and_parent_sessions_are_untouched():
    """Non-subagent platforms age nothing out."""
    messages = subagent_fixture(10)
    api_messages, aged_idxs = _build_api_messages(messages, platform="cli")

    assert aged_idxs == set()
    assert len(api_messages) == len(messages)
    assert [m["tool_call_id"] for m in api_messages if m["role"] == "tool"] == [
        f"call_{t}" for t in range(10)
    ]
    assert all("active" not in m for m in messages)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return SessionDB(db_path=tmp_path / "state.db")


def _persist_fixture(db, messages):
    session_id = db.create_session("subagent-age-out", "subagent")
    for msg in messages:
        db.append_message(
            session_id,
            msg["role"],
            content=msg.get("content"),
            tool_calls=msg.get("tool_calls"),
            tool_call_id=msg.get("tool_call_id"),
            tool_name=msg.get("tool_name"),
        )
    return session_id


def test_age_out_persists_to_db_and_survives_reload(db):
    """A reload must not resurrect an aged-out tool result as active context."""
    messages = subagent_fixture(10)
    session_id = _persist_fixture(db, messages)

    api_messages, aged_idxs = _build_api_messages(
        messages, platform="subagent", session_db=db, session_id=session_id
    )
    assert aged_idxs, "fixture must age something out"

    reloaded = db.get_messages(session_id)
    reloaded_results = [m["tool_call_id"] for m in reloaded if m["role"] == "tool"]
    assert reloaded_results == ["call_6", "call_7", "call_8", "call_9"]

    # Raw rows are soft-archived, never deleted, and still readable.
    with_inactive = db.get_messages(session_id, include_inactive=True)
    archived = [
        m for m in with_inactive
        if m["role"] == "tool" and m["tool_call_id"] == "call_0"
    ]
    assert len(archived) == 1
    assert archived[0]["content"].startswith("RESULT-0")

    # A second pass over the same transcript is a no-op (no repeat writes).
    assert db.deactivate_tool_results(session_id, ["call_0", "call_1"]) == 0


def test_deactivate_tool_results_only_touches_named_tool_rows(db):
    """The persist helper must not deactivate assistant/user rows."""
    messages = subagent_fixture(3)
    session_id = _persist_fixture(db, messages)

    flipped = db.deactivate_tool_results(session_id, ["call_0"])
    assert flipped == 1

    remaining = db.get_messages(session_id)
    assert [m["role"] for m in remaining].count("user") == 1
    assert [m["role"] for m in remaining].count("assistant") == 3
    assert [m["tool_call_id"] for m in remaining if m["role"] == "tool"] == [
        "call_1",
        "call_2",
    ]
