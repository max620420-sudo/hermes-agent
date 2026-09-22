"""Tests for the interim_assistant_callback config gating in tui_gateway.

These tests exercise the real _agent_cbs() wiring rather than a local
imitation, so a break in the production callback registration is caught.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch


def test_load_interim_assistant_messages_defaults_true():
    from tui_gateway.server import _load_interim_assistant_messages

    with patch("tui_gateway.server._load_cfg", return_value={}):
        assert _load_interim_assistant_messages() is True


def test_agent_cbs_includes_interim_callback_when_enabled():
    """_agent_cbs() includes interim_assistant_callback when the config is on.

    Exercises the real _agent_cbs() wiring: the callback must be present in
    the returned dict and, when invoked, must emit a message.interim event
    with the text and already_streamed flag passed through.
    """
    from tui_gateway.server import _agent_cbs

    emitted: list[tuple] = []

    def fake_emit(event_type, sid, payload=None):
        emitted.append((event_type, sid, payload))

    with patch("tui_gateway.server._load_cfg", return_value={}), \
         patch("tui_gateway.server._emit", side_effect=fake_emit):
        cbs = _agent_cbs("test-session")

        assert "interim_assistant_callback" in cbs
        cb = cbs["interim_assistant_callback"]
        assert callable(cb)

        # Invoke the real callback inside the patch context — the lambda
        # resolves _emit by name at call time, so it must be called while
        # the patch is active.
        cb("hello world", already_streamed=True)

    assert len(emitted) == 1
    assert emitted[0][0] == "message.interim"
    assert emitted[0][1] == "test-session"
    assert emitted[0][2]["text"] == "hello world"
    assert emitted[0][2]["already_streamed"] is True


def test_interim_callback_records_first_visible_response_once():
    """The first visible commentary becomes a turn/session latency metric."""
    from tui_gateway import server

    agent = SimpleNamespace()
    session = {
        "agent": agent,
        "history_lock": threading.RLock(),
        "inflight_turn": {
            "started_at": 100.0,
            "started_monotonic": 50.0,
            "streaming": True,
        },
    }

    with patch.dict(server._sessions, {"test-session": session}, clear=True), \
         patch("tui_gateway.server._load_cfg", return_value={}), \
         patch("tui_gateway.server._emit"), \
         patch("tui_gateway.server.time.time", side_effect=[90.0, 108.0]), \
         patch("tui_gateway.server.time.monotonic", side_effect=[52.5, 58.0]):
        callback = getattr(server, "_agent_cbs")("test-session")["interim_assistant_callback"]
        callback("first update", already_streamed=True)
        callback("second update", already_streamed=True)

    assert session["inflight_turn"]["first_visible_response_s"] == 2.5
    assert agent._last_first_visible_response_s == 2.5
    assert agent._first_visible_response_history == [2.5]


def test_usage_reports_latest_and_recent_average_first_visible_response():
    from tui_gateway.server import _get_usage

    agent = SimpleNamespace(
        model="test-model",
        _first_visible_response_history=[2.0, 4.0],
        _last_first_visible_response_s=4.0,
    )

    usage = _get_usage(agent)

    assert usage["first_response_s"] == 4.0
    assert usage["avg_first_response_s"] == 3.0


def test_usage_omits_unmeasured_first_visible_response():
    from tui_gateway.server import _get_usage

    usage = _get_usage(SimpleNamespace(model="test-model"))

    assert "first_response_s" not in usage
    assert "avg_first_response_s" not in usage


def test_final_only_response_records_first_visible_response_before_usage():
    from tui_gateway import server

    agent = SimpleNamespace(model="test-model", provider="test-provider")
    session = {
        "agent": agent,
        "history_lock": threading.RLock(),
        "inflight_turn": {
            "assistant": "",
            "started_at": 100.0,
            "started_monotonic": 50.0,
            "streaming": True,
        },
    }
    turn = getattr(server, "_TurnRun")(
        agent=agent,
        one_turn_restore=None,
        terminal_callback=None,
        receipt_committed=False,
    )
    turn.result = {"final_response": "final only"}

    with patch.dict(server._sessions, {"test-session": session}, clear=True), \
         patch("tui_gateway.server.time.monotonic", return_value=52.25):
        payload, raw, status = getattr(server, "_complete_turn_payload")(
            "test-session", session, turn, None, 120)

    assert raw == "final only"
    assert status == "complete"
    assert payload["usage"]["first_response_s"] == 2.2
    assert agent._first_visible_response_history == [2.25]


def test_error_only_completion_does_not_create_assistant_response_metric():
    from tui_gateway import server

    agent = SimpleNamespace(model="test-model", provider="test-provider")
    session = {
        "agent": agent,
        "history_lock": threading.RLock(),
        "inflight_turn": {
            "assistant": "",
            "started_at": 100.0,
            "started_monotonic": 50.0,
            "streaming": True,
        },
    }
    turn = getattr(server, "_TurnRun")(
        agent=agent,
        one_turn_restore=None,
        terminal_callback=None,
        receipt_committed=False,
    )
    turn.result = {
        "error": "provider unavailable",
        "failed": True,
        "final_response": "",
    }

    with patch.dict(server._sessions, {"test-session": session}, clear=True), \
         patch("tui_gateway.server.time.monotonic", return_value=52.0):
        payload, raw, status = getattr(server, "_complete_turn_payload")(
            "test-session", session, turn, None, 120)

    assert raw == "Error: provider unavailable"
    assert status == "error"
    assert "first_response_s" not in payload["usage"]
    assert not hasattr(agent, "_first_visible_response_history")


def _complete_metric_result(server, agent, result):
    session = {
        "agent": agent,
        "history_lock": threading.RLock(),
        "inflight_turn": {
            "assistant": "",
            "started_at": 100.0,
            "started_monotonic": 50.0,
            "streaming": True,
        },
    }
    turn = getattr(server, "_TurnRun")(
        agent=agent,
        one_turn_restore=None,
        terminal_callback=None,
        receipt_committed=False,
    )
    turn.result = result

    with patch.dict(server._sessions, {"test-session": session}, clear=True), \
         patch("tui_gateway.server.time.monotonic", return_value=52.0):
        return getattr(server, "_complete_turn_payload")(
            "test-session", session, turn, None, 120)


def test_suppressed_interrupt_sentinel_does_not_create_response_metric():
    from tui_gateway import server

    agent = SimpleNamespace(model="test-model", provider="test-provider")
    sentinel = f"{server.INTERRUPT_WAITING_FOR_MODEL_PREFIX} (cancelled)"

    payload, raw, status = _complete_metric_result(
        server, agent, {"final_response": sentinel, "interrupted": True})

    assert raw == ""
    assert status == "interrupted"
    assert "first_response_s" not in payload["usage"]
    assert not hasattr(agent, "_first_visible_response_history")


def test_suppressed_interrupt_sentinel_preserves_existing_response_metrics():
    from tui_gateway import server

    agent = SimpleNamespace(
        model="test-model",
        provider="test-provider",
        _last_first_visible_response_s=1.5,
        _first_visible_response_history=[1.5],
    )
    sentinel = f"{server.INTERRUPT_WAITING_FOR_MODEL_PREFIX} (cancelled)"

    payload, raw, status = _complete_metric_result(
        server, agent, {"final_response": sentinel, "interrupted": True})

    assert raw == ""
    assert status == "interrupted"
    assert payload["usage"]["first_response_s"] == 1.5
    assert agent._first_visible_response_history == [1.5]


def test_interrupted_visible_assistant_text_records_response_metric():
    from tui_gateway import server

    agent = SimpleNamespace(model="test-model", provider="test-provider")

    payload, raw, status = _complete_metric_result(
        server, agent, {"final_response": "partial answer", "interrupted": True})

    assert raw == "partial answer"
    assert status == "interrupted"
    assert payload["usage"]["first_response_s"] == 2.0
    assert agent._first_visible_response_history == [2.0]


def test_invoke_agent_override_records_first_nonblank_event_once():
    from tui_gateway import server

    class _Stop:
        def set(self):
            pass

    class _Thread:
        def join(self):
            pass

    class _Agent:
        def __init__(self):
            self.interim_assistant_callback = None

        def run_conversation(self, _message, **kwargs):
            kwargs["stream_callback"]("  ")
            assert self.interim_assistant_callback is not None
            self.interim_assistant_callback("commentary", already_streamed=True)
            kwargs["stream_callback"]("final")
            return {"final_response": "final"}

    agent = _Agent()
    session = {
        "agent": agent,
        "history_lock": threading.RLock(),
        "session_key": "stored-session",
        "inflight_turn": {
            "assistant": "",
            "started_at": 100.0,
            "started_monotonic": 50.0,
            "streaming": True,
        },
    }
    turn = getattr(server, "_TurnRun")(
        agent=agent,
        one_turn_restore=None,
        terminal_callback=None,
        receipt_committed=False,
    )
    emitted = []

    with patch.dict(server._sessions, {"test-session": session}, clear=True), \
         patch("tui_gateway.server._load_interim_assistant_messages", return_value=True), \
         patch("tui_gateway.server._start_usage_ticker", return_value=(_Stop(), _Thread())), \
         patch("tui_gateway.server._emit", side_effect=lambda event, sid, payload: emitted.append((event, sid, payload))), \
         patch("tui_gateway.server.time.monotonic", return_value=52.0):
        getattr(server, "_invoke_agent")(
            "test-session", session, turn, "prompt", "prompt", None, [], None, None)

    assert getattr(agent, "_first_visible_response_history") == [2.0]
    assert [event for event, _sid, _payload in emitted] == [
        "message.delta", "message.interim", "message.delta"]


def test_first_visible_response_history_keeps_last_ten_turns():
    from tui_gateway import server

    agent = SimpleNamespace()
    session = {"agent": agent, "history_lock": threading.RLock()}

    with patch.dict(server._sessions, {"test-session": session}, clear=True), \
         patch("tui_gateway.server.time.monotonic", side_effect=range(1, 13)):
        for _ in range(12):
            session["inflight_turn"] = {"started_monotonic": 0.0, "streaming": True}
            getattr(server, "_mark_first_visible_response")("test-session", "test")

    assert agent._first_visible_response_history == [float(value) for value in range(3, 13)]


