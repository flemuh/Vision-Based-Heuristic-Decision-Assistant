from __future__ import annotations

import ctypes
from unittest import mock

import pytest

from vision_engine.input.geometry import AutoPlayPoint
from vision_engine.input import win32


class _RejectingUser32:
    def SetCursorPos(self, _x, _y):
        return False


def test_intermediate_api_rejection_is_not_fatal(monkeypatch):
    start = AutoPlayPoint(100, 100)
    target = AutoPlayPoint(400, 300)
    # Only start + final physical checks are consumed. Intermediate waypoints
    # deliberately do not call GetCursorPos.
    monkeypatch.setattr(win32.sys, "platform", "win32")
    monkeypatch.setattr(win32, "_windows_user32", lambda: _RejectingUser32())
    monkeypatch.setattr(win32, "_cursor_position", mock.Mock(side_effect=[start, target]))
    monkeypatch.setattr(win32, "_sendinput_move", lambda *_a: False)
    monkeypatch.setattr(win32, "_sendinput_left_click", lambda *_a: True)
    monkeypatch.setattr(win32.time, "sleep", lambda *_a: None)
    monkeypatch.setattr(ctypes, "set_last_error", lambda *_a: None, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 0, raising=False)
    backend = win32.windows_move_and_left_click(target, duration_ms=64)
    assert backend == "SetCursorPos"


def test_final_position_still_guards_the_click(monkeypatch):
    start = AutoPlayPoint(100, 100)
    wrong = AutoPlayPoint(250, 200)
    target = AutoPlayPoint(400, 300)
    monkeypatch.setattr(win32.sys, "platform", "win32")
    monkeypatch.setattr(win32, "_windows_user32", lambda: _RejectingUser32())
    monkeypatch.setattr(win32, "_cursor_position", mock.Mock(return_value=wrong))
    # First GetCursorPos must be a valid start.
    positions = iter([start, wrong, wrong])
    monkeypatch.setattr(win32, "_cursor_position", lambda *_a: next(positions))
    monkeypatch.setattr(win32, "_sendinput_move", lambda *_a: False)
    monkeypatch.setattr(win32, "_probe_rejection", lambda *_a, **_k: {"classification": "PERSISTENT_REJECTION"})
    click = mock.Mock(return_value=True)
    monkeypatch.setattr(win32, "_sendinput_left_click", click)
    monkeypatch.setattr(win32.time, "sleep", lambda *_a: None)
    monkeypatch.setattr(ctypes, "set_last_error", lambda *_a: None, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 0, raising=False)
    with pytest.raises(RuntimeError, match="final click target"):
        win32.windows_move_and_left_click(target, duration_ms=64)
    click.assert_not_called()


def test_final_probe_recovery_allows_click(monkeypatch):
    start = AutoPlayPoint(100, 100)
    wrong = AutoPlayPoint(330, 260)
    target = AutoPlayPoint(400, 300)
    fake = type("U", (), {"SetCursorPos": lambda self, x, y: True})()
    monkeypatch.setattr(win32.sys, "platform", "win32")
    monkeypatch.setattr(win32, "_windows_user32", lambda: fake)
    monkeypatch.setattr(win32, "_cursor_position", mock.Mock(side_effect=[start, wrong, target]))
    monkeypatch.setattr(win32, "_probe_rejection", lambda *_a, **_k: {"classification": "TRANSIENT_REJECTION"})
    click = mock.Mock(return_value=True)
    monkeypatch.setattr(win32, "_sendinput_left_click", click)
    monkeypatch.setattr(win32.time, "sleep", lambda *_a: None)
    monkeypatch.setattr(ctypes, "set_last_error", lambda *_a: None, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 0, raising=False)
    backend = win32.windows_move_and_left_click(target, duration_ms=64)
    assert backend == "delayed final recovery"
    click.assert_called_once()
