from pathlib import Path

from vision_engine import autoplay


def test_non_windows_preflight_is_safe(monkeypatch):
    monkeypatch.setattr(autoplay.sys, "platform", "linux")
    ok, detail = autoplay.windows_input_preflight()
    assert ok is False
    assert "Windows" in detail


def test_autoplay_source_has_cursor_fallback_and_privilege_diagnostic():
    src = Path(autoplay.__file__).read_text(encoding="utf-8")
    assert "_sendinput_move" in src
    assert "SendInput fallback" in src
    assert "same privilege level as MU" in src


def test_box_control_remains_optional_metadata_only():
    src = (Path(__file__).parents[1] / "vision_engine" / "gui.py").read_text(encoding="utf-8")
    assert 'text="Box (optional):"' in src
    assert "optional research metadata" in src
