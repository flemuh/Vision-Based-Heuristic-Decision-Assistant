"""R3.1: regression of the always-visible Game/ML operation-mode toolbar."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

GUI = Path(__file__).resolve().parents[1] / "vision_engine" / "gui.py"
source = GUI.read_text(encoding="utf-8")
parsed = ast.parse(source)
cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "BingoApp")
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in (
    "_build", "_build_learning_tab", "_change_operation_mode",
)]


def test_operation_buttons_always_above_both_tabs():
    body = next(n for n in methods if n.name == "_build")
    text = ast.get_source_segment(source, body)
    assert text.index('self.mode_toolbar.pack(') < text.index('self.notebook = ttk.Notebook(self)')
    assert 'mode_actions, text="OBSERVABLE (sem cálculo)"' in text
    assert 'mode_actions, text="PLAY / JOGAR (solver 3/4)"' in text
    assert "variable=self.operation_var" in text
    assert "value=\"observe\"" in text
    assert "value=\"recommend\"" in text
    assert "self.mode_toolbar = ttk.LabelFrame(self" in text


@pytest.mark.skipif(os.getenv("SPEEDLORA_TK_TEST") != "1", reason="requires an X11 Tk display")
def test_modes_remain_visible_with_game_and_ml_selected():
    import tkinter as tk
    from tkinter import ttk

    synthetic_cls = ast.ClassDef(name="SyntheticBingo", bases=[], keywords=[], body=methods, decorator_list=[])
    mod = ast.fix_missing_locations(ast.Module(body=[synthetic_cls], type_ignores=[]))
    env = {"tk": tk, "ttk": ttk, "CENTER": 12}
    exec(compile(mod, str(GUI), "exec"), env)
    solver_ui = env["SyntheticBingo"]

    class Host(tk.Tk):
        _build = solver_ui._build
        _build_learning_tab = solver_ui._build_learning_tab
        _change_operation_mode = solver_ui._change_operation_mode

        def __init__(self, initial):
            super().__init__()
            self.app_cfg = SimpleNamespace(operation_mode=initial,
                                            save=Mock(), risk_profile="auto")
            self.app_cfg_path = "/unused/config.json"
            self.running = False
            self.board = None
            self.current_jewel = None
            self.solver_bridge = SimpleNamespace(_drop_pending_decisions=Mock())
            self._invalidate_recommendation = Mock()
            self._show_observe_state = Mock()
            for cb in ("auto_setup", "_open_repair_dialog", "toggle",
                       "_on_box_color_change", "_clear_box_color",
                       "toggle_cell", "_on_main_tab_changed", "_rebuild_learning_history_async"):
                setattr(self, cb, Mock())
            self._build()
            self.update_idletasks()

    for starting_mode in ("observe", "recommend"):
        ui = Host(starting_mode)
        try:
            assert ui.observable_mode_button.winfo_manager() == "pack"
            assert ui.play_mode_button.winfo_manager() == "pack"
            assert ui.observable_mode_button.winfo_ismapped()
            assert ui.play_mode_button.winfo_ismapped()
            assert ui.operation_var.get() == starting_mode
            for tab in (0, 1):
                ui.notebook.select(tab)
                ui.update_idletasks()
                assert ui.observable_mode_button.winfo_ismapped()
                assert ui.play_mode_button.winfo_ismapped()
            ui.observable_mode_button.invoke()
            ui.update_idletasks()
            assert ui.app_cfg.operation_mode == "observe"
            assert not hasattr(ui, "autoplay_btn")  # recommendation-only build
            assert "OBSERVABLE ativo" in ui.mode_status_var.get()
            assert ui.notebook.index(ui.notebook.select()) == 0
            ui.play_mode_button.invoke()
            ui.update_idletasks()
            assert ui.app_cfg.operation_mode == "recommend"
            assert "PLAY ativo" in ui.mode_status_var.get()
        finally:
            ui.destroy()
