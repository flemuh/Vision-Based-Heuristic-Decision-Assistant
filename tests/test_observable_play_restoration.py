"""Regression for restored passive Observable and active Jogar modes."""
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path
import ast

import pytest

# The GUI depends on mss/Windows capture. Evaluate the actual selected GUI
# method ASTs to regression-test mode gates on Linux without mocking the app.
_GUI = Path(__file__).resolve().parents[1] / "vision_engine" / "gui.py"
_tree = ast.parse(_GUI.read_text(encoding="utf8"))
_cls = next(n for n in _tree.body if isinstance(n, ast.ClassDef) and n.name == "BingoApp")
_names = {"_recommend", "_compute_recommendations", "_solver_request", "_change_operation_mode",
          "_start_remote_solver", "_start_solver_stage", "_schedule_precompute", "_schedule_speculative_best_child"}
_methods = [n for n in _cls.body if isinstance(n, ast.FunctionDef) and n.name in _names]
_ast = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                        ast.ClassDef(name="BingoApp", bases=[], keywords=[], body=_methods, decorator_list=[])],
                  type_ignores=[])
_ast = ast.fix_missing_locations(_ast)
_env = {}
exec(compile(_ast, str(_GUI), "exec"), _env)
BingoApp = _env["BingoApp"]


def test_observable_recommend_is_strict_no_solver():
    host = SimpleNamespace(app_cfg=SimpleNamespace(operation_mode="observe"),
                           _show_observe_state=Mock())
    # If anything below the mode gate is accessed, this test fails.
    BingoApp._recommend(host)
    host._show_observe_state.assert_called_once_with()
    host._show_observe_state.reset_mock()
    assert BingoApp._compute_recommendations(host, 0, "LI") == []
    with pytest.raises(RuntimeError, match="Observable"):
        BingoApp._solver_request(host, 0, "LI", 20)


def test_observable_switch_cancels_work():
    cfg = SimpleNamespace(operation_mode="recommend", save=Mock())
    host = SimpleNamespace(
        app_cfg=cfg, app_cfg_path="config.json", operation_var=Mock(get=Mock(return_value="observe")),
        _invalidate_recommendation=Mock(),
        solver_bridge=SimpleNamespace(_drop_pending_decisions=Mock()),
        _show_observe_state=Mock(), _recommend=Mock(),
        db=Mock(), board=object(), box_color=None, episode_id="id", session_id="session",
    )
    host.app_cfg.risk_profile = "auto"
    BingoApp._change_operation_mode(host)
    assert cfg.operation_mode == "observe"
    host._invalidate_recommendation.assert_called_once()
    host.solver_bridge._drop_pending_decisions.assert_called_once_with()
    host._show_observe_state.assert_called_once()
    host._recommend.assert_not_called()
    host.db.ensure_game.assert_called_once()
    cfg.save.assert_called_once_with("config.json")


def test_switch_back_to_play_restarts_solver_if_monitoring():
    cfg = SimpleNamespace(operation_mode="observe", save=Mock(), risk_profile="auto")
    host = SimpleNamespace(
        app_cfg=cfg, app_cfg_path="config.json", operation_var=Mock(get=Mock(return_value="recommend")),
        _invalidate_recommendation=Mock(),
        status=Mock(), _recommend=Mock(), board=object(), current_jewel="LI", running=True,
        db=Mock(), box_color=None, episode_id="id", session_id="session",
    )
    BingoApp._change_operation_mode(host)
    assert cfg.operation_mode == "recommend"
    host._recommend.assert_called_once()
    host._invalidate_recommendation.assert_called_once()


def test_observe_disallows_async_precompute_and_solver_submissions():
    host = SimpleNamespace(app_cfg=SimpleNamespace(operation_mode="observe"))
    # Both remote and local solving short-circuit without any loaded state.
    assert BingoApp._start_remote_solver(host, 1, (), 0, "LI") is None
    assert BingoApp._start_solver_stage(host, 1, (), 0, "LI", "fast", 20) is None
    assert BingoApp._schedule_precompute(host) is None
    assert BingoApp._schedule_speculative_best_child(host, 0, []) is None


def test_gui_has_both_mode_controls_and_ml_learning_tab():
    from pathlib import Path
    source = Path(__file__).resolve().parents[1].joinpath("vision_engine", "gui.py").read_text(encoding="utf8")
    assert 'text="OBSERVABLE (sem cálculo)"' in source
    assert 'text="PLAY / JOGAR (solver 3/4)"' in source
    assert 'text="Aprendizado (ML)"' in source
    assert 'if self.app_cfg.operation_mode not in {"observe", "recommend"}:' in source
