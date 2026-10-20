"""R3.2: concise learning dashboard, no changes to decision math or live training."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

GUI = Path(__file__).resolve().parents[1] / 'vision_engine' / 'gui.py'
SOURCE = GUI.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)
APP = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'BingoApp')
METHODS = {n.name: n for n in APP.body if isinstance(n, ast.FunctionDef)}


def test_learning_dashboard_defaults_to_simple_summary_and_retains_advanced_values():
    src = ast.get_source_segment(SOURCE, METHODS['_build_learning_tab'])
    assert 'O ML está ajudando?' in src
    assert 'Última partida analisada' in src
    for name in ('learning_games_count_var', 'learning_wins_count_var',
                 'learning_opps_count_var', 'learning_changes_count_var'):
        assert name in src
    assert 'self.learning_details_visible = False' in src
    assert 'self.learning_details_frame.pack(' not in src
    assert 'self.learning_pattern_tree' in src
    assert 'self.learning_metrics_var' in src
    assert 'Reprocessar aprendizado do histórico' in src
    assert 'self.mode_toolbar' not in src  # unchanged always-visible mode switch


def test_learning_dashboard_does_not_call_solver_or_retrain_on_refresh():
    src = ast.get_source_segment(SOURCE, METHODS['_refresh_learning_panel'])
    for phrase in ('solve_request(', '.train(', '_start_solver_stage(', 'MonteCarlo'):
        assert phrase not in src
    assert "influence['changes']" in src
    assert "audit['recoverable_gt1000']" in src
    assert 'não representa uma vitória real' in src
    assert 'Ainda NÃO foi medido' in src


@pytest.mark.skipif(os.getenv('SPEEDLORA_TK_TEST') != '1', reason='requires Tk display')
def test_learning_dashboard_renders_with_real_tk_and_preserves_modes():
    import tkinter as tk
    from tkinter import ttk

    names = ('_build', '_build_learning_tab', '_toggle_learning_details', '_refresh_learning_panel',
             '_change_operation_mode')
    synthetic_cls = ast.ClassDef(name='SyntheticBingo', bases=[], keywords=[],
                                 body=[METHODS[n] for n in names], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[synthetic_cls], type_ignores=[]))
    def snapshot():
        return {
            'completed_games': 151, 'verified_games': 1,
            'retrospective': {'analyzed': 130, 'recoverable_gt1000': 39},
            'live_influence': {'matches': 0, 'changes': 0, 'decisions': 42},
            'cf_status': 'LEARNING', 'cf_samples': 781, 'cf_validation_games': 7,
            'outcome_status': 'WAITING', 'outcome_samples': 0, 'outcome_validation_games': 0,
            'hindsight_status': 'LEARNING', 'hindsight_samples': 2544, 'hindsight_games': 57,
            'hindsight_validation_games': 19, 'required_validation_games': 8,
            'pattern_stage': 'ELIGIBLE FOR VALIDATION', 'pattern_games': 151,
            'repeated_layouts': 5, 'rng_influence_enabled': False,
            'outcome_improvement': None, 'outcome_residual_mae': None,
            'mae_p3': 0.093, 'mae_gt1000': 0.116, 'mae_score': 53.8,
            'hindsight_mae_p3': 0.26, 'hindsight_mae_gt1000': 0.362,
            'hindsight_mae_score': 65.8,
            'jewel_rates': [{'jewel': x, 'observed': 1/6, 'expected': 1/6, 'delta': 0}
                            for x in ('BL', 'SO', 'LI', 'CR', 'HA', 'CH')],
        }
    class FakeDB:
        def connect(self):
            class Conn:
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def execute(self, *args):
                    return SimpleNamespace(fetchone=lambda: (json.dumps({
                        'valid': True, 'observed_total': 999, 'best_total': 999,
                        'first_lost_gt1000': None, 'first_better_choice': None
                    }),))
            return Conn()
    env = {'tk': tk, 'ttk': ttk, 'CENTER': 12,
           'build_learning_snapshot': lambda *args, **kwargs: snapshot(),
           'json': json, 'Board': SimpleNamespace(rc=lambda x: (x//5+1, x%5+1))}
    exec(compile(module, str(GUI), 'exec'), env)
    synthetic = env['SyntheticBingo']

    class Host(tk.Tk):
        _build = synthetic._build
        _build_learning_tab = synthetic._build_learning_tab
        _toggle_learning_details = synthetic._toggle_learning_details
        _refresh_learning_panel = synthetic._refresh_learning_panel
        _change_operation_mode = synthetic._change_operation_mode

        def __init__(self):
            super().__init__()
            self.geometry('810x660')
            self.app_cfg = SimpleNamespace(operation_mode='observe',
                           ml_min_validation_games=8, rng_min_games=50,
                           empirical_bias_blend=0, risk_profile='auto', save=Mock())
            self.app_cfg_path = 'unused'
            self.running = False
            self.board = None
            self.current_jewel = None
            self.db = FakeDB()
            self.experience = SimpleNamespace(info=object(), winning_memory=SimpleNamespace(
                examples=[object() for _ in range(27)], _by_state={i: 1 for i in range(378)}))
            self.solver_bridge = SimpleNamespace(_drop_pending_decisions=Mock())
            self._invalidate_recommendation = Mock()
            self._show_observe_state = Mock()
            for name in ('auto_setup', '_open_repair_dialog', 'toggle', '_on_box_color_change',
                         '_clear_box_color', 'toggle_cell',
                         '_on_main_tab_changed', '_rebuild_learning_history_async'):
                setattr(self, name, Mock())
            self._build()

    host = Host()
    try:
        host.notebook.select(host.learning_tab)
        host._refresh_learning_panel()
        host.update_idletasks()
        assert host.learning_games_count_var.get() == '151'
        assert host.learning_wins_count_var.get() == '27'
        assert host.learning_opps_count_var.get() == '39'
        assert host.learning_changes_count_var.get() == '0'
        assert 'não alterou' in host.learning_status_var.get()
        assert 'Fez 999 pontos' in host.learning_last_audit_var.get()
        assert not host.learning_details_frame.winfo_manager()
        assert host.observable_mode_button.winfo_ismapped()
        assert host.play_mode_button.winfo_ismapped()
        assert host.learning_scroll.winfo_ismapped()
        host.learning_scroll.yview_moveto(1.0)
        host.update_idletasks()
        assert host.learning_details_button.winfo_rooty() < host.winfo_rooty() + host.winfo_height()
        host.learning_details_button.invoke()
        host.update_idletasks()
        assert host.learning_details_visible
        assert host.learning_pattern_tree.get_children()
        host.learning_scroll.yview_moveto(1.0)
        host.update_idletasks()
        assert host.learning_pattern_tree.winfo_rooty() < host.winfo_rooty() + host.winfo_height()
        assert host.observable_mode_button.winfo_ismapped()
        host.learning_details_button.invoke()
        host.update_idletasks()
        assert not host.learning_details_visible
        assert not host.learning_details_frame.winfo_manager()
        # No rebuilding needed to switch tabs and modes.
        host.notebook.select(0)
        host.update_idletasks()
        assert host.play_mode_button.winfo_ismapped()
    finally:
        host.destroy()
