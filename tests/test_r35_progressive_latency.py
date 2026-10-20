"""R3.5: early display is provisional and can never be logged as a decision or train ML."""
import ast
import time
from pathlib import Path
from types import SimpleNamespace

from vision_engine.config import AppConfig
from vision_engine.state_machine import GamePhase

# Extract the real methods so this test runs without Windows/Tk capture deps.
GUI_PATH = Path(__file__).resolve().parents[1] / "vision_engine/gui.py"
parsed = ast.parse(GUI_PATH.read_text(encoding="utf-8"))
app_cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "BingoApp")
names = {"_start_remote_solver", "_handle_remote_solver_result", "_handle_solver_payload"}
methods = [n for n in app_cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
synthetic = ast.ClassDef(name="SyntheticGUI", bases=[], keywords=[], body=methods, decorator_list=[])
module = ast.fix_missing_locations(ast.Module(body=[synthetic], type_ignores=[]))
ns = {"GamePhase": GamePhase, "time": time, "SolverRequest": SimpleNamespace}
exec(compile(module, str(GUI_PATH), "exec"), ns)
BingoApp = ns["SyntheticGUI"]


class FakeBridge:
    def __init__(self):
        self.requests = []

    def solve_async(self, request, result, error):
        self.requests.append((request, result, error))
        return {"accepted": True, "generation": len(self.requests), "superseded": False}


class FakeGUI:
    _start_remote_solver = BingoApp._start_remote_solver
    _handle_remote_solver_result = BingoApp._handle_remote_solver_result

    def __init__(self):
        self.app_cfg = AppConfig()
        self.app_cfg.operation_mode = "recommend"
        self.recommend_request_id = 7
        self.current_key = ("board", 0, "BL")
        self.phase = GamePhase.PLAYING
        self.selected_mask = 0
        self.remote_solver_active_key = None
        self.cached_recommend_key = None
        self.last_recs = []
        self.live_coordinator = SimpleNamespace(accepts=lambda _res: True)
        self.solver_bridge = FakeBridge()
        self.db = SimpleNamespace(save_telemetry=lambda *x: None)
        self.episode_id = "episode"
        self.rendered = []
        self.ml_persisted = 0
        self.plan_adopted = 0
        self.speculative = 0
        self.events = []

    def _recommend_state_key(self):
        return self.current_key

    def _protocol_solver_request(self, request_id, mask, jewel, rollouts, *, purpose):
        return SimpleNamespace(request_id=request_id, rollouts=rollouts, purpose=purpose)

    def _ui_call(self, func, *args):
        func(*args)

    def _render_recommendations(self, mask, jewel, recs, refined=True):
        self.rendered.append((mask, jewel, list(recs), refined))

    def _runtime_record(self, kind, **data):
        self.events.append((kind, data))

    def _persist_counterfactuals(self, mask, jewel, recs):
        self.ml_persisted += 1

    def _adopt_auto_plan(self, recs):
        self.plan_adopted += 1

    def _schedule_speculative_best_child(self, mask, recs):
        self.speculative += 1

    def _auto_log(self, *args):
        pass

    def _refresh_system_panel(self):
        pass


def response(method='monte-carlo'):
    return SimpleNamespace(
        recommendations=[SimpleNamespace(position=3, method=method)],
        elapsed_ms=300, cache_hit=False,
    )


def test_remote_preview_precedes_final_but_does_not_commit():
    gui = FakeGUI()
    gui._start_remote_solver(7, gui.current_key, 0, 'BL')
    assert gui.solver_bridge.requests[0][0].rollouts == 24
    request, complete, error = gui.solver_bridge.requests[0]
    complete(response())
    assert len(gui.rendered) == 1
    assert gui.rendered[0][3] is False
    assert gui.cached_recommend_key is None
    assert gui.remote_solver_active_key == gui.current_key
    assert (gui.ml_persisted, gui.plan_adopted, gui.speculative) == (0, 0, 0)
    assert gui.solver_bridge.requests[1][0].rollouts == 120
    gui.solver_bridge.requests[1][1](response())
    assert gui.rendered[-1][3] is True
    assert gui.cached_recommend_key == gui.current_key
    assert gui.remote_solver_active_key is None
    assert (gui.ml_persisted, gui.plan_adopted, gui.speculative) == (1, 1, 1)


def test_exact_preview_skips_refinement_and_is_authoritative():
    gui = FakeGUI()
    gui._start_remote_solver(7, gui.current_key, 0, 'BL')
    gui.solver_bridge.requests[0][1](response('immediate-lucky-fastpath'))
    assert len(gui.solver_bridge.requests) == 1
    assert gui.rendered[-1][3] is True
    assert gui.ml_persisted == 0
    assert gui.plan_adopted == 1


def test_stale_preview_never_displays_or_submits_final():
    gui = FakeGUI()
    gui._start_remote_solver(7, gui.current_key, 0, 'BL')
    gui.current_key = ("next-board", 1, "SO")
    gui.solver_bridge.requests[0][1](response())
    assert gui.rendered == []
    assert len(gui.solver_bridge.requests) == 1
    assert gui.ml_persisted == 0


def test_observable_never_submits_solver_request():
    gui = FakeGUI()
    gui.app_cfg.operation_mode = 'observe'
    gui._start_remote_solver(7, gui.current_key, 0, 'BL')
    assert gui.solver_bridge.requests == []


def test_user_can_disable_progressive_mode_without_legacy_1600():
    gui = FakeGUI()
    gui.app_cfg.progressive_live_decisions = False
    gui._start_remote_solver(7, gui.current_key, 0, 'BL')
    assert gui.solver_bridge.requests[0][0].rollouts == 120
    assert gui.solver_bridge.requests[0][0].purpose == 'live-final'
    gui.solver_bridge.requests[0][1](response())
    assert len(gui.solver_bridge.requests) == 1
    assert gui.rendered[-1][3] is True


def test_local_fast_stage_runs_24_then_120_and_does_not_log_preview():
    gui = FakeGUI()
    gui.app_cfg.progressive_live_decisions = True
    gui.app_cfg.exact_horizon = 6
    gui.selected_mask = 0
    gui.status = SimpleNamespace(set=lambda *a: None)
    gui.after = lambda delay, callback, *args: scheduled.append((delay, callback, args))
    scheduled = []
    gui._start_solver_stage = lambda *args: gui.events.append(('local-start', args))
    # Invoked as a real method, but without creating Tk or Windows inputs.
    BingoApp._handle_solver_payload(gui, 7, gui.current_key, 0, 'BL', 'fast',
                                    'ok', [SimpleNamespace(position=3, method='monte-carlo')],
                                    0.2, 'process')
    assert gui.rendered[-1][3] is False
    assert gui.cached_recommend_key is None
    assert gui.ml_persisted == 0
    assert scheduled[0][2][-1] == 120
    assert scheduled[0][2][4] == 'refine'


def test_real_gui_never_persists_preview_as_last_authoritative_recommendation():
    source = GUI_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BingoApp')
    render = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '_render_recommendations')
    writes = [n for n in ast.walk(render) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Attribute) and t.attr == 'last_recs' for t in n.targets)]
    assert len(writes) == 1
    assert writes[0].lineno > next(n.lineno for n in ast.walk(render) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'refined')


def test_monitor_repaints_grid_only_when_board_or_mask_changed():
    source = GUI_PATH.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'BingoApp')
    method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '_monitor_refresh_grid_if_changed')
    mod = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Refresh', bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[]))
    namespace = {'board_hash': lambda board: board}
    exec(compile(mod, str(GUI_PATH), 'exec'), namespace)
    host = namespace['Refresh']()
    host.board = 'board1'
    host.episode_id = 'episode'
    host.selected_mask = 0
    host._last_monitor_grid_signature = None
    seen = []
    host._refresh_grid = lambda: seen.append((host.board, host.selected_mask))
    host._monitor_refresh_grid_if_changed()
    host._monitor_refresh_grid_if_changed()
    assert seen == [('board1', 0)]
    host.selected_mask = 1
    host._monitor_refresh_grid_if_changed()
    assert seen[-1] == ('board1', 1)
    host.board = 'board2'
    host._monitor_refresh_grid_if_changed()
    assert seen[-1] == ('board2', 1)
