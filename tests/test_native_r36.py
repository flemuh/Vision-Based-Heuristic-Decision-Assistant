"""R3.6 native persistent backend and adaptive-budget invariants."""
from __future__ import annotations
from types import SimpleNamespace


from vision_engine.config import AppConfig
from vision_engine.solver.engine import SolverEngine
from vision_engine.solver.native import ensure_native_solver
from vision_engine.solver.protocol import SolverRequest, SolverSettings
from vision_engine.live.state import GameState
from vision_engine.solver.worker import advisor_signature


BOARD = ('CH','HA','SO','CH','HA', 'SO','BL','BL','CR','LI',
         'BL','CH',None,'CR','HA', 'BL','SO','CH','LI','SO',
         'LI','CR','CR','LI','HA')


def _request(purpose='live-final', rollouts=120):
    state = GameState('native-test',0,0,'hash',BOARD,0,'BL','auto')
    return SolverRequest(1,state,SolverSettings(rollouts=rollouts),purpose=purpose,session_id='s')


def _rec(*,confidence='LOW_SAMPLE',method='monte-carlo',tie=False):
    return SimpleNamespace(position=2, method=method, primary_stat_tie=tie,
                           tie_kind='', confidence=confidence)


def test_native_defaults_and_legacy_override(tmp_path,monkeypatch):
    assert AppConfig().solver_backend == 'native'
    assert AppConfig().solver_port == 57642
    config = tmp_path/'app_config.json'
    config.write_text('{"solver_backend":"wsl", "solver_port":57641}')
    monkeypatch.setenv('SPEEDLORA_SOLVER_BACKEND','native')
    cfg = AppConfig.load(config)
    assert (cfg.solver_backend, cfg.solver_port, cfg.solver_workers) == ('native',57642,2)
    assert 'wsl' in config.read_text()  # migration does not overwrite user files


def test_worker_signature_ignores_budget_but_retains_horizon():
    request = dict(board_cells=BOARD,mask=0,jewel='BL',rollouts=120,rollout_exact_horizon=3)
    assert advisor_signature(request) == advisor_signature({**request,'rollouts':1000})
    assert advisor_signature(request) != advisor_signature({**request,'rollout_exact_horizon':4})


def test_no_refinement_for_exact_or_clear():
    assert not SolverEngine._needs_refinement([_rec(method='immediate-lucky-fastpath'),_rec()])
    assert not SolverEngine._needs_refinement([_rec(method='reward-tier-exact'),_rec()])
    assert not SolverEngine._needs_refinement([_rec(confidence='RANKED'),_rec()])
    assert SolverEngine._needs_refinement([_rec(),_rec()])


def test_adaptive_stops_for_slow_or_clear(monkeypatch):
    # Don't start a worker here; mock the persistent pool to assert budget choice.
    class FakePool:
        def __init__(self,slow=False,clear=False):
            self.calls=[]
        def submit(self,fn,req):
            self.calls.append(req['rollouts'])
            return SimpleNamespace(result=lambda:[_rec(confidence='RANKED'),_rec()])
    engine=object.__new__(SolverEngine)
    engine.foreground=FakePool()
    recs,budget=engine._solve_adaptive(_request())
    assert budget == 120
    assert engine.foreground.calls == [120]


def test_non_live_final_doesnt_escalate(monkeypatch):
    class FakePool:
        def __init__(self): self.calls=[]
        def submit(self,fn,req):
            self.calls.append(req['rollouts'])
            return SimpleNamespace(result=lambda:[_rec(),_rec()])
    engine=object.__new__(SolverEngine)
    engine.foreground=FakePool()
    assert engine._solve_adaptive(_request('live-preview',24))[1]==24
    assert engine.foreground.calls==[24]


def test_native_doesnt_start_if_non_loopback(tmp_path):
    ok,process,info = ensure_native_solver(tmp_path,'192.0.2.1',57642,2)
    assert not ok and process is None
    assert 'loopback' in info


def test_fast_ambiguous_decision_can_escalate_to_1000():
    class FastPool:
        calls=[]
        def submit(self,fn,req):
            self.calls.append(req['rollouts'])
            return SimpleNamespace(result=lambda:[_rec(confidence='LOW_SAMPLE'),_rec()])
    engine=object.__new__(SolverEngine)
    engine.foreground=FastPool()
    _, budget=engine._solve_adaptive(_request('live-final',120))
    assert budget == 1000
    assert engine.foreground.calls == [120,240,500,1000]


def test_expensive_ambiguous_decision_does_not_escalate(monkeypatch):
    class SlowPool:
        def __init__(self): self.calls=[]
        def submit(self,fn,req):
            import time
            self.calls.append(req['rollouts'])
            time.sleep(0.05)
            return SimpleNamespace(result=lambda:[_rec(confidence='LOW_SAMPLE'),_rec()])
    engine=object.__new__(SolverEngine)
    engine.foreground=SlowPool()
    # At the 120 budget a short artificial >1.2s estimate for 240 should skip.
    import vision_engine.solver.engine as impl
    timestamps=iter([0.,0.8,0.8])
    monkeypatch.setattr(impl.time,'perf_counter',lambda: next(timestamps))
    _,budget=engine._solve_adaptive(_request('live-final',120))
    assert budget == 120
    assert engine.foreground.calls == [120]
