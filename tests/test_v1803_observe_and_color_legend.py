from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_observe_check_happens_before_refined_board_painting():
    source = (ROOT / 'vision_engine' / 'gui.py').read_text(encoding='utf-8')
    start = source.index('def _render_recommendations')
    end = source.index('def ', start + 10)
    body = source[start:end]
    observe = body.index('if self.app_cfg.operation_mode == "observe":')
    paint = body.index('if refined:')
    assert observe < paint
    assert 'self._ranked_grid_recs = []' in body[observe:paint]
    assert 'No ranking colors are shown in Observe mode.' in body


def test_live_and_analysis_explain_every_semantic_color():
    source = (ROOT / 'vision_engine' / 'gui.py').read_text(encoding='utf-8')
    for marker in (
        'GREEN Best/Equal',
        'YELLOW Close',
        'ORANGE Less flex',
        'AMBER Forced slack',
        'PURPLE Fallback',
        'BLUE Already placed',
        'Color guide: GREEN = best or exact equal',
    ):
        assert marker in source


def test_v1803_does_not_change_core_contracts():
    from vision_engine import __version__
    from vision_engine.versioning import PROTOCOL_VERSION, SCHEMA_VERSION, MODEL_VERSION, STRATEGY_PLANNER_VERSION
    assert __version__ == '18.0.26-STATE-INTEGRITY-P1'
    assert PROTOCOL_VERSION == 4
    assert SCHEMA_VERSION == 9
    assert MODEL_VERSION == 6
    assert STRATEGY_PLANNER_VERSION == 22
