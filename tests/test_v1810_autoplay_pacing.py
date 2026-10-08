import random

from vision_engine.autoplay import bounded_random_ms, paced_preclick_delay_ms
from vision_engine.config import AppConfig


def test_v1810_defaults_target_about_ten_hours_for_700_rounds():
    cfg = AppConfig()
    assert cfg.autoplay_target_game_seconds == 51.4
    total_hours = cfg.autoplay_target_game_seconds * 700 / 3600.0
    assert 9.8 <= total_hours <= 10.2
    assert cfg.autoplay_transition_timeout_ms >= 6000


def test_v1810_safety_floor_is_never_shortened_by_pacing():
    delay = paced_preclick_delay_ms(
        campaign_started_at=0.0, now=999.0, completed_games=10, placed_count=5,
        target_game_seconds=51.4, expected_transition_ms=550,
        safety_delay_ms=1775, max_pacing_wait_ms=4200,
    )
    assert delay == 1775


def test_v1810_pacing_waits_when_run_is_ahead():
    # At campaign t=0, first click target is roughly one 1/14 slot minus
    # expected transition time; pacing should dominate the safety floor.
    delay = paced_preclick_delay_ms(
        campaign_started_at=0.0, now=0.4, completed_games=0, placed_count=0,
        target_game_seconds=51.4, expected_transition_ms=550,
        safety_delay_ms=1200, max_pacing_wait_ms=4200,
    )
    assert 2500 <= delay <= 3000


def test_v1810_current_random_safety_bounds_are_conservative():
    cfg = AppConfig()
    rng = random.Random(1810)
    totals = [
        bounded_random_ms(cfg.autoplay_click_delay_min_ms, cfg.autoplay_click_delay_max_ms, rng=rng)
        + bounded_random_ms(cfg.autoplay_extra_settle_min_ms, cfg.autoplay_extra_settle_max_ms, rng=rng)
        for _ in range(1000)
    ]
    assert min(totals) >= 1100
    assert max(totals) <= 2300
    assert len(set(totals)) > 300


def test_v1810_load_migrates_exact_v1809_autoplay_defaults(tmp_path):
    import json
    cfg_path = tmp_path / "app_config.json"
    cfg_path.write_text(json.dumps({
        "autoplay_click_delay_min_ms": 550,
        "autoplay_click_delay_max_ms": 950,
        "autoplay_extra_settle_min_ms": 50,
        "autoplay_extra_settle_max_ms": 500,
        "autoplay_move_duration_min_ms": 180,
        "autoplay_move_duration_max_ms": 320,
        "autoplay_transition_timeout_ms": 4500,
    }))
    cfg = AppConfig.load(cfg_path)
    assert (cfg.autoplay_click_delay_min_ms, cfg.autoplay_click_delay_max_ms) == (900, 1600)
    assert (cfg.autoplay_extra_settle_min_ms, cfg.autoplay_extra_settle_max_ms) == (200, 700)
    assert (cfg.autoplay_move_duration_min_ms, cfg.autoplay_move_duration_max_ms) == (220, 380)
    assert cfg.autoplay_transition_timeout_ms == 6500


def test_v1810_load_preserves_user_custom_autoplay_timing(tmp_path):
    import json
    cfg_path = tmp_path / "app_config.json"
    cfg_path.write_text(json.dumps({
        "autoplay_click_delay_min_ms": 777,
        "autoplay_click_delay_max_ms": 1333,
        "autoplay_extra_settle_min_ms": 222,
        "autoplay_extra_settle_max_ms": 666,
        "autoplay_move_duration_min_ms": 210,
        "autoplay_move_duration_max_ms": 390,
        "autoplay_transition_timeout_ms": 5200,
    }))
    cfg = AppConfig.load(cfg_path)
    assert (cfg.autoplay_click_delay_min_ms, cfg.autoplay_click_delay_max_ms) == (777, 1333)
    assert (cfg.autoplay_extra_settle_min_ms, cfg.autoplay_extra_settle_max_ms) == (222, 666)
    assert (cfg.autoplay_move_duration_min_ms, cfg.autoplay_move_duration_max_ms) == (210, 390)
    assert cfg.autoplay_transition_timeout_ms == 5200
