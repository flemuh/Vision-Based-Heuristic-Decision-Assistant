from vision_engine.config import AppConfig
from vision_engine.autoplay import bounded_random_ms
import random


def test_v1809_extra_settle_feature_remains_bounded():
    cfg = AppConfig()
    assert 0 <= cfg.autoplay_click_delay_min_ms <= cfg.autoplay_click_delay_max_ms
    assert 0 <= cfg.autoplay_extra_settle_min_ms <= cfg.autoplay_extra_settle_max_ms
    assert cfg.autoplay_click_delay_min_ms + cfg.autoplay_extra_settle_min_ms > 0


def test_v1809_total_preclick_delay_varies_with_current_bounds():
    cfg = AppConfig()
    rng = random.Random(1809)
    totals = [
        bounded_random_ms(cfg.autoplay_click_delay_min_ms, cfg.autoplay_click_delay_max_ms, rng=rng)
        + bounded_random_ms(cfg.autoplay_extra_settle_min_ms, cfg.autoplay_extra_settle_max_ms, rng=rng)
        for _ in range(500)
    ]
    assert min(totals) >= cfg.autoplay_click_delay_min_ms + cfg.autoplay_extra_settle_min_ms
    assert max(totals) <= cfg.autoplay_click_delay_max_ms + cfg.autoplay_extra_settle_max_ms
    assert len(set(totals)) > 100
