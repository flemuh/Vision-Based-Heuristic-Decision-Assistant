from __future__ import annotations

import random


def bounded_random_ms(
    minimum_ms: int,
    maximum_ms: int,
    *,
    rng: random.Random | None = None,
) -> int:
    lo = max(0, int(minimum_ms))
    hi = max(lo, int(maximum_ms))
    r = rng or random.SystemRandom()
    return r.randint(lo, hi)


def paced_preclick_delay_ms(
    *,
    campaign_started_at: float,
    now: float,
    completed_games: int,
    placed_count: int,
    target_game_seconds: float,
    expected_transition_ms: int,
    safety_delay_ms: int,
    max_pacing_wait_ms: int,
) -> int:
    target_s = max(1.0, float(target_game_seconds))
    slot_s = target_s / 14.0
    move_index = max(0, int(completed_games)) * 14 + max(0, int(placed_count)) + 1
    target_click_at = (
        float(campaign_started_at)
        + move_index * slot_s
        - max(0, int(expected_transition_ms)) / 1000.0
    )
    pacing_wait = max(0, int(round((target_click_at - float(now)) * 1000.0)))
    pacing_wait = min(pacing_wait, max(0, int(max_pacing_wait_ms)))
    return max(max(0, int(safety_delay_ms)), pacing_wait)
