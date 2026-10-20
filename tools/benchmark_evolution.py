"""Optimization evolution benchmark: pure Python -> Numba -> Numba + policy-key cache.

Runs the same board, seed and rollout budget through three engine configurations,
checks that all three recommend the same moves with the same utilities, and writes
CSV / JSON / PNG results.

    python tools/benchmark_evolution.py --rollouts 120 1000 10000 --repeats 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_engine.advisor import Advisor
from vision_engine.board import Board
from vision_engine.route_math import multivariate_at_least_probability
from vision_engine.scoring import score_mask
from vision_engine.solver.r39_plan_kernel import board_bitmasks, compiled_features, kernel

BOARD = ('CH', 'HA', 'SO', 'CH', 'HA', 'SO', 'BL', 'BL', 'CR', 'LI',
         'BL', 'CH', None, 'CR', 'HA', 'BL', 'SO', 'CH', 'LI', 'SO',
         'LI', 'CR', 'CR', 'LI', 'HA')
MASK = sum(1 << i for i in (0, 6, 5, 9, 16, 13))
CURRENT_JEWEL = 'BL'

# (key, label, compiled route features, policy-key cache)
VARIANTS = (
    ('python', 'Pure Python route features', False, False),
    ('numba', 'Numba-compiled route features', True, False),
    ('numba_cache', 'Numba + context-aware policy-key cache', True, True),
)
COLORS = {'python': '#94a3b8', 'numba': '#3b82f6', 'numba_cache': '#10b981'}


def warm_numba() -> float:
    t0 = time.perf_counter()
    compiled_features(0, board_bitmasks(Board(BOARD)), kernel())
    return time.perf_counter() - t0


def solve(rollouts: int, compiled: bool, cache: bool, seed: int = 1337):
    score_mask.cache_clear()
    multivariate_at_least_probability.cache_clear()
    advisor = Advisor(Board(BOARD), mode='auto', exact_horizon=6, rollout_exact_horizon=3,
                      rollouts=rollouts, rng_seed=seed, compiled_plan_features=compiled,
                      policy_key_cache_enabled=cache, lazy_slack_enabled=True)
    t0 = time.perf_counter()
    recs = advisor.recommend(MASK, CURRENT_JEWEL)
    return time.perf_counter() - t0, recs


def same_decision(a, b) -> bool:
    if [r.position for r in a] != [r.position for r in b]:
        return False
    return all(abs(x - y) <= 1e-9 for ra, rb in zip(a, b)
               for x, y in zip(ra.utility.as_tuple(), rb.utility.as_tuple()))


def plot(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = summary['results']
    budgets = [str(r['rollouts']) for r in rows]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={'width_ratios': [3, 2]})
    width = 0.26
    for k, (key, label, _, _) in enumerate(VARIANTS):
        xs = [i + (k - 1) * width for i in range(len(rows))]
        vals = [r['median_s'][key] for r in rows]
        ax.bar(xs, vals, width=width, label=label, color=COLORS[key])
        if key != 'python':
            for x, r in zip(xs, rows):
                v = r['median_s'][key]
                ax.text(x, v, f"{r['speedup_vs_python'][key]:.2f}×", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(range(len(rows)), budgets)
    ax.set_xlabel('Monte Carlo rollouts per decision')
    ax.set_ylabel('Median time per decision (s)')
    ax.set_title('Time per decision (lower is better)\nlabels = speedup vs pure Python')
    ax.legend(fontsize=9)

    last = rows[-1]
    steps = [last['median_s'][k] for k, *_ in VARIANTS]
    names = ['Pure Python', '+ Numba\nroute features', '+ policy-key\ncache']
    bars = ax2.bar(range(3), steps, color=[COLORS[k] for k, *_ in VARIANTS])
    for i, (b, v) in enumerate(zip(bars, steps)):
        note = f'{v:.2f}s' if i == 0 else f'{v:.2f}s\n−{100 * (1 - v / steps[i - 1]):.0f}% vs previous'
        ax2.text(b.get_x() + b.get_width() / 2, v, note, ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(range(3), names)
    ax2.set_ylim(0, max(steps) * 1.25)
    ax2.set_title(f"What each optimization added\n({last['rollouts']} rollouts, total −{last['reduction_pct_vs_python']['numba_cache']:.0f}%)")
    fig.suptitle(f"Same board, same seed, identical decisions · {summary['machine_short']} · {summary['repeats']} repeats",
                 fontsize=10, color='#475569')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(rollouts: list[int], repeats: int, output: Path) -> dict:
    warmup = warm_numba()
    rows = []
    for budget in rollouts:
        for it in range(repeats):
            order = VARIANTS if it % 2 == 0 else tuple(reversed(VARIANTS))  # alternate order to reduce drift bias
            results = {}
            for key, _, compiled, cache in order:
                results[key] = solve(budget, compiled, cache)
            ref = results['python'][1]
            for key in results:
                if not same_decision(ref, results[key][1]):
                    raise AssertionError(f'Decision mismatch: {key} vs python at rollouts={budget} repeat={it}')
            for key, *_ in VARIANTS:
                rows.append({'rollouts': budget, 'iteration': it, 'variant': key,
                             'seconds': results[key][0], 'best_position': results[key][1][0].position})
            print(f"rollouts={budget} repeat={it + 1}: " +
                  " | ".join(f"{k} {results[k][0]:.3f}s" for k, *_ in VARIANTS), flush=True)

    grouped = []
    for budget in rollouts:
        med = {k: statistics.median(r['seconds'] for r in rows if r['rollouts'] == budget and r['variant'] == k)
               for k, *_ in VARIANTS}
        grouped.append({
            'rollouts': budget,
            'median_s': med,
            'speedup_vs_python': {k: med['python'] / med[k] for k in med},
            'reduction_pct_vs_python': {k: 100 * (1 - med[k] / med['python']) for k in med},
            'cache_gain_vs_numba_pct': 100 * (1 - med['numba_cache'] / med['numba']),
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'benchmark_evolution.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import numba  # noqa: F401  (version recorded for reproducibility)
    summary = {
        'machine': platform.platform(), 'machine_short': f"{platform.system()} · {os.cpu_count()} CPUs · Python {platform.python_version()}",
        'processor': platform.processor(), 'cpu_count': os.cpu_count(), 'python': platform.python_version(),
        'numba': getattr(numba, '__version__', '?'), 'jit_warmup_s': warmup, 'repeats': repeats,
        'variants': {k: label for k, label, *_ in VARIANTS}, 'all_decisions_identical': True, 'results': grouped,
    }
    (output / 'benchmark_evolution.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    plot(summary, output / 'benchmark_evolution.png')
    print(f"\nSaved to {output}")
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollouts', type=int, nargs='+', default=[120, 1000, 10000])
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--output', type=Path, default=Path('benchmark-results/evolution'))
    a = p.parse_args()
    run(a.rollouts, a.repeats, a.output)
