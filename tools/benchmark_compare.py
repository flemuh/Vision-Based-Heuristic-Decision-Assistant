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

BOARD = ('CH','HA','SO','CH','HA','SO','BL','BL','CR','LI',
         'BL','CH',None,'CR','HA','BL','SO','CH','LI','SO',
         'LI','CR','CR','LI','HA')
MASK = sum(1 << i for i in (0, 6, 5, 9, 16, 13))
CURRENT_JEWEL = 'BL'


def warm_numba() -> float:
    t0 = time.perf_counter()
    fn = kernel()
    compiled_features(0, board_bitmasks(Board(BOARD)), fn)
    return time.perf_counter() - t0


def timed_solve(*, rollouts: int, optimized: bool, seed: int = 1337):
    score_mask.cache_clear()
    multivariate_at_least_probability.cache_clear()
    advisor = Advisor(
        Board(BOARD),
        mode='auto',
        exact_horizon=6,
        rollout_exact_horizon=3,
        rollouts=rollouts,
        rng_seed=seed,
        compiled_plan_features=True,
        policy_key_cache_enabled=optimized,
        lazy_slack_enabled=True,
    )
    t0 = time.perf_counter()
    recommendations = advisor.recommend(MASK, CURRENT_JEWEL)
    duration = time.perf_counter() - t0
    return duration, recommendations, advisor


def architecture_image(path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def box(x, y, w, h, title, lines):
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6,rounding_size=2', linewidth=1.5, facecolor='#f5f7fb', edgecolor='#334155')
        ax.add_patch(patch)
        ax.text(x + w/2, y + h - 2.5, title, ha='center', va='top', fontsize=12, fontweight='bold')
        ax.text(x + 3, y + h - 6, '\n'.join(lines), ha='left', va='top', fontsize=8.5, linespacing=1.15)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12, linewidth=1.2, color='#475569'))

    ax.text(25, 95, 'Baseline architecture', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(75, 95, 'Optimized architecture', ha='center', va='center', fontsize=14, fontweight='bold')

    box(5, 72, 40, 15, 'State input', ['board mask', 'current jewel', 'mode / rollout budget'])
    box(5, 50, 40, 16, 'Planner path', ['compiled route features', 'Expectimax recursion', 'repeated policy-key evaluation'])
    box(5, 27, 40, 16, 'Per-node work', ['policy key rebuilt every visit', 'Lucky slack often computed eagerly', 'fewer memoized comparisons'])
    box(5, 5, 40, 14, 'Output', ['recommendation', 'probabilities', 'expected score'])
    arrow(25, 72, 25, 66)
    arrow(25, 50, 25, 43)
    arrow(25, 27, 25, 19)

    box(55, 72, 40, 15, 'State input', ['board mask', 'current jewel', 'mode / rollout budget'])
    box(55, 50, 40, 16, 'Planner path', ['compiled route features', 'Expectimax recursion', 'context-aware policy-key cache'])
    box(55, 27, 40, 16, 'Per-node work', ['memoized ranking tuples', 'Lucky slack computed only when needed', 'fewer repeated comparisons'])
    box(55, 5, 40, 14, 'Output', ['recommendation', 'probabilities', 'expected score'])
    arrow(75, 72, 75, 66)
    arrow(75, 50, 75, 43)
    arrow(75, 27, 75, 19)

    ax.text(50, -3, 'Same decisions (parity-checked) · less repeated work per node', ha='center', va='center', fontsize=11, color='#0f766e', fontweight='bold')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def plot_results(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = summary['results']
    labels = [str(r['rollouts']) for r in rows]
    x = list(range(len(labels)))
    baseline = [r['baseline_median_s'] for r in rows]
    optimized = [r['optimized_median_s'] for r in rows]
    speedups = [r['speedup'] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([i - 0.18 for i in x], baseline, width=0.36, label='Baseline', alpha=0.9)
    ax.bar([i + 0.18 for i in x], optimized, width=0.36, label='Optimized', alpha=0.9)
    ax.set_xticks(x, labels)
    ax.set_xlabel('Requested simulations / rollouts')
    ax.set_ylabel('Median wall time (seconds)')
    ax.set_title('Heuristic simulation benchmark — baseline vs optimized architecture')
    ax.legend()
    for i, sp in enumerate(speedups):
        top = max(baseline[i], optimized[i])
        ax.text(i, top + max(top * 0.03, 0.02), f'{sp:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(*, rollouts: list[int], repeats: int, output: Path) -> dict:
    warmup_s = warm_numba()
    rows: list[dict] = []
    for budget in rollouts:
        for iteration in range(repeats):
            order = (False, True) if iteration % 2 == 0 else (True, False)
            results = {}
            for optimized in order:
                sec, recs, adv = timed_solve(rollouts=budget, optimized=optimized)
                results[optimized] = (sec, recs, adv)
            a = results[False][1]
            b = results[True][1]
            if [r.position for r in a] != [r.position for r in b]:
                raise AssertionError(f'Position mismatch at rollouts={budget} repeat={iteration}')
            for ra, rb in zip(a, b):
                if any(abs(x - y) > 1e-9 for x, y in zip(ra.utility.as_tuple(), rb.utility.as_tuple())):
                    raise AssertionError(f'Utility mismatch at rollouts={budget} repeat={iteration}')
            for optimized in (False, True):
                sec, recs, adv = results[optimized]
                rows.append({
                    'rollouts': int(budget),
                    'iteration': iteration,
                    'variant': 'optimized' if optimized else 'baseline',
                    'seconds': sec,
                    'best_position': recs[0].position,
                    'sample_count': sum(r.samples for r in recs),
                    'plan_cache_entries': len(adv._plan_cache),
                    'policy_key_cache_entries': len(getattr(adv, '_policy_key_cache', {})),
                    'exact_cache_entries': len(adv._cache),
                    'compiled_plan': True,
                })
            print(f'rollouts={budget} repeat={iteration+1}: baseline {results[False][0]:.3f}s vs optimized {results[True][0]:.3f}s', flush=True)

    grouped = []
    for budget in rollouts:
        base = [r['seconds'] for r in rows if r['rollouts'] == budget and r['variant'] == 'baseline']
        opt = [r['seconds'] for r in rows if r['rollouts'] == budget and r['variant'] == 'optimized']
        grouped.append({
            'rollouts': int(budget),
            'baseline_median_s': statistics.median(base),
            'optimized_median_s': statistics.median(opt),
            'absolute_gain_s': statistics.median(base) - statistics.median(opt),
            'speedup': statistics.median(base) / statistics.median(opt),
            'reduction_pct': 100.0 * (statistics.median(base) - statistics.median(opt)) / statistics.median(base),
        })

    output.mkdir(parents=True, exist_ok=True)
    with (output / 'benchmark_compare.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'machine': platform.platform(),
        'cpu_count': os.cpu_count(),
        'python': platform.python_version(),
        'jit_warmup_s': warmup_s,
        'board_seed': 'reference-board-r39',
        'current_jewel': CURRENT_JEWEL,
        'repeats': repeats,
        'baseline': 'compiled route features without Expectimax policy-key cache',
        'optimized': 'compiled route features with context-aware policy-key cache and reduced repeated work',
        'all_decisions_identical': True,
        'results': grouped,
    }
    (output / 'benchmark_compare.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    plot_results(summary, output / 'benchmark_compare.png')
    architecture_image(output / 'architecture_compare.png')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, nargs='+', default=[16, 60, 120, 10000])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('benchmark-results/compare'))
    args = parser.parse_args()
    run(rollouts=args.rollouts, repeats=args.repeats, output=args.output)
