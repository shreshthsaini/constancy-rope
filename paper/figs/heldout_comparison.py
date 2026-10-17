"""Paired prompt-bootstrap differences from archived campaign records.

Run on a compute node with the crope uv environment. No seeds are pooled.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import orx_figstyle
from orx_figstyle import use_style, PALETTE, save

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / '.agent/notes/main2'
METHODS = [
    ('temp_only_sbase', 'Temperature only', 'black', 'o'),
    ('constancy_tm_sbase', 'Time-matched constancy', 'orange', 's'),
    ('sega_ctm_v2', 'SEGA + constancy', 'red', 'D'),
    ('dype', 'DyPE', 'blue', '^'),
]
METRICS = [('imagereward', 'ImageReward difference'), ('pickscore', 'PickScore difference'),
           ('musiq', 'Full-image MUSIQ difference'), ('musiq_crop1024', 'Crop MUSIQ difference')]


def rows(name):
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))


def main():
    records = rows('heldout_per_prompt.csv')
    reference = {r['prompt_id']: r for r in records if r['method'] == 'sega_official'}
    paired = {(r['method'], r['metric']): r for r in rows('heldout_paired_vs_sega_official.csv')}
    use_style()
    # This article uses letter paper with 1-inch margins: actual text width 6.5in.
    orx_figstyle.TEXT = 6.5
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.55), sharey=True)
    fig.subplots_adjust(left=.255, right=.98, bottom=.14, top=.94, wspace=.23, hspace=.62)
    for panel, (ax, (metric, label)) in enumerate(zip(axes.flat, METRICS)):
        ax.axvline(0, color='.5', linewidth=.8, linestyle='--', zorder=0)
        for position, (method, _, color, marker) in enumerate(METHODS):
            candidate = [r for r in records if r['method'] == method]
            assert len(candidate) == len(reference) == 187
            assert {int(r['idx']) for r in candidate} == set(range(8, 195))
            assert all(int(r['seed']) == 0 for r in candidate)
            differences = np.array([float(r[metric])-float(reference[r['prompt_id']][metric]) for r in candidate])
            mean = differences.mean()
            draws = np.random.default_rng(0).choice(differences, (2000, 187), replace=True).mean(1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            saved = paired[method, metric]
            np.testing.assert_allclose([mean, lo, hi], [float(saved[k]) for k in ('delta', 'lo', 'hi')],
                                       atol=1e-10, rtol=0)
            ax.errorbar(mean, 3-position, xerr=[[mean-lo], [hi-mean]], fmt=marker,
                        color=PALETTE[color], capsize=2, markersize=4, elinewidth=1)
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylim(-.55, 3.55)
        ax.set_yticks([3, 2, 1, 0], [m[1] for m in METHODS], fontsize=8)
        if panel % 2 == 0:
            ax.set_ylabel('Variant', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        ax.text(0, 1.05, f'({chr(97+panel)})', transform=ax.transAxes, fontsize=8, weight='bold')
        ax.spines[['top', 'right']].set_visible(False)
    stem = Path(__file__).with_suffix('')
    save(fig, str(stem), formats=('pdf', 'svg'), close=False)
    fig.savefig(stem.with_suffix('.png'), dpi=180)
    plt.close(fig)
    print('Verified and plotted 16 paired effects, n=187, seed0, 2000 bootstrap draws.')


if __name__ == '__main__':
    main()
