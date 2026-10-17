"""Verify archived second-campaign scores and render the current LaTeX table."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / '.agent/notes/main2'
METHOD = 'sega_ctm_v2'
REFERENCE = 'sega_official'
METRICS = [
    ('imagereward', 'ImageReward'), ('pickscore', 'PickScore'),
    ('clip', 'CLIP'), ('musiq', 'MUSIQ, full image'),
    ('musiq_crop1024', 'MUSIQ, 1024 crops'),
    ('clipiqa', 'CLIP-IQA, full image'),
    ('clipiqa_crop1024', 'CLIP-IQA, 1024 crops'),
    ('spectrum_log_rmse', 'Spectrum log RMSE'),
    ('spectrum_slope_error', 'Spectrum slope error'),
    ('tiling1024', 'Tiling at 1024'),
]


def read(name):
    with (EVIDENCE / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    records = read('heldout_per_prompt.csv')
    candidates = [r for r in records if r['method'] == METHOD]
    references = [r for r in records if r['method'] == REFERENCE]
    for group in (candidates, references):
        assert len(group) == 187
        assert len({r['prompt_id'] for r in group}) == 187
        assert {int(r['idx']) for r in group} == set(range(8, 195))
        assert all(int(r['seed']) == 0 for r in group)
    reference = {r['prompt_id']: r for r in references}
    assert set(reference) == {r['prompt_id'] for r in candidates}
    paired = {r['metric']: r for r in read('heldout_paired_vs_sega_official.csv')
              if r['method'] == METHOD}
    means = {(r['method'], r['metric']): r for r in read('heldout_means.csv')}
    lines = [r'\begin{table}[ht]', r'\centering\small',
             r'\caption{Held-out seed-0 means and paired differences, candidate minus released SEGA, over 187 prompts. Higher is better except the last three rows.}',
             r'\label{tab:heldout-current}', r'\begin{tabular}{@{}lrrr@{}}',
             r'\toprule', r'Metric & SEGA & SEGA + constancy & Difference [95\% CI]\\',
             r'\midrule']
    max_error = 0.0
    for metric, label in METRICS:
        x = np.array([float(r[metric]) for r in candidates])
        y = np.array([float(reference[r['prompt_id']][metric]) for r in candidates])
        assert np.isfinite(x).all() and np.isfinite(y).all()
        difference = x - y
        draws = np.random.default_rng(0).choice(difference, (2000, 187), replace=True).mean(1)
        low, high = np.percentile(draws, [2.5, 97.5])
        observed = [difference.mean(), low, high, x.mean(), y.mean()]
        expected = [float(paired[metric][k]) for k in ('delta', 'lo', 'hi')]
        expected += [float(means[(m, metric)]['mean']) for m in (METHOD, REFERENCE)]
        error = float(np.max(np.abs(np.array(observed) - expected)))
        assert error < 1e-10, (metric, error)
        max_error = max(max_error, error)
        lines.append(f'{label} & {y.mean():.5f} & {x.mean():.5f} & '
                     f'${difference.mean():+.5f}$ [$ {low:+.5f}, {high:+.5f} $]' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (ROOT / 'paper/sec/current_table.tex').write_text('\n'.join(lines))
    receipt = {'candidate': METHOD, 'pairs': 187, 'metrics_checked': len(METRICS),
               'maximum_absolute_error': max_error, 'bootstrap_draws': 2000,
               'bootstrap_seed': 0, 'fresh_image_scoring': False,
               'source_sha256': {n: hashlib.sha256((EVIDENCE / n).read_bytes()).hexdigest()
                                 for n in ('heldout_per_prompt.csv', 'heldout_means.csv',
                                           'heldout_paired_vs_sega_official.csv')}}
    (ROOT / '.agent/notes/paper_verification_20261017.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
