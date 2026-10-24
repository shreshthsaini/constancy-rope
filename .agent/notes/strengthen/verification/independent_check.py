"""Independent recomputation of the strengthening-campaign contrasts from the raw evaluator CSVs.

Written separately from scripts/strengthen_analysis.py (it imports nothing from the project). It reads the
per-image metric CSVs and ImageReward CSVs on scratch, forms paired per-prompt contrasts in manifest order,
and bootstraps with numpy default_rng(0).integers (2,000 draws, percentile 2.5/97.5, linear), as written in
.agent/notes/prereg_strengthen.md. Output: JSON with every recomputed point and interval.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

S = Path('/scratch/09032/saini_2/constancy_rope')
E = S / 'eval_krea'
W = Path('/work/09032/saini_2/vista/Projects/Exploration/constancy_rope')
TEN = ['imagereward', 'pickscore', 'clip', 'musiq', 'musiq_crop1024', 'clipiqa', 'clipiqa_crop1024',
       'spectrum_log_rmse', 'spectrum_slope_error', 'tiling1024']


def manifest(path):
    return [json.loads(l)['id'] for l in Path(path).read_text().splitlines() if l.strip()]


def seed_of(path):
    return int(re.search(r'_seed(\d+)\.png$', str(path)).group(1))


def table(per_image, ir, method, alias=None, seed=None):
    """Return DataFrame indexed by prompt_id with TEN metric columns for one method/seed."""
    p = pd.read_csv(per_image)
    p = p[p.method == method].copy()
    p['seed_'] = p['path'].map(seed_of)
    if 'seed' in p:
        assert (p['seed'].astype(int) == p['seed_']).all(), 'seed column disagrees with path'
    if seed is not None:
        p = p[p.seed_ == seed]
    q = pd.read_csv(ir)
    q = q[q.method == method].copy()
    q['seed_'] = q['png'].map(seed_of)
    if seed is not None:
        q = q[q.seed_ == seed]
    assert not p.prompt_id.duplicated().any() and not q.prompt_id.duplicated().any(), 'duplicate prompt'
    p = p.set_index('prompt_id')
    q = q.set_index('prompt_id')
    out = p[[m for m in TEN if m != 'imagereward']].copy()
    out['imagereward'] = q['imagereward'].reindex(out.index)
    # path identity: the IR row must point at the same PNG as the metric row
    same = (q['png'].reindex(out.index).astype(str) == p['path'].astype(str))
    assert same.all(), f'IR/metric path mismatch for {method}'
    return out[TEN]


def boot(x):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 1 and len(x) and np.isfinite(x).all()
    idx = np.random.default_rng(0).integers(0, len(x), size=(2000, len(x)))
    d = x[idx].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5], method='linear')
    return float(x.mean()), float(lo), float(hi)


def contrast(arms, weights, ids, metric):
    vals = sum(w * arms[a].loc[ids, metric].to_numpy(dtype=np.float64) for a, w in weights.items())
    return boot(vals)


def tost(x, margin):
    x = np.asarray(x, dtype=np.float64)
    e = x[np.random.default_rng(0).integers(0, len(x), size=(2000, len(x)))].mean(1) - x.mean()
    pg = (1 + np.count_nonzero(e >= x.mean() + margin)) / 2001     # H0 delta <= -margin
    pl = (1 + np.count_nonzero(e <= x.mean() - margin)) / 2001     # H0 delta >= +margin
    return max(pg, pl)


def sup(x):
    x = np.asarray(x, dtype=np.float64)
    e = x[np.random.default_rng(0).integers(0, len(x), size=(2000, len(x)))].mean(1) - x.mean()
    return (1 + np.count_nonzero(e >= x.mean())) / 2001


res = {}
A4K = manifest(S / 'prompts/aesthetic4k_eval.jsonl')
HELD = A4K[8:195]
# ---------------- P1 factorial
arms = {'ntk2': table(E / 'main/per_image.csv', E / 'ir/main.csv', 'ntk2', seed=0),
        'sega_official': table(E / 'main/per_image.csv', E / 'ir/main.csv', 'sega_official', seed=0),
        'temp_only_sbase': table(E / 'mainsb/per_image.csv', E / 'ir/mainsb.csv', 'temp_only', seed=0),
        'sega_base_none': table(E / 'strengthen/factorial/per_image.csv', E / 'strengthen/factorial/ir.csv', 'sega_base_none', seed=0),
        'ntk2_sref': table(E / 'strengthen/factorial/per_image.csv', E / 'strengthen/factorial/ir.csv', 'ntk2_sref', seed=0)}
for a, t in arms.items():
    assert set(HELD) <= set(t.index), f'{a} misses held-out prompts'
FACT = {'base_main_effect': {'sega_base_none': .5, 'temp_only_sbase': .5, 'ntk2': -.5, 'ntk2_sref': -.5},
        'temperature_main_effect': {'ntk2_sref': .5, 'temp_only_sbase': .5, 'ntk2': -.5, 'sega_base_none': -.5},
        'interaction': {'temp_only_sbase': 1, 'sega_base_none': -1, 'ntk2_sref': -1, 'ntk2': 1},
        'allocation': {'sega_official': 1, 'temp_only_sbase': -1},
        'base_at_none': {'sega_base_none': 1, 'ntk2': -1}, 'base_at_sref': {'temp_only_sbase': 1, 'ntk2_sref': -1},
        'temperature_at_ntk2': {'ntk2_sref': 1, 'ntk2': -1}, 'temperature_at_sega': {'temp_only_sbase': 1, 'sega_base_none': -1}}
res['P1'] = {f'{c}|{m}': contrast(arms, w, HELD, m) for c, w in FACT.items() for m in TEN}
res['P1_means'] = {f'{a}|{m}': float(arms[a].loc[HELD, m].mean()) for a in arms for m in TEN}
res['P1_n'] = len(HELD)
# ---------------- P3 sweep (rows 8-62)
SW = A4K[8:63]
sw = {'tau1p0': arms['sega_base_none'], 'tau_sref': arms['temp_only_sbase']}
for t in ['tau1p1', 'tau_logn', 'tau1p45']:
    sw[t] = table(E / 'strengthen/tsweep/per_image.csv', E / 'strengthen/tsweep/ir.csv', t, seed=0)
res['P3'] = {f'{t}|{m}': contrast(sw, {t: 1, 'tau_sref': -1}, SW, m) for t in sw if t != 'tau_sref' for m in TEN}
res['P3_means'] = {f'{t}|{m}': float(sw[t].loc[SW, m].mean()) for t in sw for m in TEN}
# ---------------- E1 fresh
FR = manifest(W / 'prompts/fresh150.jsonl')
fr = {a: table(E / 'strengthen/fresh/per_image.csv', E / 'strengthen/fresh/ir.csv', a, seed=0)
      for a in ['sega_official', 'temp_only_sbase', 'dype', 'ntk2', 'sega_base_none', 'ntk2_sref']}
res['E1'] = {f'{a}|{m}': contrast(fr, {a: 1, 'sega_official': -1}, FR, m) for a in fr if a != 'sega_official' for m in TEN}
res['E1_n'] = len(FR)
# registered family: pooled H1 from the raw old CSVs (independent of heldout_per_prompt.csv), fresh H2
IRM, PM = 0.08197713636363636, 0.10
fam = {}
for m, margin, key in [('imagereward', IRM, 'H1_IR'), ('pickscore', PM, 'H1_Pick')]:
    old = (arms['temp_only_sbase'].loc[HELD, m] - arms['sega_official'].loc[HELD, m]).to_numpy(np.float64)
    new = (fr['temp_only_sbase'].loc[FR, m] - fr['sega_official'].loc[FR, m]).to_numpy(np.float64)
    x = np.concatenate([old, new])
    fam[key] = (*boot(x), tost(x, margin), len(x))
x = (fr['sega_official'].loc[FR, 'musiq'] - fr['temp_only_sbase'].loc[FR, 'musiq']).to_numpy(np.float64)
fam['H2_MUSIQ'] = (*boot(x), sup(x), len(x))
ps = np.array([fam[k][3] for k in ['H1_IR', 'H1_Pick', 'H2_MUSIQ']])
o = np.argsort(ps, kind='stable')
adj = np.empty(3)
adj[o] = np.minimum(1, np.maximum.accumulate((3 - np.arange(3)) * ps[o]))
res['family'] = {k: {'delta': v[0], 'lo': v[1], 'hi': v[2], 'p': v[3], 'n': v[4], 'p_holm': float(a)}
                 for (k, v), a in zip(fam.items(), adj)}
# ---------------- E2 seeds
s1_old = {a: table(E / 'seed1/per_image.csv', E / 'ir/seed1.csv', raw, seed=1) for a, raw in
          [('sega_official', 'sega_official'), ('temp_only_sbase', 'temp_only')]}
s1_new = {a: table(E / 'strengthen/seeds_seed1/per_image.csv', E / 'strengthen/seeds_seed1/ir.csv', a, seed=1)
          for a in ['sega_official', 'temp_only_sbase']}
s2 = {a: table(E / 'strengthen/seeds_seed2/per_image.csv', E / 'strengthen/seeds_seed2/ir.csv', a, seed=2)
      for a in ['sega_official', 'temp_only_sbase']}
seeds = {}
for a in ['sega_official', 'temp_only_sbase']:
    reused = s1_old[a].loc[[p for p in HELD if p in s1_old[a].index and p not in s1_new[a].index]]
    one = pd.concat([reused, s1_new[a]])
    assert not one.index.duplicated().any() and set(HELD) <= set(one.index), f'seed1 coverage {a}'
    seeds[a] = {0: arms[a].loc[HELD], 1: one.loc[HELD], 2: s2[a].loc[HELD]}
res['E2_seed1_reused'] = int(len([p for p in HELD if p in s1_old['sega_official'].index and p not in s1_new['sega_official'].index]))
for lab, ks in [('seeds012', [0, 1, 2]), ('seeds12', [1, 2]), ('seed0', [0]), ('seed1', [1]), ('seed2', [2])]:
    pm = {a: sum(seeds[a][k] for k in ks) / len(ks) for a in seeds}
    res[f'E2_{lab}'] = {m: contrast(pm, {'temp_only_sbase': 1, 'sega_official': -1}, HELD, m) for m in TEN}
json.dump(res, open(sys.argv[1], 'w'), indent=1)
print('wrote', sys.argv[1], 'keys', list(res))
