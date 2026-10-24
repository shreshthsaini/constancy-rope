#!/usr/bin/env python3
"""Collect strengthening analysis outputs into flat CSVs and a readable RESULTS.md.

Reads only finished analysis directories written by strengthen_analysis.py
(.agent/notes/strengthen/analysis/{P1,P3,E1,E2,E3}) and strengthen_diagnostic_report.py
(.agent/notes/strengthen/analysis/{ckpt_matrix,default_shift}). It computes nothing new:
every number is copied from those CSV/JSON files. A block whose analysis is missing or
incomplete is listed as such. Verdict convention: tie when the nominal 95% CI contains 0,
otherwise better or worse in the metric's own direction; Holm marks come from p_holm <= 0.05
inside the registered family.

Usage: strengthen_report.py [--analysis DIR] [--export DIR] [--results FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / '.agent/notes/strengthen/analysis'
EXPORT = Path('/scratch/09032/saini_2/constancy_rope/eval_krea/strengthen')
RESULTS = ROOT / '.agent/notes/strengthen/RESULTS.md'
METRICS = ['imagereward', 'pickscore', 'clip', 'musiq', 'musiq_crop1024', 'clipiqa',
           'clipiqa_crop1024', 'spectrum_log_rmse', 'spectrum_slope_error', 'tiling1024']
LOWER = {'spectrum_log_rmse', 'spectrum_slope_error', 'tiling1024'}
SHORT = {'imagereward': 'IR', 'pickscore': 'Pick', 'clip': 'CLIP', 'musiq': 'MUSIQ',
         'musiq_crop1024': 'MUSIQ crop', 'clipiqa': 'CLIP-IQA', 'clipiqa_crop1024': 'CLIP-IQA crop',
         'spectrum_log_rmse': 'Spec. RMSE', 'spectrum_slope_error': 'Slope err.', 'tiling1024': 'Tiling'}
DIGITS = {'imagereward': 3, 'pickscore': 3, 'clip': 3, 'musiq': 2, 'musiq_crop1024': 2, 'clipiqa': 4,
          'clipiqa_crop1024': 4, 'spectrum_log_rmse': 4, 'spectrum_slope_error': 4, 'tiling1024': 4}
FILES = {'P1': ['means', 'paired', 'contrasts_holm'], 'P3': ['means', 'paired', 'contrasts_holm'],
         'E1': ['means', 'paired', 'contrasts_holm', 'confirmation_family'],
         'E2': ['means', 'paired', 'contrasts_holm'], 'E3': ['means', 'paired', 'contrasts_holm'],
         'ckpt_matrix': ['means', 'paired', 'lattice_summary', 'lattice_per_prompt'],
         'default_shift': ['means', 'paired']}
EXPORT_NAME = {'P1': 'factorial', 'P3': 'tsweep', 'E1': 'fresh', 'E2': 'seeds', 'E3': 'qwen',
               'ckpt_matrix': 'ckpt_matrix', 'default_shift': 'default_shift'}


def verdict(row, metric=None):
    metric = metric or row['metric']
    lo, hi, d = row.get('lo'), row.get('hi'), row.get('delta')
    if lo is None or pd.isna(lo):
        return 'missing'
    if lo <= 0 <= hi:
        return 'tie'
    return 'better' if ((d < 0) if metric in LOWER else (d > 0)) else 'worse'


def num(x, digits, signed=True):
    if x is None or pd.isna(x):
        return 'n/a'
    return f'{x:+.{digits}f}' if signed else f'{x:.{digits}f}'


def ci(row, metric, signed=True):
    k = DIGITS.get(metric, 3)
    return f"{num(row['delta'] if signed else row['mean'], k, signed)} [{num(row['lo'], k, signed)}, {num(row['hi'], k, signed)}]"


def load(block, root):
    d = root / block
    out = {}
    if not d.is_dir():
        return None
    for name in FILES[block]:
        p = d / f'{name}.csv'
        if p.exists() and p.stat().st_size:
            out[name] = pd.read_csv(p)
    for name in ('coverage', 'hypotheses', 'confirmation_family', 'analysis_protocol', 'report'):
        p = d / f'{name}.json'
        if p.exists():
            out[name + '_json'] = json.loads(p.read_text())
    return out


def complete(data):
    if data is None:
        return False
    if 'analysis_protocol_json' in data:
        return bool(data['analysis_protocol_json'].get('complete'))
    if 'report_json' in data:
        return bool(data['report_json'].get('complete'))
    return False


def contrast_table(frame, contrasts, label='seed0', metrics=METRICS):
    rows = ['| Contrast | ' + ' | '.join(SHORT[m] for m in metrics) + ' |',
            '| --- | ' + ' | '.join('---' for _ in metrics) + ' |']
    sub = frame[frame.analysis == label]
    for c in contrasts:
        cells = []
        for m in metrics:
            r = sub[(sub.contrast == c) & (sub.metric == m)]
            if r.empty:
                cells.append('n/a')
                continue
            r = r.iloc[0].to_dict()
            mark = ' H' if r.get('holm_reject') else ''
            cells.append(f"{ci(r, m)} {verdict(r)}{mark}")
        rows.append(f'| {c} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def means_table(frame, methods, label='seed0', metrics=METRICS):
    rows = ['| Method | ' + ' | '.join(SHORT[m] for m in metrics) + ' |',
            '| --- | ' + ' | '.join('---' for _ in metrics) + ' |']
    sub = frame[frame.analysis == label] if 'analysis' in frame else frame
    for method in methods:
        cells = []
        for m in metrics:
            r = sub[(sub.method == method) & (sub.metric == m)]
            cells.append('n/a' if r.empty else f"{num(r.iloc[0]['mean'], DIGITS[m], False)} (n={int(r.iloc[0]['n'])})")
        rows.append(f'| {method} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def hypothesis_lines(family):
    lines = []
    for r in family:
        k = 3 if r['hypothesis'] != 'H2_MUSIQ' else 2
        margin = '' if r.get('margin') is None else f", margin +/-{r['margin']:.4f}"
        lines.append(f"- {r['hypothesis']} (n={r['n']}{margin}): delta {num(r.get('delta'), k)} "
                     f"[{num(r.get('lo'), k)}, {num(r.get('hi'), k)}], p {r['p']:.4g}, Holm p {r['p_holm']:.4g}: "
                     f"**{r['status']}**")
    return lines


def section_p1(data):
    out = ['## (a) Factorial base x temperature plus allocation (P1, held-out rows 8-194, seed 0)']
    if data is None:
        return out + ['Not analysed yet.']
    out.append(f"Complete block: {complete(data)}. Holm family 80 (8 contrasts x 10 metrics), two-sided "
               "absolute-tail bootstrap p. Cells: delta [95% CI] verdict; H = survives Holm.")
    order = ['base_main_effect', 'temperature_main_effect', 'interaction', 'allocation',
             'base_at_none', 'base_at_sref', 'temperature_at_ntk2', 'temperature_at_sega']
    if 'contrasts_holm' in data:
        out.append(contrast_table(data['contrasts_holm'], order))
    if 'means' in data:
        out += ['', 'Arm means (95% CI omitted here; see CSV):',
                means_table(data['means'], ['ntk2', 'ntk2_sref', 'sega_base_none', 'temp_only_sbase', 'sega_official'])]
    return out


def section_p3(data):
    out = ['## (c) Temperature sweep (P3, SEGA base, temp_only, held-out rows 8-62, seed 0)']
    if data is None:
        return out + ['Not analysed yet.']
    out.append(f"Complete block: {complete(data)}. Holm family 40 (4 settings minus sref x 10 metrics).")
    taus = ['tau1p0', 'tau1p1', 'tau_sref', 'tau_logn', 'tau1p45']
    if 'means' in data:
        out.append(means_table(data['means'], taus))
    if 'contrasts_holm' in data:
        out += ['', 'Each setting minus sref (multiplier 4**0.16 = 1.2483):',
                contrast_table(data['contrasts_holm'], [t for t in taus if t != 'tau_sref'])]
    return out


def section_e1(data):
    out = ['## (d) Fresh150 confirmation (E1) and amended H1/H2']
    if data is None:
        return out + ['Not analysed yet.']
    out.append(f"Complete block: {complete(data)}.")
    conf = data.get('confirmation_family_json', {})
    if conf.get('family'):
        out.append(f"Amended confirmatory family (Holm 3): H1 pooled historical 187 + fresh 150; H2 fresh 150. "
                   f"Overall H1 **{conf.get('H1')}**, H2 **{conf.get('H2')}**, H3 {conf.get('H3')}.")
        out += hypothesis_lines(conf['family'])
    hyp = data.get('hypotheses_json', {}).get('seed0')
    if hyp:
        out.append(f"Fresh-only descriptive summary (not a substitute for the pooled test): H1 {hyp['H1']}, H2 {hyp['H2']}.")
        out += hypothesis_lines(hyp['family'])
    if 'contrasts_holm' in data:
        out += ['', 'Secondary: each arm minus released SEGA, Holm family 50:',
                contrast_table(data['contrasts_holm'], ['temp_only_sbase', 'dype', 'ntk2', 'sega_base_none', 'ntk2_sref'])]
    return out


def section_e2(data):
    out = ['## (e) Seeds 0/1/2 (E2, held-out rows 8-194)']
    if data is None:
        return out + ['Not analysed yet.']
    out.append(f"Complete block: {complete(data)}. Primary: prompt-level mean over seeds 0,1,2, then paired "
               "bootstrap (n=187 prompts). Holm family 10 (temp_only_sbase minus SEGA x 10 metrics).")
    if 'contrasts_holm' in data:
        for label in ['seeds012', 'seeds12', 'seed0', 'seed1', 'seed2']:
            out += ['', f'{label}:', contrast_table(data['contrasts_holm'], ['temp_only_sbase'], label)]
    hyp = data.get('hypotheses_json', {})
    for label in ['seeds012', 'seeds12']:
        if label in hyp:
            out.append(f"Replication summary {label}: H1 {hyp[label]['H1']}, H2 {hyp[label]['H2']}.")
            out += hypothesis_lines(hyp[label]['family'])
    return out


def section_e3(data):
    out = ['## (f) Qwen-Image transfer (E3, fresh150, seed 0, exploratory)']
    if data is None:
        return out + ['Not analysed yet.']
    out.append(f"Complete block: {complete(data)}. Holm family 20 (2 arms minus Qwen SEGA x 10 metrics).")
    if 'contrasts_holm' in data:
        out.append(contrast_table(data['contrasts_holm'], ['temp_only_sbase', 'base_default']))
    hyp = data.get('hypotheses_json', {}).get('seed0')
    if hyp:
        out.append(f"Exploratory H1/H2-style summary: H1 {hyp['H1']}, H2 {hyp['H2']}.")
        out += hypothesis_lines(hyp['family'])
    return out


def section_diag(name, data):
    title = {'ckpt_matrix': '## (b) Checkpoint x guidance matrix (P2, official DyPE, dev rows 0-7, seed 42)',
             'default_shift': '## P7 default flow shift (dev rows 0-7, seed 0, descriptive)'}[name]
    out = [title]
    if data is None:
        return out + ['Not analysed yet.']
    if 'lattice_summary' in data:
        out.append('Lattice statistic: FLUX.1-dev / Krea whitened power at the dev-selected peak (1/64 < f < 1/8), '
                   'geometric mean over 8 prompts, bootstrap CI on logs:')
        for r in data['lattice_summary'].to_dict('records'):
            out.append(f"- guidance {r['guidance']}: ratio {r['mean']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] (n={r['n']})")
    if 'paired' in data:
        out.append('Paired contrasts (descriptive, no multiplicity correction):')
        p = data['paired'].rename(columns={'mean': 'delta', 'ci_low': 'lo', 'ci_high': 'hi'})
        for r in p.to_dict('records'):
            if r['metric'] in ('imagereward', 'pickscore', 'musiq', 'musiq_crop1024', 'clip'):
                out.append(f"- {r['contrast']} {SHORT[r['metric']]}: {ci(r, r['metric'])} {verdict(r)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--analysis', type=Path, default=ANALYSIS)
    ap.add_argument('--export', type=Path, default=EXPORT)
    ap.add_argument('--results', type=Path, default=RESULTS)
    a = ap.parse_args()
    blocks = {b: load(b, a.analysis) for b in FILES}
    a.export.mkdir(parents=True, exist_ok=True)
    exported = []
    for b, data in blocks.items():
        if not data:
            continue
        for name, frame in data.items():
            if isinstance(frame, pd.DataFrame):
                path = a.export / f'analysis_{EXPORT_NAME[b]}_{name}.csv'
                frame.to_csv(path, index=False, float_format='%.17g')
                exported.append(str(path))
    lines = ['# Strengthening results (generated by scripts/strengthen_report.py)', '',
             'Every number below is copied from the analysis directories under .agent/notes/strengthen/analysis/. '
             'Delta = arm minus reference (released SEGA unless stated). tie = nominal 95% CI contains 0. '
             'H = Holm-adjusted p <= 0.05 within the registered family.', '',
             'Block status: ' + ', '.join(f"{b}={'complete' if complete(d) else ('partial' if d else 'missing')}"
                                         for b, d in blocks.items()), '']
    for part in (section_p1(blocks['P1']), section_diag('ckpt_matrix', blocks['ckpt_matrix']),
                 section_p3(blocks['P3']), section_e1(blocks['E1']), section_e2(blocks['E2']),
                 section_e3(blocks['E3']), section_diag('default_shift', blocks['default_shift'])):
        lines += part + ['']
    marker = '<!-- AUTO-TABLES: everything below is regenerated by scripts/strengthen_report.py -->'
    head = ''
    if a.results.exists() and marker in a.results.read_text():
        head = a.results.read_text().split(marker)[0]
    a.results.write_text(head + marker + '\n\n' + '\n'.join(lines) + '\n')
    print(json.dumps({'results': str(a.results), 'exported': len(exported),
                      'blocks': {b: ('complete' if complete(d) else 'partial' if d else 'missing')
                                 for b, d in blocks.items()}}))


if __name__ == '__main__':
    main()
