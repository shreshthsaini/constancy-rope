#!/usr/bin/env python3
"""Second-analyst driver for the strengthening campaign (CPU only; no scheduler calls).

Usage (project root, OMP_NUM_THREADS=8):
  strengthen_confirm.py status            per-block 90_ task states in taskq
  strengthen_confirm.py lattice           P2 symmetric checkpoint x guidance lattice index (needs PNGs only)
  strengthen_confirm.py run               merge finished blocks, run the registered analyses, export CSVs

Why a separate file: every published 90_ evaluator task verifies a checksum list that includes
scripts/strengthen_analysis.py, scripts/strengthen_evaluation.py, scripts/eval.py and crope/metrics.
Editing those files before all blocks are merged would fail the queued tasks, so this driver only
calls them. Analysis directories: .agent/notes/strengthen/analysis/<P1|P3|E1|E2|ckpt_matrix|...>.
Flat CSVs: /scratch/09032/saini_2/constancy_rope/eval_krea/strengthen/analysis_*.csv.

Lattice index (defined 2026-10-24 before any P2 metric was read; descriptive, eight dev prompts):
per prompt, average the log whitened spectra (strengthen_diagnostic_report.whitened_array, the F5
statistic) of the four checkpoint x guidance cells, take the argmax bin with period 8 to 64 px, and
read every cell's whitened power at that one bin. The selection is shared by all cells, so no cell is
favoured by its own maximum. Contrasts are log ratios per prompt (checkpoint at each guidance,
guidance within each checkpoint, interaction); geometric means with a percentile bootstrap
(2,000 draws, numpy default_rng(0) recreated per contrast). The historical dev-selected peak ratio
stays in the diagnostic report (lattice_summary.csv).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
QUEUE = SCRATCH / 'taskq'
NOTES = ROOT / '.agent/notes/strengthen'
ANALYSIS = NOTES / 'analysis'
EXPORT = SCRATCH / 'eval_krea/strengthen'
PY = '/work/09032/saini_2/vista/library/uv-envs/crope/bin/python'
AESTHETIC_REFS = SCRATCH / 'prompts/aesthetic4k_refs.jsonl'
CELLS = ('official_dype_dev_g3p5', 'official_dype_dev_g4p5', 'official_dype_krea_g3p5', 'official_dype_krea_g4p5')
# block -> evaluation queue manifests that must be merged first
NEEDS = {'P1': ['factorial'], 'P3': ['factorial', 'tsweep'], 'E1': ['fresh'], 'E2': ['seeds'],
         'ckpt_matrix': ['ckpt_matrix'], 'default_shift': ['default_shift']}
EVAL_BLOCKS = ['factorial', 'tsweep', 'fresh', 'seeds', 'ckpt_matrix', 'default_shift']


def tasks_of(block):
    record = json.loads((NOTES / 'evaluation' / block / 'evaluation_tasks.json').read_text())
    return sorted(record['tasks'])


def task_state(name):
    for state in ['done', 'failed', 'pending']:
        if (QUEUE / state / name).exists():
            return state
    if list((QUEUE / 'running').glob(name + '.*')):
        return 'running'
    return 'absent'


def status():
    report = {}
    for block in EVAL_BLOCKS:
        states = {name: task_state(name) for name in tasks_of(block)}
        report[block] = {'tasks': len(states), **{s: sum(v == s for v in states.values())
                         for s in ['pending', 'running', 'done', 'failed', 'absent']}}
    return report


def merged(block):
    """Run the frozen CPU merge once all of the block's tasks are done; True when canonical outputs exist."""
    states = [task_state(name) for name in tasks_of(block)]
    if any(s != 'done' for s in states):
        return False
    result = subprocess.run([PY, 'scripts/strengthen_evaluation.py', str(NOTES / f'{block}_queue.json'), '--merge'],
                            cwd=ROOT, capture_output=True, text=True)
    if result.returncode not in (0, 2):
        raise RuntimeError(f'merge failed for {block}: {result.stderr[-2000:]}')
    return json.loads((NOTES / 'evaluation' / block / 'merge_status.json').read_text())['complete'] is True


def whitened(path):
    from scripts.strengthen_diagnostic_report import spectrum
    return spectrum(path)


def geometric_boot(log_values):
    x = np.asarray(log_values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('bootstrap needs finite log values')
    draws = x[np.random.default_rng(0).integers(0, len(x), size=(2000, len(x)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5], method='linear')
    return float(np.exp(x.mean())), float(np.exp(lo)), float(np.exp(hi))


def lattice(out=ANALYSIS / 'ckpt_matrix_lattice2x2'):
    from crope.utils import image_stem, read_prompts
    manifest = json.loads((NOTES / 'ckpt_matrix_queue.json').read_text())
    prompts = read_prompts(manifest['prompts'])[:8]
    root = SCRATCH / 'diag/ckpt_matrix'
    rows, frequency = [], None
    for prompt in prompts:
        spectra = {}
        for cell in CELLS:
            png = root / cell / '4096x4096' / (image_stem(prompt['id'], 42) + '.png')
            sidecar = json.loads(png.with_suffix('.json').read_text())
            if sidecar['config']['method'] != cell or sidecar['config']['prompt_id'] != prompt['id']:
                raise ValueError(f'identity mismatch: {png}')
            f, p = whitened(png)
            frequency = f if frequency is None else frequency
            band = (frequency > 1 / 64) & (frequency < 1 / 8)
            if not np.array_equal(frequency, f) or not (p[band] > 0).all():
                raise ValueError(f'spectrum grid or in-band positivity differs: {png}')
            spectra[cell] = p
        # Empty low-frequency radial bins hold zero power; only the fixed 8-64 px band is used.
        mean_log = np.mean([np.log(np.where(band, spectra[c], 1.0)) for c in CELLS], axis=0)
        index = int(np.argmax(np.where(band, mean_log, -np.inf)))
        rows.append({'prompt_id': prompt['id'], 'bin': index, 'frequency': float(frequency[index]),
                     'period_pixels': float(1 / frequency[index]),
                     **{cell: float(spectra[cell][index]) for cell in CELLS}})
    frame = pd.DataFrame(rows)
    logs = {cell: np.log(frame[cell].to_numpy()) for cell in CELLS}
    dev35, dev45, krea35, krea45 = (logs[c] for c in CELLS)
    contrasts = {
        'checkpoint_dev_over_krea_at_g3p5': dev35 - krea35,
        'checkpoint_dev_over_krea_at_g4p5': dev45 - krea45,
        'guidance_g4p5_over_g3p5_dev': dev45 - dev35,
        'guidance_g4p5_over_g3p5_krea': krea45 - krea35,
        'interaction_checkpoint_ratio_g4p5_over_g3p5': (dev45 - krea45) - (dev35 - krea35),
    }
    summary = [{'quantity': f'cell_{cell}', 'n': len(frame), 'geometric_mean': g, 'ci_low': lo, 'ci_high': hi}
               for cell in CELLS for g, lo, hi in [geometric_boot(logs[cell])]]
    summary += [{'quantity': name, 'n': len(values), 'geometric_mean': g, 'ci_low': lo, 'ci_high': hi}
                for name, values in contrasts.items() for g, lo, hi in [geometric_boot(values)]]
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'per_prompt.csv', index=False, float_format='%.17g')
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False, float_format='%.17g')
    (out / 'definition.json').write_text(json.dumps({
        'statistic': 'whitened power (f^2 x radial mean power, Hann-windowed luminance) at one shared bin per prompt',
        'bin_rule': 'argmax over 1/64 < f < 1/8 of the mean log spectrum of the four cells',
        'cells': CELLS, 'prompts': [p['id'] for p in prompts], 'seed': 42,
        'bootstrap': '2000 draws, default_rng(0) per quantity, percentile 2.5/97.5 of mean log, exponentiated',
        'role': 'descriptive development diagnostic; not a held-out test'}, indent=2) + '\n')
    EXPORT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(EXPORT / 'analysis_ckpt_matrix_lattice2x2_per_prompt.csv', index=False, float_format='%.17g')
    pd.DataFrame(summary).to_csv(EXPORT / 'analysis_ckpt_matrix_lattice2x2_summary.csv', index=False,
                                 float_format='%.17g')
    return summary


def analyse(block):
    out = ANALYSIS / block
    if block in ('ckpt_matrix', 'default_shift'):
        if (out / 'report.json').exists():
            return 'kept_existing'
        cmd = [PY, 'scripts/strengthen_diagnostic_report.py', block, '--out', str(out)]
    else:
        cmd = [PY, 'scripts/strengthen_analysis.py', str(NOTES / f'analysis_{block}.json'), str(out)]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode not in (0, 2):
        raise RuntimeError(f'analysis failed for {block}: {result.stderr[-3000:]}')
    return 'complete' if result.returncode == 0 else 'incomplete'


def extra_exports():
    """Seed-level prompt means and the fixed sweep curve, copied from analysis outputs (no new statistic)."""
    written = []
    e2 = ANALYSIS / 'E2' / 'prompt_means.csv'
    if e2.exists():
        pd.read_csv(e2).to_csv(EXPORT / 'analysis_seeds_prompt_means.csv', index=False, float_format='%.17g')
        written.append('analysis_seeds_prompt_means.csv')
    p3 = ANALYSIS / 'P3' / 'means.csv'
    if p3.exists():
        from scripts.strengthen_analysis import SWEEP_TEMPERATURES
        curve = pd.read_csv(p3)
        curve = curve[curve.analysis == 'seed0'].copy()
        curve.insert(2, 'logit_multiplier', curve.method.map(SWEEP_TEMPERATURES))
        curve.sort_values(['metric', 'logit_multiplier']).to_csv(EXPORT / 'analysis_tsweep_curve.csv', index=False,
                                                                 float_format='%.17g')
        written.append('analysis_tsweep_curve.csv')
    return written


def run(blocks):
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    ready = {name: merged(name) for name in sorted({b for block in blocks for b in NEEDS[block]})}
    outcome = {}
    for block in blocks:
        if all(ready[b] for b in NEEDS[block]):
            outcome[block] = analyse(block)
        else:
            outcome[block] = 'waiting_for_evaluation'
    report = subprocess.run([PY, 'scripts/strengthen_report.py'], cwd=ROOT, capture_output=True, text=True)
    if report.returncode != 0:
        raise RuntimeError(report.stderr[-2000:])
    outcome['exports'] = extra_exports()
    outcome['merged'] = ready
    return outcome


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['status', 'lattice', 'run'])
    ap.add_argument('--blocks', nargs='+', default=['P1', 'P3', 'E1', 'E2', 'ckpt_matrix', 'default_shift'])
    a = ap.parse_args()
    result = status() if a.command == 'status' else lattice() if a.command == 'lattice' else run(a.blocks)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
