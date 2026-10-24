#!/usr/bin/env python3
"""Completion-gated descriptive reports for P2 and P7; no generation or queue changes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crope.utils import file_sha256, image_stem, read_prompts, runtime_environment

SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
NOTES = ROOT / '.agent/notes/strengthen'
METRICS = ('imagereward', 'pickscore', 'clip', 'musiq', 'musiq_crop1024',
           'clipiqa', 'clipiqa_crop1024', 'spectrum_log_rmse', 'spectrum_slope_error', 'tiling1024')
LOWER = {'spectrum_log_rmse', 'spectrum_slope_error', 'tiling1024'}
CELLS = ('official_dype_dev_g3p5', 'official_dype_dev_g4p5',
         'official_dype_krea_g3p5', 'official_dype_krea_g4p5')
LATTICE_DEFINITION = {
    'original_source': 'paper/figs/f5_checkpoint.py:whitened_spectrum,peak_ratio',
    'luminance': '0.2126R + 0.7152G + 0.0722B on RGB/255; no linear-sRGB conversion',
    'window': 'subtract full-image luminance mean, multiply separable numpy.hanning(n)',
    'power': 'abs(fft2(windowed_luminance))**2 / n**2',
    'radial_edges': 'numpy.geomspace(2/n, 0.5, 49)',
    'radial_average': 'numpy.digitize, weighted bincount / bin count',
    'whitening': 'multiply bin-average power by geometric-bin-center frequency squared',
    'peak': 'argmax dev whitened power where 1/64 < frequency < 1/8, first tie',
    'ratio': 'dev / Krea whitened power at the dev-selected peak frequency',
    'summary': 'geometric mean of eight prompt ratios; percentile bootstrap on their logarithms',
    'scope': 'descriptive, frequency selected from dev image; not an unbiased lattice-effect estimator',
}


def validate_design(manifest, samples, prompts):
    block = manifest['block']
    methods = CELLS if block == 'ckpt_matrix' else ('sega_default_shift',)
    seed = 42 if block == 'ckpt_matrix' else 0
    if block not in ('ckpt_matrix', 'default_shift') or len(prompts) < 8:
        raise ValueError('Unsupported diagnostic design')
    if len({row['id'] for row in prompts[:8]}) != 8:
        raise ValueError('Development prompt identities must be unique')
    expected = {(method, row['id'], seed) for method in methods for row in prompts[:8]}
    actual = [(s['method'], s['prompt_id'], s['seed']) for s in samples]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f'{block} requires exactly {len(expected)} fixed development images')


def bootstrap(values, *, geometric=False):
    import numpy as np
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (8,) or not np.isfinite(x).all() or (geometric and np.any(x <= 0)):
        raise ValueError('Descriptive intervals require all eight finite prompt values')
    if geometric:
        x = np.log(x)
    rng = np.random.default_rng(0)
    draws = x[rng.integers(0, 8, size=(2000, 8))].mean(axis=1)
    result = np.r_[x.mean(), np.percentile(draws, [2.5, 97.5], method='linear')]
    if geometric:
        result = np.exp(result)
    return {'n': 8, 'mean': float(result[0]), 'ci_low': float(result[1]), 'ci_high': float(result[2])}


def whitened_array(rgb):
    """Exact arithmetic/order of the historical F5 statistic, without its path/stat cache."""
    import numpy as np
    a = np.asarray(rgb, dtype=np.float64) / 255.0
    if a.ndim != 3 or a.shape[0] != a.shape[1] or a.shape[2] != 3:
        raise ValueError('Whitened spectrum requires a square RGB image')
    y = .2126*a[..., 0] + .7152*a[..., 1] + .0722*a[..., 2]
    n = y.shape[0]
    w = np.hanning(n)
    y = (y-y.mean()) * w[:, None] * w[None, :]
    power = np.abs(np.fft.fft2(y))**2 / (n*n)
    frequency = np.fft.fftfreq(n)
    radius = np.hypot(frequency[:, None], frequency[None, :]).ravel()
    edges = np.geomspace(2.0/n, .5, 49)
    indices = np.digitize(radius, edges)
    sums = np.bincount(indices, weights=power.ravel(), minlength=len(edges)+1)[1:len(edges)]
    count = np.bincount(indices, minlength=len(edges)+1)[1:len(edges)]
    centers = np.sqrt(edges[1:]*edges[:-1])
    return centers, sums / np.maximum(count, 1) * centers**2


def spectrum(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        if image.size != (4096, 4096):
            raise ValueError(f'Expected full 4K image: {path}')
        return whitened_array(np.asarray(image.convert('RGB')))


def peak_statistic(frequency, dev, krea):
    import numpy as np
    f, d, k = map(lambda x: np.asarray(x, dtype=np.float64), (frequency, dev, krea))
    eligible = (f > 1/64) & (f < 1/8)
    if f.ndim != 1 or f.shape != d.shape or f.shape != k.shape or not eligible.any():
        raise ValueError('Incompatible spectral vectors or empty fixed peak band')
    if not np.isfinite(f).all() or not np.isfinite(d).all() or not np.isfinite(k).all():
        raise ValueError('Nonfinite spectral power')
    index = int(np.argmax(np.where(eligible, d, -np.inf)))
    if d[index] <= 0 or k[index] <= 0:
        raise ValueError('Peak ratio requires positive power in both arms')
    return {'bin': index, 'frequency': float(f[index]), 'period_pixels': float(1/f[index]),
            'dev_power': float(d[index]), 'krea_power': float(k[index]),
            'ratio': float(d[index]/k[index])}


def load_metrics(samples, per_image, imagereward):
    """Join scorers by exact PNG path, with method/prompt/seed cross-checks."""
    import numpy as np
    import pandas as pd
    metrics, reward = pd.read_csv(per_image), pd.read_csv(imagereward)
    for frame, column in ((metrics, 'path'), (reward, 'png')):
        frame['_path'] = frame[column].map(lambda p: str(Path(p).resolve()))
        if frame['_path'].duplicated().any():
            raise ValueError('Duplicate PNG paths in score CSV')
        frame.set_index('_path', inplace=True)
    rows = []
    for sample in samples:
        path = str(Path(sample['path']).resolve())
        if path not in metrics.index or path not in reward.index:
            raise ValueError(f'Missing fixed-metric scores: {path}')
        a, b = metrics.loc[path], reward.loc[path]
        for frame in (a, b):
            for key in ('method', 'prompt_id', 'res'):
                if str(frame[key]) != str(sample[key]):
                    raise ValueError(f'Score identity mismatch for {key}: {path}')
        if float(a['seed']) != sample['seed']:
            raise ValueError(f'Score seed mismatch: {path}')
        row = {k: sample[k] for k in ('method', 'prompt_id', 'seed', 'path', 'res')}
        for metric in METRICS:
            value = float(b[metric] if metric == 'imagereward' else a[metric])
            if not np.isfinite(value):
                raise ValueError(f'Nonfinite {metric}: {path}')
            row[metric] = value
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_samples(prompts):
    from scripts.strengthen_analysis import inspect_sidecar
    from scripts.strengthen_diagnostics import KREA_REVISION
    root = SCRATCH / 'out_krea/pilot/sega_official/4096x4096'
    template = SCRATCH / 'taskq/done/30_pilot_sega_official_4096x4096_00000-00008_076ef49b9670.sh'
    text = template.read_text()
    if ('models--black-forest-labs--FLUX.1-Krea-dev/snapshots/'+KREA_REVISION) not in text:
        raise ValueError('Historical baseline lacks pinned Krea transformer ancestry')
    commands = [shlex.split(line) for line in text.splitlines() if ' scripts/sample.py ' in line]
    if len(commands) != 1:
        raise ValueError('Historical baseline generation command is ambiguous')
    command = commands[0]
    for flag, value in {'--method':'sega_official', '--out':str(root), '--start':'0', '--end':'8',
                        '--steps':'28', '--guidance':'4.5', '--seed':'0'}.items():
        if command[command.index(flag)+1] != value:
            raise ValueError('Historical baseline generation settings differ')
    samples, witnesses = [], []
    for row in prompts[:8]:
        path = root / (image_stem(row['id'], 0)+'.png')
        sample = {'method':'sega_official', 'raw_method':'sega_official', 'prompt_id':row['id'],
                  'seed':0, 'path':str(path), 'res':'4096x4096', 'prompt':row['prompt']}
        witness = inspect_sidecar(sample, row['prompt'], {'block':'P1'}, {})
        metadata = json.loads(path.with_suffix('.json').read_text())
        if metadata.get('png_sha256') != file_sha256(path):
            raise ValueError('Historical baseline PNG hash differs')
        samples.append(sample)
        witnesses.append({**witness, 'path':str(path), 'sidecar':str(path.with_suffix('.json')),
                          'sidecar_sha256':file_sha256(path.with_suffix('.json')),
                          'png_sha256':file_sha256(path), 'config':metadata['config']})
    return samples, witnesses, template


def summaries(frame, block, prompt_ids):
    means, paired = [], []
    for method in frame.method.unique():
        selected = frame[frame.method == method].set_index('prompt_id').loc[prompt_ids]
        for metric in METRICS:
            means.append({'method':method, 'metric':metric, **bootstrap(selected[metric])})
    contrasts = ({
        'dev_minus_krea_g3p5': {CELLS[0]:1, CELLS[2]:-1},
        'dev_minus_krea_g4p5': {CELLS[1]:1, CELLS[3]:-1},
        'g4p5_minus_g3p5_dev': {CELLS[1]:1, CELLS[0]:-1},
        'g4p5_minus_g3p5_krea': {CELLS[3]:1, CELLS[2]:-1},
        'guidance_effect_dev_minus_krea': {CELLS[1]:1, CELLS[0]:-1, CELLS[3]:-1, CELLS[2]:1},
    } if block == 'ckpt_matrix' else {
        'default_minus_official_shift': {'sega_default_shift':1, 'sega_official':-1}})
    for name, weights in contrasts.items():
        for metric in METRICS:
            pivot = frame.pivot(index='prompt_id', columns='method', values=metric).loc[prompt_ids]
            values = sum(weight*pivot[method] for method,weight in weights.items())
            paired.append({'contrast':name, 'metric':metric, 'weights':weights,
                           'higher_is_better':metric not in LOWER, **bootstrap(values)})
    return means, paired


def build_report(block, output, eval_root=None):
    import numpy as np
    import pandas as pd
    from scripts.strengthen_evaluation import expected_samples, completion_status, verify_samples
    manifest_path = NOTES / (block+'_queue.json')
    manifest = json.loads(manifest_path.read_text())
    prompts = read_prompts(manifest['prompts'])
    samples = expected_samples(manifest)
    validate_design(manifest, samples, prompts)
    status = completion_status(manifest, samples, SCRATCH/'taskq')
    if not status['complete']:
        raise ValueError(f'{block} incomplete: {len(status["missing_image_pairs"])} image pairs missing; '
                         f'{len(status["unfinished_tasks"])} tasks unfinished')
    for path, digest in manifest['sources'].items():
        if file_sha256(path) != digest:
            raise ValueError(f'Frozen generation source differs: {path}')
    witnesses = verify_samples(manifest, samples)
    base = Path(eval_root) if eval_root else SCRATCH/'eval_krea/strengthen'/block
    score_sources = [base/'per_image.csv', base/'ir.csv']
    frame = load_metrics(samples, *score_sources)
    if block == 'default_shift':
        previous, baseline_witnesses, template = baseline_samples(prompts)
        historical = [SCRATCH/'eval_krea/pilot/4096x4096/per_image.csv', SCRATCH/'eval_krea/ir/pilot.csv']
        frame = pd.concat([frame, load_metrics(previous, *historical)], ignore_index=True)
        score_sources.extend(historical)
        score_sources.append(template)
        witnesses.extend(baseline_witnesses)
    order = [row['id'] for row in prompts[:8]]
    means, paired = summaries(frame, block, order)
    report = {'schema':'strengthen-descriptive-diagnostics-v1', 'block':block, 'complete':True,
              'role':'development diagnostic; no confirmatory hypothesis, pass/fail rule or general failure rate',
              'bootstrap':{'n':2000, 'seed':0, 'unit':'paired prompt', 'interval':'nominal 95% percentile, linear quantile'},
              'multiplicity':'No correction or significance claims; report all fixed contrasts and metrics',
              'prompt_ids':order, 'prompt_rows':list(range(8)), 'metrics':list(METRICS),
              'means':means, 'paired':paired, 'images':witnesses,
              'input_sha256':{str(p):file_sha256(p) for p in [manifest_path, Path(manifest['prompts']), *score_sources]},
              'reporter_sha256':file_sha256(__file__), 'created_utc':datetime.now(timezone.utc).isoformat()}
    if block == 'ckpt_matrix':
        spectral, arrays = [], {}
        for sample in samples:
            frequency, power = spectrum(sample['path'])
            key = (sample['method'], sample['prompt_id'])
            arrays[key] = (frequency, power)
            spectral.append({'method':sample['method'], 'prompt_id':sample['prompt_id'],
                             'frequency':frequency.tolist(), 'whitened_power':power.tolist()})
        lattice, summaries_ = [], []
        for guidance, dev, krea in ((3.5,CELLS[0],CELLS[2]),(4.5,CELLS[1],CELLS[3])):
            values=[]
            for prompt_id in order:
                f,d = arrays[(dev,prompt_id)]
                fk,k = arrays[(krea,prompt_id)]
                if not np.array_equal(f,fk):
                    raise ValueError('Spectral frequency grids differ')
                value=peak_statistic(f,d,k)
                lattice.append({'guidance':guidance,'prompt_id':prompt_id,**value})
                values.append(value['ratio'])
            summaries_.append({'guidance':guidance,'statistic':'dev/Krea power at dev-selected peak',
                               **bootstrap(values,geometric=True)})
        report.update(lattice_definition=LATTICE_DEFINITION, spectra=spectral,
                      lattice_per_prompt=lattice, lattice_summary=summaries_)
    else:
        report['baseline_scope'] = ('Historical official-shift Krea pilot, matched prompts, prompt-derived seed0, '
            '28 steps and guidance4.5. Source revision and hardware may differ. Actual SDPA dispatch was not '
            'recorded; the legacy attention label is not backend verification. Timing is descriptive only.')
        report['schedule_comparison'] = []
        for witness in witnesses:
            metadata=json.loads(Path(witness['sidecar']).read_text())
            sigma=np.asarray(metadata['sigmas'],dtype=np.float64)
            report['schedule_comparison'].append({'method':witness['method'],'prompt_id':witness['prompt_id'],
                'last_evaluated_sigma':float(sigma[-1]),'evaluations_below_0p3':int((sigma<.3).sum())})
    output=Path(output)
    if output.exists():
        raise FileExistsError(f'Preserve prior report by choosing a new output directory: {output}')
    # No output directory or partial result is written until every requirement above passes.
    output.mkdir(parents=True)
    frame.to_csv(output/'per_prompt.csv',index=False,float_format='%.17g')
    pd.DataFrame(means).to_csv(output/'means.csv',index=False,float_format='%.17g')
    pd.DataFrame([{**row,'weights':json.dumps(row['weights'],sort_keys=True)} for row in paired]).to_csv(
        output/'paired.csv',index=False,float_format='%.17g')
    if block == 'ckpt_matrix':
        pd.DataFrame(report['lattice_summary']).to_csv(output/'lattice_summary.csv',index=False,float_format='%.17g')
        pd.DataFrame(report['lattice_per_prompt']).to_csv(output/'lattice_per_prompt.csv',index=False,float_format='%.17g')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def verify_report_images(report):
    if report.get('schema') != 'strengthen-descriptive-diagnostics-v1' or not report.get('complete'):
        raise ValueError('A complete validated diagnostic report is required')
    block = report.get('block')
    if block not in ('ckpt_matrix', 'default_shift'):
        raise ValueError('Unsupported diagnostic report block')
    ids = report.get('prompt_ids', [])
    if len(ids) != 8 or len(set(ids)) != 8 or report.get('prompt_rows') != list(range(8)):
        raise ValueError('Diagnostic report must contain the eight fixed development prompts')
    methods = CELLS if block == 'ckpt_matrix' else ('sega_default_shift', 'sega_official')
    seed = 42 if block == 'ckpt_matrix' else 0
    expected = {(method, prompt_id, seed) for method in methods for prompt_id in ids}
    actual = [(image['method'], image['prompt_id'], image['seed']) for image in report.get('images', [])]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Diagnostic report has an incomplete image set')
    for image in report['images']:
        if file_sha256(image['path']) != image['png_sha256'] or file_sha256(image['sidecar']) != image['sidecar_sha256']:
            raise ValueError(f'Diagnostic evidence changed: {image["path"]}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('block',choices=('ckpt_matrix','default_shift'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--eval-root',type=Path)
    args=p.parse_args()
    runtime_environment()
    try:
        report=build_report(args.block,args.out,args.eval_root)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f'REPORT NOT WRITTEN: {error}',file=sys.stderr)
        return 2
    print(json.dumps({'block':args.block,'images':len(report['images']),'report':str(args.out/'report.json')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
