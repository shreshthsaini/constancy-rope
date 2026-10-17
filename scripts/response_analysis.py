"""Summarize validated local response measurements without making efficacy claims."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    done = json.loads((root / 'completion.json').read_text())
    if (not done.get('gpu_probe_executed') or not done['trajectory_bitwise_equal'] or not done['baseline_history_equal']
            or done['baseline_sha256'] != done['probed_sha256']):
        raise ValueError('A changed baseline trajectory invalidates response analysis')
    if sorted(done['observed_probe_steps']) != sorted(manifest['probe_steps']):
        raise ValueError('Incomplete probe steps')
    rows, total_calls = [], 0
    for step in manifest['probe_steps']:
        record = json.loads((root / f'step_{step:03d}.json').read_text())
        if record['step'] != step or record['replay_max_abs_velocity_error'] != 0:
            raise ValueError('Invalid step or failed unperturbed replay')
        if record['epsilon'] != manifest['epsilon']:
            raise ValueError('Perturbation strength changed')
        if [d['name'] for d in record['directions']] != done['directions']:
            raise ValueError('Direction set changed between steps')
        total_calls += record['extra_calls']
        baseline = np.asarray(record['baseline'], dtype=float)
        if baseline.ndim != 1 or not np.isfinite(baseline).all() or record['epsilon'] <= 0:
            raise ValueError('Invalid baseline statistics or perturbation strength')
        split = len(record['bins'][0]['lower'])
        for direction in record['directions']:
            values = [np.asarray(direction[name], dtype=float) for name in
                      ('plus', 'minus', 'half_plus', 'half_minus')]
            if any(v.shape != baseline.shape or not np.isfinite(v).all() for v in values):
                raise ValueError('Invalid response vectors')
            plus, minus, half_plus, half_minus = values
            jacobian = (plus - minus) / (2 * record['epsilon'])
            observed = np.r_[half_plus - baseline, half_minus - baseline]
            predicted = np.r_[jacobian * record['epsilon'] / 2, -jacobian * record['epsilon'] / 2]
            effect_norm = float(np.linalg.norm(observed))
            axes = direction['preserve_axes']
            off_axis = None
            if axes:
                other = jacobian[split:] if axes == [1] else jacobian[:split]
                off_axis = float(np.linalg.norm(other) / max(np.linalg.norm(jacobian), 1e-30))
            rows.append({'prompt_index': manifest['prompt_index'], 'step': step,
                         'sigma': record['sigma'], 'direction': direction['name'],
                         'half_effect_l2': effect_norm,
                         'linear_residual_l2': float(np.linalg.norm(observed - predicted)),
                         'relative_prediction_error': (float(np.linalg.norm(observed - predicted) / effect_norm)
                                                       if effect_norm > 1e-12 else None),
                         'fraction_negative_jacobian': float(np.mean(jacobian < 0)),
                         'off_axis_response_fraction': off_axis,
                         'jacobian': jacobian.tolist()})
    if total_calls != done['extra_forward_calls']:
        raise ValueError('Extra-call accounting mismatch')
    return {'purpose': 'Local response diagnosis, not quality or generalization evidence',
            'manifest': manifest, 'validation': done, 'rows': rows,
            'interpretation': 'Prediction errors near or above one do not beat the zero-change predictor. '
                              'Tiny responses and numerical precision need separate inspection. '
                              'Three tested controls do not identify the full reachable subspace.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
