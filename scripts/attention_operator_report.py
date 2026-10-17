"""Offline integrity and FP64 operator analysis for frozen BF16 captures."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from scripts.attention_operator_capture import CAPTURE, pinned_json, validate_protocol
from scripts.observer_pretrained_audit import read_json, require, sha256, tensor_hash
from scripts.observer_pretrained_report import compare_numbers, difference_stats, valid_digest, verify_gain, verify_tensor

POLICY = {'cell_shape': [16, 16], 'exact_cell_chebyshev_radius': 2,
          'text_exact_once': True, 'dtype': 'float64', 'rank_projection': None}
METHODS = ('sparse', 'zero_order', 'global_mean', 'complement_mean', 'weighted_cells')


def raster_cells(value, grid, cell):
    """H,(height*width),D raster -> H,cells,row-major tokens-in-cell,D."""
    height, width = grid
    cy, cx = cell
    require(height > 0 and width > 0 and cy > 0 and cx > 0
            and height % cy == width % cx == 0, 'Grid must contain complete cells')
    require(value.ndim == 3 and value.shape[1] == height*width, 'Raster shape mismatch')
    return value.reshape(value.shape[0], height//cy, cy, width//cx, cx, value.shape[-1]).permute(
        0, 1, 3, 2, 4, 5).reshape(value.shape[0], (height//cy)*(width//cx), cy*cx, value.shape[-1])


def exact_cells(coordinate, grid, cell, radius):
    y, x = coordinate
    require(0 <= y < grid[0] and 0 <= x < grid[1] and type(radius) is int and radius >= 0,
            'Invalid query geometry')
    rows, cols = grid[0]//cell[0], grid[1]//cell[1]
    row, col = y//cell[0], x//cell[1]
    return [r*cols+c for r in range(max(0, row-radius), min(rows, row+radius+1))
            for c in range(max(0, col-radius), min(cols, col+radius+1))]


def metrics(value, reference, minimum, maximum):
    delta = value-reference
    violation = torch.maximum(minimum-value, value-maximum).clamp_min(0)
    norm = reference.norm().item()
    return {'absolute_l2': delta.norm().item(), 'max_abs': delta.abs().max().item(),
            'relative_l2': delta.norm().item()/norm if norm else None,
            'dense_l2': norm, 'finite': bool(torch.isfinite(value).all()),
            'coordinate_range_violation_max': violation.max().item(),
            'coordinate_range_violating_count': int((violation > 0).sum())}


def analyze_tensors(query, key, value, sdpa_output, *, grid, cell, radius, text_tokens, coordinates, heads):
    """One captured call, CPU FP64 math; cross-moments computed once per call.

    Equations match pisa_complement_cpu.compare (tested independently). Retaining
    all queries here avoids recomputing large within-cell moments per support.
    """
    require(all(x.device.type == 'cpu' and bool(torch.isfinite(x).all()) for x in (query, key, value, sdpa_output)),
            'Finite CPU tensors required')
    q, k, v, saved = [x.double() for x in (query, key, value, sdpa_output)]
    require(q.ndim == k.ndim == v.ndim == saved.ndim == 3 and q.shape == saved.shape
            and k.shape == v.shape and q.shape[0] == k.shape[0] == len(heads)
            and q.shape[-1] == k.shape[-1] and q.shape[1] == len(coordinates)
            and k.shape[1] == text_tokens+grid[0]*grid[1], 'Analysis shape mismatch')
    kc, vc = raster_cells(k[:, text_tokens:], grid, cell), raster_cells(v[:, text_tokens:], grid, cell)
    mean_k, sum_v = kc.mean(2), vc.sum(2)
    cross = torch.einsum('hcbi,hcbj->hcij', vc, kc-mean_k[:, :, None])
    global_h = cross.mean(1)
    t = q/math.sqrt(q.shape[-1])
    logits = t @ k.transpose(1, 2)
    dense_lse = logits.logsumexp(-1)
    dense = logits.softmax(-1) @ v
    minimum_values, maximum_values = v.amin(1), v.amax(1)
    require(all(bool(torch.isfinite(x).all()) for x in (cross, logits, dense_lse, dense)),
            'Nonfinite FP64 reference or cell moments')
    ids = torch.arange(grid[0]*grid[1]).reshape(1, -1, 1)
    cell_ids = raster_cells(ids, grid, cell)[0, :, :, 0]+text_tokens
    rows = []
    for qi, coordinate in enumerate(coordinates):
        selected = exact_cells(coordinate, grid, cell, radius)
        remaining = [j for j in range(kc.shape[1]) if j not in selected]
        kept = torch.cat((torch.arange(text_tokens), cell_ids[selected].flatten()))
        kept_logits = logits[:, qi, kept]
        tq = t[:, qi]
        alpha_logits = torch.einsum('hd,hcd->hc', tq, mean_k[:, remaining])
        shift = torch.cat((kept_logits, alpha_logits), -1).amax(-1)
        weights = (kept_logits-shift[:, None]).exp()
        alpha = (alpha_logits-shift[:, None]).exp()
        den = weights.sum(-1)+kc.shape[2]*alpha.sum(-1)
        n0 = torch.einsum('hn,hnv->hv', weights, v[:, kept])+torch.einsum('hc,hcv->hv', alpha, sum_v[:, remaining])
        conditional_h = cross[:, remaining].mean(1) if remaining else torch.zeros_like(global_h)
        corrections = {
            'zero_order': torch.zeros_like(n0),
            'global_mean': alpha.sum(-1)[:, None]*torch.einsum('hvd,hd->hv', global_h, tq),
            'complement_mean': alpha.sum(-1)[:, None]*torch.einsum('hvd,hd->hv', conditional_h, tq),
            'weighted_cells': torch.einsum('hc,hcvd,hd->hv', alpha, cross[:, remaining], tq),
        }
        outputs = {name: (n0+term)/den[:, None] for name, term in corrections.items()}
        outputs['sparse'] = torch.einsum('hn,hnv->hv', kept_logits.softmax(-1), v[:, kept])
        approx_lse = shift+den.log()
        exact_lse = kept_logits.logsumexp(-1)
        true_exact_mass = (exact_lse-dense_lse[:, qi]).exp()
        approx_exact_mass = weights.sum(-1)/den
        require(all(bool(torch.isfinite(x).all()) for x in
                    (*outputs.values(), approx_lse, exact_lse, true_exact_mass, approx_exact_mass)),
                'Nonfinite operator output or mass')
        for hi, head in enumerate(heads):
            minimum, maximum = minimum_values[hi], maximum_values[hi]
            controls = {}
            for name, output in outputs.items():
                lse = exact_lse[hi] if name == 'sparse' else approx_lse[hi]
                mass = 1. if name == 'sparse' else approx_exact_mass[hi].item()
                controls[name] = {**metrics(output[hi], dense[hi, qi], minimum, maximum),
                                  'output': output[hi].tolist(), 'log_mass': lse.item(),
                                  'log_mass_minus_dense': (lse-dense_lse[hi, qi]).item(),
                                  'exact_support_probability': mass,
                                  'exact_support_probability_minus_true': mass-true_exact_mass[hi].item()}
            rows.append({'head': head, 'query_index': qi, 'coordinate': coordinate,
                         'image_raster_index': coordinate[0]*grid[1]+coordinate[1],
                         'selected_cells': selected, 'exact_key_count': len(kept),
                         'dense_output': dense[hi, qi].tolist(), 'dense_log_mass': dense_lse[hi, qi].item(),
                         'true_exact_support_probability': true_exact_mass[hi].item(), 'controls': controls,
                         'sdpa_bfloat16_vs_fp64': metrics(saved[hi, qi], dense[hi, qi], minimum, maximum)})
    return rows


def verify_sources(root, sources):
    require(bool(sources), 'Missing source hashes')
    for name, digest in sources.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and valid_digest(digest), 'Invalid source receipt')
        require(sha256(root/path) == digest, f'Frozen source changed: {name}')


def load_payload(path, receipt, *, grid, text_tokens, coordinates, heads, dim, gain_hash):
    require(sha256(path) == receipt['file_sha256'], 'Capture file hash mismatch')
    data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    sequence = text_tokens+grid[0]*grid[1]
    specs = {'query': ((1, len(heads), len(coordinates), dim), torch.bfloat16),
             'key': ((1, len(heads), sequence, dim), torch.bfloat16),
             'value': ((1, len(heads), sequence, dim), torch.bfloat16),
             'sdpa_output': ((1, len(heads), len(coordinates), dim), torch.bfloat16),
             'gain_bfloat16': ((dim,), torch.bfloat16), 'head_indices': ((len(heads),), torch.int64),
             'query_joint_indices': ((len(coordinates),), torch.int64),
             'query_image_indices': ((len(coordinates),), torch.int64), 'key_joint_indices': ((sequence,), torch.int64)}
    require(isinstance(data, dict) and set(data) == set(receipt['tensors']) == set(specs), 'Capture tensor fields mismatch')
    for name, (shape, dtype) in specs.items():
        tensor, declared = data[name], receipt['tensors'][name]
        require(isinstance(tensor, torch.Tensor) and tensor.shape == shape and tensor.dtype == dtype
                and bool(torch.isfinite(tensor).all()), f'Invalid captured tensor: {name}')
        require(declared == {'shape': list(shape), 'dtype': str(dtype), 'sha256': tensor_hash(tensor)},
                f'Tensor receipt mismatch: {name}')
    image = torch.tensor([y*grid[1]+x for y, x in coordinates])
    for name, expected in {'head_indices': torch.tensor(heads), 'query_image_indices': image,
                           'query_joint_indices': image+text_tokens, 'key_joint_indices': torch.arange(sequence)}.items():
        require(torch.equal(data[name], expected), f'Token identity/order mismatch: {name}')
    require(tensor_hash(data['gain_bfloat16']) == gain_hash, 'Captured gain changed')
    return data


def verify_replay(directory, completion, shape):
    require(completion['completed_forwards'] == 2 and completion['replay']['within_job_byte_equal'] is True,
            'Incomplete or failed capture replay')
    values = []
    require(set(completion['outputs']) == {'baseline', 'capture'}, 'Missing replay output receipt')
    for arm in ('baseline', 'capture'):
        entry = completion['outputs'][arm]
        require(entry['shape'] == list(shape) and entry['dtype'] == 'torch.bfloat16', 'Replay output geometry mismatch')
        value, _ = verify_tensor(directory/f'{arm}_output.pt', 'output', shape, torch.bfloat16,
                                 file_digest=entry['file_sha256'], tensor_digest=entry['sha256'])
        values.append(value)
    require(tensor_hash(values[0]) == tensor_hash(values[1]), 'Saved replay outputs are not byte equal')
    compare_numbers(completion['replay']['within_job'], difference_stats(values[1], values[0]), 'Replay contrast')
    return {'baseline_sha256': tensor_hash(values[0]), 'capture_sha256': tensor_hash(values[1]), 'byte_equal': True}


def build_report(protocol_path, expected_sha, root):
    require(valid_digest(expected_sha) and sha256(protocol_path) == expected_sha, 'Protocol pin mismatch')
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    require(protocol.get('analysis_policy') == POLICY, 'Frozen analysis policy mismatch')
    require({'scripts/attention_operator_report.py', 'tests/test_attention_operator_report.py',
             'scripts/pisa_complement_cpu.py'} <= set(protocol['source_sha256']), 'Analysis sources not frozen')
    verify_sources(ROOT, protocol['source_sha256'])
    dependencies = {name: pinned_json(protocol[name]) for name in
                    ('accepted_cr18_report', 'accepted_cr18_receipt', 'cr18_protocol', 'witness_retention')}
    prior = dependencies['accepted_cr18_report']
    require(dependencies['accepted_cr18_receipt']['report_sha256'] == protocol['accepted_cr18_report']['sha256'],
            'Accepted CR18 report receipt mismatch')
    verify_sources(ROOT, dependencies['cr18_protocol']['source_sha256'])
    manifests, completions, replays = {}, {}, {}
    for index in protocol['development_prompt_indices']:
        directory = root/f'prompt{index}'
        require(not (directory/'failure.json').exists(), 'Capture has failure marker')
        manifest, completion = read_json(directory/'manifest.json'), read_json(directory/'completion.json')
        accepted = prior['raw_json'][f'prompt{index}']['manifest.json']
        require(manifest['prompt_index'] == index and manifest['protocol_sha256'] == expected_sha
                and manifest['source_sha256'] == protocol['source_sha256'] and manifest['capture'] == CAPTURE,
                'Capture manifest provenance mismatch')
        for key in ('model', 'prompt', 'conditioning_sha256', 'torch', 'cuda', 'diffusers'):
            require(manifest[key] == accepted[key], f'CR18 provenance mismatch: {key}')
        require(manifest['input_sha256'] == accepted['state']['input_sha256']
                and manifest['accepted_cr18_report_sha256'] == protocol['accepted_cr18_report']['sha256']
                and manifest['cr18_protocol_sha256'] == protocol['cr18_protocol']['sha256'], 'Input ancestry mismatch')
        require(manifest['layout'] == 'B,H,S,D; joint keys are text first then image raster order'
                and manifest['sdpa_target'] == {'scale': 'default 1/sqrt(128)', 'dropout_p': 0., 'is_causal': False, 'attn_mask': None}
                and manifest['forward_order'] == ['baseline', 'capture'] and manifest['forward_budget'] == 2,
                'Attention target changed')
        original = prior['raw_json'][f'prompt{index}']['original_g0.json']
        gain = verify_gain(original['gain'])
        require(manifest['gain_bfloat16_sha256'] == original['consumed_gain_bfloat16_sha256'] == tensor_hash(gain.bfloat16())
                and manifest['gain_float32_sha256'] == tensor_hash(gain), 'CR18 gain mismatch')
        require(completion['captured_attention_calls'] == completion['captured_sdpa_calls'] == 57
                and set(completion['captures']) == {str(i) for i in CAPTURE['attention_call_indices_zero_based']},
                'Capture call accounting mismatch')
        require(read_json(directory/'replay_validation.json') == completion['replay'], 'Replay receipt mismatch')
        require(read_json(directory/'progress.json') == {'status': 'completed', 'completed_forwards': 2}, 'Capture not completed')
        require(completion['on_trajectory'] is False and completion['efficacy_evaluated'] is False, 'Scope changed')
        replays[str(index)] = verify_replay(directory, completion, (1, 65536, 64))
        manifests[str(index)], completions[str(index)] = manifest, completion
    rows, artifact_hashes = [], {}
    for index in protocol['development_prompt_indices']:
        directory = root/f'prompt{index}'
        for call in CAPTURE['attention_call_indices_zero_based']:
            receipt = completions[str(index)]['captures'][str(call)]
            path = directory/f'attention_{call:02d}.pt'
            require(Path(receipt['path']).name == path.name and Path(receipt['path']).is_absolute(), 'Capture file identity mismatch')
            require(read_json(directory/f'attention_{call:02d}.json') == receipt, 'Capture sidecar mismatch')
            data = load_payload(path, receipt, grid=protocol['grid'], text_tokens=512,
                                coordinates=CAPTURE['image_query_grid_coordinates'], heads=CAPTURE['heads_zero_based'],
                                dim=128, gain_hash=manifests[str(index)]['gain_bfloat16_sha256'])
            result = analyze_tensors(*(data[name][0] for name in ('query', 'key', 'value', 'sdpa_output')),
                                     grid=protocol['grid'], cell=POLICY['cell_shape'], radius=2, text_tokens=512,
                                     coordinates=CAPTURE['image_query_grid_coordinates'], heads=CAPTURE['heads_zero_based'])
            rows.extend(dict(prompt_index=index, attention_call=call, **row) for row in result)
            artifact_hashes[f'{index}/{call}'] = receipt['file_sha256']
    require(len(rows) == 192, 'Missing per-query analyses')
    return {'accepted': True, 'protocol_sha256': expected_sha, 'source_sha256': protocol['source_sha256'],
            'analyzer_sha256': sha256(__file__), 'policy': POLICY, 'threads': torch.get_num_threads(),
            'torch': torch.__version__, 'replay_verification': replays, 'artifact_sha256': artifact_hashes,
            'query_count': len(rows), 'rows': rows, 'automatic_promotion': False,
            'limits': ['Captured artificial CR18 states, not natural trajectories or image-quality evidence.',
                       'Approximation errors use saved consumed BF16 inputs evaluated in FP64. SDPA BF16 gap is separate; no error subtraction.',
                       'Coordinatewise value bounds are necessary for convex-hull membership, not a full convex-hull test.',
                       'All controls use the same approximate centroid denominator except exact sparse; no PISA theorem bound.',
                       'No rank projection, GPU speed, efficacy, novelty or selected-winner claim.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = build_report(args.protocol, args.protocol_sha256, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'accepted': report['accepted'], 'query_count': report['query_count']}))


if __name__ == '__main__':
    main()
