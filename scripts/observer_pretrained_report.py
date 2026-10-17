"""CPU validation/archive for the bounded artificial-state observer campaign.

Saved tensors are checked independently. Conditioning and full projected-feature
hashes remain frozen-worker runtime receipts because those tensors were not saved.
No result is promoted to an efficacy, novelty or generalization claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.observer_pretrained_audit import (
    ARMS, SHIFT_SPEC, artificial_state, load_verified_inputs, observer_record,
    read_json, require, sha256, target_observer, tensor_hash, validate_protocol,
)

GRID = (256, 256)
STATE_SHAPE = (1, 65536, 64)
OBSERVER_SHAPE = (1, 65536)


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def compare_numbers(actual, expected, label, *, rtol=1e-9, atol=1e-11):
    """Strict structure/bools, finite numeric comparison with explicit tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f'{label}: keys mismatch')
        for key in expected:
            compare_numbers(actual[key], expected[key], f'{label}.{key}', rtol=rtol, atol=atol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f'{label}: list mismatch')
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare_numbers(a, b, f'{label}[{i}]', rtol=rtol, atol=atol)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected, f'{label}: value mismatch')
    elif isinstance(expected, (int, float)):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), f'{label}: numeric mismatch')
    else:
        raise ValueError(f'{label}: unsupported value type')


def difference_stats(value, reference):
    a, b = value.double(), reference.double()
    require(a.shape == b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
            'Nonfinite or mismatched contrast tensors')
    difference = a - b
    denominator = torch.linalg.vector_norm(b).item()
    magnitude = torch.linalg.vector_norm(difference).item()
    return {'max_abs': difference.abs().max().item(), 'l2': magnitude,
            'relative_l2': magnitude / denominator if denominator else None,
            'reference_l2': denominator, 'bitwise_equal': torch.equal(value, reference)}


def recompute_contrasts(outputs):
    require(set(outputs) == set(ARMS), 'All five output arms required')
    base, repeat, adaptive, fixed_shift, gain_only = [outputs[name] for name in ARMS]
    return {
        'exact_repeat': difference_stats(repeat, base),
        'adaptive_total_shifted_g1_minus_original_g0': difference_stats(adaptive, base),
        'fixed_g0_shift_leakage': difference_stats(fixed_shift, base),
        'gain_only_original_g1_minus_original_g0': difference_stats(gain_only, base),
        'mediation_residual_shifted_g1_minus_original_g1': difference_stats(adaptive, gain_only),
        'gain_effect_at_shifted_features': difference_stats(adaptive, fixed_shift),
        'shift_gain_interaction': difference_stats(adaptive.double() - gain_only.double(),
                                                  fixed_shift.double() - base.double()),
    }


def safe_ratio(numerator, denominator):
    require(math.isfinite(numerator) and math.isfinite(denominator) and numerator >= 0 and denominator >= 0,
            'Ratio requires finite nonnegative norms')
    return {'numerator': numerator, 'denominator': denominator,
            'ratio': numerator / denominator if denominator else None,
            'zero_denominator': denominator == 0,
            'both_zero': numerator == denominator == 0}


def continuous_measures(outputs, contrasts):
    def norm(name):
        return contrasts[name]['l2']
    gain = norm('gain_only_original_g1_minus_original_g0')
    fixed = norm('fixed_g0_shift_leakage')
    mediation = norm('mediation_residual_shifted_g1_minus_original_g1')
    interaction = norm('shift_gain_interaction')
    adaptive = norm('adaptive_total_shifted_g1_minus_original_g0')
    base, _, shifted, _, gain_only = [outputs[name].double() for name in ARMS]
    a, b = shifted-base, gain_only-base
    denominator = a.norm().item() * b.norm().item()
    return {'gain_only_to_fixed_gain_residual': safe_ratio(gain, fixed),
            'adaptive_to_fixed_gain_residual': safe_ratio(adaptive, fixed),
            'mediation_residual_to_gain_only': safe_ratio(mediation, gain),
            'interaction_to_gain_only': safe_ratio(interaction, gain),
            'interaction_to_fixed_gain_residual': safe_ratio(interaction, fixed),
            'adaptive_and_gain_only_cosine': (a.flatten() @ b.flatten()).item() / denominator if denominator else None,
            'interpretation': 'Continuous descriptive ratios only; zero denominators remain null with explicit flags.'}


def verify_tensor(path, key, shape, dtype, *, file_digest=None, tensor_digest=None):
    path = Path(path)
    require(path.is_file(), f'Missing tensor evidence: {path}')
    actual_file = sha256(path)
    require(file_digest is None or actual_file == file_digest, f'Tensor file hash mismatch: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    require(isinstance(payload, dict) and set(payload) == {key}, f'Tensor payload mismatch: {path}')
    tensor = payload[key]
    require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == tuple(shape) and tensor.dtype == dtype,
            f'Tensor shape/dtype mismatch: {path}')
    require(bool(torch.isfinite(tensor).all()), f'Nonfinite tensor: {path}')
    digest = tensor_hash(tensor)
    require(tensor_digest is None or digest == tensor_digest, f'Tensor contents hash mismatch: {path}')
    return tensor, {'path': str(path.resolve()), 'file_sha256': actual_file, 'tensor_sha256': digest,
                    'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'independently_verified_on_cpu': True}


def verify_gain(record):
    require(set(record) == {'float32', 'cast_bfloat16', 'float32_sha256', 'bfloat16_sha256'}, 'Gain receipt keys mismatch')
    value = torch.tensor(record['float32'], dtype=torch.float32)
    cast = torch.tensor(record['cast_bfloat16'], dtype=torch.float32)
    require(value.shape == cast.shape == (128,) and bool(torch.isfinite(value).all()) and bool((value >= 1).all()),
            'Expected 128 finite SEGA gains at least one')
    require(torch.equal(value[::2], value[1::2]) and torch.equal(value[:16], torch.ones(16)), 'Gain axis/pair mismatch')
    require(torch.equal(value.bfloat16().float(), cast), 'BF16 gain cast mismatch')
    require(tensor_hash(value) == record['float32_sha256'] and
            tensor_hash(value.bfloat16()) == record['bfloat16_sha256'], 'Gain hash mismatch')
    return value


def validate_crossings(records):
    require(set(records) == set(ARMS), 'All five arm receipts required')
    values = {arm: verify_gain(records[arm]['gain']) for arm in ARMS}
    for arm in ARMS:
        expected = values['shifted_g1' if arm in ('shifted_g1', 'original_g1') else 'original_g0']
        require(torch.equal(values[arm], expected), f'Wrong gain crossing: {arm}')
        record = records[arm]
        require(record.get('consumed_gain_bfloat16_sha256') == tensor_hash(expected.bfloat16()),
                f'Consumed gain cast mismatch: {arm}')
        require(record.get('attention_calls') == 57 and record.get('attention_query_dtype') == 'torch.bfloat16',
                f'Attention runtime receipt mismatch: {arm}')
        require(record.get('incoming_features_equal_baseline') is True, 'Incoming feature equality check failed')
        feature_hash = record.get('realized_feature_sha256')
        require(valid_digest(feature_hash), 'Missing realized feature hash')
        reference = records['shifted_g1' if arm in ('shifted_g1', 'shifted_g0') else 'original_g0']
        require(feature_hash == reference['realized_feature_sha256'], f'Cached feature crossing mismatch: {arm}')
    require(records['shifted_g1']['centered_feature_error'] == records['shifted_g0']['centered_feature_error'],
            'Cached shifted feature geometry differs across arms')
    geometry = records['shifted_g1']['centered_feature_error']
    for key in ('max_abs', 'l2', 'realized_common_channel_delta_l2', 'realized_transverse_delta_l2', 'realized_full_delta_l2'):
        require(isinstance(geometry.get(key), (float, int)) and math.isfinite(geometry[key]) and geometry[key] >= 0,
                'Nonfinite or invalid runtime feature geometry')
    for key in ('relative_l2', 'transverse_fraction_of_full_delta'):
        value = geometry.get(key)
        require(value is None or (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0),
                'Nonfinite runtime feature ratio')
    common, transverse, full = [geometry[key] for key in ('realized_common_channel_delta_l2',
                                                        'realized_transverse_delta_l2', 'realized_full_delta_l2')]
    require(math.isclose(common**2 + transverse**2, full**2, rel_tol=1e-8, abs_tol=1e-12),
            'Feature geometry violates orthogonal decomposition')
    compare_numbers(geometry['l2'], transverse, 'Feature transverse norm')
    compare_numbers(geometry['transverse_fraction_of_full_delta'], transverse/full if full else None,
                    'Feature transverse fraction')
    for arm in ('original_g0', 'repeat_original_g0', 'original_g1'):
        require(records[arm]['centered_feature_error'] is None and records[arm]['target_error'] is None,
                'Unshifted arm has shifted feature geometry')
    return values


def verify_campaign_presence(jobroot):
    root = Path(jobroot)
    for index in (0, 1):
        path = root / f'prompt{index}'
        require((path / 'completion.json').is_file(), 'Both prompts must be completed before analysis')
        require(not (path / 'failure.json').exists(), 'Campaign includes a failed prompt')
        for name in ('manifest.json', 'validation.json', 'repeat_validation.json', 'input.pt'):
            require((path / name).is_file(), f'Incomplete prompt evidence: {path / name}')
        for arm in ARMS:
            for suffix in ('.json', '_output.pt', '_observer.pt'):
                require((path / f'{arm}{suffix}').is_file(), f'Incomplete arm evidence: {arm}{suffix}')


def verify_observers(path, records):
    means, targets, verified = {}, {}, {}
    for arm in ARMS:
        tensor_path = path / f'{arm}_observer.pt'
        data = torch.load(tensor_path, map_location='cpu', weights_only=True, mmap=True)
        require(isinstance(data, dict) and set(data) == {'actual_observer', 'target'}, 'Observer payload mismatch')
        for key in data:
            require(isinstance(data[key], torch.Tensor) and tuple(data[key].shape) == OBSERVER_SHAPE and
                    data[key].dtype == torch.float32 and bool(torch.isfinite(data[key]).all()),
                    'Observer/target shape, dtype or finiteness mismatch')
        means[arm], targets[arm] = data['actual_observer'], data['target']
        declared = records[arm]['observer']
        require(tensor_hash(means[arm]) == declared['sha256'], 'Observer hash mismatch')
        recomputed = observer_record(means[arm], GRID)
        compare_numbers(declared, recomputed, f'{arm} CPU observer spectrum', rtol=5e-5, atol=1e-6)
        if arm in ('shifted_g1', 'shifted_g0'):
            compare_numbers(records[arm]['target_error'], difference_stats(means[arm], targets[arm]),
                            f'{arm} target error', rtol=1e-8, atol=1e-10)
        verified[arm] = {'path': str(tensor_path.resolve()), 'file_sha256': sha256(tensor_path),
                         'observer_sha256': tensor_hash(means[arm]), 'target_sha256': tensor_hash(targets[arm]),
                         'independently_verified_on_cpu': True}
    for arm in ARMS:
        reference = 'shifted_g1' if arm in ('shifted_g1', 'shifted_g0') else 'original_g0'
        require(torch.equal(means[arm], means[reference]) and tensor_hash(means[arm]) == tensor_hash(means[reference]),
                'Observer crossing mismatch')
        require(torch.equal(targets[arm], targets['original_g0']) and
                tensor_hash(targets[arm]) == tensor_hash(targets['original_g0']), 'Cached target changed between arms')
    reconstructed = target_observer(means['original_g0'], GRID)
    scale = max(1., means['original_g0'].abs().max().item())
    require(torch.allclose(targets['original_g0'], reconstructed, rtol=1e-5, atol=1e-6*scale),
            'Target violates frozen harmonic/DC/amplitude construction')
    return {'files': verified, 'target_cpu_reconstruction_error': difference_stats(targets['original_g0'], reconstructed),
            'comparison_tolerances': {'observer_spectra_rtol': 5e-5, 'observer_spectra_atol': 1e-6,
                                      'target_reconstruction_rtol': 1e-5, 'target_reconstruction_atol': 1e-6*scale},
            'tolerance_scope': 'CPU/CUDA arithmetic reproduction only, not an efficacy or response-size threshold'}


def validate_manifest(manifest, protocol, protocol_hash, submission, index, prompt, state_meta, accepted_manifest):
    expected = {'protocol_sha256': protocol_hash, 'prompt_index': index, 'prompt': prompt,
                'state': state_meta, 'source_sha256': protocol['source_sha256'], 'model': protocol['model'],
                'accepted_report_sha256': protocol['accepted_report_sha256'],
                'accepted_receipt_sha256': protocol['accepted_receipt_sha256'],
                'endpoint_file_sha256': protocol['endpoint_file_sha256'][str(index)],
                'allocation_numerics': protocol['allocation_numerics'], 'sigma': .5, 'guidance': 4.5,
                'resolution': 4096, 'forward_calls_budget': 5, 'arm_order': list(ARMS),
                'observer_shift': SHIFT_SPEC, 'slurm_job_id': str(submission['job_id']),
                'torch': accepted_manifest['torch'], 'cuda': accepted_manifest['cuda'],
                'diffusers': accepted_manifest['experiment_configuration']['diffusers'],
                'model_parameter_dtypes': ['torch.bfloat16']}
    for key, value in expected.items():
        require(manifest.get(key) == value, f'Manifest {key} mismatch')
    hashes = manifest.get('conditioning_sha256', {})
    require(set(hashes) == {'encoder_hidden_states', 'pooled_projections', 'txt_ids', 'img_ids'} and
            all(valid_digest(value) for value in hashes.values()), 'Conditioning runtime hash receipt incomplete')
    require(isinstance(manifest.get('hardware'), str) and bool(manifest['hardware']), 'Missing hardware receipt')
    require(math.isfinite(manifest.get('model_loading_seconds', math.nan)) and manifest['model_loading_seconds'] >= 0,
            'Invalid loading time')


def validate_completion(completion, outputs, records):
    require(completion.get('completed_forwards') == 5 and completion.get('exact_repeat_equal') is True,
            'Completion count/repeat failure')
    require(completion.get('arms') == records, 'Completion arm receipts differ from raw arm files')
    require(all(completion.get(key) is False for key in ('on_trajectory', 'efficacy_evaluated', 'novelty_established')),
            'Invalid completion scope')
    contrasts = recompute_contrasts(outputs)
    require(contrasts['exact_repeat']['bitwise_equal'] and
            tensor_hash(outputs['original_g0']) == tensor_hash(outputs['repeat_original_g0']),
            'Saved outputs fail exact repeat, including raw tensor bytes')
    compare_numbers(completion.get('contrasts'), contrasts, 'Declared contrasts')
    gains = validate_crossings(records)
    g0, g1 = gains['original_g0'], gains['shifted_g1']
    compare_numbers(completion.get('gain_change_float32'), difference_stats(g1, g0), 'Gain FP32 change')
    compare_numbers(completion.get('gain_change_bfloat16'), difference_stats(g1.bfloat16(), g0.bfloat16()), 'Gain BF16 change')
    dead = torch.equal(g0.bfloat16(), g1.bfloat16())
    require(completion.get('gain_quantization_dead') is dead, 'Gain quantization-dead mismatch')
    require(completion.get('gain_changed_bfloat16_coordinates') == int((g0.bfloat16() != g1.bfloat16()).sum()),
            'Changed BF16 gain count mismatch')
    return contrasts


def build_report(jobroot, protocol_path, submission_path):
    root = Path(jobroot).resolve()
    verify_campaign_presence(root)
    protocol, submission = read_json(protocol_path), read_json(submission_path)
    validate_protocol(protocol)
    digest = sha256(protocol_path)
    require(submission.get('protocol_sha256') == digest, 'Submission protocol hash mismatch')
    require(submission.get('source_commit') == protocol.get('source_commit'), 'Submission source commit mismatch')
    expected_files = [str(root / f'prompt{i}/completion.json') for i in (0, 1)]
    require(submission.get('registry', {}).get('completion_files') == expected_files, 'Submitted job root mismatch')
    require(root.name == f'job{submission["job_id"]}', 'Job directory ID mismatch')
    results, raw, tensor_receipts = {}, {}, {}
    for index in (0, 1):
        path = root / f'prompt{index}'
        endpoint, prompt, accepted = load_verified_inputs(protocol, index)
        expected_state, state_meta = artificial_state(endpoint, index)
        manifest = read_json(path / 'manifest.json')
        validate_manifest(manifest, protocol, digest, submission, index, prompt, state_meta, accepted)
        input_tensor, input_receipt = verify_tensor(path / 'input.pt', 'input', STATE_SHAPE, torch.bfloat16,
            file_digest=manifest['input_file_sha256'], tensor_digest=state_meta['input_sha256'])
        require(torch.equal(input_tensor, expected_state), 'Input differs from reconstructed artificial state')
        completion, validation = read_json(path / 'completion.json'), read_json(path / 'validation.json')
        require(completion == validation, 'Completion/validation receipt mismatch')
        records = {arm: read_json(path / f'{arm}.json') for arm in ARMS}
        outputs, output_receipts = {}, {}
        for arm, record in records.items():
            outputs[arm], output_receipts[arm] = verify_tensor(path / f'{arm}_output.pt', 'output',
                STATE_SHAPE, torch.bfloat16, file_digest=record['output_file_sha256'], tensor_digest=record['output_sha256'])
            seconds = record.get('forward_seconds_including_hook_diagnostics')
            require(isinstance(seconds, (float, int)) and math.isfinite(seconds) and seconds >= 0, 'Invalid arm timing')
        contrasts = validate_completion(completion, outputs, records)
        repeat = read_json(path / 'repeat_validation.json')
        compare_numbers(repeat, {'output': contrasts['exact_repeat'], 'replayed_gain_equal': True}, 'Repeat receipt')
        observers = verify_observers(path, records)
        raw[f'prompt{index}'] = {'manifest.json': manifest, 'completion.json': completion,
                                'validation.json': validation, 'repeat_validation.json': repeat,
                                **{f'{arm}.json': records[arm] for arm in ARMS}}
        tensor_receipts[f'prompt{index}'] = {'input': input_receipt, 'outputs': output_receipts,
                                          'observers': observers}
        results[f'prompt{index}'] = {'prompt': prompt, 'synthetic_state': state_meta, 'contrasts': contrasts,
                                    'continuous_measures': continuous_measures(outputs, contrasts),
                                    'gain_change_float32': completion['gain_change_float32'],
                                    'gain_change_bfloat16': completion['gain_change_bfloat16'],
                                    'gain_quantization_dead': completion['gain_quantization_dead'],
                                    'runtime_feature_geometry': records['shifted_g1']['centered_feature_error'],
                                    'timing': {'model_loading_seconds': manifest['model_loading_seconds'],
                                      'five_forwards_including_hook_diagnostics_seconds': sum(
                                          v['forward_seconds_including_hook_diagnostics'] for v in records.values())}}
    manifests = [raw[f'prompt{i}']['manifest.json'] for i in (0, 1)]
    for key in ('model', 'source_sha256', 'torch', 'cuda', 'diffusers', 'model_parameter_dtypes'):
        require(manifests[0][key] == manifests[1][key], f'Cross-prompt {key} mismatch')
    require(manifests[0]['conditioning_sha256']['img_ids'] == manifests[1]['conditioning_sha256']['img_ids'],
            'Cross-prompt image position conditioning receipt mismatch')
    return {'purpose': 'Two-prompt artificial-state observer audit; descriptive causal contrasts only',
            'on_trajectory': False, 'efficacy_evaluated': False, 'novelty_established': False,
            'automatic_promotion': False, 'job_id': str(submission['job_id']), 'protocol_sha256': digest,
            'completed_prompts': [0, 1], 'total_transformer_forwards': 10,
            'equality_semantics': 'Exact original replay requires equal tensor bytes/hashes. Other worker-compatible bitwise_equal contrast fields use torch.equal numerical equality, which treats signed zeros as equal.',
            'independent_verification': ['CR17 accepted inputs/endpoint hashes and receipts',
                'CPU RNG and artificial-state reconstruction', 'saved BF16 inputs and velocity outputs',
                'exact original output repeat', 'observer/target tensors and spectra with explicit CPU/CUDA tolerance',
                'gain-vector hashes, BF16 casts and reported crossing consistency', 'all saved output contrasts'],
            'runtime_receipts_not_independently_recomputed': [
                'Text conditioning tensor hashes: full conditioning tensors were not saved.',
                'Full projected-feature equality, cached-feature hashes and transverse geometry: full features were not saved.',
                'Gain-consumption hashes, Q/K dtype and attention call count: checked at runtime by frozen instrumentation.',
                'OfficialSEGA allocation computation is not rerun from full projected features, which were not saved.'],
            'limitations': ['Artificial endpoint/noise mixture at sigma .5, not a captured trajectory state.',
                'Two previously used development prompts, one state each; no quality metrics or confirmation set.',
                'One exact original repeat proves repeatability of that pair only.',
                'Fixed-gain BF16/architecture residual is measured, not assumed to be pure rounding.',
                'Timing includes hook diagnostics and some serialization, excludes conditioning encoding and unmeasured I/O; not a benchmark.'],
            'results': results, 'tensor_verification': tensor_receipts, 'raw_json': raw,
            'inputs': {'protocol': protocol, 'submission': submission},
            'input_file_sha256': {str(Path(protocol_path).resolve()): digest,
                                  str(Path(submission_path).resolve()): sha256(submission_path)},
            'reporter_source_sha256': sha256(__file__)}


def archive_report(report, jobroot, out):
    out, root = Path(out), Path(jobroot)
    require(not out.exists(), 'Archive must be a fresh directory')
    require(str(out.resolve()).startswith('/work/'), 'JSON crux archive must be on work storage')
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    source_hashes = {str((root / prompt / name).resolve()): sha256(root / prompt / name)
                     for prompt, files in report['raw_json'].items() for name in files}
    out.mkdir(parents=True)
    (out / 'report.json').write_text(encoded)
    for prompt, files in report['raw_json'].items():
        folder = out / prompt
        folder.mkdir()
        for name, data in files.items():
            (folder / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    receipt = {'report_sha256': sha256(out / 'report.json'), 'reporter_source_sha256': sha256(__file__),
               'source_json_sha256': source_hashes, 'bulk_tensors_copied': False}
    (out / 'archive_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobroot', type=Path)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = build_report(args.jobroot, args.protocol, args.submission)
    print(json.dumps(archive_report(report, args.jobroot, args.out), indent=2))


if __name__ == '__main__':
    main()
