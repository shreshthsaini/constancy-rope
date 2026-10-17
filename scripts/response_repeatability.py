"""Locate strict-history failure without changing the frozen CR15 experiment."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crope.utils import atomic_json, read_prompts, runtime_environment
from response_probe import check_frozen_sources, select_directions, tensor_hash

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')


def differences(a, b, path=''):
    """Keep every differing scalar path, including exact discrepancy magnitude."""
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            result += differences(a.get(key), b.get(key), f'{path}/{key}')
        return result
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [row for i, (x, y) in enumerate(zip(a, b))
                for row in differences(x, y, f'{path}/{i}')]
    if a == b:
        return []
    numeric = type(a) in (int, float) and type(b) in (int, float)
    return [{'path': path, 'a': a, 'b': b, 'abs_difference': abs(a-b) if numeric else None}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if SCRATCH not in args.out.resolve().parents:
        raise ValueError('Use project scratch for diagnostic outputs')
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError('Fresh output directory required')
    protocol_file = ROOT / '.agent/notes/response_repeatability_protocol.json'
    protocol = json.loads(protocol_file.read_text())
    for name, expected in protocol['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Repeatability source changed: {name}')
    runtime_environment()
    import torch
    import wandb
    from crope.response import SameStateProbe, band_directions
    from crope.runtime import FluxExperiment, load_pipeline
    from crope.sega_official import OfficialSEGA
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated GPU required')
    frozen = check_frozen_sources()
    original_protocol = ROOT / '.agent/notes/response_probe_protocol.json'
    if hashlib.sha256(original_protocol.read_bytes()).hexdigest() != protocol['cr15_protocol_sha256']:
        raise ValueError('Frozen CR15 protocol changed')
    source = SCRATCH / 'prompts/aesthetic4k_eval.jsonl'
    if hashlib.sha256(source.read_bytes()).hexdigest() != protocol['prompt_sha256']:
        raise ValueError('Prompt source changed')
    if Path(os.environ['CROPE_TRANSFORMER']).name != protocol['transformer_revision']:
        raise ValueError('Pinned Krea transformer required')
    row = read_prompts(source)[args.prompt_index]
    manifest = {'purpose': 'Strict-history repeatability diagnosis; not efficacy evidence',
                'protocol_sha256': hashlib.sha256(protocol_file.read_bytes()).hexdigest(),
                'protocol': protocol, 'prompt_index': args.prompt_index, 'prompt': row,
                'job_id': os.environ.get('SLURM_JOB_ID'), 'frozen_sources': frozen,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name()}
    atomic_json(args.out / 'manifest.json', manifest)
    run = wandb.init(project='constancy-rope', group='response-repeatability',
                     name=f'repeatability-{args.prompt_index}', job_type='diagnostic',
                     dir=str(SCRATCH / 'wandb'), config=manifest,
                     mode=os.environ.get('WANDB_MODE', 'offline'))
    exp = None
    try:
        pipe = load_pipeline('cuda', False, False)
        exp = FluxExperiment(pipe, 'sega_official', (2048, 2048))
        pipe.transformer.eval()
        if exp.rope.method != 'sega_official':
            raise ValueError('Frozen SEGA RoPE required')
        configuration = exp.config(row, 0, 4, 4.5)
        configuration.pop('attention', None)
        manifest.update(configuration=configuration,
                        base_snapshot=pipe.config['_name_or_path'],
                        attention='Existing SDPA policy; kernel selection not instrumented')
        atomic_json(args.out / 'manifest.json', manifest)
        captured, observations, results = [], [], {}
        def capture(module, inputs, projected):
            if not captured:
                captured.append(projected.detach().clone())
        def observe(step, sigma, xt, x0):
            observations.append({'step': step, 'sigma': sigma,
                'input_sha256': tensor_hash(xt), 'clean_estimate_sha256': tensor_hash(x0),
                'gain_float32': exp.controller.qk_scale.float().cpu().tolist(),
                'gain_bfloat16': exp.controller.qk_scale.bfloat16().float().cpu().tolist()})
        def velocity(module, inputs, kwargs, output):
            observations[-1]['velocity_sha256'] = tensor_hash(output[0])
        h = pipe.transformer.x_embedder.register_forward_hook(capture)
        v = pipe.transformer.register_forward_hook(velocity, with_kwargs=True)
        try:
            for name in ('baseline_a', 'baseline_b', 'probed_c', 'baseline_d'):
                observations = []
                probe_error = None
                try:
                    if name == 'probed_c':
                        with SameStateProbe(exp, [1], select_directions(band_directions(exp)),
                                .1, lambda r: atomic_json(args.out / 'probe_step.json', r)) as probe:
                            output, seconds, memory = exp.generate(row, 0, 4, 4.5,
                                                                   output_type='latent', observer=observe)
                        extra_calls = probe.extra_calls
                    else:
                        output, seconds, memory = exp.generate(row, 0, 4, 4.5,
                                                               output_type='latent', observer=observe)
                        extra_calls = 0
                except RuntimeError as exc:
                    if name != 'probed_c':
                        raise
                    probe_error = str(exc)
                    output, seconds, memory, extra_calls = None, None, None, probe.extra_calls
                result = {'history': deepcopy(exp.controller.history),
                          'observations': observations, 'seconds': seconds,
                          'peak_memory_bytes': memory, 'extra_calls': extra_calls,
                          'error': probe_error, 'endpoint_sha256': None}
                if output is not None:
                    result['endpoint_sha256'] = tensor_hash(output)
                    torch.save(output.detach().cpu(), args.out / f'{name}_endpoint.pt')
                results[name] = result
                atomic_json(args.out / f'{name}.json', result)
                run.log({'run_index': len(results), 'extra_calls': extra_calls,
                         'runtime_error': int(probe_error is not None)})
        finally:
            h.remove(); v.remove()
        # Directly repeat only SEGA's feature-to-gain arithmetic on identical data.
        # This separate controller never feeds gains into the sampler.
        isolated = {}
        old_determinism = torch.are_deterministic_algorithms_enabled()
        try:
            for deterministic in (False, True):
                torch.use_deterministic_algorithms(deterministic)
                controller = OfficialSEGA(exp.grid, exp.axes_dims)
                trials = []
                try:
                    for _ in range(8):
                        gains = controller.update(captured[0], 1.0)
                        trials.append({'gain_float32': gains.float().cpu().tolist(),
                                       'gain_bfloat16': gains.bfloat16().float().cpu().tolist(),
                                       'diagnostics': deepcopy(controller.diagnostics)})
                    isolated[str(deterministic)] = {'trials': trials,
                        'differences_from_first': [differences(trials[0], t) for t in trials[1:]]}
                except RuntimeError as exc:
                    isolated[str(deterministic)] = {'error': str(exc), 'trials': trials}
        finally:
            torch.use_deterministic_algorithms(old_determinism)
        atomic_json(args.out / 'isolated_allocation.json', isolated)
        comparisons = {f'{a}_vs_{b}': {
            'history': differences(results[a]['history'], results[b]['history']),
            'observations': differences(results[a]['observations'], results[b]['observations']),
            'endpoint_equal': results[a]['endpoint_sha256'] is not None and
                              results[a]['endpoint_sha256'] == results[b]['endpoint_sha256']}
            for a, b in (('baseline_a', 'baseline_b'), ('baseline_b', 'probed_c'),
                         ('baseline_b', 'baseline_d'))}
        atomic_json(args.out / 'comparison.json', comparisons)
        atomic_json(args.out / 'completion.json', {
            'purpose': 'Diagnostic executed; completion is not successful CR15 validation',
            'comparisons': {k: {'history_differences': len(v['history']),
                              'observation_differences': len(v['observations']),
                              'endpoint_equal': v['endpoint_equal']} for k, v in comparisons.items()},
            'cr15_status': 'Failed under original frozen protocol; no automatic retry/relaxation'})
    finally:
        if exp is not None:
            exp.close()
        run.finish()


if __name__ == '__main__':
    main()
