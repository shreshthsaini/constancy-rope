"""Bounded same-state SEGA response diagnostic on development prompts only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crope.utils import atomic_json, read_prompts, resolution, runtime_environment

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
KREA_REVISION = '8162a9c7b05a641be098422bf2fcf335615c2f28'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt-index', type=int, required=True, choices=range(8))
    p.add_argument('--res', type=resolution, default=(4096, 4096))
    p.add_argument('--steps', type=int, default=28)
    p.add_argument('--probe-steps', default='6,14,22')
    p.add_argument('--epsilon', type=float, default=.1)
    p.add_argument('--out', type=Path, required=True)
    return p


def select_directions(directions):
    """One prespecified middle octave per axis and the spatial scalar control."""
    chosen = []
    for axis in (1, 2):
        axial = [d for d in directions if d.preserve_axes == (axis,)]
        if not axial:
            raise ValueError('No nontrivial spatial octave direction')
        chosen.append(axial[len(axial) // 2])
    return chosen + [d for d in directions if d.name == 'spatial_scalar']


def check_frozen_sources():
    frozen = json.loads((ROOT / '.agent/notes/continuation3/frozen_manifest.json').read_text())
    checked = {}
    for name, expected in frozen['source_sha256'].items():
        path = Path(name)
        if ROOT not in path.parents:
            continue  # Spool files move during CR14; do not inspect task ownership here.
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Frozen CR14 source changed: {path}')
        checked[name] = actual
    return checked


def tensor_hash(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8)
                          .numpy().tobytes()).hexdigest()


def check_response_protocol(args):
    path = ROOT / '.agent/notes/response_deterministic_protocol.json'
    protocol = json.loads(path.read_text())
    for name, expected in protocol['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Response protocol source changed: {name}')
    modes = [protocol['smoke'], protocol['diagnostic']]
    selected = [int(v) for v in args.probe_steps.split(',')]
    if (args.prompt_index not in protocol['development_prompt_indices']
            or args.epsilon != protocol['epsilon']
            or not any(tuple(args.res) == (m['resolution'], m['resolution'])
                       and args.steps == m['steps'] and selected == m['probe_steps'] for m in modes)):
        raise ValueError('Requested run differs from frozen response protocol')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args = parser().parse_args()
    selected_steps = [int(v) for v in args.probe_steps.split(',')]
    if not selected_steps or len(set(selected_steps)) != len(selected_steps) or any(
            step < 0 or step >= args.steps for step in selected_steps):
        raise ValueError('Probe step indices must be unique and within the sampling schedule')
    protocol_hash = check_response_protocol(args)
    if SCRATCH not in args.out.resolve().parents:
        raise ValueError('Bulk probe outputs must be under project scratch')
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError('Use a fresh output directory; interrupted evidence is preserved')
    runtime_environment()
    import torch
    import wandb
    from crope.response import SameStateProbe, band_directions
    from crope.deterministic_allocation import DeterministicAllocationSEGA
    from crope.runtime import FluxExperiment, load_pipeline
    frozen_sources = check_frozen_sources()
    transformer = Path(os.environ['CROPE_TRANSFORMER'])
    if transformer.name != KREA_REVISION:
        raise ValueError('Pinned Krea transformer required')
    prompt_file = SCRATCH / 'prompts/aesthetic4k_eval.jsonl'
    prompt_hash = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
    frozen = json.loads((ROOT / '.agent/notes/continuation3/frozen_manifest.json').read_text())
    if prompt_hash != frozen['source_sha256'][str(prompt_file)]:
        raise ValueError('Frozen development prompt source changed')
    row = read_prompts(prompt_file)[args.prompt_index]
    if not torch.cuda.is_available():
        raise RuntimeError('Real model probe requires an allocated GPU')
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (ROOT / 'crope/response.py', Path(__file__).resolve(), ROOT / 'crope/deterministic_allocation.py')}
    manifest = {'purpose': 'Same-state response diagnostic, not a candidate efficacy result',
                'prompt_index': args.prompt_index, 'prompt': row, 'seed': 0, 'guidance': 4.5,
                'res': args.res, 'steps': args.steps, 'probe_steps': selected_steps,
                'epsilon': args.epsilon, 'source_sha256': source_hashes,
                'frozen_cr14_source_sha256': frozen_sources,
                'transformer_revision': transformer.name,
                'transformer_config_sha256': hashlib.sha256((transformer / 'transformer/config.json').read_bytes()).hexdigest(),
                'prompt_file_sha256': prompt_hash,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                                                     '--format=csv,noheader'], text=True).strip(),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    atomic_json(args.out / 'manifest.json', manifest)
    run = wandb.init(project='constancy-rope', group='same-state-response',
                     name=f'response-{args.prompt_index}-{args.res[0]}', job_type='diagnostic',
                     dir=str(SCRATCH / 'wandb'), config=manifest,
                     mode=os.environ.get('WANDB_MODE', 'online' if os.environ.get('WANDB_API_KEY') else 'offline'))
    experiment = None
    try:
        hub = Path(os.environ['HF_HOME']) / 'hub/models--black-forest-labs--FLUX.1-dev'
        base_revision = (hub / 'refs/main').read_text().strip()
        base_snapshot = hub / 'snapshots' / base_revision
        pipe = load_pipeline('cuda', False, False)
        if (hub / 'refs/main').read_text().strip() != base_revision:
            raise RuntimeError('Base model cache revision changed during loading')
        if Path(pipe.config['_name_or_path']).resolve() != base_snapshot.resolve():
            raise RuntimeError('Loaded base snapshot differs from recorded source')
        experiment = FluxExperiment(pipe, 'sega_official', args.res)
        experiment.official = DeterministicAllocationSEGA(experiment.grid, experiment.axes_dims)
        if experiment.rope.method != 'sega_official':
            raise ValueError('Response protocol requires sega_official RoPE')
        configuration = experiment.config(row, seed=0, steps=args.steps, guidance=4.5)
        configuration.pop('attention', None)
        configuration['allocation_numerics'] = 'deterministic_sega_update_only'
        configuration['attention_backend'] = 'PyTorch SDPA policy; actual selected kernel not instrumented'
        manifest.update(experiment_configuration=configuration,
                        base_snapshot=str(base_snapshot), base_revision=base_revision,
                        base_model_index_sha256=hashlib.sha256((base_snapshot / 'model_index.json').read_bytes()).hexdigest(),
                        response_protocol_sha256=protocol_hash)
        atomic_json(args.out / 'manifest.json', manifest)
        run.config.update(manifest, allow_val_change=True)
        pipe.transformer.eval()
        baseline, base_seconds, base_memory = experiment.generate(
            row, seed=0, steps=args.steps, guidance=4.5, output_type='latent')
        baseline = baseline.detach().cpu()
        base_history = list(experiment.controller.history)
        atomic_json(args.out / 'baseline_history.json', base_history)
        directions = select_directions(band_directions(experiment))
        def sink(record):
            atomic_json(args.out / f'step_{record["step"]:03d}.json', record)
            run.log({'step': record['step'], 'sigma': record['sigma'],
                     'extra_seconds': record['extra_seconds'], 'extra_calls': record['extra_calls'],
                     'replay_max_abs_error': record['replay_max_abs_velocity_error']})
        with SameStateProbe(experiment, selected_steps, directions, args.epsilon, sink) as probe:
            result, probe_seconds, probe_memory = experiment.generate(
                row, seed=0, steps=args.steps, guidance=4.5, output_type='latent')
        result = result.detach().cpu()
        atomic_json(args.out / 'probed_history.json', experiment.controller.history)
        torch.save({'baseline': baseline, 'probed': result}, args.out / 'trajectory_endpoints.pt')
        receipt = {'baseline_seconds': base_seconds, 'probed_seconds': probe_seconds,
                   'baseline_peak_bytes': base_memory, 'probed_peak_bytes': probe_memory,
                   'extra_forward_calls': probe.extra_calls,
                   'baseline_sha256': tensor_hash(baseline), 'probed_sha256': tensor_hash(result),
                   'trajectory_bitwise_equal': bool(torch.equal(baseline, result)),
                   'baseline_history_equal': base_history == experiment.controller.history,
                   'observed_probe_steps': [r['step'] for r in probe.records],
                   'directions': [d.name for d in directions], 'gpu_probe_executed': True,
                   'sigmas': pipe.scheduler.sigmas.detach().float().cpu().tolist(),
                   'timing_interpretation': 'Diagnostic accounting only; first baseline and later probed run are not warmup-matched performance benchmarks.',
                   'efficacy_evaluated': False}
        atomic_json(args.out / 'validation.json', receipt)
        if not receipt['trajectory_bitwise_equal'] or not receipt['baseline_history_equal']:
            raise RuntimeError('Probe changed the baseline trajectory or gain history')
        if sorted(receipt['observed_probe_steps']) != sorted(selected_steps):
            raise RuntimeError('Requested probe steps did not all execute')
        run.log(receipt)
        atomic_json(args.out / 'completion.json', receipt)
    finally:
        if experiment is not None:
            experiment.close()
        run.finish()


if __name__ == '__main__':
    main()
