"""Frozen eight-caption native comparator, observational against existing CR14 4K.

This reuses FluxExperiment and its ordinary PNG/JSON sidecars. At native scale,
released SEGA has base rotary frequencies and unit gains. No new method, shared
noise realization across sizes, or isolated resolution-effect claim is implied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crope.utils import atomic_json, file_sha256, image_seed, image_stem, read_prompts, runtime_environment

SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
BASE = '3de623fc3c33e44ffbe2bad470d0f45bccf2eb21'
KREA = '8162a9c7b05a641be098422bf2fcf335615c2f28'
PINNED = {
    'cr18_protocol': 'b98de615bdb3c475b38f355f35b13fbce50240f614aaac9492ebf803a77374da',
    'cr19_protocol': 'a0fb07d23ad61224f275829741287d4a242e862418002f00f8e052bec1081dea',
    'accepted_cr18_report': '127ef1c2a2b5b27e590a1d80ee7baead59bf6ec9375c59a4f5616dcbef84df49',
    'prompt_file': 'ce10315688d5cfd70824cd88fe9f2332a8bd1f3ccb056d3a1315227408a4f8f1',
}
DEV_IDS = ['a4k_eval_4096_'+suffix for suffix in (
    'minus6873203881450547484', '3614061571823823193', '7354534197758042049',
    '788875264758341041', 'minus5023570400257826529', 'minus8041742109062037958',
    '5968998903196670637', 'minus8648158440845565467')]
SCHEDULER_EXPECTATION = {
    'num_train_timesteps': 1000, 'shift': 3., 'use_dynamic_shifting': True,
    'base_shift': .5, 'max_shift': 1.15, 'base_image_seq_len': 256, 'max_image_seq_len': 4096,
    'invert_sigmas': False, 'shift_terminal': None, 'use_karras_sigmas': False,
    'use_exponential_sigmas': False, 'use_beta_sigmas': False,
    'time_shift_type': 'exponential', 'stochastic_sampling': False,
}
FIXED = dict(schema_version=1, status='frozen', development_prompt_indices=list(range(8)),
    chunks=[[0, 4], [4, 8]], method='sega_official', resolution=[1024, 1024], steps=28,
    guidance=4.5, seed=0, dtype='bfloat16', scope='observational_native_vs_existing_4k',
    noise_coupling='uncoupled_across_resolutions', new_method=False)
REQUIRED_SOURCES = {'scripts/native_comparator.py', 'tests/test_native_comparator.py',
    'crope/runtime.py', 'crope/gain.py', 'crope/rope.py', 'crope/sega_official.py',
    'crope/utils.py', 'scripts/sample.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_pin(entry):
    require(isinstance(entry, dict) and set(entry) == {'path', 'sha256'} and
            isinstance(entry['path'], str) and Path(entry['path']).is_absolute() and
            valid_digest(entry['sha256']), 'Invalid absolute file pin')


def pinned_json(entry):
    require(file_sha256(entry['path']) == entry['sha256'], 'Pinned JSON changed: '+entry['path'])
    return read_json(entry['path'])


def validate_protocol(protocol, start, end):
    require(type(start) is int and type(end) is int and (start, end) in ((0, 4), (4, 8)),
            'Only complete chunks [0,4) and [4,8) are permitted')
    for key, value in FIXED.items():
        require(protocol.get(key) == value, 'Protocol mismatch: '+key)
    for key in (*PINNED, 'cr14_model_manifest', 'rubric'):
        validate_pin(protocol.get(key))
        if key in PINNED:
            require(protocol[key]['sha256'] == PINNED[key], 'Wrong accepted dependency: '+key)
    selected = protocol.get('selected_prompts', [])
    require(len(selected) == 8, 'Exactly eight development captions required')
    for index, row in enumerate(selected):
        require(row.get('index') == index and row.get('id') == DEV_IDS[index] and
                isinstance(row.get('prompt'), str) and bool(row['prompt'].strip()) and
                row.get('derived_seed') == image_seed(DEV_IDS[index], 0), 'Development prompt identity/order changed')
        for key in ('reference_png', 'reference_metadata'):
            validate_pin(row.get(key))
    sources = protocol.get('source_sha256', {})
    require(REQUIRED_SOURCES <= set(sources), 'Native worker sources are not frozen')
    require(all(not Path(name).is_absolute() and '..' not in Path(name).parts and valid_digest(digest)
                for name, digest in sources.items()), 'Invalid source hash')
    require(protocol.get('scheduler_expectation') == SCHEDULER_EXPECTATION, 'Native scheduler expectation changed')
    sigmas = protocol.get('expected_sigmas', [])
    require(len(sigmas) == 28 and all(type(x) in (int, float) and math.isfinite(x) for x in sigmas)
            and sigmas[0] == 1. and all(0 < b < a <= 1 for a, b in zip(sigmas, sigmas[1:])),
            'Expected complete decreasing executed sigma schedule')


def verify_sources(sources):
    for name, digest in sources.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and valid_digest(digest)
                and file_sha256(ROOT/name) == digest, 'Frozen source changed: '+name)


def verify_png(path, expected_shape, digest):
    from PIL import Image
    require(file_sha256(path) == digest, 'PNG hash mismatch')
    with Image.open(path) as image:
        require(image.format == 'PNG' and image.size == tuple(expected_shape[::-1]) and image.mode == 'RGB',
                'PNG shape, mode or format mismatch')
        image.verify()


def validate_dependencies(protocol, start, end):
    """Validate all eight old references and frozen source sets before any model load."""
    validate_protocol(protocol, start, end)
    verify_sources(protocol['source_sha256'])
    pinned_json(protocol['rubric'])
    cr18, cr19 = (pinned_json(protocol[name]) for name in ('cr18_protocol', 'cr19_protocol'))
    require(len(cr18['source_sha256']) == 16 and len(cr19['source_sha256']) == 15,
            'Accepted frozen source sets changed')
    verify_sources(cr18['source_sha256'])
    verify_sources(cr19['source_sha256'])
    prior = pinned_json(protocol['accepted_cr18_report'])
    require(prior['completed_prompts'] == [0, 1] and prior['total_transformer_forwards'] == 10
            and prior['protocol_sha256'] == PINNED['cr18_protocol'], 'Unaccepted CR18 report')
    accepted = prior['raw_json']['prompt0']['manifest.json']
    require(accepted['model'] == cr18['model'] and accepted['model']['base_revision'] == BASE and
            accepted['model']['transformer_revision'] == KREA, 'Model provenance mismatch')
    old_model = pinned_json(protocol['cr14_model_manifest'])
    require(old_model['transformer_revision'] == KREA and
            old_model['transformer_config_sha256'] == accepted['model']['transformer_config_sha256'] and
            old_model['base_pipeline'] == 'black-forest-labs/FLUX.1-dev', 'CR14 model manifest mismatch')
    require(file_sha256(protocol['prompt_file']['path']) == PINNED['prompt_file'], 'Prompt file changed')
    rows = read_prompts(protocol['prompt_file']['path'])[:8]
    for index, (row, selected) in enumerate(zip(rows, protocol['selected_prompts'])):
        require(row['id'] == selected['id'] and row['prompt'] == selected['prompt'], 'Pinned caption changed')
        metadata = pinned_json(selected['reference_metadata'])
        png = Path(selected['reference_png']['path'])
        require(png.stem == image_stem(row['id'], 0) and
                Path(selected['reference_metadata']['path']) == png.with_suffix('.json'), 'Reference filename mismatch')
        verify_png(png, [4096, 4096], selected['reference_png']['sha256'])
        require(metadata['complete'] is True and metadata['png_sha256'] == selected['reference_png']['sha256']
                and metadata['sigmas'] == protocol['expected_sigmas'], 'CR14 reference completion/schedule mismatch')
        config = metadata['config']
        expected = dict(method='sega_official', res='4096x4096', prompt_id=row['id'], prompt=row['prompt'],
            seed=0, derived_seed=image_seed(row['id'], 0), steps=28, guidance=4.5,
            model_revision=BASE, dtype='bfloat16')
        require(all(config.get(k) == v for k, v in expected.items()), 'CR14 reference settings mismatch')
        require(config['torch'] == accepted['torch'] and config['diffusers'] == accepted['diffusers'],
                'CR14 and CR18 software mismatch')
    require(len(rows) == 8, 'Missing development captions')
    return rows, accepted


def validate_output(png, metadata_path, row, protocol, expected_config):
    """Independently reopen the ordinary sample sidecar and PNG before completion."""
    from crope.runtime import fingerprint
    record = read_json(metadata_path)
    require(record.get('complete') is True and record.get('config') == expected_config and
            record.get('config_fingerprint') == fingerprint(expected_config), 'Output config/fingerprint mismatch')
    config = record['config']
    expected = dict(method='sega_official', res='1024x1024', prompt_id=row['id'], prompt=row['prompt'],
        seed=0, derived_seed=image_seed(row['id'], 0), steps=28, guidance=4.5,
        dtype='bfloat16', grid=[64, 64], side_scale=[1., 1.], area_scale=1.)
    require(all(config.get(k) == v for k, v in expected.items()), 'Native output identity/settings mismatch')
    require(all(config['scheduler'].get(k) == v for k, v in SCHEDULER_EXPECTATION.items()), 'Output scheduler mismatch')
    require(Path(record['image']).resolve() == png.resolve() and png.stem == image_stem(row['id'], 0)
            and metadata_path == png.with_suffix('.json'), 'Output file identity mismatch')
    verify_png(png, [1024, 1024], record['png_sha256'])
    require(record['sigmas'] == protocol['expected_sigmas'], 'Executed native sigmas differ from frozen expectation')
    history = record.get('gains', [])
    require(len(history) == 28, 'Incomplete native gain history')
    for index, step in enumerate(history):
        require(step['step'] == index and step['sigma'] == protocol['expected_sigmas'][index]
                and step['mode'] == 'sega_official', 'Gain history step/sigma mismatch')
        require(step['log_g'] == step['controller_log_g'] == [0.]*128 and
                step['log_g_axes'] == {'index': [0.]*16, 'height': [0.]*56, 'width': [0.]*56},
                'Native gains are not exactly one')
        official = step['official']
        require(all(official.get(k) == v for k, v in {'reference_qk_gain': 1., 'ntk_factor': 1.,
                'spatial_scale': 1., 'spectral_spread': 0., 'sigma': step['sigma']}.items()),
                'Native SEGA diagnostics mismatch')
        require(math.isfinite(step['model_timestep']), 'Nonfinite model timestep')
    require(math.isfinite(record['seconds']) and record['seconds'] >= 0, 'Invalid generation timing')
    return {'prompt_id': row['id'], 'prompt': row['prompt'], 'seed': 0,
            'derived_seed': image_seed(row['id'], 0), 'steps': 28, 'guidance': 4.5,
            'shape': [1024, 1024, 3], 'png': str(png.resolve()), 'png_sha256': file_sha256(png),
            'metadata': str(metadata_path.resolve()), 'metadata_sha256': file_sha256(metadata_path),
            'generation_seconds': record['seconds'], 'all_native_gains_one': True,
            'executed_sigmas': record['sigmas']}


class NativeGainAudit:
    """After the existing allocator hook, check FP32/BF16 controller and rotary gains."""
    def __init__(self, experiment):
        self.experiment, self.records = experiment, []

    def __call__(self, module, args, output):
        import torch
        e = self.experiment
        gain = e.controller.qk_scale.detach()
        embedded = e.rope.embedding_gain(device=gain.device)
        require(e.step == len(self.records) and gain.shape == embedded.shape == (128,), 'Native gain audit order/shape mismatch')
        require(gain.dtype == torch.float32 and torch.equal(gain, torch.ones_like(gain)) and
                torch.equal(gain.bfloat16(), torch.ones_like(gain.bfloat16())) and
                torch.equal(embedded, torch.ones_like(embedded)), 'Non-unit native gain before attention')
        self.records.append({'step': e.step, 'sigma': e.sigma, 'controller_float32_all_one': True,
                             'controller_bfloat16_all_one': True, 'rotary_gain_all_one': True})


class InitialInputAudit:
    """Save first packed transformer input, without asserting it is raw RNG noise."""
    def __init__(self):
        self.path, self.receipt = None, None

    def __call__(self, module, args, kwargs):
        import torch
        if self.receipt is not None:
            return
        value = kwargs.get('hidden_states', args[0] if args else None)
        require(isinstance(value, torch.Tensor) and value.shape == (1, 4096, 64) and
                value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()), 'Initial transformer input mismatch')
        value = value.detach().cpu().contiguous()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'initial_transformer_input': value}, self.path)
        self.receipt = {'path': str(self.path.resolve()), 'file_sha256': file_sha256(self.path),
            'tensor_sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
            'shape': [1, 4096, 64], 'dtype': 'torch.bfloat16',
            'label': 'initial_transformer_input',
            'scope': 'Saved actual first packed transformer input; not a claim of raw RNG noise or cross-resolution coupling.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    runtime_environment()
    import torch
    torch.set_num_threads(2)
    protocol = read_json(args.protocol)
    rows, accepted = validate_dependencies(protocol, args.start, args.end)
    require(torch.cuda.is_available(), 'Allocated CUDA GPU required; no CPU model execution')
    require(not args.out.exists() and args.out.resolve().is_relative_to(SCRATCH), 'Fresh scratch output directory required')
    require(not os.environ.get('CROPE_ROPE'), 'CROPE_ROPE override is prohibited')
    model = accepted['model']
    transformer_path = Path(os.environ['CROPE_TRANSFORMER'])
    hub = Path(os.environ['HF_HOME'])/'hub/models--black-forest-labs--FLUX.1-dev'
    base_snapshot = hub/'snapshots'/BASE
    require(transformer_path.name == KREA and file_sha256(transformer_path/'transformer/config.json') ==
            model['transformer_config_sha256'], 'Transformer snapshot/config mismatch')
    require((hub/'refs/main').read_text().strip() == BASE and file_sha256(base_snapshot/'model_index.json') ==
            model['base_model_index_sha256'], 'Base snapshot/config mismatch')
    import diffusers
    require(torch.__version__ == accepted['torch'] and torch.version.cuda == accepted['cuda'] and
            diffusers.__version__ == accepted['diffusers'], 'Software differs from accepted CR18')
    from crope.runtime import FluxExperiment, load_pipeline
    import wandb
    args.out.mkdir(parents=True)
    manifest = {'protocol_sha256': file_sha256(args.protocol), 'source_sha256': protocol['source_sha256'],
        'start': args.start, 'end': args.end, 'prompt_ids': [r['id'] for r in rows[args.start:args.end]],
        'model': model, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'diffusers': diffusers.__version__,
        'scope': FIXED['scope'], 'noise_coupling': FIXED['noise_coupling'], 'new_method': False,
        'seed_device': 'cpu', 'seed_policy': 'crope.utils.image_seed(prompt_id,0)',
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True).strip()}
    atomic_json(args.out/'manifest.json', manifest)
    run, experiment, gain_handle, input_handle = None, None, None, None
    try:
        atomic_json(args.out/'progress.json', {'status': 'loading_model', 'completed_images': 0})
        mode = os.environ.get('WANDB_MODE', 'online' if os.environ.get('WANDB_API_KEY') else 'offline')
        try:
            run = wandb.init(project='constancy-rope', group='native-comparator', job_type='generation',
                name=f'native-{args.start}-{args.end}', dir=str(SCRATCH/'wandb'), config=manifest, mode=mode)
        except Exception:
            if mode == 'offline':
                raise
            run = wandb.init(project='constancy-rope', group='native-comparator', job_type='generation',
                name=f'native-{args.start}-{args.end}', dir=str(SCRATCH/'wandb'), config=manifest, mode='offline')
        tick = time.perf_counter()
        pipe = load_pipeline('cuda', False, False)
        require(Path(pipe.config['_name_or_path']).resolve() == base_snapshot.resolve(), 'Loaded base revision mismatch')
        require(len(pipe.transformer.transformer_blocks) == 19 and len(pipe.transformer.single_transformer_blocks) == 38 and
                pipe.transformer.config.in_channels == 64 and tuple(pipe.transformer.config.axes_dims_rope) == (16, 56, 56),
                'Unexpected FLUX geometry')
        require(sorted({str(p.dtype) for p in pipe.transformer.parameters()}) == accepted['model_parameter_dtypes'],
                'Model parameter dtypes changed')
        pipe.transformer.eval()
        experiment = FluxExperiment(pipe, 'sega_official', (1024, 1024))
        require(all(pipe.scheduler.config.get(k) == v for k, v in SCHEDULER_EXPECTATION.items()), 'Loaded scheduler mismatch')
        manifest['model_loading_seconds'] = time.perf_counter()-tick
        manifest['native_rope'] = experiment.rope.metadata()
        atomic_json(args.out/'manifest.json', manifest)
        audit = NativeGainAudit(experiment)
        gain_handle = pipe.transformer.x_embedder.register_forward_hook(audit)
        input_audit = InitialInputAudit()
        input_handle = pipe.transformer.register_forward_pre_hook(input_audit, with_kwargs=True)
        images = args.out/'sega_official'/'1024x1024'
        records = []
        for index in range(args.start, args.end):
            row = rows[index]
            audit.records.clear()
            stem = image_stem(row['id'], 0)
            input_audit.path, input_audit.receipt = images/f'{stem}_initial_transformer_input.pt', None
            atomic_json(args.out/'progress.json', {'status': 'generating', 'prompt_index': index, 'completed_images': len(records)})
            expected_config = experiment.config(row, 0, 28, 4.5)
            experiment.sample(row, images, 0, 28, 4.5)
            require(len(audit.records) == 28 and [r['sigma'] for r in audit.records] == protocol['expected_sigmas'],
                    'Incomplete runtime gain/sigma audit')
            record = validate_output(images/f'{stem}.png', images/f'{stem}.json', row, protocol, expected_config)
            require(input_audit.receipt is not None, 'Missing initial transformer input')
            records.append({'prompt_index': index, **record, 'runtime_gain_audit': list(audit.records),
                            'initial_transformer_input': input_audit.receipt})
            atomic_json(args.out/'progress.json', {'status': 'image_completed', 'prompt_index': index, 'completed_images': len(records)})
            run.log({'completed_images': len(records), 'generation_seconds': record['generation_seconds']})
        require(len(records) == 4, 'Chunk incomplete')
        atomic_json(args.out/'completion.json', {'protocol_sha256': file_sha256(args.protocol), 'start': args.start,
            'end': args.end, 'completed_images': 4, 'records': records, 'scope': FIXED['scope'],
            'noise_coupling': FIXED['noise_coupling'], 'new_method': False, 'quality_scored': False,
            'interpretation': 'Native comparator versus existing4K is observational. Equal seed labels do not couple differently shaped noise. No isolated causal resolution or method-gain claim.'})
        atomic_json(args.out/'progress.json', {'status': 'completed', 'completed_images': 4})
    except Exception as error:
        atomic_json(args.out/'failure.json', {'error_type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if gain_handle is not None:
            gain_handle.remove()
        if input_handle is not None:
            input_handle.remove()
        if experiment is not None:
            experiment.close()
        if run is not None:
            run.finish()


if __name__ == '__main__':
    main()
