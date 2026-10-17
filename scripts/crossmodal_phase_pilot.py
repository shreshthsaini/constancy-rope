"""Bounded two-caption cross-modal phase pilot; no quality claim or scoring here."""
from __future__ import annotations

import argparse
import copy
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
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
from crope.utils import atomic_json, file_sha256, image_seed, image_stem, runtime_environment
from scripts import native_comparator as native

ARMS = ('baseline', 'fixedmass', 'unnormalized')
NATIVE_PROTOCOL_SHA256 = 'ba80b1ddd2ca794a84e2607be6773acf5253f80402ece5ddf71d53b400775a18'
NEW_SOURCES = {'scripts/crossmodal_phase_pilot.py', 'tests/test_crossmodal_phase_pilot.py',
               'crope/crossmodal_phase.py', 'tests/test_crossmodal_phase.py'}
FIXED = dict(schema_version=1, status='frozen', development_prompt_indices=[0, 1],
    arms=list(ARMS), resolution=[4096, 4096], smoke_resolution=[1024, 1024],
    steps=28, smoke_steps=4, guidance=4.5, seed=0, dtype='bfloat16', text_tokens=512,
    native_grid=[64, 64], axes_dims=[16, 56, 56], transformer_blocks=57,
    query_chunk_size=65536, backend='aten._scaled_dot_product_cudnn_attention',
    coordinate_map='endpoint_linear_0_to_63', text_ids='all_zero',
    native_endpoint_identity='byte_exact', deployed_smoke_max_relative_l2=1e-3,
    deployed_smoke_cosine_diagnostic_min=.99999,
    unnormalized_implementation='image_only_and_new_text_lse_mixture',
    scope='two_existing_satisfied_development_controls', quality_scored=False)
require = native.require


def expected_sigmas(steps):
    """Independent scheduler expectation, without pipeline/model construction."""
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(**native.SCHEDULER_EXPECTATION)
    scheduler.set_timesteps(sigmas=np.linspace(1., 1 / steps, steps), mu=1.15)
    return scheduler.sigmas[:-1].float().tolist()


def example_protocol(native_protocol_path):
    """Return an unfrozen template for the parent to finish and freeze separately."""
    path = Path(native_protocol_path).resolve()
    old = native.read_json(path)
    return {**copy.deepcopy(FIXED), 'native_protocol': {'path': str(path), 'sha256': NATIVE_PROTOCOL_SHA256},
        'source_sha256': {name: file_sha256(ROOT / name) for name in sorted(NEW_SOURCES)},
        'expected_sigmas': old['expected_sigmas'], 'expected_smoke_sigmas': expected_sigmas(4)}


def validate_protocol(protocol, prompt_index):
    require(type(prompt_index) is int and prompt_index in (0, 1), 'Only development prompt indices 0 and 1')
    for key, value in FIXED.items():
        require(json.dumps(protocol.get(key), sort_keys=True) == json.dumps(value, sort_keys=True),
                'Pilot protocol mismatch: ' + key)
    native.validate_pin(protocol.get('native_protocol'))
    require(protocol['native_protocol']['sha256'] == NATIVE_PROTOCOL_SHA256, 'Wrong accepted native protocol')
    sources = protocol.get('source_sha256', {})
    require(isinstance(sources, dict) and NEW_SOURCES <= set(sources), 'Missing frozen pilot sources')
    require(all(not Path(name).is_absolute() and '..' not in Path(name).parts and native.valid_digest(digest)
                for name, digest in sources.items()), 'Invalid pilot source hash')
    for key, length in (('expected_sigmas', 28), ('expected_smoke_sigmas', 4)):
        sigmas = protocol.get(key, [])
        require(isinstance(sigmas, list) and len(sigmas) == length and
            all(type(s) in (int, float) and math.isfinite(s) for s in sigmas) and sigmas[0] == 1. and
            all(0 < b < a <= 1 for a, b in zip(sigmas, sigmas[1:])), 'Invalid ' + key)


def validate_dependencies(protocol, prompt_index):
    validate_protocol(protocol, prompt_index)
    native.verify_sources(protocol['source_sha256'])
    old = native.pinned_json(protocol['native_protocol'])
    rows, accepted = native.validate_dependencies(old, 0, 4)
    require(protocol['expected_sigmas'] == old['expected_sigmas'] == expected_sigmas(28),
            'Pilot/accepted/independent sigma schedules differ')
    require(protocol['expected_smoke_sigmas'] == expected_sigmas(4), 'Smoke sigma schedule differs')
    row = rows[prompt_index]
    require(row['id'] == native.DEV_IDS[prompt_index], 'Prompt identity differs')
    return row, accepted


def tensor_hash(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def save_tensor(path, name, value):
    import torch
    value = value.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: value}, path)
    return {'path': str(path.resolve()), 'file_sha256': file_sha256(path),
            'tensor_sha256': tensor_hash(value), 'shape': list(value.shape), 'dtype': str(value.dtype), 'key': name}


class PackedInputAudit(native.InitialInputAudit):
    """Reuse the accepted native audit and extend its shape to the fixed 4K grid."""
    def __init__(self, tokens, expected_sha256=None):
        super().__init__()
        self.tokens, self.expected_sha256 = tokens, expected_sha256

    def __call__(self, module, args, kwargs):
        import torch
        if self.tokens == 4096:
            super().__call__(module, args, kwargs)
            require(self.expected_sha256 is None or self.receipt['tensor_sha256'] == self.expected_sha256,
                    'Initial input differs across matched arms')
            return
        if self.receipt is not None:
            return
        value = kwargs.get('hidden_states', args[0] if args else None)
        require(isinstance(value, torch.Tensor) and value.shape == (1, self.tokens, 64) and
                value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()), 'Initial transformer input mismatch')
        self.receipt = save_tensor(self.path, 'initial_transformer_input', value)
        self.receipt.update(label='initial_transformer_input', scope='Actual first packed transformer input')
        require(self.expected_sha256 is None or self.receipt['tensor_sha256'] == self.expected_sha256,
                'Initial input differs across matched arms')


class GeometryGainAudit:
    def __init__(self, experiment, expected_conditioning=None):
        self.experiment, self.geometry, self.gains = experiment, [], []
        self.initial_ids = None
        self.expected_conditioning = expected_conditioning

    def before(self, module, args, kwargs):
        import torch
        e, grid = self.experiment, self.experiment.grid
        ids, txt = kwargs.get('img_ids'), kwargs.get('txt_ids')
        packed = kwargs.get('hidden_states', args[0] if args else None)
        require(isinstance(packed, torch.Tensor) and packed.shape == (1, math.prod(grid), 64) and
                packed.dtype == torch.bfloat16, 'Packed input geometry/dtype mismatch')
        require(isinstance(txt, torch.Tensor) and txt.shape == (512, 3) and bool(torch.isfinite(txt).all())
                and not bool(torch.count_nonzero(txt)), 'Text IDs must be 512 by 3 and all zero')
        require(isinstance(ids, torch.Tensor) and ids.shape == (math.prod(grid), 3) and
                bool(torch.isfinite(ids).all()), 'Image IDs geometry mismatch')
        require(kwargs['encoder_hidden_states'].shape == (1, 512, 4096) and
                kwargs['pooled_projections'].shape == (1, 768), 'Conditioning geometry mismatch')
        if self.initial_ids is None:
            h, w = grid
            expected = torch.stack((torch.zeros(h*w, device=ids.device),
                torch.arange(h, device=ids.device).repeat_interleave(w),
                torch.arange(w, device=ids.device).repeat(h)), dim=-1).to(ids)
            require(torch.equal(ids, expected), 'Image coordinates differ from declared row-major grid')
            self.initial_ids = ids.detach().clone()
            self.conditioning = {k: tensor_hash(kwargs[k]) for k in
                ('encoder_hidden_states', 'pooled_projections', 'txt_ids', 'img_ids')}
            require(self.expected_conditioning is None or self.conditioning == self.expected_conditioning,
                    'Initial conditioning differs across matched arms')
        else:
            require(torch.equal(ids, self.initial_ids), 'Image IDs changed during trajectory')
        require(e.step == len(self.geometry), 'Unexpected transformer call order')
        self.geometry.append({'step': e.step, 'sigma': e.sigma, 'grid': list(grid), 'text_ids_all_zero': True})

    def projected(self, module, args, output):
        import torch
        e = self.experiment
        gain = e.controller.qk_scale.detach()
        rotary = e.rope.embedding_gain(device=gain.device)
        require(e.step == len(self.gains) and gain.shape == rotary.shape == (128,) and
                gain.dtype == torch.float32 and bool(torch.isfinite(gain).all()) and bool((gain > 0).all())
                and torch.equal(gain[::2], gain[1::2]), 'Gain geometry/order/positivity/pair mismatch')
        require(bool(torch.isfinite(rotary).all()) and bool((rotary > 0).all()), 'Invalid rotary amplitude')
        if e.grid == (64, 64):
            require(torch.equal(gain, torch.ones_like(gain)) and torch.equal(rotary, torch.ones_like(rotary)),
                    'Native gain is not exactly one')
        self.gains.append({'step': e.step, 'sigma': e.sigma, 'qk_scale': gain.cpu().tolist(),
                           'rotary_embedding_gain': rotary.cpu().tolist(), 'pair_equal': True})


def phase_receipt_valid(receipt, steps, arm, grid):
    from crope.crossmodal_phase import NUMERICAL_POLICY
    native_identity = grid == [64, 64]
    expected = dict(mode=arm, grid=grid, text_tokens=512, query_chunk_size=FIXED['query_chunk_size'],
        backend=FIXED['backend'], fallback=None, geometry_id_validations=steps,
        native_identity_baseline_path=native_identity, all_text_keys_including_padding=True)
    require(all(k in receipt and receipt[k] == v for k, v in expected.items()), 'Attention arm/geometry/backend receipt mismatch')
    require(receipt.get('calls') == 57 * steps, 'Incomplete attention call count')
    counts = receipt.get('per_block_calls', {})
    names = {f'transformer_blocks.{i}.attn.processor' for i in range(19)} | {
             f'single_transformer_blocks.{i}.attn.processor' for i in range(38)}
    require(set(counts) == names and set(counts.values()) == {steps}, 'Incomplete per-block attention calls')
    branches = 0 if arm == 'baseline' or native_identity else math.ceil(math.prod(grid)/FIXED['query_chunk_size']) * 57 * steps
    expected_calls = dict(full_joint_calls=57*steps, new_text_calls=branches,
                         old_text_calls=branches if arm == 'fixedmass' else 0,
                         image_only_calls=branches if arm == 'unnormalized' else 0)
    require(receipt.get('kernel_calls') == expected_calls, 'Attention kernel branches differ from declared arm')
    require(receipt.get('numerical_policy') == NUMERICAL_POLICY, 'Attention numerical policy differs from pinned source')
    ranges = receipt.get('mass_ranges', {})
    mass_key = 'old_text_mass' if arm == 'fixedmass' else 'new_text_mass'
    require(set(ranges) == ({mass_key} if branches else set()), 'Missing/unexpected text-mass diagnostic')
    if branches:
        bounds = ranges[mass_key]
        ceiling = float(torch.tensor(NUMERICAL_POLICY['log_mass_upper_tolerance'], dtype=torch.float32).exp()) if arm == 'fixedmass' else 1.
        require(len(bounds) == 2 and all(math.isfinite(x) for x in bounds) and 0 <= bounds[0] <= bounds[1] <= ceiling,
                'Nonfinite or invalid text-mass range')


def validate_trajectory(record, row, arm, protocol, smoke=False, protocol_hash=None):
    from crope.runtime import fingerprint
    steps, resolution = (4, [1024, 1024]) if smoke else (28, [4096, 4096])
    schedule = protocol['expected_smoke_sigmas' if smoke else 'expected_sigmas']
    config = record.get('config', {})
    expected = dict(method=arm, prompt_id=row['id'], prompt=row['prompt'], seed=0,
        derived_seed=image_seed(row['id'], 0), steps=steps, guidance=4.5, dtype='bfloat16',
        res='x'.join(map(str, resolution)), grid=[v//16 for v in resolution], phase_arm=arm,
        phase_source_sha256=protocol['source_sha256']['crope/crossmodal_phase.py'],
        pilot_source_sha256=protocol['source_sha256'])
    require(record.get('complete') is True and all(json.dumps(config.get(k), sort_keys=True) == json.dumps(v, sort_keys=True)
            for k, v in expected.items())
            and record.get('config_fingerprint') == fingerprint(config), 'Trajectory identity/config mismatch')
    require(native.valid_digest(config.get('phase_protocol_sha256')) and
            (protocol_hash is None or config['phase_protocol_sha256'] == protocol_hash), 'Trajectory protocol pin differs')
    require(record.get('sigmas') == schedule and len(record.get('gains', [])) == steps, 'Trajectory sigma/history mismatch')
    require(all(g['step'] == i and g['sigma'] == schedule[i] and g['mode'] == 'sega_official'
                for i, g in enumerate(record['gains'])), 'Gain step/sigma/controller mismatch')
    audit = record.get('runtime_audit', {})
    require([g['sigma'] for g in audit.get('gains', [])] == schedule and
            [g['sigma'] for g in audit.get('geometry', [])] == schedule and
            all(g['text_ids_all_zero'] for g in audit['geometry']), 'Runtime geometry/gain schedule incomplete')
    phase = record.get('phase', {})
    if arm != 'deployed':
        phase_receipt_valid(phase, steps, arm, expected['grid'])
    for key, shape in (('initial_transformer_input', [1, math.prod(expected['grid']), 64]),
                       ('endpoint_latent', [math.prod(expected['grid']), 64])):
        item = record.get(key, {})
        require(item.get('shape') == shape and item.get('dtype') == 'torch.bfloat16' and
                native.valid_digest(item.get('tensor_sha256')) and
                file_sha256(item['path']) == item.get('file_sha256'), 'Missing or corrupt ' + key)
        payload = torch.load(item['path'], map_location='cpu', weights_only=True, mmap=True)
        value = payload.get(key)
        require(set(payload) == {key} and isinstance(value, torch.Tensor) and list(value.shape) == shape and
                value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()) and
                tensor_hash(value) == item['tensor_sha256'], 'Saved tensor disagrees with ' + key + ' receipt')
    require(math.isfinite(record['seconds']) and record['seconds'] >= 0 and
            type(record['peak_memory_bytes']) is int and record['peak_memory_bytes'] > 0, 'Invalid timing/memory')


def validate_arm_set(records, row, protocol, smoke=False, protocol_hash=None):
    arms = ('deployed', *ARMS) if smoke else ARMS
    require(len(records) == len(arms) and [r.get('config', {}).get('method') for r in records] == list(arms),
            'Missing, duplicated or reordered arms')
    for record, arm in zip(records, arms):
        validate_trajectory(record, row, arm, protocol, smoke, protocol_hash)
        if not smoke:
            metadata, png = Path(record['metadata']), Path(record['image'])
            require(file_sha256(metadata) == record['metadata_sha256'] and
                    native.read_json(metadata) == {k: v for k, v in record.items() if k not in ('metadata', 'metadata_sha256')},
                    'Published sidecar differs from completed arm')
            require(metadata == png.with_suffix('.json') and png.stem == image_stem(row['id'], 0),
                    'Published PNG/sidecar identity differs')
            native.verify_png(png, [4096, 4096], record['png_sha256'])
    for key in ('initial_transformer_input',):
        require(len({r[key]['tensor_sha256'] for r in records}) == 1, 'Initial input differs across matched arms')
    require(all(r['runtime_audit']['conditioning_sha256'] == records[0]['runtime_audit']['conditioning_sha256']
                for r in records), 'Initial conditioning differs across matched arms')


def endpoint_comparison(reference, value):
    import torch
    require(reference.shape == value.shape and reference.dtype == value.dtype and
            bool(torch.isfinite(reference).all()) and bool(torch.isfinite(value).all()), 'Invalid endpoint comparison')
    a, b = reference.double().flatten(), value.double().flatten()
    norm_a, norm_b, delta = float(a.norm()), float(b.norm()), float((a-b).norm())
    cosine = float(torch.dot(a, b)/(norm_a*norm_b)) if norm_a*norm_b else (1. if delta == 0 else 0.)
    return {'byte_exact': tensor_hash(reference) == tensor_hash(value),
        'relative_l2': delta / norm_a if norm_a else (0. if delta == 0 else None),
        'max_abs': float((a-b).abs().max()), 'cosine': cosine, 'reference_l2': norm_a}


# This subclass is deliberately in the new worker. The accepted runtime hash stays intact.
from crope.runtime import FluxExperiment, fingerprint
import torch


class PilotExperiment(FluxExperiment):
    def __init__(self, pipe, arm, resolution, protocol, protocol_hash):
        require(arm in ('deployed', *ARMS), 'Unknown phase arm')
        self.arm, self.protocol, self.protocol_hash = arm, protocol, protocol_hash
        self.phase_handle, self.endpoint = None, None
        super().__init__(pipe, 'sega_official', resolution)
        if arm != 'deployed':
            from crope.crossmodal_phase import install_crossmodal_phase
            self.phase_handle = install_crossmodal_phase(pipe.transformer, mode=arm, grid=self.grid,
                controller=self.controller, text_tokens=512, query_chunk_size=FIXED['query_chunk_size'])

    def reset(self, observer=None):
        super().reset(observer)
        if self.phase_handle is not None:
            self.phase_handle.reset()
        self.endpoint = None

    def close(self):
        if self.phase_handle is not None:
            self.phase_handle.close()
            self.phase_handle = None
        super().close()

    def config(self, row, seed, steps, guidance):
        value = super().config(row, seed, steps, guidance)
        value.update(method=self.arm, phase_arm=self.arm,
            attention='deployed_pytorch_sdpa' if self.arm == 'deployed' else FIXED['backend'],
            phase_protocol_sha256=self.protocol_hash,
            phase_source_sha256=self.protocol['source_sha256']['crope/crossmodal_phase.py'],
            pilot_source_sha256=self.protocol['source_sha256'], native_reference_grid=[64, 64],
            phase_coordinate_map=FIXED['coordinate_map'], unnormalized_implementation=FIXED['unnormalized_implementation'])
        return value

    @torch.inference_mode()
    def generate(self, row, seed=0, steps=28, guidance=4.5, output_type='pil', observer=None):
        # Same frozen runtime sequence plus a read-only final scheduler-latent callback.
        self.reset(observer)
        h, w = self.resolution
        if h*w >= 2048*4096:
            self.pipe.vae.enable_tiling()
        generator = torch.Generator(device='cpu').manual_seed(image_seed(row['id'], seed))
        self.pipe.scheduler.register_to_config(max_image_seq_len=max(4096, math.prod(self.grid)))
        def endpoint(pipe, index, timestep, values):
            if index == steps-1:
                self.endpoint = values['latents'].detach().clone()[0]
            return values
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        result = self.pipe(prompt=row['prompt'], height=h, width=w, num_inference_steps=steps,
            guidance_scale=guidance, generator=generator, max_sequence_length=512, output_type=output_type,
            callback_on_step_end=endpoint, callback_on_step_end_tensor_inputs=['latents'])
        torch.cuda.synchronize()
        elapsed, peak = time.perf_counter()-tick, torch.cuda.max_memory_allocated()
        require(self.step == steps and self.endpoint is not None, 'Incomplete trajectory/endpoint callback')
        require(self.endpoint.shape == (math.prod(self.grid), 64) and self.endpoint.dtype == torch.bfloat16
                and bool(torch.isfinite(self.endpoint).all()), 'Invalid final scheduler endpoint')
        return result.images[0], elapsed, peak


def run_arm(pipe, row, arm, protocol, protocol_hash, out, smoke=False, reference=None):
    resolution, steps = ((1024, 1024), 4) if smoke else ((4096, 4096), 28)
    folder = out/'smoke'/arm if smoke else out/'images'/arm/'4096x4096'
    folder.mkdir(parents=True, exist_ok=False)
    stem = image_stem(row['id'], 0)
    experiment = PilotExperiment(pipe, arm, resolution, protocol, protocol_hash)
    audit = GeometryGainAudit(experiment, reference['runtime_audit']['conditioning_sha256'] if reference else None)
    initial = PackedInputAudit(math.prod(experiment.grid),
                              reference['initial_transformer_input']['tensor_sha256'] if reference else None)
    initial.path = folder/(stem+'_initial_transformer_input.pt')
    handles = [pipe.transformer.register_forward_pre_hook(audit.before, with_kwargs=True),
        pipe.transformer.register_forward_pre_hook(initial, with_kwargs=True),
        pipe.transformer.x_embedder.register_forward_hook(audit.projected)]
    try:
        config = experiment.config(row, 0, steps, 4.5)
        image, seconds, peak = experiment.generate(row, 0, steps, 4.5, output_type='latent' if smoke else 'pil')
        endpoint = experiment.endpoint.detach().cpu().contiguous()
        record = dict(complete=True, config=config, config_fingerprint=fingerprint(config),
            seconds=seconds, peak_memory_bytes=peak,
            seconds_scope='Encoding, denoising, endpoint GPU clone and optional VAE; excludes saving and model loading',
            gains=experiment.controller.history, sigmas=[g['sigma'] for g in experiment.controller.history],
            runtime_audit={'geometry': audit.geometry, 'gains': audit.gains, 'conditioning_sha256': audit.conditioning},
            phase=experiment.phase_handle.receipt() if experiment.phase_handle is not None else {'mode': 'deployed'},
            initial_transformer_input=initial.receipt,
            endpoint_latent=save_tensor(folder/(stem+'_endpoint_latent.pt'), 'endpoint_latent', endpoint))
        if not smoke:
            png = folder/(stem+'.png')
            image.save(png, format='PNG')
            record.update(image=str(png.resolve()), png_sha256=file_sha256(png))
            native.verify_png(png, [4096, 4096], record['png_sha256'])
        validate_trajectory(record, row, arm, protocol, smoke, protocol_hash)
        metadata = folder/(stem+'.json')
        atomic_json(metadata, record)
        record.update(metadata=str(metadata.resolve()), metadata_sha256=file_sha256(metadata))
        return record, endpoint
    finally:
        for handle in handles:
            handle.remove()
        experiment.close()


def verify_model_environment(accepted):
    import diffusers
    model = accepted['model']
    require(torch.cuda.is_available(), 'Allocated CUDA GPU required; CPU model execution prohibited')
    require(not os.environ.get('CROPE_ROPE'), 'CROPE_ROPE override prohibited')
    transformer = Path(os.environ['CROPE_TRANSFORMER'])
    hub = Path(os.environ['HF_HOME'])/'hub/models--black-forest-labs--FLUX.1-dev'
    base = hub/'snapshots'/native.BASE
    require(transformer.name == native.KREA and file_sha256(transformer/'transformer/config.json') ==
            model['transformer_config_sha256'], 'Transformer revision/config mismatch')
    require((hub/'refs/main').read_text().strip() == native.BASE and file_sha256(base/'model_index.json') ==
            model['base_model_index_sha256'], 'Base revision/config mismatch')
    require(torch.__version__ == accepted['torch'] and torch.version.cuda == accepted['cuda'] and
            diffusers.__version__ == accepted['diffusers'], 'Software differs from accepted CR18')
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-index', type=int, choices=[0, 1], required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    runtime_environment()
    torch.set_num_threads(2)
    require(not args.out.exists() and args.out.resolve().is_relative_to(native.SCRATCH), 'Fresh scratch output required')
    args.out.mkdir(parents=True)
    run = None
    try:
        require(native.valid_digest(args.protocol_sha256) and file_sha256(args.protocol) == args.protocol_sha256,
                'Protocol hash differs from submitted pin')
        protocol = native.read_json(args.protocol)
        row, accepted = validate_dependencies(protocol, args.prompt_index)
        base = verify_model_environment(accepted)
        manifest = dict(protocol_sha256=args.protocol_sha256, prompt_index=args.prompt_index,
            prompt=row, source_sha256=protocol['source_sha256'], model=accepted['model'],
            torch=torch.__version__, cuda=torch.version.cuda, diffusers=accepted['diffusers'],
            slurm_job_id=os.environ.get('SLURM_JOB_ID'), scope=FIXED['scope'], quality_scored=False,
            seed_device='cpu', seed_policy='crope.utils.image_seed(prompt_id,0)',
            hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                '--format=csv,noheader'], text=True).strip())
        atomic_json(args.out/'manifest.json', manifest)
        atomic_json(args.out/'progress.json', {'status': 'kernel_smoke_before_model_loading', 'completed_images': 0})
        from crope.crossmodal_phase import kernel_smoke
        kernel = kernel_smoke(device='cuda')
        atomic_json(args.out/'kernel_smoke.json', kernel)
        require(kernel.get('passed') is True, 'GPU kernel smoke failed before model load')
        import wandb
        mode = os.environ.get('WANDB_MODE', 'online' if os.environ.get('WANDB_API_KEY') else 'offline')
        kwargs = dict(project='constancy-rope', group='crossmodal-phase-pilot', job_type='generation',
            name=f'crossmodal-phase-{args.prompt_index}', dir=str(native.SCRATCH/'wandb'), config=manifest)
        try:
            run = wandb.init(**kwargs, mode=mode)
        except Exception:
            if mode == 'offline':
                raise
            run = wandb.init(**kwargs, mode='offline')
        atomic_json(args.out/'progress.json', {'status': 'loading_model', 'completed_images': 0})
        from crope.runtime import load_pipeline
        tick = time.perf_counter()
        pipe = load_pipeline('cuda', False, False)
        torch.set_num_threads(2)
        require(Path(pipe.config['_name_or_path']).resolve() == base.resolve(), 'Loaded base revision mismatch')
        require(len(pipe.transformer.transformer_blocks) == 19 and len(pipe.transformer.single_transformer_blocks) == 38 and
                pipe.transformer.config.in_channels == 64 and tuple(pipe.transformer.config.axes_dims_rope) == (16, 56, 56),
                'Unexpected FLUX model geometry')
        require(sorted({str(p.dtype) for p in pipe.transformer.parameters()}) == accepted['model_parameter_dtypes'],
                'Model parameter dtypes changed')
        require(all(pipe.scheduler.config.get(k) == v for k, v in native.SCHEDULER_EXPECTATION.items()),
                'Loaded scheduler differs from frozen expectation')
        pipe.transformer.eval()
        manifest['model_loading_seconds'] = time.perf_counter()-tick
        atomic_json(args.out/'manifest.json', manifest)
        smoke, endpoints = [], {}
        for arm in ('deployed', *ARMS):
            atomic_json(args.out/'progress.json', {'status': 'native_smoke', 'arm': arm, 'completed_images': 0})
            record, endpoint = run_arm(pipe, row, arm, protocol, args.protocol_sha256, args.out,
                                       smoke=True, reference=smoke[0] if smoke else None)
            smoke.append(record)
            endpoints[arm] = endpoint
        validate_arm_set(smoke, row, protocol, smoke=True, protocol_hash=args.protocol_sha256)
        comparisons = {arm: endpoint_comparison(endpoints['baseline'], value) for arm, value in endpoints.items()}
        atomic_json(args.out/'native_smoke.json', {'protocol_sha256': args.protocol_sha256,
            'prompt_index': args.prompt_index, 'records': smoke, 'comparisons_to_baseline': comparisons,
            'no_vae_decode': True, 'cosine_threshold_is_diagnostic': True})
        require(comparisons['deployed']['relative_l2'] is not None and
                comparisons['deployed']['relative_l2'] <= FIXED['deployed_smoke_max_relative_l2'],
                'Deployed/new-backend native endpoint discrepancy exceeds frozen tolerance')
        require(all(comparisons[arm]['byte_exact'] for arm in ('fixedmass', 'unnormalized')),
                'Native phase arms are not byte-identical to matched baseline')
        del endpoints
        records = []
        for arm in ARMS:
            atomic_json(args.out/'progress.json', {'status': 'generating_4k', 'arm': arm, 'completed_images': len(records)})
            record, endpoint = run_arm(pipe, row, arm, protocol, args.protocol_sha256, args.out,
                                       reference=records[0] if records else None)
            records.append(record)
            del endpoint
            run.log({'completed_images': len(records), arm+'/generation_seconds': record['seconds'],
                     arm+'/peak_memory_bytes': record['peak_memory_bytes']})
            atomic_json(args.out/'progress.json', {'status': 'image_completed', 'arm': arm, 'completed_images': len(records)})
        validate_arm_set(records, row, protocol, protocol_hash=args.protocol_sha256)
        atomic_json(args.out/'completion.json', {'protocol_sha256': args.protocol_sha256,
            'prompt_index': args.prompt_index, 'completed_images': 3, 'arms': list(ARMS), 'records': records,
            'quality_scored': False, 'smoke_passed': True, 'scope': FIXED['scope'],
            'cost_limit': 'Unnormalized control adds image-only attention; not an equal-cost candidate.',
            'timing_limit': 'Sequential arms are not a warmup-matched performance benchmark.'})
        atomic_json(args.out/'progress.json', {'status': 'completed', 'completed_images': 3})
    except Exception as error:
        atomic_json(args.out/'failure.json', {'error_type': type(error).__name__, 'message': str(error)})
        atomic_json(args.out/'progress.json', {'status': 'failed', 'error_type': type(error).__name__})
        raise
    finally:
        if run is not None:
            run.finish()


if __name__ == '__main__':
    main()
