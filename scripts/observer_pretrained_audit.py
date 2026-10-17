"""Five-forward pretrained observer audit on an artificial endpoint/noise state.

Preparation does not execute a model. CLI execution requires a frozen protocol,
accepted CR17 evidence for both development prompts, and an allocated CUDA GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from crope.utils import atomic_json, runtime_environment

SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
KREA = '8162a9c7b05a641be098422bf2fcf335615c2f28'
BASE = '3de623fc3c33e44ffbe2bad470d0f45bccf2eb21'
ARMS = ('original_g0', 'repeat_original_g0', 'shifted_g1', 'shifted_g0', 'original_g1')
REQUIRED_SOURCES = ('scripts/observer_pretrained_audit.py', 'crope/deterministic_allocation.py',
                    'crope/runtime.py', 'crope/gain.py', 'crope/rope.py',
                    'crope/sega_official.py', 'crope/utils.py')
SHIFT_SPEC = dict(width_harmonic=3, height_harmonic=1, height_amplitude=.3,
                  amplitude='baseline_observer_population_std', unit_rms=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def validate_protocol(protocol):
    fixed = dict(schema_version=1, development_prompt_indices=[0, 1], resolution=4096,
                 sigma=.5, guidance=4.5, noise_seeds={'0': 1729, '1': 1730},
                 noise_device='cpu', noise_dtype='float32', mixture_dtype='float32',
                 model_dtype='bfloat16', forward_calls_per_prompt=5,
                 allocation_numerics='deterministic_sega_update_only', observer_shift=SHIFT_SPEC)
    for name, value in fixed.items():
        require(protocol.get(name) == value, f'Protocol {name} mismatch')
    for key in ('accepted_report', 'accepted_receipt'):
        require(isinstance(protocol.get(key), str) and Path(protocol[key]).is_absolute(),
                f'Protocol requires absolute {key}')
    require(set(protocol.get('endpoint_file_sha256', {})) == {'0', '1'}, 'Both endpoint hashes required')
    require(len(protocol.get('prompt_ids', [])) == 2 and
            len(set(protocol['prompt_ids'])) == 2, 'Two distinct prompt IDs required')
    model = protocol.get('model', {})
    require(model.get('transformer_revision') == KREA and model.get('base_revision') == BASE,
            'Pinned Krea/base revisions required')
    hashes = [protocol.get(k) for k in ('accepted_report_sha256', 'accepted_receipt_sha256',
                                       'prompt_file_sha256')]
    hashes += list(protocol['endpoint_file_sha256'].values())
    hashes += [model.get(k) for k in ('transformer_config_sha256', 'base_model_index_sha256')]
    sources = protocol.get('source_sha256', {})
    require(set(REQUIRED_SOURCES) <= set(sources), 'Required worker sources are not frozen')
    hashes += list(sources.values())
    require(all(isinstance(value, str) and len(value) == 64 and
                all(char in '0123456789abcdef' for char in value) for value in hashes),
            'Invalid SHA256 in protocol')
    require(all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in sources),
            'Source paths must be relative repository paths')


def validate_report_metadata(report, protocol):
    """Validate the pinned report's completed four-stage identities before loading tensors."""
    keys = {f'prompt{i}/{stage}' for i in (0, 1) for stage in ('smoke', '4k')}
    require(set(report.get('stages', {})) == keys, 'Both prompts require completed smoke and 4K stages')
    require(report.get('automatic_promotion') is False, 'Unexpected report promotion')
    prompts = report['inputs']['development_prompts']
    require(len(prompts) == 2 and [p['id'] for p in prompts] == protocol['prompt_ids'], 'Prompt IDs mismatch')
    require(report['inputs']['prompt_file_sha256'] == protocol['prompt_file_sha256'], 'Prompt file hash mismatch')
    for index in (0, 1):
        for stage in ('smoke', '4k'):
            record = report['stages'][f'prompt{index}/{stage}']
            manifest = record['summary']['manifest']
            raw = record['raw_json']
            require(raw.get('manifest.json') == manifest, 'Archived manifest mismatch')
            receipt = raw.get('completion.json')
            require(isinstance(receipt, dict) and receipt == raw.get('validation.json'),
                    'Missing or mismatched completion receipt')
            require(all(receipt.get(k) is True for k in ('gpu_probe_executed',
                    'trajectory_bitwise_equal', 'baseline_history_equal')), 'CR17 stage did not pass validation')
            require(manifest.get('prompt_index') == index and manifest.get('prompt') == prompts[index],
                    'Prompt identity mismatch')
            require(manifest.get('res') == [2048 if stage == 'smoke' else 4096] * 2,
                    'Resolution mismatch')
            require(manifest.get('seed') == 0 and manifest.get('guidance') == 4.5,
                    'CR17 seed/guidance mismatch')
            require(manifest.get('prompt_file_sha256') == protocol['prompt_file_sha256'],
                    'Manifest prompt file hash mismatch')
            require(all(manifest.get(k) == v for k, v in protocol['model'].items()), 'Model provenance mismatch')
            config = manifest.get('experiment_configuration', {})
            require(config.get('allocation_numerics') == protocol['allocation_numerics'] and
                    config.get('dtype') == 'bfloat16' and config.get('batch_size') == 1 and
                    config.get('method') == 'sega_official', 'Allocation/model configuration mismatch')
            endpoint = record['endpoint_verification']
            require(endpoint.get('independently_verified_on_cpu') is True, 'Endpoint not independently verified')
            require(endpoint.get('shape') == [(128 if stage == 'smoke' else 256)**2, 64] and
                    endpoint.get('dtype') == 'torch.bfloat16', 'Endpoint shape/dtype mismatch')
            require(endpoint.get('tensor_sha256') == receipt.get('baseline_sha256') == receipt.get('probed_sha256'),
                    'Endpoint tensor hash mismatch')
            if stage == '4k':
                require(endpoint['file_sha256'] == protocol['endpoint_file_sha256'][str(index)],
                        'Pinned endpoint file hash mismatch')
    return prompts


def load_verified_inputs(protocol, prompt_index):
    """Fail before any model load if either prompt's accepted evidence is incomplete."""
    validate_protocol(protocol)
    require(prompt_index in (0, 1), 'Only development prompts 0 and 1 are permitted')
    for name, digest in protocol['source_sha256'].items():
        require(sha256(ROOT / name) == digest, f'Frozen source changed: {name}')
    report_path, receipt_path = Path(protocol['accepted_report']), Path(protocol['accepted_receipt'])
    require(sha256(report_path) == protocol['accepted_report_sha256'], 'Accepted report hash mismatch')
    require(sha256(receipt_path) == protocol['accepted_receipt_sha256'], 'Archive receipt hash mismatch')
    report, archive = read_json(report_path), read_json(receipt_path)
    require(archive.get('report_sha256') == protocol['accepted_report_sha256'], 'Receipt does not certify report')
    prompts = validate_report_metadata(report, protocol)
    selected = None
    for index in (0, 1):
        for stage in ('smoke', '4k'):
            record = report['stages'][f'prompt{index}/{stage}']
            for name in ('manifest.json', 'completion.json', 'validation.json'):
                archived = report_path.parent / f'prompt{index}/{stage}' / name
                require(read_json(archived) == record['raw_json'][name], 'Archived raw JSON mismatch')
        record = report['stages'][f'prompt{index}/4k']
        endpoint = record['endpoint_verification']
        path = Path(endpoint['path'])
        require(sha256(path) == protocol['endpoint_file_sha256'][str(index)], 'Endpoint file changed')
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        require(set(payload) == {'baseline', 'probed'}, 'Unexpected endpoint payload')
        base, probed = payload['baseline'], payload['probed']
        require(base.shape == probed.shape == (65536, 64), 'Expected exact unbatched packed endpoint shape')
        require(base.dtype == probed.dtype == torch.bfloat16 and bool(torch.isfinite(base).all()),
                'Endpoint must be finite BF16')
        require(torch.equal(base, probed), 'Endpoint repeat mismatch')
        require(tensor_hash(base) == endpoint['tensor_sha256'], 'Endpoint tensor contents changed')
        if index == prompt_index:
            selected = base.clone()
    return selected, prompts[prompt_index], report['stages'][f'prompt{prompt_index}/4k']['summary']['manifest']


def artificial_state(endpoint, prompt_index):
    require(prompt_index in (0, 1), 'Only development prompts 0 and 1 are permitted')
    require(endpoint.shape == (65536, 64) and endpoint.dtype == torch.bfloat16 and
            endpoint.device.type == 'cpu', 'Expected exact CPU BF16 packed endpoint')
    require(bool(torch.isfinite(endpoint).all()), 'Nonfinite endpoint')
    generator = torch.Generator(device='cpu').manual_seed(1729 + prompt_index)
    noise = torch.randn(endpoint.shape, dtype=torch.float32, device='cpu', generator=generator)
    mixed = .5 * endpoint.float() + .5 * noise
    value = mixed.bfloat16().unsqueeze(0)
    return value, {'construction': 'artificial x=.5*CR17_final_endpoint+.5*CPU_FP32_noise, then BF16',
                   'on_trajectory': False, 'noise_seed': 1729 + prompt_index,
                   'rng': 'torch.Generator(device=cpu)', 'noise_dtype': 'torch.float32',
                   'mixture_dtype': 'torch.float32', 'input_dtype': str(value.dtype),
                   'shape': list(value.shape), 'noise_sha256': tensor_hash(noise),
                   'mixture_float32_sha256': tensor_hash(mixed), 'input_sha256': tensor_hash(value)}


def target_observer(mean, grid):
    require(mean.shape == (1, grid[0] * grid[1]), 'Observer shape mismatch')
    yy, xx = torch.meshgrid(torch.arange(grid[0], device=mean.device, dtype=torch.float32),
                           torch.arange(grid[1], device=mean.device, dtype=torch.float32), indexing='ij')
    pattern = (torch.sin(2 * torch.pi * 3 * xx / grid[1]) +
               .3 * torch.cos(2 * torch.pi * yy / grid[0])).reshape(1, -1)
    pattern = pattern - pattern.mean()
    pattern = pattern / pattern.square().mean().sqrt()
    return mean.mean() + mean.std(correction=0) * pattern


def shift_features(features, target):
    require(features.ndim == 3 and features.shape[0] == 1 and
            target.shape == features.shape[:2], 'Feature/target shape mismatch')
    fp32 = features.float()
    shifted = (fp32 + (target - fp32.mean(-1))[..., None]).to(features.dtype)
    return shifted


def error_stats(value, reference):
    a, b = value.detach().double(), reference.detach().double()
    require(a.shape == b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),
            'Contrast requires matching finite tensors')
    delta, denominator = a - b, b.norm().item()
    return {'max_abs': delta.abs().max().item(), 'l2': delta.norm().item(),
            'relative_l2': delta.norm().item() / denominator if denominator else None,
            'reference_l2': denominator, 'bitwise_equal': torch.equal(value, reference)}


def contrast_summary(outputs):
    require(set(outputs) == set(ARMS), 'Five arms are required')
    a, repeat, b, d, c = (outputs[name] for name in ARMS)
    return {
        'exact_repeat': error_stats(repeat, a),
        'adaptive_total_shifted_g1_minus_original_g0': error_stats(b, a),
        'fixed_g0_shift_leakage': error_stats(d, a),
        'gain_only_original_g1_minus_original_g0': error_stats(c, a),
        'mediation_residual_shifted_g1_minus_original_g1': error_stats(b, c),
        'gain_effect_at_shifted_features': error_stats(b, d),
        'shift_gain_interaction': error_stats(b.double() - c.double(), d.double() - a.double()),
    }


def validate_repeat(original, repeated, original_gain, repeated_gain):
    require(torch.equal(original, repeated) and original_gain == repeated_gain,
            'Exact no-op repeat failed; diagnostics preserved, remaining interventions skipped')


def feature_delta_geometry(original, shifted):
    # Chunk to avoid several simultaneous full-width FP64 residual arrays at 4K.
    delta2 = base2 = common2 = full2 = 0.
    maximum = 0.
    for start in range(0, original.shape[1], 1024):
        a = original[:, start:start+1024].double()
        b = shifted[:, start:start+1024].double()
        delta = b - a
        common = delta.mean(-1, keepdim=True)
        transverse = delta - common
        delta2 += transverse.square().sum().item()
        common2 += (common.square().sum() * original.shape[-1]).item()
        full2 += delta.square().sum().item()
        base2 += (a - a.mean(-1, keepdim=True)).square().sum().item()
        maximum = max(maximum, transverse.abs().max().item())
    return {'max_abs': maximum, 'l2': delta2**.5,
            'relative_l2': (delta2/base2)**.5 if base2 else None,
            'realized_common_channel_delta_l2': common2**.5,
            'realized_transverse_delta_l2': delta2**.5,
            'realized_full_delta_l2': full2**.5,
            'transverse_fraction_of_full_delta': (delta2/full2)**.5 if full2 else None,
            'interpretation': 'Realized BF16 feature delta decomposed into common-channel and orthogonal components'}


def observer_record(mean, grid):
    from crope.sega_official import axis_profiles, radial_profile
    spatial = mean.float().reshape(grid)
    centered = spatial - spatial.mean()
    return {'sha256': tensor_hash(mean), 'mean': spatial.mean().item(),
            'population_std': spatial.std(correction=0).item(),
            'axis_profiles': [v.cpu().tolist() for v in axis_profiles(centered)],
            'radial_profile': radial_profile(centered).cpu().tolist()}


def gain_record(value):
    cast = value.bfloat16().float()
    return {'float32': value.cpu().tolist(), 'cast_bfloat16': cast.cpu().tolist(),
            'float32_sha256': tensor_hash(value), 'bfloat16_sha256': tensor_hash(value.bfloat16())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime_environment()
    protocol = read_json(args.protocol)
    endpoint, prompt, accepted_manifest = load_verified_inputs(protocol, args.prompt_index)
    require(SCRATCH in args.out.resolve().parents, 'Output must be under project scratch')
    require(not args.out.exists(), 'Use a fresh output directory')
    require(torch.cuda.is_available(), 'Allocated CUDA GPU required; no CPU model execution')
    transformer_path = Path(os.environ['CROPE_TRANSFORMER'])
    require(transformer_path.name == KREA and
            sha256(transformer_path / 'transformer/config.json') == protocol['model']['transformer_config_sha256'],
            'Pinned Krea snapshot/configuration mismatch')
    hub = Path(os.environ['HF_HOME']) / 'hub/models--black-forest-labs--FLUX.1-dev'
    require((hub / 'refs/main').read_text().strip() == BASE, 'Base cache revision mismatch')
    base_snapshot = hub / 'snapshots' / BASE
    require(sha256(base_snapshot / 'model_index.json') == protocol['model']['base_model_index_sha256'],
            'Base model configuration mismatch')
    import diffusers
    require(torch.__version__ == accepted_manifest['torch'] and
            diffusers.__version__ == accepted_manifest['experiment_configuration']['diffusers'],
            'Software versions differ from accepted CR17 environment')
    from crope.runtime import load_pipeline
    from crope.deterministic_allocation import DeterministicAllocationSEGA
    from crope.gain import install_gain
    import crope.gain as gain_module
    import wandb
    args.out.mkdir(parents=True)
    state, state_meta = artificial_state(endpoint, args.prompt_index)
    torch.save({'input': state}, args.out / 'input.pt')
    manifest = {'purpose': 'Artificial-state observer consistency audit; no quality or novelty claim',
                'protocol_sha256': sha256(args.protocol), 'prompt_index': args.prompt_index,
                'prompt': prompt, 'state': state_meta, 'source_sha256': protocol['source_sha256'],
                'model': protocol['model'], 'accepted_report_sha256': protocol['accepted_report_sha256'],
                'accepted_receipt_sha256': protocol['accepted_receipt_sha256'],
                'endpoint_file_sha256': protocol['endpoint_file_sha256'][str(args.prompt_index)],
                'input_file_sha256': sha256(args.out / 'input.pt'),
                'allocation_numerics': protocol['allocation_numerics'], 'sigma': .5, 'guidance': 4.5,
                'resolution': 4096, 'forward_calls_budget': 5, 'arm_order': list(ARMS),
                'observer_shift': SHIFT_SPEC, 'torch': torch.__version__, 'diffusers': diffusers.__version__,
                'observer_shift_equation': 'mu0=mean_channels_FP32(h0); target=mean_space(mu0)+std_space_population(mu0)*unit_RMS_zero_mean_pattern; h1=BF16(FP32(h0)+(target-mu0)*ones_channels)',
                'gain_repeat_scope': 'Original repeat reuses G0; this tests forward repeatability, not allocator recomputation.',
                'cuda': torch.version.cuda, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                                                     '--format=csv,noheader'], text=True).strip()}
    atomic_json(args.out / 'manifest.json', manifest)
    atomic_json(args.out / 'progress.json', {'status': 'loading_model', 'completed_forwards': 0})
    run = wandb.init(project='constancy-rope', group='observer-null-audit', job_type='diagnostic',
                     name=f'observer-null-{args.prompt_index}', dir=str(SCRATCH / 'wandb'), config=manifest,
                     mode=os.environ.get('WANDB_MODE', 'online' if os.environ.get('WANDB_API_KEY') else 'offline'))
    handle = None
    original_attention = gain_module.gained_attention
    consumed = {}

    def recorded_attention(query, key, value, qk_scale, attention_mask=None):
        require(query.dtype == key.dtype == torch.bfloat16, 'Attention Q/K dtype differs from BF16 protocol')
        cast = qk_scale.to(device=query.device, dtype=query.dtype)
        if 'cast' not in consumed:
            consumed['cast'] = cast.clone()
        require(torch.equal(cast, consumed['cast']), 'Gain cast changed between attention blocks')
        consumed['calls'] = consumed.get('calls', 0) + 1
        return original_attention(query, key, value, qk_scale, attention_mask)

    gain_module.gained_attention = recorded_attention
    try:
        tick = time.perf_counter()
        pipe = load_pipeline('cuda', False, False)
        require(Path(pipe.config['_name_or_path']).resolve() == base_snapshot.resolve(), 'Loaded base differs')
        require((hub / 'refs/main').read_text().strip() == BASE, 'Base revision changed during loading')
        pipe.transformer.eval()
        loading_seconds = time.perf_counter() - tick
        with torch.inference_mode():
            prompt_embeds, pooled, txt_ids = pipe.encode_prompt(prompt=prompt['prompt'], prompt_2=None,
                device=torch.device('cuda'), num_images_per_prompt=1, max_sequence_length=512)
            img_ids = pipe._prepare_latent_image_ids(1, 256, 256, torch.device('cuda'), torch.bfloat16)
            kwargs = dict(hidden_states=state.cuda(), encoder_hidden_states=prompt_embeds,
                          pooled_projections=pooled, txt_ids=txt_ids, img_ids=img_ids,
                          timestep=torch.tensor([.5], device='cuda', dtype=torch.bfloat16),
                          guidance=torch.tensor([4.5], device='cuda', dtype=torch.float32), return_dict=False)
            require(pipe.transformer.config.in_channels == 64 and
                    tuple(pipe.transformer.config.axes_dims_rope) == (16, 56, 56), 'Unexpected transformer geometry')
            official = DeterministicAllocationSEGA((256, 256), (16, 56, 56))
            pipe.transformer.pos_embed = official.rope
            controller = SimpleNamespace(qk_scale=None)
            install_gain(pipe.transformer, controller)
            manifest.update(model_loading_seconds=loading_seconds,
                            conditioning_sha256={key: tensor_hash(kwargs[key]) for key in
                                                 ('encoder_hidden_states', 'pooled_projections', 'txt_ids', 'img_ids')},
                            model_parameter_dtypes=sorted({str(p.dtype) for p in pipe.transformer.parameters()}))
            atomic_json(args.out / 'manifest.json', manifest)
            run.config.update(manifest, allow_val_change=True)
            outputs, gains, arm_records = {}, {}, {}
            target = baseline_features = cached_shifted_features = shifted_geometry = None
            feature_hashes = {}
            for arm_index, arm in enumerate(ARMS):
                hook_records = []
                consumed.clear()
                atomic_json(args.out / 'progress.json', {'status': 'forward_running', 'arm': arm,
                            'completed_forwards': arm_index, 'total_forwards': 5})

                def hook(_module, _inputs, h):
                    nonlocal target, baseline_features, cached_shifted_features, shifted_geometry
                    require(h.shape == (1, 65536, 3072) and h.dtype == torch.bfloat16,
                            'Expected actual BF16 FLUX projected features')
                    if baseline_features is None:
                        baseline_features = h.clone()
                        original_mean = baseline_features.float().mean(-1)
                        target = target_observer(original_mean, (256, 256))
                        cached_shifted_features = shift_features(baseline_features, target)
                        shifted_geometry = feature_delta_geometry(baseline_features, cached_shifted_features)
                        feature_hashes['original'] = tensor_hash(baseline_features)
                        feature_hashes['shifted'] = tensor_hash(cached_shifted_features)
                    require(torch.equal(h, baseline_features), 'Projected original features changed between arms')
                    shifted = arm in ('shifted_g1', 'shifted_g0')
                    actual = cached_shifted_features if shifted else h
                    if arm == 'original_g0':
                        gains['g0'] = official.update(actual, .5).clone()
                    elif arm == 'shifted_g1':
                        gains['g1'] = official.update(actual, .5).clone()
                    controller.qk_scale = gains['g1' if arm in ('shifted_g1', 'original_g1') else 'g0'].clone()
                    mean = actual.float().mean(-1)
                    record = {'observer': observer_record(mean, (256, 256)),
                              'gain': gain_record(controller.qk_scale),
                              'incoming_features_equal_baseline': True,
                              'realized_feature_sha256': feature_hashes['shifted' if shifted else 'original'],
                              'target_error': error_stats(mean, target) if shifted else None,
                              'centered_feature_error': shifted_geometry if shifted else None}
                    torch.save({'actual_observer': mean.cpu(), 'target': target.cpu()}, args.out / f'{arm}_observer.pt')
                    hook_records.append(record)
                    return actual

                handle = pipe.transformer.x_embedder.register_forward_hook(hook)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                try:
                    with pipe.transformer.cache_context('cond'):
                        output = pipe.transformer(**kwargs)[0]
                    torch.cuda.synchronize()
                finally:
                    handle.remove()
                    handle = None
                seconds = time.perf_counter() - tick
                require(len(hook_records) == 1, 'Expected one x_embedder hook per forward')
                expected_attention_calls = len(pipe.transformer.transformer_blocks) + len(pipe.transformer.single_transformer_blocks)
                require(consumed.get('calls') == expected_attention_calls, 'Attention receipt count mismatch')
                require(torch.equal(consumed['cast'], controller.qk_scale.bfloat16()), 'Consumed gain mismatch')
                require(output.shape == (1, 65536, 64) and bool(torch.isfinite(output).all()), 'Invalid transformer output')
                outputs[arm] = output.cpu()
                torch.save({'output': outputs[arm]}, args.out / f'{arm}_output.pt')
                arm_records[arm] = {**hook_records[0], 'forward_seconds_including_hook_diagnostics': seconds,
                                    'attention_calls': consumed['calls'], 'attention_query_dtype': 'torch.bfloat16',
                                    'consumed_gain_bfloat16_sha256': tensor_hash(consumed['cast']),
                                    'output_sha256': tensor_hash(outputs[arm]),
                                    'output_file_sha256': sha256(args.out / f'{arm}_output.pt')}
                atomic_json(args.out / f'{arm}.json', arm_records[arm])
                atomic_json(args.out / 'progress.json', {'status': 'forward_completed', 'arm': arm,
                            'completed_forwards': arm_index + 1, 'total_forwards': 5})
                run.log({'arm_index': arm_index, 'forward_seconds': seconds, 'completed_forwards': arm_index + 1})
                if arm == 'repeat_original_g0':
                    atomic_json(args.out / 'repeat_validation.json', {
                        'output': error_stats(outputs[arm], outputs['original_g0']),
                        'replayed_gain_equal': arm_records[arm]['gain'] == arm_records['original_g0']['gain']})
                    validate_repeat(outputs['original_g0'], outputs[arm],
                                    arm_records['original_g0']['gain'], arm_records[arm]['gain'])
            contrasts = contrast_summary(outputs)
            g0, g1 = gains['g0'].cpu(), gains['g1'].cpu()
            repeat_equal = (torch.equal(outputs['original_g0'], outputs['repeat_original_g0']) and
                            arm_records['original_g0']['gain'] == arm_records['repeat_original_g0']['gain'])
            validation = {'completed_forwards': 5, 'exact_repeat_equal': repeat_equal,
                          'contrasts': contrasts, 'arms': arm_records,
                          'gain_change_float32': error_stats(g1, g0),
                          'gain_change_bfloat16': error_stats(g1.bfloat16(), g0.bfloat16()),
                          'gain_quantization_dead': torch.equal(g1.bfloat16(), g0.bfloat16()),
                          'gain_changed_bfloat16_coordinates': int((g1.bfloat16() != g0.bfloat16()).sum()),
                          'on_trajectory': False, 'efficacy_evaluated': False, 'novelty_established': False,
                          'interpretation': 'Fixed-gain BF16/architecture and mediation residuals are observed measurements, not assumed pure rounding. One original repeat establishes repeatability of that pair only.',
                          'timing_interpretation': 'Includes hook geometry, serialization and synchronization; not a speed benchmark.'}
            atomic_json(args.out / 'validation.json', validation)
            require(repeat_equal, 'Exact no-op repeat failed; diagnostics preserved, no completion success')
            atomic_json(args.out / 'completion.json', validation)
            atomic_json(args.out / 'progress.json', {'status': 'completed', 'completed_forwards': 5})
    except Exception as error:
        atomic_json(args.out / 'failure.json', {'error_type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        gain_module.gained_attention = original_attention
        if handle is not None:
            handle.remove()
        run.finish()


if __name__ == '__main__':
    main()
