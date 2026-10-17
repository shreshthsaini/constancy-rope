"""Tiny CPU common-channel observer intervention with SEGA gain mediation.

The intervention is on x_embedder output, not a claim that any spatial map is
realizable by a single parameter reparameterization for all latent inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import diffusers
import torch
from diffusers.models import embeddings, normalization
from diffusers.models.transformers import transformer_flux

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crope import gain, rope, sega_official
from crope.gain import install_gain
from crope.sega_official import OfficialSEGA, axis_profiles, radial_profile
from scripts import observer_shift_audit
from scripts.observer_shift_audit import errors, feature_summary, tiny_fixture


def force_channel_mean(features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Add a common-channel scalar per token to impose the target mean."""
    if target.shape != features.shape[:-1]:
        raise ValueError('Target must match batch and token dimensions')
    return features + (target - features.mean(-1))[..., None]


def released_observer(features, grid):
    spatial = features.reshape(features.shape[0], *grid, -1).float().mean(dim=(0, -1))
    spatial = spatial - spatial.mean()
    return {'spatial_centered_map': spatial.tolist(),
            'axis_profiles': [profile.tolist() for profile in axis_profiles(spatial)],
            'radial_profile': radial_profile(spatial).tolist()}


@torch.no_grad()
def run_audit(seed=1729):
    started = time.perf_counter()
    model, kwargs, _ = tiny_fixture(seed)
    grid = (4, 128)
    generator = torch.Generator(device='cpu').manual_seed(seed + 1)
    kwargs['hidden_states'] = torch.randn(1, grid[0] * grid[1], 8,
                                          dtype=torch.float64, generator=generator)
    yy, xx = torch.meshgrid(torch.arange(grid[0], dtype=torch.float64),
                           torch.arange(grid[1], dtype=torch.float64), indexing='ij')
    kwargs['img_ids'] = torch.stack((torch.zeros_like(yy), yy, xx), dim=-1).reshape(-1, 3)
    # Frozen before execution: width harmonic 3, height harmonic 1, amplitudes 1/.3.
    target = (torch.sin(2 * torch.pi * 3 * xx / grid[1])
              + .3 * torch.cos(2 * torch.pi * yy / grid[0])).reshape(1, -1)
    original = model.x_embedder(kwargs['hidden_states'])
    shifted = force_channel_mean(original, target)
    official = OfficialSEGA(grid, axes_dims=(4, 8, 8))
    model.pos_embed = official.rope
    sigma = float(kwargs['timestep'].item())
    original_gain = official.update(original, sigma).clone()
    original_diagnostics = dict(official.diagnostics)
    shifted_gain = official.update(shifted, sigma).clone()
    shifted_diagnostics = dict(official.diagnostics)
    controller = SimpleNamespace(qk_scale=original_gain.clone())
    install_gain(model, controller)
    outputs, records = {}, {}
    configurations = [
        ('fixed_original', False, original_gain, False),
        ('fixed_shifted', True, original_gain, False),
        ('adaptive_original', False, original_gain, True),
        ('adaptive_shifted', True, original_gain, True),
        ('original_with_shifted_gain', False, shifted_gain, False),
    ]
    for name, use_shift, fixed_gain, adaptive in configurations:
        controller.qk_scale = fixed_gain.clone()
        observed = []

        def hook(_module, _inputs, output):
            feature = force_channel_mean(output, target) if use_shift else output
            if adaptive:
                controller.qk_scale = official.update(feature, sigma).clone()
            observed.append(feature.detach().clone())
            return feature

        handle = model.x_embedder.register_forward_hook(hook)
        tick = time.perf_counter()
        try:
            output = model(**kwargs)[0]
        finally:
            handle.remove()
        elapsed = time.perf_counter() - tick
        if len(observed) != 1:
            raise RuntimeError(f'Expected one x_embedder observation, got {len(observed)}')
        outputs[name] = output
        records[name] = {'shifted': use_shift, 'adaptive': adaptive,
                         'executed_qk_gain': controller.qk_scale.tolist(),
                         'forward_seconds': elapsed,
                         'output_vs_fixed_original': errors(output, outputs['fixed_original'])}
    contrasts = {
        'fixed_gain_shift_vs_original': errors(outputs['fixed_shifted'], outputs['fixed_original']),
        'adaptive_original_vs_fixed_original': errors(outputs['adaptive_original'], outputs['fixed_original']),
        'adaptive_shift_vs_original': errors(outputs['adaptive_shifted'], outputs['adaptive_original']),
        'mediation_original_shifted_gain_vs_adaptive_shifted': errors(
            outputs['original_with_shifted_gain'], outputs['adaptive_shifted']),
    }
    centered_original = original - original.mean(-1, keepdim=True)
    centered_shifted = shifted - shifted.mean(-1, keepdim=True)
    source_modules = [observer_shift_audit, gain, rope, sega_official,
                      transformer_flux, normalization, embeddings]
    source_paths = [Path(__file__)] + [Path(inspect.getfile(module)) for module in source_modules]
    return {
        'schema_version': 1,
        'scope': 'Single tiny random FLUX forward, common-channel feature intervention and SEGA gain mediation',
        'limitations': [
            'Not pretrained, BF16, image quality, trajectory, novelty, or efficacy evidence.',
            'The forced observer is an activation intervention for this input, not a universal parameter transform.',
            'Installed SEGA observer/gains use FP32; transformer parameters/residuals/output use FP64.',
            'Installed rotary code internally casts Q/K to FP32.',
            'Same SEGA rotary base, conditioning, inputs, and weights in every comparison.',
        ],
        'seed': seed, 'latent_seed': seed + 1, 'grid': list(grid), 'sigma': sigma,
        'threads': torch.get_num_threads(),
        'versions': {'torch': torch.__version__, 'diffusers': diffusers.__version__},
        'model_config': dict(model.config),
        'attention_processors': {name: type(processor).__name__
                                 for name, processor in model.attn_processors.items()},
        'shift_specification': {'width_harmonic': 3, 'height_harmonic': 1,
                                'width_amplitude': 1., 'height_amplitude': .3,
                                'formula': 'h_shifted = h + (target - mean_channels(h)) * ones_channels'},
        'original': {'feature_summary': feature_summary(original, grid),
                     'released_observer': released_observer(original, grid),
                     'qk_gain': original_gain.tolist(), 'diagnostics': original_diagnostics},
        'shifted': {'feature_summary': feature_summary(shifted, grid),
                    'released_observer': released_observer(shifted, grid),
                    'qk_gain': shifted_gain.tolist(), 'diagnostics': shifted_diagnostics},
        'gain_change': errors(shifted_gain.double(), original_gain.double()),
        'target_mean_error': errors(shifted.mean(-1), target),
        'channel_centered_feature_error': errors(centered_shifted, centered_original),
        'runs': records, 'contrasts': contrasts,
        'source_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths},
        'total_wall_seconds_excluding_imports': time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'gain_change': report['gain_change'], 'contrasts': report['contrasts']}))


if __name__ == '__main__':
    main()
