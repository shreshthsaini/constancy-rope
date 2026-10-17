"""CPU numerical audit of a common-channel x_embedder shift in tiny FLUX.

This tests a random miniature with fixed default attention, not pretrained
quality or a closed-loop SEGA trajectory. No arbitrary channel rotation is used.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time

import diffusers
import torch
from diffusers import FluxTransformer2DModel
from diffusers.models import embeddings, normalization
from diffusers.models.transformers import transformer_flux


def tiny_fixture(seed: int = 712):
    """Actual installed architecture, one joint block and one single block."""
    torch.manual_seed(seed)
    model = FluxTransformer2DModel(
        patch_size=1, in_channels=8, out_channels=8, num_layers=1,
        num_single_layers=1, attention_head_dim=20, num_attention_heads=1,
        joint_attention_dim=12, pooled_projection_dim=6, guidance_embeds=True,
        axes_dims_rope=(4, 8, 8)).double().eval()
    grid = (8, 8)
    ids = torch.zeros(64, 3, dtype=torch.float64)
    ids[:, 1] = torch.arange(8).repeat_interleave(8)
    ids[:, 2] = torch.arange(8).repeat(8)
    kwargs = {
        'hidden_states': torch.randn(1, 64, 8, dtype=torch.float64),
        'encoder_hidden_states': torch.randn(1, 3, 12, dtype=torch.float64),
        'pooled_projections': torch.randn(1, 6, dtype=torch.float64),
        'guidance': torch.tensor([4.5], dtype=torch.float64),
        'timestep': torch.tensor([.8], dtype=torch.float64),
        'img_ids': ids, 'txt_ids': torch.zeros(3, 3, dtype=torch.float64),
        'return_dict': False,
    }
    return model, kwargs, grid


def errors(value: torch.Tensor, reference: torch.Tensor) -> dict:
    delta = value - reference
    reference_norm = reference.norm().item()
    return {'max_abs': delta.abs().max().item(), 'l2': delta.norm().item(),
            'relative_l2': delta.norm().item() / reference_norm
            if reference_norm else None}


def feature_summary(features: torch.Tensor, grid: tuple[int, int]) -> dict:
    spatial = features.mean(dim=(0, 2)).reshape(grid)
    spatial_centered = spatial - spatial.mean()
    centered_features = features - features.mean(-1, keepdim=True)
    spectrum = torch.fft.fft2(spatial_centered, norm='ortho').abs().square()
    axis_spectra = [torch.fft.fft(spatial_centered, dim=axis, norm='ortho')
                    .abs().square().mean(dim=1-axis).tolist() for axis in (0, 1)]
    return {
        'channel_mean_map': spatial.tolist(),
        'spatial_centered_channel_mean_energy': spatial_centered.square().sum().item(),
        'spatial_centered_channel_mean_fft2_power': spectrum.tolist(),
        'spatial_centered_channel_mean_axis_power': axis_spectra,
        'channel_centered_feature_energy': centered_features.square().sum().item(),
        'max_abs_channel_mean': spatial.abs().max().item(),
    }


@torch.no_grad()
def run_audit(seed: int = 712) -> dict:
    started = time.perf_counter()
    model, kwargs, grid = tiny_fixture(seed)
    weight = model.x_embedder.weight.detach().clone()
    bias = model.x_embedder.bias.detach().clone()
    channels, input_dim = weight.shape
    # Moderate fixed shifts are chosen without consulting output errors.
    a = torch.linspace(-.7, .9, input_dim, dtype=torch.float64)
    c = .37
    common = torch.ones(channels, dtype=torch.float64)
    wrong = torch.linspace(-1., 1., channels, dtype=torch.float64)
    wrong /= wrong.square().mean().sqrt()  # match common vector RMS
    zero_weight = weight - weight.mean(0, keepdim=True)
    zero_bias = bias - bias.mean()
    variants = {
        'baseline': (weight, bias),
        'common_shift': (weight + common[:, None] * a, bias + c * common),
        'mean_zeroed': (zero_weight, zero_bias),
        'redirected': (zero_weight + common[:, None] * a, zero_bias + c * common),
        'channel_dependent_negative_control': (
            weight + wrong[:, None] * a, bias + c * wrong),
    }
    outputs, features = {}, {}
    elapsed = {}
    for name, (new_weight, new_bias) in variants.items():
        model.x_embedder.weight.copy_(new_weight)
        model.x_embedder.bias.copy_(new_bias)
        features[name] = model.x_embedder(kwargs['hidden_states'])
        tick = time.perf_counter()
        outputs[name] = model(**kwargs)[0]
        elapsed[name] = time.perf_counter() - tick
    base_feature = features['baseline']
    base_centered = base_feature - base_feature.mean(-1, keepdim=True)
    expected_shift = kwargs['hidden_states'] @ a + c
    records = {}
    for name, feature in features.items():
        centered = feature - feature.mean(-1, keepdim=True)
        records[name] = {
            'output_vs_baseline': errors(outputs[name], outputs['baseline']),
            'channel_centered_feature_vs_baseline': errors(centered, base_centered),
            'feature_summary': feature_summary(feature, grid),
            'forward_seconds': elapsed[name],
        }
    ln = torch.nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6).double()
    layernorm_error = errors(ln(base_feature + expected_shift[..., None]), ln(base_feature))
    source_paths = [Path(__file__), Path(inspect.getfile(transformer_flux)),
                    Path(inspect.getfile(normalization)), Path(inspect.getfile(embeddings))]
    root = Path(__file__).resolve().parents[1]
    source_paths.append(root / 'crope' / 'sega_official.py')
    tolerance = 1e-10
    invariant_names = ('common_shift', 'mean_zeroed', 'redirected')
    invariance_pass = all(records[name]['output_vs_baseline']['max_abs'] <= tolerance
                          for name in invariant_names)
    return {
        'schema_version': 1,
        'scope': 'CPU random tiny FLUX, fixed default attention, common-channel shifts only',
        'limitations': [
            'Not pretrained, BF16, image-quality, or trajectory evidence.',
            'Prompt, timestep and guidance conditioning are fixed; no ControlNet or IP adapter.',
            'No adaptive SEGA gains are executed. The measured projected channel mean is its observer input.',
            'Parameters, residuals and outputs are float64; installed rotary code internally casts Q/K to float32.',
            'Reported FFTs use orthonormal normalization, not released SEGA binning or floors.',
            'The redirectable map is x @ a + c, not an arbitrary spatial map.',
        ],
        'seed': seed, 'grid': list(grid), 'threads': torch.get_num_threads(),
        'versions': {'torch': torch.__version__, 'diffusers': diffusers.__version__,
                     'python': platform.python_version()},
        'model_config': dict(model.config),
        'model_parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
        'output_dtype': str(outputs['baseline'].dtype),
        'attention_processors': {name: type(processor).__name__
                                 for name, processor in model.attn_processors.items()},
        'normalization_modules': {name: type(module).__name__
                                  for name, module in model.named_modules()
                                  if isinstance(module, torch.nn.LayerNorm)},
        'parameter_transform': {'a': a.tolist(), 'c': c, 'negative_channel_vector': wrong.tolist()},
        'analytic_checks': {
            'common_shift_mean_vs_expected': errors(
                features['common_shift'].mean(-1), base_feature.mean(-1) + expected_shift),
            'redirected_mean_vs_expected': errors(features['redirected'].mean(-1), expected_shift),
            'layernorm_common_shift_error': layernorm_error,
        },
        'variants': records,
        'numerical_checks': {
            'max_abs_output_tolerance': tolerance, 'all_common_shift_variants_within_tolerance': invariance_pass,
            'negative_control_output_max_abs_exceeds_1e_minus_4':
                records['channel_dependent_negative_control']['output_vs_baseline']['max_abs'] > 1e-4,
        },
        'source_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths},
        'total_wall_seconds': time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=712)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run_audit(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(report['numerical_checks'], sort_keys=True))


if __name__ == '__main__':
    main()
