"""Test common-channel shifts on the actual tiny diffusers architecture."""

import pytest
import torch

from scripts.observer_shift_audit import feature_summary, run_audit


@pytest.fixture(scope='module')
def audit():
    torch.set_num_threads(2)
    return run_audit()


def test_actual_architecture_and_precision(audit):
    assert audit['model_config']['num_layers'] == 1
    assert audit['model_config']['num_single_layers'] == 1
    assert audit['model_parameter_dtypes'] == ['torch.float64']
    assert audit['output_dtype'] == 'torch.float64'
    norms = audit['normalization_modules']
    for name in ('transformer_blocks.0.norm1.norm', 'transformer_blocks.0.norm2',
                 'single_transformer_blocks.0.norm.norm', 'norm_out.norm'):
        assert norms[name] == 'LayerNorm'
    assert set(audit['attention_processors'].values()) == {'FluxAttnProcessor'}


@pytest.mark.parametrize('name', ['common_shift', 'mean_zeroed', 'redirected'])
def test_common_channel_symmetry(audit, name):
    record = audit['variants'][name]
    assert record['output_vs_baseline']['max_abs'] < 1e-10
    assert record['channel_centered_feature_vs_baseline']['max_abs'] < 1e-13
    baseline = audit['variants']['baseline']['feature_summary']
    assert record['feature_summary']['channel_centered_feature_energy'] == pytest.approx(
        baseline['channel_centered_feature_energy'], rel=1e-13)


def test_observer_zeroing_redirection_and_layernorm(audit):
    variants = audit['variants']
    assert variants['mean_zeroed']['feature_summary']['max_abs_channel_mean'] < 1e-14
    assert variants['baseline']['feature_summary']['spatial_centered_channel_mean_energy'] > .01
    assert variants['mean_zeroed']['feature_summary']['spatial_centered_channel_mean_energy'] < 1e-26
    for record in audit['analytic_checks'].values():
        assert record['max_abs'] < 1e-13
    assert variants['redirected']['feature_summary']['spatial_centered_channel_mean_energy'] > 1


def test_negative_control_is_appreciable(audit):
    record = audit['variants']['channel_dependent_negative_control']
    assert record['output_vs_baseline']['max_abs'] > 1e-4
    assert record['channel_centered_feature_vs_baseline']['max_abs'] > .1
    # This zero-mean channel direction also gives the converse: the observer
    # stays fixed even though the model output changes appreciably.
    observed = torch.tensor(record['feature_summary']['channel_mean_map'], dtype=torch.float64)
    baseline = torch.tensor(audit['variants']['baseline']['feature_summary']['channel_mean_map'],
                            dtype=torch.float64)
    assert torch.allclose(observed, baseline, atol=1e-13, rtol=1e-13)


def test_spectrum_is_of_channel_mean_and_spatial_dc_is_removed():
    signal = torch.tensor([1., -1., 1., -1.], dtype=torch.float64)
    features = (signal + 5).reshape(1, 4, 1).expand(1, 4, 3)
    summary = feature_summary(features, (2, 2))
    assert summary['channel_centered_feature_energy'] == 0
    assert summary['spatial_centered_channel_mean_energy'] == 4
    spectrum = torch.tensor(summary['spatial_centered_channel_mean_fft2_power'])
    assert spectrum[0, 0] == 0
    assert spectrum.sum() == 4
