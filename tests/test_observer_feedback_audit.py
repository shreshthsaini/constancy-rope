"""Actual SEGA/FLUX mediation controls, with parameters fixed before execution."""

import pytest
import torch

from scripts.observer_feedback_audit import force_channel_mean, run_audit


@pytest.fixture(scope='module')
def audit():
    torch.set_num_threads(2)
    return run_audit()


def test_actual_sega_and_installed_processor(audit):
    assert audit['grid'] == [4, 128]
    assert audit['original']['diagnostics']['spatial_scale'] == 2
    assert audit['original']['diagnostics']['ntk_factor'] > 1
    assert set(audit['attention_processors'].values()) == {'FluxGainAttnProcessor'}
    assert len(audit['attention_processors']) == 2
    assert audit['original']['qk_gain'][:4] == [1.] * 4
    assert audit['shifted']['qk_gain'][:4] == [1.] * 4


def test_force_mean_preserves_channel_centered_features(audit):
    assert audit['target_mean_error']['max_abs'] < 1e-13
    assert audit['channel_centered_feature_error']['max_abs'] < 1e-13


def test_fixed_gain_shift_preserves_outputs(audit):
    assert audit['contrasts']['fixed_gain_shift_vs_original']['max_abs'] < 1e-10
    assert audit['runs']['fixed_original']['executed_qk_gain'] == audit['runs']['fixed_shifted']['executed_qk_gain']


def test_adaptive_original_reproduces_fixed_original(audit):
    assert audit['contrasts']['adaptive_original_vs_fixed_original']['max_abs'] == 0
    assert audit['runs']['adaptive_original']['executed_qk_gain'] == audit['original']['qk_gain']
    assert audit['runs']['adaptive_shifted']['executed_qk_gain'] == audit['shifted']['qk_gain']


def test_observer_intervention_changes_gains_and_adaptive_output(audit):
    assert audit['gain_change']['max_abs'] > 1e-4
    assert audit['contrasts']['adaptive_shift_vs_original']['max_abs'] > 1e-5


def test_gain_mediation_reproduces_shifted_adaptive_output(audit):
    assert audit['contrasts']['mediation_original_shifted_gain_vs_adaptive_shifted']['max_abs'] < 1e-10
    assert audit['runs']['original_with_shifted_gain']['executed_qk_gain'] == audit['runs']['adaptive_shifted']['executed_qk_gain']
    assert not audit['runs']['original_with_shifted_gain']['shifted']
    assert not audit['runs']['original_with_shifted_gain']['adaptive']


def test_forcing_map_rejects_implicit_broadcast():
    with pytest.raises(ValueError, match='batch and token'):
        force_channel_mean(torch.zeros(1, 4, 3), torch.zeros(4))
