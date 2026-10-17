"""CPU-only geometry/provenance tests. No pretrained model is loaded."""
import copy

import pytest
import torch

from scripts.observer_pretrained_audit import (
    ARMS, BASE, KREA, REQUIRED_SOURCES, SHIFT_SPEC, artificial_state,
    contrast_summary, feature_delta_geometry, shift_features, target_observer,
    validate_protocol, validate_report_metadata, validate_repeat,
)


def protocol_fixture():
    digest = 'a' * 64
    return dict(schema_version=1, development_prompt_indices=[0, 1], resolution=4096,
                sigma=.5, guidance=4.5, noise_seeds={'0': 1729, '1': 1730},
                noise_device='cpu', noise_dtype='float32', mixture_dtype='float32',
                model_dtype='bfloat16', forward_calls_per_prompt=5,
                allocation_numerics='deterministic_sega_update_only', observer_shift=dict(SHIFT_SPEC),
                accepted_report='/example/report.json', accepted_receipt='/example/archive_receipt.json',
                accepted_report_sha256=digest, accepted_receipt_sha256=digest,
                endpoint_file_sha256={'0': digest, '1': digest}, prompt_file_sha256=digest,
                prompt_ids=['dev0', 'dev1'], source_sha256={name: digest for name in REQUIRED_SOURCES},
                model=dict(transformer_revision=KREA, base_revision=BASE,
                           transformer_config_sha256=digest, base_model_index_sha256=digest))


def report_fixture(protocol):
    prompts = [dict(id=f'dev{i}', prompt=f'example prompt {i}') for i in (0, 1)]
    report = {'automatic_promotion': False, 'inputs': {'development_prompts': prompts,
              'prompt_file_sha256': protocol['prompt_file_sha256']}, 'stages': {}}
    for index in (0, 1):
        for stage in ('smoke', '4k'):
            manifest = dict(prompt_index=index, prompt=prompts[index], seed=0, guidance=4.5,
                            res=[2048 if stage == 'smoke' else 4096] * 2,
                            prompt_file_sha256=protocol['prompt_file_sha256'], **protocol['model'],
                            experiment_configuration=dict(allocation_numerics=protocol['allocation_numerics'],
                                                          dtype='bfloat16', batch_size=1, method='sega_official'))
            done = dict(gpu_probe_executed=True, trajectory_bitwise_equal=True,
                        baseline_history_equal=True, baseline_sha256='b'*64, probed_sha256='b'*64)
            report['stages'][f'prompt{index}/{stage}'] = {
                'summary': {'manifest': manifest},
                'raw_json': {'manifest.json': copy.deepcopy(manifest),
                             'completion.json': done, 'validation.json': copy.deepcopy(done)},
                'endpoint_verification': {'independently_verified_on_cpu': True,
                    'shape': [(128 if stage == 'smoke' else 256)**2, 64], 'dtype': 'torch.bfloat16',
                    'tensor_sha256': 'b'*64, 'file_sha256': protocol['endpoint_file_sha256'][str(index)]}}
    return report


def test_valid_protocol_and_four_stage_report():
    protocol = protocol_fixture()
    validate_protocol(protocol)
    assert len(validate_report_metadata(report_fixture(protocol), protocol)) == 2


@pytest.mark.parametrize('key,value', [('sigma', .4), ('forward_calls_per_prompt', 4),
    ('development_prompt_indices', [0, 2]), ('noise_dtype', 'bfloat16'),
    ('noise_seeds', {'0': 0, '1': 1}), ('allocation_numerics', 'ordinary')])
def test_rejects_protocol_drift(key, value):
    protocol = protocol_fixture()
    protocol[key] = value
    with pytest.raises(ValueError):
        validate_protocol(protocol)


@pytest.mark.parametrize('case', ['missing_stage', 'failed_repeat', 'wrong_model', 'wrong_prompt',
                                 'wrong_shape', 'wrong_endpoint', 'missing_completion'])
def test_rejects_incomplete_or_mismatched_evidence(case):
    protocol = protocol_fixture()
    report = report_fixture(protocol)
    stage = report['stages']['prompt1/4k']
    if case == 'missing_stage':
        report['stages'].pop('prompt1/smoke')
    elif case == 'failed_repeat':
        for name in ('completion.json', 'validation.json'):
            stage['raw_json'][name]['trajectory_bitwise_equal'] = False
    elif case in ('wrong_model', 'wrong_prompt'):
        key, value = ('transformer_revision', 'wrong') if case == 'wrong_model' else ('prompt_index', 5)
        stage['summary']['manifest'][key] = value
        stage['raw_json']['manifest.json'][key] = value
    elif case == 'wrong_shape':
        stage['endpoint_verification']['shape'] = [1, 65536, 64]
    elif case == 'wrong_endpoint':
        stage['endpoint_verification']['file_sha256'] = 'c'*64
    else:
        stage['raw_json'].pop('completion.json')
    with pytest.raises(ValueError):
        validate_report_metadata(report, protocol)


def test_target_uses_frozen_harmonic_and_population_scale():
    mean = torch.arange(512, dtype=torch.float32).reshape(1, -1) / 100 + .7
    target = target_observer(mean, (4, 128))
    assert target.mean() == pytest.approx(mean.mean().item(), abs=1e-6)
    assert target.std(correction=0) == pytest.approx(mean.std(correction=0).item(), abs=1e-6)
    # Harmonic ratios are specified rather than selected from output responses.
    power = torch.fft.fft2((target-target.mean()).reshape(4, 128)).abs().square()
    assert power[1, 0] / power[0, 3] == pytest.approx(.3**2, rel=1e-5)


def test_bf16_shift_is_add_float32_then_cast_and_reports_transverse_component():
    generator = torch.Generator().manual_seed(18)
    features = torch.randn(1, 512, 20, generator=generator).bfloat16()
    target = target_observer(features.float().mean(-1), (4, 128))
    shifted = shift_features(features, target)
    expected = (features.float() + (target-features.float().mean(-1))[..., None]).bfloat16()
    assert torch.equal(shifted, expected)
    geometry = feature_delta_geometry(features, shifted)
    assert geometry['realized_transverse_delta_l2'] > 0
    assert geometry['realized_full_delta_l2']**2 == pytest.approx(
        geometry['realized_transverse_delta_l2']**2 + geometry['realized_common_channel_delta_l2']**2,
        rel=1e-12)


def test_four_arm_contrast_and_interaction_math():
    a = torch.tensor([1., 2.])
    d = a + torch.tensor([.25, -.25])
    c = a + torch.tensor([2., 3.])
    b = c + (d-a)
    outputs = dict(zip(ARMS, (a, a.clone(), b, d, c)))
    contrasts = contrast_summary(outputs)
    assert contrasts['exact_repeat']['bitwise_equal']
    assert contrasts['fixed_g0_shift_leakage']['max_abs'] == .25
    assert contrasts['mediation_residual_shifted_g1_minus_original_g1']['max_abs'] == .25
    assert contrasts['shift_gain_interaction']['l2'] == 0
    with pytest.raises(ValueError, match='Five arms'):
        contrast_summary({key: value for key, value in outputs.items() if key != ARMS[-1]})


def test_noop_failure_stops_before_intervention_arms():
    original = torch.tensor([1., 2.], dtype=torch.bfloat16)
    validate_repeat(original, original.clone(), {'gain': [1.]}, {'gain': [1.]})
    with pytest.raises(ValueError, match='remaining interventions skipped'):
        validate_repeat(original, original + 1, {'gain': [1.]}, {'gain': [1.]})
    with pytest.raises(ValueError, match='remaining interventions skipped'):
        validate_repeat(original, original.clone(), {'gain': [1.]}, {'gain': [2.]})


def test_state_sampling_is_cpu_fp32_and_reproducible():
    endpoint = torch.zeros(65536, 64, dtype=torch.bfloat16)
    first, info = artificial_state(endpoint, 0)
    second, other = artificial_state(endpoint, 0)
    assert first.shape == (1, 65536, 64) and first.dtype == torch.bfloat16
    assert torch.equal(first, second) and info == other and info['on_trajectory'] is False
    generator = torch.Generator(device='cpu').manual_seed(1729)
    expected = (.5 * torch.randn((65536, 64), generator=generator, dtype=torch.float32)).bfloat16()
    assert torch.equal(first[0], expected)


@pytest.mark.parametrize('shape,dtype,index', [((1, 65536, 64), torch.bfloat16, 0),
    ((65536, 63), torch.bfloat16, 0), ((65536, 64), torch.float32, 0),
    ((65536, 64), torch.bfloat16, 2)])
def test_rejects_wrong_endpoint_shape_dtype_or_prompt(shape, dtype, index):
    with pytest.raises(ValueError):
        artificial_state(torch.zeros(shape, dtype=dtype), index)
