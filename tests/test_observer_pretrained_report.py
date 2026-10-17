"""Bounded CPU checks of saved evidence, causal contrasts and corrupt receipts."""
import copy
import json
import math
from pathlib import Path
import tempfile

import pytest
import torch

import scripts.observer_pretrained_report as report
from scripts.observer_pretrained_audit import ARMS, gain_record, observer_record, target_observer, tensor_hash
from scripts.observer_pretrained_audit import BASE, KREA, REQUIRED_SOURCES


def evidence_fixture():
    base = torch.tensor([1., 2.], dtype=torch.bfloat16)
    fixed = base + torch.tensor([.25, -.25], dtype=torch.bfloat16)
    gain_only = base + torch.tensor([2., 3.], dtype=torch.bfloat16)
    adaptive = gain_only + (fixed-base)
    outputs = dict(zip(ARMS, (base, base.clone(), adaptive, fixed, gain_only)))
    g0 = torch.ones(128)
    g1 = torch.ones(128)
    g1[16:] = 1.125
    geometry = dict(max_abs=.1, l2=math.sqrt(2), relative_l2=.2,
                    realized_common_channel_delta_l2=2., realized_transverse_delta_l2=math.sqrt(2),
                    realized_full_delta_l2=math.sqrt(6), transverse_fraction_of_full_delta=math.sqrt(2/6))
    records = {}
    for arm in ARMS:
        shifted = arm in ('shifted_g1', 'shifted_g0')
        value = g1 if arm in ('shifted_g1', 'original_g1') else g0
        records[arm] = dict(gain=gain_record(value), consumed_gain_bfloat16_sha256=tensor_hash(value.bfloat16()),
            attention_calls=57, attention_query_dtype='torch.bfloat16', incoming_features_equal_baseline=True,
            realized_feature_sha256=('b' if shifted else 'a')*64,
            centered_feature_error=copy.deepcopy(geometry) if shifted else None,
            target_error={'placeholder': True} if shifted else None)
    completion = dict(completed_forwards=5, exact_repeat_equal=True, arms=copy.deepcopy(records),
                      on_trajectory=False, efficacy_evaluated=False, novelty_established=False,
                      contrasts=report.recompute_contrasts(outputs),
                      gain_change_float32=report.difference_stats(g1, g0),
                      gain_change_bfloat16=report.difference_stats(g1.bfloat16(), g0.bfloat16()),
                      gain_quantization_dead=False, gain_changed_bfloat16_coordinates=112)
    return outputs, records, completion


def test_known_additive_mediation_is_recomputed():
    outputs, records, completion = evidence_fixture()
    contrasts = report.validate_completion(completion, outputs, records)
    assert contrasts['shift_gain_interaction']['l2'] == 0
    assert contrasts['fixed_g0_shift_leakage']['max_abs'] == .25
    assert contrasts['gain_only_original_g1_minus_original_g0']['l2'] == pytest.approx(math.sqrt(13))
    ratios = report.continuous_measures(outputs, contrasts)
    assert ratios['gain_only_to_fixed_gain_residual']['ratio'] == pytest.approx(math.sqrt(104))
    assert ratios['interaction_to_gain_only']['ratio'] == 0


def test_zero_effects_have_explicit_zero_denominator_handling():
    outputs = {arm: torch.zeros(2, dtype=torch.bfloat16) for arm in ARMS}
    contrasts = report.recompute_contrasts(outputs)
    measures = report.continuous_measures(outputs, contrasts)
    assert measures['gain_only_to_fixed_gain_residual'] == dict(numerator=0., denominator=0.,
        ratio=None, zero_denominator=True, both_zero=True)
    assert measures['adaptive_and_gain_only_cosine'] is None
    assert report.safe_ratio(1., 0.)['ratio'] is None
    assert report.safe_ratio(1., 0.)['both_zero'] is False


def test_rejects_miscomputed_contrast():
    outputs, records, completion = evidence_fixture()
    completion['contrasts']['gain_only_original_g1_minus_original_g0']['l2'] += 1
    with pytest.raises(ValueError, match='numeric mismatch'):
        report.validate_completion(completion, outputs, records)


def test_rejects_corrupted_completion_receipt():
    outputs, records, completion = evidence_fixture()
    completion['arms']['original_g0']['incoming_features_equal_baseline'] = False
    with pytest.raises(ValueError, match='raw arm files'):
        report.validate_completion(completion, outputs, records)


def test_rejects_false_exact_repeat_claim():
    outputs, records, completion = evidence_fixture()
    outputs['repeat_original_g0'] = outputs['original_g0'] + 1
    with pytest.raises(ValueError, match='exact repeat'):
        report.validate_completion(completion, outputs, records)


def test_rejects_signed_zero_in_claimed_bitwise_repeat():
    outputs, records, completion = evidence_fixture()
    outputs['original_g0'] = torch.tensor([0., 2.], dtype=torch.bfloat16)
    outputs['repeat_original_g0'] = torch.tensor([-0., 2.], dtype=torch.bfloat16)
    assert torch.equal(outputs['original_g0'], outputs['repeat_original_g0'])
    completion['contrasts'] = report.recompute_contrasts(outputs)
    with pytest.raises(ValueError, match='raw tensor bytes'):
        report.validate_completion(completion, outputs, records)


@pytest.mark.parametrize('case', ['gain_crossing', 'consumed_cast', 'feature_crossing', 'feature_geometry', 'cast_vector'])
def test_rejects_wrong_crossings_or_realized_casts(case):
    _, records, _ = evidence_fixture()
    if case == 'gain_crossing':
        records['original_g1']['gain'] = copy.deepcopy(records['original_g0']['gain'])
    elif case == 'consumed_cast':
        records['original_g1']['consumed_gain_bfloat16_sha256'] = 'c'*64
    elif case == 'feature_crossing':
        records['shifted_g0']['realized_feature_sha256'] = 'c'*64
    elif case == 'feature_geometry':
        for arm in ('shifted_g1', 'shifted_g0'):
            records[arm]['centered_feature_error']['realized_full_delta_l2'] = 100
    else:
        records['shifted_g1']['gain']['cast_bfloat16'][20] = 1.5
    with pytest.raises(ValueError):
        report.validate_crossings(records)


def test_tensor_validation_rejects_hash_shape_and_nonfinite(tmp_path):
    path = tmp_path / 'output.pt'
    value = torch.tensor([[1., 2.]], dtype=torch.bfloat16)
    torch.save({'output': value}, path)
    loaded, receipt = report.verify_tensor(path, 'output', (1, 2), torch.bfloat16,
        file_digest=report.sha256(path), tensor_digest=tensor_hash(value))
    assert torch.equal(loaded, value) and receipt['independently_verified_on_cpu']
    with pytest.raises(ValueError, match='file hash'):
        report.verify_tensor(path, 'output', (1, 2), torch.bfloat16, file_digest='a'*64)
    with pytest.raises(ValueError, match='contents hash'):
        report.verify_tensor(path, 'output', (1, 2), torch.bfloat16, tensor_digest='a'*64)
    with pytest.raises(ValueError, match='shape/dtype'):
        report.verify_tensor(path, 'output', (2,), torch.bfloat16)
    invalid = tmp_path / 'nonfinite.pt'
    torch.save({'output': torch.tensor([float('nan')], dtype=torch.bfloat16)}, invalid)
    with pytest.raises(ValueError, match='Nonfinite'):
        report.verify_tensor(invalid, 'output', (1,), torch.bfloat16)


def test_requires_both_completed_prompts_before_reading_evidence(tmp_path):
    (tmp_path / 'prompt0').mkdir()
    (tmp_path / 'prompt0/completion.json').write_text('{}')
    # Even malformed first-prompt evidence cannot make an incomplete campaign valid.
    with pytest.raises(ValueError):
        report.verify_campaign_presence(tmp_path)
    with pytest.raises(ValueError):
        report.build_report(tmp_path, tmp_path/'absent-protocol.json', tmp_path/'absent-submission.json')


@pytest.fixture
def observer_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'GRID', (4, 128))
    monkeypatch.setattr(report, 'OBSERVER_SHAPE', (1, 512))
    generator = torch.Generator().manual_seed(123)
    mean = torch.randn(1, 512, generator=generator)
    target = target_observer(mean, (4, 128))
    shifted = target.bfloat16().float()
    _, records, _ = evidence_fixture()
    for arm in ARMS:
        actual = shifted if arm in ('shifted_g1', 'shifted_g0') else mean
        torch.save({'actual_observer': actual, 'target': target}, tmp_path/f'{arm}_observer.pt')
        records[arm]['observer'] = observer_record(actual, (4, 128))
        records[arm]['target_error'] = report.difference_stats(actual, target) if 'shifted' in arm else None
    return tmp_path, records, mean, target


def test_saved_observers_targets_and_crossings_are_verified(observer_evidence):
    path, records, _, _ = observer_evidence
    validation = report.verify_observers(path, records)
    assert validation['target_cpu_reconstruction_error']['max_abs'] == 0
    assert len(validation['files']) == 5


def test_observer_hash_corruption_is_rejected(observer_evidence):
    path, records, _, _ = observer_evidence
    records['shifted_g0']['observer']['sha256'] = 'c'*64
    with pytest.raises(ValueError, match='Observer hash'):
        report.verify_observers(path, records)


def test_numeric_comparison_does_not_accept_nan_or_bool_as_number():
    with pytest.raises(ValueError):
        report.compare_numbers(float('nan'), 0., 'metric')
    with pytest.raises(ValueError):
        report.compare_numbers(True, 1., 'metric')
    with pytest.raises(ValueError):
        report.compare_numbers({'one': 1}, {'two': 1}, 'schema')


@pytest.mark.parametrize('changed_key', [None, 'source_sha256', 'model', 'state', 'conditioning_sha256',
                                       'slurm_job_id', 'protocol_sha256'])
def test_manifest_provenance_rejects_drift(changed_key):
    protocol = dict(source_sha256={'worker.py': 'a'*64}, model={'revision': 'fixed'},
                    accepted_report_sha256='b'*64, accepted_receipt_sha256='c'*64,
                    endpoint_file_sha256={'0': 'd'*64}, allocation_numerics='deterministic_sega_update_only')
    prompt, state = {'id': 'dev0', 'prompt': 'caption'}, {'input_sha256': 'e'*64, 'noise_seed': 1729}
    accepted = {'torch': 'test', 'cuda': None, 'experiment_configuration': {'diffusers': 'test'}}
    manifest = dict(protocol_sha256='f'*64, prompt_index=0, prompt=prompt, state=state,
        source_sha256=protocol['source_sha256'], model=protocol['model'],
        accepted_report_sha256=protocol['accepted_report_sha256'], accepted_receipt_sha256=protocol['accepted_receipt_sha256'],
        endpoint_file_sha256=protocol['endpoint_file_sha256']['0'], allocation_numerics=protocol['allocation_numerics'],
        sigma=.5, guidance=4.5, resolution=4096, forward_calls_budget=5, arm_order=list(ARMS),
        observer_shift=report.SHIFT_SPEC, slurm_job_id='123', torch='test', cuda=None, diffusers='test',
        model_parameter_dtypes=['torch.bfloat16'], hardware='CPU fixture only', model_loading_seconds=0.,
        conditioning_sha256={key: 'a'*64 for key in ('encoder_hidden_states', 'pooled_projections', 'txt_ids', 'img_ids')})
    arguments = (protocol, 'f'*64, {'job_id': '123'}, 0, prompt, state, accepted)
    if changed_key is None:
        report.validate_manifest(manifest, *arguments)
    else:
        manifest[changed_key] = 'changed'
        with pytest.raises(ValueError):
            report.validate_manifest(manifest, *arguments)


@pytest.fixture
def complete_campaign(tmp_path, monkeypatch):
    """Exercise report orchestration with small tensors, no model or external data."""
    monkeypatch.setattr(report, 'GRID', (4, 128))
    monkeypatch.setattr(report, 'OBSERVER_SHAPE', (1, 512))
    monkeypatch.setattr(report, 'STATE_SHAPE', (1, 512, 64))
    digest = 'a'*64
    protocol = dict(schema_version=1, development_prompt_indices=[0, 1], resolution=4096,
        sigma=.5, guidance=4.5, noise_seeds={'0': 1729, '1': 1730}, noise_device='cpu',
        noise_dtype='float32', mixture_dtype='float32', model_dtype='bfloat16',
        forward_calls_per_prompt=5, allocation_numerics='deterministic_sega_update_only',
        observer_shift=report.SHIFT_SPEC, accepted_report='/unused/report.json',
        accepted_receipt='/unused/receipt.json', accepted_report_sha256=digest,
        accepted_receipt_sha256=digest, endpoint_file_sha256={'0': digest, '1': digest},
        prompt_file_sha256=digest, prompt_ids=['dev0', 'dev1'],
        source_sha256={name: digest for name in REQUIRED_SOURCES}, source_commit='unit-test-source',
        model=dict(transformer_revision=KREA, base_revision=BASE,
                   transformer_config_sha256=digest, base_model_index_sha256=digest))
    protocol_path = tmp_path/'protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    root = tmp_path/'job123'
    root.mkdir()
    submission = dict(protocol_sha256=report.sha256(protocol_path), job_id='123', source_commit='unit-test-source',
                      registry={'completion_files': [str(root/f'prompt{i}/completion.json') for i in (0, 1)]})
    submission_path = tmp_path/'submission.json'
    submission_path.write_text(json.dumps(submission))
    accepted = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                'experiment_configuration': {'diffusers': 'unit-test-version'}}
    endpoint = torch.zeros(512, 64, dtype=torch.bfloat16)

    def fake_verified_inputs(_protocol, index):
        return endpoint, {'id': f'dev{index}', 'prompt': f'unit test {index}'}, accepted

    def small_artificial_state(end, index):
        generator = torch.Generator().manual_seed(1729+index)
        state = (.5*end.float() + .5*torch.randn(end.shape, generator=generator)).bfloat16().unsqueeze(0)
        return state, {'on_trajectory': False, 'noise_seed': 1729+index,
                       'input_sha256': tensor_hash(state), 'shape': list(state.shape)}

    monkeypatch.setattr(report, 'load_verified_inputs', fake_verified_inputs)
    monkeypatch.setattr(report, 'artificial_state', small_artificial_state)
    for index in (0, 1):
        path = root/f'prompt{index}'
        path.mkdir()
        _, prompt, _ = fake_verified_inputs(protocol, index)
        state, state_meta = small_artificial_state(endpoint, index)
        torch.save({'input': state}, path/'input.pt')
        outputs, records, completion = evidence_fixture()
        outputs = {arm: value.repeat(16384).reshape(1, 512, 64) for arm, value in outputs.items()}
        mean = torch.randn(1, 512, generator=torch.Generator().manual_seed(123+index))
        target = target_observer(mean, (4, 128))
        for arm in ARMS:
            actual_mean = target.bfloat16().float() if arm in ('shifted_g1', 'shifted_g0') else mean
            torch.save({'actual_observer': actual_mean, 'target': target}, path/f'{arm}_observer.pt')
            torch.save({'output': outputs[arm]}, path/f'{arm}_output.pt')
            records[arm].update(observer=observer_record(actual_mean, (4, 128)),
                target_error=report.difference_stats(actual_mean, target) if arm in ('shifted_g1', 'shifted_g0') else None,
                output_file_sha256=report.sha256(path/f'{arm}_output.pt'), output_sha256=tensor_hash(outputs[arm]),
                forward_seconds_including_hook_diagnostics=1.)
            (path/f'{arm}.json').write_text(json.dumps(records[arm]))
        completion.update(arms=records, contrasts=report.recompute_contrasts(outputs))
        for name in ('completion.json', 'validation.json'):
            (path/name).write_text(json.dumps(completion))
        (path/'repeat_validation.json').write_text(json.dumps({'output': completion['contrasts']['exact_repeat'],
                                                             'replayed_gain_equal': True}))
        manifest = dict(protocol_sha256=submission['protocol_sha256'], prompt_index=index, prompt=prompt,
            state=state_meta, source_sha256=protocol['source_sha256'], model=protocol['model'],
            accepted_report_sha256=digest, accepted_receipt_sha256=digest, endpoint_file_sha256=digest,
            allocation_numerics=protocol['allocation_numerics'], sigma=.5, guidance=4.5, resolution=4096,
            forward_calls_budget=5, arm_order=list(ARMS), observer_shift=report.SHIFT_SPEC,
            slurm_job_id='123', torch=torch.__version__, cuda=torch.version.cuda, diffusers='unit-test-version',
            model_parameter_dtypes=['torch.bfloat16'], hardware='synthetic test only', model_loading_seconds=1.,
            input_file_sha256=report.sha256(path/'input.pt'), conditioning_sha256={
                key: digest for key in ('encoder_hidden_states', 'pooled_projections', 'txt_ids', 'img_ids')})
        (path/'manifest.json').write_text(json.dumps(manifest))
    return root, protocol_path, submission_path


def test_build_report_complete_synthetic_campaign(complete_campaign):
    result = report.build_report(*complete_campaign)
    assert result['completed_prompts'] == [0, 1]
    assert result['total_transformer_forwards'] == 10
    assert result['automatic_promotion'] is False and result['on_trajectory'] is False
    assert all(value['contrasts']['exact_repeat']['bitwise_equal'] for value in result['results'].values())
    assert result['runtime_receipts_not_independently_recomputed']


def test_archive_fresh_work_directory_contains_only_verified_json(complete_campaign):
    root, protocol_path, submission_path = complete_campaign
    result = report.build_report(root, protocol_path, submission_path)
    # This small temporary work-storage directory is removed after the test.
    # The scratch tensor fixtures are preserved by the project test harness.
    with tempfile.TemporaryDirectory(prefix='observer-report-test-', dir=report.ROOT.parent) as temporary:
        out = Path(temporary)/'archive'
        receipt = report.archive_report(result, root, out)
        assert receipt['report_sha256'] == report.sha256(out/'report.json')
        assert receipt['reporter_source_sha256'] == report.sha256(report.__file__)
        assert receipt['bulk_tensors_copied'] is False
        assert all(path.suffix == '.json' for path in out.rglob('*') if path.is_file())
        restored = json.loads((out/'report.json').read_text())
        assert restored['input_file_sha256'][str(protocol_path.resolve())] == report.sha256(protocol_path)
        assert restored['input_file_sha256'][str(submission_path.resolve())] == report.sha256(submission_path)
        assert restored['inputs']['protocol'] == json.loads(protocol_path.read_text())
        assert restored['completed_prompts'] == [0, 1]
        for source, digest in receipt['source_json_sha256'].items():
            assert report.sha256(source) == digest
        for index in (0, 1):
            assert json.loads((out/f'prompt{index}/completion.json').read_text()) == restored['raw_json'][f'prompt{index}']['completion.json']
        with pytest.raises(ValueError, match='fresh directory'):
            report.archive_report(result, root, out)


@pytest.mark.parametrize('case', ['submission_hash', 'saved_input', 'submission_root'])
def test_build_report_rejects_broken_campaign_linkage(complete_campaign, case):
    root, protocol_path, submission_path = complete_campaign
    if case == 'saved_input':
        with (root/'prompt1/input.pt').open('ab') as handle:
            handle.write(b'corrupt')
    else:
        submission = json.loads(submission_path.read_text())
        if case == 'submission_hash':
            submission['protocol_sha256'] = 'b'*64
        else:
            submission['registry']['completion_files'][1] = str(root/'prompt2/completion.json')
        submission_path.write_text(json.dumps(submission))
    with pytest.raises(ValueError):
        report.build_report(root, protocol_path, submission_path)
