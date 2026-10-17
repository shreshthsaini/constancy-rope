"""CPU corruption checks for bounded pilot scope, identities and matched inputs."""
import copy
from types import SimpleNamespace

import pytest
import torch
from crope.crossmodal_phase import NUMERICAL_POLICY

from crope.runtime import fingerprint
from crope.utils import image_seed
from scripts import crossmodal_phase_pilot as pilot


@pytest.fixture
def protocol():
    return {**copy.deepcopy(pilot.FIXED),
        'native_protocol': {'path': '/scratch/accepted.json', 'sha256': pilot.NATIVE_PROTOCOL_SHA256},
        'source_sha256': {name: 'a'*64 for name in pilot.NEW_SOURCES},
        'expected_sigmas': pilot.expected_sigmas(28), 'expected_smoke_sigmas': pilot.expected_sigmas(4)}


def test_only_two_existing_prompts_and_fixed_arms(protocol):
    pilot.validate_protocol(protocol, 0)
    pilot.validate_protocol(protocol, 1)
    assert pilot.ARMS == ('baseline', 'fixedmass', 'unnormalized')
    assert protocol['expected_sigmas'][0] == protocol['expected_smoke_sigmas'][0] == 1.


@pytest.mark.parametrize('index', [-1, 2, 8, 64, True, '0'])
def test_no_fresh_prompts_or_boolean_index(protocol, index):
    with pytest.raises(ValueError, match='prompt indices'):
        pilot.validate_protocol(protocol, index)


@pytest.mark.parametrize('key,value', [('steps', 29), ('seed', 1), ('seed', False), ('guidance', 3.5),
    ('arms', ['baseline', 'fixedmass']), ('arms', ['baseline', 'fixedmass', 'fixedmass']),
    ('arms', ['fixedmass', 'baseline', 'unnormalized']), ('text_tokens', 256),
    ('resolution', [2048, 4096]), ('native_grid', [63, 63]), ('query_chunk_size', 1024),
    ('backend', 'math'), ('text_ids', 'assumed_zero'), ('coordinate_map', 'p_over_4'),
    ('unnormalized_implementation', 'subtractive_cancellation')])
def test_settings_drift_rejected(protocol, key, value):
    protocol[key] = value
    with pytest.raises(ValueError, match='protocol mismatch'):
        pilot.validate_protocol(protocol, 0)


@pytest.mark.parametrize('kind', ['wrong_native', 'relative_pin', 'missing_source', 'unsafe_source', 'bad_schedule'])
def test_source_and_dependency_drift(protocol, kind):
    if kind == 'wrong_native': protocol['native_protocol']['sha256'] = 'b'*64
    elif kind == 'relative_pin': protocol['native_protocol']['path'] = 'native.json'
    elif kind == 'missing_source': protocol['source_sha256'].pop('crope/crossmodal_phase.py')
    elif kind == 'unsafe_source': protocol['source_sha256']['../other.py'] = 'c'*64
    else: protocol['expected_smoke_sigmas'][1] = float('nan')
    with pytest.raises(ValueError):
        pilot.validate_protocol(protocol, 0)


def test_parent_extra_pins_and_documentary_fields_allowed(protocol):
    protocol['source_sha256']['scripts/crossmodal_phase_score.py'] = 'c'*64
    protocol['review_plan'] = {'path': '/work/review.json', 'sha256': 'd'*64}
    protocol['scoring_reference'] = {'path': '/work/scoring.json', 'sha256': 'e'*64}
    pilot.validate_protocol(protocol, 0)


def test_native_input_audit_saves_actual_bytes_and_detects_changed_arm(tmp_path):
    torch.set_num_threads(2)
    value = torch.zeros(1, 4096, 64, dtype=torch.bfloat16)
    first = pilot.PackedInputAudit(4096)
    first.path = tmp_path/'first.pt'
    first(None, (), {'hidden_states': value})
    saved = torch.load(first.path, weights_only=True)['initial_transformer_input']
    assert torch.equal(saved, value)
    second = pilot.PackedInputAudit(4096, first.receipt['tensor_sha256'])
    second.path = tmp_path/'second.pt'
    second(None, (), {'hidden_states': value.clone()})
    value[0, 0, 0] = 1.
    changed = pilot.PackedInputAudit(4096, first.receipt['tensor_sha256'])
    changed.path = tmp_path/'changed.pt'
    with pytest.raises(ValueError, match='Initial input differs'):
        changed(None, (), {'hidden_states': value})


def test_input_audit_fails_nonfinite_and_wrong_geometry(tmp_path):
    audit = pilot.PackedInputAudit(16)
    audit.path = tmp_path/'input.pt'
    for value in (torch.zeros(1, 15, 64, dtype=torch.bfloat16),
                  torch.full((1, 16, 64), float('nan'), dtype=torch.bfloat16),
                  torch.zeros(1, 16, 64, dtype=torch.float32)):
        with pytest.raises(ValueError, match='Initial transformer input'):
            audit(None, (), {'hidden_states': value})
    assert not audit.path.exists()


def geometry_kwargs(grid):
    h, w = grid
    return {'hidden_states': torch.zeros(1, h*w, 64, dtype=torch.bfloat16),
        'txt_ids': torch.zeros(512, 3), 'img_ids': torch.stack((torch.zeros(h*w),
            torch.arange(h).repeat_interleave(w), torch.arange(w).repeat(h)), dim=-1),
        'encoder_hidden_states': torch.zeros(1, 512, 4096, dtype=torch.bfloat16),
        'pooled_projections': torch.zeros(1, 768, dtype=torch.bfloat16)}


def test_actual_text_ids_and_coordinate_order_checked():
    experiment = SimpleNamespace(grid=(2, 3), step=0, sigma=1.)
    good = geometry_kwargs(experiment.grid)
    audit = pilot.GeometryGainAudit(experiment)
    audit.before(None, (), good)
    assert audit.geometry[0]['text_ids_all_zero'] is True
    for change in ('txt', 'grid', 'sequence'):
        kwargs = geometry_kwargs(experiment.grid)
        if change == 'txt': kwargs['txt_ids'][17, 2] = 1
        elif change == 'grid': kwargs['img_ids'] = kwargs['img_ids'].flip(0)
        else: kwargs['encoder_hidden_states'] = kwargs['encoder_hidden_states'][:, :511]
        with pytest.raises(ValueError):
            pilot.GeometryGainAudit(experiment).before(None, (), kwargs)


def test_initial_conditioning_identity_is_checked_before_attention():
    experiment = SimpleNamespace(grid=(2, 2), step=0, sigma=1.)
    first = pilot.GeometryGainAudit(experiment)
    first.before(None, (), geometry_kwargs(experiment.grid))
    second = pilot.GeometryGainAudit(experiment, first.conditioning)
    changed = geometry_kwargs(experiment.grid)
    changed['encoder_hidden_states'][0, 0, 0] = 1
    with pytest.raises(ValueError, match='conditioning differs'):
        second.before(None, (), changed)


def test_endpoint_zero_and_dtype_cases_do_not_fake_relative_error():
    zero = torch.zeros(5, 2, dtype=torch.bfloat16)
    assert pilot.endpoint_comparison(zero, zero)['relative_l2'] == 0
    assert pilot.endpoint_comparison(zero, zero)['cosine'] == 1
    changed = zero.clone(); changed[0, 0] = 1
    assert pilot.endpoint_comparison(zero, changed)['relative_l2'] is None
    assert not pilot.endpoint_comparison(zero, changed)['byte_exact']
    with pytest.raises(ValueError, match='Invalid endpoint comparison'):
        pilot.endpoint_comparison(zero, zero.float())


def test_finite_endpoint_discrepancy_against_independent_norm():
    ref = torch.tensor([[3., 4.]], dtype=torch.bfloat16)
    value = torch.tensor([[0., 4.]], dtype=torch.bfloat16)
    result = pilot.endpoint_comparison(ref, value)
    assert result['relative_l2'] == pytest.approx(3/5)
    assert result['cosine'] == pytest.approx(.8)
    assert result['max_abs'] == 3


@pytest.fixture
def records(protocol, tmp_path):
    row = {'id': pilot.native.DEV_IDS[0], 'prompt': 'The frozen caption.'}
    initial = pilot.save_tensor(tmp_path/'initial.pt', 'initial_transformer_input',
                                torch.zeros(1, 4096, 64, dtype=torch.bfloat16))
    endpoint = pilot.save_tensor(tmp_path/'endpoint.pt', 'endpoint_latent',
                                 torch.zeros(4096, 64, dtype=torch.bfloat16))
    result = []
    for arm in ('deployed', *pilot.ARMS):
        config = dict(method=arm, phase_arm=arm, prompt_id=row['id'], prompt=row['prompt'], seed=0,
            derived_seed=image_seed(row['id'], 0), steps=4, guidance=4.5, dtype='bfloat16',
            res='1024x1024', grid=[64, 64], phase_source_sha256='a'*64,
            phase_protocol_sha256='d'*64, pilot_source_sha256=copy.deepcopy(protocol['source_sha256']))
        phase = dict(mode=arm, grid=[64, 64], text_tokens=512, query_chunk_size=pilot.FIXED['query_chunk_size'],
            backend=pilot.FIXED['backend'], fallback=None, geometry_id_validations=4,
            native_identity_baseline_path=True, all_text_keys_including_padding=True,
            calls=57*4, per_block_calls={f'transformer_blocks.{i}.attn.processor': 4 for i in range(19)} |
            {f'single_transformer_blocks.{i}.attn.processor': 4 for i in range(38)},
            kernel_calls=dict(full_joint_calls=57*4, old_text_calls=0, new_text_calls=0, image_only_calls=0), mass_ranges={})
        phase['numerical_policy'] = copy.deepcopy(NUMERICAL_POLICY)
        result.append(dict(complete=True, config=config, config_fingerprint=fingerprint(config),
            sigmas=list(protocol['expected_smoke_sigmas']),
            gains=[dict(step=i, sigma=s, mode='sega_official') for i, s in enumerate(protocol['expected_smoke_sigmas'])],
            runtime_audit=dict(gains=[{'sigma': s} for s in protocol['expected_smoke_sigmas']],
                geometry=[dict(sigma=s, text_ids_all_zero=True) for s in protocol['expected_smoke_sigmas']],
                conditioning_sha256={'encoder_hidden_states': 'b'*64}),
            phase=phase,
            initial_transformer_input=dict(initial), endpoint_latent=dict(endpoint),
            seconds=.5, peak_memory_bytes=1024))
    return result, row


def test_complete_arm_set_and_native_inputs(protocol, records):
    values, row = records
    pilot.validate_arm_set(values, row, protocol, smoke=True)


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'order', 'seed', 'caption', 'sigma', 'block',
    'text_ids', 'input_hash', 'input_file', 'conditioning', 'endpoint_shape', 'phase_hash', 'fingerprint'])
def test_corrupt_completed_arms_rejected(protocol, records, kind):
    values, row = records
    if kind == 'missing': values.pop()
    elif kind == 'duplicate': values[2] = copy.deepcopy(values[1])
    elif kind == 'order': values[1], values[2] = values[2], values[1]
    elif kind == 'seed': values[1]['config']['derived_seed'] += 1
    elif kind == 'caption': values[1]['config']['prompt'] = 'New held-out caption'
    elif kind == 'sigma': values[1]['sigmas'][2] = 0.
    elif kind == 'block': values[1]['phase']['per_block_calls']['single_transformer_blocks.0.attn.processor'] = 3
    elif kind == 'text_ids': values[1]['runtime_audit']['geometry'][2]['text_ids_all_zero'] = False
    elif kind == 'input_hash': values[1]['initial_transformer_input']['tensor_sha256'] = 'c'*64
    elif kind == 'input_file': values[1]['initial_transformer_input']['file_sha256'] = 'c'*64
    elif kind == 'conditioning': values[1]['runtime_audit']['conditioning_sha256']['encoder_hidden_states'] = 'c'*64
    elif kind == 'endpoint_shape': values[1]['endpoint_latent']['shape'] = [65536, 64]
    elif kind == 'phase_hash': values[1]['config']['phase_source_sha256'] = 'c'*64
    else: values[1]['config_fingerprint'] = 'c'*64
    # Refreshing a fingerprint must not bypass independent field expectations.
    if kind in ('seed', 'caption', 'phase_hash'):
        values[1]['config_fingerprint'] = fingerprint(values[1]['config'])
    with pytest.raises(ValueError):
        pilot.validate_arm_set(values, row, protocol, smoke=True)


@pytest.mark.parametrize('field,value', [('mode', 'fixedmass'), ('grid', [256, 256]), ('text_tokens', 511),
    ('fallback', 'math'), ('geometry_id_validations', 3), ('all_text_keys_including_padding', False),
    ('native_identity_baseline_path', False), ('query_chunk_size', 4096), ('backend', 'public_sdpa')])
def test_complete_counts_cannot_hide_wrong_attention_arm(protocol, records, field, value):
    values, row = records
    values[1]['phase'][field] = value
    with pytest.raises(ValueError, match='Attention arm/geometry/backend'):
        pilot.validate_arm_set(values, row, protocol, smoke=True)


def test_declared_native_noop_cannot_run_unrequested_branches(protocol, records):
    values, row = records
    values[2]['phase']['kernel_calls']['old_text_calls'] = 4
    with pytest.raises(ValueError, match='kernel branches'):
        pilot.validate_arm_set(values, row, protocol, smoke=True)


def test_protocol_pin_checked_independently_of_refreshed_fingerprint(protocol, records):
    values, row = records
    values[1]['config']['phase_protocol_sha256'] = 'e'*64
    values[1]['config_fingerprint'] = fingerprint(values[1]['config'])
    with pytest.raises(ValueError, match='protocol pin'):
        pilot.validate_arm_set(values, row, protocol, smoke=True, protocol_hash='d'*64)


def test_actual_saved_input_hash_cannot_be_forged_by_editing_every_receipt(protocol, records):
    values, row = records
    for value in values:
        value['initial_transformer_input']['tensor_sha256'] = 'c'*64
    with pytest.raises(ValueError, match='Saved tensor disagrees'):
        pilot.validate_arm_set(values, row, protocol, smoke=True)


@pytest.mark.parametrize('arm', ['baseline', 'fixedmass', 'unnormalized'])
def test_production_phase_branch_counts_and_mass_tolerance(records, arm):
    values, _ = records
    phase = copy.deepcopy(values[1]['phase'])
    phase.update(mode=arm, grid=[256, 256], native_identity_baseline_path=False,
                 calls=57*28, geometry_id_validations=28)
    phase['per_block_calls'] = {k: 28 for k in phase['per_block_calls']}
    branch = 0 if arm == 'baseline' else 57*28
    phase['kernel_calls'] = dict(full_joint_calls=57*28, new_text_calls=branch,
        old_text_calls=branch if arm == 'fixedmass' else 0,
        image_only_calls=branch if arm == 'unnormalized' else 0)
    if arm != 'baseline':
        ceiling = float(torch.tensor(NUMERICAL_POLICY['log_mass_upper_tolerance']).exp()) if arm == 'fixedmass' else 1.
        phase['mass_ranges'] = {'old_text_mass' if arm == 'fixedmass' else 'new_text_mass': [0., ceiling]}
    pilot.phase_receipt_valid(phase, 28, arm, [256, 256])
    phase['kernel_calls']['new_text_calls'] += 1
    with pytest.raises(ValueError, match='kernel branches'):
        pilot.phase_receipt_valid(phase, 28, arm, [256, 256])
