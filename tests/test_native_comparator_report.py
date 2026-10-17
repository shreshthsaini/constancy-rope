"""Focused checks for an independent native-config expectation."""
import copy
import hashlib

import pytest

from crope.rope import FluxRoPE
from crope.runtime import fingerprint
from crope.utils import image_seed
from scripts import native_comparator_report as report


@pytest.fixture
def configs(monkeypatch):
    row = {'id': 'fixed-development-row', 'prompt': 'Two people hold hands.'}
    accepted = {'torch': 'pinned-torch', 'diffusers': 'pinned-diffusers'}
    reference = dict(schema_version=1, method='sega_official', res='4096x4096',
        prompt_id=row['id'], prompt=row['prompt'], seed=0, derived_seed=image_seed(row['id'], 0),
        steps=28, guidance=4.5, model='black-forest-labs/FLUX.1-dev', model_revision=report.BASE,
        dtype='bfloat16', batch_size=1, attention='pytorch_sdpa_flash', **accepted,
        code_revision='a'*40, implementation_sha256='b'*64, grid=[256,256], eta=1., target='profile',
        temperature='logN', profile_sha256=None, static_profile_sha256=None,
        rope=FluxRoPE(method='sega_official', grid=(256,256)).metadata(), probe_band=None, probe_factor=1.5,
        scale_choice='spatial axes per-side; scalar temperature geometric-mean side', side_scale=[4.,4.],
        area_scale=16., sega_sref_area_alternative=16**.08,
        scheduler={**report.SCHEDULER_EXPECTATION, 'max_image_seq_len':65536,
                   '_use_default_values':['invert_sigmas','shift_terminal'], '_class_name':'FlowMatchEulerDiscreteScheduler'})
    native = copy.deepcopy(reference)
    native.update(res='1024x1024', grid=[64,64], side_scale=[1.,1.], area_scale=1.,
                  sega_sref_area_alternative=1., rope=FluxRoPE(method='sega_official',grid=(64,64)).metadata())
    native['scheduler']['max_image_seq_len'] = 4096
    native['scheduler']['_use_default_values'].reverse()
    protocol = {'source_sha256': {'frozen.py':'c'*64}}
    monkeypatch.setattr(report, 'verify_runtime_revision', lambda revision, sources: None)
    monkeypatch.setattr('crope.runtime.implementation_hash', lambda: 'b'*64)
    return native, row, protocol, accepted, reference


def test_independent_expectation_accepts_only_bookkeeping_default_order(configs):
    candidate, row, protocol, accepted, reference = configs
    before = copy.deepcopy(reference)
    expected = report.expected_native_config(candidate, row, protocol, accepted, reference)
    assert expected == candidate and expected is not candidate
    assert reference == before  # Pinned reference must never be mutated.
    assert fingerprint(expected) == fingerprint(candidate)


@pytest.mark.parametrize('field,value', [
    ('model_revision','wrong-model'), ('model','other-backbone'), ('dtype','float32'),
    ('batch_size',2), ('profile_sha256','d'*64), ('static_profile_sha256','e'*64),
    ('eta',.5), ('temperature','none'), ('target','slope'), ('probe_band',3),
    ('seed',1), ('steps',27), ('guidance',3.5), ('implementation_sha256','f'*64),
    ('diffusers','unfrozen-version'), ('grid',[128,128]), ('unexpected_field',True)])
def test_corruption_rejected_even_when_self_fingerprint_recomputed(configs, field, value):
    candidate, row, protocol, accepted, reference = configs
    candidate[field] = value
    # This is the old reporter failure mode: candidate plus matching self-hash.
    forged_sidecar = {'config':candidate, 'config_fingerprint':fingerprint(candidate)}
    assert forged_sidecar['config_fingerprint'] == fingerprint(forged_sidecar['config'])
    with pytest.raises(ValueError, match='independent frozen expectation'):
        report.expected_native_config(candidate, row, protocol, accepted, reference)


@pytest.mark.parametrize('change', ['method','theta','embedded_gain','native_grid','schedule','default_name','duplicate_default'])
def test_rotary_and_scheduler_corruption_rejected(configs, change):
    candidate, row, protocol, accepted, reference = configs
    if change == 'method': candidate['rope']['method'] = 'ntk2'
    elif change == 'theta': candidate['rope']['theta'] = 12000.
    elif change == 'embedded_gain': candidate['rope']['embedded_qk_gain'][0] = 1.25
    elif change == 'native_grid': candidate['rope']['native_grid_tokens'] = [32,32]
    elif change == 'schedule': candidate['scheduler']['max_shift'] = 1.2
    elif change == 'default_name': candidate['scheduler']['_use_default_values'][0] = 'other'
    else: candidate['scheduler']['_use_default_values'].append('invert_sigmas')
    with pytest.raises(ValueError): report.expected_native_config(candidate, row, protocol, accepted, reference)


def test_reference_implementation_must_match_current_frozen_implementation(configs):
    candidate, row, protocol, accepted, reference = configs
    reference['implementation_sha256'] = candidate['implementation_sha256'] = 'e'*64
    with pytest.raises(ValueError, match='implementation fingerprint'):
        report.expected_native_config(candidate, row, protocol, accepted, reference)


def test_runtime_commit_may_add_notes_but_cannot_change_frozen_source(monkeypatch):
    report.verify_runtime_revision.cache_clear()
    content = b'fixed executable contents\n'
    digest = hashlib.sha256(content).hexdigest()
    calls = []
    def show(command, cwd):
        calls.append(command)
        return content
    monkeypatch.setattr(report.subprocess,'check_output',show)
    report.verify_runtime_revision('a'*40, (('scripts/worker.py',digest),))
    assert calls == [['git','show','a'*40+':scripts/worker.py']]
    with pytest.raises(ValueError, match='Runtime commit source mismatch'):
        report.verify_runtime_revision('b'*40, (('scripts/worker.py','0'*64),))
    with pytest.raises(ValueError, match='Invalid runtime code revision'):
        report.verify_runtime_revision('--unexpected-option', (('scripts/worker.py',digest),))
    report.verify_runtime_revision.cache_clear()
