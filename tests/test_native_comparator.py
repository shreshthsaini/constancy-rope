"""CPU-only scope, saved-output corruption and native algebra checks."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from crope.runtime import fingerprint
from crope.utils import file_sha256, image_seed, image_stem
from scripts.native_comparator import (DEV_IDS, FIXED, PINNED, REQUIRED_SOURCES,
    SCHEDULER_EXPECTATION, InitialInputAudit, NativeGainAudit, validate_output, validate_protocol)


@pytest.fixture
def protocol():
    pin = {'path': '/scratch/test/reference.json', 'sha256': 'a'*64}
    return {**copy.deepcopy(FIXED), **{k: {**pin, 'sha256': v} for k, v in PINNED.items()},
        'cr14_model_manifest': dict(pin), 'rubric': dict(pin),
        'selected_prompts': [{'index': i, 'id': pid, 'prompt': 'Caption '+str(i),
            'derived_seed': image_seed(pid, 0), 'reference_png': dict(pin), 'reference_metadata': dict(pin)}
            for i, pid in enumerate(DEV_IDS)],
        'source_sha256': {name: 'b'*64 for name in REQUIRED_SOURCES},
        'scheduler_expectation': dict(SCHEDULER_EXPECTATION),
        'expected_sigmas': [1.-i/28 for i in range(28)]}


def test_exact_eight_ids_and_two_chunks(protocol):
    validate_protocol(protocol, 0, 4)
    validate_protocol(protocol, 4, 8)
    assert len(set(DEV_IDS)) == 8
    assert DEV_IDS[0] == 'a4k_eval_4096_minus6873203881450547484'
    assert DEV_IDS[-1] == 'a4k_eval_4096_minus8648158440845565467'
    assert protocol['selected_prompts'][1]['derived_seed'] == 4141338643831228041


@pytest.mark.parametrize('bounds', [(-1,4), (0,3), (1,4), (0,8), (4,9), (True,4)])
def test_partial_or_new_prompt_chunks_rejected(protocol, bounds):
    with pytest.raises(ValueError, match='chunks'):
        validate_protocol(protocol, *bounds)


@pytest.mark.parametrize('corruption', ['extra_id', 'swap', 'seed', 'prompt_pin', 'rubric', 'schedule', 'scope', 'source'])
def test_protocol_rejects_identity_and_scope_drift(protocol, corruption):
    if corruption == 'extra_id': protocol['selected_prompts'][7]['id'] = 'heldout8'
    elif corruption == 'swap': protocol['selected_prompts'][0], protocol['selected_prompts'][1] = protocol['selected_prompts'][1], protocol['selected_prompts'][0]
    elif corruption == 'seed': protocol['selected_prompts'][0]['derived_seed'] += 1
    elif corruption == 'prompt_pin': protocol['prompt_file']['sha256'] = 'd'*64
    elif corruption == 'rubric': del protocol['rubric']
    elif corruption == 'schedule': protocol['expected_sigmas'][3] = float('nan')
    elif corruption == 'scope': protocol['noise_coupling'] = 'shared_noise'
    else: del protocol['source_sha256']['scripts/native_comparator.py']
    with pytest.raises(ValueError): validate_protocol(protocol, 0, 4)


@pytest.fixture
def output(tmp_path, protocol):
    row = protocol['selected_prompts'][0]
    png = tmp_path/(image_stem(row['id'], 0)+'.png')
    metadata = png.with_suffix('.json')
    Image.new('RGB', (1024, 1024), (17, 29, 43)).save(png)
    config = dict(method='sega_official', res='1024x1024', prompt_id=row['id'], prompt=row['prompt'],
        seed=0, derived_seed=row['derived_seed'], steps=28, guidance=4.5, dtype='bfloat16',
        grid=[64,64], side_scale=[1.,1.], area_scale=1., scheduler=dict(SCHEDULER_EXPECTATION))
    history = [dict(step=i, sigma=sigma, model_timestep=sigma, mode='sega_official',
        log_g=[0.]*128, controller_log_g=[0.]*128,
        log_g_axes={'index':[0.]*16, 'height':[0.]*56, 'width':[0.]*56},
        official=dict(reference_qk_gain=1., ntk_factor=1., spatial_scale=1., spectral_spread=0., sigma=sigma))
        for i, sigma in enumerate(protocol['expected_sigmas'])]
    record = dict(complete=True, config=config, config_fingerprint=fingerprint(config), image=str(png),
        png_sha256=file_sha256(png), sigmas=protocol['expected_sigmas'], gains=history, seconds=.5)
    metadata.write_text(json.dumps(record))
    return png, metadata, row, config, record


def test_valid_output_reopened_and_enumerated(output, protocol):
    png, metadata, row, config, _ = output
    receipt = validate_output(png, metadata, row, protocol, config)
    assert receipt['shape'] == [1024,1024,3] and receipt['all_native_gains_one']
    assert receipt['metadata_sha256'] == file_sha256(metadata)


@pytest.mark.parametrize('corruption', ['shape', 'pixels', 'config', 'history', 'sigma', 'missing_step', 'official', 'fingerprint'])
def test_corrupt_output_rejected_even_with_refreshed_hash_when_relevant(output, protocol, corruption):
    png, metadata, row, config, record = output
    if corruption == 'shape':
        Image.new('RGB', (512,1024)).save(png)
        record['png_sha256'] = file_sha256(png)
    elif corruption == 'pixels': Image.new('RGB', (1024,1024), 'red').save(png)
    elif corruption == 'config':
        record['config'] = {**config, 'prompt': 'Different caption'}
        record['config_fingerprint'] = fingerprint(record['config'])
    elif corruption == 'history': record['gains'][12]['controller_log_g'][27] = .01
    elif corruption == 'sigma': record['sigmas'] = [1.]*28
    elif corruption == 'missing_step': record['gains'].pop()
    elif corruption == 'official': record['gains'][0]['official']['ntk_factor'] = 4.
    else: record['config_fingerprint'] = '0'*64
    metadata.write_text(json.dumps(record))
    with pytest.raises(ValueError): validate_output(png, metadata, row, protocol, config)


def test_native_released_sega_is_unit_gain_and_base_frequency_without_model():
    from crope.rope import FluxRoPE
    from crope.sega_official import OfficialSEGA
    torch.set_num_threads(2)
    sega, base = FluxRoPE(method='sega_official', grid=(64,64)), FluxRoPE(method='base', grid=(64,64))
    for axis in range(3):
        assert torch.equal(sega.frequencies(axis), base.frequencies(axis))
    official = OfficialSEGA((64,64))
    # Deliberately different projected features cannot change this native branch.
    for features in (torch.zeros(1,4096,2), torch.ones(1,4096,2)*137.):
        gain = official.update(features, .5)
        assert torch.equal(gain, torch.ones(128))
    assert torch.equal(sega.embedding_gain(), torch.ones(128))


def test_runtime_gain_audit_checks_values_and_step_order():
    rope = SimpleNamespace(embedding_gain=lambda device: torch.ones(128, device=device))
    experiment = SimpleNamespace(controller=SimpleNamespace(qk_scale=torch.ones(128)), rope=rope, step=0, sigma=1.)
    audit = NativeGainAudit(experiment)
    audit(None,None,None)
    with pytest.raises(ValueError, match='order'): audit(None,None,None)
    experiment.step = 1
    experiment.controller.qk_scale[20] = 1.01
    with pytest.raises(ValueError, match='Non-unit'): audit(None,None,None)


def test_initial_input_is_saved_once_and_labeled_not_raw_noise(tmp_path):
    audit = InitialInputAudit()
    audit.path = tmp_path/'initial.pt'
    value = torch.arange(4096*64).reshape(1,4096,64).bfloat16()
    audit(None,(),{'hidden_states':value})
    first = copy.deepcopy(audit.receipt)
    audit(None,(),{'hidden_states':value+1})
    assert audit.receipt == first
    loaded = torch.load(audit.path, weights_only=True)['initial_transformer_input']
    assert torch.equal(loaded,value)
    assert audit.receipt['label'] == 'initial_transformer_input'
    audit.receipt = None
    with pytest.raises(ValueError, match='Initial transformer'): audit(None,(),{'hidden_states':value.float()})
