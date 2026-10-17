"""Offline geometry, independent equations and corrupt-artifact checks."""
import json
import math

import pytest
import torch

from scripts.attention_operator_capture import tensor_receipt
from scripts.attention_operator_report import (
    analyze_tensors, build_report, exact_cells, load_payload, metrics,
    raster_cells, verify_replay, verify_sources,
)
from scripts.observer_pretrained_audit import sha256, tensor_hash
from scripts.observer_pretrained_report import difference_stats
from scripts.pisa_complement_cpu import compare


@pytest.fixture
def tensors():
    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(1729)
    q = torch.randn(2, 3, 4, generator=rng).bfloat16()
    k = torch.randn(2, 27, 4, generator=rng).bfloat16()
    v = torch.randn(2, 27, 4, generator=rng).bfloat16()
    saved = ((q.double()/2 @ k.double().transpose(1, 2)).softmax(-1) @ v.double()).bfloat16()
    return q, k, v, saved


KWARGS = dict(grid=[4, 6], cell=[2, 3], radius=0, text_tokens=3,
              coordinates=[[0, 0], [3, 5], [1, 4]], heads=[0, 12])


def test_non_square_raster_orientation_and_clipping():
    raster = torch.arange(24).reshape(1, 24, 1)
    cells = raster_cells(raster, [4, 6], [2, 3])
    assert cells[0, :, :, 0].tolist() == [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11],
                                          [12, 13, 14, 18, 19, 20], [15, 16, 17, 21, 22, 23]]
    assert exact_cells([3, 5], [4, 6], [2, 3], 0) == [3]
    assert exact_cells([0, 0], [4, 6], [2, 3], 1) == [0, 1, 2, 3]
    assert exact_cells([0, 0], [256, 256], [16, 16], 2) == [r*16+c for r in range(3) for c in range(3)]


def test_cached_operator_matches_reusable_equal_cell_reference(tensors):
    q, k, v, saved = tensors
    rows = analyze_tensors(*tensors, **KWARGS)
    kc, vc = [raster_cells(x[:, 3:].double(), [4, 6], [2, 3]) for x in (k, v)]
    for row in rows:
        hi, qi = KWARGS['heads'].index(row['head']), row['query_index']
        reference = compare(q[:, qi:qi+1].double(), kc, vc, row['selected_cells'],
                            exact_k=k[:, :3].double(), exact_v=v[:, :3].double())
        assert row['exact_key_count'] == 9  # Six image tokens, three text tokens once.
        for name, entry in row['controls'].items():
            expected = reference['sparse'] if name == 'sparse' else reference['outputs'][name]
            torch.testing.assert_close(torch.tensor(entry['output'], dtype=torch.float64), expected[hi, 0], atol=1e-12, rtol=1e-12)
        assert row['controls']['zero_order']['log_mass'] == pytest.approx(reference['approximate_lse'][hi, 0].item())
        assert row['controls']['zero_order']['exact_support_probability'] == pytest.approx(reference['approximate_exact_mass'][hi, 0].item())
        assert row['sdpa_bfloat16_vs_fp64']['absolute_l2'] > 0


def test_all_exact_and_identical_keys_limits(tensors):
    q, k, v, saved = tensors
    all_exact = analyze_tensors(*tensors, **{**KWARGS, 'radius': 2})
    for row in all_exact:
        assert row['exact_key_count'] == 27
        for result in row['controls'].values():
            assert result['absolute_l2'] < 1e-12
            assert result['log_mass_minus_dense'] == pytest.approx(0, abs=1e-12)
            assert result['exact_support_probability'] == pytest.approx(1)
    k = torch.ones_like(k)
    same_keys = analyze_tensors(q, k, v, saved, **KWARGS)
    for row in same_keys:
        assert row['true_exact_support_probability'] == pytest.approx(9/27)
        for name, result in row['controls'].items():
            if name != 'sparse':
                assert result['absolute_l2'] < 1e-12


def test_coordinate_range_is_only_a_necessary_hull_condition():
    # V points (0,0),(1,1): (0,1) passes ranges but is outside their line segment.
    bounds = metrics(torch.tensor([0., 1.]), torch.tensor([.5, .5]), torch.zeros(2), torch.ones(2))
    assert bounds['coordinate_range_violating_count'] == 0
    bad = metrics(torch.tensor([-1., 2.]), torch.tensor([.5, .5]), torch.zeros(2), torch.ones(2))
    assert bad['coordinate_range_violating_count'] == 2 and bad['coordinate_range_violation_max'] == 1


def test_large_common_key_shift_is_stable(tensors):
    q, k, v, saved = [x.double() for x in tensors]
    base = analyze_tensors(q, k, v, saved, **KWARGS)
    shifted = analyze_tensors(q, k+10000., v, saved, **KWARGS)
    for a, b in zip(base, shifted):
        for name in a['controls']:
            torch.testing.assert_close(torch.tensor(a['controls'][name]['output']),
                                       torch.tensor(b['controls'][name]['output']), atol=1e-6, rtol=1e-6)


def make_payload(tmp_path, tensors):
    q, k, v, saved = tensors
    images = torch.tensor([0, 23, 10])
    data = {'query': q[None], 'key': k[None], 'value': v[None], 'sdpa_output': saved[None],
            'head_indices': torch.tensor([0, 12]), 'query_image_indices': images,
            'query_joint_indices': images+3, 'key_joint_indices': torch.arange(27),
            'gain_bfloat16': torch.ones(4, dtype=torch.bfloat16)}
    path = tmp_path/'attention_00.pt'
    torch.save(data, path)
    receipt = {'file_sha256': sha256(path), 'tensors': {name: tensor_receipt(x) for name, x in data.items()}}
    kwargs = dict(grid=[4, 6], text_tokens=3, coordinates=KWARGS['coordinates'], heads=[0, 12], dim=4,
                  gain_hash=tensor_hash(data['gain_bfloat16']))
    return path, receipt, kwargs, data


def test_payload_verified_before_analysis(tmp_path, tensors):
    path, receipt, kwargs, data = make_payload(tmp_path, tensors)
    loaded = load_payload(path, receipt, **kwargs)
    assert torch.equal(loaded['key'], data['key'])


@pytest.mark.parametrize('corruption', ['file', 'receipt', 'order', 'dtype', 'gain'])
def test_corrupted_artifact_rejected(tmp_path, tensors, corruption):
    path, receipt, kwargs, data = make_payload(tmp_path, tensors)
    if corruption == 'file':
        receipt['file_sha256'] = '0'*64
    elif corruption == 'receipt':
        receipt['tensors']['query']['sha256'] = '0'*64
    elif corruption == 'gain':
        kwargs['gain_hash'] = '0'*64
    else:
        if corruption == 'order':
            data['key_joint_indices'] = data['key_joint_indices'].flip(0)
        else:
            data['query'] = data['query'].float()
        torch.save(data, path)
        receipt = {'file_sha256': sha256(path), 'tensors': {name: tensor_receipt(x) for name, x in data.items()}}
    with pytest.raises(ValueError):
        load_payload(path, receipt, **kwargs)


def test_replay_recomputed_not_trusted(tmp_path):
    baseline = torch.ones(1, 2, 3, dtype=torch.bfloat16)
    captured = baseline.clone()
    outputs = {}
    for name, value in [('baseline', baseline), ('capture', captured)]:
        torch.save({'output': value}, tmp_path/f'{name}_output.pt')
        outputs[name] = {'file_sha256': sha256(tmp_path/f'{name}_output.pt'), **tensor_receipt(value)}
    completion = {'completed_forwards': 2, 'replay': {'within_job_byte_equal': True,
                  'within_job': difference_stats(captured, baseline)}, 'outputs': outputs}
    assert verify_replay(tmp_path, completion, (1, 2, 3))['byte_equal']
    captured[0, 0, 0] = 2
    torch.save({'output': captured}, tmp_path/'capture_output.pt')
    completion['outputs']['capture'] = {'file_sha256': sha256(tmp_path/'capture_output.pt'), **tensor_receipt(captured)}
    with pytest.raises(ValueError, match='not byte equal'):
        verify_replay(tmp_path, completion, (1, 2, 3))


def test_source_and_protocol_hashes_rejected(tmp_path):
    source = tmp_path/'source.py'
    source.write_text('original')
    sources = {'source.py': sha256(source)}
    verify_sources(tmp_path, sources)
    source.write_text('changed')
    with pytest.raises(ValueError, match='Frozen source changed'):
        verify_sources(tmp_path, sources)
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps({}))
    with pytest.raises(ValueError, match='Protocol pin mismatch'):
        build_report(protocol, '0'*64, tmp_path)
