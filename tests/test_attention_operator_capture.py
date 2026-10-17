"""CPU tests of actual SDPA observation, fixed selections and failure cleanup."""
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import crope.gain as gain_module
import scripts.attention_operator_capture as capture


def fixture_data():
    generator = torch.Generator().manual_seed(105)
    tensors = [torch.randn(1, 9, 3, 4, generator=generator).bfloat16() for _ in range(3)]
    gain = torch.tensor([1.001, 1.1, 1.25, 1.4], dtype=torch.float32)
    return *tensors, gain


def make_capture(gain, sink, **updates):
    args = dict(expected_gain=gain, grid=(2, 3), text_tokens=3, total_heads=3, head_dim=4,
                selected_heads=[0, 2], coordinates=[[0, 1], [1, 2]], selected_calls=[0],
                expected_calls=1, sink=sink)
    args.update(updates)
    return capture.AttentionCapture(**args)


def test_observes_actual_gained_sdpa_arguments_and_preserves_output():
    q, k, v, gain = fixture_data()
    baseline = gain_module.gained_attention(q, k, v, gain)
    saved = {}
    recorder = make_capture(gain, lambda i, payload, receipts: saved.update(payload=payload, receipts=receipts))
    old_attention, old_sdpa = gain_module.gained_attention, F.scaled_dot_product_attention
    with recorder.installed(gain_module):
        output = gain_module.gained_attention(q, k, v, gain)
    recorder.validate_complete()
    assert gain_module.gained_attention is old_attention and F.scaled_dot_product_attention is old_sdpa
    capture.validate_replay(baseline, output)
    payload = saved['payload']
    consumed_q, consumed_k = (q * gain.bfloat16()).transpose(1, 2), (k * gain.bfloat16()).transpose(1, 2)
    heads, queries = torch.tensor([0, 2]), torch.tensor([4, 8])
    assert torch.equal(payload['query'], consumed_q.index_select(1, heads).index_select(2, queries))
    assert torch.equal(payload['key'], consumed_k.index_select(1, heads))
    assert torch.equal(payload['value'], v.transpose(1, 2).index_select(1, heads))
    assert torch.equal(payload['sdpa_output'], baseline.transpose(1, 2).index_select(1, heads).index_select(2, queries))
    assert payload['query_image_indices'].tolist() == [1, 5]
    assert payload['query_joint_indices'].tolist() == [4, 8]
    assert payload['key_joint_indices'].tolist() == list(range(9))
    assert payload['head_indices'].tolist() == [0, 2]
    assert payload['key'].dtype == torch.bfloat16 and payload['key'].shape == (1, 2, 9, 4)
    for name, tensor in payload.items():
        assert saved['receipts'][name] == capture.tensor_receipt(tensor)


def test_selected_calls_are_zero_based_and_other_calls_are_not_saved():
    q, k, v, gain = fixture_data()
    selected = []
    recorder = make_capture(gain, lambda index, *_: selected.append(index), selected_calls=[0, 2], expected_calls=3)
    with recorder.installed(gain_module):
        for _ in range(3):
            gain_module.gained_attention(q, k, v, gain)
    recorder.validate_complete()
    assert selected == [0, 2] and recorder.calls == recorder.sdpa_calls == 3


@pytest.mark.parametrize('case', ['dtype', 'shape', 'gain', 'mask', 'nonfinite'])
def test_rejects_unexpected_inputs_and_always_restores(case):
    q, k, v, gain = fixture_data()
    recorder = make_capture(gain, lambda *_: None)
    mask = None
    if case == 'dtype':
        q = q.float()
    elif case == 'shape':
        k = k[:, :-1]
    elif case == 'gain':
        gain = gain + 1
    elif case == 'mask':
        mask = torch.zeros(9, 9)
    else:
        v[0, 0, 0, 0] = float('nan')
    old_attention, old_sdpa = gain_module.gained_attention, F.scaled_dot_product_attention
    with pytest.raises(ValueError):
        with recorder.installed(gain_module):
            gain_module.gained_attention(q, k, v, gain, mask)
    assert gain_module.gained_attention is old_attention and F.scaled_dot_product_attention is old_sdpa


@pytest.mark.parametrize('case', ['scale', 'causal', 'dropout', 'mask', 'missing_call', 'double_call', 'bad_output'])
def test_sdpa_contract_and_call_counts_are_guarded(case, monkeypatch):
    q, k, v, gain = fixture_data()
    original_sdpa = F.scaled_dot_product_attention

    def unusual_attention(q, k, v, m, attention_mask=None):
        if case == 'missing_call':
            return q
        kwargs = dict(attn_mask=None, dropout_p=0., is_causal=False)
        if case == 'scale':
            kwargs['scale'] = .5
        elif case == 'causal':
            kwargs['is_causal'] = True
        elif case == 'dropout':
            kwargs['dropout_p'] = .1
        elif case == 'mask':
            kwargs['attn_mask'] = torch.zeros(9, 9)
        result = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **kwargs)
        if case == 'double_call':
            F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **kwargs)
        return result.transpose(1, 2)

    if case == 'bad_output':
        monkeypatch.setattr(F, 'scaled_dot_product_attention', lambda *a, **k: torch.zeros(1))
    monkeypatch.setattr(gain_module, 'gained_attention', unusual_attention)
    expected_sdpa = F.scaled_dot_product_attention
    recorder = make_capture(gain, lambda *_: None)
    with pytest.raises(ValueError):
        with recorder.installed(gain_module):
            gain_module.gained_attention(q, k, v, gain)
    assert gain_module.gained_attention is unusual_attention and F.scaled_dot_product_attention is expected_sdpa
    if case != 'bad_output':
        assert expected_sdpa is original_sdpa


def test_sink_failure_restores_both_wrappers():
    q, k, v, gain = fixture_data()
    def failing_sink(*_):
        raise RuntimeError('storage failure')
    recorder = make_capture(gain, failing_sink)
    originals = gain_module.gained_attention, F.scaled_dot_product_attention
    with pytest.raises(RuntimeError, match='storage failure'):
        with recorder.installed(gain_module):
            gain_module.gained_attention(q, k, v, gain)
    assert (gain_module.gained_attention, F.scaled_dot_product_attention) == originals


def test_rejects_missing_and_extra_outer_calls():
    q, k, v, gain = fixture_data()
    recorder = make_capture(gain, lambda *_: None)
    with pytest.raises(ValueError, match='count mismatch'):
        recorder.validate_complete()
    with recorder.installed(gain_module):
        gain_module.gained_attention(q, k, v, gain)
        with pytest.raises(ValueError, match='Too many'):
            gain_module.gained_attention(q, k, v, gain)


def test_replay_requires_byte_equality_including_signed_zero():
    original = torch.tensor([0.], dtype=torch.bfloat16)
    repeated = torch.tensor([-0.], dtype=torch.bfloat16)
    assert torch.equal(original, repeated)
    with pytest.raises(ValueError, match='byte equal'):
        capture.validate_replay(original, repeated)


@pytest.mark.parametrize('coordinates', [[[0, 1], [0, 1]], [[2, 0]], [[1.5, 0]], []])
def test_rejects_invalid_query_selection(coordinates):
    with pytest.raises(ValueError):
        capture.query_indices((2, 3), coordinates, 3)


def protocol_fixture():
    pin = lambda name, digest: {'path': f'/unused/{name}', 'sha256': digest}
    return dict(schema_version=1, status='frozen', development_prompt_indices=[0, 1],
        forwards_per_prompt=2, forward_order=['baseline', 'capture'], resolution=4096,
        grid=[256, 256], sigma=.5, guidance=4.5, text_tokens=512, attention_heads=24,
        head_dim=128, expected_attention_calls=57, dtype='bfloat16', capture=copy.deepcopy(capture.CAPTURE),
        accepted_cr18_report=pin('report.json', capture.CR18_REPORT_SHA256),
        accepted_cr18_receipt=pin('receipt.json', 'a'*64),
        cr18_protocol=pin('protocol.json', capture.CR18_PROTOCOL_SHA256),
        witness_retention=pin('retention.json', 'b'*64),
        source_sha256={name: 'c'*64 for name in capture.REQUIRED_SOURCES})


def test_protocol_is_explicit_and_does_not_need_files_during_schema_validation():
    capture.validate_protocol(protocol_fixture())


@pytest.mark.parametrize('key,value', [('status', 'draft'), ('forwards_per_prompt', 3),
    ('sigma', .7), ('expected_attention_calls', 56), ('capture', {}), ('source_sha256', {})])
def test_protocol_rejects_scope_drift(key, value):
    protocol = protocol_fixture()
    protocol[key] = value
    with pytest.raises(ValueError):
        capture.validate_protocol(protocol)


def test_retention_rejects_missing_duplicate_or_corrupt_witness(tmp_path):
    path = tmp_path/'input.pt'
    path.write_bytes(b'paper witness')
    record = {'source': '/old/input.pt', 'preserved': str(path), 'sha256': capture.sha256(path)}
    receipt = {'archive_root': str(tmp_path), 'files': [record]}
    assert capture.retained_path(receipt, record['source'], record['sha256']) == path
    with pytest.raises(ValueError):
        capture.retained_path(receipt, '/other/input.pt', record['sha256'])
    with pytest.raises(ValueError):
        capture.retained_path({**receipt, 'files': [record, record]}, record['source'], record['sha256'])
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):
        capture.retained_path(receipt, record['source'], record['sha256'])


def test_actual_tiny_flux_joint_and_single_processors_preserve_bf16_forward():
    from diffusers import FluxTransformer2DModel
    torch.manual_seed(720)
    model = FluxTransformer2DModel(patch_size=1, in_channels=8, out_channels=8,
        num_layers=1, num_single_layers=1, attention_head_dim=20, num_attention_heads=2,
        joint_attention_dim=12, pooled_projection_dim=6, guidance_embeds=True,
        axes_dims_rope=(4, 8, 8)).bfloat16().eval()
    gain = torch.linspace(1., 1.4, 20)
    gain_module.install_gain(model, SimpleNamespace(qk_scale=gain))
    image_ids = torch.tensor([[0, y, x] for y in range(2) for x in range(3)], dtype=torch.bfloat16)
    kwargs = dict(hidden_states=torch.randn(1, 6, 8).bfloat16(),
        encoder_hidden_states=torch.randn(1, 3, 12).bfloat16(),
        pooled_projections=torch.randn(1, 6).bfloat16(), timestep=torch.tensor([.5]).bfloat16(),
        guidance=torch.tensor([4.5]), img_ids=image_ids, txt_ids=torch.zeros(3, 3).bfloat16(), return_dict=False)
    saved = {}
    recorder = make_capture(gain, lambda i, p, _: saved.update({i: p}), total_heads=2, head_dim=20,
        selected_heads=[0, 1], selected_calls=[0, 1], expected_calls=2)
    with torch.inference_mode():
        baseline = model(**kwargs)[0]
        with recorder.installed(gain_module):
            observed = model(**kwargs)[0]
    recorder.validate_complete()
    capture.validate_replay(baseline, observed)
    assert set(saved) == {0, 1}
    assert all(payload['key'].shape == (1, 2, 9, 20) for payload in saved.values())
