"""CPU algebra and routing tests; these do not establish CUDA kernel support."""
import math
from types import SimpleNamespace

import pytest
import torch

from crope.crossmodal_phase import (AXES, CrossmodalPhaseProcessor, _dense_attention,
    cudnn_attention, fixed_mass_output, install_crossmodal_phase,
    native_rotary_embedding, phase_attention, unnormalized_output)
from crope.rope import FluxRoPE


@pytest.fixture
def tensors():
    rng = torch.Generator().manual_seed(142)
    q, k, v = [torch.randn(1, 2, 9, 8, dtype=torch.float64, generator=rng) for _ in range(3)]
    alt = torch.randn(1, 2, 6, 8, dtype=torch.float64, generator=rng)
    return q, k, v, alt


def direct_operator(q, k, v, alt, mode, text=3):
    logits = torch.einsum('bhid,bhjd->bhij', q, k)/math.sqrt(q.shape[-1])
    original_weights = logits.softmax(-1)
    replacement = torch.einsum('bhid,bhjd->bhij', alt, k[:, :, :text])/math.sqrt(q.shape[-1])
    if mode == 'fixedmass':
        replacement += logits[:, :, text:, :text].logsumexp(-1, keepdim=True)-replacement.logsumexp(-1, keepdim=True)
    logits[:, :, text:, :text] = replacement
    weights = logits.softmax(-1)
    return weights @ v, weights, original_weights


@pytest.mark.parametrize('mode', ['fixedmass', 'unnormalized'])
@pytest.mark.parametrize('chunk', [1, 4, 65536])
def test_exact_fp64_full_attention_oracle_and_chunking(tensors, mode, chunk):
    q, k, v, alt = tensors
    actual, counts, ranges = phase_attention(q, k, v, alt, text_tokens=3, mode=mode,
                                            query_chunk_size=chunk, backend=_dense_attention)
    expected, weights, old_weights = direct_operator(*tensors, mode)
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(weights[:, :, :3], old_weights[:, :, :3], rtol=0, atol=0)
    if mode == 'fixedmass':
        torch.testing.assert_close(weights[:, :, 3:, 3:], old_weights[:, :, 3:, 3:], rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(weights[:, :, 3:, :3].sum(-1), old_weights[:, :, 3:, :3].sum(-1), rtol=2e-14, atol=2e-14)
    assert counts['full_joint_calls'] == 1
    assert counts['new_text_calls'] == math.ceil(6/chunk)
    assert counts['image_only_calls'] == (math.ceil(6/chunk) if mode == 'unnormalized' else 0)
    assert counts['old_text_calls'] == (math.ceil(6/chunk) if mode == 'fixedmass' else 0)
    assert all(bool((value >= 0).all() & (value <= 1).all()) for value in ranges.values())


@pytest.mark.parametrize('mode', ['baseline', 'fixedmass', 'unnormalized'])
def test_declared_native_identity_runs_only_identical_baseline(tensors, mode):
    q, k, v, _ = tensors
    baseline, _ = _dense_attention(q, k, v)
    actual, counts, ranges = phase_attention(q, k, v, None, text_tokens=3, mode=mode,
                                            backend=_dense_attention, phase_identity=True)
    assert torch.equal(actual.view(torch.uint8), baseline.view(torch.uint8))
    assert sum(counts.values()) == 1 and not ranges


@pytest.mark.parametrize('mode', ['fixedmass', 'unnormalized'])
def test_non_bypassed_identical_phase_matches_baseline_algebra(tensors, mode):
    q, k, v, _ = tensors
    baseline, _ = _dense_attention(q, k, v)
    actual, _, _ = phase_attention(q, k, v, q[:, :, 3:], text_tokens=3, mode=mode, backend=_dense_attention)
    torch.testing.assert_close(actual, baseline, rtol=3e-14, atol=3e-14)


@pytest.mark.parametrize('logit_delta', [-2000., -80., 0., 80., 2000.])
def test_stable_unnormalized_extreme_text_image_ratios(logit_delta):
    image = torch.tensor([[[[2., -3.]]]], dtype=torch.float64)
    text = torch.tensor([[[[-7., 11.]]]], dtype=torch.float64)
    li = torch.zeros(1, 1, 1, dtype=torch.float64)
    lt = li+logit_delta
    result, mass = unnormalized_output(image, text, li, lt)
    weights = torch.tensor([0., logit_delta], dtype=torch.float64).softmax(0)
    torch.testing.assert_close(result, weights[0]*image+weights[1]*text, rtol=1e-14, atol=1e-14)
    assert torch.isfinite(result).all() and 0 <= mass.item() <= 1


def test_invalid_mass_nonfinite_and_wrong_geometry_are_rejected(tensors):
    q, k, v, alt = tensors
    with pytest.raises(ValueError, match='old text mass'):
        fixed_mass_output(q, q, q, torch.ones(q.shape[:-1]), torch.zeros(q.shape[:-1]))
    with pytest.raises(ValueError, match='Nonfinite conditional'):
        unnormalized_output(q, q, torch.full(q.shape[:-1], float('nan')), torch.zeros(q.shape[:-1]))
    with pytest.raises(ValueError, match='shape mismatch'):
        phase_attention(q.flatten(), k, v, alt, text_tokens=3, mode='fixedmass')
    with pytest.raises(ValueError, match='Alternate image'):
        phase_attention(q, k, v, alt[:, :, :-1], text_tokens=3, mode='fixedmass', backend=_dense_attention)
    with pytest.raises(ValueError, match='Positive query'):
        phase_attention(q, k, v, alt, text_tokens=3, mode='fixedmass', query_chunk_size=0)
    for text in (0, 9, True):
        with pytest.raises(ValueError, match='partitions'):
            phase_attention(q, k, v, alt, text_tokens=text, mode='fixedmass')


def raster(grid):
    ids = torch.zeros(*grid, 3)
    ids[..., 1] = torch.arange(grid[0])[:, None]
    ids[..., 2] = torch.arange(grid[1])[None, :]
    return ids.reshape(-1, 3)


def test_native_embedding_identity_and_endpoint_mapping():
    actual = native_rotary_embedding((64, 64), 'cpu')
    deployed = FluxRoPE(axes_dims=AXES, theta=10000, method='sega_official', grid=(64, 64))(raster((64, 64)))
    assert all(torch.equal(a, b) for a, b in zip(actual, deployed))
    transported = native_rotary_embedding((4, 8), 'cpu')
    endpoints = raster((4, 8))*torch.tensor([1., 21., 9.])
    expected = FluxRoPE(axes_dims=AXES, theta=10000, method='base', grid=(64, 64))(endpoints)
    assert all(torch.equal(a, b) for a, b in zip(transported, expected))
    assert torch.equal(transported[0][:, :16], torch.ones(32, 16))
    assert torch.equal(transported[1][:, :16], torch.zeros(32, 16))


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = FluxRoPE(axes_dims=AXES, theta=10000, method='sega_official', grid=(4, 8))
        self.attn_processors = {'transformer_blocks.0.attn.processor': object(),
                                'single_transformer_blocks.0.attn.processor': object()}

    def set_attn_processor(self, processors):
        self.attn_processors = dict(processors)

    def forward(self, *, txt_ids, img_ids):
        return img_ids


def test_all_text_policy_ids_lifecycle_and_receipts():
    transformer = FakeTransformer()
    original = dict(transformer.attn_processors)
    with pytest.raises(ValueError, match='all512'):
        install_crossmodal_phase(transformer, mode='fixedmass', grid=(4, 8), controller=None, text_tokens=511)
    handle = install_crossmodal_phase(transformer, mode='fixedmass', grid=(4, 8), controller=None)
    transformer(txt_ids=torch.zeros(512, 3), img_ids=raster((4, 8)))
    with pytest.raises(ValueError, match='padding'):
        transformer(txt_ids=torch.ones(512, 3), img_ids=raster((4, 8)))
    with pytest.raises(ValueError, match='raster'):
        transformer(txt_ids=torch.zeros(512, 3), img_ids=raster((4, 8)).flip(0))
    handle.record(next(iter(original)), dict(full_joint_calls=1, new_text_calls=1, old_text_calls=1, image_only_calls=0),
                  {'old_text_mass': torch.tensor([.1, .8])})
    receipt = handle.receipt()
    assert receipt['calls'] == 1 and receipt['geometry_id_validations'] == 1
    assert receipt['query_chunk_size'] == 65536 and sum(receipt['per_block_calls'].values()) == 1
    handle.reset()
    assert handle.receipt()['calls'] == 0 and not handle.receipt()['mass_ranges']
    handle.close()
    handle.close()
    assert transformer.attn_processors == original and not transformer._forward_pre_hooks


def test_no_cpu_fallback_and_mask_rejection():
    q = torch.ones(1, 2, 8, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='CUDA BF16'):
        cudnn_attention(q, q, q)
    with pytest.raises(ValueError, match='Masked'):
        CrossmodalPhaseProcessor(None, 'block')(None, None, attention_mask=torch.zeros(1), image_rotary_emb=(None, None))


def test_private_operator_abi_and_lse_layout_with_mocked_cuda_device(monkeypatch):
    # This deliberately tests the adapter ABI only. The tensor remains on CPU;
    # the real CUDA operation is replaced, so this is not CUDA validation.
    monkeypatch.setattr(torch.Tensor, 'device', property(lambda self: torch.device('cuda:0')))
    query = torch.ones(1, 3, 2, 128, dtype=torch.bfloat16).transpose(1, 2)
    key = torch.ones(1, 5, 2, 128, dtype=torch.bfloat16).transpose(1, 2)
    observed = []
    def fake_op(*args, **kwargs):
        observed.append((args, kwargs))
        return query.clone(), torch.zeros(1, 2, 3, 1, dtype=torch.float32)
    monkeypatch.setattr(torch.ops.aten._scaled_dot_product_cudnn_attention, 'default', fake_op)
    result, lse = cudnn_attention(query, key, key)
    assert result.shape == query.shape and lse.shape == (1, 2, 3)
    args, kwargs = observed[0]
    assert args[3:] == (None, True, 0., False, False)
    assert kwargs == {'scale': 1/math.sqrt(128)}
    def failing(*args, **kwargs):
        raise RuntimeError('unsupported simulated plan')
    monkeypatch.setattr(torch.ops.aten._scaled_dot_product_cudnn_attention, 'default', failing)
    with pytest.raises(RuntimeError, match='no math or alternate-backend fallback'):
        cudnn_attention(query, key, key)


def test_identical_text_values_and_negligible_mass_are_zero_effect():
    joint = torch.tensor([[[[2., -4.]]]], dtype=torch.float64)
    old_pool = torch.tensor([[[[10., 20.]]]], dtype=torch.float64)
    lse = torch.zeros(1, 1, 1, dtype=torch.float64)
    unchanged, _ = fixed_mass_output(joint, old_pool, old_pool, lse-1., lse)
    assert torch.equal(unchanged, joint)
    negligible, mass = fixed_mass_output(joint, old_pool, -old_pool, lse-1000., lse)
    assert mass.item() == 0 and torch.equal(negligible, joint)


@pytest.mark.parametrize('joint', [False, True])
def test_joint_and_single_processor_preserve_consumed_gains_and_routing(monkeypatch, joint):
    import diffusers.models.transformers.transformer_flux as flux
    from diffusers.models.embeddings import apply_rotary_emb
    generator = torch.Generator().manual_seed(991)
    text, image, heads = 3, 4, 2
    all_qkv = [torch.randn(1, text+image, heads*128, generator=generator) for _ in range(3)]
    gains = torch.linspace(.8, 1.3, 64).repeat_interleave(2)
    state = SimpleNamespace(mode='fixedmass', native_identity=False, text_tokens=text, grid=(2, 2),
        controller=SimpleNamespace(qk_scale=gains), query_chunk_size=65536,
        target_embedding=lambda device: native_rotary_embedding((2, 2), device), record=lambda *args: None)
    seen = []
    def backend(q, k, v):
        seen.append((q.clone(), k.clone(), v.clone()))
        out, lse = _dense_attention(q, k, v)
        return out.float(), lse.float()
    state.backend = backend
    if joint:
        projections = tuple(x[:, text:] for x in all_qkv)+tuple(x[:, :text] for x in all_qkv)
    else:
        projections = tuple(all_qkv)+(None, None, None)
    monkeypatch.setattr(flux, '_get_qkv_projections', lambda *args: projections)
    identity = torch.nn.Identity()
    attn = SimpleNamespace(heads=heads, norm_q=identity, norm_k=identity,
        norm_added_q=identity, norm_added_k=identity, added_kv_proj_dim=256 if joint else None,
        to_out=[identity, identity], to_add_out=identity)
    ids = torch.cat((torch.zeros(text, 3), raster((2, 2))))
    rope = FluxRoPE(axes_dims=AXES, theta=10000, method='sega_official', grid=(2, 2))(ids)
    hidden = all_qkv[0][:, text:] if joint else all_qkv[0]
    encoder = all_qkv[0][:, :text] if joint else None
    output = CrossmodalPhaseProcessor(state, 'block')(attn, hidden, encoder, image_rotary_emb=rope)
    unflattened = all_qkv[0].unflatten(-1, (heads, 128))
    expected_old = (apply_rotary_emb(unflattened, rope, sequence_dim=1)*gains).transpose(1, 2)
    expected_new = (apply_rotary_emb(unflattened[:, text:], state.target_embedding('cpu'), sequence_dim=1)*gains).transpose(1, 2)
    assert torch.equal(seen[0][0], expected_old)
    assert torch.equal(seen[1][0], expected_new)
    assert torch.equal(seen[1][1], seen[0][1][:, :, :text])
    if joint:
        assert output[0].shape == (1, image, 256) and output[1].shape == (1, text, 256)
    else:
        assert output.shape == (1, text+image, 256)
