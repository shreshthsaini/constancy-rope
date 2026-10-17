"""Isolated crossmodal phase pilot using explicit cuDNN output and LSE.

Only image-query/text-key logits change at each call. The unnormalized control
deliberately recomputes image-only attention to avoid subtractive recovery of a
small image contribution from rounded joint outputs. No backend fallback exists.
"""
from __future__ import annotations

import math
import torch

MODES = ('baseline', 'fixedmass', 'unnormalized')
AXES = (16, 56, 56)
NUMERICAL_POLICY = {'kernel_output_relative_l2': .01, 'kernel_output_max_abs': .02,
                    'kernel_lse_max_abs': 2e-4, 'operator_relative_l2': .02,
                    'operator_max_abs': .03, 'log_mass_upper_tolerance': 2e-4}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def native_rotary_embedding(grid, device):
    """Base-theta endpoint transport, evaluated before BF16 query multiplication."""
    from .rope import FluxRoPE
    require(len(grid) == 2 and all(type(x) is int and x >= 2 for x in grid), 'Invalid image grid')
    h, w = grid
    ids = torch.zeros(h, w, 3, device=device, dtype=torch.float32)
    ids[..., 1] = torch.arange(h, device=device, dtype=torch.float32)[:, None]*(63/(h-1))
    ids[..., 2] = torch.arange(w, device=device, dtype=torch.float32)[None, :]*(63/(w-1))
    return FluxRoPE(axes_dims=AXES, theta=10000, method='base', grid=(64, 64))(ids.reshape(-1, 3))


def cudnn_attention(query, key, value):
    """B,H,S,D BF16 CUDA attention; unsupported cuDNN geometry fails explicitly."""
    require(query.ndim == key.ndim == value.ndim == 4 and key.shape == value.shape and
            query.shape[:2] == key.shape[:2] and query.shape[-1] == key.shape[-1] == 128 and
            min(query.shape[-2], key.shape[-2]) > 0, 'Unsupported attention geometry')
    require(all(x.device.type == 'cuda' and x.device == query.device and x.dtype == torch.bfloat16
                and x.stride(-1) == 1 for x in (query, key, value)), 'cuDNN requires CUDA BF16 inputs with unit last stride')
    try:
        result = torch.ops.aten._scaled_dot_product_cudnn_attention.default(
            query, key, value, None, True, 0., False, False, scale=1/math.sqrt(128))
    except RuntimeError as error:
        raise RuntimeError('Explicit cuDNN attention failed; no math or alternate-backend fallback') from error
    output, lse = result[:2]
    if lse.shape == (*query.shape[:3], 1):
        lse = lse[..., 0]
    require(output.shape == query.shape and output.dtype == torch.bfloat16 and
            lse.shape == query.shape[:3] and lse.dtype == torch.float32 and
            bool(torch.isfinite(lse).all()), 'Invalid cuDNN output/LSE contract')
    return output, lse


def _working_dtype(value):
    return torch.float64 if value.dtype == torch.float64 else torch.float32


def fixed_mass_output(joint_output, old_text_output, new_text_output, old_text_lse, joint_lse):
    dtype = _working_dtype(joint_output)
    log_mass = old_text_lse.to(dtype)-joint_lse.to(dtype)
    require(bool(torch.isfinite(log_mass).all()) and
            bool((log_mass <= NUMERICAL_POLICY['log_mass_upper_tolerance']).all()), 'Invalid old text mass')
    mass = log_mass.exp()
    out = joint_output.to(dtype)+mass[..., None]*(new_text_output.to(dtype)-old_text_output.to(dtype))
    require(bool(torch.isfinite(out).all()), 'Nonfinite fixed-mass output')
    return out.to(joint_output.dtype), mass


def unnormalized_output(image_output, new_text_output, image_lse, new_text_lse):
    """Stable conditional mixture, with no baseline-minus-text subtraction."""
    dtype = _working_dtype(image_output)
    li, lt = image_lse.to(dtype), new_text_lse.to(dtype)
    require(bool(torch.isfinite(li).all()) and bool(torch.isfinite(lt).all()), 'Nonfinite conditional LSE')
    total = torch.logaddexp(li, lt)
    wi, wt = (li-total).exp(), (lt-total).exp()
    out = wi[..., None]*image_output.to(dtype)+wt[..., None]*new_text_output.to(dtype)
    require(bool(torch.isfinite(out).all()), 'Nonfinite unnormalized phase output')
    return out.to(image_output.dtype), wt


def _update_range(ranges, name, value):
    bounds = torch.stack((value.min(), value.max())).detach()
    if name in ranges:
        previous = ranges[name]
        bounds = torch.stack((torch.minimum(previous[0], bounds[0]), torch.maximum(previous[1], bounds[1])))
    ranges[name] = bounds


@torch.no_grad()
def phase_attention(query, key, value, alternate_image_query, *, text_tokens, mode,
                    query_chunk_size=65536, backend=cudnn_attention, phase_identity=False):
    """Consumed B,H,S,D tensors, including unchanged gains; test backend injectable."""
    require(query.ndim == key.ndim == value.ndim == 4 and query.shape == key.shape == value.shape,
            'Joint attention shape mismatch')
    require(mode in MODES and type(text_tokens) is int and 0 < text_tokens < query.shape[-2],
            'Require nonempty text and image partitions and a supported mode')
    require(type(query_chunk_size) is int and query_chunk_size > 0, 'Positive query chunk size required')
    output, full_lse = backend(query, key, value)
    counts = {'full_joint_calls': 1, 'old_text_calls': 0, 'new_text_calls': 0, 'image_only_calls': 0}
    ranges = {}
    if mode == 'baseline' or phase_identity:
        return output, counts, ranges
    require(alternate_image_query is not None and alternate_image_query.shape == query[:, :, text_tokens:].shape
            and alternate_image_query.dtype == query.dtype and alternate_image_query.device == query.device,
            'Alternate image query shape/dtype/device mismatch')
    image_tokens = query.shape[-2]-text_tokens
    for start in range(0, image_tokens, query_chunk_size):
        end = min(start+query_chunk_size, image_tokens)
        original_q = query[:, :, text_tokens+start:text_tokens+end]
        alternate_q = alternate_image_query[:, :, start:end]
        new_pool, new_lse = backend(alternate_q, key[:, :, :text_tokens], value[:, :, :text_tokens])
        counts['new_text_calls'] += 1
        if mode == 'fixedmass':
            old_pool, old_lse = backend(original_q, key[:, :, :text_tokens], value[:, :, :text_tokens])
            counts['old_text_calls'] += 1
            updated, mass = fixed_mass_output(output[:, :, text_tokens+start:text_tokens+end],
                old_pool, new_pool, old_lse, full_lse[:, :, text_tokens+start:text_tokens+end])
            _update_range(ranges, 'old_text_mass', mass)
        else:
            image_pool, image_lse = backend(original_q, key[:, :, text_tokens:], value[:, :, text_tokens:])
            counts['image_only_calls'] += 1
            updated, mass = unnormalized_output(image_pool, new_pool, image_lse, new_lse)
            _update_range(ranges, 'new_text_mass', mass)
        output[:, :, text_tokens+start:text_tokens+end] = updated
    return output, counts, ranges


class CrossmodalPhaseProcessor:
    """Standard joint/single FLUX processor with one isolated crossmodal branch."""
    def __init__(self, state, name):
        self.state, self.name = state, name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        from diffusers.models.embeddings import apply_rotary_emb
        from diffusers.models.transformers.transformer_flux import _get_qkv_projections
        require(attention_mask is None and image_rotary_emb is not None, 'Masked or missing rotary attention is unsupported')
        state = self.state
        q, k, v, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        q, k, v = (x.unflatten(-1, (attn.heads, -1)) for x in (q, k, v))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if attn.added_kv_proj_dim is not None:
            require(encoder_hidden_states is not None and encoder_hidden_states.shape[1] == state.text_tokens,
                    'Joint text conditioning length mismatch')
            eq, ek, ev = (x.unflatten(-1, (attn.heads, -1)) for x in (eq, ek, ev))
            eq, ek = attn.norm_added_q(eq), attn.norm_added_k(ek)
            q, k, v = (torch.cat([a, b], dim=1) for a, b in ((eq, q), (ek, k), (ev, v)))
        require(q.shape == k.shape == v.shape and q.shape[0] == 1 and
                q.shape[1] == state.text_tokens+math.prod(state.grid) and q.shape[-1] == 128,
                'Unexpected FLUX joint sequence geometry')
        alternate = None
        if state.mode != 'baseline' and not state.native_identity:
            # Use the actual pre-RoPE normalized Q, not an inverse of rounded Q.
            alternate = apply_rotary_emb(q[:, state.text_tokens:], state.target_embedding(q.device), sequence_dim=1)
        q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
        k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        gain = state.controller.qk_scale.to(device=q.device, dtype=q.dtype)
        require(gain.shape == (128,) and bool(torch.isfinite(gain).all()) and
                bool((gain > 0).all()) and torch.equal(gain[::2], gain[1::2]), 'Invalid shared rotary-pair gain')
        consumed_q, consumed_k = (q*gain).transpose(1, 2), (k*gain).transpose(1, 2)
        if alternate is not None:
            alternate = (alternate*gain).transpose(1, 2)
        output, counts, ranges = phase_attention(consumed_q, consumed_k, v.transpose(1, 2), alternate,
            text_tokens=state.text_tokens, mode=state.mode, query_chunk_size=state.query_chunk_size,
            backend=state.backend, phase_identity=state.native_identity)
        state.record(self.name, counts, ranges)
        hidden_states = output.transpose(1, 2).flatten(2, 3).to(q.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1]-encoder_hidden_states.shape[1]], dim=1)
            hidden_states = attn.to_out[1](attn.to_out[0](hidden_states.contiguous()))
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        return hidden_states


class CrossmodalPhaseHandle:
    def __init__(self, transformer, *, mode, grid, controller, text_tokens, query_chunk_size):
        require(mode in MODES and len(grid) == 2 and all(type(x) is int and x >= 2 for x in grid), 'Invalid phase configuration')
        require(text_tokens == 512 and type(query_chunk_size) is int and query_chunk_size > 0,
                'Pilot includes all512 text keys and requires a positive query chunk size')
        rope = transformer.pos_embed
        require(getattr(rope, 'method', None) == 'sega_official' and tuple(rope.grid) == tuple(grid)
                and tuple(rope.axes_dims) == AXES and rope.theta == 10000., 'Expected deployed SEGA rotary embedding')
        self.transformer, self.mode, self.grid, self.controller = transformer, mode, tuple(grid), controller
        self.text_tokens, self.query_chunk_size = text_tokens, query_chunk_size
        self.native_identity = self.grid == (64, 64)
        self.backend, self._embeddings = cudnn_attention, {}
        self.original_processors = dict(transformer.attn_processors)
        require(len(self.original_processors) > 0, 'Transformer has no attention processors')
        self.reset()
        transformer.set_attn_processor({name: CrossmodalPhaseProcessor(self, name) for name in self.original_processors})
        self.pre_hook = transformer.register_forward_pre_hook(self.validate_ids, with_kwargs=True)

    def target_embedding(self, device):
        key = str(device)
        if key not in self._embeddings:
            self._embeddings[key] = native_rotary_embedding(self.grid, device)
        return self._embeddings[key]

    def validate_ids(self, module, args, kwargs):
        text, image = kwargs.get('txt_ids'), kwargs.get('img_ids')
        require(isinstance(text, torch.Tensor) and text.shape == (self.text_tokens, 3)
                and bool((text == 0).all()), 'All text positions, including padding, must be zero')
        require(isinstance(image, torch.Tensor) and image.shape == (math.prod(self.grid), 3), 'Image ID shape mismatch')
        h, w = self.grid
        expected = torch.zeros(h, w, 3, dtype=image.dtype, device=image.device)
        expected[..., 1] = torch.arange(h, dtype=image.dtype, device=image.device)[:, None]
        expected[..., 2] = torch.arange(w, dtype=image.dtype, device=image.device)[None, :]
        require(torch.equal(image, expected.reshape(-1, 3)), 'Expected zero-axis raster image IDs')
        self.id_validations += 1

    def reset(self):
        self.calls, self.id_validations = 0, 0
        self.per_block_calls = {name: 0 for name in self.original_processors}
        self.kernel_calls = {name: 0 for name in ('full_joint_calls', 'old_text_calls', 'new_text_calls', 'image_only_calls')}
        self.mass_ranges = {}

    def record(self, name, counts, ranges):
        self.calls += 1
        self.per_block_calls[name] += 1
        for key, value in counts.items():
            self.kernel_calls[key] += value
        for key, value in ranges.items():
            _update_range(self.mass_ranges, key, value)

    def receipt(self):
        return {'mode': self.mode, 'calls': self.calls, 'per_block_calls': dict(self.per_block_calls),
            'geometry_id_validations': self.id_validations, 'grid': list(self.grid), 'text_tokens': self.text_tokens,
            'all_text_keys_including_padding': True, 'query_chunk_size': self.query_chunk_size,
            'kernel_calls': dict(self.kernel_calls), 'mass_ranges': {k: v.cpu().tolist() for k, v in self.mass_ranges.items()},
            'native_identity_baseline_path': self.native_identity,
            'backend': 'aten._scaled_dot_product_cudnn_attention', 'fallback': None,
            'phase_rule': 'R_base(p*63/(L-1)) on pre-RoPE normalized image Q; axis0 unchanged; same BF16 pair gains',
            'unnormalized_policy': 'Separate image-only conditional attention and new-text conditional attention, stable LSE mixture; full original text-query rows retained',
            'numerical_policy': dict(NUMERICAL_POLICY)}

    def close(self):
        if self.pre_hook is not None:
            self.pre_hook.remove()
            self.pre_hook = None
            self.transformer.set_attn_processor(self.original_processors.copy())


def install_crossmodal_phase(transformer, *, mode, grid, controller, text_tokens=512, query_chunk_size=65536):
    return CrossmodalPhaseHandle(transformer, mode=mode, grid=grid, controller=controller,
                                 text_tokens=text_tokens, query_chunk_size=query_chunk_size)


def _dense_attention(query, key, value):
    logits = query.double() @ key.double().transpose(-2, -1)/math.sqrt(query.shape[-1])
    return logits.softmax(-1) @ value.double(), logits.logsumexp(-1)


@torch.no_grad()
def kernel_smoke(device='cuda'):
    """Prospective tiny actual-CUDA oracle check. Never invoked by CPU tests."""
    require(str(device).startswith('cuda') and torch.cuda.is_available(), 'GPU smoke requires an allocated CUDA device')
    generator = torch.Generator(device='cpu').manual_seed(20261017)
    q, k, v = [torch.randn(1, 2, 40, 128, generator=generator, dtype=torch.float32).to(device, torch.bfloat16)
               for _ in range(3)]
    alternate = torch.randn(1, 2, 32, 128, generator=generator).to(device, torch.bfloat16)
    output, lse = cudnn_attention(q, k, v)
    reference, reference_lse = _dense_attention(q, k, v)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        public = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
    checks = {}
    def compare(name, actual, expected, relative_tolerance, absolute_tolerance):
        difference = actual.double()-expected.double()
        relative = (difference.norm()/expected.double().norm().clamp_min(1e-30)).item()
        maximum = difference.abs().max().item()
        require(relative <= relative_tolerance and maximum <= absolute_tolerance, name+' tolerance failed')
        checks[name] = {'relative_l2': relative, 'max_abs': maximum}
    compare('private_vs_fp64', output, reference, NUMERICAL_POLICY['kernel_output_relative_l2'], NUMERICAL_POLICY['kernel_output_max_abs'])
    compare('private_vs_public_cudnn', output, public, NUMERICAL_POLICY['kernel_output_relative_l2'], NUMERICAL_POLICY['kernel_output_max_abs'])
    require((lse.double()-reference_lse).abs().max().item() <= NUMERICAL_POLICY['kernel_lse_max_abs'], 'LSE oracle tolerance failed')
    checks['lse_max_abs'] = (lse.double()-reference_lse).abs().max().item()
    old_logits = q.double() @ k.double().transpose(-2, -1)/math.sqrt(128)
    new_text_logits = alternate.double() @ k[:, :, :8].double().transpose(-2, -1)/math.sqrt(128)
    for mode in ('fixedmass', 'unnormalized'):
        changed = old_logits.clone()
        replacement = new_text_logits
        if mode == 'fixedmass':
            replacement = replacement+old_logits[:, :, 8:, :8].logsumexp(-1, keepdim=True)-replacement.logsumexp(-1, keepdim=True)
        changed[:, :, 8:, :8] = replacement
        direct = changed.softmax(-1) @ v.double()
        actual, counts, ranges = phase_attention(q, k, v, alternate, text_tokens=8, mode=mode, query_chunk_size=16)
        compare(mode+'_vs_fp64', actual, direct, NUMERICAL_POLICY['operator_relative_l2'], NUMERICAL_POLICY['operator_max_abs'])
        require(torch.equal(actual[:, :, :8].contiguous().view(torch.uint8),
                            output[:, :, :8].contiguous().view(torch.uint8)), 'Text-query output changed')
        identical, _, _ = phase_attention(q, k, v, q[:, :, 8:], text_tokens=8, mode=mode, phase_identity=True)
        require(torch.equal(identical.contiguous().view(torch.uint8), output.contiguous().view(torch.uint8)),
                'Identical phase baseline path is not byte equal')
        checks[mode+'_identity_byte_equal'] = True
        for key, bounds in ranges.items():
            require(bool((bounds >= 0).all()) and bool((bounds <= math.exp(NUMERICAL_POLICY['log_mass_upper_tolerance'])).all()), 'Mass bounds failed')
            checks[mode+'_'+key] = bounds.cpu().tolist()
    # This is linear in image-token count: no full image/image attention probe.
    long_generator = torch.Generator(device=device).manual_seed(20261018)
    # Match the installed B,S,H,D projection followed by transpose, including
    # head stride128 and sequence stride24*128. Batch is always one.
    long_q = torch.randn(1, 65536, 24, 128, device=device, dtype=torch.bfloat16,
                         generator=long_generator).transpose(1, 2)
    short_k, short_v = [torch.randn(1, 512, 24, 128, device=device, dtype=torch.bfloat16,
                                  generator=long_generator).transpose(1, 2) for _ in range(2)]
    long_output, long_lse = cudnn_attention(long_q, short_k, short_v)
    require(bool(torch.isfinite(long_output).all()) and bool(torch.isfinite(long_lse).all()),
            'Production-shaped text-branch probe is nonfinite')
    checks['production_text_branch'] = {'query_shape': list(long_q.shape), 'key_shape': list(short_k.shape),
        'output_shape': list(long_output.shape), 'lse_shape': list(long_lse.shape), 'finite': True,
        'query_stride': list(long_q.stride()), 'key_stride': list(short_k.stride()),
        'cuda_rng_seed': 20261018, 'sampling_dtype': 'bfloat16',
        'scope': 'Geometry/support probe only, without full-size FP64 oracle or image/image branch.'}
    return {'passed': True, 'backend': 'aten._scaled_dot_product_cudnn_attention',
            'tolerances': dict(NUMERICAL_POLICY), 'checks': checks, 'seed': 20261017,
            'geometry': {'batch': 1, 'heads': 2, 'text_tokens': 8, 'image_tokens': 32, 'head_dim': 128},
            'full_4k_shape_executed': False, 'gpu_runtime_verified': True,
            'scope': 'Tiny kernel oracle and identical-phase checks; not full-model quality or speed evidence.'}
