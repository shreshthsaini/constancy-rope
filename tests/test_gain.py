"""CPU tests for all-token attention gains and spectral controller contracts."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import FluxTransformer2DModel

import crope.gain as gain_module
import crope.runtime as runtime
from crope.gain import FluxGainAttnProcessor, GainController, gained_attention, install_gain, sega_modulator
from crope.rope import FluxRoPE
from crope.spectral import NativeProfile, axis_power, log_interpolate, token_grid


@pytest.fixture(scope="module", autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_flux():
    torch.manual_seed(123)
    return FluxTransformer2DModel(in_channels=64, out_channels=64, num_layers=1,
                                  num_single_layers=1, num_attention_heads=2,
                                  attention_head_dim=16, joint_attention_dim=12,
                                  pooled_projection_dim=8, axes_dims_rope=(4, 6, 6)).eval()


def model_inputs():
    generator = torch.Generator().manual_seed(719)
    yy, xx = torch.meshgrid(torch.arange(2), torch.arange(3), indexing="ij")
    return {
        "hidden_states": torch.randn(1, 6, 64, generator=generator),
        "encoder_hidden_states": torch.randn(1, 4, 12, generator=generator),
        "pooled_projections": torch.randn(1, 8, generator=generator),
        "timestep": torch.tensor([0.77]),
        "img_ids": torch.stack((torch.zeros(6), yy.flatten(), xx.flatten()), dim=-1),
        "txt_ids": torch.zeros(4, 3),
        "return_dict": False,
    }


def test_ones_reproduce_complete_joint_and_single_flux_exactly(tiny_flux):
    kwargs = model_inputs()
    with torch.inference_mode():
        expected = tiny_flux(**kwargs)[0]
        controller = GainController("none", grid=(2, 3), axes_dims=(4, 6, 6))
        controller.update(token_grid(kwargs["hidden_states"], (2, 3)), 0.77)
        install_gain(tiny_flux, controller)
        assert len(tiny_flux.attn_processors) == 2
        assert all(isinstance(p, FluxGainAttnProcessor) for p in tiny_flux.attn_processors.values())
        actual = tiny_flux(**kwargs)[0]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("logit_gain", [0.5, 1.5, 3.0])
def test_uniform_qk_gain_is_logit_multiplier(logit_gain):
    generator = torch.Generator().manual_seed(11)
    q, k, v = [torch.randn(1, 9, 2, 8, generator=generator, dtype=torch.float64) for _ in range(3)]
    logits = torch.einsum("bshd,bthd->bhst", q, k) * logit_gain / math.sqrt(8)
    expected = torch.einsum("bhst,bthd->bshd", logits.softmax(-1), v)
    actual = gained_attention(q, k, v, torch.full((8,), math.sqrt(logit_gain), dtype=torch.float64))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def _manual_rotate(value, embeddings):
    cos, sin = (component[None, :, None, :] for component in embeddings)
    rotated = torch.stack((-value[..., 1::2], value[..., 0::2]), dim=-1).flatten(-2)
    return value * cos + rotated * sin


@pytest.mark.parametrize("joint", [True, False])
def test_spatial_gains_reach_every_token_after_rope(tiny_flux, monkeypatch, joint):
    attn = tiny_flux.transformer_blocks[0].attn if joint else tiny_flux.single_transformer_blocks[0].attn
    generator = torch.Generator().manual_seed(91)
    image = torch.randn(1, 6, 32, generator=generator)
    text = torch.randn(1, 4, 32, generator=generator)
    states = image if joint else torch.cat((text, image), dim=1)
    source = model_inputs()
    ids = torch.cat((source["txt_ids"], source["img_ids"]), dim=0)
    embeddings = FluxRoPE(axes_dims=(4, 6, 6), grid=(2, 3))(ids)
    gain = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 1.3, 1.3,
                         1.7, 1.7, 0.6, 0.6, 1.1, 1.1, 1.5, 1.5])
    controller = SimpleNamespace(qk_scale=gain)
    q, k, v = [layer(states).unflatten(-1, (attn.heads, -1))
               for layer in (attn.to_q, attn.to_k, attn.to_v)]
    q, k = attn.norm_q(q), attn.norm_k(k)
    if joint:
        eq, ek, ev = [layer(text).unflatten(-1, (attn.heads, -1))
                      for layer in (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj)]
        q = torch.cat((attn.norm_added_q(eq), q), dim=1)
        k = torch.cat((attn.norm_added_k(ek), k), dim=1)
        v = torch.cat((ev, v), dim=1)
    q, k = _manual_rotate(q, embeddings), _manual_rotate(k, embeddings)
    calls = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def inspect_sdpa(actual_q, actual_k, actual_v, **kwargs):
        calls.append(1)
        for actual, expected in ((actual_q, (q * gain).transpose(1, 2)),
                                 (actual_k, (k * gain).transpose(1, 2)),
                                 (actual_v, v.transpose(1, 2))):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # A spatial gain changes text q/k too, although their spatial ids are zero.
        assert not torch.equal(actual_q[:, :, :4, 4:], q[:, :4, :, 4:].transpose(1, 2))
        return sdpa(actual_q, actual_k, actual_v, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", inspect_sdpa)
    actual = FluxGainAttnProcessor(controller)(attn, states, text if joint else None,
                                               image_rotary_emb=embeddings)
    logits = torch.einsum("bshd,bthd->bhst", q * gain, k * gain) / math.sqrt(16)
    expected = torch.einsum("bhst,bthd->bshd", logits.softmax(-1), v).flatten(2, 3)
    assert calls == [1]
    if joint:
        expected_text = attn.to_add_out(expected[:, :4].contiguous())
        expected_image = attn.to_out[1](attn.to_out[0](expected[:, 4:].contiguous()))
        for a, b in zip(actual, (expected_image, expected_text)):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.fixture(scope="module")
def reference_sega():
    path = Path("/work/09032/saini_2/vista/Projects/Exploration/research_reviews/"
                "sega_2605.22668_2026-10-14/sega_stats.py")
    namespace = {"__name__": "sega_reference_for_tests"}
    # Compile the approved source in memory without writing a sibling pyc file.
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("length", [64, 128, 192])
@pytest.mark.parametrize("ntk2", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sega_matches_provided_reference_and_axis_order(reference_sega, length, ntk2, dtype):
    rng = np.random.default_rng(830)
    yy, xx = np.meshgrid(np.arange(length), np.arange(length), indexing="ij")
    value = rng.standard_normal((length, length))
    value += 2 * np.sin(2 * np.pi * xx * 7 / length) + np.cos(2 * np.pi * yy * 3 / length)
    value = value.astype(dtype)
    theta = FluxRoPE(method="ntk2" if ntk2 else "base", grid=(length, length)).frequencies(1)
    expected, expected_sigma = reference_sega.modulator(value, theta.numpy())
    actual, actual_sigma = sega_modulator(torch.from_numpy(value), theta)
    # NumPy2 and PyTorch use distinct float32 FFT/reduction kernels. Keep the
    # same input dtype and allow their small arithmetic differences near M=0.
    rtol, atol = (2e-5, 1e-5) if dtype == np.float32 else (1e-10, 1e-12)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual_sigma.numpy(), expected_sigma, rtol=rtol, atol=atol)
    assert not np.allclose(expected[0], expected[1])


@pytest.fixture(scope="module")
def native_latents():
    generator = torch.Generator().manual_seed(561)
    return torch.randn(64, 64, 64, generator=generator, dtype=torch.float64)


def blur(value, sigma=1.2):
    h, w = value.shape[-2:]
    fy = torch.fft.fftfreq(h, dtype=value.dtype)[:, None]
    fx = torch.fft.fftfreq(w, dtype=value.dtype)[None, :]
    response = torch.exp(-2 * math.pi ** 2 * sigma ** 2 * (fy.square() + fx.square()))
    return torch.fft.ifft2(torch.fft.fft2(value) * response).real


def test_constancy_native_profile_has_zero_deficit(native_latents):
    controller = GainController("constancy", profile=NativeProfile.from_latents(native_latents))
    controller.update(native_latents, 0.02, native_latents)
    for axis in controller.history[-1]["bins"]:
        assert axis["anchor_count"] > 0
        np.testing.assert_allclose(axis["deficit"], 0, atol=1e-12)
        np.testing.assert_allclose(axis["correction"], 0, atol=1e-12)
    torch.testing.assert_close(controller.qk_scale, torch.ones(128), rtol=0, atol=0)


def test_constancy_blur_produces_positive_high_bin_correction(native_latents):
    blurred = blur(native_latents)
    controller = GainController("constancy", profile=NativeProfile.from_latents(native_latents))
    controller.update(blurred, 0.001, blurred)
    for axis in controller.history[-1]["bins"]:
        frequency = np.asarray(axis["frequency"])
        high = (frequency >= 1 / 8) & (frequency <= 1 / 4)
        assert high.any() and axis["anchor_count"] > 0
        assert np.asarray(axis["deficit"])[high].min() > 0.3
        assert np.asarray(axis["correction"])[high].min() > 0.3
    assert controller.log_g[16] > 0 and controller.log_g[72] > 0


def test_constancy_gate_is_small_at_pure_noise(native_latents):
    sigma = 0.999
    noise = torch.randn(native_latents.shape, generator=torch.Generator().manual_seed(86), dtype=torch.float64)
    controller = GainController("constancy", profile=NativeProfile.from_latents(native_latents))
    controller.update(sigma * noise, sigma, blur(native_latents))
    for axis in controller.history[-1]["bins"]:
        assert max(axis["measured_gate"]) < 0.05
        assert axis["anchor_count"] == 0
    torch.testing.assert_close(controller.qk_scale, torch.ones(128), rtol=0, atol=0)


def test_long_wavelengths_and_first_step_receive_temperature_only():
    grid = (128, 128)
    clean = torch.randn(8, *grid, generator=torch.Generator().manual_seed(301))
    controller = GainController("constancy", grid=grid, profile=NativeProfile.from_latents(clean))
    expected = math.log(math.log(math.prod(grid)) / math.log(4096))
    controller.update(clean, 0.9)
    torch.testing.assert_close(controller.log_g, torch.full((128,), expected))
    assert controller.history[-1]["first_step_no_correction"]
    controller.update(blur(clean), 0.001, blur(clean))
    for axis, offset in ((1, 16), (2, 72)):
        below_grid = (controller.frequencies(axis, "cpu") / (2 * math.pi) < 1 / grid[axis - 1]).repeat_interleave(2)
        assert below_grid.any()
        torch.testing.assert_close(controller.log_g[offset:offset + 56][below_grid],
                                   torch.full((int(below_grid.sum()),), expected))
    torch.testing.assert_close(controller.log_g[:16], torch.full((16,), expected))


def test_nogate_and_sref_ablations(native_latents):
    profile = NativeProfile.from_latents(native_latents)
    controller = GainController("constancy_nogate", profile=profile)
    controller.update(native_latents, 1.0, blur(native_latents))
    assert all(set(axis["gate"]) == {1.0} for axis in controller.history[-1]["bins"])
    assert any(max(axis["correction"]) > 0.1 for axis in controller.history[-1]["bins"])
    controller = GainController("constancy_sref", grid=(128, 256), profile=profile)
    controller.update(torch.zeros(1, 128, 256), 1.0)
    torch.testing.assert_close(controller.qk_scale[16:72], torch.full((56,), 2 ** 0.08))
    torch.testing.assert_close(controller.qk_scale[72:], torch.full((56,), 4 ** 0.08))
    torch.testing.assert_close(controller.qk_scale[:16], torch.full((16,), math.sqrt(8) ** 0.08))


@pytest.fixture
def time_profile(native_latents):
    fields = [blur(native_latents, s) for s in (2.0, 1.0, 0.5, 0.0)]
    profiles = [NativeProfile.from_latents(x) for x in fields]
    return NativeProfile(profiles[-1].frequencies, profiles[-1].log_power,
                         torch.tensor([0.9, 0.6, 0.3, 0.1], dtype=torch.float64),
                         tuple(torch.stack([p.log_power[a] for p in profiles]) for a in (0, 1))), fields


@pytest.mark.parametrize("mode", ["constancy_tm", "constancy_tm_nogate"])
def test_time_matched_native_trajectory_has_zero_deficit_at_every_step(time_profile, mode):
    profile, fields = time_profile
    controller = GainController(mode, profile=profile)
    for sigma, x0 in zip(profile.sigma_steps, fields):
        controller.update(5 * fields[-1], sigma, x0)
        for axis in controller.history[-1]["bins"]:
            assert axis["anchor_count"] > 0
            np.testing.assert_allclose(axis["deficit"], 0, atol=1e-12)
            np.testing.assert_allclose(axis["correction"], 0, atol=1e-12)
        torch.testing.assert_close(controller.qk_scale, torch.ones(128), rtol=0, atol=0)


def test_time_matched_blur_corrects_high_bands_at_mid_sigma(time_profile):
    profile, fields = time_profile
    sigma, x0 = 0.6, blur(fields[1], 1.2)
    controller = GainController("constancy_tm", profile=profile)
    controller.update(5 * fields[-1], sigma, x0)
    for axis in controller.history[-1]["bins"]:
        f = np.asarray(axis["frequency"])
        high = (f >= 1 / 8) & (f <= 1 / 4)
        assert np.asarray(axis["deficit"])[high].min() > 0.3
        assert np.asarray(axis["correction"])[high].min() > 0.3


def test_time_matched_anchor_keeps_hidden_bands_and_gate_off_is_equivalent(time_profile):
    profile, fields = time_profile
    xt, x0 = torch.zeros_like(fields[1]), blur(fields[1])
    gated = GainController("constancy_tm", profile=profile)
    gated.update(xt, 0.6, x0)
    for axis in gated.history[-1]["bins"]:
        assert axis["anchor_count"] > 0 and max(axis["deficit"]) > 0.1
        assert set(axis["measured_gate"]) == {0.0}
        assert set(axis["correction"]) == {0.0}
    ablation = GainController("constancy_tm_nogate", profile=profile)
    alias = GainController("constancy_tm", profile=profile, gate="off")
    ablation.update(xt, 0.6, x0)
    alias.update(xt, 0.6, x0)
    assert alias.mode == "constancy_tm_nogate"
    assert alias.history == ablation.history
    assert any(max(axis["correction"]) > 0.1 for axis in alias.history[-1]["bins"])
    assert all(set(axis["gate"]) == {1.0} for axis in alias.history[-1]["bins"])


def test_time_matched_no_anchor_uses_zero_offset(time_profile):
    profile, _ = time_profile
    x0 = torch.randn(8, 8, 8, generator=torch.Generator().manual_seed(93), dtype=torch.float64)
    controller = GainController("constancy_tm_nogate", grid=(8, 8), profile=profile)
    controller.update(x0, 0.6, x0)
    for a, axis in enumerate(controller.history[-1]["bins"]):
        assert axis["anchor_count"] == 0 and axis["anchor_offset"] == 0
        power = axis_power(x0)[a]
        expected = 0.5 * (profile.evaluate_time(a, power.frequency, 0.6) - power.power.log())
        np.testing.assert_allclose(axis["deficit"], expected.numpy(), atol=1e-12)
        assert expected.abs().max() > 0.1


@pytest.mark.parametrize("mode", ["constancy_tm", "constancy_tm_nogate"])
def test_time_matched_first_step_and_axis_zero_get_only_temperature(time_profile, mode):
    profile, _ = time_profile
    clean = torch.randn(8, 128, 256, generator=torch.Generator().manual_seed(19))
    controller = GainController(mode, grid=(128, 256), profile=profile)
    expected = math.log(math.log(128 * 256) / math.log(4096))
    controller.update(clean, 0.9)
    torch.testing.assert_close(controller.log_g, torch.full((128,), expected))
    assert controller.history[-1]["first_step_no_correction"]
    controller.update(clean, 0.6, blur(clean))
    torch.testing.assert_close(controller.log_g[:16], torch.full((16,), expected))
    torch.testing.assert_close(controller.qk_scale[::2], controller.qk_scale[1::2], rtol=0, atol=0)


@pytest.mark.parametrize("grid", [(64, 64), (256, 256), (128, 256)])
def test_yarn_temperature_scales_both_queries_and_keys(grid):
    controller = GainController("yarn_temp", grid=grid)
    controller.update(torch.zeros(1, *grid), 0.5)
    scale = math.sqrt(math.prod(grid)) / 64
    amplitude = 1 + 0.1 * math.log(scale)
    torch.testing.assert_close(controller.qk_scale, torch.full((128,), amplitude))
    torch.testing.assert_close(controller.log_g.exp(), torch.full((128,), amplitude ** 2))


@pytest.mark.parametrize("grid", [(32, 64), (48, 96)])
def test_rectangular_axis_power_parseval_and_noise_floor(grid):
    channels = 64
    value = torch.randn(channels, *grid, generator=torch.Generator().manual_seed(71), dtype=torch.float64)
    powers = axis_power(value)
    for axis, power in enumerate(powers):
        other = grid[1 - axis]
        centered = value - value.mean(dim=axis + 1, keepdim=True)
        expected = centered.square().sum() / other
        torch.testing.assert_close((power.power * power.count).sum(), expected, rtol=1e-12, atol=1e-6)
        torch.testing.assert_close(power.noise, torch.full_like(power.noise, channels))
        assert power.power.shape == power.frequency.shape == power.noise.shape
        assert torch.all(power.lower < power.frequency) and torch.all(power.frequency < power.upper)
        torch.testing.assert_close(power.power / power.noise, torch.ones_like(power.power), rtol=0.05, atol=0.01)


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_power_law_density_and_constancy_transfer_from_64_to_256(alpha):
    fields = []
    for length in (64, 256):
        frequency = torch.fft.fftfreq(length, dtype=torch.float64)
        radius = torch.hypot(frequency[:, None], frequency[None, :])
        amplitude = radius.clamp_min(1e-9).pow(-alpha / 2)
        amplitude[0, 0] = 0
        white = torch.randn(32, length, length, dtype=torch.float64,
                            generator=torch.Generator().manual_seed(113 + length))
        # Sample the same continuous 1/f^alpha spectrum in cycles/token on each grid.
        fields.append(torch.fft.ifft2(torch.fft.fft2(white) * amplitude).real)
    native, target = map(axis_power, fields)
    for small, large in zip(native, target):
        indices = torch.searchsorted(large.frequency, small.frequency)
        torch.testing.assert_close(large.frequency[indices], small.frequency)
        # Finite realizations and sparse native DFT bins limit band agreement.
        torch.testing.assert_close(large.power[indices], small.power, rtol=0.12, atol=0)
    controller = GainController("constancy", grid=(256, 256), profile=NativeProfile.from_latents(fields[0]))
    controller.update(fields[1], 0.01, fields[1])
    for axis in controller.history[-1]["bins"]:
        assert axis["anchor_count"] > 0
        np.testing.assert_allclose(axis["deficit"], 0, atol=0.05)


@pytest.mark.parametrize("mode", ["constancy", "constancy_tm", "constancy_tm_nogate"])
def test_all_resolvable_ntk2_pairs_interpolate_corrections_at_4096_pixels(mode):
    grid = (256, 256)  # FLUX packs 4096x4096 pixels into 256x256 tokens.
    clean = torch.randn(8, *grid, generator=torch.Generator().manual_seed(19), dtype=torch.float64)
    native = axis_power(clean[:, :64, :64])
    profile = NativeProfile(tuple(a.frequency for a in native),
                            tuple(-2 * a.frequency.log() for a in native))
    if mode.startswith("constancy_tm"):
        profile = NativeProfile(profile.frequencies, profile.log_power, torch.tensor([0.9, 0.01]),
                                tuple(torch.stack((p - 1, p)) for p in profile.log_power))
    rope = FluxRoPE(method="ntk2", grid=grid)
    controller = GainController(mode, grid=grid, profile=profile, rope=rope)
    controller.update(clean, 0.01, clean)
    temperature = math.log(math.log(math.prod(grid)) / math.log(4096))
    for a, (power, offset) in enumerate(zip(axis_power(clean), (16, 72))):
        frequency = rope.frequencies(a + 1) / (2 * math.pi)
        resolvable = frequency >= 1 / grid[a]
        membership = (frequency[:, None] >= power.lower) & (frequency[:, None] < power.upper)
        assert (resolvable & ~membership.any(1)).any()
        bins = controller.history[-1]["bins"][a]
        expected = np.interp(np.log(frequency.numpy()), np.log(bins["frequency"]), bins["correction"])
        expected[~resolvable.numpy()] = 0
        actual = controller.log_g[offset:offset + 56:2] - temperature
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        assert (actual[resolvable].abs() > 1e-3).all()
        torch.testing.assert_close(actual[~resolvable], torch.zeros_like(actual[~resolvable]))


def test_correction_interpolation_holds_outermost_values():
    frequency = torch.tensor([0.01, 0.04], dtype=torch.float64)
    values = torch.tensor([2.0, 4.0], dtype=torch.float64)
    query = torch.tensor([0.005, 0.01, 0.02, 0.04, 0.08], dtype=torch.float64)
    expected = torch.tensor([2.0, 2.0, 3.0, 4.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(log_interpolate(query, frequency, values, extrapolate=False), expected)
    torch.testing.assert_close(log_interpolate(query, frequency[:1], values[:1], extrapolate=False),
                               torch.full_like(query, 2))


def test_native_low_frequency_extrapolation_is_log_linear():
    frequencies = torch.tensor([1 / 64, 1 / 32, 1 / 16], dtype=torch.float64)
    values = 3 - 1.7 * frequencies.log()
    query = torch.tensor([1 / 256, 1 / 128, 1 / 48, 1 / 16], dtype=torch.float64)
    torch.testing.assert_close(log_interpolate(query, frequencies, values), 3 - 1.7 * query.log())


@pytest.mark.parametrize("method", ["base", "sega_official", "sega_official_static",
                                    "constancy_tm", "constancy_tm_nogate"])
def test_runtime_updates_once_and_uses_exact_scheduler_sigma(tiny_flux, monkeypatch, tmp_path, method):
    monkeypatch.setattr(runtime, "code_revision", lambda: "test")
    monkeypatch.setattr(runtime, "model_revision", lambda: "test")
    sigmas = torch.tensor([0.777712345, 0.333312345, 0], dtype=torch.float64)
    pipe = SimpleNamespace(transformer=tiny_flux, scheduler=SimpleNamespace(sigmas=sigmas, config={}))
    original_rope, original_processors = tiny_flux.pos_embed, dict(tiny_flux.attn_processors)
    static = None
    if method == "sega_official_static":
        static = tmp_path / "official_static.npz"
        np.savez(static, mean_qk_scale=np.repeat([1, 1, 1.1, 1.2, 1.3, 1.2, 1.3, 1.4], 2),
                 grid=[2, 3], source_method="sega_official")
    profile = None
    if method.startswith("constancy_tm"):
        profile = tmp_path / "native_tm.npz"
        f = torch.tensor([1 / 64, 1 / 16, 1 / 4])
        NativeProfile((f, f), (f * 0, f * 0), torch.tensor([0.9, 0.1]),
                      (torch.zeros(2, 3), torch.zeros(2, 3))).save(profile)
    resolution = (64, 64) if profile is not None else (32, 48)
    experiment = runtime.FluxExperiment(pipe, method, resolution, profile=profile, static_profile=static)
    if method == "sega_official_static":
        assert experiment.rope.method == "sega_official" and experiment.official is None
    calls, observed = [], []
    target = experiment.official if method == "sega_official" else experiment.controller
    original_update = target.update

    def update(*args):
        calls.append(tuple(value.clone() if isinstance(value, torch.Tensor) else value for value in args))
        return original_update(*args)

    monkeypatch.setattr(target, "update", update)
    experiment.reset(lambda step, sigma, xt, x0: observed.append((step, sigma, xt.clone(), x0.clone())))
    kwargs = model_inputs()
    if profile is not None:
        kwargs["hidden_states"] = torch.randn(1, 16, 64)
        yy, xx = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        kwargs["img_ids"] = torch.stack((torch.zeros(16), yy.flatten(), xx.flatten()), dim=-1)
    try:
        with torch.inference_mode():
            for index in range(2):
                kwargs["timestep"] = sigmas[index:index + 1].to(torch.bfloat16)
                output = tiny_flux(**kwargs)[0]
                expected = token_grid(kwargs["hidden_states"], experiment.grid).float()
                expected = expected - float(sigmas[index]) * token_grid(output.float(), experiment.grid)
                torch.testing.assert_close(experiment.previous_x0, expected, atol=0, rtol=0)
                assert experiment.previous_x0.dtype == torch.float32
                assert calls[index][1] == float(sigmas[index])
                assert calls[index][1] != float(kwargs["timestep"][0])
        assert len(calls) == len(experiment.controller.history) == experiment.step == 2
        assert [row[:2] for row in observed] == [(0, float(sigmas[0])), (1, float(sigmas[1]))]
        if method != "sega_official":
            assert calls[0][2] is None
            torch.testing.assert_close(calls[1][2], observed[0][3], atol=0, rtol=0)
        else:
            assert calls[0][0].shape == (1, 6, 32)
    finally:
        experiment.close()
    assert tiny_flux.pos_embed is original_rope
    assert all(tiny_flux.attn_processors[key] is value for key, value in original_processors.items())
