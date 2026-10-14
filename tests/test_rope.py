"""CPU numerical contracts for frequencies and upstream baseline behavior."""

import ast
import math
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pytest
import torch
from diffusers.models.transformers.transformer_flux import FluxPosEmbed

from crope.rope import FluxRoPE, METHODS
from crope.sega_official import OfficialSEGA, allocation, axis_profiles, radial_profile, spectral_spread


THIRD_PARTY = Path(os.environ.get("CROPE_THIRD_PARTY", "/scratch/09032/saini_2/constancy_rope/third_party"))


def _source_definitions(path, names):
    """Execute only numerical definitions, with no imports or model loading."""
    if not path.is_file():
        pytest.skip(f"Upstream parity checkout absent: {path}")
    parsed = ast.parse(path.read_text())
    nodes = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    assert len(nodes) == len(names), "Upstream numerical API changed"
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = dict(torch=torch, np=np, math=math, nn=torch.nn, List=List, Optional=Optional, Union=Union)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def upstream_dype():
    names = {"find_correction_factor", "find_correction_range", "linear_ramp_mask",
             "find_newbase_ntk", "get_1d_rotary_pos_embed", "FluxPosEmbed"}
    return _source_definitions(THIRD_PARTY / "DyPE/flux/transformer_flux.py", names)


@pytest.fixture(scope="module")
def upstream_sega():
    names = {"compute_base_mscale", "compute_sega_allocation", "get_1d_rotary_pos_embed",
             "compute_spectral_energy_profile", "compute_axis_spectral_profiles",
             "compute_dynamic_spread", "FluxPosEmbed"}
    return _source_definitions(THIRD_PARTY / "sega/flux_sega/transformer_flux.py", names)


def ids_for(grid):
    generator = torch.Generator().manual_seed(7)
    ids = torch.stack((torch.randint(0, 3, (31,), generator=generator),
                       torch.randint(0, grid[0], (31,), generator=generator),
                       torch.randint(0, grid[1], (31,), generator=generator)), dim=-1).float()
    ids[0, 1], ids[0, 2] = grid[0] - 1, grid[1] - 1
    return ids


@pytest.mark.parametrize("dimensions", [(16, 56, 56), (4, 6, 6)])
def test_base_is_exact_diffusers(dimensions):
    ids = ids_for((128, 256))
    expected = FluxPosEmbed(theta=10000, axes_dim=list(dimensions))(ids)
    actual = FluxRoPE(axes_dims=dimensions, grid=(128, 256))(ids)
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)


@pytest.mark.parametrize("method", METHODS)
def test_native_identity(method):
    ids = ids_for((64, 64))
    expected = FluxRoPE(method="base")(ids)
    actual = FluxRoPE(method=method)(ids)
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)


@pytest.mark.parametrize("method,strength", [("ntk", 1), ("ntk2", 2)])
def test_ntk_formula_and_per_axis_scaling(method, strength):
    rope = FluxRoPE(method=method, grid=(128, 256))
    for axis, scale in ((1, 2), (2, 4)):
        new_base = 10000 * scale ** (strength * 56 / 54)
        expected = 1 / new_base ** (torch.arange(0, 56, 2, dtype=torch.float64) / 56)
        torch.testing.assert_close(rope.frequencies(axis), expected, atol=0, rtol=0)
    torch.testing.assert_close(rope.frequencies(0), FluxRoPE().frequencies(0), atol=0, rtol=0)


def test_pi_and_design_yarn():
    for axis, scale in ((1, 2), (2, 4)):
        base = FluxRoPE().frequencies(axis)
        pi = FluxRoPE(method="pi", grid=(128, 256)).frequencies(axis)
        torch.testing.assert_close(pi, base / scale, atol=0, rtol=0)
        ratio = 2 * math.pi / base / 64
        ramp = ((ratio - 1) / 31).clamp(0, 1)
        expected = base * (1 - ramp) + base / scale * ramp
        yarn = FluxRoPE(method="yarn", grid=(128, 256)).frequencies(axis)
        torch.testing.assert_close(yarn, expected, atol=0, rtol=0)
        torch.testing.assert_close(yarn[ratio <= 1], base[ratio <= 1])
        torch.testing.assert_close(yarn[ratio >= 32], pi[ratio >= 32])


@pytest.mark.parametrize("grid", [(256, 256), (128, 256), (64, 256)])
@pytest.mark.parametrize("sigma", [1.0, 0.7, 0.3, 0.05])
def test_dype_exact_upstream_cos_sin(upstream_dype, grid, sigma):
    ids = ids_for(grid)
    official = upstream_dype["FluxPosEmbed"](theta=10000, axes_dim=[16, 56, 56], method="yarn", dype=True)
    official.set_timestep(sigma)
    port = FluxRoPE(method="dype", grid=grid)
    port.set_timestep(sigma)
    expected, actual = official(ids), port(ids)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_dype_amplitudes_and_endpoint():
    rope = FluxRoPE(method="dype", grid=(128, 256))
    gain = rope.embedding_gain()
    assert torch.equal(gain[:16], torch.ones(16))
    torch.testing.assert_close(gain[16:72], torch.full((56,), 1 + 0.1 * math.log(2)))
    torch.testing.assert_close(gain[72:], torch.full((56,), 1 + 0.1 * math.log(4)))
    rope.set_timestep(0)
    for axis in (0, 1, 2):
        torch.testing.assert_close(rope.frequencies(axis), FluxRoPE().frequencies(axis), rtol=0, atol=0)
    assert all(torch.isfinite(value).all() for value in rope(ids_for((128, 256))))


@pytest.mark.parametrize("grid", [(64, 64), (128, 256)])
def test_official_sega_helpers_and_full_gain(upstream_sega, grid):
    generator = torch.Generator().manual_seed(71)
    features = torch.randn((1, grid[0] * grid[1], 7), generator=generator)
    spatial = features.reshape(1, *grid, 7).float().mean(dim=(0, -1))
    spatial = spatial - spatial.mean()
    expected_axes = upstream_sega["compute_axis_spectral_profiles"](
        features, *grid, max(grid[0] // 2, 8), max(grid[1] // 2, 8))
    for actual, expected in zip(axis_profiles(spatial), expected_axes):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_radial = upstream_sega["compute_spectral_energy_profile"](
        features, *grid, max(grid) // 2)
    torch.testing.assert_close(radial_profile(spatial), expected_radial, rtol=0, atol=0)
    spread = upstream_sega["compute_dynamic_spread"](expected_radial)
    assert spectral_spread(expected_radial) == spread
    controller = OfficialSEGA(grid)
    actual_gain = controller.update(features, 0.5)
    official = upstream_sega["FluxPosEmbed"](theta=10000, axes_dim=[16, 56, 56],
                                            training_res_pixels=1024,
                                            base_mscale_coefficient=0.08)
    expected_cos, expected_sin = official(
        ids_for(grid), ntk_factor=controller.ntk_factor, mscale_spread=spread,
        energy_profile_h=expected_axes[0], energy_profile_w=expected_axes[1],
        target_res=max(grid) * 16)
    port_cos, port_sin = controller.rope(ids_for(grid))
    for actual, expected in ((port_cos * actual_gain, expected_cos),
                             (port_sin * actual_gain, expected_sin)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for axis, profile in enumerate(expected_axes, 1):
        frequencies = controller.rope.frequencies(axis)
        expected = upstream_sega["compute_sega_allocation"](
            profile, frequencies, controller.reference_gain, spread)
        actual = allocation(profile, frequencies, controller.reference_gain, spread)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_validation():
    with pytest.raises(ValueError):
        FluxRoPE(method="unknown")
    with pytest.raises(ValueError):
        FluxRoPE(axes_dims=(16, 55, 56))
    with pytest.raises(ValueError):
        FluxRoPE(grid=(0, 256))
    with pytest.raises(ValueError):
        FluxRoPE().set_timestep(2)
