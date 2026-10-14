"""Independent numerical implementation of the released SEGA Flux behavior.

This optional baseline preserves the released code's projected-feature input,
frequency-rank mapping, clipping, and shared longest-side scale. The requested
reference implementation remains available separately in ``crope.gain``. See
THIRD_PARTY.md for the source revision and the differences from design.md.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .rope import FluxRoPE


def axis_profiles(spatial_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive-frequency axis profiles with the released code's rebinning."""
    result = []
    for axis, length in enumerate(spatial_map.shape):
        count = max(length // 2, 8)
        power = torch.fft.fft(spatial_map, dim=axis).abs().square().mean(dim=1 - axis)
        half = length // 2 + 1
        power = power[:half]
        index = (torch.linspace(0, 1, half, device=power.device) * count).long().clamp(0, count - 1)
        total = torch.zeros(count, device=power.device, dtype=torch.float32)
        number = torch.zeros_like(total)
        total.scatter_add_(0, index, power.float())
        number.scatter_add_(0, index, torch.ones_like(power, dtype=torch.float32))
        result.append(total / (number + 1e-8))
    return tuple(result)


def radial_profile(spatial_map: torch.Tensor) -> torch.Tensor:
    """Radial average using normalized corner distance and a bin containing DC."""
    height, width = spatial_map.shape
    count = max(height, width) // 2
    if count < 1:
        raise ValueError("SEGA requires a spatial side of at least two tokens")
    power = torch.fft.fftshift(torch.fft.fft2(spatial_map)).abs().square()
    yy = torch.arange(height, device=power.device, dtype=torch.float32) - height / 2
    xx = torch.arange(width, device=power.device, dtype=torch.float32) - width / 2
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    radius = torch.sqrt(yy.square() + xx.square())
    radius = radius / (radius.max() + 1e-8)
    index = (radius * count).long().clamp(0, count - 1).flatten()
    total = torch.zeros(count, device=power.device, dtype=torch.float32)
    number = torch.zeros_like(total)
    total.scatter_add_(0, index, power.flatten())
    number.scatter_add_(0, index, torch.ones_like(power).flatten())
    return total / (number + 1e-8)


def spectral_spread(energy: torch.Tensor) -> float:
    energy = energy.clamp(min=1e-8)
    geometric = energy.log().mean().exp()
    flatness = (geometric / (energy.mean() + 1e-8)).clamp(0, 1).item()
    # Keep the two subtractions in the released implementation for parity.
    concentration = 1.0 - flatness
    return 1.0 - (1.0 - concentration) ** 1.5


def allocation(energy: torch.Tensor, frequencies: torch.Tensor,
               reference_gain: float, spread: float) -> torch.Tensor:
    """Released default q/k allocation: alpha=.15, beta=1.5, minimum=1."""
    if spread <= 0:
        return torch.full_like(frequencies, reference_gain, dtype=torch.float32)
    log_period = (2 * math.pi / frequencies.clamp(min=1e-8)).log()
    span = log_period.max() - log_period.min()
    if span.item() > 1e-6:
        rank = (log_period - log_period.min()) / span
    else:
        rank = torch.zeros_like(log_period)
    location = (1 - rank) * (len(energy) - 1)
    lo = location.floor().long().clamp(0, len(energy) - 1)
    hi = (lo + 1).clamp(0, len(energy) - 1)
    fraction = (location - lo.to(location.dtype)).clamp(0, 1)
    log_energy = energy.to(frequencies.device).clamp(min=1e-8).log()
    sampled = log_energy[lo] * (1 - fraction) + log_energy[hi] * fraction
    z = sampled - sampled.mean()
    z = z / z.std().clamp(min=1e-8)
    centered = (1.5 * z).tanh()
    centered = centered - centered.mean()
    return (reference_gain * (1 - 0.15 * spread * centered)).clamp(min=1).float()


class OfficialSEGA:
    """Update once from ``x_embedder(x_t)`` before a Flux attention block runs."""

    def __init__(self, grid: Sequence[int], axes_dims: Sequence[int] = (16, 56, 56),
                 theta: float = 10000.0) -> None:
        self.rope = FluxRoPE(axes_dims=axes_dims, theta=theta, method="sega_official", grid=grid)
        self.grid = self.rope.grid
        self.axes_dims = self.rope.axes_dims
        self.scale = max(self.rope.scales)
        self.reference_gain = max(self.scale, 1.0) ** 0.08
        dim = self.axes_dims[1]
        self.ntk_factor = self.scale ** (2 * dim / (dim - 2)) / (1 + 0.1 * math.log(self.scale))
        self.diagnostics: dict = {}

    @torch.no_grad()
    def update(self, projected_features: torch.Tensor, sigma: float) -> torch.Tensor:
        """Return a full q/k multiplier; axis 0 remains exactly one."""
        height, width = self.grid
        if projected_features.ndim != 3 or projected_features.shape[1] != height * width:
            raise ValueError("projected_features must have shape [batch, height*width, channels]")
        gains = torch.ones(sum(self.axes_dims), device=projected_features.device, dtype=torch.float32)
        spread = 0.0
        if self.ntk_factor > 1:
            spatial = projected_features.reshape(projected_features.shape[0], height, width, -1)
            spatial_map = spatial.float().mean(dim=(0, -1))
            spatial_map = spatial_map - spatial_map.mean()
            profiles = axis_profiles(spatial_map)
            spread = spectral_spread(radial_profile(spatial_map))
            offset = self.axes_dims[0]
            freq_dtype = torch.float32 if gains.device.type in ("mps", "npu") else torch.float64
            for axis, (dim, profile) in enumerate(zip(self.axes_dims[1:], profiles), start=1):
                frequencies = self.rope.frequencies(axis, device=gains.device, dtype=freq_dtype)
                if spread > 0 and self.reference_gain > 1 + 1e-8:
                    pair_gains = allocation(profile, frequencies, self.reference_gain, spread)
                else:
                    pair_gains = torch.full_like(frequencies, self.reference_gain, dtype=torch.float32)
                gains[offset:offset + dim] = pair_gains.repeat_interleave(2)
                offset += dim
        self.diagnostics = {
            "implementation": "sega_official_ddaab08552c7",
            "input": "x_embedder(x_t)",
            "sigma": float(sigma),
            "spectral_spread": spread,
            "reference_qk_gain": self.reference_gain,
            "ntk_factor": self.ntk_factor,
            "spatial_scale": self.scale,
            "alpha": 0.15,
            "beta": 1.5,
            "minimum_qk_gain": 1.0,
        }
        return gains
