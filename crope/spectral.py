"""Channel-summed spectral densities in cycles per token."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch


POWER_NORMALIZATION = "channel_sum_mean_ortho_fft2_power"


def token_grid(packed, grid):
    """Convert FLUX's packed B,N,64 tensor into C,H,W without unpacking pixels."""
    if packed.ndim != 3 or packed.shape[0] != 1 or packed.shape[1] != math.prod(grid):
        raise ValueError("Expected batch-one packed latents matching the token grid")
    return packed[0].reshape(*grid, packed.shape[-1]).permute(2, 0, 1)


@dataclass
class AxisPower:
    lower: torch.Tensor
    upper: torch.Tensor
    frequency: torch.Tensor
    power: torch.Tensor
    noise: torch.Tensor
    count: torch.Tensor


def axis_power(x):
    if x.ndim != 3:
        raise ValueError("Expected C,H,W token-grid latents")
    x = x.to(torch.float64 if x.dtype == torch.float64 else torch.float32)
    c, h, w = x.shape
    p = torch.fft.fft2(x - x.mean((-2, -1), keepdim=True), norm="ortho").abs().square().sum(0)
    result = []
    for length, marginal in ((h, p.mean(1)), (w, p.mean(0))):
        # Common powers of sqrt(2) make native/target bins share absolute frequencies.
        low_exp = -math.ceil(math.log2(length))
        edges = torch.pow(2.0, torch.arange(low_exp, -0.999, 0.5, device=x.device, dtype=x.dtype))
        freq = torch.fft.fftfreq(length, device=x.device, dtype=x.dtype).abs()
        masks = (freq[None] >= edges[:-1, None]) & (freq[None] < edges[1:, None])
        masks[-1] |= freq == 0.5
        valid = masks.any(1)
        masks = masks[valid]
        lower, upper = edges[:-1][valid], edges[1:][valid]
        count = masks.sum(1).to(x.dtype)
        result.append(AxisPower(lower, upper, (lower * upper).sqrt(),
                                (masks.to(x.dtype) @ marginal) / count,
                                torch.full_like(count, c), count))
    return result


def log_interpolate(x, xp, yp, *, extrapolate=True):
    """Linear interpolation/extrapolation of a log-power law in log frequency."""
    xp, yp = xp.to(x), yp.to(x)
    if not extrapolate:
        if xp.numel() == 1:
            return torch.full_like(x, yp[0])
        x = x.clamp(xp[0], xp[-1])
    if xp.numel() < 2:
        raise ValueError("At least two native bins are required")
    a, b = x.log(), xp.log()
    index = torch.searchsorted(b.contiguous(), a.contiguous()).clamp(1, len(b) - 1)
    fraction = (a - b[index - 1]) / (b[index] - b[index - 1])
    return yp[index - 1] + fraction * (yp[index] - yp[index - 1])


@dataclass
class NativeProfile:
    frequencies: tuple
    log_power: tuple

    def __post_init__(self):
        if len(self.frequencies) != 2 or len(self.log_power) != 2:
            raise ValueError("A native profile must contain height and width axes")
        for f, p in zip(self.frequencies, self.log_power):
            if f.ndim != 1 or p.shape != f.shape or len(f) < 2:
                raise ValueError("Profile frequencies and log powers must be equal length vectors")
            if not torch.isfinite(f).all() or not torch.isfinite(p).all() or not (f > 0).all():
                raise ValueError("Profile frequencies and log powers must be finite with positive frequencies")
            if not (f[1:] > f[:-1]).all():
                raise ValueError("Profile frequencies must increase strictly")

    @classmethod
    def from_latents(cls, x):
        axes = axis_power(x)
        return cls(tuple(a.frequency.cpu() for a in axes),
                   tuple(a.power.clamp_min(1e-30).log().cpu() for a in axes))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data.get("power_normalization", "")) != POWER_NORMALIZATION:
                raise ValueError("Native profile uses incompatible power normalization; recalibrate it")
            return cls(tuple(torch.from_numpy(data[f"frequency_{a}"].copy()) for a in ("h", "w")),
                       tuple(torch.from_numpy(data[f"log_power_{a}"].copy()) for a in ("h", "w")))

    def evaluate(self, axis, frequencies, target="profile"):
        xp, yp = self.frequencies[axis].to(frequencies), self.log_power[axis].to(frequencies)
        if target == "slope":
            dx, dy = xp.log() - xp.log().mean(), yp - yp.mean()
            slope = (dx * dy).sum() / dx.square().sum()
            return yp.mean() + slope * (frequencies.log() - xp.log().mean())
        if target != "profile":
            raise ValueError(f"Unknown target {target}")
        return log_interpolate(frequencies, xp, yp)

    def save(self, path, **extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import os
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        with temporary.open("wb") as handle:
            np.savez(handle, **{f"frequency_{a}": self.frequencies[i].cpu().numpy() for i, a in enumerate(("h", "w"))},
                     **{f"log_power_{a}": self.log_power[i].cpu().numpy() for i, a in enumerate(("h", "w"))},
                     power_normalization=POWER_NORMALIZATION,
                     **extra)
        os.replace(temporary, path)
