"""FLUX rotary embeddings, with independent scaling of the spatial axes.

The DyPE branch adapts the MIT-licensed upstream Flux implementation. Its
copyright and license are reproduced in THIRD_PARTY.md.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


METHODS = ("base", "pi", "ntk", "ntk2", "yarn", "dype", "sega_official")


def _ramp(low: float, high: float, count: int, device, dtype) -> torch.Tensor:
    # Upstream builds this ramp in float32 before converting to frequency dtype.
    if low == high:
        high += 0.001
    return ((torch.arange(count, device=device, dtype=torch.float32) - low) /
            (high - low)).clamp(0, 1).to(dtype)


def _dype_mask(dim: int, theta: float, native: int, rotations: tuple[float, float],
               device, dtype) -> torch.Tensor:
    boundaries = [dim * math.log(native / (r * 2 * math.pi)) /
                  (2 * math.log(theta)) for r in rotations]
    low = max(math.floor(boundaries[0]), 0)
    high = min(math.ceil(boundaries[1]), dim - 1, dim // 2)
    return 1 - _ramp(low, high, dim // 2, device, dtype)


class FluxRoPE(nn.Module):
    """Drop-in replacement for diffusers 0.39 ``FluxPosEmbed``.

    ``grid`` and ``native_grid`` are packed image-token grids, in height/width
    order. Axis 0 always retains its original frequencies. ``yarn`` uses the
    wavelength-ratio ramp in design.md; its global temperature belongs to the
    gain controller. ``dype`` preserves the upstream spatial amplitude inside
    the cosine/sine arrays, so it must use a controller with no extra temperature.
    """

    def __init__(self, axes_dims: Sequence[int] = (16, 56, 56),
                 theta: float = 10000.0, method: str = "base",
                 grid: Sequence[int] = (64, 64),
                 native_grid: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown RoPE method {method!r}; choose from {METHODS}")
        self.axes_dims = tuple(int(d) for d in axes_dims)
        if len(self.axes_dims) != 3 or any(d <= 0 or d % 2 for d in self.axes_dims):
            raise ValueError("axes_dims must contain three positive even dimensions")
        if not math.isfinite(theta) or theta <= 1:
            raise ValueError("theta must be finite and greater than 1")
        self.axes_dim = list(self.axes_dims)
        self.theta = float(theta)
        self.method = method
        self.grid = self._validate_grid(grid)
        self.native_grid = self._validate_grid(native_grid)
        if method in ("ntk", "ntk2", "dype", "sega_official") and min(self.axes_dims[1:]) <= 2:
            raise ValueError("NTK scaling requires spatial dimensions greater than 2")
        self.current_timestep = 1.0

    @staticmethod
    def _validate_grid(grid: Sequence[int]) -> tuple[int, int]:
        if len(grid) != 2 or any(int(n) != n or n <= 0 for n in grid):
            raise ValueError("grid must contain two positive integer token counts")
        return tuple(int(n) for n in grid)

    @property
    def scales(self) -> tuple[float, float]:
        return tuple(target / native for target, native in zip(self.grid, self.native_grid))

    def set_timestep(self, sigma: float | torch.Tensor) -> None:
        """Set the current flow sigma, with 1 denoting pure noise."""
        sigma = float(sigma.detach().item()) if isinstance(sigma, torch.Tensor) else float(sigma)
        if not math.isfinite(sigma) or not 0 <= sigma <= 1:
            raise ValueError("sigma must be finite and in [0, 1]")
        self.current_timestep = sigma

    def frequencies(self, axis: int, *, device=None,
                    dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Return one angular frequency per rotary pair for axis 0, 1, or 2."""
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        dim = self.axes_dims[axis]
        exponent = torch.arange(0, dim, 2, device=device, dtype=dtype) / dim
        base = 1.0 / (self.theta ** exponent)
        if axis == 0 or self.method == "base":
            return base
        scale = self.scales[axis - 1]
        if scale == 1 and self.method != "sega_official":
            return base
        if self.method == "pi":
            return base / scale
        if self.method in ("ntk", "ntk2"):
            strength = 1 if self.method == "ntk" else 2
            new_theta = self.theta * scale ** (strength * dim / (dim - 2))
            return 1.0 / (new_theta ** exponent)
        if self.method == "sega_official":
            # Upstream uses the longest side and the height-axis rotary dimension.
            max_scale = max(self.scales)
            spatial_dim = self.axes_dims[1]
            factor = max_scale ** (2 * spatial_dim / (spatial_dim - 2))
            factor /= 1 + 0.1 * math.log(max_scale)
            return 1.0 / ((self.theta * max(factor, 1.0)) ** exponent)
        if self.method == "yarn":
            # The specification defines r = wavelength/native_length explicitly.
            ratio = (2 * math.pi / base) / self.native_grid[axis - 1]
            interpolate = ((ratio - 1.0) / (32.0 - 1.0)).clamp(0, 1)
            return base * (1 - interpolate) + (base / scale) * interpolate
        if scale <= 1:
            return base
        if self.current_timestep == 0:
            # Upstream divides by zero at sigma=0. The default scheduler never
            # evaluates this endpoint; define it with base frequencies.
            return base
        native = self.native_grid[axis - 1]
        scale_t = torch.tensor(scale, device=device, dtype=dtype)
        linear = 1.0 / (scale_t * (self.theta ** exponent))
        new_theta = self.theta * (scale_t ** (dim / (dim - 2)))
        ntk = 1.0 / torch.pow(new_theta, exponent)
        kappa = self.current_timestep ** 2.0
        mask = _dype_mask(dim, self.theta, native, (1.25 * kappa, 0.75 * kappa), device, dtype)
        mixed = linear * (1 - mask) + ntk * mask
        mask = _dype_mask(dim, self.theta, native, (16 * kappa, 2 * kappa), device, dtype)
        return mixed * (1 - mask) + base * mask

    def embedding_gain(self, *, device=None,
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return q/k amplitudes already encoded in ``forward`` cosines/sines."""
        gains = torch.ones(sum(self.axes_dims), device=device, dtype=dtype)
        if self.method == "dype":
            offset = self.axes_dims[0]
            for dim, scale in zip(self.axes_dims[1:], self.scales):
                if scale > 1:
                    # Match upstream float64 evaluation followed by float32 gain.
                    scale_t = torch.tensor(scale, dtype=torch.float64)
                    gains[offset:offset + dim] = (1 + 0.1 * torch.log(scale_t)).item()
                offset += dim
        return gains

    def metadata(self) -> dict:
        """JSON-ready conventions for the generation record."""
        return {
            "method": self.method,
            "axes_dims": list(self.axes_dims),
            "theta": self.theta,
            "grid_tokens": list(self.grid),
            "native_grid_tokens": list(self.native_grid),
            "spatial_scales": list(self.scales),
            "yarn_ramp": "r=wavelength/native_axis_length; alpha=1; beta=32",
            "yarn_temperature": "controller logit gain; per-side geometric-mean scale",
            "dype_schedule": "kappa=sigma**2; beta=(1.25,.75); gamma=(16,2)",
            "dype_amplitude": "spatial q/k scale=1+0.1*ln(axis_scale); axis0=1",
            "embedded_qk_gain": self.embedding_gain().tolist(),
        }

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.ndim != 2 or ids.shape[1] != 3:
            raise ValueError("FLUX position ids must have shape [tokens, 3]")
        pos = ids.float()
        freq_dtype = torch.float32 if ids.device.type in ("mps", "npu") else torch.float64
        amplitudes = self.embedding_gain(device=ids.device)
        cosine, sine = [], []
        offset = 0
        for axis, dim in enumerate(self.axes_dims):
            frequencies = self.frequencies(axis, device=ids.device, dtype=freq_dtype)
            phase = torch.outer(pos[:, axis], frequencies)
            if ids.device.type == "npu":
                phase = phase.float()
            cos = phase.cos().repeat_interleave(2, dim=1, output_size=dim).float()
            sin = phase.sin().repeat_interleave(2, dim=1, output_size=dim).float()
            if self.method == "dype" and axis > 0 and self.scales[axis - 1] > 1:
                cos = cos * amplitudes[offset:offset + dim]
                sin = sin * amplitudes[offset:offset + dim]
            cosine.append(cos)
            sine.append(sin)
            offset += dim
        return torch.cat(cosine, dim=-1), torch.cat(sine, dim=-1)
