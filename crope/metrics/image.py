"""Deterministic crops and luminance diagnostics without model dependencies."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image


def stable_seed(identity: str, seed: int = 0) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{identity}".encode()).digest()[:8], "little")


def crop_boxes(size: tuple[int, int], identity: str, *, seed: int = 0,
               count: int = 4, crop_size: int = 1024) -> list[tuple[int, int, int, int]]:
    """Use identical crop coordinates across methods with the same image size."""
    width, height = size
    if min(size) < crop_size:
        raise ValueError(f"A {crop_size}px crop needs both dimensions >= {crop_size}, got {size}")
    if count < 1:
        raise ValueError("Crop count must be positive")
    rng = np.random.default_rng(stable_seed(identity, seed))
    boxes = []
    for _ in range(count):
        left, top = int(rng.integers(width - crop_size + 1)), int(rng.integers(height - crop_size + 1))
        boxes.append((left, top, left + crop_size, top + crop_size))
    return boxes


def linear_luminance(image: Image.Image | np.ndarray) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    rgb = rgb.astype(np.float32) / (255.0 if rgb.dtype == np.uint8 else 1.0)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected an RGB image")
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def octave_profile(luminance: np.ndarray, edges: np.ndarray | None = None) -> dict:
    """Log mean 2D spectral density in octave bands, in cycles per pixel.

    Frequencies below 1/4096 and above 1/2 are excluded. Mean subtraction
    removes DC; the FFT is unwindowed. Power is |FFT|^2/(H*W), so the radial
    density has the same normalization at different image dimensions.
    """
    y = np.asarray(luminance, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("Expected a two-dimensional luminance image")
    h, w = y.shape
    edges = 2.0 ** np.arange(-12, 0) if edges is None else np.asarray(edges, np.float64)
    power = np.abs(np.fft.rfft2(y - y.mean())) ** 2 / (h * w)
    frequency = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None, :])
    multiplicity = np.full(power.shape[1], 2.0)
    multiplicity[0] = 1.0
    if w % 2 == 0:
        multiplicity[-1] = 1.0
    weights = np.broadcast_to(multiplicity, power.shape)
    counts = np.histogram(frequency, bins=edges, weights=weights)[0]
    sums = np.histogram(frequency, bins=edges, weights=power * weights)[0]
    logs = np.full_like(sums, np.nan)
    np.log(np.maximum(np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0), 1e-20),
           out=logs, where=counts > 0)
    return {"frequency": np.sqrt(edges[:-1] * edges[1:]), "log_power": logs, "counts": counts}


def profile_error(profile: dict, reference_log_power: np.ndarray) -> dict:
    current, target = np.asarray(profile["log_power"]), np.asarray(reference_log_power)
    valid = np.isfinite(current) & np.isfinite(target)
    error = current - target
    result = {f"spectrum_log_error_b{i:02d}": float(value) if valid[i] else None
              for i, value in enumerate(error)}
    result["spectrum_log_rmse"] = float(np.sqrt(np.mean(error[valid] ** 2))) if valid.any() else None
    if valid.sum() >= 2:
        logf = np.log(np.asarray(profile["frequency"])[valid])
        slope = np.polyfit(logf, current[valid], 1)[0]
        target_slope = np.polyfit(logf, target[valid], 1)[0]
        result.update(spectrum_slope=float(slope), spectrum_slope_error=float(abs(slope - target_slope)))
    else:
        result.update(spectrum_slope=None, spectrum_slope_error=None)
    return result


def autocorrelation(luminance: np.ndarray, lag: int, axis: int) -> float | None:
    """Normalized, non-wrapped autocorrelation on the overlapping region."""
    if lag <= 0 or lag >= luminance.shape[axis]:
        return None
    first, second = [slice(None)] * 2, [slice(None)] * 2
    first[axis], second[axis] = slice(None, -lag), slice(lag, None)
    a, b = luminance[tuple(first)].astype(np.float64), luminance[tuple(second)].astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return None if denominator <= 1e-20 else float(np.sum(a * b) / denominator)


def tiling_metrics(luminance: np.ndarray, lags=(1024, 2048)) -> dict:
    result = {}
    for lag in lags:
        valid = []
        for axis in (0, 1):
            peak = autocorrelation(luminance, lag, axis)
            control = autocorrelation(luminance, round(1.5 * lag), axis)
            if peak is not None and control is not None:
                valid.append((peak, control))
        key = f"tiling{lag}"
        result[key] = float(np.mean([a - b for a, b in valid])) if valid else None
        result[key + "_acf"] = float(np.mean([a for a, _ in valid])) if valid else None
        result[key + "_control_acf"] = float(np.mean([b for _, b in valid])) if valid else None
        result[key + "_valid_axes"] = len(valid)
    return result
