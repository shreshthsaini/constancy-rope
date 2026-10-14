"""Statistics treat prompts, including all their seeds and crops, as clusters."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np


def confidence(values: np.ndarray, estimate: float, n: int) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    low, high = np.quantile(finite, [0.025, 0.975]) if finite.size and n > 1 else (None, None)
    return {"value": float(estimate), "ci_low": None if low is None else float(low),
            "ci_high": None if high is None else float(high), "n_prompts": int(n),
            "bootstrap_valid": int(finite.size)}


def bootstrap_mean(values, *, draws: int = 1000, seed: int = 0) -> dict:
    """Percentile interval of an equally weighted vector of prompt means."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"value": None, "ci_low": None, "ci_high": None, "n_prompts": 0,
                "bootstrap_valid": 0}
    rng = np.random.default_rng(seed)
    samples = [np.mean(values[rng.integers(len(values), size=len(values))])
               for _ in range(draws)]
    return confidence(samples, values.mean(), len(values))


def scalar_summaries(rows: list[dict], *, draws: int = 1000, seed: int = 0) -> list[dict]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric, value in row["metrics"].items():
            if value is not None and np.isfinite(value):
                grouped[(row["method"], row["res"], metric)][row["prompt_id"]].append(value)
    return [{"method": method, "res": res, "metric": metric,
             **bootstrap_mean([np.mean(v) for _, v in sorted(prompts.items())],
                              draws=draws, seed=seed)}
            for (method, res, metric), prompts in sorted(grouped.items())]


def paired_scalar_summaries(rows: list[dict], reference_method: str = "sega", *,
                            draws: int = 1000, seed: int = 0) -> list[dict]:
    """Match resolution, prompt ID and seed before aggregating prompt deltas."""
    reference = {(r["res"], r["prompt_id"], str(r["seed"])): r for r in rows
                 if r["method"] == reference_method}
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["method"] == reference_method:
            continue
        ref = reference.get((row["res"], row["prompt_id"], str(row["seed"])))
        if ref is None:
            continue
        for metric in row["metrics"].keys() & ref["metrics"].keys():
            value, baseline = row["metrics"][metric], ref["metrics"][metric]
            if value is not None and baseline is not None and np.isfinite(value + baseline):
                grouped[(row["method"], row["res"], metric)][row["prompt_id"]].append(value - baseline)
    return [{"method": method, "res": res, "reference_method": reference_method,
             "metric": metric, "delta_definition": "method_minus_reference",
             **bootstrap_mean([np.mean(v) for _, v in sorted(prompts.items())],
                              draws=draws, seed=seed)}
            for (method, res, metric), prompts in sorted(grouped.items())]


def _features(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Features must be two matrices with the same feature dimension")
    if len(x) < 2 or len(y) < 2:
        raise ValueError("FID and unbiased KID require at least two images per set")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite Inception features")
    return x, y


def fid(x, y) -> float:
    """Exact empirical FID, using a low-rank covariance identity for small N.

    The nuclear norm of X_centered @ Y_centered.T equals the covariance
    square-root trace. This avoids a 2048-square eigensolve when N is small.
    """
    x, y = _features(x, y)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = (x - mx) / np.sqrt(len(x) - 1), (y - my) / np.sqrt(len(y) - 1)
    if max(len(x), len(y)) <= x.shape[1]:
        cross_trace = np.linalg.svd(xc @ yc.T, compute_uv=False).sum()
    else:
        cx, cy = xc.T @ xc, yc.T @ yc
        eig, vectors = np.linalg.eigh(cx)
        root = (vectors * np.sqrt(np.maximum(eig, 0))) @ vectors.T
        product = root @ cy @ root
        cross_trace = np.sqrt(np.maximum(np.linalg.eigvalsh((product + product.T) / 2), 0)).sum()
    value = np.sum((mx - my) ** 2) + np.sum(xc * xc) + np.sum(yc * yc) - 2 * cross_trace
    return float(max(value, 0.0))


def kid(x, y) -> float:
    """Unbiased degree-three polynomial MMD on all available feature rows."""
    x, y = _features(x, y)
    dim = x.shape[1]
    xx, yy, xy = (x @ x.T / dim + 1) ** 3, (y @ y.T / dim + 1) ** 3, (x @ y.T / dim + 1) ** 3
    return float((xx.sum() - np.trace(xx)) / (len(x) * (len(x) - 1))
                 + (yy.sum() - np.trace(yy)) / (len(y) * (len(y) - 1)) - 2 * xy.mean())


def distribution_bootstrap(generated: dict[str, np.ndarray], real: dict[str, np.ndarray],
                           statistic: Callable, *, draws: int = 1000, seed: int = 0,
                           baseline: dict[str, np.ndarray] | None = None) -> dict:
    """Resample prompt clusters, with the same reference draw for paired deltas.

    A paired caller must first align seeds within each prompt. The baseline
    and generated sets then use precisely the same sampled prompt indices.
    Real-image prompt clusters are sampled independently of generated prompts.
    """
    keys = sorted(generated.keys() if baseline is None else generated.keys() & baseline.keys())
    real_keys = sorted(real)
    if len(keys) < 2 or len(real_keys) < 2:
        raise ValueError("Distribution intervals require at least two prompt clusters in each set")
    combine = lambda bank, selected: np.concatenate([bank[k] for k in selected], axis=0)
    x, y = combine(generated, keys), combine(real, real_keys)
    estimate = statistic(x, y)
    if baseline is not None:
        estimate -= statistic(combine(baseline, keys), y)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = [keys[i] for i in rng.integers(len(keys), size=len(keys))]
        selected_real = [real_keys[i] for i in rng.integers(len(real_keys), size=len(real_keys))]
        yb = combine(real, selected_real)
        value = statistic(combine(generated, selected), yb)
        if baseline is not None:
            value -= statistic(combine(baseline, selected), yb)
        samples.append(value)
    return {**confidence(samples, estimate, len(keys)), "n_reference_prompts": len(real_keys),
            "n_feature_rows": len(x), "n_reference_feature_rows": len(y)}
