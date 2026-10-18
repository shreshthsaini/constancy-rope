"""Helpers shared by f1_hero.py and f4_qualitative.py: verified image sources and the
method-agnostic crop rule.

Crop rule (fixed before any output was viewed): average the Rec. 709 luminance of every
method shown for a prompt at native 4096 px, take the 4-neighbour Laplacian of that mean,
and pick the square tile of the requested size with the largest mean squared Laplacian
(tiles on a stride of size/4, kept clear of the image border). Because the statistic is
computed on the across-method mean, it favours structure that all methods share and never
looks at any single method. The second location, where used, is the exact image center.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import crope_style as cs

RAW = {v: k for k, v in cs.RAW_TO_KEY.items()}  # figure key -> method name in the sidecar


def sources(idx, methods, table=None, check_hash=True):
    """method key -> Path of the seed-0 4096 px PNG for manifest row idx, with sidecar checks."""
    d = cs.heldout() if table is None else table
    out, ref = {}, None
    for m in methods:
        r = d[(d.idx == idx) & (d.method == m)]
        assert len(r) == 1, (idx, m, len(r))
        p = Path(r.iloc[0].path)
        side = json.loads(p.with_suffix(".json").read_text())
        cfg = side["config"]
        assert side["complete"] and cfg["seed"] == 0 and cfg["res"] == "4096x4096"
        assert cfg["prompt_id"] == r.iloc[0].prompt_id
        assert cfg["method"] == RAW.get(m, m), (cfg["method"], m)
        if ref is None:
            ref = cfg
        else:
            assert all(cfg[k] == ref[k] for k in ("prompt_id", "prompt", "seed", "derived_seed", "steps", "guidance"))
        if check_hash:
            assert hashlib.sha256(p.read_bytes()).hexdigest() == side["png_sha256"], p
        out[m] = p
    return out


def luma(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    assert a.shape == (4096, 4096, 3)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def mean_luma(paths):
    acc = None
    for p in paths:
        y = luma(p)
        acc = y if acc is None else acc + y
    return acc / len(paths)


def detail_box(mean_y, size, margin=64, exclude=None):
    """(x0, y0, x1, y1) of the size x size tile with maximum Laplacian energy of mean_y.
    exclude: list of boxes; candidate tiles overlapping any of them are skipped."""
    lap = np.zeros_like(mean_y)
    lap[1:-1, 1:-1] = (mean_y[:-2, 1:-1] + mean_y[2:, 1:-1] + mean_y[1:-1, :-2] + mean_y[1:-1, 2:]
                       - 4.0 * mean_y[1:-1, 1:-1])
    e = lap.astype(np.float64) ** 2
    ii = np.zeros((e.shape[0] + 1, e.shape[1] + 1))
    ii[1:, 1:] = e.cumsum(0).cumsum(1)
    stride = max(size // 4, 1)
    H, W = mean_y.shape
    best, box = -1.0, None
    for y0 in range(margin, H - margin - size + 1, stride):
        for x0 in range(margin, W - margin - size + 1, stride):
            b = (x0, y0, x0 + size, y0 + size)
            if exclude and any(not (b[2] <= q[0] or b[0] >= q[2] or b[3] <= q[1] or b[1] >= q[3]) for q in exclude):
                continue
            s = ii[y0 + size, x0 + size] - ii[y0, x0 + size] - ii[y0 + size, x0] + ii[y0, x0]
            if s > best:
                best, box = s, b
    return box


def center_box(size, full=4096):
    c = full // 2
    return (c - size // 2, c - size // 2, c + size // 2, c + size // 2)
