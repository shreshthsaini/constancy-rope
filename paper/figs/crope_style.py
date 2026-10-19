"""Shared design system for the paper figures: method names and colors, metric labels,
data access, the paired bootstrap, image helpers, and JPEG embedding for photo panels.

Every figure script imports this module so names, colors and typography stay fixed
across figures (see DESIGN_NOTES.md). Sizes, the audit and save() come from the
vendored orx_figstyle module.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

import orx_figstyle
from orx_figstyle import COLUMN, TEXT, WIDE, save, use_style  # noqa: F401 (re-exported)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOTES = ROOT / ".agent" / "notes"
SCR = Path("/scratch/09032/saini_2/constancy_rope")
CACHE = SCR / "cache/figures"
PROMPTS = NOTES / "main2" / "prompts.jsonl"
HELDOUT = NOTES / "main2" / "heldout_per_prompt.csv"          # seed 0, rows 8-194, all roots
PAIRED = SCR / "eval_krea" / "heldout" / "heldout_paired_vs_sega_official.csv"
PAIRED_V2 = NOTES / "main2" / "heldout_paired_vs_sega_official.csv"  # same rows + sega_ctm_v2

# ==== colors
INK = "#1F2329"        # text and strong strokes
INK2 = "#5B6270"       # secondary text
RULE = "#C9CDD3"       # hairlines, zero lines
PANEL = "#F3F4F6"      # light panel fill
GRIDC = "#E3E5E8"

COBALT = "#1F4E8C"     # released SEGA (the reference everywhere)
SKY = "#7EA9D9"        # SEGA base + global temperature, no allocation
STONE = "#9A9A9A"      # NTK base (baseline grey)
TEAL = "#2E9C85"       # DyPE
SAFFRON = "#E1A21B"    # Constancy (NTK base)
TERRA = "#D35F3A"      # Constancy-TM (SEGA base + temperature)
PLUM = "#A2467F"       # SEGA + TM correction
SIENNA = "#8A5A33"     # alias-count temperature (dev only)
OXBLOOD = "#9E2A2B"    # SEGA as written in the paper equations

# key -> label, short label, color, marker, group
METHODS = {
    "ntk2":               dict(label="NTK", short="NTK", color=STONE, marker="o", group="baseline"),
    "dype":               dict(label="DyPE", short="DyPE", color=TEAL, marker="o", group="baseline"),
    "sega_official":      dict(label="SEGA (released)", short="SEGA", color=COBALT, marker="o", group="sega"),
    "temp_only_sbase":    dict(label="SEGA base + temp.", short="Base + temp.", color=SKY, marker="o", group="sega"),
    "constancy":          dict(label="Constancy", short="Constancy", color=SAFFRON, marker="D", group="ours"),
    "constancy_tm_sbase": dict(label="Constancy-TM", short="Constancy-TM", color=TERRA, marker="D", group="ours"),
    "sega_ctm_v2":        dict(label="SEGA + TM", short="SEGA + TM", color=PLUM, marker="D", group="ours"),
    "alias_temp":         dict(label="Alias-T", short="Alias-T", color=SIENNA, marker="D", group="ours"),
    "sega":               dict(label="SEGA (paper eqs.)", short="SEGA eqs.", color=OXBLOOD, marker="s", group="sega"),
}
# aliases used by raw roots (mainsb rows have no suffix inside their own per_image.csv)
RAW_TO_KEY = {"temp_only": "temp_only_sbase", "constancy_tm": "constancy_tm_sbase", "sega_ctm": "sega_ctm_v2"}


# Darker tints for TEXT set in a method color (headers, direct labels); light hues fail contrast on white.
TEXT_TINT = {SKY: "#3F73B0", SAFFRON: "#A8770C", STONE: "#767676"}
for _m in METHODS.values():
    _m["text"] = TEXT_TINT.get(_m["color"], _m["color"])


def color(key):
    return METHODS[key]["color"]


def text_color(key):
    """Color for text naming a method; markers and lines keep color(key)."""
    return METHODS[key]["text"]


def label(key, short=False):
    return METHODS[key]["short" if short else "label"]


# ==== metrics
METRICS = {
    "imagereward":          dict(label="ImageReward", group="Reward and alignment", lower=False),
    "pickscore":            dict(label="PickScore", group="Reward and alignment", lower=False),
    "clip":                 dict(label="CLIP score", group="Reward and alignment", lower=False),
    "musiq":                dict(label="MUSIQ, full frame", group="Image quality", lower=False),
    "musiq_crop1024":       dict(label="MUSIQ, 1024 crops", group="Image quality", lower=False),
    "clipiqa":              dict(label="CLIP-IQA, full frame", group="Image quality", lower=False),
    "clipiqa_crop1024":     dict(label="CLIP-IQA, 1024 crops", group="Image quality", lower=False),
    "spectrum_log_rmse":    dict(label="Spectral log-RMSE", group="Spectrum and structure", lower=True),
    "spectrum_slope_error": dict(label="Spectral slope error", group="Spectrum and structure", lower=True),
    "tiling1024":           dict(label="Tiling index, 1024 px", group="Spectrum and structure", lower=True),
}

# Display precision of estimates, matching tables/make_tables.py. Axis ticks use
# the precision needed by their step; estimates retain a fixed precision per metric.
METRIC_PRECISION = {
    "imagereward": 3, "pickscore": 2, "clip": 2, "musiq": 2, "musiq_crop1024": 2,
    "clipiqa": 3, "clipiqa_crop1024": 3, "spectrum_log_rmse": 3,
    "spectrum_slope_error": 3, "tiling1024": 4,
}


def style():
    """House rcParams on top of orx_figstyle.use_style()."""
    use_style()
    mpl.rcParams.update({
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "grid.color": GRIDC, "image.composite_image": False, "image.interpolation": "none",
        "legend.fontsize": 7, "axes.titlesize": 8,
    })


def panel_tag(ax_or_fig, x, y, letter, transform=None, **kw):
    """Bold lower-case panel letter (a, b, ...) in the house style."""
    tgt = ax_or_fig
    tr = transform if transform is not None else (tgt.transAxes if hasattr(tgt, "transAxes") else tgt.transFigure)
    return tgt.text(x, y, letter, transform=tr, fontsize=9, fontweight="bold", color=INK,
                    ha="left", va="baseline", **kw)


# ==== data
def prompt_order():
    return {json.loads(line)["id"]: i for i, line in enumerate(open(PROMPTS))}


def prompts():
    return [json.loads(line) for line in open(PROMPTS)]


def heldout():
    """Seed-0 held-out per-prompt table (rows 8-194) with method keys as in METHODS."""
    return pd.read_csv(HELDOUT)


def boot(x, n=2000, seed=0):
    """Mean and percentile 95% CI; identical to scripts/heldout_analysis.py."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(df, method, metric, ref="sega_official"):
    a = df[df.method == method].set_index("prompt_id")[metric]
    b = df[df.method == ref].set_index("prompt_id")[metric]
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).dropna()
    return (*boot(diff), len(diff))


def seed1():
    """Seed-1 replication table (rows 8-62, n=55) with ImageReward merged, keys as in METHODS."""
    d = pd.read_csv(SCR / "eval_krea" / "seed1" / "per_image.csv")
    ir = pd.read_csv(SCR / "eval_krea" / "ir" / "seed1.csv")
    keys = ["method", "prompt_id", "res"]
    d = d.merge(ir[keys + ["imagereward", "png"]], on=keys, how="left", validate="one_to_one")
    same = d.path.map(lambda p: str(Path(p).resolve())) == d.png.map(lambda p: str(Path(p).resolve()))
    assert same.all(), "seed-1 evaluation and ImageReward scored different files"
    d["idx"] = d.prompt_id.map(prompt_order())
    d = d[(d.idx >= 8) & (d.idx < 195)].copy()
    assert (d.seed == 1).all()
    d["method"] = d.method.map(lambda m: RAW_TO_KEY.get(m, m))
    return d


# ==== images
def _cached(tag, path, fn):
    from PIL import Image
    CACHE.mkdir(parents=True, exist_ok=True)
    st = os.stat(path)
    key = hashlib.sha1(f"{path}|{st.st_size}|{st.st_mtime_ns}|{tag}".encode()).hexdigest()[:20]
    out = CACHE / f"{key}.png"
    if not out.exists():
        with Image.open(path) as im:
            fn(im.convert("RGB")).save(out, compress_level=1)
    with Image.open(out) as im:
        return np.asarray(im.convert("RGB"))


def thumb(path, px):
    """Full frame reduced to px x px with Lanczos (no other processing)."""
    from PIL import Image
    return _cached(f"thumb{px}", path, lambda im: im.resize((px, px), Image.Resampling.LANCZOS))


def crop(path, box):
    """Native-resolution crop, box = (x0, y0, x1, y1) in source pixels; pixels untouched."""
    return _cached(f"crop{tuple(box)}", path, lambda im: im.crop(tuple(box)))


def gray_full(path, px=1024):
    a = thumb(path, px).astype(np.float32) / 255.0
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


# ==== PDF post-processing
_PDFPY = "/home1/09032/saini_2/.local/share/uv/python/cpython-3.11.13-linux-aarch64-gnu/bin/python3.11"
_PDFLIB = "/work/09032/saini_2/vista/library/uv-envs/pdfrender/lib/python3.11/site-packages"
_JPEG_SNIPPET = r"""
import io, os, sys, fitz
from PIL import Image, ImageFile
ImageFile.MAXBLOCK = 1 << 26  # optimize=True needs the whole scan in one buffer
src, q, minpx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
doc = fitz.open(src); n = 0
for page in doc:
    for info in page.get_images(full=True):
        xref, smask, w, h = info[0], info[1], info[2], info[3]
        if smask or w * h < minpx:
            continue
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha != 3:
            continue
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples if not pix.alpha else fitz.Pixmap(pix, 0).samples)
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=q, subsampling=0, optimize=True)
        page.replace_image(xref, stream=buf.getvalue()); n += 1
tmp = src + ".tmp.pdf"; doc.save(tmp, garbage=4, deflate=True); doc.close(); os.replace(tmp, src)
print(f"jpeg-embedded {n} photo image(s) in {os.path.basename(src)}: {os.path.getsize(src)/1e6:.2f} MB")
"""


def jpegify(pdf_path, quality=94, min_pixels=20000):
    """Re-encode the photographic images inside a matplotlib PDF as JPEG (4:4:4).
    Vector text and plots are untouched; image placement is unchanged."""
    env = dict(os.environ, PYTHONPATH=_PDFLIB, OMP_NUM_THREADS="8")
    r = subprocess.run([_PDFPY, "-s", "-c", _JPEG_SNIPPET, str(pdf_path), str(quality), str(min_pixels)],
                       env=env, capture_output=True, text=True)
    sys.stderr.write(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError("jpegify failed")


def finish(fig, stem, photo=False, formats=None):
    """save() with the audit, then JPEG-embed photo panels. Photo figures skip SVG (base64 PNG bloat)."""
    stem = str(stem)
    formats = formats or (("pdf",) if photo else ("pdf", "svg"))
    save(fig, stem, formats=formats)
    if "svg" in formats:
        svg = Path(stem + ".svg")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    if photo:
        jpegify(stem + ".pdf")
