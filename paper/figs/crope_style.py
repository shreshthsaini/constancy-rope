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

# ==== paper palette (figv3). PALETTE.md mirrors these hex values for the tables.
# One method hue (SEGA released), two component hues, one neutral gray, black. At most four
# hues in any one figure; the paper-equations pink appears only in the appendix comparison.
INK = "#1A1A1A"        # black: all text, strong strokes, the NTK baseline
INK2 = "#595959"       # secondary text (tick labels may use INK)
GRAY = "#8C8C8C"       # neutral gray: DyPE, secondary marks
RULE = "#BFBFBF"       # hairlines, reference lines
GRIDC = "#E6E6E6"      # light gray gridlines
PANEL = "#F2F2F2"      # gray band (regions of interest, prompt bands, panel fill)

METHOD = "#0068A8"     # SEGA (released): the method hue
TEMP = "#009870"       # component: SEGA base + global temperature (no allocation)
LAW = "#D06008"        # component: per-band laws (Constancy, Constancy-TM, SEGA + TM, Alias-T)
PAPER = "#C878A0"      # SEGA as written in the paper equations (appendix only)

ZOOM = "#FF0000"       # zoom boxes on photos (1 pt) and the frame of the enlarged crop
ZOOM2 = "#00B0F0"      # second zoom box when one frame carries two


def tint(hex_color, t):
    """Mix a hex color with white: t = 1 keeps the color, t = 0.2 is a 20 percent tint."""
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], float)
    m = np.round(255.0 * (1 - t) + c * t).astype(int)
    return "#" + "".join(f"{v:02X}" for v in m)


FILL = {k: tint(v, 0.20) for k, v in dict(METHOD=METHOD, TEMP=TEMP, LAW=LAW, GRAY=GRAY, PAPER=PAPER).items()}
SHADE = {k: tint(v, 0.10) for k, v in dict(METHOD=METHOD, TEMP=TEMP, LAW=LAW, GRAY=GRAY, PAPER=PAPER).items()}

# Legacy names kept so older scripts still run; they now resolve to the paper palette.
COBALT, SKY, STONE, TEAL = METHOD, TEMP, INK, GRAY
SAFFRON = TERRA = PLUM = SIENNA = LAW
OXBLOOD = PAPER

# key -> label, short label, color, marker, line style, group
METHODS = {
    "ntk2":               dict(label="NTK", short="NTK", color=INK, marker="o", ls="--", group="baseline"),
    "dype":               dict(label="DyPE", short="DyPE", color=GRAY, marker="s", ls="--", group="baseline"),
    "sega_official":      dict(label="SEGA (released)", short="SEGA", color=METHOD, marker="o", ls="-", group="sega"),
    "temp_only_sbase":    dict(label="SEGA base + temp.", short="Base + temp.", color=TEMP, marker="^", ls="-", group="sega"),
    "constancy":          dict(label="Constancy", short="Constancy", color=LAW, marker="o", ls=":", group="ours"),
    "constancy_tm_sbase": dict(label="Constancy-TM", short="Constancy-TM", color=LAW, marker="D", ls="-", group="ours"),
    "sega_ctm_v2":        dict(label="SEGA + TM", short="SEGA + TM", color=LAW, marker="v", ls="-.", group="ours"),
    "alias_temp":         dict(label="Alias-T", short="Alias-T", color=LAW, marker="s", ls=":", group="ours"),
    "sega":               dict(label="SEGA (paper eqs.)", short="SEGA eqs.", color=PAPER, marker="s", ls="--", group="sega"),
}
# aliases used by raw roots (mainsb rows have no suffix inside their own per_image.csv)
RAW_TO_KEY = {"temp_only": "temp_only_sbase", "constancy_tm": "constancy_tm_sbase", "sega_ctm": "sega_ctm_v2"}

# Text stays in INK; a method name is never set in its hue (labels are neutral, marks carry color).
for _m in METHODS.values():
    _m["text"] = INK

# Type sizes at final print size (points).
FS_TICK, FS_LABEL, FS_PANEL, FS_VALUE = 7, 8, 9, 7
FS_MIN = 6.5     # smallest glyph in any figure at print size, mathtext sub- and superscripts included


def script_floor(base_pt=FS_LABEL, floor_pt=FS_MIN):
    """Print first-level mathtext sub- and superscripts of `base_pt` text at `floor_pt` or larger.

    Mathtext sets scripts at 0.7 of the base size, so an 8 pt label such as $\\ln P_{\\mathrm{SEGA}}$ prints a
    5.6 pt subscript, and matplotlib has no public switch for that ratio. This raises its module constant for
    the whole process: call it once, right after style() and before anything is drawn (layouts are cached)."""
    from matplotlib import _mathtext
    assert isinstance(_mathtext.SHRINK_FACTOR, float), "matplotlib moved mathtext's SHRINK_FACTOR"
    _mathtext.SHRINK_FACTOR = max(_mathtext.SHRINK_FACTOR, floor_pt / base_pt)


def min_text_pt(fig):
    """Smallest glyph size (points, at the figure's own size) over all visible text. Unlike the orx_figstyle
    audit, which reads each Text's nominal size, this lays out mathtext, so scripts count at their real size."""
    from matplotlib.cbook import is_math_text
    from matplotlib.mathtext import MathTextParser
    parser = MathTextParser("path")
    sizes = []
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        if t.get_parse_math() and not t.get_usetex() and is_math_text(s):
            glyphs = parser.parse(s, dpi=72, prop=t.get_fontproperties()).glyphs
            sizes.append(min((g[1] for g in glyphs), default=t.get_fontsize()))
        else:
            sizes.append(t.get_fontsize())
    return min(sizes)


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


def panel_tag(ax_or_fig, x, y, letter, name=None, transform=None, **kw):
    """Panel tag "(a) Name" at 9 pt, upper left, as in the gold-standard papers.
    `letter` may be "a" or "(a)"; `name` is 1 to 3 words or None."""
    tgt = ax_or_fig
    tr = transform if transform is not None else (tgt.transAxes if hasattr(tgt, "transAxes") else tgt.transFigure)
    tag = letter if letter.startswith("(") else f"({letter})"
    if name:
        tag = f"{tag} {name}"
    kw.setdefault("fontsize", FS_PANEL)
    return tgt.text(x, y, tag, transform=tr, color=INK, ha="left", va="baseline", **kw)


def clean_axes(ax, grid="y", spines=("left", "bottom")):
    """Gold plot axes: no top/right spines, light gray gridlines behind the data, small ticks."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in spines)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(INK2)
    ax.tick_params(length=2.5, width=0.6, labelsize=FS_TICK, colors=INK2, labelcolor=INK)
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid in ("x", "y", "both"):
        ax.grid(True, axis=grid, color=GRIDC, lw=0.5)
    return ax


def top_legend(target, handles, labels=None, ncol=None, y=1.02, **kw):
    """One shared legend above a panel or a row of panels, no frame."""
    labels = labels if labels is not None else [h.get_label() for h in handles]
    kw = dict(dict(loc="lower center", bbox_to_anchor=(0.5, y), frameon=False, fontsize=FS_LABEL,
                   handlelength=1.6, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0), **kw)
    return target.legend(handles, labels, ncol=ncol or len(handles), **kw)


def crop_box(ax, xywh, color=ZOOM, lw=1.0):
    """Zoom box on a full frame: 1 pt square outline at (x, y, w, h) in data (pixel) coordinates."""
    from matplotlib.patches import Rectangle
    x, y, w, h = xywh
    return ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=color, lw=lw, joinstyle="miter", zorder=5))


def crop_frame(ax, color=ZOOM, lw=1.0):
    """Frame of the enlarged crop: a 1 pt outline drawn just inside the image edge."""
    from matplotlib.patches import Rectangle
    return ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=color,
                                  lw=lw, joinstyle="miter", clip_on=False, zorder=5))


METRIC_ARROW = {"imagereward": "↑", "pickscore": "↑", "clip": "↑", "musiq": "↑",
                "musiq_crop1024": "↑", "clipiqa": "↑", "clipiqa_crop1024": "↑",
                "spectrum_log_rmse": "↓", "spectrum_slope_error": "↓", "tiling1024": "↓"}
METRIC_SHORT = {"imagereward": "ImageReward", "pickscore": "PickScore", "clip": "CLIP",
                "musiq": "MUSIQ", "musiq_crop1024": "MUSIQ crops", "clipiqa": "CLIP-IQA",
                "clipiqa_crop1024": "CLIP-IQA crops", "spectrum_log_rmse": "Log-RMSE",
                "spectrum_slope_error": "Slope error", "tiling1024": "Tiling"}


def metric_label(key, arrow=True):
    """Short metric name with its better direction, e.g. "MUSIQ crops ↑"."""
    return METRIC_SHORT[key] + (f" {METRIC_ARROW[key]}" if arrow else "")


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
    smallest = min_text_pt(fig)
    save(fig, stem, formats=formats)
    if smallest < FS_MIN - 1e-6:
        print(f"FIGURE AUDIT {stem}: smallest text {smallest:.2f} pt is under the {FS_MIN} pt floor "
              "(mathtext scripts included; cs.script_floor raises them)", file=sys.stderr)
    else:
        print(f"figure audit {stem}: smallest text {smallest:.2f} pt (mathtext scripts included)", file=sys.stderr)
    if "svg" in formats:
        svg = Path(stem + ".svg")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    if photo:
        jpegify(stem + ".pdf")
