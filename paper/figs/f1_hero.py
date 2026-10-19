"""F1 (page-1 hero): released SEGA's 4K gain over the NTK base comes mostly from its RoPE base and one
global attention temperature; the per-band allocation adds little. Held-out results only.

(a) One held-out prompt at 4096 px, seed 0: manifest row 28 (cathedral interior), chosen by subject
    before any output was viewed, and outside rows 8-10 so it does not repeat F4. Full frame =
    released SEGA, Lanczos-reduced. Three arms in narrative order: NTK (RoPE base only), SEGA base +
    temp. (SEGA's released base and global temperature), SEGA (released) (adds the per-band allocation).
    Crops are native pixels, never resampled, at the same two locations for every arm: A = tile with
    the maximum Laplacian energy of the mean luminance of the three arms shown
    (qual_helpers.detail_box), B = image centre. 384 px each, shown at 0.80 in (480 px per inch).
(b) Share of SEGA's gain over NTK recovered by SEGA base + temp.:
    100 * (mean_B - mean_NTK) / (mean_SEGA - mean_NTK) on held-out rows 8-194, seed 0 (n = 187), with
    a 95% percentile interval from 2,000 joint bootstrap resamples of prompts (ratio recomputed per
    resample, numpy default_rng(0)). Computed exactly as tables/make_tables.py table_share() and
    checked against the N@sh@... macros in tables/num_results.tex at run time. Where the share is
    below 100%, a cobalt segment fills the remainder up to 100%: the part of the gain that the
    per-band allocation adds. The value column prints each share with its interval.

Regenerate: OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f1_hero.py
"""
from __future__ import annotations

import re

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import crope_style as cs
import qual_helpers as qh

ROW = 28
CROP = 384                     # native crop side in px (fixed before rendering; not tuned on content)
THUMB = 768                    # Lanczos-reduced full frame, px
SHOWN = ["ntk2", "temp_only_sbase", "sega_official"]          # narrative order
DESC = {"ntk2": "base only", "temp_only_sbase": "+ base, temperature",
        "sega_official": "+ per-band allocation"}
METRICS = ["imagereward", "pickscore", "musiq", "musiq_crop1024"]
MACRO = {"imagereward": "ir", "pickscore": "pick", "musiq": "musiq", "musiq_crop1024": "musiqc"}
MACROS = cs.HERE.parent / "tables" / "num_results.tex"

# ==== geometry (inches)
W = cs.TEXT
NAME_Y = 0.115        # baseline of panel letters and method names, from the top
DESC_Y = 0.225        # baseline of the grey descriptor line
IMG_TOP = 0.275       # top of the photos and of panel b's plot block
F = 1.05              # full-frame thumbnail side
G = 0.13              # thumbnail to crop grid (holds the row letters A, B)
C = 0.80              # crop side on the page
GG = 0.09             # between method columns (keeps the bold headers 5 pt apart)
GI = 2 / 72           # between rows A and B (2 pt)
GAP_AB = 0.17         # panel a to panel b
X_GRID = F + G
X_B = X_GRID + 3 * C + 2 * GG + GAP_AB          # left edge of panel b
GRID_BOTTOM = IMG_TOP + 2 * C + GI
H = GRID_BOTTOM + 0.015


# ==== numbers
def ratio_ci(a, b, c, n_boot=2000, seed=0):
    """Share (%) of the c - a gain carried by b - a; one fresh default_rng(seed) per metric, as make_tables."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    share = (b - a)[idx].mean(1) / (c - a)[idx].mean(1)
    lo, hi = np.percentile(share, [2.5, 97.5])
    return 100 * float((b - a).mean() / (c - a).mean()), 100 * float(lo), 100 * float(hi), len(a)


def shares():
    d = cs.heldout()
    assert d.idx.between(8, 194).all() and (d.seed == 0).all()
    piv = {m: d[d.method == m].set_index("prompt_id") for m in SHOWN}
    common = piv["sega_official"].index.intersection(piv["ntk2"].index).intersection(piv["temp_only_sbase"].index)
    out = {}
    for met in METRICS:
        a, b, c = (piv[m].loc[common, met].to_numpy(float) for m in SHOWN)
        out[met] = ratio_ci(a, b, c)
    return out


def check_macros(vals):
    """Print the figure's shares next to the paper macros and require the rounded values to agree."""
    text = MACROS.read_text() if MACROS.exists() else ""
    get = lambda k: (re.search(r"\\csname N@%s\\endcsname\{([^}]*)\}" % re.escape(k), text) or [None, None])[1]  # noqa: E731
    for met, (pt, lo, hi, n) in vals.items():
        fig_p, fig_ci = str(round(pt)), "[%d, %d]" % (round(lo), round(hi))
        tex_p, tex_ci = get(f"sh@p@{MACRO[met]}"), get(f"sh@ci@{MACRO[met]}")
        print(f"{met:15s} share {pt:7.2f} [{lo:6.2f}, {hi:6.2f}] n={n}  figure {fig_p}% {fig_ci}"
              f"  macros {tex_p}% {tex_ci}")
        if tex_p is None:
            print(f"  warning: no N@sh@p@{MACRO[met]} macro in {MACROS}")
            continue
        assert (fig_p, fig_ci) == (tex_p, tex_ci), (met, fig_p, fig_ci, tex_p, tex_ci)
    n_tex = get("sh@n")
    assert n_tex is None or int(n_tex) == vals[METRICS[0]][3], n_tex


# ==== drawing helpers
HALO = [pe.withStroke(linewidth=1.8, foreground="#111317")]


def axes_at(fig, x, top, w, h, **kw):
    """Axes from x (left) and top (both in inches from the top-left corner of the figure)."""
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H], **kw)


def fy(top):
    return 1 - top / H


def photo(ax, img, frame=False):
    ax.imshow(img, interpolation="none")
    ax.axis("off")
    if frame:  # 0.4 pt hairline around 1:1 crops
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=cs.RULE,
                               linewidth=0.4, zorder=5, clip_on=False))


def crop_box(ax, box, scale, letter, side):
    x0, y0, x1, y1 = (v * scale for v in box)
    r = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=0.8)
    r.set_path_effects([pe.withStroke(linewidth=2.0, foreground="#111317"), pe.Normal()])
    ax.add_patch(r)
    pad = 0.03 * THUMB
    x = x1 + pad if side == "right" else x0 - pad
    ax.text(x, (y0 + y1) / 2, letter, color="white", fontsize=7, fontweight="bold",
            ha="left" if side == "right" else "right", va="center", path_effects=HALO)


def header(fig, x, name, color, desc, ha="center", weight="bold"):
    fig.text(x / W, fy(NAME_Y), name, color=color, fontsize=7, fontweight=weight, ha=ha, va="baseline")
    fig.text(x / W, fy(DESC_Y), desc, color=cs.INK2, fontsize=6.5, ha=ha, va="baseline")


# ==== figure
def panel_a(fig):
    src = qh.sources(ROW, SHOWN)
    mean_y = qh.mean_luma([src[m] for m in SHOWN])
    box_a, box_b = qh.detail_box(mean_y, CROP), qh.center_box(CROP)
    print("crop A (detail)", box_a, " crop B (centre)", box_b)

    cs.panel_tag(fig, 0, fy(NAME_Y), "a")
    header(fig, 0.15, "Full frame", cs.INK, "SEGA (released), 4096 px", ha="left", weight="normal")
    ax = axes_at(fig, 0, IMG_TOP, F, F)
    photo(ax, cs.thumb(src["sega_official"], THUMB))
    s = THUMB / 4096
    # the letter goes on the side of the box facing away from the other box
    side_a = "right" if box_a[0] >= box_b[0] else "left"
    crop_box(ax, box_a, s, "A", side_a)
    crop_box(ax, box_b, s, "B", "left" if side_a == "right" else "right")
    note = ["A: highest-detail tile", "B: image centre", f"native {CROP} px crops,", "identical coordinates"]
    for i, line in enumerate(note):
        fig.text(0, fy(IMG_TOP + F + 0.13 + 0.1 * i), line, fontsize=6.5, color=cs.INK2, va="baseline")

    for j, m in enumerate(SHOWN):
        x = X_GRID + j * (C + GG)
        for i, box in enumerate((box_a, box_b)):
            photo(axes_at(fig, x, IMG_TOP + i * (C + GI), C, C), cs.crop(src[m], box), frame=True)
        header(fig, x + C / 2, cs.label(m), cs.text_color(m), DESC[m])
    for i, letter in enumerate("AB"):
        fig.text((X_GRID - G / 2) / W, fy(IMG_TOP + i * (C + GI) + C / 2), letter, fontsize=7,
                 fontweight="bold", color=cs.INK, ha="center", va="center")


def panel_b(fig, vals):
    cs.panel_tag(fig, X_B / W, fy(NAME_Y), "b")
    x0 = X_B + 0.14
    header(fig, x0, cs.label("temp_only_sbase"), cs.text_color("temp_only_sbase"),
           "share of SEGA's gain over NTK", ha="left")
    xmax, val_w = 115.0, 0.34
    A = W - val_w - 0.04 - x0                                  # plot width; value column on the right
    top, bottom = IMG_TOP + 0.02, GRID_BOTTOM - 0.14
    ax = axes_at(fig, x0, top, A, bottom - top)
    n = len(METRICS)
    ax.set_xlim(0, xmax)
    ax.set_ylim(n - 0.45, -0.75)
    bar = 0.40
    halo = [pe.withStroke(linewidth=1.5, foreground="white"), pe.Normal()]
    for i, met in enumerate(METRICS):
        pt, lo, hi, _ = vals[met]
        ax.barh(i, pt, height=bar, left=0, color=cs.SKY, lw=0, zorder=2)
        if pt < 100:   # remainder up to released SEGA: what the per-band allocation adds
            ax.barh(i, 100 - pt, height=bar, left=pt, color=cs.COBALT, lw=0, zorder=2)
        ax.plot([lo, hi], [i, i], color=cs.INK, lw=0.7, solid_capstyle="butt", zorder=4, path_effects=halo)
        ax.plot([lo, hi], [i, i], ls="none", marker="|", ms=3.0, mew=0.7, color=cs.INK, zorder=4,
                path_effects=halo)
        ax.annotate(f"{round(pt)}%", (1.0, i), xycoords=("axes fraction", "data"),
                    xytext=((val_w + 0.04) * 72, 0), textcoords="offset points",
                    ha="right", va="baseline", fontsize=7.5, fontweight="bold", color=cs.INK, zorder=5)
        ax.annotate(f"[{round(lo)}, {round(hi)}]", (1.0, i), xycoords=("axes fraction", "data"),
                    xytext=((val_w + 0.04) * 72, -8.0), textcoords="offset points",
                    ha="right", va="baseline", fontsize=6.5, color=cs.INK2, zorder=5)
    # metric names are the categorical y ticks, set inside the plot above each bar
    ax.set_yticks([i - bar / 2 - 0.12 for i in range(n)], [cs.METRICS[m]["label"] for m in METRICS])
    ax.tick_params(axis="y", length=0, pad=-2.5, labelsize=7, labelcolor=cs.INK)
    for t in ax.get_yticklabels():
        t.set_horizontalalignment("left")
        t.set_verticalalignment("baseline")
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
    ax.axvline(100, color=cs.COBALT, lw=1.0, zorder=3)
    ax.axvline(0, color=cs.STONE, lw=1.0, zorder=3)
    ax.set_xticks([0, 50, 100], ["NTK", "", cs.label("sega_official")])
    for t, col in zip(ax.get_xticklabels(), (cs.text_color("ntk2"), cs.INK2, cs.text_color("sega_official"))):
        t.set_color(col)
    ax.tick_params(axis="x", length=2.0, width=0.6, pad=1.8, labelsize=7, color=cs.RULE)
    ax.grid(False)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(cs.RULE)


def main():
    cs.style()
    vals = shares()
    check_macros(vals)
    fig = plt.figure(figsize=(W, H))
    panel_a(fig)
    panel_b(fig, vals)
    print(f"figure size {W:.2f} x {H:.3f} in")
    cs.finish(fig, cs.HERE / "f1_hero", photo=True)


if __name__ == "__main__":
    main()
