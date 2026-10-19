"""F1 (page-1 hero): released SEGA's 4K gain over the NTK base comes mostly from its RoPE base and one
global attention temperature; the per-band allocation adds little. Held-out results only.

(a) One held-out prompt at 4096 px, seed 0: manifest row 32 (isolated wooden cabin), chosen by subject
    from captions during the senior-review revision before viewing this row; no metric-based selection. Full frame =
    released SEGA, Lanczos-reduced. Three arms in narrative order: NTK (RoPE base only), SEGA base +
    temp. (SEGA's released base and global temperature), SEGA (released) (adds the per-band allocation).
    Crops are native pixels, never resampled, at the same two locations for every arm: A = cabin, centred at(2048,3200); B = mountain, centred at(2048,1536).
    These shared coordinates were chosen to retain common subject matter after viewing the three frames. 384 px each, shown at 0.80 in (480 px per inch).
(b) Two paired contrasts on 187 held-out prompts, seed 0: base + temperature minus NTK,
    and released SEGA minus base + temperature (allocation). Each metric has two labelled
    rows on one shared axis in its native units. Dots and unshifted whiskers show the mean
    and nominal 95% paired bootstrap CI (2000 resamples, seed 0).

Regenerate: OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f1_hero.py
"""
from __future__ import annotations

import re

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

import crope_style as cs
import qual_helpers as qh

ROW = 32
CROP = 384                     # crop coordinates were fixed after viewing, as disclosed in the caption
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
X_A = (W - (F + G + 3 * C + 2 * GG)) / 2
X_GRID = X_A + F + G
GRID_BOTTOM = IMG_TOP + 2 * C + GI
B_TOP = GRID_BOTTOM + 0.13
H = B_TOP + 1.54


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
    box_a = (1856, 3008, 2240, 3392)
    box_b = (1856, 1344, 2240, 1728)
    print("crop A (cabin)", box_a, " crop B (mountain)", box_b)

    cs.panel_tag(fig, X_A / W, fy(NAME_Y), "a")
    header(fig, X_A + 0.15, "Full frame", cs.INK, "SEGA (released), 4096 px", ha="left", weight="normal")
    ax = axes_at(fig, X_A, IMG_TOP, F, F)
    photo(ax, cs.thumb(src["sega_official"], THUMB))
    s = THUMB / 4096
    # the letter goes on the side of the box facing away from the other box
    side_a = "right" if box_a[0] >= box_b[0] else "left"
    crop_box(ax, box_a, s, "A", side_a)
    crop_box(ax, box_b, s, "B", "left" if side_a == "right" else "right")
    note = ["A: cabin", "B: mountain", f"native {CROP} px crops,", "identical coordinates"]
    for i, line in enumerate(note):
        fig.text(X_A / W, fy(IMG_TOP + F + 0.13 + 0.1 * i), line, fontsize=6.5, color=cs.INK2, va="baseline")

    for j, m in enumerate(SHOWN):
        x = X_GRID + j * (C + GG)
        for i, box in enumerate((box_a, box_b)):
            photo(axes_at(fig, x, IMG_TOP + i * (C + GI), C, C), cs.crop(src[m], box), frame=True)
        header(fig, x + C / 2, cs.label(m), cs.text_color(m), DESC[m])
    for i, letter in enumerate("AB"):
        fig.text((X_GRID - G / 2) / W, fy(IMG_TOP + i * (C + GI) + C / 2), letter, fontsize=7,
                 fontweight="bold", color=cs.INK, ha="center", va="center")


def paired_gains():
    d = cs.heldout()
    vals = {m: (cs.paired(d, "temp_only_sbase", m, "ntk2"),
                cs.paired(d, "sega_official", m, "temp_only_sbase")) for m in METRICS}
    # Verify every displayed mean and CI against the same generated macros used by the text.
    source = MACROS.read_text()
    for metric, contrasts in vals.items():
        precision = cs.METRIC_PRECISION[metric]
        for key, (mean, lo, hi, n) in zip(("basetemp", "alloc"), contrasts):
            for suffix, expected in (("d", f"${mean:+.{precision}f}$"),
                                     ("ci", f"[${lo:+.{precision}f}$, ${hi:+.{precision}f}$]")):
                name = f"ho@{suffix}@{key}@{MACRO[metric]}"
                actual = re.search(r"\\csname N@%s\\endcsname\{([^}]*)\}" % re.escape(name), source)
                assert actual and actual[1] == expected, (name, expected, actual[1] if actual else None)
            assert n == 187
    return vals


def panel_b(fig, vals):
    cs.panel_tag(fig, 0, fy(B_TOP + .03), "b")
    fig.text(.16 / W, fy(B_TOP + .03), "Paired changes in score (95% CI)",
             fontsize=7.5, color=cs.INK, fontweight="bold")
    col_width, gutter, label_width = 2.65, .20, 1.13
    names = {"imagereward": "ImageReward", "pickscore": "PickScore",
             "musiq": "MUSIQ, full frame", "musiq_crop1024": "MUSIQ, 1024 crops"}
    for i, met in enumerate(METRICS):
        top = B_TOP + .12 + (i // 2) * .70
        x0 = (i % 2) * (col_width + gutter)
        ax = axes_at(fig, x0 + label_width, top + .12, col_width - label_width - .04, .43)
        ax.set_xlabel(names[met], fontsize=7.5, color=cs.INK, loc="left")
        ax.xaxis.set_label_coords(x0 / W, fy(top + .07), transform=fig.transFigure)
        low = min(0, *(v[1] for v in vals[met]))
        high = max(v[2] for v in vals[met])
        span = high - low
        ax.set_xlim(low - .13 * span, high + .14 * span)
        ax.set_ylim(-.35, 1.65)
        ax.axvline(0, color=cs.INK2, alpha=.65, lw=.75, zorder=0)
        precision = cs.METRIC_PRECISION[met]
        for y, contrast, method, name in zip((1, 0), vals[met],
                ("temp_only_sbase", "sega_official"), ("base + temperature", "allocation")):
            mean, lo, hi, n = contrast
            ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o", markersize=3.5,
                        color=cs.color(method), markeredgecolor=cs.text_color(method),
                        markeredgewidth=.5, elinewidth=1.1, capsize=2.2, capthick=.8, zorder=2)
            ax.annotate(f"{mean:+.{precision}f}", (mean, y), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7,
                        color=cs.text_color(method))
        ax.set_yticks([1, 0], ["base + temperature", "allocation"])
        ax.tick_params(axis="y", length=0, pad=6, labelsize=7)
        for tick, method in zip(ax.get_yticklabels(), ("temp_only_sbase", "sega_official")):
            tick.set_color(cs.text_color(method))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.2g}"))
        ax.tick_params(axis="x", labelsize=7, length=2, pad=2)
        for side in ["left", "top", "right"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(cs.RULE)
        ax.grid(False)
        print(met, "base+temperature", vals[met][0], "allocation", vals[met][1])


def main():
    cs.style()
    check_macros(shares())
    vals = paired_gains()
    fig = plt.figure(figsize=(W, H))
    panel_a(fig)
    panel_b(fig, vals)
    print(f"figure size {W:.2f} x {H:.3f} in")
    cs.finish(fig, cs.HERE / "f1_hero", photo=True, formats=("pdf", "svg"))


if __name__ == "__main__":
    main()
