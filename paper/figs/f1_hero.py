"""F1 (page-1 hero): released SEGA's 4K gain over the NTK base comes mostly from its RoPE base and one
global attention temperature; the per-band allocation adds little. Held-out results only.

(a) Paired gains on 187 held-out prompts, seed 0, one small panel per metric (ImageReward, PickScore,
    MUSIQ, MUSIQ crops), each on its own scale: SEGA base + temp. minus NTK (green) and SEGA (released)
    minus SEGA base + temp. ("allocation", blue). Bars are means, whiskers nominal 95% paired bootstrap
    intervals (2000 resamples, seed 0), values printed at the marks in table precision.
(b) One held-out prompt at 4096 px, seed 0: manifest row 32 (isolated wooden cabin), chosen by subject
    from captions during the senior-review revision before viewing this row; no metric-based selection.
    Columns in narrative order: NTK (RoPE base only), SEGA base + temp. (SEGA's released base and global
    temperature), SEGA (released) (adds the per-band allocation). Top row: full frames, Lanczos-reduced.
    Below each frame, two native 384 px crops at the same locations for every arm: cabin (red box),
    centred at (2048, 3200), and mountain (cyan box), centred at (2048, 1536). These shared coordinates
    were chosen after viewing the three frames, to keep common subject matter.

Regenerate: OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f1_hero.py
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import crope_style as cs
import qual_helpers as qh

ROW = 32
CROP = 384                     # crop coordinates were fixed after viewing, as disclosed in the caption
THUMB = 768                    # Lanczos-reduced full frame, px
SHOWN = ["ntk2", "temp_only_sbase", "sega_official"]          # narrative order
BOXES = [((1856, 3008, 2240, 3392), cs.ZOOM),                  # cabin, centred at (2048, 3200)
         ((1856, 1344, 2240, 1728), cs.ZOOM2)]                 # mountain, centred at (2048, 1536)
METRICS = ["imagereward", "pickscore", "musiq", "musiq_crop1024"]
MACRO = {"imagereward": "ir", "pickscore": "pick", "musiq": "musiq", "musiq_crop1024": "musiqc"}
MACROS = cs.HERE.parent / "tables" / "num_results.tex"
CONTRASTS = (("temp_only_sbase", "base + temp."), ("sega_official", "allocation"))  # bar hue, legend

# ==== geometry (inches, measured from the top-left corner)
W = cs.TEXT
TAG_Y = 0.115                  # baseline of the panel tags
TOP = 0.18                     # top of the full frames
T = 0.90                       # side of every photo tile
GR = 0.035                     # gutter inside a column (frame, crop, crop)
GC = 0.11                      # gutter between arms (keeps the 8 pt labels apart)
TILES_BOTTOM = TOP + 3 * T + 2 * GR
LAB_Y = TILES_BOTTOM + 0.13    # baseline of the arm and metric labels
H = LAB_Y + 0.04
X_B = W - (3 * T + 2 * GC)     # left edge of panel b (right-aligned to the text width)
A_W = X_B - 0.24               # width of panel a
PGAP = 0.12                    # gap between the two columns of metric panels
PW = (A_W - PGAP) / 2          # width of one metric panel
PH = 1.05                      # height of one metric panel (bars, whiskers, printed values)
RGAP = 0.36                    # from the bottom of a panel row to the top of the next one
LOFF = 0.12                    # from the bottom of a metric panel to its label baseline
LEG_Y = 0.30                   # centre of the legend row
ZERO_AT = 0.10                 # zero line height as a fraction of the panel
HEADROOM = 0.16                # space above the highest whisker for its printed value
BW = 0.30                      # bar width (as wide as a printed value)
DX = 0.19                      # bar centre offset from the panel centre (0.08 in between bars)


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
    """Print the shares next to the paper macros and require the rounded values to agree."""
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


def signed(v, precision):
    """Signed value with a true minus sign, e.g. +0.620 or −0.026."""
    return f"{v:+.{precision}f}".replace("-", "−")


# ==== drawing helpers
def axes_at(fig, x, top, w, h, **kw):
    """Axes from x (left) and top (both in inches from the top-left corner of the figure)."""
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H], **kw)


def fy(top):
    return 1 - top / H


def photo(ax, img):
    ax.imshow(img, interpolation="none")
    ax.set_axis_off()


# ==== figure
def panel_a(fig, vals):
    cs.panel_tag(fig, 0, fy(TAG_Y), "a", "Paired gains")
    handles = [Rectangle((0, 0), 1, 1, fc=cs.color(m), ec="none") for m, _ in CONTRASTS]
    fig.legend(handles, [name for _, name in CONTRASTS], loc="center", bbox_to_anchor=(A_W / 2 / W, fy(LEG_Y)),
               ncol=2, frameon=False, fontsize=cs.FS_LABEL, handlelength=0.8, handleheight=0.8,
               handletextpad=0.4, columnspacing=1.4, borderpad=0, borderaxespad=0)
    bottoms = (LAB_Y - LOFF - PH - RGAP, LAB_Y - LOFF)   # lower metric labels share the arm label baseline
    for i, met in enumerate(METRICS):
        row, col = divmod(i, 2)
        x0, bottom = col * (PW + PGAP), bottoms[row]
        ax = axes_at(fig, x0, bottom - PH, PW, PH)
        ax.set_axis_off()
        # Own scale per metric: the zero line sits at the same height in every panel and the
        # highest whisker leaves HEADROOM for its printed value.
        top = max(0.0, *(c[2] for c in vals[met]))
        per_inch = top / (PH * (1 - ZERO_AT) - HEADROOM)
        ax.set_ylim(-ZERO_AT * PH * per_inch, (1 - ZERO_AT) * PH * per_inch)
        ax.set_xlim(0, PW)
        assert min(c[1] for c in vals[met]) > -ZERO_AT * PH * per_inch, met
        precision = cs.METRIC_PRECISION[met]
        for (method, _), (mean, lo, hi, n), x in zip(CONTRASTS, vals[met], (PW / 2 - DX, PW / 2 + DX)):
            ax.bar(x, mean, width=BW, color=cs.color(method), lw=0, zorder=2)
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor=cs.INK,
                        elinewidth=0.7, capsize=2.2, capthick=0.7, zorder=4)
            ax.annotate(signed(mean, precision), (x, max(hi, 0.0)), xytext=(0, 2.0), textcoords="offset points",
                        ha="center", va="bottom", fontsize=cs.FS_VALUE, color=cs.INK)
        ax.plot([PW / 2 - DX - BW / 2 - 0.07, PW / 2 + DX + BW / 2 + 0.07], [0, 0], color=cs.INK, lw=0.8,
                solid_capstyle="butt", zorder=3)
        fig.text((x0 + PW / 2) / W, fy(bottom + LOFF), cs.metric_label(met), fontsize=cs.FS_LABEL,
                 color=cs.INK, ha="center", va="baseline")
        print(met, "base+temperature", vals[met][0], "allocation", vals[met][1])


def panel_b(fig):
    src = qh.sources(ROW, SHOWN)
    print("crops (cabin, mountain):", [b for b, _ in BOXES])
    cs.panel_tag(fig, X_B / W, fy(TAG_Y), "b", "4K samples")
    s = THUMB / 4096
    for j, m in enumerate(SHOWN):
        x = X_B + j * (T + GC)
        ax = axes_at(fig, x, TOP, T, T)
        photo(ax, cs.thumb(src[m], THUMB))
        for (x0, y0, x1, y1), hue in BOXES:   # pixel edges of the reduced frame sit at k - 0.5
            cs.crop_box(ax, (x0 * s - 0.5, y0 * s - 0.5, (x1 - x0) * s, (y1 - y0) * s), color=hue)
        for i, (box, hue) in enumerate(BOXES):
            assert box[2] - box[0] == box[3] - box[1] == CROP
            ax = axes_at(fig, x, TOP + (i + 1) * (T + GR), T, T)
            photo(ax, cs.crop(src[m], box))
            cs.crop_frame(ax, hue)
        fig.text((x + T / 2) / W, fy(LAB_Y), cs.label(m), fontsize=cs.FS_LABEL, color=cs.INK, ha="center",
                 va="baseline", fontweight="bold" if m == "sega_official" else "normal")


def main():
    cs.style()
    check_macros(shares())
    vals = paired_gains()
    fig = plt.figure(figsize=(W, H))
    panel_a(fig, vals)
    panel_b(fig)
    print(f"figure size {W:.2f} x {H:.3f} in")
    cs.finish(fig, cs.HERE / "f1_hero", photo=True)


if __name__ == "__main__":
    main()
