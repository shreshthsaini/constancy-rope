"""F3: held-out paired differences from released SEGA for every method and metric, with the seed-1 replication.

Small multiples: one compact axis per metric (own x-scale, shared zero line = released SEGA), method rows in a
fixed order and three groups (baselines | SEGA ablation | per-band laws). Held-out results only: seed 0 on rows
8-194 (n = 187) for every row; seed 1 on rows 8-62 (n = 55) for SEGA base + temp. and Constancy-TM. The partial
Constancy run (NTK base, 41 prompts) is not drawn. Intervals are 95% percentile intervals of 2,000 paired bootstrap
resamples over prompts (crope_style.paired), and the seed-0 values are asserted equal to
.agent/notes/main2/heldout_paired_vs_sega_official.csv within 1e-9.

Axis rule: limits cover every interval of DyPE, SEGA base + temp., Constancy-TM, SEGA + TM (both seeds) and zero.
NTK is added if that at most multiplies the span by 1.5; otherwise its row is clipped at the edge with an outward
triangle and its value printed.

Regenerate (from paper/figs):  OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f3_forest.py
"""
from __future__ import annotations

import math

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

import crope_style as cs

ORDER = ["ntk2", "dype", "temp_only_sbase", "constancy_tm_sbase", "sega_ctm_v2"]
NOT_DRAWN = {"constancy"}                # partial run (41 prompts): appendix only, never in a main-text figure
SEED1 = ["temp_only_sbase", "constancy_tm_sbase"]
FAR = {"ntk2": 1.5}                      # row -> largest allowed widening of the core span before clipping
GROUP_BREAK = {"temp_only_sbase", "constancy_tm_sbase"}   # extra space above these rows (baselines | SEGA ablation | laws)
FADE = 0.60                              # share of colour kept when the 95% CI includes zero

# two figure rows of five panels; each entry is (group header, [(metric, panel title), ...])
LAYOUT = [
    [("Reward and alignment", [("imagereward", "ImageReward"), ("pickscore", "PickScore"), ("clip", "CLIP score")]),
     ("Image quality: MUSIQ", [("musiq", "Full frame"), ("musiq_crop1024", "1024-px crops")])],
    [("Image quality: CLIP-IQA", [("clipiqa", "Full frame"), ("clipiqa_crop1024", "1024-px crops")]),
     ("Spectrum and structure", [("spectrum_log_rmse", "Log-RMSE"), ("spectrum_slope_error", "Slope error"),
                                 ("tiling1024", "Tiling, 1024 px")])],
]

# geometry (inches)
W = cs.TEXT
LABEL_W = 1.04
GUT = 0.13
RIGHT = 0.03
PW = (W - LABEL_W - 4 * GUT - RIGHT) / 5
TOP = 0.03
HEAD = 0.19          # group header + rule
TITLE = 0.14         # metric names
PITCH = 0.125        # inches per method row
TICKS = 0.13
XLAB = 0.11
GAP = 0.12
KEY = 0.17
SUB = 0.5            # seed-1 offset below the seed-0 row, in row units
BREAK = 0.28         # extra space between method groups, in row units
TICK_PT = 7.0


def ypos():
    y, pos, sub = 0.0, {}, {}
    for i, m in enumerate(ORDER):
        if i:
            y += 1.0 + (SUB if ORDER[i - 1] in SEED1 else 0.0) + (BREAK if m in GROUP_BREAK else 0.0)
        pos[m] = y
        if m in SEED1:
            sub[m] = y + SUB
    return pos, sub


POS, SUBPOS = ypos()
YLIM = (max(list(POS.values()) + list(SUBPOS.values())) + 0.6, -0.6)
PH = (YLIM[0] - YLIM[1]) * PITCH
BLOCK = HEAD + TITLE + PH + TICKS + XLAB
H = TOP + 2 * BLOCK + GAP + KEY


# ------------------------------------------------------------------ data
def compute():
    h = cs.heldout()
    assert h.idx.between(8, 194).all() and (h.seed == 0).all()
    ref = pd.read_csv(cs.PAIRED_V2)
    old = pd.read_csv(cs.PAIRED)
    assert set(ref.method) == set(ORDER) | NOT_DRAWN and set(ref.metric) == set(cs.METRICS)
    s1 = cs.seed1()
    res, worst = {}, 0.0
    for m in ORDER:
        for k in cs.METRICS:
            d, lo, hi, n = cs.paired(h, m, k)
            for table in (ref, old):
                r = table[(table.method == m) & (table.metric == k)]
                if table is old and r.empty:
                    continue
                r = r.iloc[0]
                err = max(abs(d - r.delta), abs(lo - r.lo), abs(hi - r.hi))
                assert int(r.n) == n and err < 1e-9, (m, k, err)
                worst = max(worst, err)
            res[(m, k, 0)] = (d, lo, hi, n)
    for m in SEED1:
        for k in cs.METRICS:
            d, lo, hi, n = cs.paired(s1, m, k)
            assert n == 55, (m, k, n)
            res[(m, k, 1)] = (d, lo, hi, n)
    print(f"seed-0 paired deltas match both CSVs (max abs diff {worst:.1e}); seed-1 n = 55")
    return res


def limits(res, k):
    core = [0.0]
    for m in ORDER:
        if m in FAR:
            continue
        core += list(res[(m, k, 0)][1:3])
        if m in SEED1:
            core += list(res[(m, k, 1)][1:3])
    lo, hi = min(core), max(core)
    span = hi - lo
    for m, factor in FAR.items():
        a, b = res[(m, k, 0)][1:3]
        if max(hi, b) - min(lo, a) <= factor * span:
            lo, hi = min(lo, a), max(hi, b)
    pad = 0.07 * (hi - lo)
    return lo - pad, hi + pad


def tick_fmt(step):
    dec = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    while abs(step * 10 ** dec - round(step * 10 ** dec)) > 1e-6:
        dec += 1

    def f(v, _pos=None):
        if abs(v) < step * 1e-6:
            return "0"
        return f"{v:.{dec}f}".replace("-", "−")
    return f


def label_width(text, size=TICK_PT):
    """Printed width in inches of a numeric tick label in Helvetica metrics (digits 0.556 em, point 0.278 em)."""
    em = sum(0.278 if ch == "." else 0.584 if ch in "\u2212+" else 0.556 for ch in text)
    return em * size / 72


def nice_ticks(lo, hi):
    """Most ticks (2 to 4, always including 0) whose labels keep a clear gap at the printed panel width."""
    span = hi - lo
    e0 = int(math.floor(math.log10(span)))
    for mults in ((1, 2, 5), (2.5,)):
        for step in sorted(m * 10 ** e for e in range(e0 - 2, e0 + 2) for m in mults):
            ticks = np.arange(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1) * step
            if not 2 <= len(ticks) <= 4:
                continue
            w = [label_width(tick_fmt(step)(t)) for t in ticks]
            need = max((w[i] + w[i + 1]) / 2 for i in range(len(w) - 1)) + 0.06
            if step / span * PW >= need:
                return ticks, step
    raise ValueError(f"no tick step for [{lo}, {hi}]")


def val_fmt(v):
    a = abs(v)
    s = f"{a:.1f}" if a >= 10 else (f"{a:.2f}" if a >= 0.095 else f"{a:.3f}")
    return ("−" if v < 0 else "+") + s


def tint(c, f):
    r, g, b = to_rgb(c)
    return (1 - f * (1 - r), 1 - f * (1 - g), 1 - f * (1 - b))


def msize(marker, hollow=False):
    return (3.7 if marker == "o" else 3.2) * (0.9 if hollow else 1.0)


# ------------------------------------------------------------------ drawing
def draw_row(ax, y, est, color, marker, lim, lower, hollow=False):
    d, lo, hi, _ = est
    x0, x1 = lim
    sig = lo > 0 or hi < 0
    c = color if sig else tint(color, FADE)
    a, b = max(lo, x0), min(hi, x1)
    if b - a < 0.03 * (x1 - x0):   # a sliver at the edge reads as noise: treat the row as off-axis
        a, b = x1, x0
    if a < b:
        ax.plot([a, b], [y, y], color=c, lw=0.85 if hollow else 1.3, solid_capstyle="butt", zorder=3)
    for side, beyond in ((x0, lo < x0), (x1, hi > x1)):
        if beyond and a < b:   # interval continues past the edge
            ax.plot([side], [y], marker="<" if side == x0 else ">", ms=2.6 if hollow else 3.0, color=c,
                    mec="none", clip_on=False, zorder=4)
    if x0 <= d <= x1:
        kw = dict(mfc="white", mec=c, mew=0.9) if hollow else dict(mfc=c, mec="white", mew=0.45)
        ax.plot([d], [y], marker=marker, ms=msize(marker, hollow), ls="none", zorder=5, **kw)
        return
    left = d < x0
    side = x0 if left else x1
    bg = cs.PANEL if (side < 0) == lower else "white"   # halo matches the half-plane the label starts in
    ax.plot([side], [y], marker="<" if left else ">", ms=4.4, color=c, mec="white", mew=0.4,
            clip_on=False, zorder=5)
    partial = a < b   # part of the interval is visible: print the value above the row, clear of the line
    ax.annotate(val_fmt(d), (side, y), xytext=(4.0 if left else -4.0, 3.2 if partial else 0),
                textcoords="offset points", ha="left" if left else "right",
                va="bottom" if partial else "center", fontsize=6.5, color=cs.INK2, zorder=6,
                path_effects=[pe.withStroke(linewidth=1.6, foreground=bg)])


def draw_panel(ax, res, k, title):
    lim = limits(res, k)
    lower = cs.METRICS[k]["lower"]
    ax.set_xlim(*lim)
    ax.set_ylim(*YLIM)
    # better half-plane, zero line (= released SEGA), row guides
    ax.axvspan(lim[0] if lower else 0.0, 0.0 if lower else lim[1], color=cs.PANEL, lw=0, zorder=0)
    for m in ORDER:
        ax.axhline(POS[m], color=cs.GRIDC, lw=0.35, zorder=1)
    ax.axvline(0.0, color=cs.RULE, lw=0.8, zorder=2)
    for m in ORDER:
        spec = cs.METHODS[m]
        draw_row(ax, POS[m], res[(m, k, 0)], spec["color"], spec["marker"], lim, lower)
        if m in SEED1:
            draw_row(ax, SUBPOS[m], res[(m, k, 1)], spec["color"], spec["marker"], lim, lower, hollow=True)
    ticks, step = nice_ticks(*lim)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(tick_fmt(step)))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(step / 2))
    ax.set_yticks([])
    ax.grid(False)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(cs.RULE)
    ax.tick_params(axis="x", length=2.2, width=0.6, pad=1.6, labelsize=TICK_PT, color=cs.RULE)
    ax.tick_params(axis="x", which="minor", length=1.3, width=0.5, color=cs.RULE)
    ax.set_xlabel("← better" if lower else "better →", loc="left" if lower else "right",
                  fontsize=6.5, color=cs.INK2, labelpad=1.2)
    ax.text(0.5, 1.0, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=cs.INK)


def label_text_width(fig, text, size=6.5):
    t = fig.text(0, 0, text, fontsize=size)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def main():
    cs.style()
    res = compute()
    fig = plt.figure(figsize=(W, H))
    fx = lambda inch: inch / W          # noqa: E731
    fy = lambda inch: 1 - inch / H      # noqa: E731  (inches measured from the top)

    for r, groups in enumerate(LAYOUT):
        top = TOP + r * (BLOCK + GAP)
        ax_top = top + HEAD + TITLE
        col, axes = 0, []
        for header, panels in groups:
            gx0 = LABEL_W + col * (PW + GUT)
            gx1 = gx0 + len(panels) * PW + (len(panels) - 1) * GUT
            fig.text(fx(gx0), fy(top + 0.11), header, fontsize=7.5, fontweight="bold", color=cs.INK,
                     ha="left", va="baseline")
            fig.add_artist(Line2D([fx(gx0), fx(gx1)], [fy(top + 0.15)] * 2, color=cs.RULE, lw=0.6))
            for k, title in panels:
                x0 = LABEL_W + col * (PW + GUT)
                ax = fig.add_axes([fx(x0), fy(ax_top + PH), fx(PW), PH / H], sharey=axes[0] if axes else None)
                draw_panel(ax, res, k, title)
                axes.append(ax)
                col += 1
        # the label column is the y axis of the block: header "Method" is the first panel's y label
        axes[0].set_ylabel("Method", rotation=0, ha="right", va="bottom", fontsize=7, color=cs.INK)
        axes[0].yaxis.set_label_coords(-0.16 / PW, 1.0 + 0.035 / PH)
        # method labels with colour keys, on the rows of this block
        tr = blended_transform_factory(fig.transFigure, axes[0].transData)
        for m in ORDER:
            spec = cs.METHODS[m]
            fig.text(fx(LABEL_W - 0.16), POS[m], spec["label"], transform=tr,
                     ha="right", va="center", fontsize=7, color=cs.INK)
            fig.add_artist(Line2D([fx(LABEL_W - 0.09)], [POS[m]], transform=tr, marker=spec["marker"],
                                  ms=msize(spec["marker"]), mfc=spec["color"], mec="white", mew=0.45, ls="none"))
            if m in SEED1:
                fig.add_artist(Line2D([fx(LABEL_W - 0.09)], [SUBPOS[m]], transform=tr, marker=spec["marker"],
                                      ms=msize(spec["marker"], True), mfc="white", mec=spec["color"], mew=0.85,
                                      ls="none"))
        # shared x label for the block, on the tick-label line
        fig.text(fx(LABEL_W - 0.16), fy(ax_top + PH + 0.105), "Method − SEGA", fontsize=6.5,
                 color=cs.INK2, ha="right", va="baseline")

    # key
    ky = fy(H - KEY / 2 + 0.005)
    items = [
        (dict(marker="|", mec=cs.RULE, mew=0.8, ms=6.0), "0 = SEGA (released)"),
        (dict(marker="o", mfc=cs.INK2, mec="white", mew=0.45, ms=3.7), "95% CI excludes 0"),
        (dict(marker="o", mfc=tint(cs.INK2, FADE), mec="white", mew=0.45, ms=3.7), "95% CI includes 0"),
        (dict(marker="o", mfc="white", mec=cs.INK2, mew=0.85, ms=3.3), "seed 1, n = 55"),
        (dict(marker="<", mfc=cs.INK2, mec="white", mew=0.4, ms=4.4), "off-axis, value printed"),
    ]
    fig.canvas.draw()
    widths = [label_text_width(fig, t) for _, t in items]
    sep = 0.2
    total = sum(widths) + len(items) * 0.09 + (len(items) - 1) * sep
    x = (W - total) / 2
    for (style, text), w in zip(items, widths):
        fig.add_artist(Line2D([fx(x + 0.03)], [ky], ls="none", **style))
        fig.text(fx(x + 0.09), ky, text, fontsize=6.5, color=cs.INK2, ha="left", va="center")
        x += 0.09 + w + sep
    cs.finish(fig, cs.HERE / "f3_forest")


if __name__ == "__main__":
    main()
