"""F3: held-out paired differences from released SEGA for every method and metric, with the seed-1 replication.

Small multiples: two rows of five panels, one per metric (own x-scale), methods as rows in a fixed order with
their names at the left of each row of panels. The dashed blue zero line is the reference arm, SEGA (released).
Held-out results only: seed 0 on rows 8-194 (n = 187) for every row; seed 1 on rows 8-62 (n = 55) for SEGA
base + temp. and Constancy-TM (hollow markers). The partial Constancy run (NTK base, 41 prompts) is not drawn.
Intervals are 95% percentile intervals of 2,000 paired bootstrap resamples over prompts (crope_style.paired),
and the seed-0 values are asserted equal to .agent/notes/main2/heldout_paired_vs_sega_official.csv within 1e-9.
A mark is pale when its interval includes 0. The caption defines every encoding; the figure carries no legend,
no printed values and no direction shading (the arrow after each metric name gives the better direction).

Axis rule: limits cover every interval of DyPE, SEGA base + temp., Constancy-TM, SEGA + TM (both seeds) and zero.
NTK is added if that at most multiplies the span by 1.5; otherwise an outward triangle at the axis limit marks it.

Print size 5.50 x 2.90 in, included at \\linewidth.
Regenerate (from paper/figs):  OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f3_forest.py
"""
from __future__ import annotations

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import crope_style as cs

ORDER = ["ntk2", "dype", "temp_only_sbase", "constancy_tm_sbase", "sega_ctm_v2"]
NOT_DRAWN = {"constancy"}                # partial run (41 prompts): appendix only, never in a main-text figure
SEED1 = ["temp_only_sbase", "constancy_tm_sbase"]
FAR = {"ntk2": 1.5}                      # row -> largest allowed widening of the core span before clipping
GROUP_BREAK = {"temp_only_sbase", "constancy_tm_sbase"}   # extra space above these rows (baselines | SEGA ablation | laws)
FADE = 0.60                              # share of colour kept when the 95% CI includes zero

# two figure rows of five panels, in the order of the metric groups (reward and alignment, image quality,
# spectrum and structure); titles are cs.metric_label(key) with the better-direction arrow
LAYOUT = [
    ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024"],
    ["clipiqa", "clipiqa_crop1024", "spectrum_log_rmse", "spectrum_slope_error", "tiling1024"],
]

# geometry (inches)
W = cs.TEXT
H = 2.90
LABEL_PT = 8.0       # method names (row labels)
TITLE_PT = 8.0       # metric names (top x labels)
TICK_PT = 7.0
LABEL_GAP = 0.11     # method name to first panel (clear of an edge triangle)
LEFT = 0.03
GUT = 0.12           # between panels
RIGHT = 0.06
TOP = 0.02
TITLE = 0.15         # metric-name line above each panel
TICKS = 0.16         # tick marks and tick labels below each panel
GAP = 0.13           # between the two rows of panels
SUB = 0.55           # seed-1 offset below the seed-0 row, in row units
BREAK = 0.30         # extra space between method groups, in row units


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
# a method name sits on its row, or midway between its seed-0 and seed-1 rows
LABEL_Y = {m: (POS[m] + SUBPOS[m]) / 2 if m in SEED1 else POS[m] for m in ORDER}
PH = (H - TOP - 2 * (TITLE + TICKS) - GAP) / 2


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


# ------------------------------------------------------------------ ticks
def tick_fmt(step):
    dec = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    while abs(step * 10 ** dec - round(step * 10 ** dec)) > 1e-6:
        dec += 1

    def f(v, _pos=None):
        if abs(v) < step * 1e-6:
            return "0"
        return f"{v:.{dec}f}".replace("-", "−")
    return f


_WIDTH = {}


def text_width(fig, text, size):
    """Printed width in inches of `text` at `size` pt, measured with the figure's renderer."""
    if (text, size) not in _WIDTH:
        t = fig.text(0, 0, text, fontsize=size)
        _WIDTH[(text, size)] = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        t.remove()
    return _WIDTH[(text, size)]


def nice_ticks(fig, lo, hi, room_left, room_right):
    """Two or three ticks (always 0), preferring three, whose labels keep 0.05 in apart and stay inside the
    panel plus `room_left` / `room_right` inches, so no label reaches the next panel's labels."""
    span = hi - lo
    e0 = int(math.floor(math.log10(span)))
    for n_want, mults in ((3, (1, 2, 5)), (3, (2.5,)), (2, (1, 2, 5)), (2, (2.5,))):
        for step in sorted(m * 10 ** e for e in range(e0 - 2, e0 + 2) for m in mults):
            ticks = np.arange(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1) * step
            if len(ticks) != n_want:
                continue
            fmt = tick_fmt(step)
            w = [text_width(fig, fmt(t), TICK_PT) for t in ticks]
            x = [(t - lo) / span * PW for t in ticks]
            apart = all(x[i + 1] - x[i] >= (w[i] + w[i + 1]) / 2 + 0.05 for i in range(len(x) - 1))
            inside = x[0] - w[0] / 2 >= -room_left and x[-1] + w[-1] / 2 <= PW + room_right
            if apart and inside:
                return ticks, step
    raise ValueError(f"no tick step for [{lo}, {hi}]")


# ------------------------------------------------------------------ drawing
MSIZE = {"o": 4.0, "s": 3.6, "D": 3.5, "^": 4.7, "v": 4.7}   # equal visual weight across shapes (points)


def msize(marker, hollow=False):
    return MSIZE[marker] * (0.9 if hollow else 1.0)


def draw_row(ax, y, est, key, lim, hollow=False, log=None):
    spec = cs.METHODS[key]
    d, lo, hi, _ = est
    x0, x1 = lim
    sig = lo > 0 or hi < 0
    c = spec["color"] if sig else cs.tint(spec["color"], FADE)
    a, b = max(lo, x0), min(hi, x1)
    if b - a < 0.03 * (x1 - x0):   # a sliver at the edge reads as noise: treat the interval as off-axis
        a, b = x1, x0
    if a < b:
        ax.plot([a, b], [y, y], color=c, lw=0.8 if hollow else 1.0, solid_capstyle="butt", zorder=3)
    if x0 <= d <= x1:
        kw = dict(mfc="white", mec=c, mew=0.85) if hollow else dict(mfc=c, mec="white", mew=0.4)
        ax.plot([d], [y], marker=spec["marker"], ms=msize(spec["marker"], hollow), ls="none", zorder=5, **kw)
        for side, beyond in ((x0, lo < x0), (x1, hi > x1)):   # interval runs past the axis limit
            if beyond:
                ax.plot([side], [y], marker="<" if side == x0 else ">", ms=3.0, color=c, mec="none",
                        clip_on=False, zorder=4)
                if log is not None:
                    log.append(f"{key} interval past the edge")
        return
    # point estimate beyond the axis limit: an outward triangle at the limit, no value
    left = d < x0
    ax.plot([x0 if left else x1], [y], marker="<" if left else ">", ms=4.2, color=c, mec="white", mew=0.4,
            clip_on=False, zorder=5)
    if log is not None:
        log.append(f"{key} {'left' if left else 'right'}{' (partial interval)' if a < b else ''}")


def draw_panel(fig, ax, res, k, room_left, room_right):
    lim = limits(res, k)
    ax.set_xlim(*lim)
    ax.set_ylim(*YLIM)
    for y in list(POS.values()) + list(SUBPOS.values()):           # row guides: the only gridlines
        ax.axhline(y, color=cs.GRIDC, lw=0.4, zorder=1)
    ax.axvline(0.0, color=cs.METHOD, lw=0.6, ls=(0, (3.0, 1.8)), zorder=2)   # reference: SEGA (released)
    log = []
    for m in ORDER:
        draw_row(ax, POS[m], res[(m, k, 0)], m, lim, log=log)
        if m in SEED1:
            draw_row(ax, SUBPOS[m], res[(m, k, 1)], m, lim, hollow=True, log=log)
    ticks, step = nice_ticks(fig, *lim, room_left, room_right)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(tick_fmt(step)))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color(cs.INK2)
    ax.tick_params(axis="x", length=2.5, width=0.6, pad=1.8, labelsize=TICK_PT, color=cs.INK2,
                   labelcolor=cs.INK)
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.grid(False)
    # the metric name is the panel's x label, set above the panel (the audit rejects axes titles)
    ax.set_xlabel(cs.metric_label(k), fontsize=TITLE_PT, color=cs.INK, labelpad=2.5)
    ax.xaxis.set_label_position("top")
    print(f"  {k:22s} lim [{lim[0]:+.4f}, {lim[1]:+.4f}]  ticks {[tick_fmt(step)(t) for t in ticks]}"
          + (f"  edge: {'; '.join(log)}" if log else ""))


def main():
    cs.style()
    res = compute()
    fig = plt.figure(figsize=(W, H))
    fig.canvas.draw()
    label_w = LEFT + max(text_width(fig, cs.label(m), LABEL_PT) for m in ORDER) + LABEL_GAP
    global PW
    PW = (W - label_w - 4 * GUT - RIGHT) / 5
    print(f"label column {label_w:.3f} in, panel {PW:.3f} x {PH:.3f} in")

    for r, keys in enumerate(LAYOUT):
        ax_top = TOP + r * (TITLE + PH + TICKS + GAP) + TITLE          # inches from the top of the figure
        axes = []
        for j, k in enumerate(keys):
            x0 = label_w + j * (PW + GUT)
            ax = fig.add_axes([x0 / W, 1 - (ax_top + PH) / H, PW / W, PH / H],
                              sharey=axes[0] if axes else None)
            room_left = GUT / 2 - 0.01 if j else 0.15
            room_right = GUT / 2 - 0.01 if j < len(keys) - 1 else RIGHT - 0.005
            draw_panel(fig, ax, res, k, room_left, room_right)
            axes.append(ax)
        # methods as rows: the first panel's categorical y tick labels name the rows of the whole row of panels
        first = axes[0]
        first.set_yticks([LABEL_Y[m] for m in ORDER])
        first.set_yticklabels([cs.label(m) for m in ORDER])
        first.tick_params(axis="y", length=0, pad=LABEL_GAP * 72, labelleft=True, labelsize=LABEL_PT,
                          labelcolor=cs.INK)
        # The row names are the shared y axis's label. orx_figstyle's audit accepts a shared axis only through an
        # axis-label string on a sibling, so the first panel carries that string undrawn (the critique deletes the
        # visible "Method" header); nothing is added to the printed figure.
        first.set_ylabel("Method")
        first.yaxis.label.set_visible(False)
    cs.finish(fig, cs.HERE / "f3_forest")


if __name__ == "__main__":
    main()
