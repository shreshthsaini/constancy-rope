"""F1 (page-1 hero): released SEGA's 4K gain over the NTK base comes from its global attention temperature; the
RoPE base adds nothing detectable and the per-band allocation adds full-frame MUSIQ. Held-out results only.

(a) Factorial effects on 187 held-out prompts (rows 8-194), seed 0, one small panel per metric (ImageReward,
    PickScore, MUSIQ, MUSIQ crops), each on its own scale. Bars in this order: temperature main effect (s^0.16 minus
    none, averaged over the two bases; green), base main effect (SEGA base minus NTK base, averaged over the two
    temperatures; gray) and allocation (SEGA (released) minus SEGA base + temp.; blue). By construction the three
    add up to the paired gain of SEGA (released) over NTK, drawn as the black zero line. The base x temperature
    interaction is not drawn: it survives Holm on no metric, as the caption says. Bars are paired means, whiskers
    nominal 95% paired prompt-bootstrap intervals (2,000 draws, RNG seed 0), values printed in table precision; an
    asterisk marks Holm-adjusted p <= 0.05 over the 80 registered factorial tests (8 contrasts by 10 metrics).
    Source: analysis_factorial_contrasts_holm.csv on scratch (scripts/strengthen_analysis.py). No number is typed:
    the script checks that CSV against the work copy, recomputes Holm, re-derives every interval from the
    per-prompt means and compares the printed values with the paper macros in tables/num_strengthen.tex.
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

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy

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
STRENGTHEN = cs.SCR / "eval_krea" / "strengthen"
CONTRASTS_CSV = STRENGTHEN / "analysis_factorial_contrasts_holm.csv"   # registered factorial, Holm over 80
PAIRED_CSV = STRENGTHEN / "analysis_factorial_paired.csv"              # each arm minus SEGA (released)
P1 = cs.ROOT / ".agent" / "notes" / "strengthen" / "analysis" / "P1"    # work copy and per-prompt means
MACROS = cs.HERE.parent / "tables" / "num_strengthen.tex"
EFFECTS = (("temperature_main_effect", "temp", "Temperature", cs.TEMP),   # contrast, macro key, legend, hue
           ("base_main_effect", "base", "Base", cs.GRAY),
           ("allocation", "alloc", "Allocation", cs.METHOD))

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
BW = 0.24                      # bar width
S = 0.34                       # bar centre spacing (two printed values side by side stay 0.09 in apart)
STAR = dict(ms=4.6, dx=2.8, dy=6.6)   # Holm star (pt): size, gap from the value's right edge to its centre, height


# ==== numbers (all read from the registered factorial analysis; nothing is typed)
def holm(p):
    """Holm step-down adjustment, as in scripts/strengthen_analysis.py."""
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adjusted


def rederive(c):
    """Re-derive every contrast's mean and interval from the per-prompt means with the analysis bootstrap
    (one fresh default_rng(0) per contrast, 2,000 draws, manifest order)."""
    pm = pd.read_csv(P1 / "prompt_means.csv")
    pm = pm[pm.analysis == "seed0"]
    for r in c.itertuples():
        weights = json.loads(r.coefficients)
        sub = pm[pm.metric == r.metric]
        order = sub[sub.method == next(iter(weights))].prompt_id.tolist()
        piv = sub.pivot(index="prompt_id", columns="method", values="mean").reindex(order)
        v = sum(w * piv[m].to_numpy(np.float64) for m, w in weights.items())
        draws = np.random.default_rng(0).choice(v, (2000, len(v)), replace=True).mean(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5], method="linear")
        assert len(v) == r.n and np.allclose([v.mean(), lo, hi], [r.delta, r.lo, r.hi], rtol=0, atol=1e-9), r


def macro(text, name):
    m = re.search(r"\\csname N@%s\\endcsname\{(.*)\}\s*$" % re.escape(name), text, re.M)
    assert m, f"no macro N@{name} in {MACROS}"
    return m[1]


def factorial():
    """Per metric, (mean, lo, hi, Holm) of the temperature, base and allocation effects."""
    c = pd.read_csv(CONTRASTS_CSV)
    assert set(c.analysis) == {"seed0"} and set(c.family) == {"P1:seed0"}, "one registered family expected"
    assert (c.family_size == len(c)).all() and c.complete_block.all() and (c.n == 187).all()
    assert np.allclose(holm(c.p), c.p_holm, rtol=0, atol=1e-12) and (c.holm_reject == (c.p_holm <= 0.05)).all()
    work = pd.read_csv(P1 / "contrasts_holm.csv").set_index(["contrast", "metric"])
    both = c.set_index(["contrast", "metric"]).join(work, rsuffix="_work", how="outer")
    for col in ("delta", "lo", "hi", "p", "p_holm"):
        assert np.allclose(both[col], both[col + "_work"], rtol=0, atol=1e-12), col
    assert (both.holm_reject == both.holm_reject_work).all()
    rederive(c)
    c = c.set_index(["contrast", "metric"])
    n_inter = int(c.loc["interaction"].holm_reject.sum())
    tex = MACROS.read_text()
    assert int(macro(tex, "st@f@n")) == 187 and int(macro(tex, "st@f@nholm@inter")) == n_inter == 0
    paired = pd.read_csv(PAIRED_CSV)
    paired = paired[paired.analysis == "seed0"].set_index(["method", "reference", "metric"])
    vals = {}
    for met in METRICS:
        prec, key = cs.METRIC_PRECISION[met], MACRO[met]
        rows = [c.loc[(name, met)] for name, *_ in EFFECTS]
        total = -float(paired.loc[("ntk2", "sega_official", met), "delta"])   # SEGA (released) minus NTK
        assert abs(sum(r.delta for r in rows) - total) < 1e-9, met              # the three effects add up
        assert macro(tex, f"st@f@tot@{key}") == f"${total:+.{prec}f}$"
        for (_, short, *_), r in zip(EFFECTS, rows):
            star = "^{*}" if r.holm_reject else ""
            assert macro(tex, f"st@f@d@{short}@{key}") == f"${r.delta:+.{prec}f}{star}$"
            assert macro(tex, f"st@f@ci@{short}@{key}") == f"[${r.lo:+.{prec}f}$, ${r.hi:+.{prec}f}$]"
            assert macro(tex, f"st@f@holm@{short}@{key}") == ("yes" if r.holm_reject else "no")
        vals[met] = [(float(r.delta), float(r.lo), float(r.hi), bool(r.holm_reject)) for r in rows]
        print(f"{met:15s}", "  ".join(f"{s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{'*' if h else ''}"
                                      for (_, s, *_), (d, lo, hi, h) in zip(EFFECTS, vals[met])), f" sum {total:+.4f}")
    print(f"Holm family of {len(c)} tests; the interaction survives on {n_inter} of "
          f"{len(c.loc['interaction'])} metrics; n = {int(c.n.iloc[0])}")
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
    cs.panel_tag(fig, 0, fy(TAG_Y), "a", "Factorial effects")
    handles = [Rectangle((0, 0), 1, 1, fc=hue, ec="none") for *_, hue in EFFECTS]
    fig.legend(handles, [name for _, _, name, _ in EFFECTS], loc="center", bbox_to_anchor=(A_W / 2 / W, fy(LEG_Y)),
               ncol=len(EFFECTS), frameon=False, fontsize=cs.FS_LABEL, handlelength=0.8, handleheight=0.8,
               handletextpad=0.4, columnspacing=1.4, borderpad=0, borderaxespad=0)
    renderer = fig.canvas.get_renderer()
    bottoms = (LAB_Y - LOFF - PH - RGAP, LAB_Y - LOFF)   # lower metric labels share the arm label baseline
    xs = [PW / 2 + k * S for k in (-1, 0, 1)]
    for i, met in enumerate(METRICS):
        row, col = divmod(i, 2)
        x0, bottom = col * (PW + PGAP), bottoms[row]
        ax = axes_at(fig, x0, bottom - PH, PW, PH)
        ax.set_axis_off()
        # Own scale per metric: the zero line sits at the same height in every panel and the
        # highest whisker leaves HEADROOM for its printed value.
        top = max(0.0, *(hi for _, _, hi, _ in vals[met]))
        per_inch = top / (PH * (1 - ZERO_AT) - HEADROOM)
        ax.set_ylim(-ZERO_AT * PH * per_inch, (1 - ZERO_AT) * PH * per_inch)
        ax.set_xlim(0, PW)
        assert min(lo for _, lo, _, _ in vals[met]) > -ZERO_AT * PH * per_inch, met
        precision = cs.METRIC_PRECISION[met]
        for (*_, hue), (mean, lo, hi, star), x in zip(EFFECTS, vals[met], xs):
            ax.bar(x, mean, width=BW, color=hue, lw=0, zorder=2)
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor=cs.INK,
                        elinewidth=0.7, capsize=2.2, capthick=0.7, zorder=4)
            value = ax.annotate(signed(mean, precision), (x, max(hi, 0.0)), xytext=(0, 2.0),
                                textcoords="offset points", ha="center", va="bottom", fontsize=cs.FS_VALUE,
                                color=cs.INK)
            if star:   # Holm-adjusted p <= 0.05: a small star hangs right of the centred value
                half = value.get_window_extent(renderer).width * 36 / fig.dpi
                at = offset_copy(ax.transData, fig=fig, x=half + STAR["dx"], y=STAR["dy"], units="points")
                ax.plot([x], [max(hi, 0.0)], ls="none", marker="*", ms=STAR["ms"], mfc=cs.INK, mec="none",
                        transform=at, clip_on=False, zorder=5)
        # Zero line: the NTK baseline in its own style (black, dashed), as reference lines are drawn paper-wide.
        ax.plot([xs[0] - BW / 2 - 0.07, xs[-1] + BW / 2 + 0.07], [0, 0], color=cs.INK, lw=0.8,
                ls=(0, (3.0, 1.8)), dash_capstyle="butt", zorder=1)   # behind the bars
        fig.text((x0 + PW / 2) / W, fy(bottom + LOFF), cs.metric_label(met), fontsize=cs.FS_LABEL,
                 color=cs.INK, ha="center", va="baseline")


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
    vals = factorial()
    fig = plt.figure(figsize=(W, H))
    panel_a(fig, vals)
    panel_b(fig)
    print(f"figure size {W:.2f} x {H:.3f} in")
    cs.finish(fig, cs.HERE / "f1_hero", photo=True)


if __name__ == "__main__":
    main()
