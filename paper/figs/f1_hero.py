"""F1 (page-1 teaser): where released SEGA's 4K gain over the NTK base comes from.

(a) One held-out prompt at 4096 px, seed 0: manifest row 28 (cathedral interior), chosen by
    subject before any output was viewed, and outside rows 8-10 so it does not repeat F4.
    Full frame = released SEGA, Lanczos-reduced. Crops are native pixels at the same two
    locations for every method: A = maximum Laplacian energy of the across-method mean
    luminance (qual_helpers.detail_box), B = image center. 512 px each.
(b) Share of SEGA's gain over NTK: 100 * (mean_X - mean_NTK) / (mean_SEGA - mean_NTK) on the
    prompts each method has, with a 95% percentile interval from 2,000 joint bootstrap
    resamples of prompts (ratio recomputed per resample, numpy default_rng(0)).
    Held-out rows 8-194 (n = 187; Constancy n = 41 on its own prompts). Alias-T exists
    only on dev rows 0-7 (n = 8) and is drawn hollow.

Regenerate: OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f1_hero.py
"""
from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import crope_style as cs
import qual_helpers as qh

ROW = 28
CROP = 512
SHOWN = ["ntk2", "temp_only_sbase", "sega_official", "constancy_tm_sbase"]
METRICS = ["imagereward", "pickscore", "musiq", "musiq_crop1024"]
ROWS = ["temp_only_sbase", "sega_ctm_v2", "constancy_tm_sbase", "constancy", "alias_temp"]
YPOS = {"temp_only_sbase": 0.0, "sega_ctm_v2": 1.4, "constancy_tm_sbase": 2.4, "constancy": 3.4,
        "alias_temp": 4.4}
ROWNAME = {"constancy": "Constancy (n = 41)", "alias_temp": "Alias-T (dev, n = 8)"}
XLIM = (-30, 140)
W, H = cs.TEXT, 3.50
MINUS = "−"


# ---------------------------------------------------------------- numbers
def ratio_ci(a, n_boot=2000, seed=0):
    """a: (n, 3) array [method, NTK, SEGA] per prompt. Share (%) and joint-bootstrap 95% CI."""
    a = np.asarray(a, float)
    pt = 100 * (a[:, 0].mean() - a[:, 1].mean()) / (a[:, 2].mean() - a[:, 1].mean())
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1)
    r = 100 * (m[:, 0] - m[:, 1]) / (m[:, 2] - m[:, 1])
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(pt), float(lo), float(hi), len(a)


def dev_table():
    """Dev rows 0-7: Alias-T plus NTK and released SEGA, ImageReward merged per file."""
    order, keys, ev = cs.prompt_order(), ["method", "prompt_id", "res"], cs.SCR / "eval_krea"

    def load(per_image, ir):
        d = pd.read_csv(ev / per_image).merge(pd.read_csv(ev / ir)[keys + ["imagereward", "png"]],
                                              on=keys, how="left", validate="one_to_one")
        assert (d.path == d.png).all() and (d.seed == 0).all()
        d["idx"] = d.prompt_id.map(order)
        return d

    al = load("dev2_alias_temp/per_image.csv", "ir/dev2_alias_temp.csv")
    mn = load("main/per_image.csv", "ir/main.csv")
    mn = mn[(mn.idx < 8) & mn.method.isin(["ntk2", "sega_official"])]
    d = pd.concat([al, mn])
    assert sorted(d.idx.unique()) == list(range(8)) and len(d) == 24
    return d


def shares():
    held, dev = cs.heldout(), dev_table()
    out, gain = {}, {}
    for met in METRICS:
        w = held.pivot(index="idx", columns="method", values=met)
        g = (w.sega_official - w.ntk2).dropna()
        gain[met] = (float(g.mean()), len(g))
        for m in ROWS:
            src = dev.pivot(index="idx", columns="method", values=met) if m == "alias_temp" else w
            out[m, met] = ratio_ci(src[[m, "ntk2", "sega_official"]].dropna())
    return out, gain


# ---------------------------------------------------------------- drawing helpers
HALO = [pe.withStroke(linewidth=1.8, foreground="#111317")]


def axes_in(fig, x, y, w, h, **kw):
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def photo(ax, img, frame=False):
    ax.imshow(img, interpolation="none")
    ax.axis("off")
    if frame:  # 0.4 pt hairline around 1:1 crops
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=cs.RULE,
                               linewidth=0.4, zorder=5, clip_on=False))


def crop_box(ax, box, scale, letter):
    x0, y0, x1, y1 = (v * scale for v in box)
    r = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=0.8)
    r.set_path_effects([pe.withStroke(linewidth=2.0, foreground="#111317"), pe.Normal()])
    ax.add_patch(r)
    ax.text(x1 + 0.012 * 4096 * scale, y0, letter, color="white", fontsize=7, fontweight="bold",
            ha="left", va="top", path_effects=HALO)


def fmt_gain(g):
    s = f"{abs(g):.2f}" if abs(g) < 10 else f"{abs(g):.1f}"
    return ("+" if g >= 0 else MINUS) + s


def main():
    cs.style()
    vals, gain = shares()
    for (m, met), v in vals.items():
        print(f"{m:20s} {met:16s} share {v[0]:6.1f}  CI [{v[1]:6.1f}, {v[2]:6.1f}]  n={v[3]}")
    for met, (g, n) in gain.items():
        print(f"SEGA - NTK  {met:16s} {g:+.4f}  n={n}")

    src = qh.sources(ROW, SHOWN)
    mean_y = qh.mean_luma([src[m] for m in SHOWN])
    box_a, box_b = qh.detail_box(mean_y, CROP), qh.center_box(CROP)
    print("crop A (detail)", box_a, " crop B (center)", box_b)

    fig = plt.figure(figsize=(W, H))
    # ---------------- (a) photo block: full frame + 2 x 4 native crops, one rectangle
    # gutters: frame to crops G, 2 pt inside a method group (and between rows A, B), GG between groups
    G, GI, GG = 0.13, 2 / 72, 0.07
    F = (W - G - 2 * GG + GI) / 3
    c = (F - GI) / 2
    xs = [F + G, F + G + c + GG, F + G + 2 * c + GG + GI, F + G + 3 * c + 2 * GG + GI]
    top = H - 0.19
    y_img = top - F
    head = H - 0.12
    ax = axes_in(fig, 0, y_img, F, F)
    px = 1024
    photo(ax, cs.thumb(src["sega_official"], px))
    crop_box(ax, box_a, px / 4096, "A")
    crop_box(ax, box_b, px / 4096, "B")
    cs.panel_tag(fig, 0, head / H, "a")
    t = fig.text(0.15 / W, head / H, cs.label("sega_official"), fontsize=7, color=cs.text_color("sega_official"),
                 fontweight="bold", va="baseline")
    fig.canvas.draw()
    x_end = fig.transFigure.inverted().transform(t.get_window_extent().p1)[0]
    fig.text(x_end + 0.04 / W, head / H, "full frame, 4096 px", fontsize=7, color=cs.INK2, va="baseline")
    for j, m in enumerate(SHOWN):
        x = xs[j]
        for i, (box, letter) in enumerate(((box_a, "A"), (box_b, "B"))):
            axc = axes_in(fig, x, top - c - i * (c + GI), c, c)
            photo(axc, cs.crop(src[m], box), frame=True)
            if j == 0:
                axc.text(0.045, 0.955, letter, transform=axc.transAxes, color="white", fontsize=7,
                         fontweight="bold", ha="left", va="top", path_effects=HALO)
        fig.text((x + c / 2) / W, head / H, cs.label(m), color=cs.text_color(m), fontsize=7,
                 fontweight="bold", ha="center", va="baseline")
        fig.add_artist(plt.Line2D([x / W, (x + c) / W], [(H - 0.152) / H] * 2, color=cs.color(m), lw=1.1,
                                  solid_capstyle="butt"))
    fig.add_artist(plt.Line2D([0, F / W], [(H - 0.152) / H] * 2, color=cs.COBALT, lw=1.1,
                              solid_capstyle="butt"))
    fig.text(W / W, (y_img - 0.105) / H, f"A: highest-detail tile, B: image center; native {CROP} px crops "
             "at identical coordinates", ha="right", va="baseline", fontsize=6.5, color=cs.INK2)

    # ---------------- (b) normalized decomposition, one small multiple per metric
    L, gp = 1.10, 0.12
    p = (W - 0.04 - L - 3 * gp) / 4
    name_y = y_img - 0.25
    sub_y = name_y - 0.115
    yb = 0.37
    hb = sub_y - 0.06 - yb
    cs.panel_tag(fig, 0, name_y / H, "b")
    first = None
    for k, met in enumerate(METRICS):
        x0 = L + k * (p + gp)
        ax = axes_in(fig, x0, yb, p, hb) if first is None else axes_in(fig, x0, yb, p, hb, sharex=first,
                                                                          sharey=first)
        first = first or ax
        ax.axvspan(0, 100, color=cs.COBALT, alpha=0.06, lw=0, zorder=0)
        ax.axvline(0, color=cs.STONE, lw=0.8, zorder=1)
        ax.axvline(100, color=cs.COBALT, lw=0.8, zorder=1)
        ax.axhline(0.7, color=cs.RULE, lw=0.5, zorder=1)
        ppu = p * 72 / (XLIM[1] - XLIM[0])  # points per percentage unit
        for m in ROWS:
            pt, lo, hi, n = vals[m, met]
            y, col = YPOS[m], cs.color(m)
            hero, hollow = m == "temp_only_sbase", m == "alias_temp"
            hi_c, lo_c = min(hi, XLIM[1]), max(lo, XLIM[0])
            ax.plot([lo_c, hi_c], [y, y], color=col, lw=1.2, solid_capstyle="butt", zorder=2)
            caps = [v for v, clipped in ((lo, lo < XLIM[0]), (hi, hi > XLIM[1])) if not clipped]
            ax.plot(caps, [y] * len(caps), ls="none", marker="|", ms=3.4, mew=0.9, color=col, zorder=2)
            for edge, mk in ((XLIM[0], "<"), (XLIM[1], ">")):
                if (lo < edge) if mk == "<" else (hi > edge):
                    ax.plot([edge], [y], marker=mk, color=col, ms=3.6, mew=0, zorder=3, clip_on=False)
            ms = 5.4 if hero else 4.5
            ax.plot([pt], [y], marker=cs.METHODS[m]["marker"], ms=ms if hollow is False else 4.2,
                    mfc="white" if hollow else col, mec=col if hollow else "white",
                    mew=1.0 if hollow else 0.6, zorder=4)
            if hero:
                off = max(ms / 2, (pt - lo) * ppu) + 2.5
                ax.annotate(f"{pt:.0f}%", (pt, y), xytext=(-off, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=6.5, color=cs.INK, fontweight="bold")
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelleft=(k == 0))
        fig.text(x0 / W, name_y / H, cs.METRICS[met]["label"], fontsize=7.5, color=cs.INK,
                 fontweight="bold", va="baseline")
        fig.text(x0 / W, sub_y / H, f"SEGA {MINUS} NTK = {fmt_gain(gain[met][0])}", fontsize=6.5,
                 color=cs.INK2, va="baseline")
    first.set_xlim(*XLIM)
    first.set_ylim(4.9, -0.6)
    first.set_xticks([0, 50, 100], ["NTK", "", "SEGA"])
    first.set_yticks([YPOS[m] for m in ROWS], [ROWNAME.get(m, cs.label(m)) for m in ROWS])
    for a in fig.axes[-4:]:
        a.tick_params(axis="x", length=2.0, pad=2.0)
        for t, col in zip(a.get_xticklabels(), (cs.text_color("ntk2"), cs.INK2, cs.text_color("sega_official"))):
            t.set_color(col)
    for t, m in zip(first.get_yticklabels(), ROWS):
        t.set_color(cs.text_color(m))
        t.set_fontweight("bold" if m == "temp_only_sbase" else "normal")
    first.set_ylabel("held-out prompts, n = 187", rotation=0, ha="right", va="baseline", fontsize=6.5,
                     color=cs.INK2)
    first.yaxis.set_label_coords(-3.5 / 72 / p, (sub_y - yb) / hb)
    first.set_xlabel("Share of released SEGA's gain over the NTK base, per metric (higher is better)",
                     fontsize=7.5)
    first.xaxis.set_label_coords((4 * p + 3 * gp) / 2 / p, -(yb - 0.05) / hb)
    first.xaxis.label.set_va("baseline")
    cs.finish(fig, cs.HERE / "f1_hero", photo=True)


if __name__ == "__main__":
    main()
