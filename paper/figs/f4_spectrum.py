"""Figure 4 (main text, Section 5.2) and its appendix version: the registered confirmation.

The file names (f4_spectrum.pdf, fA_spectrum_full.pdf) were fixed by the integration plan; the content is
the registered-confirmation figure (label fig:fs-confirm), not the luminance spectrum (Figure 3,
f8_spectrum_main.pdf, unchanged).

f4_spectrum.pdf (2.75 x 1.83 in, a right wrapfigure included at \\linewidth inside {0.5\\linewidth}): the two
reward panels only, ImageReward and PickScore, both as SEGA base + temp. minus SEGA (released); the sign is
the header above both panels. Rows: Pooled (337 prompts, the registered H1 test, filled), Fresh (150),
Held-out (187, seed 0) and Held-out averaged over seeds 0 to 2 (hollow). The gray band with dotted edges is
the registered equivalence margin (+/-0.082 ImageReward, +/-0.10 PickScore); the dashed line is zero.

fA_spectrum_full.pdf (5.50 x 1.62 in, appendix, \\linewidth): the same two panels plus full-frame MUSIQ as
SEGA (released) minus SEGA base + temp. (the allocation, registered H2 on the 150 fresh prompts, filled);
each sign is written in a header above its panels, so the flip between reward and MUSIQ panels is explicit.

Inputs (read only), written by scripts/strengthen_analysis.py and read through strengthen/fs_confirm.py,
so nothing is recomputed: .agent/notes/strengthen/analysis/E1/confirmation_family.csv (pooled H1, fresh H2),
E1/hypotheses.json (fresh, seed 0) and E2/hypotheses.json (held-out seed 0, and seeds 0 to 2). Intervals are
nominal 95% paired prompt-bootstrap intervals (2,000 draws, RNG seed 0). Every plotted estimate is checked
against the paper macros in tables/num_strengthen.tex where one exists.

Regenerate: OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python f4_spectrum.py
"""
from __future__ import annotations

import json
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.pyplot as plt
import pandas as pd

import crope_style as cs
import orx_figstyle

sys.path.insert(0, str(cs.HERE / "strengthen"))
import fs_confirm as fc   # noqa: E402  (data access of the original confirmation figure)

MACROS = cs.HERE.parent / "tables" / "num_strengthen.tex"
REWARD = "SEGA base + temp. − SEGA (released)"
ALLOC = "SEGA (released) − SEGA base + temp."
PANELS = [("H1_IR", "imagereward", "Δ ImageReward", cs.TEMP),
          ("H1_Pick", "pickscore", "Δ PickScore", cs.TEMP),
          ("H2_MUSIQ", "musiq", "Δ MUSIQ", cs.METHOD)]
MACRO_KEY = {"H1_IR": "ir", "H1_Pick": "pick", "H2_MUSIQ": "musiq"}
PRECISION = {"H1_IR": 3, "H1_Pick": 3, "H2_MUSIQ": 2}


def load():
    for block in ("E1", "E2"):
        proto = json.loads((fc.ANALYSIS / block / "analysis_protocol.json").read_text())
        if not proto.get("complete"):
            raise SystemExit(f"{block} analysis incomplete; refusing to draw")
    data = [(label, fc.family(fc.ANALYSIS, block, key)) for label, block, key in fc.ROWS]
    check_macros(dict(data))
    return data


def macro(text, name):
    m = re.search(r"\\csname N@%s\\endcsname\{(.*)\}\s*$" % re.escape(name), text, re.M)
    assert m, f"no macro N@{name}"
    return m[1]


def check_macros(fams):
    """The registered estimates, intervals, margins and n must equal the numbers the text prints."""
    tex = MACROS.read_text()
    for hyp, key in MACRO_KEY.items():
        r, p = fams["Pooled"][hyp], PRECISION[hyp]
        assert macro(tex, f"st@h@d@{key}") == f"${r['delta']:+.{p}f}$", hyp
        assert macro(tex, f"st@h@ci@{key}") == f"[${r['lo']:+.{p}f}$, ${r['hi']:+.{p}f}$]", hyp
        assert int(macro(tex, f"st@h@n@{key}")) == int(r["n"]), hyp
    assert macro(tex, "st@h@m@ir") == f"{fams['Pooled']['H1_IR']['margin']:.3f}"
    assert macro(tex, "st@h@m@pick") == f"{fams['Pooled']['H1_Pick']['margin']:.2f}"
    for hyp, key, p in (("H1_IR", "ir", 3), ("H1_Pick", "pick", 2)):   # held-out, seeds 0 to 2
        r = fams["Held-out, 3 seeds"][hyp]
        assert macro(tex, f"st@rseeds012@d@tsb@{key}") == f"${r['delta']:+.{p}f}$", hyp
        assert macro(tex, f"st@rseeds012@ci@tsb@{key}") == f"[${r['lo']:+.{p}f}$, ${r['hi']:+.{p}f}$]", hyp
    print("macros: registered estimates, intervals, margins and n match tables/num_strengthen.tex")


def panel(ax, hyp, color, data, ms):
    margin = None
    for i, (label, fam) in enumerate(data):
        r = fam.get(hyp)
        if r is None or pd.isna(r.get("lo")) or (label == "Pooled" and hyp == "H2_MUSIQ"):
            continue            # H2 is fresh-only; the pooled family repeats the fresh estimate
        y = len(data) - 1 - i
        registered = (label == "Pooled") if hyp != "H2_MUSIQ" else (label == "Fresh")
        ax.plot([r["lo"], r["hi"]], [y, y], color=color, lw=1.0, solid_capstyle="butt", zorder=3)
        ax.plot([r["delta"]], [y], "o", ms=ms, color=color, mfc=color if registered else "white", mew=0.9,
                zorder=4)
        if r.get("margin") is not None and not pd.isna(r.get("margin")):
            margin = float(r["margin"])
    if margin is not None:
        ax.axvspan(-margin, margin, color=cs.PANEL, zorder=0, lw=0)
        for edge in (-margin, margin):
            ax.axvline(edge, color=cs.GRAY, lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
        reach = max([margin] + [abs(v) for _, fam in data for k in ("lo", "hi")
                                for v in [fam.get(hyp, {}).get(k)] if v is not None and not pd.isna(v)])
        ax.set_xlim(-1.18 * reach, 1.18 * reach)
    ax.axvline(0, color=cs.INK2, lw=0.6, ls=(0, (3.0, 1.8)), zorder=2)
    cs.clean_axes(ax, grid="x")
    ax.set_ylim(-0.6, len(data) - 0.4)


def header(fig, axes, text, gap=0.07, rule_gap=0.035):
    """Signed header above a group of panels: 8 pt text over a thin rule spanning the group (as a booktabs
    cmidrule), centred on the group and clamped inside the canvas."""
    W, H = fig.get_size_inches()
    x0, x1 = axes[0].get_position().x0 * W, axes[-1].get_position().x1 * W
    top = axes[0].get_position().y1 * H
    y_rule = top + rule_gap
    fig.add_artist(matplotlib.lines.Line2D([x0 / W, x1 / W], [y_rule / H] * 2, transform=fig.transFigure,
                                           color=cs.RULE, lw=0.6, solid_capstyle="butt"))
    t = fig.text((x0 + x1) / 2 / W, (y_rule + gap) / H, text, ha="center", va="baseline", fontsize=cs.FS_LABEL,
                 color=cs.INK)
    r = fig.canvas.get_renderer()
    bb = t.get_window_extent(r)
    over = bb.x1 / fig.dpi - (W - 0.02)
    if over > 0:
        t.set_x(((x0 + x1) / 2 - over) / W)
    return t


def finish_at_width(fig, stem):
    """cs.finish at a wrapfigure width: orx_figstyle's audit knows 3.25 / 5.50 / 6.75 in, so the known
    column width is set to this figure's width for the audit only (as f8_spectrum.py does)."""
    saved = orx_figstyle.COLUMN
    orx_figstyle.COLUMN = fig.get_figwidth()
    try:
        cs.finish(fig, stem)
    finally:
        orx_figstyle.COLUMN = saved


def axes_row(fig, left, right, bottom, top, ratios, gap):
    """Panels placed in inches: left/right edges of the row, bottom/top of the plot area, width ratios."""
    W, H = fig.get_size_inches()
    unit = (right - left - gap * (len(ratios) - 1)) / sum(ratios)
    axes, x = [], left
    for k, ratio in enumerate(ratios):
        ax = fig.add_axes([x / W, bottom / H, ratio * unit / W, (top - bottom) / H],
                          sharey=axes[0] if axes else None)
        axes.append(ax)
        x += ratio * unit + gap
    return axes


def rows(ax, data, two_line):
    labels = [d[0] for d in data][::-1]
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([lab.replace(", ", ",\n") if two_line else lab for lab in labels], linespacing=1.0)
    ax.set_ylabel("Prompts", labelpad=3)


def wrap(data):
    W, H = cs.TEXT / 2, round(cs.TEXT / 2 / 1.5, 3)          # 2.75 x 1.833 in, about 1.5:1
    fig = plt.figure(figsize=(W, H))
    axes = axes_row(fig, left=0.63, right=W - 0.06, bottom=0.40, top=H - 0.27, ratios=(1, 1), gap=0.12)
    for ax, (hyp, _, xlabel, color) in zip(axes, PANELS[:2]):
        panel(ax, hyp, color, data, ms=3.6)
        ax.set_xlabel(xlabel, labelpad=2)
        ax.locator_params(axis="x", nbins=3)
    rows(axes[0], data, two_line=True)
    axes[1].tick_params(axis="y", length=0, labelleft=False)
    header(fig, axes, REWARD)
    finish_at_width(fig, cs.HERE / "f4_spectrum")


def full(data):
    W, H = cs.TEXT, 1.62
    fig = plt.figure(figsize=(W, H))
    axes = axes_row(fig, left=0.66, right=W - 0.04, bottom=0.40, top=H - 0.27, ratios=(1, 1, 1.3), gap=0.16)
    for ax, (hyp, _, xlabel, color) in zip(axes, PANELS):
        panel(ax, hyp, color, data, ms=3.6)
        ax.set_xlabel(xlabel, labelpad=2)
        ax.locator_params(axis="x", nbins=4)
    rows(axes[0], data, two_line=True)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0, labelleft=False)
    a, b = header(fig, axes[:2], REWARD), header(fig, axes[2:], ALLOC)
    r = fig.canvas.get_renderer()
    assert a.get_window_extent(r).x1 + 0.15 * fig.dpi < b.get_window_extent(r).x0, "headers collide"
    cs.finish(fig, cs.HERE / "fA_spectrum_full")


def main():
    data = load()
    cs.style()
    for label, fam in data:
        print(f"{label:18s}", "  ".join(f"{h} {fam[h]['delta']:+.4f} [{fam[h]['lo']:+.4f}, {fam[h]['hi']:+.4f}]"
                                        for h, *_ in PANELS if h in fam))
    wrap(data)
    full(data)


if __name__ == "__main__":
    main()
