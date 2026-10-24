"""FS1: factorial decomposition of released SEGA on held-out prompts (P1).

Regenerate from the project root:
    OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python paper/figs/strengthen/fs_factorial.py

Input (work, read only): .agent/notes/strengthen/analysis/P1/contrasts_holm.csv from scripts/strengthen_analysis.py
(held-out rows 8-194, seed 0, paired prompt bootstrap, 2,000 draws, seed 0, nominal 95% CI). Nothing is recomputed.
Rows: base main effect (SEGA base minus NTK base, averaged over temperature), temperature main effect (s^0.16 minus
none, averaged over base), base x temperature interaction, and the allocation contrast (SEGA released minus SEGA
base + temp.). Panels: the three primary metrics. A mark is pale when its interval covers zero. Refuses to draw an
incomplete block unless --allow-incomplete is given (layout tests write to the scratchpad only).
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import crope_style as cs

ROOT = HERE.parents[2]
P1 = ROOT / ".agent/notes/strengthen/analysis/P1"
ROWS = [("base_main_effect", "Base", cs.INK), ("temperature_main_effect", "Temperature", cs.TEMP),
        ("interaction", "Base × temp.", cs.GRAY), ("allocation", "Allocation", cs.METHOD)]
PANELS = ["imagereward", "pickscore", "musiq"]
FADE = 0.60


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, default=P1)
    ap.add_argument("--stem", default=str(HERE / "fs_factorial"))
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    import json
    proto = json.loads((a.analysis / "analysis_protocol.json").read_text())
    if not proto.get("complete") and not a.allow_incomplete:
        raise SystemExit("P1 analysis incomplete; refusing to draw")
    c = pd.read_csv(a.analysis / "contrasts_holm.csv")
    c = c[c.analysis == "seed0"]
    cs.style()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(cs.TEXT, 1.55), sharey=True)
    for ax, metric in zip(axes, PANELS):
        for i, (name, label, color) in enumerate(ROWS):
            r = c[(c.contrast == name) & (c.metric == metric)]
            if r.empty or pd.isna(r.iloc[0]["lo"]):
                continue
            r = r.iloc[0]
            y = len(ROWS) - 1 - i
            alpha = FADE if r["lo"] <= 0 <= r["hi"] else 1.0
            ax.plot([r["lo"], r["hi"]], [y, y], color=color, lw=1.0, alpha=alpha, solid_capstyle="butt")
            ax.plot([r["delta"]], [y], "o", color=color, ms=3.5, alpha=alpha, mec="none")
        ax.axvline(0, color=cs.INK2, lw=0.6, ls=(0, (3.0, 1.8)), zorder=0)
        cs.clean_axes(ax, grid="x")
        ax.set_xlabel("Δ " + cs.metric_label(metric))
        ax.locator_params(axis="x", nbins=4)
    axes[0].set_yticks(range(len(ROWS)))
    axes[0].set_yticklabels([r[1] for r in ROWS][::-1])
    axes[0].set_ylabel("Contrast")
    axes[0].set_ylim(-0.6, len(ROWS) - 0.4)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.27, top=0.97, wspace=0.12)
    cs.finish(fig, a.stem)


if __name__ == "__main__":
    main()
