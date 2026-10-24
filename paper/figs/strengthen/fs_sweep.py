"""FS2: global temperature sweep on the SEGA base (P3).

Regenerate from the project root:
    OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python paper/figs/strengthen/fs_sweep.py

Input (work, read only): .agent/notes/strengthen/analysis/P3/contrasts_holm.csv from scripts/strengthen_analysis.py
(held-out rows 8-62, seed 0; each multiplier minus the SEGA reference s^0.16 = 4**0.16 = 1.248; paired prompt bootstrap,
2,000 draws, seed 0, nominal 95% CI). Nothing is recomputed. The five multipliers were fixed before generation; the
dashed vertical marks the reference, not a selected value. A mark is pale when its interval covers zero.
"""
import argparse
import json
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
P3 = ROOT / ".agent/notes/strengthen/analysis/P3"
TAU = {"tau1p0": 1.0, "tau1p1": 1.1, "tau_sref": 4 ** 0.16, "tau_logn": 4 / 3, "tau1p45": 1.45}
PANELS = ["imagereward", "pickscore", "musiq", "musiq_crop1024"]
FADE = 0.60


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, default=P3)
    ap.add_argument("--stem", default=str(HERE / "fs_sweep"))
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    proto = json.loads((a.analysis / "analysis_protocol.json").read_text())
    if not proto.get("complete") and not a.allow_incomplete:
        raise SystemExit("P3 analysis incomplete; refusing to draw")
    c = pd.read_csv(a.analysis / "contrasts_holm.csv")
    c = c[c.analysis == "seed0"]
    cs.style()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(cs.TEXT, 1.6))
    for ax, metric in zip(axes, PANELS):
        xs, ys = [], []
        for name, x in sorted(TAU.items(), key=lambda kv: kv[1]):
            if name == "tau_sref":
                xs.append(x), ys.append(0.0)
                ax.plot([x], [0.0], "o", ms=3.5, mfc="white", mec=cs.TEMP, mew=1.0, zorder=3)
                continue
            r = c[(c.contrast == name) & (c.metric == metric)]
            if r.empty or pd.isna(r.iloc[0]["lo"]):
                continue
            r = r.iloc[0]
            alpha = FADE if r["lo"] <= 0 <= r["hi"] else 1.0
            ax.plot([x, x], [r["lo"], r["hi"]], color=cs.TEMP, lw=1.0, alpha=alpha, solid_capstyle="butt")
            ax.plot([x], [r["delta"]], "o", color=cs.TEMP, ms=3.5, alpha=alpha, mec="none", zorder=3)
            xs.append(x), ys.append(r["delta"])
        ax.plot(xs, ys, color=cs.TEMP, lw=0.8, alpha=0.8, zorder=2)
        ax.axhline(0, color=cs.INK2, lw=0.6, ls=(0, (3.0, 1.8)), zorder=0)
        ax.axvline(4 ** 0.16, color=cs.RULE, lw=0.6, ls=(0, (3.0, 1.8)), zorder=0)
        cs.clean_axes(ax, grid="y")
        ax.set_xlabel("Logit multiplier")
        ax.set_ylabel("Δ " + cs.metric_label(metric))
        ax.set_xticks([1.0, 1.2, 1.4])
        ax.locator_params(axis="y", nbins=4)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.26, top=0.96, wspace=0.6)
    cs.finish(fig, a.stem)


if __name__ == "__main__":
    main()
