"""FS3: registered confirmation of the decomposition (H1 equivalence, H2 superiority) with its replications.

Regenerate from the project root:
    OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python paper/figs/strengthen/fs_confirm.py

Inputs (work, read only), all written by scripts/strengthen_analysis.py; nothing is recomputed here:
  .agent/notes/strengthen/analysis/E1/confirmation_family.csv   pooled 337-prompt H1 and fresh-150 H2 (registered)
  .agent/notes/strengthen/analysis/E1/hypotheses.json           fresh-150 estimates (seed 0)
  .agent/notes/strengthen/analysis/E2/hypotheses.json           held-out 187 at seed 0, and prompt means over seeds 0-2
Panels: ImageReward and PickScore show SEGA base + temp. minus SEGA (released) with the registered equivalence
margin as a gray band; full-frame MUSIQ shows SEGA (released) minus SEGA base + temp. (H2 direction). Filled marks
are the registered tests (pooled H1, fresh H2); hollow marks are the separately reported populations. Intervals are
nominal 95% paired prompt-bootstrap (2,000 draws, seed 0). Refuses an incomplete E1 or E2 unless --allow-incomplete.
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
ANALYSIS = ROOT / ".agent/notes/strengthen/analysis"
# (label, source, hypotheses-json analysis key or None for the pooled family)
ROWS = [("Pooled", "E1", None), ("Fresh", "E1", "seed0"),
        ("Held-out", "E2", "seed0"), ("Held-out, 3 seeds", "E2", "seeds012")]
PANELS = [("H1_IR", "imagereward", "Δ ImageReward"), ("H1_Pick", "pickscore", "Δ PickScore"),
          ("H2_MUSIQ", "musiq", "Δ MUSIQ, allocation")]


def family(analysis, block, key):
    if key is None:
        frame = pd.read_csv(analysis / block / "confirmation_family.csv")
        return {r["hypothesis"]: r for r in frame.to_dict("records")}
    rows = json.loads((analysis / block / "hypotheses.json").read_text())[key]["family"]
    return {r["hypothesis"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, default=ANALYSIS)
    ap.add_argument("--stem", default=str(HERE / "fs_confirm"))
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--png", action="store_true", help="also write a PNG preview")
    a = ap.parse_args()
    for block in ("E1", "E2"):
        proto = json.loads((a.analysis / block / "analysis_protocol.json").read_text())
        if not proto.get("complete") and not a.allow_incomplete:
            raise SystemExit(f"{block} analysis incomplete; refusing to draw")
    data = [(label, family(a.analysis, block, key)) for label, block, key in ROWS]
    cs.style()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(cs.TEXT, 1.45), sharey=True)
    for ax, (hyp, metric, xlabel) in zip(axes, PANELS):
        color = cs.METHOD if hyp == "H2_MUSIQ" else cs.TEMP
        margin = None
        for i, (label, fam) in enumerate(data):
            r = fam.get(hyp)
            if r is None or pd.isna(r.get("lo")):
                continue
            y = len(data) - 1 - i
            registered = (label == "Pooled" and hyp != "H2_MUSIQ") or (label == "Fresh" and hyp == "H2_MUSIQ")
            if label == "Pooled" and hyp == "H2_MUSIQ":
                continue  # H2 is fresh-only; the pooled family repeats the fresh estimate
            ax.plot([r["lo"], r["hi"]], [y, y], color=color, lw=1.0, solid_capstyle="butt")
            ax.plot([r["delta"]], [y], "o", ms=3.6, color=color, mfc=color if registered else "white", mew=0.9)
            if r.get("margin") is not None and not pd.isna(r.get("margin")):
                margin = float(r["margin"])
        if margin is not None:
            # Margin band with visible edges; the x range always shows both edges and every interval.
            ax.axvspan(-margin, margin, color=cs.PANEL, zorder=0, lw=0)
            for edge in (-margin, margin):
                ax.axvline(edge, color=cs.GRAY, lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
            reach = max([margin] + [abs(v) for _, fam in data for k in ("lo", "hi")
                                    for v in [fam.get(hyp, {}).get(k)] if v is not None and not pd.isna(v)])
            ax.set_xlim(-1.18 * reach, 1.18 * reach)
        ax.axvline(0, color=cs.INK2, lw=0.6, ls=(0, (3.0, 1.8)), zorder=1)
        cs.clean_axes(ax, grid="x")
        ax.set_xlabel(xlabel)
        ax.locator_params(axis="x", nbins=4)
    axes[0].set_yticks(range(len(data)))
    axes[0].set_yticklabels([d[0] for d in data][::-1])
    axes[0].set_ylim(-0.6, len(data) - 0.4)
    axes[0].set_ylabel("Prompts")
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.20, right=0.99, bottom=0.28, top=0.97, wspace=0.14)
    cs.finish(fig, a.stem, formats=("pdf", "svg", "png") if a.png else None)


if __name__ == "__main__":
    main()
