"""Held-out comparison across evaluation roots: per-image metrics + ImageReward on prompts with index >= START,
method means with bootstrap 95% CIs and paired deltas vs a reference method on common prompts.

usage: heldout_analysis.py EVAL_ROOT PROMPTS OUT_DIR [--start 8] [--ref sega_official]
EVAL_ROOT holds {main,mainsb,...}/per_image.csv and ir/{main,mainsb,...}.csv; mainsb methods get the suffix _sbase.
"""
import argparse, json, os
import numpy as np, pandas as pd

METRICS = ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024", "clipiqa", "clipiqa_crop1024",
           "spectrum_log_rmse", "spectrum_slope_error", "tiling1024"]
LOWER_BETTER = {"spectrum_log_rmse", "spectrum_slope_error", "tiling1024"}


def load(eval_root, name, suffix):
    d = pd.read_csv(os.path.join(eval_root, name, "per_image.csv"))
    ir = pd.read_csv(os.path.join(eval_root, "ir", f"{name}.csv"))
    d = d.merge(ir[["method", "prompt_id", "imagereward"]], on=["method", "prompt_id"], how="left")
    d["method"] = d["method"] + suffix
    return d


def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("eval_root"); p.add_argument("prompts"); p.add_argument("out")
    p.add_argument("--start", type=int, default=8); p.add_argument("--ref", default="sega_official")
    a = p.parse_args()
    order = {json.loads(l)["id"]: i for i, l in enumerate(open(a.prompts))}
    d = pd.concat([load(a.eval_root, "main", ""), load(a.eval_root, "mainsb", "_sbase")], ignore_index=True)
    d["idx"] = d.prompt_id.map(order); d = d[d.idx >= a.start]
    os.makedirs(a.out, exist_ok=True)
    rows, deltas = [], []
    for m, g in d.groupby("method"):
        for k in METRICS:
            v = g[k].dropna()
            if len(v):
                mu, lo, hi = boot(v); rows.append(dict(method=m, metric=k, n=len(v), mean=mu, lo=lo, hi=hi))
    ref = d[d.method == a.ref].set_index("prompt_id")
    for m, g in d.groupby("method"):
        if m == a.ref:
            continue
        g = g.set_index("prompt_id"); common = g.index.intersection(ref.index)
        for k in METRICS:
            diff = (g.loc[common, k] - ref.loc[common, k]).dropna()
            if len(diff):
                mu, lo, hi = boot(diff)
                better = (mu < 0) if k in LOWER_BETTER else (mu > 0)
                sig = (hi < 0 or lo > 0)
                deltas.append(dict(method=m, metric=k, n=len(diff), delta=mu, lo=lo, hi=hi,
                                   verdict=("better" if better else "worse") if sig else "tie"))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "heldout_means.csv"), index=False)
    pd.DataFrame(deltas).to_csv(os.path.join(a.out, "heldout_paired_vs_" + a.ref + ".csv"), index=False)
    t = pd.DataFrame(rows).pivot_table(index="method", columns="metric", values="mean")[METRICS]
    pd.set_option("display.width", 250); print(t.round(3).to_string()); print()
    dd = pd.DataFrame(deltas)
    print(dd[dd.method.isin(["temp_only_sbase", "constancy_tm_sbase", "dype", "ntk2"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
