"""Held-out comparison across evaluation roots: per-image metrics + ImageReward on prompts with index >= START,
method means with bootstrap 95% CIs and paired deltas vs a reference method on common prompts.

usage: heldout_analysis.py EVAL_ROOT PROMPTS OUT_DIR [--start 8] [--ref sega_official]
EVAL_ROOT holds {main,mainsb,main2}/per_image.csv and ir/{main,mainsb,main2}.csv.
mainsb methods get suffix _sbase; optional main2 methods get suffix _v2.
"""
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd

METRICS = ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024", "clipiqa", "clipiqa_crop1024",
           "spectrum_log_rmse", "spectrum_slope_error", "tiling1024"]
LOWER_BETTER = {"spectrum_log_rmse", "spectrum_slope_error", "tiling1024"}


def load(eval_root, name, suffix):
    d = pd.read_csv(os.path.join(eval_root, name, "per_image.csv"))
    ir = pd.read_csv(os.path.join(eval_root, "ir", f"{name}.csv"))
    keys = ["method", "prompt_id", "res"]
    d = d.merge(ir[keys + ["imagereward", "png"]], on=keys, how="left", validate="one_to_one")
    matched = d.png.notna()
    if not (d.loc[matched, "path"].map(lambda p: str(Path(p).resolve())) ==
            d.loc[matched, "png"].map(lambda p: str(Path(p).resolve()))).all():
        raise ValueError(f"{name}: evaluation and ImageReward scored different image paths")
    d["method"] = d["method"] + suffix
    return d


def select_heldout(d, order, start, end, reference):
    """Exclude every dev row and require complete, finite v2/reference pairs."""
    if start < 8 or end <= start or end > len(order):
        raise ValueError("Held-out interval must exclude dev rows 0-7 and fit the prompt manifest")
    d = d.copy()
    required = [reference] + [m for m in d.method.unique() if m.endswith("_v2")]
    d["idx"] = d.prompt_id.map(order)
    if d.idx.isna().any():
        raise ValueError("Evaluation contains an unknown prompt ID")
    d = d[(d.idx >= start) & (d.idx < end)].copy()
    if d.duplicated(["method", "prompt_id"]).any():
        raise ValueError("Duplicate held-out method/prompt pairs")
    expected = {pid for pid, index in order.items() if start <= index < end}
    for method in required:
        rows = d[d.method == method]
        if set(rows.prompt_id) != expected:
            raise ValueError(f"{method}: incomplete held-out prompt coverage")
        if not ((rows.seed == 0) & (rows.res == "4096x4096")).all():
            raise ValueError(f"{method}: expected seed 0 at 4096x4096")
        if not np.isfinite(rows[METRICS].to_numpy(dtype=float)).all():
            raise ValueError(f"{method}: nonfinite or missing held-out metrics")
    return d


def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("eval_root"); p.add_argument("prompts"); p.add_argument("out")
    p.add_argument("--start", type=int, default=8); p.add_argument("--ref", default="sega_official")
    p.add_argument("--end", type=int, default=195, help="Exclusive prompt row; default 195")
    p.add_argument("--require-main2", action="store_true", help="Fail if the selected v2 evaluation is absent")
    a = p.parse_args()
    order = {json.loads(l)["id"]: i for i, l in enumerate(open(a.prompts))}
    roots = [("main", ""), ("mainsb", "_sbase")]
    has_main2 = Path(a.eval_root, "main2", "per_image.csv").is_file()
    if a.require_main2 and not has_main2:
        raise ValueError("Required main2 evaluation is absent")
    if has_main2:
        roots.append(("main2", "_v2"))
    d = pd.concat([load(a.eval_root, name, suffix) for name, suffix in roots], ignore_index=True)
    if has_main2 and not d.method.str.endswith("_v2").any():
        raise ValueError("main2 contains no candidate rows")
    d = select_heldout(d, order, a.start, a.end, a.ref)
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
    d.to_csv(os.path.join(a.out, "heldout_per_prompt.csv"), index=False)
    Path(a.out, "analysis_protocol.json").write_text(json.dumps({
        "start_inclusive": a.start, "end_exclusive": a.end, "seed": 0,
        "reference": a.ref, "roots": roots, "paired_bootstrap_draws": 2000,
        "bootstrap_seed": 0, "ci_percentiles": [2.5, 97.5],
        "purpose": "Held-out paired comparisons; dev rows 0-7 excluded",
    }, indent=2) + "\n")
    t = pd.DataFrame(rows).pivot_table(index="method", columns="metric", values="mean")[METRICS]
    pd.set_option("display.width", 250); print(t.round(3).to_string()); print()
    dd = pd.DataFrame(deltas)
    show = dd.method.isin(["temp_only_sbase", "constancy_tm_sbase", "dype", "ntk2"]) | dd.method.str.endswith("_v2")
    print(dd[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
