#!/usr/bin/env python
"""Every results table and every result number cited in the paper text.

Run from the project root:
    OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python paper/tables/make_tables.py

Inputs.
  Held-out, seed 0 (prompt rows 8-194, n = 187): the archived per-prompt table written by
  scripts/heldout_analysis.py, .agent/notes/main2/heldout_per_prompt.csv (on work). The registered
  paired CSV beside it is recomputed here and must match to 1e-9.
  Seed 1 (rows 8-62, n = 55), the eight-prompt dev pilot (rows 0-7) and all dev-selection runs:
  scratch evaluation roots eval_krea/<root>/per_image.csv plus ImageReward in eval_krea/ir/<root>.csv.
  Every scratch row used is copied to paper/tables/data/ on the first run; when scratch is gone the
  snapshot is used instead, so the paper rebuilds after a scratch purge.

Statistics. Means and paired deltas use the percentile bootstrap of scripts/heldout_analysis.py:
2,000 resamples of prompts, numpy default_rng(0), 2.5 and 97.5 percentiles, one fresh generator per
(method, metric). A delta is marked better or worse only when its 95 percent interval excludes zero.
Intervals are per metric and not adjusted for multiple comparisons. Stars in the two held-out
delta tables mark retrospective Holm-adjusted centered-bootstrap tests over that table.

Outputs (paper/tables/): tab_results.tex, tab_alloc.tex, tab_seed1_small.tex (main text), tab_main.tex, tab_share.tex, tab_means_full.tex, tab_deltas_full.tex,
tab_seed1.tex, tab_pilot.tex, tab_dev.tex, tab_octave.tex, num_results.tex.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "tables"
SNAP = OUT / "data"
SCR = Path("/scratch/09032/saini_2/constancy_rope")
EVAL = SCR / "eval_krea"
PROMPTS = SCR / "prompts" / "aesthetic4k_eval.jsonl"
HELDOUT = ROOT / ".agent" / "notes" / "main2"

METRICS = ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024", "clipiqa", "clipiqa_crop1024",
           "spectrum_log_rmse", "spectrum_slope_error", "tiling1024"]
KEY = dict(zip(METRICS, ["ir", "pick", "clip", "musiq", "musiqc", "iqa", "iqac", "spec", "slope", "tile"]))
LOWER = {"spectrum_log_rmse", "spectrum_slope_error", "tiling1024"}
PREC = {"imagereward": 3, "pickscore": 2, "clip": 2, "musiq": 2, "musiq_crop1024": 2, "clipiqa": 3,
        "clipiqa_crop1024": 3, "spectrum_log_rmse": 3, "spectrum_slope_error": 3, "tiling1024": 4}
LABEL = {"imagereward": r"ImageReward$\uparrow$", "pickscore": r"PickScore$\uparrow$", "clip": r"CLIP$\uparrow$",
         "musiq": r"MUSIQ$\uparrow$", "musiq_crop1024": r"MUSIQ\textsubscript{crop}$\uparrow$",
         "clipiqa": r"CLIP-IQA$\uparrow$", "clipiqa_crop1024": r"CLIP-IQA\textsubscript{crop}$\uparrow$",
         "spectrum_log_rmse": r"Spec.\ err.$\downarrow$", "spectrum_slope_error": r"Slope err.$\downarrow$",
         "tiling1024": r"Tiling$\downarrow$"}
BANDS = [f"spectrum_log_error_b{i:02d}" for i in range(11)]
HEAD = {"imagereward": r"\makecell[r]{Image-\\Reward$\uparrow$}", "pickscore": r"\makecell[r]{Pick-\\Score$\uparrow$}",
        "clip": r"\makecell[r]{CLIP\\score$\uparrow$}", "musiq": r"\makecell[r]{MUSIQ\\full$\uparrow$}",
        "musiq_crop1024": r"\makecell[r]{MUSIQ\\crop$\uparrow$}", "clipiqa": r"\makecell[r]{CLIP-IQA\\full$\uparrow$}",
        "clipiqa_crop1024": r"\makecell[r]{CLIP-IQA\\crop$\uparrow$}", "spectrum_log_rmse": r"\makecell[r]{Spectral\\error$\downarrow$}",
        "spectrum_slope_error": r"\makecell[r]{Slope\\error$\downarrow$}", "tiling1024": r"\makecell[r]{Tiling\\$\downarrow$}"}

# Held-out method labels in the archived table -> (macro key, display macro)
HO = {"sega_official": ("sega", r"\segaofficial"), "ntk2": ("ntk", r"\ntk"), "temp_only_sbase": ("tsb", r"\tempsb"),
      "dype": ("dype", r"\dype"), "constancy_tm_sbase": ("ctm", r"\constancytm"),
      "sega_ctm_v2": ("sctm", r"\segactm"), "constancy": ("cons", r"\constancy")}
HO_ORDER = ["ntk2", "temp_only_sbase", "dype", "constancy", "constancy_tm_sbase", "sega_ctm_v2"]
HOH = {"sega_official": r"\segaofficialH", "ntk2": r"\ntkH", "temp_only_sbase": r"\tempsbH", "dype": r"\dypeH",
       "constancy_tm_sbase": r"\constancytmH", "sega_ctm_v2": r"\segactmH", "constancy": r"\constancyH"}

macros = []          # (name, value) for num_results.tex
log = []             # stdout summary
HOLM = {}
STORE = {}          # computed intervals reused by the main-text tables


def boot(x, n=2000, seed=0):
    """Identical to scripts/heldout_analysis.py:boot."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def num(v, d, sign=False):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    if s.lstrip("+-").strip("0.") == "":           # rounds to zero: keep the true sign, never "-0" for 0
        s = ("-" if v < 0 else ("+" if sign else "")) + f"{0:.{d}f}"
    return s.replace("-", "-")


def m_(s):
    return "$" + s + "$"


def mac(name, value):
    macros.append((name, value))


def verdict(metric, d, lo, hi):
    if lo <= 0 <= hi:
        return "t"
    return "b" if ((d < 0) if metric in LOWER else (d > 0)) else "w"


def dcell(metric, d, lo, hi, prec=None, holm=False):
    p = PREC[metric] if prec is None else prec
    v = verdict(metric, d, lo, hi)
    point = num(d, p, True) + (r"^{*}" if holm else "")
    return r"\dcell%s{%s}{%s}{%s}" % (v.upper(), point, num(lo, p, True), num(hi, p, True))


def historical_marks(d):
    """Retain the original archived row order, including the partial Constancy arm."""
    full = [(m, k) for m in HO_ORDER for k in METRICS]
    small = [(m, k) for m in HO_ORDER if m != "constancy" for k in
             ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024", "clipiqa"]]
    rows = []
    for family, pairs in [("historical_full60", full), ("historical_condensed30", small)]:
        ps = []
        for method, metric in pairs:
            a = d[d.method == method].set_index("prompt_id")[metric]
            b = d[d.method == "sega_official"].set_index("prompt_id")[metric]
            common = a.index.intersection(b.index)
            x = (a.loc[common] - b.loc[common]).dropna().to_numpy(float)
            draws = np.random.default_rng(0).choice(x, (2000, len(x)), replace=True).mean(1)
            p = (1 + np.count_nonzero(np.abs(draws - x.mean()) >= abs(x.mean()))) / 2001
            ps.append(p)
        order = np.argsort(ps, kind="stable")
        adjusted, previous = np.ones(len(ps)), 0.
        for rank, idx in enumerate(order):
            previous = max(previous, (len(ps) - rank) * ps[idx])
            adjusted[idx] = min(1., previous)
        for (method, metric), p, ph in zip(pairs, ps, adjusted):
            HOLM[(family, method, metric)] = bool(ph <= .05)
            rows.append(dict(family=family, family_size=len(pairs), method=method, metric=metric,
                             p=p, p_holm=ph, holm_reject=bool(ph <= .05)))
        log.append(f"{family}: {sum(adjusted <= .05)}/{len(ps)} Holm marks")
    pd.DataFrame(rows).to_csv(SNAP / "historical_holm.csv", index=False)


def register(prefix, key, metric, d, lo, hi, n, signed=True):
    """Macros: \\<prefix>d{key}{metric}, ci, abs, n; see macros.tex for the accessors."""
    p = PREC[metric]
    k = KEY[metric]
    mac(f"{prefix}@d@{key}@{k}", m_(num(d, p, signed)))
    mac(f"{prefix}@a@{key}@{k}", num(abs(d), p))
    mac(f"{prefix}@ci@{key}@{k}", "[" + m_(num(lo, p, signed)) + ", " + m_(num(hi, p, signed)) + "]")
    mac(f"{prefix}@n@{key}", str(n))
    mac(f"{prefix}@v@{key}@{k}", {"b": "better", "w": "worse", "t": "tie"}[verdict(metric, d, lo, hi)]
        if signed else "")


# ----------------------------------------------------------------------------------------------
# Data loading with scratch snapshot
def prompt_order():
    snap = SNAP / "prompt_order.csv"
    if PROMPTS.is_file():
        order = pd.DataFrame([{"prompt_id": json.loads(l)["id"], "idx": i} for i, l in enumerate(open(PROMPTS))])
        SNAP.mkdir(parents=True, exist_ok=True)
        order.to_csv(snap, index=False)
    order = pd.read_csv(snap)
    return dict(zip(order.prompt_id, order.idx))


def load_root(name, cols_extra=()):
    """per_image + ImageReward for one scratch root, snapshotted to data/<name>.csv."""
    snap = SNAP / f"{name}.csv"
    pi = EVAL / name / "per_image.csv"
    if not pi.is_file():
        pi = EVAL / name / "4096x4096" / "per_image.csv"
    ir = EVAL / "ir" / f"{name}.csv"
    if pi.is_file() and ir.is_file():
        d = pd.read_csv(pi)
        r = pd.read_csv(ir)
        keys = ["method", "prompt_id", "res"]
        d = d.merge(r[keys + ["imagereward", "png"]], on=keys, how="left", validate="one_to_one")
        ok = d.png.notna()
        if not (d.loc[ok, "path"].map(lambda p: os.path.realpath(p)) ==
                d.loc[ok, "png"].map(lambda p: os.path.realpath(p))).all():
            raise ValueError(f"{name}: evaluation and ImageReward scored different images")
        keep = ["method", "prompt_id", "res", "seed", "path", "seconds"] + METRICS + BANDS + list(cols_extra)
        d = d[[c for c in keep if c in d.columns]]
        SNAP.mkdir(parents=True, exist_ok=True)
        d.to_csv(snap, index=False)
    return pd.read_csv(snap)


def select(d, order, lo, hi, seed):
    d = d.copy()
    d["idx"] = d.prompt_id.map(order)
    if d.idx.isna().any():
        raise ValueError("unknown prompt id")
    d = d[(d.idx >= lo) & (d.idx < hi) & (d.seed == seed) & (d.res == "4096x4096")]
    if d.duplicated(["method", "prompt_id"]).any():
        raise ValueError("duplicate method/prompt rows")
    return d


def paired(d, method, ref, metric):
    a = d[d.method == method].set_index("prompt_id")
    b = d[d.method == ref].set_index("prompt_id")
    common = a.index.intersection(b.index)
    diff = (a.loc[common, metric] - b.loc[common, metric]).dropna()
    return boot(diff) + (len(diff),)


# ----------------------------------------------------------------------------------------------
def heldout(order):
    d = pd.read_csv(HELDOUT / "heldout_per_prompt.csv")
    assert set(d.method) == set(HO), sorted(set(d.method))
    assert d.idx.between(8, 194).all() and (d.seed == 0).all()
    reg = pd.read_csv(HELDOUT / "heldout_paired_vs_sega_official.csv")
    worst = 0.0
    res = {}
    for m, g in d.groupby("method", sort=False):
        if m == "sega_official":
            continue
        g = g.set_index("prompt_id")
        ref = d[d.method == "sega_official"].set_index("prompt_id")
        common = g.index.intersection(ref.index)
        for k in METRICS:
            diff = (g.loc[common, k] - ref.loc[common, k]).dropna()
            mu, lo, hi = boot(diff)
            res[(m, k)] = (mu, lo, hi, len(diff))
            row = reg[(reg.method == m) & (reg.metric == k)].iloc[0]
            worst = max(worst, abs(row.delta - mu), abs(row.lo - lo), abs(row.hi - hi))
            assert int(row.n) == len(diff)
    assert worst < 1e-9, worst
    log.append(f"held-out paired deltas reproduce the registered CSV (max abs diff {worst:.1e})")
    means = {}
    for m, g in d.groupby("method", sort=False):
        for k in METRICS:
            means[(m, k)] = boot(g[k].dropna()) + (int(g[k].notna().sum()),)
    return d, res, means


def table_main(res, means):
    cols = ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024", "clipiqa"]
    L = [r"\begin{tabular}{@{}l " + "r" * len(cols) + r"@{}}", r"\toprule",
         r"Method & " + " & ".join(HEAD[c] for c in cols) + r" \\", r"\midrule"]
    L.append(r"\segaofficial{} & " +
             " & ".join(r"\refcell{%s}" % num(means[("sega_official", c)][0], PREC[c]) for c in cols) + r" \\")
    L.append(r"\midrule")
    groups = [("Decomposition of \\segaofficial", ["ntk2", "temp_only_sbase"]),
              ("Other baseline", ["dype"]),
              ("Principled per-band laws (ours)", ["constancy_tm_sbase", "sega_ctm_v2"])]
    for gi, (title, ms) in enumerate(groups):
        L.append(r"\multicolumn{%d}{@{}l}{\emph{%s}} \\[1pt]" % (len(cols) + 1, title))
        for m in ms:
            n = res[(m, cols[0])][3]
            name = HO[m][1] + ("{}" if n == means[("sega_official", cols[0])][3] else r"{}$^{\dagger}$")
            cells = [dcell(c, *res[(m, c)][:3], holm=HOLM[("historical_condensed30", m, c)]) for c in cols]
            L.append(name + " & " + " & ".join(cells) + r" \\")
        if gi < len(groups) - 1:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_main.tex").write_text("\n".join(L) + "\n")


def table_share(d):
    """Share of released SEGA's gain over the NTK base carried by base + global temperature."""
    cols = ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024", "clipiqa"]
    piv = {m: d[d.method == m].set_index("prompt_id") for m in ["ntk2", "temp_only_sbase", "sega_official"]}
    common = piv["sega_official"].index.intersection(piv["ntk2"].index).intersection(piv["temp_only_sbase"].index)
    n = len(common)
    L = [r"\begin{tabular}{@{}l rrr rr r@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Mean ($n=%d$)} & \multicolumn{2}{c}{Paired gain} & \\" % n,
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
         r"Metric & \ntkH & \tempsbH & \segaofficialH & \hdr{base\\+\,temp.} & \hdr{alloc-\\ation} & \hdr{share of\\base\,+\,temp.} \\",
         r"\midrule"]
    for k in cols:
        a = piv["ntk2"].loc[common, k].to_numpy(float)
        b = piv["temp_only_sbase"].loc[common, k].to_numpy(float)
        c = piv["sega_official"].loc[common, k].to_numpy(float)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n, (2000, n))
        ab = (b - a)[idx].mean(1)
        ac = (c - a)[idx].mean(1)
        share_b = ab / ac
        sh = (b - a).mean() / (c - a).mean()
        slo, shi = np.percentile(share_b, [2.5, 97.5])
        p = PREC[k]
        g_bt = boot(b - a)
        g_al = boot(c - b)
        sig_total = verdict(k, *boot(c - a))
        share_txt = (r"%d\%% \tiny[%d, %d]" % (round(100 * sh), round(100 * slo), round(100 * shi))
                     if sig_total == "b" else "n/a")
        L.append(LABEL[k] + " & " + " & ".join(m_(num(x.mean(), p)) for x in (a, b, c)) + " & " +
                 dcell(k, *g_bt) + " & " + dcell(k, *g_al) + " & " + share_txt + r" \\")
        kk = KEY[k]
        mac(f"sh@p@{kk}", str(round(100 * sh)))
        mac(f"sh@ci@{kk}", "[%d, %d]" % (round(100 * slo), round(100 * shi)))
        register("ho", "basetemp", k, *g_bt, n)
        register("ho", "alloc", k, *g_al, n)
        register("ho", "total", k, *boot(c - a), n)
        STORE[("basetemp", k)], STORE[("alloc", k)] = g_bt, g_al
        STORE[("share", k)] = share_txt
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_share.tex").write_text("\n".join(L) + "\n")
    mac("sh@n", str(n))


def table_full(res, means):
    order = ["sega_official"] + HO_ORDER
    head = r"Metric & " + " & ".join(HOH[m] for m in order) + r" \\"
    L = [r"\begin{tabular}{@{}l" + "r" * len(order) + r"@{}}", r"\toprule", head,
         r" & " + " & ".join(r"\tiny $n=%d$" % means[(m, METRICS[0])][3] for m in order) + r" \\", r"\midrule"]
    for k in METRICS:
        p = PREC[k]
        cells = [r"\mcell{%s}{%s}{%s}" % (num(means[(m, k)][0], p), num(means[(m, k)][1], p), num(means[(m, k)][2], p))
                 for m in order]
        L.append(LABEL[k] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_means_full.tex").write_text("\n".join(L) + "\n")
    L = [r"\begin{tabular}{@{}l" + "r" * len(HO_ORDER) + r"@{}}", r"\toprule",
         r"Metric & " + " & ".join(HOH[m] for m in HO_ORDER) + r" \\",
         r" & " + " & ".join(r"\tiny $n=%d$" % res[(m, METRICS[0])][3] for m in HO_ORDER) + r" \\", r"\midrule"]
    for k in METRICS:
        L.append(LABEL[k] + " & " + " & ".join(
            dcell(k, *res[(m, k)][:3], holm=HOLM[("historical_full60", m, k)]) for m in HO_ORDER) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_deltas_full.tex").write_text("\n".join(L) + "\n")
    mac("ho@ncomp", str(len(HO_ORDER) * len(METRICS)))
    for (m, k), (mu, lo, hi, n) in res.items():
        register("ho", HO[m][0], k, mu, lo, hi, n)
    for (m, k), (mu, lo, hi, n) in means.items():
        p = PREC[k]
        mac(f"hom@{HO[m][0]}@{KEY[k]}", m_(num(mu, p)))
        mac(f"homci@{HO[m][0]}@{KEY[k]}", "[" + m_(num(lo, p)) + ", " + m_(num(hi, p)) + "]")


def contrasts(d):
    """Extra paired contrasts cited in the text."""
    for key, a, b in [("ctmtsb", "constancy_tm_sbase", "temp_only_sbase"),
                      ("sctmtsb", "sega_ctm_v2", "temp_only_sbase"),
                      ("consntk", "constancy", "ntk2"),
                      ("dypetsb", "dype", "temp_only_sbase")]:
        for k in METRICS:
            mu, lo, hi, n = paired(d, a, b, k)
            register("ho", key, k, mu, lo, hi, n)
            STORE[(key, k)] = (mu, lo, hi)
    # SEGA minus NTK on exactly the prompts where Constancy exists (n = 41)
    sub = d[d.prompt_id.isin(set(d[d.method == "constancy"].prompt_id))]
    for k in METRICS:
        register("ho", "total41", k, *paired(sub, "sega_official", "ntk2", k))


def constants():
    """Derived constants of the setting (no data): token counts, scales, schedule shift, temperatures."""
    side_native, side_target, patch = 1024, 4096, 16           # VAE factor 8 x 2x2 patches = 16 px per token
    n_tr, n_tg = (side_native // patch) ** 2, (side_target // patch) ** 2
    s = side_target / side_native
    mac("c@ntrain", f"{n_tr:,}".replace(",", "{,}"))
    mac("c@ntgt", f"{n_tg:,}".replace(",", "{,}"))
    mac("c@tokside", str(side_target // patch))
    mac("c@tokratio", str(n_tg // n_tr))
    mac("c@s", f"{s:g}")
    mac("c@taulogn", f"{np.log(n_tg) / np.log(n_tr):.3f}")
    mac("c@segatemp", f"{s ** 0.16:.3f}")
    mac("c@segaqk", f"{s ** 0.08:.3f}")
    mac("c@ntkbasefac", f"{s ** (2 * 56 / 54):.2f}")  # NTK multiplier for one 56-dim spatial axis, D = 56
    mac("c@segabasediv", f"{1 + 0.1 * np.log(s):.3f}")
    mac("c@dypegain", f"{(1 + 0.1 * np.log(s)) ** 2:.2f}")
    # SEGA paper-equation bin map at 4096^2 on the strengthened NTK base: pairs whose wavelength exceeds
    # the token side all clip to the first frequency bin.
    D, theta = 56, 1e4
    base = theta * s ** (2 * D / (D - 2))
    lam = 2 * np.pi * base ** (2 * np.arange(D // 2) / D)
    mac("c@collapsedpairs", str(int((lam > side_target // patch).sum())))
    mac("c@pairsperaxis", str(D // 2))
    # diffusers FluxPipeline default: mu = base_shift + (max_shift - base_shift) * (L - 256) / (4096 - 256)
    base_shift, max_shift, steps = 0.5, 1.15, 28
    mu_default = base_shift + (max_shift - base_shift) * (n_tg - 256) / (4096 - 256)
    last = 1.0 / steps                                         # last evaluated unshifted sigma (linspace(1, 1/28, 28))
    shift = lambda mu, t: np.exp(mu) / (np.exp(mu) + (1 / t - 1))
    mac("c@mudefault", f"{mu_default:.2f}")
    mac("c@muofficial", f"{max_shift:.2f}")
    mac("c@sigmalastdefault", f"{shift(mu_default, last):.5f}")
    mac("c@sigmalastofficial", f"{shift(max_shift, last):.5f}")
    mac("c@steps", str(steps))


def fid_seed1():
    """Seed-1 paired FID/KID from the evaluator (bootstrap over prompts), to show their noise at n = 55."""
    snap = SNAP / "seed1_paired_deltas.csv"
    src = EVAL / "seed1" / "paired_deltas.csv"
    if src.is_file():
        pd.read_csv(src).to_csv(snap, index=False)
    p = pd.read_csv(snap)
    for m, key in [("temp_only", "tsb"), ("constancy_tm", "ctm")]:
        for metric in ["fid", "kid"]:
            r = p[(p.method == m) & (p.metric == metric)]
            if len(r):
                r = r.iloc[0]
                dgt = 2 if metric == "fid" else 4
                mac(f"fid@{key}@{metric}", m_(num(r.value, dgt, True)))
                mac(f"fid@{key}@{metric}@ci", "[" + m_(num(r.ci_low, dgt, True)) + ", " + m_(num(r.ci_high, dgt, True)) + "]")
                mac(f"fid@n", str(int(r.n_prompts)))


def seed1_tables(order, d0):
    raw = load_root("seed1")
    d1 = select(raw, order, 8, 63, 1)
    names = {"sega_official": "sega", "temp_only": "tsb", "constancy_tm": "ctm"}
    assert set(d1.method) == set(names), set(d1.method)
    for m in names:
        assert (d1.method == m).sum() == 55, (m, (d1.method == m).sum())
    s0 = d0[d0.idx.between(8, 62)].copy()
    s0["method"] = s0.method.map({"sega_official": "sega_official", "temp_only_sbase": "temp_only",
                                  "constancy_tm_sbase": "constancy_tm"})
    s0 = s0[s0.method.notna()]
    out = {}
    for m in ["temp_only", "constancy_tm"]:
        for k in METRICS:
            out[("s1", m, k)] = paired(d1, m, "sega_official", k)
            out[("s0", m, k)] = paired(s0, m, "sega_official", k)
            register("so", names[m], k, *out[("s1", m, k)])
            register("hs", names[m], k, *out[("s0", m, k)])
            STORE[("so", names[m], k)], STORE[("hs", names[m], k)] = out[("s1", m, k)], out[("s0", m, k)]
    for m, key in names.items():                     # seed-1 means, quoted beside the paired deltas
        for k in METRICS:
            mac(f"som@{key}@{KEY[k]}", m_(num(d1.loc[d1.method == m, k].mean(), PREC[k])))
    L = [r"\begin{tabular}{@{}l rr rr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{\tempsb{} $-$ \segaofficial} & \multicolumn{2}{c}{\constancytm{} $-$ \segaofficial} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Metric & seed 0 & seed 1 & seed 0 & seed 1 \\", r"\midrule"]
    for k in METRICS:
        cells = []
        for m in ["temp_only", "constancy_tm"]:
            for s in ["s0", "s1"]:
                cells.append(dcell(k, *out[(s, m, k)][:3]))
        L.append(LABEL[k] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_seed1.tex").write_text("\n".join(L) + "\n")
    # agreement count: same verdict at seed 0 and seed 1 on the same 55 prompts
    agree = sum(verdict(k, *out[("s0", m, k)][:3]) == verdict(k, *out[("s1", m, k)][:3])
                for m in ["temp_only", "constancy_tm"] for k in METRICS)
    mac("so@agree", str(agree))
    mac("so@cells", str(2 * len(METRICS)))
    log.append(f"seed 1: n=55 per method; verdict agreement with seed 0 on same prompts {agree}/{2*len(METRICS)}")


PILOT = [("sega_official", "sega", r"\segaofficial"), ("sega_official_static", "segast", r"\segaofficialstatic"),
         ("sega", "segap", r"\segapaper"), ("sega_static", "segapst", r"\segapaperstatic"),
         ("ntk2", "ntk", r"\ntk"), ("yarn", "yarn", r"\yarn"), ("dype", "dype", r"\dype"),
         ("temp_only", "temp", r"\tempntk"), ("constancy", "cons", r"\constancy")]


def pilot_table(order):
    raw = load_root("pilot")
    d = select(raw, order, 0, 8, 0)
    for m, _, _ in PILOT:
        assert (d.method == m).sum() == 8, m
    cols = ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024", "clipiqa", "spectrum_log_rmse"]
    L = [r"\begin{tabular}{@{}l " + "r" * len(cols) + r"@{}}", r"\toprule",
         r"Method & " + " & ".join(HEAD[c] for c in cols) + r" \\", r"\midrule"]
    for m, key, disp in PILOT:
        g = d[d.method == m]
        cells = []
        for c in cols:
            mu = g[c].mean()
            mac(f"pim@{key}@{KEY[c]}", m_(num(mu, PREC[c])))
            if m == "sega_official":
                cells.append(r"\refcell{%s}" % num(mu, PREC[c]))
            else:
                dd, lo, hi, n = paired(d, m, "sega_official", c)
                register("pi", key, c, dd, lo, hi, n)
                cells.append(r"\mcellP{%s}{%s}" % (num(mu, PREC[c]), verdict(c, dd, lo, hi)))
        L.append(disp + " & " + " & ".join(cells) + r" \\")
        if m in ("sega_static",):
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_pilot.tex").write_text("\n".join(L) + "\n")
    return d


def dev_table(order):
    """Dev selection, three campaigns, rows 0-7, seed 0. Selection only, never results."""
    cols = ["imagereward", "musiq", "clip", "pickscore", "musiq_crop1024"]
    pilot = select(load_root("pilot"), order, 0, 8, 0)
    camp1 = [("pilot", "temp_only", r"\tempntk", "dev only"),
             ("pilot", "constancy", r"\constancy", "partial held-out$^{a}$"),
             ("devtm_eta1", "constancy_tm", r"\constancytm{} on \ntk{} base, $\eta=1$", "dev only"),
             ("devtm_eta2", "constancy_tm", r"\constancytm{} on \ntk{} base, $\eta=2$", "dev only"),
             ("devsb", "temp_only", r"\tempsb", "held-out (control)"),
             ("devsb", "constancy_tm", r"\constancytm", "held-out (candidate)")]
    camp2 = [("dev2_alias_temp", "alias_temp", r"\aliastemp"), ("dev2_alias_temp_sref", "alias_temp_sref", r"\aliastemps"),
             ("dev2_sega_ctm_eta05", "sega_ctm", r"\segactm, $\eta=0.5$"),
             ("dev2_sega_ctm_eta10", "sega_ctm", r"\segactm, $\eta=1$")]
    camp3 = [("dev3_sega_rms_normalized", None, r"\segarms"), ("dev3_sega_rms_uniform", None, r"\segauniform"),
             ("dev3_sega_ctm_lag", None, r"\segalag")]

    def stats(df, ref):
        r = {}
        for c in cols:
            a = df.set_index("prompt_id")[c]
            b = ref.set_index("prompt_id")[c]
            common = a.index.intersection(b.index)
            assert len(common) == 8
            r[c] = (a.loc[common].mean(), (a.loc[common] - b.loc[common]).mean())
        return r

    def gate(r):
        return (r["imagereward"][1] > 0) and (r["musiq"][1] > 0) and (r["clip"][1] >= -0.3)

    def row(disp, r, decision):
        cells = []
        for c in cols:
            cells.append(r"\devcell{%s}{%s}" % (num(r[c][0], PREC[c]), num(r[c][1], PREC[c], True)))
        return disp + " & " + " & ".join(cells) + " & " + decision + r" \\"

    L = [r"\begin{tabular}{@{}l " + "r" * len(cols) + r" l@{}}", r"\toprule",
         r"Variant & " + " & ".join(HEAD[c] for c in cols) + r" & Outcome \\", r"\midrule"]
    ref1 = pilot[pilot.method == "sega_official"]
    L.append(r"\multicolumn{%d}{@{}l}{\emph{Campaign 1 (reference: \segaofficial, pilot run)}} \\[1pt]" % (len(cols) + 2))
    L.append(row(r"\segaofficial", {c: (ref1[c].mean(), 0.0) for c in cols}, "reference"))
    for root, m, disp, decision in camp1:
        df = pilot if root == "pilot" else select(load_root(root), order, 0, 8, 0)
        r1 = stats(df[df.method == m], ref1)
        L.append(row(disp, r1, decision))
        for c in cols:
            mac(f"dv@d@{root.replace('_', '')}{m.replace('_', '')}@{KEY[c]}", m_(num(r1[c][1], PREC[c], True)))
    L.append(r"\addlinespace[2pt]")
    L.append(r"\multicolumn{%d}{@{}l}{\emph{Campaign 2, written gate (reference: \segaofficial, pilot run)}} \\[1pt]"
             % (len(cols) + 2))
    sel2 = json.loads((ROOT / ".agent/notes/dev2/selection.json").read_text()) if \
        (ROOT / ".agent/notes/dev2/selection.json").is_file() else None
    for root, m, disp in camp2:
        df = select(load_root(root), order, 0, 8, 0)
        r = stats(df[df.method == m], ref1)
        L.append(row(disp, r, "passes, held-out" if gate(r) else "fails gate"))
        mac(f"dv@{root.replace('_', '')}@gate", "passes" if gate(r) else "fails")
    L.append(r"\addlinespace[2pt]")
    ref3 = select(load_root("dev3_sega_official"), order, 0, 8, 0)
    L.append(r"\multicolumn{%d}{@{}l}{\emph{Campaign 3, same gate (reference: \segaofficial, fresh rerun)}} \\[1pt]"
             % (len(cols) + 2))
    L.append(row(r"\segaofficial{} (rerun)", {c: (ref3[c].mean(), ref3[c].mean() - ref1[c].mean()) for c in cols},
                 "reference$^{b}$"))
    for c in cols:
        mac(f"dv@d@rerun@{KEY[c]}", m_(num(ref3[c].mean() - ref1[c].mean(), PREC[c], True)))
    for root, m, disp in camp3:
        df = select(load_root(root), order, 0, 8, 0)
        mm = sorted(set(df.method))
        assert len(mm) == 1, mm
        r = stats(df, ref3)
        L.append(row(disp, r, "passes, not confirmed$^{c}$" if gate(r) else "fails gate"))
        mac(f"dv@{root.replace('_', '')}@gate", "passes" if gate(r) else "fails")
        for c in cols:
            mac(f"dv@d@{root.replace('_', '')}@{KEY[c]}", m_(num(r[c][1], PREC[c], True)))
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_dev.tex").write_text("\n".join(L) + "\n")
    if sel2:
        log.append("dev2 selection.json present; gate recomputed here: " +
                   ", ".join(f"{k}={v}" for k, v in macros if k.startswith("dv@dev2")))


def octave_table(d):
    order = ["sega_official"] + HO_ORDER
    freq = 2.0 ** (np.arange(-12, -1) + 0.5)
    L = [r"\begin{tabular}{@{}l" + "r" * len(BANDS) + r"@{}}", r"\toprule",
         r"Octave band & " + " & ".join(str(i) for i in range(len(BANDS))) + r" \\",
         r"Centre (cyc/px) & " + " & ".join(r"\tiny $2^{%.1f}$" % (e + 0.5) for e in range(-12, -1)) + r" \\", r"\midrule"]
    for m in order:
        g = d[d.method == m]
        vals = [g[b].mean() for b in BANDS]
        L.append(HO[m][1] + " & " + " & ".join(m_(num(v, 2, True)) for v in vals) + r" \\")
        for i, v in enumerate(vals):
            mac(f"oct@{HO[m][0]}@b{i}", m_(num(v, 2, True)))
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_octave.tex").write_text("\n".join(L) + "\n")
    # Nyquist of a 224-pixel view of a 4096-pixel image, in original cycles per pixel
    nyq = 0.5 * 224 / 4096
    first_blind = int(np.argmax(2.0 ** np.arange(-12, -1) >= nyq))   # lowest band whose lower edge is above it
    mac("nyq@cpp", f"{nyq:.4f}")
    mac("nyq@log", f"{np.log2(nyq):.1f}")
    mac("nyq@firstband", str(first_blind))
    mac("nyq@nbands", str(len(BANDS) - first_blind))
    log.append(f"224-px Nyquist = {nyq:.4f} cyc/px (2^{np.log2(nyq):.2f}); bands {first_blind}-10 invisible")


def shares_all(d):
    """Each held-out arm's gain over NTK as a share of released SEGA's gain over NTK (same prompts),
    with a joint paired bootstrap interval for the appendix. Figure 1b draws the two absolute gains."""
    cols = ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024", "clipiqa"]
    for m in ["temp_only_sbase", "dype", "constancy", "constancy_tm_sbase", "sega_ctm_v2"]:
        key = HO[m][0]
        idx_m = set(d[d.method == m].prompt_id)
        common = sorted(idx_m & set(d[d.method == "ntk2"].prompt_id) & set(d[d.method == "sega_official"].prompt_id))
        piv = {x: d[d.method == x].set_index("prompt_id").loc[common] for x in [m, "ntk2", "sega_official"]}
        n = len(common)
        for k in cols:
            a = piv["ntk2"][k].to_numpy(float)
            b = piv[m][k].to_numpy(float)
            c = piv["sega_official"][k].to_numpy(float)
            rng = np.random.default_rng(0)
            ix = rng.integers(0, n, (2000, n))
            sb = (b - a)[ix].mean(1) / (c - a)[ix].mean(1)
            sh = (b - a).mean() / (c - a).mean()
            lo, hi = np.percentile(sb, [2.5, 97.5])
            kk = KEY[k]
            mac(f"shx@p@{key}@{kk}", str(round(100 * sh)))
            mac(f"shx@ci@{key}@{kk}", "[%d, %d]" % (round(100 * lo), round(100 * hi)))
            mac(f"shx@ex@{key}@{kk}", "excludes" if (lo > 1 or hi < 1) else "includes")
        mac(f"shx@n@{key}", str(n))



# ----------------------------------------------------------------------------------------------
# Main-text tables (gold layout): shaded study arms, bold best, underlined second, gray context rows.
MAIN_COLS = ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024", "clipiqa"]


def _rank_marks(vals, metric):
    """Bold for the best displayed mean, underline for the second (ties share a mark)."""
    shown = {m: float(num(v, PREC[metric])) for m, v in vals.items()}
    order = sorted(set(shown.values()), reverse=metric not in LOWER)
    return {m: ("b" if v == order[0] else "u" if len(order) > 1 and v == order[1] else "")
            for m, v in shown.items()}


def table_results(res, means):
    """Table 2: held-out means of every full-n arm plus the decomposition of SEGA's gain over NTK."""
    groups = [("Frequency baselines", ["ntk2", "dype"]),
              ("\\segaofficial{} and its decomposition control", ["sega_official", "temp_only_sbase"]),
              ("Per-band laws (ours)", ["constancy_tm_sbase", "sega_ctm_v2"])]
    shaded, context = {"sega_official"}, {"dype"}   # figs/PALETTE.md: shade only the reference row
    arms = [m for _, ms in groups for m in ms]
    marks = {c: _rank_marks({m: means[(m, c)][0] for m in arms}, c) for c in MAIN_COLS}
    nc = len(MAIN_COLS) + 1
    L = [r"\begin{tabular}{@{}l " + "r" * len(MAIN_COLS) + r"@{}}", r"\toprule",
         r"Arm & " + " & ".join(HEAD[c] for c in MAIN_COLS) + r" \\", r"\midrule"]
    for gi, (title, ms) in enumerate(groups):
        L.append(r"\multicolumn{%d}{@{}l}{\emph{%s}} \\" % (nc, title))
        for m in ms:
            cells = []
            for c in MAIN_COLS:
                v = num(means[(m, c)][0], PREC[c])
                v = {"b": r"\mathbf{%s}" % v, "u": r"\underline{%s}" % v, "": v}[marks[c][m]]
                if m == "sega_official":
                    cells.append(r"\refcell{%s}" % v)
                    continue
                mu, lo, hi, _ = res[(m, c)]
                star = r"^{*}" if HOLM[("historical_condensed30", m, c)] else ""
                cells.append(r"\rcell%s{%s%s}" % (verdict(c, mu, lo, hi).upper(), v, star))
            row = HO[m][1] + "{} & " + " & ".join(cells) + r" \\"
            if m in context:
                row = r"\ctxrow{" + HO[m][1] + "{}} & " + " & ".join(r"\ctxcell{%s}" % x for x in cells) + r" \\"
            L.append((r"\rowcolor{papertint}" if m in shaded else "") + row)
        L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{@{}l}{\emph{Decomposition of \segaofficial's gain over \ntk{} (paired, $n=%d$)}} \\"
             % (nc, means[("sega_official", MAIN_COLS[0])][3]))
    for key, lab in [("basetemp", r"Base\,+\,temp.\ gain"), ("alloc", r"Allocation gain")]:
        cells = []
        for c in MAIN_COLS:
            star = key == "alloc" and HOLM[("historical_condensed30", "temp_only_sbase", c)]
            cells.append(dcell(c, *STORE[(key, c)], holm=star))
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L.append(r"Share of base\,+\,temp.\ & " + " & ".join(STORE[("share", c)] for c in MAIN_COLS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_results.tex").write_text("\n".join(L) + "\n")


ABL_ROWS = ["imagereward", "pickscore", "musiq", "musiq_crop1024", "spectrum_log_rmse"]


def table_alloc(res):
    """Table 3: three per-band allocations added to the same released base and temperature.
    SEGA's allocation is the negated held-out contrast tsb - SEGA (same resamples, exact negation)."""
    L = [r"\begin{tabular}{@{}l >{\columncolor{papertint}}r rr@{}}", r"\toprule",
         r"Metric & \hdr{\sega\\alloc.} & \hdr{TM\\corr.} & \hdr{\sega\\+\,TM} \\", r"\midrule"]
    for k in ABL_ROWS:
        mu, lo, hi, _ = res[("temp_only_sbase", k)]
        cells = [dcell(k, -mu, -hi, -lo)] + [dcell(k, *STORE[(key, k)]) for key in ("ctmtsb", "sctmtsb")]
        L.append(LABEL[k] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_alloc.tex").write_text("\n".join(L) + "\n")


def table_seed1_small():
    """Table 4: seed-1 replication, point estimates with nominal verdict triangles."""
    L = [r"\begin{tabular}{@{}l rr rr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{\tempsb} & \multicolumn{2}{c}{\constancytm} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Metric & seed 0 & seed 1 & seed 0 & seed 1 \\", r"\midrule"]
    for k in ABL_ROWS:
        cells = []
        for m in ("tsb", "ctm"):
            for s in ("hs", "so"):
                mu, lo, hi = STORE[(s, m, k)][:3]
                cells.append(r"\rcell%s{%s}" % (verdict(k, mu, lo, hi).upper(), num(mu, PREC[k], True)))
        L.append(LABEL[k] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_seed1_small.tex").write_text("\n".join(L) + "\n")


def write_macros():
    L = ["% Generated by paper/tables/make_tables.py. Do not edit by hand.",
         r"\makeatletter"]
    for k, v in macros:
        L.append(r"\expandafter\def\csname N@%s\endcsname{%s}" % (k, v))
    L.append(r"\makeatother")
    (OUT / "num_results.tex").write_text("\n".join(L) + "\n")


def main():
    order = prompt_order()
    d, res, means = heldout(order)
    historical_marks(d)
    table_main(res, means)
    table_full(res, means)
    table_share(d)
    shares_all(d)
    contrasts(d)
    seed1_tables(order, d)
    table_results(res, means)
    table_alloc(res)
    table_seed1_small()
    pilot_table(order)
    dev_table(order)
    octave_table(d)
    constants()
    fid_seed1()
    mac("ho@nprompts", str(d[d.method == "sega_official"].shape[0]))
    write_macros()
    print("\n".join(log))
    print(f"{len(macros)} macros -> {OUT / 'num_results.tex'}")


if __name__ == "__main__":
    sys.exit(main())
