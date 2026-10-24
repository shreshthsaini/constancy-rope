"""Tables and number macros for the strengthening campaign (factorial, sweep, checkpoint, fresh, seeds, Qwen).

Regenerate from the project root:
    OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python paper/tables/strengthen_tables.py

Inputs (work, read only): .agent/notes/strengthen/analysis/<block>/ written by scripts/strengthen_analysis.py
(P1 factorial, P3 sweep, E1 fresh150, E2 seeds, E3 Qwen) and scripts/strengthen_diagnostic_report.py
(ckpt_matrix). This script computes no statistic: every point, interval, Holm decision and hypothesis
outcome is copied from those files. A block whose analysis is missing or incomplete produces no table
and no macros (pass --allow-incomplete only for layout tests; such files are marked DESCRIPTIVE).

Outputs: paper/tables/num_strengthen.tex (macros, prefix st, read with \\N{st@...}) and
paper/tables/tab_strengthen_*.tex (bare tabulars). Each table file starts with its caption text in a
comment, ending with a "Key result:" sentence that is composed from the copied verdicts.
Conventions follow make_tables.py: precision PREC, signed deltas, verdict tie when the nominal 95% CI
contains zero, \\dcellB/W/T cells, Holm mark ^{*} when the Holm-adjusted p <= 0.05 inside the
registered family, bold best and underlined second among displayed means, papertint on the reference.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / ".agent/notes/strengthen/analysis"
OUT = Path(__file__).resolve().parent
LOWER = {"spectrum_log_rmse", "spectrum_slope_error", "tiling1024"}
PREC = {"imagereward": 3, "pickscore": 2, "clip": 2, "musiq": 2, "musiq_crop1024": 2, "clipiqa": 3,
        "clipiqa_crop1024": 3, "spectrum_log_rmse": 3, "spectrum_slope_error": 3, "tiling1024": 4}
KEY = {"imagereward": "ir", "pickscore": "pick", "clip": "clip", "musiq": "musiq", "musiq_crop1024": "musiqc",
       "clipiqa": "iqa", "clipiqa_crop1024": "iqac", "spectrum_log_rmse": "spec",
       "spectrum_slope_error": "slope", "tiling1024": "tile"}
NAME = {"imagereward": "ImageReward", "pickscore": "PickScore", "clip": "CLIP score", "musiq": "full-frame MUSIQ",
        "musiq_crop1024": "crop MUSIQ", "clipiqa": "CLIP-IQA", "clipiqa_crop1024": "crop CLIP-IQA",
        "spectrum_log_rmse": "spectral error", "spectrum_slope_error": "slope error", "tiling1024": "tiling"}
HEAD = {"imagereward": r"\makecell[r]{Image-\\Reward$\uparrow$}", "pickscore": r"\makecell[r]{Pick-\\Score$\uparrow$}",
        "clip": r"\makecell[r]{CLIP\\score$\uparrow$}", "musiq": r"\makecell[r]{MUSIQ\\full$\uparrow$}",
        "musiq_crop1024": r"\makecell[r]{MUSIQ\\crop$\uparrow$}", "clipiqa": r"\makecell[r]{CLIP-IQA\\full$\uparrow$}",
        "clipiqa_crop1024": r"\makecell[r]{CLIP-IQA\\crop$\uparrow$}", "spectrum_log_rmse": r"\makecell[r]{Spectral\\error$\downarrow$}",
        "spectrum_slope_error": r"\makecell[r]{Slope\\error$\downarrow$}", "tiling1024": r"\makecell[r]{Tiling\\$\downarrow$}"}
SIX = ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024", "clipiqa"]
TEN = SIX + ["clipiqa_crop1024", "spectrum_log_rmse", "spectrum_slope_error", "tiling1024"]
PROVIDE = (r"\providecommand{\segabase}{SEGA base}\providecommand{\ntktemp}{NTK\,+\,temp.}"
           r"\providecommand{\qwen}{Qwen-Image}")
# contrast name in strengthen_analysis.py -> (macro key, LaTeX row label)
FACT = {"base_main_effect": ("base", r"Base: \sega{} $-$ \ntk{}"),
        "temperature_main_effect": ("temp", r"Temperature: $s^{0.16}$ $-$ none"),
        "interaction": ("inter", r"Base $\times$ temperature"),
        "allocation": ("alloc", r"Allocation: \segaofficial{} $-$ \tempsb{}"),
        "base_at_none": ("basenone", r"Base, no temperature"),
        "base_at_sref": ("basesref", r"Base, at $s^{0.16}$"),
        "temperature_at_ntk2": ("tempntk", r"Temperature, \ntk{} base"),
        "temperature_at_sega": ("tempsega", r"Temperature, \sega{} base")}
ARMS = {"ntk2": ("ntk", r"\ntk{}"), "ntk2_sref": ("ntktemp", r"\ntktemp{}"),
        "sega_base_none": ("segabase", r"\segabase{}"), "temp_only_sbase": ("tsb", r"\tempsb{}"),
        "sega_official": ("sega", r"\segaofficial{}"), "dype": ("dype", r"\dype{}"),
        "base_default": ("qbase", r"Default extrapolation")}
TAUS = {"tau1p0": ("t100", "1.000", "none"), "tau1p1": ("t110", "1.100", ""),
        "tau_sref": ("tsref", "1.248", "$s^{0.16}$"), "tau_logn": ("tlogn", "1.333", r"$\tau_N$"),
        "tau1p45": ("t145", "1.450", "")}
macros, notes = [], []


def num(v, d, sign=False):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    if s.lstrip("+-").strip("0.") == "":
        s = ("-" if v < 0 else ("+" if sign else "")) + f"{0:.{d}f}"
    return s


def m_(s):
    return "$" + s + "$"


def mac(name, value):
    macros.append((name, value))


def verdict(metric, d, lo, hi):
    if lo <= 0 <= hi:
        return "t"
    return "b" if ((d < 0) if metric in LOWER else (d > 0)) else "w"


WORD = {"b": "better", "w": "worse", "t": "tie"}


def dcell(metric, d, lo, hi, holm=False):
    p = PREC[metric]
    point = num(d, p, True) + (r"^{*}" if holm else "")
    return r"\dcell%s{%s}{%s}{%s}" % (verdict(metric, d, lo, hi).upper(), point, num(lo, p, True), num(hi, p, True))


def register(prefix, key, metric, d, lo, hi, holm=None):
    p, k = PREC[metric], KEY[metric]
    mac(f"st@{prefix}@d@{key}@{k}", m_(num(d, p, True) + (r"^{*}" if holm else "")))
    mac(f"st@{prefix}@a@{key}@{k}", num(abs(d), p))
    mac(f"st@{prefix}@ci@{key}@{k}", "[" + m_(num(lo, p, True)) + ", " + m_(num(hi, p, True)) + "]")
    mac(f"st@{prefix}@v@{key}@{k}", WORD[verdict(metric, d, lo, hi)])
    if holm is not None:
        mac(f"st@{prefix}@holm@{key}@{k}", "yes" if holm else "no")


def load(block, allow):
    d = ANALYSIS / block
    if not d.is_dir():
        notes.append(f"{block}: no analysis directory")
        return None
    meta = d / ("analysis_protocol.json" if block not in ("ckpt_matrix", "default_shift") else "report.json")
    complete = meta.exists() and bool(json.loads(meta.read_text()).get("complete"))
    if not complete and not allow:
        notes.append(f"{block}: incomplete, skipped")
        return None
    out = {"complete": complete}
    for f in d.glob("*.csv"):
        if f.stat().st_size:
            out[f.stem] = pd.read_csv(f)
    for f in d.glob("*.json"):
        out[f.stem + "_json"] = json.loads(f.read_text())
    return out


def write(name, caption, body, data):
    flag = "" if data["complete"] else "% DESCRIPTIVE: incomplete block, layout test only\n"
    text = ("% Generated by paper/tables/strengthen_tables.py. Do not edit by hand.\n" + flag +
            "% Caption: " + caption + "\n" + body)
    (OUT / f"tab_strengthen_{name}.tex").write_text(text)


def tabular(cols, metrics, rows):
    head = " & ".join([cols] + [HEAD[m] for m in metrics]) + r" \\"
    lines = [r"\begin{tabular}{@{}l " + "r" * len(metrics) + "@{}}", r"\toprule", head, r"\midrule"]
    lines += rows + [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def group(label, n):
    return r"\multicolumn{%d}{@{}l}{\emph{%s}} \\" % (n + 1, label)


def contrast_rows(frame, names, labels, metrics, label="seed0", prefix=None, keyer=None):
    rows, found = [], {}
    sub = frame[frame.analysis == label]
    for name in names:
        cells = []
        for m in metrics:
            r = sub[(sub.contrast == name) & (sub.metric == m)]
            if r.empty or pd.isna(r.iloc[0]["lo"]):
                cells.append(r"\nom")
                continue
            r = r.iloc[0]
            holm = bool(r["holm_reject"])
            cells.append(dcell(m, r["delta"], r["lo"], r["hi"], holm))
            found[(name, m)] = (r["delta"], r["lo"], r["hi"], holm, r["p_holm"])
        rows.append(labels[name] + " & " + " & ".join(cells) + r" \\")
    if prefix:
        for (name, m), (d, lo, hi, holm, _) in found.items():
            register(prefix, keyer(name), m, d, lo, hi, holm)
    return rows, found


def rank_marks(values, metric):
    """Bold best, underline second, compared on the rounded displayed value."""
    shown = {k: round(v, PREC[metric]) for k, v in values.items()}
    order = sorted(set(shown.values()), reverse=metric not in LOWER)
    best, second = order[0], (order[1] if len(order) > 1 else None)
    out = {}
    for k, v in values.items():
        s = num(v, PREC[metric])
        out[k] = (r"\mathbf{%s}" % s if shown[k] == best else r"\underline{%s}" % s if shown[k] == second else s)
    return out


def summary_words(found, names, metrics):
    """Per contrast: which displayed metrics are better/worse with Holm marks."""
    out = {}
    for name in names:
        good = [NAME[m] for m in metrics if (name, m) in found and found[(name, m)][3]
                and verdict(m, *found[(name, m)][:3]) == "b"]
        bad = [NAME[m] for m in metrics if (name, m) in found and found[(name, m)][3]
               and verdict(m, *found[(name, m)][:3]) == "w"]
        out[name] = (good, bad)
    return out


def listing(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def effect_phrase(label, good, bad):
    parts = []
    if good:
        parts.append(f"raises {listing(good)}")
    if bad:
        parts.append(f"lowers {listing(bad)}")
    return f"{label} " + (" and ".join(parts) if parts else "has no Holm-surviving effect on the displayed metrics")


# ----------------------------------------------------------------------------------------------
def factorial(data):
    c = data["contrasts_holm"]
    order = ["base_main_effect", "temperature_main_effect", "interaction", "allocation"]
    simple = ["base_at_none", "base_at_sref", "temperature_at_ntk2", "temperature_at_sega"]
    labels = {k: v[1] for k, v in FACT.items()}
    rows_main, found = contrast_rows(c, order[:3], labels, SIX)
    rows_alloc, found2 = contrast_rows(c, order[3:], labels, SIX)
    found.update(found2)
    body = tabular("Contrast", SIX, [group("Main effects, averaged over the other factor", len(SIX))] + rows_main
                   + [r"\midrule", group("Per-band allocation", len(SIX))] + rows_alloc)
    words = summary_words(found, order, SIX)
    n = int(c[c.analysis == "seed0"]["n"].max())
    key = ("; ".join(effect_phrase(t, *words[k]) for k, t in
                     [("base_main_effect", "the base"), ("temperature_main_effect", "the temperature"),
                      ("interaction", "the interaction"), ("allocation", "the allocation")]))
    cap = (r"\textbf{Factorial decomposition of \segaofficial.} Paired held-out contrasts on rows 8 to 194 "
           f"(n={n} prompts, seed 0) from a 2$\\times$2 design: RoPE base (\\ntk{{}} or \\sega) by global "
           r"logit temperature (none or $s^{0.16}$), plus the allocation contrast. Main effects average the "
           r"two levels of the other factor; the interaction is the temperature effect on the \sega{} base "
           r"minus that on the \ntk{} base. Cells give the paired mean difference with its nominal 95\% "
           r"interval; $^{*}$ marks Holm-adjusted $p\le0.05$ over the registered 80 tests (8 contrasts by 10 "
           r"metrics, \cref{tab:strengthen-factorial-full}). Key result: " + key[0].upper() + key[1:] + ".")
    write("factorial", cap, body, data)
    # Full appendix version: eight contrasts, ten metrics.
    rows_a, f_a = contrast_rows(c, order[:3], labels, TEN, prefix="f", keyer=lambda k: FACT[k][0])
    rows_b, f_b = contrast_rows(c, simple, labels, TEN, prefix="f", keyer=lambda k: FACT[k][0])
    rows_c, f_c = contrast_rows(c, order[3:], labels, TEN, prefix="f", keyer=lambda k: FACT[k][0])
    body = tabular("Contrast", TEN, [group("Main effects", len(TEN))] + rows_a + [r"\midrule",
                   group("Simple effects", len(TEN))] + rows_b + [r"\midrule", group("Allocation", len(TEN))] + rows_c)
    total = int(c[c.analysis == "seed0"]["holm_reject"].sum())
    cap = (r"\textbf{All factorial contrasts.} Every registered contrast and metric of the held-out factorial "
           rf"(n={n}, seed 0), with the same cells and Holm family as \cref{{tab:strengthen-factorial}}. "
           f"Key result: {total} of 80 contrast-metric tests survive Holm.")
    write("factorial_full", cap, body, data)
    mac("st@f@n", str(n))
    mac("st@f@nholm", str(total))
    # Cell means: five arms, ten metrics, bold best among displayed rounded means.
    means = data["means"]
    means = means[means.analysis == "seed0"]
    arms = ["ntk2", "ntk2_sref", "sega_base_none", "temp_only_sbase", "sega_official"]
    marked = {}
    for m in TEN:
        vals = {a: float(means[(means.method == a) & (means.metric == m)]["mean"].iloc[0]) for a in arms}
        marked[m] = rank_marks(vals, m)
        for a in arms:
            mac(f"st@fm@{ARMS[a][0]}@{KEY[m]}", m_(num(vals[a], PREC[m])))
    rows = []
    for a in arms:
        line = ARMS[a][1] + " & " + " & ".join(m_(marked[m][a]) for m in TEN) + r" \\"
        rows.append((r"\rowcolor{papertint}" if a == "sega_official" else "") + line)
    base = {"ntk2": r"\ntk{}", "ntk2_sref": r"\ntk{}", "sega_base_none": r"\sega{}", "temp_only_sbase": r"\sega{}",
            "sega_official": r"\sega{}"}
    body = tabular("Arm", TEN, rows)
    cap = (r"\textbf{Factorial cell means.} Held-out means (n=" + str(n) + r", seed 0) of the four "
           r"base-by-temperature cells and \segaofficial{} (shaded reference). Bold is best and underlined "
           r"second among the displayed rounded values. Key result: see the paired contrasts in "
           r"\cref{tab:strengthen-factorial}; marginal means do not carry the paired comparison.")
    write("factorial_means", cap, body, data)
    del base


def sweep(data):
    c = data["contrasts_holm"]
    means = data["means"][data["means"].analysis == "seed0"]
    n = int(c[c.analysis == "seed0"]["n"].max())
    rows = []
    for tau, (k, val, tag) in TAUS.items():
        label = f"${val}$" + (f" ({tag})" if tag else "")
        if tau == "tau_sref":
            cells = [r"\refcell{%s}" % num(float(means[(means.method == tau) & (means.metric == m)]["mean"].iloc[0]),
                                           PREC[m]) for m in SIX]
            rows.append(r"\rowcolor{papertint}" + label + " & " + " & ".join(cells) + r" \\")
            continue
        r_, found = contrast_rows(c, [tau], {tau: label}, SIX)
        rows += r_
    for tau, (k, val, tag) in TAUS.items():
        for m in TEN:
            sel = means[(means.method == tau) & (means.metric == m)]
            if not sel.empty:
                r = sel.iloc[0]
                mac(f"st@sm@{k}@{KEY[m]}", m_(num(r["mean"], PREC[m])))
                mac(f"st@smci@{k}@{KEY[m]}", "[" + m_(num(r["lo"], PREC[m])) + ", " + m_(num(r["hi"], PREC[m])) + "]")
    _, found = contrast_rows(c, [t for t in TAUS if t != "tau_sref"], {t: t for t in TAUS}, TEN, prefix="s",
                             keyer=lambda t: TAUS[t][0])
    total = int(c[c.analysis == "seed0"]["holm_reject"].sum())
    mac("st@s@n", str(n))
    mac("st@s@nholm", str(total))
    sig = [f"{TAUS[t][1]} on {NAME[m]}" for (t, m), v in found.items() if v[3] and m in SIX]
    key = (f"{total} of 40 tests survive Holm" + (f" ({listing(sig)})" if sig else "") + ".")
    body = tabular(r"Multiplier", SIX, rows)
    cap = (r"\textbf{Global temperature sweep.} \tempsb{} with the logit multiplier fixed in advance at five "
           rf"values, held-out rows 8 to 62 (n={n}, seed 0, first 55 rows of the manifest). The shaded row is "
           r"\sega's reference $s^{0.16}=1.248$ and shows means; other rows show the paired difference from it "
           r"with nominal 95\% intervals; $^{*}$ marks Holm-adjusted $p\le0.05$ over 40 tests (4 settings by "
           r"10 metrics). No value was selected from these results. Key result: " + key)
    write("sweep", cap, body, data)


def ckpt(data):
    rows = []
    summ = data.get("lattice_summary")
    if summ is None:
        notes.append("ckpt_matrix: no lattice summary")
        return
    for r in summ.to_dict("records"):
        g = float(r["guidance"])
        gk = f"g{str(g).replace('.', 'p')}"
        ratio, lo, hi = (float(r[k]) for k in ("mean", "ci_low", "ci_high"))
        mac(f"st@k@ratio@{gk}", num(ratio, 1))
        mac(f"st@k@ci@{gk}", f"[{num(lo, 1)}, {num(hi, 1)}]")
        rows.append((g, ratio, lo, hi))
    paired = data.get("paired")
    lines = [r"\multicolumn{3}{@{}l}{\emph{Historical F5 statistic, peak bin chosen on the \fluxdev{} image}} \\"]
    for g, ratio, lo, hi in sorted(rows):
        lines.append(f"\\fluxdev/\\krea, guidance ${g:.1f}$ & ${num(ratio, 1)}\\times$ & $[{num(lo, 1)}, {num(hi, 1)}]$ \\\\")
    shared = ANALYSIS / "ckpt_matrix_lattice2x2" / "summary.csv"
    twobytwo = {}
    if shared.exists():
        frame = pd.read_csv(shared).set_index("quantity")
        labels = [("checkpoint_dev_over_krea_at_g3p5", r"\fluxdev/\krea, guidance $3.5$", "ckg35"),
                  ("checkpoint_dev_over_krea_at_g4p5", r"\fluxdev/\krea, guidance $4.5$", "ckg45"),
                  ("guidance_g4p5_over_g3p5_dev", r"Guidance $4.5/3.5$, \fluxdev", "gdev"),
                  ("guidance_g4p5_over_g3p5_krea", r"Guidance $4.5/3.5$, \krea", "gkrea")]
        lines += [r"\midrule", r"\multicolumn{3}{@{}l}{\emph{Shared bin, chosen on the mean of all four cells}} \\"]
        for q, label, k in labels:
            g, lo, hi = (float(frame.loc[q, c]) for c in ("geometric_mean", "ci_low", "ci_high"))
            twobytwo[k] = (g, lo, hi)
            digits = 1 if g >= 2 else 2
            mac(f"st@k@two@{k}", num(g, digits))
            mac(f"st@k@twoci@{k}", f"[{num(lo, digits)}, {num(hi, digits)}]")
            lines.append(f"{label} & ${num(g, digits)}\\times$ & $[{num(lo, digits)}, {num(hi, digits)}]$ \\\\")
    body = "\n".join([r"\begin{tabular}{@{}l rr@{}}", r"\toprule",
                      r"Power ratio at the lattice bin & \hdr{Geometric\\mean} & \hdr{95\%\\interval} \\", r"\midrule"]
                     + lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"
    persists = all(lo > 1 for _, _, lo, _ in rows)
    key = ("\\fluxdev{} carries more lattice power than \\krea{} at both matched guidance values"
           if persists else "the checkpoint ratio interval reaches 1 at some guidance value")
    if twobytwo:
        gd, gk = twobytwo["gdev"], twobytwo["gkrea"]
        dev_drop = gd[2] < 1
        krea_flat = gk[1] <= 1 <= gk[2]
        key += ("; raising guidance from 3.5 to 4.5 " +
                ("lowers it on \\fluxdev{}" if dev_drop else "leaves it unresolved on \\fluxdev{}") +
                (" while the \\krea{} interval covers 1" if krea_flat else
                 " and changes \\krea{} as well" if gk[2] < 1 or gk[1] > 1 else "") +
                ", so guidance modulates the lattice but does not create the checkpoint gap." if persists else ".")
    else:
        key += "."
    cap = (r"\textbf{Checkpoint by guidance.} Official \dype{} code on development rows 0 to 7 (seed 42, "
           r"28 steps, $4096^2$). Each entry is a ratio of whitened luminance power (radial power times $f^2$, "
           r"Hann window) at one lattice bin with a period of 8 to 64 pixels: geometric mean over the 8 prompts "
           r"with a 95\% bootstrap interval on the logs. The historical statistic picks the bin on the \fluxdev{} "
           r"image and is biased upward; the shared bin is picked once per prompt from the mean log spectrum of "
           r"all four cells. Descriptive development evidence, not a held-out test. Key result: " + key)
    write("ckpt", cap, body, data)
    if paired is not None:
        for r in paired.to_dict("records"):
            if r["metric"] in KEY:
                register("k", r["contrast"].replace("_", ""), r["metric"], r["mean"], r["ci_low"], r["ci_high"])


def fresh(data):
    conf = data.get("confirmation_family_json", {})
    fam = conf.get("family", [])
    label = {"H1_IR": (r"H1 equivalence, ImageReward", "337 pooled", "ir"),
             "H1_Pick": (r"H1 equivalence, PickScore", "337 pooled", "pick"),
             "H2_MUSIQ": (r"H2 superiority, full MUSIQ", "150 fresh", "musiq")}
    status = {"supported": "supported", "not_established": "not established",
              "incompatible_with_equivalence": "incompatible", "unresolved": "unresolved",
              "reverse_finding_nominal": "reverse", "incomplete": "incomplete"}
    lines = []
    for r in fam:
        name, pop, k = label[r["hypothesis"]]
        d = PREC["imagereward"] if k == "ir" else PREC["pickscore"] if k == "pick" else PREC["musiq"]
        margin = r"$\pm%s$" % num(r["margin"], 3) if r.get("margin") else "$>0$"
        lines.append(f"{name} & {pop} & {m_(num(r['delta'], d, True))} & "
                     f"$[{num(r['lo'], d, True)}, {num(r['hi'], d, True)}]$ & {margin} & "
                     f"${r['p_holm']:.3f}$ & {status[r['status']]} \\\\")
        mac(f"st@h@d@{k}", m_(num(r["delta"], d, True)))
        mac(f"st@h@ci@{k}", "[" + m_(num(r["lo"], d, True)) + ", " + m_(num(r["hi"], d, True)) + "]")
        mac(f"st@h@p@{k}", f"{r['p_holm']:.3f}")
        mac(f"st@h@status@{k}", status[r["status"]])
    mac("st@h@H1", status.get(conf.get("H1"), str(conf.get("H1"))))
    mac("st@h@H2", status.get(conf.get("H2"), str(conf.get("H2"))))
    if lines:
        body = "\n".join([r"\begin{tabular}{@{}l l r r r r l@{}}", r"\toprule",
                          r"Test & Prompts & $\Delta$ & 95\% interval & Margin & Holm $p$ & Outcome \\",
                          r"\midrule"] + lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"
        cap = (r"\textbf{Registered confirmation.} $\Delta$ is \tempsb{} minus \segaofficial{} for H1 and the "
               r"reverse for H2. H1 pools the 187 held-out and 150 fresh prompts (seed 0) and needs both nominal "
               r"intervals inside the margins and both Holm-adjusted equivalence $p\le0.05$; H2 uses the fresh "
               r"prompts only and needs a positive lower bound and Holm $p\le0.05$. Holm runs over these three "
               rf"tests. Key result: H1 is {status.get(conf.get('H1'), conf.get('H1'))} and H2 is "
               rf"{status.get(conf.get('H2'), conf.get('H2'))}.")
        write("fresh_tests", cap, body, data)
    c = data.get("contrasts_holm")
    if c is not None:
        arms = ["temp_only_sbase", "sega_base_none", "ntk2_sref", "dype", "ntk2"]
        rows, found = contrast_rows(c, arms, {a: ARMS[a][1] for a in arms}, SIX)
        contrast_rows(c, arms, {a: a for a in arms}, TEN, prefix="e", keyer=lambda a: ARMS[a][0])
        n = int(c[c.analysis == "seed0"]["n"].max())
        tsb = summary_words(found, ["temp_only_sbase"], SIX)["temp_only_sbase"]
        body = tabular("Arm minus \\segaofficial", SIX, rows)
        cap = (r"\textbf{Fresh prompts.} Paired differences from \segaofficial{} on 150 unseen Aesthetic-4K "
               rf"training captions (n={n}, seed 0), nominal 95\% intervals; $^{{*}}$ marks Holm-adjusted "
               r"$p\le0.05$ over 50 tests (5 arms by 10 metrics). Key result: "
               + effect_phrase(r"\tempsb{}", *tsb) + ".")
        write("fresh_arms", cap, body, data)


def seeds(data):
    c = data["contrasts_holm"]
    labels = {"seeds012": "Seeds 0, 1, 2 (mean)", "seeds12": "Seeds 1, 2 (mean)", "seed0": "Seed 0",
              "seed1": "Seed 1", "seed2": "Seed 2"}
    rows = []
    for lab in labels:
        r_, found = contrast_rows(c, ["temp_only_sbase"], {"temp_only_sbase": labels[lab]}, SIX, label=lab,
                                  prefix="r" + lab, keyer=lambda a: "tsb")
        if lab != "seeds012":   # Holm marks belong to the primary seed-mean family only
            r_ = [x.replace("^{*}", "") for x in r_]
        rows += r_
        if lab == "seeds012":
            rows.append(r"\midrule")
            primary = found
    n = int(c[c.analysis == "seeds012"]["n"].max())
    words = summary_words(primary, ["temp_only_sbase"], SIX)["temp_only_sbase"]
    body = tabular(r"\tempsb{} $-$ \segaofficial", SIX, rows)
    cap = (r"\textbf{Seed replication.} Held-out rows 8 to 194 (n=" + str(n) + r" prompts). The first row "
           r"averages seeds 0, 1 and 2 within each prompt before the paired bootstrap; $^{*}$ marks Holm-adjusted "
           r"$p\le0.05$ over its 10 metrics. Lower rows are sensitivity analyses without Holm marks. Key result: "
           + effect_phrase(r"\tempsb{}", *words) + ".")
    write("seeds", cap, body, data)


def qwen(data):
    c = data["contrasts_holm"]
    arms = ["temp_only_sbase", "base_default"]
    labels = {"temp_only_sbase": r"\sega{} base\,+\,temp.", "base_default": r"Default extrapolation"}
    rows, found = contrast_rows(c, arms, labels, SIX)
    contrast_rows(c, arms, labels, TEN, prefix="q", keyer=lambda a: ARMS[a][0])
    n = int(c[c.analysis == "seed0"]["n"].max())
    words = summary_words(found, arms, SIX)
    body = tabular(r"Arm minus \sega{} (released)", SIX, rows)
    cap = (r"\textbf{Transfer to \qwen.} Within-family paired differences from \qwen{} with released \sega{} "
           rf"settings on the 150 fresh prompts (n={n}, seed 0, 40 steps, true CFG 4.0); nominal 95\% intervals, "
           r"$^{*}$ marks Holm-adjusted $p\le0.05$ over 20 tests. Exploratory. Key result: "
           + effect_phrase(r"base\,+\,temp.", *words["temp_only_sbase"]) + "; "
           + effect_phrase("the default extrapolation", *words["base_default"]) + ".")
    write("qwen", cap, body, data)


def main():
    global ANALYSIS, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-incomplete", action="store_true", help="layout tests only")
    ap.add_argument("--analysis", type=Path, default=ANALYSIS, help="tests only")
    ap.add_argument("--out", type=Path, default=OUT, help="tests only")
    a = ap.parse_args()
    ANALYSIS, OUT = a.analysis, a.out
    OUT.mkdir(parents=True, exist_ok=True)
    mac("st@provide", PROVIDE)
    for block, fn in [("P1", factorial), ("P3", sweep), ("ckpt_matrix", ckpt), ("E1", fresh), ("E2", seeds),
                      ("E3", qwen)]:
        data = load(block, a.allow_incomplete)
        if data is not None:
            try:
                fn(data)
                mac(f"st@done@{block.replace('_', '')}", "yes" if data["complete"] else "descriptive")
            except (KeyError, IndexError, ValueError) as err:
                notes.append(f"{block}: {type(err).__name__} {err}")
    lines = ["% Generated by paper/tables/strengthen_tables.py. Do not edit by hand.", PROVIDE, r"\makeatletter"]
    lines += [r"\expandafter\def\csname N@%s\endcsname{%s}" % (k, v) for k, v in macros if k != "st@provide"]
    lines.append(r"\makeatother")
    (OUT / "num_strengthen.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({"macros": len(macros), "notes": notes}))


if __name__ == "__main__":
    main()
