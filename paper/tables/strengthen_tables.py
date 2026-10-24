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
contains zero, \\dcellB/W/T cells (direction triangles), Holm mark ^{*} when the Holm-adjusted p <= 0.05
inside the registered family, papertint on the reference row. Bold: largest gain per column in decomposition
tables (only intervals that exclude zero); in arm-minus-reference tables the best arm when its interval
separates it from the reference, or the reference when every arm is worse; best and underlined second among
displayed means in mean tables. Blocks: P1 factorial, P3 sweep, ckpt_matrix, default_shift, E1 fresh, E2 seeds,
E3 Qwen (absent: not run). The Key result sentence of each caption is composed from the copied decisions.
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


def dcell(metric, d, lo, hi, holm=False, bold=False):
    p = PREC[metric]
    point = num(d, p, True)
    point = (r"\mathbf{%s}" % point if bold else point) + (r"^{*}" if holm else "")
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


MAIN_TABLES = ("factorial", "fresh_tests")   # main text; every other table goes to the appendix
APPENDIX_ORDER = ("factorial_full", "factorial_means", "sweep", "fresh_arms", "seeds", "ckpt", "default_shift")
written = []


def write(name, caption, body, data):
    flag = "" if data["complete"] else "% DESCRIPTIVE: incomplete block, layout test only\n"
    head = "% Generated by paper/tables/strengthen_tables.py. Do not edit by hand.\n" + flag
    (OUT / f"tab_strengthen_{name}.tex").write_text(head + "% Caption: " + caption + "\n" + body)
    main = name in MAIN_TABLES
    size, sep = (r"\footnotesize", "3.4pt") if main else (r"\scriptsize", "1.8pt")
    float_ = [r"\begin{table}[%s]" % ("t" if main else "!htbp"), r"\centering", r"\caption{" + caption + "}",
              r"\label{tab:strengthen-%s}" % name.replace("_", "-"), size, r"\setlength{\tabcolsep}{%s}" % sep,
              r"\input{tables/tab_strengthen_%s.tex}" % name, r"\end{table}"]
    (OUT / f"float_strengthen_{name}.tex").write_text(head + "\n".join(float_) + "\n")
    written.append(name)


def appendix_file():
    """One appendix section holding every appendix table float and the sweep figure, in a fixed order."""
    names = [n for n in APPENDIX_ORDER if n in written]
    if not names:
        return
    what = {"factorial_full": r"\cref{tab:strengthen-factorial-full} expands \cref{tab:strengthen-factorial} to all "
                              r"eight contrasts and ten metrics",
            "factorial_means": r"\cref{tab:strengthen-factorial-means} gives the factorial cell means",
            "sweep": r"\cref{tab:strengthen-sweep,fig:fs-sweep} give the temperature sweep",
            "fresh_arms": r"\cref{tab:strengthen-fresh-arms} compares every fresh-prompt arm with \segaofficial",
            "seeds": r"\cref{tab:strengthen-seeds} gives the seed replication",
            "ckpt": r"\cref{tab:strengthen-ckpt} the checkpoint-by-guidance lattice",
            "default_shift": r"\cref{tab:strengthen-default-shift} the default flow shift"}
    lines = ["% Generated by paper/tables/strengthen_tables.py. Do not edit by hand.",
             r"\section{Confirmation campaign: full tables}\label{app:confirm}",
             r"This appendix expands \cref{sec:confirm}. " + "; ".join(what[n] for n in names).replace(r"\cref",
                                                                                                    r"\Cref", 1)
             + ". Every table "
             r"lists all registered metrics, including unfavourable ones, with nominal 95\% paired prompt-bootstrap "
             r"intervals (2{,}000 draws, seed 0).", ""]
    for n in names:
        lines.append(r"\input{tables/float_strengthen_%s}" % n)
        if n == "sweep" and (Path(__file__).resolve().parents[1] / "figs/strengthen/fs_sweep.pdf").exists():
            lines += [r"\begin{figure}[!htbp]", r"\centering",
                      r"\includegraphics[width=\linewidth]{figs/strengthen/fs_sweep.pdf}",
                      r"\caption{\textbf{Temperature sweep.} \tempsb{} at five logit multipliers fixed before "
                      r"generation, minus the reference $s^{0.16}$ (hollow mark, dashed vertical), on held-out rows 8 "
                      r"to 62 ($n=\N{st@s@n}$, seed 0). Whiskers are nominal 95\% paired intervals; pale marks cover "
                      r"zero. Reward models and full-frame MUSIQ fall on both sides of $s^{0.16}$, while crop MUSIQ "
                      r"keeps rising with the multiplier.}", r"\label{fig:fs-sweep}", r"\end{figure}"]
    (OUT / "float_strengthen_appendix.tex").write_text("\n".join(lines) + "\n")


def tabular(cols, metrics, rows):
    head = " & ".join([cols] + [HEAD[m] for m in metrics]) + r" \\"
    lines = [r"\begin{tabular}{@{}l " + "r" * len(metrics) + "@{}}", r"\toprule", head, r"\midrule"]
    lines += rows + [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def group(label, n):
    return r"\multicolumn{%d}{@{}l}{\emph{%s}} \\" % (n + 1, label)


ROWLAB = {"imagereward": r"ImageReward$\uparrow$", "pickscore": r"PickScore$\uparrow$", "clip": r"CLIP$\uparrow$",
          "musiq": r"MUSIQ$\uparrow$", "musiq_crop1024": r"MUSIQ\textsubscript{crop}$\uparrow$",
          "clipiqa": r"CLIP-IQA$\uparrow$", "clipiqa_crop1024": r"CLIP-IQA\textsubscript{crop}$\uparrow$",
          "spectrum_log_rmse": r"Spec.\ err.$\downarrow$", "spectrum_slope_error": r"Slope err.$\downarrow$",
          "tiling1024": r"Tiling$\downarrow$"}


def ttabular(corner, heads, columns, metrics, shade=None):
    """Transposed table (as tab_deltas_full): one row per metric, one column per contrast or arm.
    columns: list of {metric: cell}; shade: index of the reference column (pale method tint)."""
    spec = "".join(r">{\columncolor{papertint}}r" if i == shade else "r" for i in range(len(heads)))
    lines = [r"\begin{tabular}{@{}l" + spec + "@{}}", r"\toprule",
             corner + " & " + " & ".join(r"\hdr{%s}" % h for h in heads) + r" \\", r"\midrule"]
    lines += [ROWLAB[m] + " & " + " & ".join(col.get(m, r"\nom") for col in columns) + r" \\" for m in metrics]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def contrast_cells(frame, name, metrics, label="seed0", bold=frozenset(), holm_marks=True):
    sub = frame[(frame.analysis == label) & (frame.contrast == name)]
    out = {}
    for m in metrics:
        r = sub[sub.metric == m]
        if r.empty or pd.isna(r.iloc[0]["lo"]):
            continue
        r = r.iloc[0]
        out[m] = dcell(m, r["delta"], r["lo"], r["hi"], bool(r["holm_reject"]) and holm_marks, (name, m) in bold)
    return out


def ref_cells(values, metrics, bold_metrics=(), marks=None):
    """Reference means as \refcell; bold where the reference is best (or use precomputed rank marks)."""
    out = {}
    for m in metrics:
        s = marks[m] if marks is not None else num(values[m], PREC[m])
        out[m] = r"\refcell{%s}" % (r"\mathbf{%s}" % s if m in bold_metrics else s)
    return out


def contrast_rows(frame, names, labels, metrics, label="seed0", prefix=None, keyer=None, bold=frozenset(),
                  holm_marks=True):
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
            cells.append(dcell(m, r["delta"], r["lo"], r["hi"], holm and holm_marks, (name, m) in bold))
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
    items = list(items)
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def effect_phrase(label, good, bad):
    """good/bad: metric display names with a Holm-surviving better/worse verdict."""
    parts = []
    if good:
        parts.append(f"improves {listing(good)}")
    if bad:
        parts.append(f"worsens {listing(bad)}")
    return f"{label} " + (" and ".join(parts) if parts else "has no Holm-surviving effect on the displayed metrics")


def favour(metric, d):
    """Signed so that positive is favourable (errors are lower-is-better)."""
    return -d if metric in LOWER else d


def holm_split(found, name, metrics):
    """Metrics on which a contrast is better / worse with a Holm-surviving test (copied decisions)."""
    hit = [m for m in metrics if (name, m) in found and found[(name, m)][3]]
    return ([m for m in hit if verdict(m, *found[(name, m)][:3]) == "b"],
            [m for m in hit if verdict(m, *found[(name, m)][:3]) == "w"])


def best_effect(found, names, metrics):
    """Bold set for decomposition tables: per column, the largest favourable effect whose nominal 95%
    interval excludes zero."""
    out = set()
    for m in metrics:
        cands = [(favour(m, found[(n, m)][0]), n) for n in names
                 if (n, m) in found and verdict(m, *found[(n, m)][:3]) == "b"]
        if cands:
            out.add((max(cands)[1], m))
    return out


def best_arm(found, names, metrics):
    """Bold set for arm-minus-reference tables. Per column: the best arm when its interval favours it over the
    reference; the reference (returned as a metric name) when every arm's interval favours the reference."""
    arms, ref = set(), set()
    for m in metrics:
        cands = [(favour(m, found[(n, m)][0]), n) for n in names if (n, m) in found]
        if not cands:
            continue
        g, n = max(cands)
        if g > 0 and verdict(m, *found[(n, m)][:3]) == "b":
            arms.add((n, m))
        elif all(verdict(m, *found[(k, m)][:3]) == "w" for _, k in cands):
            ref.add(m)
    return arms, ref


def pt(found, name, metric, prec=None):
    return m_(num(found[(name, metric)][0], prec or PREC[metric], True))


def ci_(found, name, metric, prec=None):
    d, lo, hi = found[(name, metric)][:3]
    p = prec or PREC[metric]
    return "[" + m_(num(lo, p, True)) + ", " + m_(num(hi, p, True)) + "]"


def names_(metrics):
    return listing([NAME[m] for m in metrics])


def with_points(found, name, metrics):
    return listing([f"{NAME[m]} ({pt(found, name, m)})" for m in metrics])


# ----------------------------------------------------------------------------------------------
GROUPS = {"base": ["base_main_effect", "base_at_none", "base_at_sref"],
          "temp": ["temperature_main_effect", "temperature_at_ntk2", "temperature_at_sega"],
          "inter": ["interaction"], "alloc": ["allocation"]}


def alloc_words(found, metrics, displayed_only):
    """Allocation phrase: Holm-surviving improvements and losses, and ties on the primary metrics."""
    ab, aw = holm_split(found, "allocation", metrics)
    parts = []
    if ab:
        parts.append("improves " + listing([f"{NAME[ab[0]]} ({pt(found, 'allocation', ab[0])})"]
                                           + [NAME[m] for m in ab[1:]]))
    if aw:
        parts.append(f"worsens {names_(aw)}")
    ties = [m for m in ["imagereward", "pickscore"] if ("allocation", m) in found
            and verdict(m, *found[("allocation", m)][:3]) == "t"]
    text = "the allocation " + (" and ".join(parts) if parts else "survives Holm on no metric")
    if ties:
        text += (" while tying " if parts else " and ties ") + names_(ties)
    return text


def factorial(data):
    c = data["contrasts_holm"]
    main_ = ["base_main_effect", "temperature_main_effect", "interaction"]
    simple = ["base_at_none", "base_at_sref", "temperature_at_ntk2", "temperature_at_sega"]
    labels = {k: v[1] for k, v in FACT.items()}
    # Every cell once, for the copied Holm decisions, the bold marks and the macros.
    _, found = contrast_rows(c, main_ + simple + ["allocation"], labels, TEN, prefix="f", keyer=lambda k: FACT[k][0])
    n = int(c[c.analysis == "seed0"]["n"].max())
    total = int(c[c.analysis == "seed0"]["holm_reject"].sum())
    hits = {g: sum(bool(found[(k, m)][3]) for k in ks for m in TEN if (k, m) in found) for g, ks in GROUPS.items()}
    bold = best_effect(found, main_ + ["allocation"], SIX)
    short = {"base_main_effect": "Base", "temperature_main_effect": "Temperature",
             "interaction": r"Base $\times$ temp.", "allocation": "Allocation"}
    rows_main, _ = contrast_rows(c, main_, short, SIX, bold=bold)
    rows_alloc, _ = contrast_rows(c, ["allocation"], short, SIX, bold=bold)
    body = tabular("Contrast", SIX, [group("Main effects, averaged over the other factor", len(SIX))] + rows_main
                   + [r"\midrule", group("Per-band allocation", len(SIX))] + rows_alloc)
    T = "temperature_main_effect"
    tb, _ = holm_split(found, T, SIX)
    temp_txt = ("improves all six displayed metrics under Holm" if len(tb) == len(SIX) else
                f"improves {names_(tb)} under Holm" if tb else "has no Holm-surviving effect on the displayed metrics")
    quiet = [w for g, w in [("base", "the RoPE base"), ("inter", "the interaction")] if hits[g] == 0]
    quiet_txt = ("neither the RoPE base nor the interaction survives Holm on any of the ten metrics"
                 if len(quiet) == 2 else f"{quiet[0]} survives Holm on none of the ten metrics" if quiet else "")
    key = (f"the global temperature {temp_txt} (ImageReward {pt(found, T, 'imagereward')}, full-frame MUSIQ "
           f"{pt(found, T, 'musiq')}), " + (quiet_txt + ", and " if quiet_txt else "")
           + alloc_words(found, SIX, True) + ".")
    cap = (r"\textbf{Factorial decomposition of \segaofficial.} Paired held-out contrasts on rows 8 to 194 "
           f"($n={n}$ prompts, seed 0) from a 2$\\times$2 design: RoPE base (\\ntk{{}} or \\sega) by global "
           r"logit temperature (none or $s^{0.16}$), plus the allocation contrast (\segaofficial{} minus \tempsb). "
           r"Main effects average the two levels of the other factor; the interaction is the temperature effect "
           r"on the \sega{} base minus that on the \ntk{} base. Base, temperature and allocation add up to the "
           r"whole gain of \segaofficial{} over \ntk{}. Cells give the paired mean difference with its nominal "
           r"95\% interval; triangles mark intervals that exclude zero, $^{*}$ marks Holm-adjusted $p\le0.05$ "
           r"over the registered 80 tests (8 contrasts by 10 metrics, \cref{tab:strengthen-factorial-full}), and "
           r"bold marks the largest gain in each column. Key result: " + key[0].upper() + key[1:])
    write("factorial", cap, body, data)
    # Full appendix version: eight contrasts, ten metrics.
    bold_full = best_effect(found, list(FACT), TEN)
    heads_a = {"base_main_effect": r"Base\\(main)", "temperature_main_effect": r"Temperature\\(main)",
               "interaction": r"Base $\times$\\temp.", "allocation": r"Allocation\\\phantom{x}"}
    heads_b = {"base_at_none": r"Base at\\no temp.", "base_at_sref": r"Base at\\$s^{0.16}$",
               "temperature_at_ntk2": r"Temp. on\\\ntk{} base", "temperature_at_sega": r"Temp. on\\\sega{} base"}
    body = (ttabular(r"\emph{Main effects}", list(heads_a.values()),
                     [contrast_cells(c, k, TEN, bold=bold_full) for k in heads_a], TEN)
            + r"\par\medskip" + "\n"
            + ttabular(r"\emph{Simple effects}", list(heads_b.values()),
                       [contrast_cells(c, k, TEN, bold=bold_full) for k in heads_b], TEN))
    ab, aw = holm_split(found, "allocation", TEN)
    tmiss = [m for m in TEN if not any(found.get((k, m), (0, 0, 0, False))[3] for k in GROUPS["temp"])]
    key = (f"{total} of 80 tests survive Holm: {hits['temp']} temperature tests"
           + (f" (every metric except {names_(tmiss)})" if tmiss else "")
           + f" and {hits['alloc']} allocation tests"
           + (" (" + "; ".join(x for x in [f"better on {names_(ab)}" if ab else "", f"worse on {names_(aw)}" if aw else ""]
                               if x) + ")" if ab or aw else "")
           + "; " + (quiet_txt if quiet_txt else f"{hits['base']} base and {hits['inter']} interaction tests survive")
           + ".")
    cap = (r"\textbf{All factorial contrasts.} Every registered contrast and metric of the held-out factorial "
           rf"($n={n}$, seed 0), with the same cells, marks and Holm family as \cref{{tab:strengthen-factorial}}, transposed; "
           r"bold marks the largest gain for each metric. Key result: " + key)
    write("factorial_full", cap, body, data)
    mac("st@f@n", str(n))
    mac("st@f@nholm", str(total))
    for g, v in hits.items():
        mac(f"st@f@nholm@{g}", str(v))
    # Whole gain of released SEGA over NTK (point only; equals base + temperature + allocation by construction).
    paired = data.get("paired")
    if paired is not None:
        sel = paired[(paired.analysis == "seed0") & (paired.method == "ntk2") & (paired.reference == "sega_official")]
        for r in sel.to_dict("records"):
            if r["metric"] in KEY:
                mac(f"st@f@tot@{KEY[r['metric']]}", m_(num(-r["delta"], PREC[r["metric"]], True)))
    # Cell means: five arms, ten metrics, bold best and underline second among displayed rounded means.
    means = data["means"]
    means = means[means.analysis == "seed0"]
    arms = ["ntk2", "sega_base_none", "ntk2_sref", "temp_only_sbase", "sega_official"]
    hot, cold = ["ntk2_sref", "temp_only_sbase", "sega_official"], ["ntk2", "sega_base_none"]
    marked, vals = {}, {}
    for m in TEN:
        vals[m] = {a: float(means[(means.method == a) & (means.metric == m)]["mean"].iloc[0]) for a in arms}
        marked[m] = rank_marks(vals[m], m)
        for a in arms:
            mac(f"st@fm@{ARMS[a][0]}@{KEY[m]}", m_(num(vals[m][a], PREC[m])))
    heads = {"ntk2": r"\ntk{}\\\phantom{x}", "sega_base_none": r"\sega{}\\base", "ntk2_sref": r"\ntk{}\\+\,temp.",
             "temp_only_sbase": r"\sega{} base\\+\,temp.", "sega_official": r"\sega{}\\(released)"}
    body = ttabular("Metric", list(heads.values()),
                    [{m: m_(marked[m][a]) for m in TEN} for a in arms], TEN, shade=len(arms) - 1)
    split = [m for m in TEN if min(favour(m, vals[m][a]) for a in hot) > max(favour(m, vals[m][a]) for a in cold)]
    rest = [m for m in TEN if m not in split]
    key = ("no metric separates the arms with and without the temperature" if not split else
           "each of the three arms with the temperature scores better than both arms without it on "
           + (f"every metric except {names_(rest)}" if 0 < len(rest) <= 3 else
              "all ten metrics" if not rest else names_(split)))
    cap = (r"\textbf{Factorial cell means.} Held-out means ($n=" + str(n) + r"$, seed 0) of the four "
           r"base-by-temperature cells and of \segaofficial{}, in the shaded column; the first two columns have no temperature. Bold is best and underlined second for each metric among the displayed rounded values; the paired "
           r"tests are in \cref{tab:strengthen-factorial}. Key result: " + key[0].upper() + key[1:] + ".")
    write("factorial_means", cap, body, data)


def sweep(data):
    c = data["contrasts_holm"]
    means = data["means"][data["means"].analysis == "seed0"]
    n = int(c[c.analysis == "seed0"]["n"].max())
    others = [t for t in TAUS if t != "tau_sref"]
    labels = {t: f"${TAUS[t][1]}$" + (f" ({TAUS[t][2]})" if TAUS[t][2] else "") for t in TAUS}
    _, found = contrast_rows(c, others, labels, TEN, prefix="s", keyer=lambda t: TAUS[t][0])
    arms_b, ref_b = best_arm(found, others, TEN)
    val = {(t, m): float(means[(means.method == t) & (means.metric == m)]["mean"].iloc[0]) for t in TAUS for m in TEN}
    heads, cols = [], []
    for tau, (k, v, tag) in TAUS.items():  # increasing multiplier
        heads.append(f"${v}$\\\\" + (tag if tag else r"\phantom{x}"))
        cols.append(ref_cells({m: val[(tau, m)] for m in TEN}, TEN, ref_b) if tau == "tau_sref"
                    else contrast_cells(c, tau, TEN, bold=arms_b))
    for tau, (k, v, tag) in TAUS.items():
        for m in TEN:
            sel = means[(means.method == tau) & (means.metric == m)]
            if not sel.empty:
                r = sel.iloc[0]
                mac(f"st@sm@{k}@{KEY[m]}", m_(num(r["mean"], PREC[m])))
                mac(f"st@smci@{k}@{KEY[m]}", "[" + m_(num(r["lo"], PREC[m])) + ", " + m_(num(r["hi"], PREC[m])) + "]")
    total = int(c[c.analysis == "seed0"]["holm_reject"].sum())
    mac("st@s@n", str(n))
    mac("st@s@nholm", str(total))
    # Key result from the copied decisions and the fixed means.
    _, w0 = holm_split(found, "tau1p0", SIX)
    parts = []
    if len(w0) == len(SIX):
        parts.append(f"removing the temperature (1.000) worsens all six displayed metrics under Holm (ImageReward "
                     f"{pt(found, 'tau1p0', 'imagereward')})")
    elif w0:
        parts.append(f"removing the temperature (1.000) worsens {names_(w0)} under Holm")
    prim = ["imagereward", "pickscore", "musiq"]
    beat = [(t, m) for t in others for m in prim if (t, m) in found and found[(t, m)][3]
            and verdict(m, *found[(t, m)][:3]) == "b"]
    parts.append("no multiplier beats $s^{0.16}$ under Holm on ImageReward, PickScore or full-frame MUSIQ" if not beat
                 else "a multiplier beats $s^{0.16}$ under Holm on " + listing(sorted({NAME[m] for _, m in beat})))
    crop = [val[(t, "musiq_crop1024")] for t in TAUS]
    if all(b > a for a, b in zip(crop, crop[1:])):
        parts.append(f"crop MUSIQ rises monotonically with the multiplier, from {num(crop[0], 2)} at 1.000 to "
                     f"{num(crop[-1], 2)} at 1.450")
    mac("st@s@cropmono", "yes" if all(b > a for a, b in zip(crop, crop[1:])) else "no")
    key = (parts[0] + "." if len(parts) == 1 else parts[0] + ", and " + parts[1] + "." if len(parts) == 2
           else parts[0] + "; " + parts[1] + ", whereas " + parts[2] + ".")
    body = ttabular("Metric", heads, cols, TEN, shade=list(TAUS).index("tau_sref"))
    cap = (r"\textbf{Global temperature sweep.} \tempsb{} with the logit multiplier fixed in advance at five "
           rf"values, held-out rows 8 to 62 ($n={n}$, seed 0, the first 55 held-out rows of the manifest). The shaded "
           r"column is \sega's reference $s^{0.16}=1.248$ and shows means; other columns show the paired difference from "
           r"it with nominal 95\% intervals; $^{*}$ marks Holm-adjusted $p\le0.05$ over 40 tests (4 settings by 10 "
           r"metrics). Bold marks the best setting for a metric when its interval separates it from the reference. "
           r"No value was selected from these results. Key result: " + key[0].upper() + key[1:])
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


STATUS = {"supported": "supported", "not_established": "not established",
          "incompatible_with_equivalence": "incompatible", "unresolved": "unresolved",
          "reverse_finding_nominal": "reverse", "incomplete": "incomplete"}
HPREC = {"H1_IR": 3, "H1_Pick": 3, "H2_MUSIQ": 2}
HKEY = {"H1_IR": "ir", "H1_Pick": "pick", "H2_MUSIQ": "musiq"}


def pval(p):
    """Two significant digits, never scientific notation (the smallest Holm p here is 3/2001)."""
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


def family_rows(fam, prefix, pop, registered):
    lines = []
    label = {"H1_IR": "H1 equivalence, ImageReward", "H1_Pick": "H1 equivalence, PickScore",
             "H2_MUSIQ": "H2 superiority, full MUSIQ"}
    for r in fam:
        h = r["hypothesis"]
        if pop.get(h) is None:
            continue
        d, k = HPREC[h], HKEY[h]
        margin = r"$\pm%s$" % num(r["margin"], 3) if r.get("margin") and not pd.isna(r.get("margin")) else "$>0$"
        outcome = STATUS[r["status"]]
        lines.append(f"{label[h]} & {pop[h]} & {m_(num(r['delta'], d, True))} & "
                     f"$[{num(r['lo'], d, True)}, {num(r['hi'], d, True)}]$ & {margin} & "
                     f"${pval(r['p_holm'])}$ & " + (r"\textbf{%s}" % outcome if registered else outcome) + r" \\")
        mac(f"st@{prefix}@d@{k}", m_(num(r["delta"], d, True)))
        mac(f"st@{prefix}@a@{k}", num(abs(r["delta"]), d))
        mac(f"st@{prefix}@ci@{k}", "[" + m_(num(r["lo"], d, True)) + ", " + m_(num(r["hi"], d, True)) + "]")
        mac(f"st@{prefix}@p@{k}", pval(r["p_holm"]))
        mac(f"st@{prefix}@status@{k}", outcome)
    return lines


def fresh(data):
    conf = data.get("confirmation_family_json", {})
    hyp = data.get("hypotheses_json", {})
    fam = conf.get("family", [])
    lines = family_rows(fam, "h", {"H1_IR": "337 pooled", "H1_Pick": "337 pooled", "H2_MUSIQ": "150 fresh"}, True)
    mac("st@h@H1", STATUS.get(conf.get("H1"), str(conf.get("H1"))))
    mac("st@h@H2", STATUS.get(conf.get("H2"), str(conf.get("H2"))))
    fresh_only = hyp.get("seed0", {}).get("family", [])
    extra = family_rows(fresh_only, "hf", {"H1_IR": "150 fresh", "H1_Pick": "150 fresh"}, False)
    if fresh_only:
        mac("st@hf@H1", STATUS.get(hyp["seed0"]["H1"], hyp["seed0"]["H1"]))
    rep = ANALYSIS / "E2" / "hypotheses.json"
    rep_lines = []
    if rep.exists() and (ANALYSIS / "E2" / "analysis_protocol.json").exists() and \
            json.loads((ANALYSIS / "E2" / "analysis_protocol.json").read_text()).get("complete"):
        seeds = json.loads(rep.read_text()).get("seeds012", {})
        rep_lines = family_rows(seeds.get("family", []), "hr", {"H1_IR": "187, 3 seeds", "H1_Pick": "187, 3 seeds",
                                                                "H2_MUSIQ": "187, 3 seeds"}, False)
        if seeds:
            mac("st@hr@H1", STATUS.get(seeds["H1"], seeds["H1"]))
            mac("st@hr@H2", STATUS.get(seeds["H2"], seeds["H2"]))
    if lines:
        body = [r"\begin{tabular}{@{}l l r r r r l@{}}", r"\toprule",
                r"Test & Prompts & $\Delta$ & 95\% interval & Margin & Holm $p$ & Outcome \\", r"\midrule",
                r"\multicolumn{7}{@{}l}{\emph{Registered tests, one Holm family}} \\"] + lines
        if extra or rep_lines:
            body += [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{Separately reported populations, same rules "
                     r"(descriptive)}} \\"] + extra + rep_lines
        body = "\n".join(body + [r"\bottomrule", r"\end{tabular}"]) + "\n"
        h1, h2 = STATUS.get(conf.get("H1"), conf.get("H1")), STATUS.get(conf.get("H2"), conf.get("H2"))
        pick_fresh = next((r for r in fresh_only if r["hypothesis"] == "H1_Pick"), None)
        tail = ""
        if pick_fresh is not None and pick_fresh["status"] != "supported":
            mg = num(pick_fresh["margin"], 2)
            side = f"$-{mg}$" if pick_fresh["lo"] <= -pick_fresh["margin"] else f"$+{mg}$"
            tail = (f"; on the fresh prompts alone the PickScore interval crosses the {side} margin, so "
                    "equivalence is not established there")
        cap = (r"\textbf{Registered confirmation.} $\Delta$ is \tempsb{} minus \segaofficial{} for H1 and the "
               r"reverse for H2. H1 pools the 187 held-out and 150 fresh prompts (seed 0) and needs both nominal "
               r"intervals inside the margins and both Holm-adjusted equivalence $p\le0.05$; H2 uses the fresh "
               r"prompts only and needs a positive lower bound and Holm $p\le0.05$. Holm runs over the three "
               r"registered tests. The lower rows apply the same rules to the fresh prompts alone and to held-out "
               r"prompt means over seeds 0 to 2, each with its own Holm family; they describe, but do not decide, "
               rf"the hypotheses. Key result: H1 is {h1} on the pooled prompts and H2 is {h2}{tail}.")
        write("fresh_tests", cap, body, data)
    c = data.get("contrasts_holm")
    if c is not None:
        arms = ["temp_only_sbase", "sega_base_none", "ntk2_sref", "dype", "ntk2"]
        lab = {a: ARMS[a][1] for a in arms}
        _, found = contrast_rows(c, arms, lab, TEN, prefix="e", keyer=lambda a: ARMS[a][0])
        arms_b, ref_b = best_arm(found, arms, TEN)
        means = data["means"][data["means"].analysis == "seed0"]
        ref = [r"\refcell{%s}" % ((r"\mathbf{%s}" if m in ref_b else "%s") % num(
            float(means[(means.method == "sega_official") & (means.metric == m)]["mean"].iloc[0]), PREC[m]))
            for m in TEN]
        for m in TEN:
            for a in ["sega_official"] + arms:
                mac(f"st@em@{ARMS[a][0]}@{KEY[m]}",
                    m_(num(float(means[(means.method == a) & (means.metric == m)]["mean"].iloc[0]), PREC[m])))
        heads = [r"\sega{}\\(released)", r"\sega{} base\\+\,temp.", r"\sega{}\\base", r"\ntk{}\\+\,temp.",
                 r"\dype{}\\\phantom{x}", r"\ntk{}\\\phantom{x}"]
        cols = [dict(zip(TEN, ref))] + [contrast_cells(c, a, TEN, bold=arms_b) for a in arms]
        n = int(c[c.analysis == "seed0"]["n"].max())
        total = int(c[c.analysis == "seed0"]["holm_reject"].sum())
        mac("st@e@n", str(n))
        mac("st@e@nholm", str(total))
        winners = [a for a in arms if holm_split(found, a, TEN)[0]]
        parts = []
        for a in winners:
            b, w = holm_split(found, a, TEN)
            parts.append(f"{ARMS[a][1]} beats \\segaofficial{{}} under Holm on {with_points(found, a, b)}"
                         + (f" while losing {with_points(found, a, w)}" if w else ""))
        if len(winners) == 1:
            parts[0] = parts[0].replace(" beats ", " is the only arm that beats ", 1)
        if not winners:
            parts.append("no arm beats \\segaofficial{} on any metric under Holm")
        tb, tw = holm_split(found, "temp_only_sbase", TEN)
        tie_ir = verdict("imagereward", *found[("temp_only_sbase", "imagereward")][:3]) == "t"
        t_txt = (r"\tempsb{} " + (f"ties ImageReward ({pt(found, 'temp_only_sbase', 'imagereward')})" if tie_ir
                                  else f"differs on ImageReward ({pt(found, 'temp_only_sbase', 'imagereward')})")
                 + (f" and loses {names_(tw)} under Holm" if tw else " and has no Holm-surviving loss")
                 + (f" and gains {names_(tb)}" if tb else ""))
        key = "; ".join(parts + [t_txt]) + "."
        body = ttabular("Metric", heads, cols, TEN, shade=0)
        cap = (r"\textbf{Fresh prompts.} Means of \segaofficial{}, in the shaded column, and paired differences from it on 150 "
               rf"unseen Aesthetic-4K training captions ($n={n}$, seed 0), with nominal 95\% intervals; $^{{*}}$ marks "
               r"Holm-adjusted $p\le0.05$ over 50 tests (5 arms by 10 metrics). Bold marks the best entry for a "
               r"metric when its interval separates it from \segaofficial. Key result: " + key[0].upper() + key[1:])
        write("fresh_arms", cap, body, data)


def seeds(data):
    c = data["contrasts_holm"]
    labels = {"seeds012": "Seeds 0, 1, 2 (mean)", "seeds12": "Seeds 1, 2 (mean)", "seed0": "Seed 0",
              "seed1": "Seed 1", "seed2": "Seed 2"}
    found = {}
    for lab in labels:
        found[lab] = contrast_rows(c, ["temp_only_sbase"], {"temp_only_sbase": labels[lab]}, TEN, label=lab)[1]
        for (name, m), (d, lo, hi, holm, _) in found[lab].items():
            # Holm marks belong to the primary seed-mean family only; sensitivity macros carry none.
            register("r" + lab, "tsb", m, d, lo, hi, holm if lab == "seeds012" else None)
    arms_b, ref_b = best_arm(found["seeds012"], ["temp_only_sbase"], TEN)
    means = data["means"]
    ref = means[(means.analysis == "seeds012") & (means.method == "sega_official")]
    heads = [r"\sega{} (rel.)\\seeds 0 to 2", r"Seeds\\0, 1, 2", r"Seeds\\1, 2", r"Seed\\0", r"Seed\\1",
             r"Seed\\2"]
    cols = [ref_cells({m: float(ref[ref.metric == m]["mean"].iloc[0]) for m in TEN}, TEN, ref_b)]
    for lab in labels:  # Holm marks: primary seed-mean family only
        cols.append(contrast_cells(c, "temp_only_sbase", TEN, label=lab,
                                   bold=arms_b if lab == "seeds012" else frozenset(), holm_marks=lab == "seeds012"))
    n = int(c[c.analysis == "seeds012"]["n"].max())
    f0 = found["seeds012"]
    a = "temp_only_sbase"
    same = all(verdict("musiq", *found[lab][(a, "musiq")][:3]) == verdict("musiq", *f0[(a, "musiq")][:3])
               for lab in labels)
    key = (f"averaged over seeds, \\tempsb{{}} minus \\segaofficial{{}} is {pt(f0, a, 'imagereward')} "
           f"{ci_(f0, a, 'imagereward')} on ImageReward and {pt(f0, a, 'pickscore', 3)} {ci_(f0, a, 'pickscore', 3)} on "
           f"PickScore, and {pt(f0, a, 'musiq')} {ci_(f0, a, 'musiq')} on full-frame MUSIQ"
           + (" (Holm)" if f0[(a, "musiq")][3] else "")
           + ("; every single seed gives the same MUSIQ verdict" if same else "; single seeds disagree on MUSIQ")
           + ".")
    body = ttabular("Metric", heads, cols, TEN, shade=0)
    cap = (r"\textbf{Seed replication.} Held-out rows 8 to 194 ($n=" + str(n) + r"$ prompts): \segaofficial{} "
           r"means over seeds 0 to 2 (shaded column) and paired differences of \tempsb{} minus \segaofficial. The "
           r"first difference column averages seeds 0, 1 and 2 within "
           r"each prompt before the paired bootstrap; $^{*}$ marks Holm-adjusted $p\le0.05$ over its 10 metrics "
           r"and bold marks the better arm where its interval excludes zero. The other columns are sensitivity analyses "
           r"without Holm marks. Key result: " + key[0].upper() + key[1:])
    write("seeds", cap, body, data)


def default_shift(data):
    means, paired, report = data["means"], data["paired"], data["report_json"]
    val = {(r["method"], r["metric"]): float(r["mean"]) for r in means.to_dict("records")}
    arms = [("sega_official", "off", r"Official shift, $\mu=1.15$"),
            ("sega_default_shift", "def", r"Default shift, $\mu=11.55$")]
    cols = []
    for meth, k, label in arms:
        cells = {}
        for m in TEN:
            best = max(arms, key=lambda x: favour(m, val[(x[0], m)]))[0] == meth
            s = num(val[(meth, m)], PREC[m])
            cells[m] = r"\refcell{%s}" % (r"\mathbf{%s}" % s if best else s)
            mac(f"st@dsm@{k}@{KEY[m]}", m_(s))
        cols.append(cells)
    sel = paired[paired.contrast == "default_minus_official_shift"]
    cells, worse = {}, []
    for m in TEN:
        r = sel[sel.metric == m].iloc[0]
        d, lo, hi = float(r["mean"]), float(r["ci_low"]), float(r["ci_high"])
        register("ds", "def", m, d, lo, hi)
        cells[m] = dcell(m, d, lo, hi)
        if verdict(m, d, lo, hi) == "w":
            worse.append(m)
        if m == "imagereward":
            ir = (d, lo, hi)
    cols.append(cells)
    sched = report.get("schedule_comparison", [])
    for meth, k, _ in arms:
        s = [x for x in sched if x["method"] == meth]
        if s:
            mac(f"st@ds@sig@{k}", num(min(x["last_evaluated_sigma"] for x in s), 4 if k == "def" else 3))
            mac(f"st@ds@low@{k}", str(max(x["evaluations_below_0p3"] for x in s)))
    mac("st@ds@n", str(len(report.get("prompt_ids", []))))
    key = (("the default shift worsens all ten metrics" if len(worse) == len(TEN) else
            f"the default shift worsens {names_(worse)}")
           + f" and lowers ImageReward by {num(-ir[0], 2)} [{num(-ir[2], 2)}, {num(-ir[1], 2)}]")
    body = ttabular("Metric", [r"Official\\$\mu=1.15$", r"Default\\$\mu=11.55$", r"Default\\$-$ official"],
                    cols, TEN, shade=0)
    cap = (r"\textbf{Default flow shift.} \segaofficial{} on development rows 0 to 7 (seed 0, 28 steps, guidance "
           r"4.5, $4096^2$) with the official shift ($\mu=1.15$, shaded) and with the diffusers default, which "
           r"extrapolates $\mu$ to 11.55 at 65{,}536 image tokens. Means over 8 prompts, bold best; the last column is "
           r"the paired difference with a nominal 95\% interval. Descriptive development evidence. Key result: "
           + key[0].upper() + key[1:] + ".")
    write("default_shift", cap, body, data)


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
           rf"settings on the 150 fresh prompts ($n={n}$, seed 0, 40 steps, true CFG 4.0); nominal 95\% intervals, "
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
    for block, fn in [("P1", factorial), ("P3", sweep), ("ckpt_matrix", ckpt), ("default_shift", default_shift),
                      ("E1", fresh), ("E2", seeds), ("E3", qwen)]:
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
    appendix_file()
    print(json.dumps({"macros": len(macros), "notes": notes, "tables": written}))


if __name__ == "__main__":
    main()
