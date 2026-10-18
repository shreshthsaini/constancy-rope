#!/usr/bin/env python3
"""Mechanism numbers for Section 5.2 (why per-band controllers tie or lose to released SEGA).

Also holds the FLUX.1-dev versus Krea probe contrast used as a checkpoint example in Section 5.4.
Reads archived data only: generation sidecars, probe CSVs, the pilot per-image scores and the
CPU SEGA study JSONs. It never writes outside paper/tables.

Outputs:
  paper/tables/num_mechanism.tex     \\newcommand macros, prefix \\Nmech, letters only
  paper/tables/mechanism_summary.md  writer notes: macro, value, source, field, caveats

Run from the project root:
  OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python paper/tables/mechanism_numbers.py

Conventions: steps are 0-based indices into the 28-step schedule; log gains are natural logs of
the attention-logit gain g (q and k are each scaled by sqrt(g)); signed values use a plain
hyphen-minus, so typeset them in math mode.
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

TABLES = Path(__file__).resolve().parent
NOTES = TABLES.parents[1] / ".agent" / "notes"
SCRATCH = Path("/scratch/09032/saini_2/constancy_rope")
OUT = SCRATCH / "out_krea"
PROMPTS = SCRATCH / "prompts" / "aesthetic4k_eval.jsonl"
CPU_STUDY = Path("/work/09032/saini_2/vista/Projects/Exploration/research_reviews/"
                 "sega_2605.22668_2026-10-14")
PILOT_SCORES = SCRATCH / "eval_krea" / "pilot" / "4096x4096" / "per_image.csv"

STEPS = 28
INDEX_DIMS = 16          # axis-0 rotary dims; spatial dims follow (height 16..71, width 72..127)
PAIRS = 28               # rotary pairs per spatial axis
DEV = range(0, 8)
HELDOUT = range(8, 195)
MATERIAL = 0.1           # log-gain threshold for a material correction (about 10.5 percent)
MID_STEP = 14            # midpoint of the schedule
LATE = range(14, STEPS)  # second half of the schedule
COARSE = [6, 7, 8]       # rotary pairs with wavelengths of about 80 to 200 tokens at 4096^2
FINE = [0, 1, 2]         # rotary pairs with wavelengths of about 6 to 15 tokens

MACROS = {}   # name -> formatted value
GROUPS = []   # summary sections


def fmt(value, digits):
    """Round to a fixed number of decimals and drop a negative sign on zero."""
    if digits == 0:
        text = str(int(round(float(value))))
    else:
        text = f"{float(value):.{digits}f}"
    if re.fullmatch(r"-0(\.0*)?", text):
        text = text[1:]
    return text


def fmt_step(value):
    """Integer steps print as integers; a median of an even count may end in .5."""
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def put(name, text):
    if not re.fullmatch(r"Nmech[A-Za-z]+", name):
        raise ValueError(f"Bad macro name {name}")
    if name in MACROS:
        raise ValueError(f"Duplicate macro {name}")
    MACROS[name] = text
    return name


def group(tag, title, source, rows, caveats):
    """rows: list of (label, [macro names]); every macro must already be registered."""
    for _, names in rows:
        for n in names:
            if n not in MACROS:
                raise KeyError(n)
    GROUPS.append({"tag": tag, "title": title, "source": source, "rows": rows, "caveats": caveats})


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.median(values)), float(values.min()), float(values.max())


# ----------------------------------------------------------------------------- loading

def prompt_rows():
    rows = {}
    with PROMPTS.open() as handle:
        for i, line in enumerate(handle):
            pid = json.loads(line)["id"]
            if pid in rows:
                raise ValueError(f"Duplicate prompt id {pid}")
            rows[pid] = i
    if len(rows) != 195:
        raise ValueError(f"Expected 195 prompts, found {len(rows)}")
    return rows


def sidecars(rel, rows, keep):
    """Yield (row, sidecar) for complete seed-0 4096^2 sidecars whose prompt row is in keep."""
    seen = set()
    for path in sorted((OUT / rel / "4096x4096").glob("*.json")):
        record = json.loads(path.read_text())
        config = record["config"]
        row = rows[config["prompt_id"]]
        if row not in keep:
            continue
        if not record.get("complete") or config["res"] != "4096x4096" or config["seed"] != 0 \
                or config["steps"] != STEPS or len(record["gains"]) != STEPS:
            raise ValueError(f"Unexpected protocol in {path}")
        if [g["step"] for g in record["gains"]] != list(range(STEPS)):
            raise ValueError(f"Gain history is not one entry per step in {path}")
        if row in seen:
            raise ValueError(f"Duplicate prompt row {row} in {rel}")
        seen.add(row)
        yield row, record


def log_tau(config):
    """The controller's global log temperature before any per-band correction."""
    grid = config["grid"]
    if grid[0] != grid[1]:
        raise ValueError("Square grids only")
    if config["temperature"] == "logN":
        return math.log(math.log(grid[0] * grid[1]) / math.log(64 * 64))
    if config["temperature"] == "sref":
        return 0.16 * math.log(grid[0] / 64)
    raise ValueError(f"Unexpected temperature {config['temperature']}")


# ----------------------------------------------------------------------------- (a) gate activity

def correction_stats(rel, rows, keep, mode):
    """Per-prompt statistics of corr = controller_log_g - log tau on the RoPE pairs."""
    stats = []
    sigma_ref = None
    for row, record in sidecars(rel, rows, keep):
        config = record["config"]
        if config["method"] != mode:
            raise ValueError(f"{rel} holds {config['method']}, expected {mode}")
        logg = np.array([g["controller_log_g"] for g in record["gains"]], dtype=np.float64)
        sigma = np.array([g["sigma"] for g in record["gains"]], dtype=np.float64)
        if sigma_ref is None:
            sigma_ref = sigma
        elif not np.array_equal(sigma, sigma_ref):
            raise ValueError("Sigma schedules differ across prompts")
        tau = log_tau(config)
        if np.abs(logg[0] - tau).max() > 1e-6 or not record["gains"][0]["first_step_no_correction"]:
            raise ValueError(f"Step 0 of row {row} is not the uncorrected global temperature")
        corr = logg - logg[0]                       # exact in float32 storage; equals log g - log tau
        if np.abs(corr[:, :INDEX_DIMS]).max() != 0:
            raise ValueError("Axis-0 dims were corrected")
        spatial = corr[:, INDEX_DIMS:]
        if not np.array_equal(spatial[:, 0::2], spatial[:, 1::2]):
            raise ValueError("Rotary pair members differ")
        pairs = np.stack([spatial[:, 0:2 * PAIRS:2], spatial[:, 2 * PAIRS::2]], 1)  # step, axis, pair
        maxabs = np.abs(pairs).reshape(STEPS, -1).max(1)
        nonzero = np.flatnonzero(maxabs > 0)
        material = np.flatnonzero(maxabs > MATERIAL)
        anchor = [any(b["anchor_count"] > 0 for b in g.get("bins", [])) for g in record["gains"]]
        band = [max((abs(c) for b in g.get("bins", []) for c in b["correction"]), default=0.0)
                for g in record["gains"]]
        stats.append({
            "row": row, "sigma": sigma, "pairs": pairs, "maxabs": maxabs,
            "onset": int(nonzero[0]) if nonzero.size else None,
            "onset_material": int(material[0]) if material.size else None,
            "active": int(nonzero.size), "active_material": int(material.size),
            "peak": float(maxabs.max()), "final": float(maxabs[-1]),
            "first_anchor": next((t for t, a in enumerate(anchor) if a), None),
            "first_band": next((t for t, b in enumerate(band) if b > 0), None),
        })
    return stats


def gate_section(rows):
    const = correction_stats("main/constancy", rows, HELDOUT, "constancy")
    const_dev = correction_stats("pilot/constancy", rows, DEV, "constancy")
    tm = correction_stats("mainsb/constancy_tm", rows, HELDOUT, "constancy_tm")
    tm_dev = correction_stats("devtm_eta1/constancy_tm", rows, DEV, "constancy_tm")
    temp = correction_stats("mainsb/temp_only", rows, HELDOUT, "temp_only")
    sigma = const[0]["sigma"]
    if not np.array_equal(sigma, tm[0]["sigma"]):
        raise ValueError("Constancy and constancy_tm schedules differ")
    if any(s["active"] for s in temp):
        raise ValueError("temp_only shows a per-band correction")
    for name, block in (("constancy", const), ("constancy dev", const_dev),
                        ("constancy_tm", tm), ("constancy_tm dev", tm_dev)):
        if any(s["onset"] is None for s in block):
            raise ValueError(f"{name}: a prompt never corrects")
        mismatch = sum(s["onset"] != s["first_band"] for s in block)
        print(f"[check] {name}: n={len(block)} rows {min(x['row'] for x in block)}..{max(x['row'] for x in block)}, "
              f"dim-level onset differs from band-level onset in {mismatch} prompts")

    def onset_macros(prefix, block, key):
        med, lo, hi = summarize([s[key] for s in block])
        put(f"Nmech{prefix}StepMed", fmt_step(med))
        put(f"Nmech{prefix}StepMin", fmt_step(lo))
        put(f"Nmech{prefix}StepMax", fmt_step(hi))
        # The schedule is identical across prompts, so the sigma at the median step is the median sigma.
        put(f"Nmech{prefix}SigmaMed", fmt(np.median([sigma[s[key]] for s in block]), 2))
        put(f"Nmech{prefix}SigmaHi", fmt(sigma[int(lo)], 2))
        put(f"Nmech{prefix}SigmaLo", fmt(sigma[int(hi)], 2))
        return [f"Nmech{prefix}{k}" for k in ("StepMed", "StepMin", "StepMax")], \
               [f"Nmech{prefix}{k}" for k in ("SigmaMed", "SigmaHi", "SigmaLo")]

    def count_macros(prefix, block, key):
        med, lo, hi = summarize([s[key] for s in block])
        names = [put(f"Nmech{prefix}Med", fmt_step(med)), put(f"Nmech{prefix}Min", fmt_step(lo)),
                 put(f"Nmech{prefix}Max", fmt_step(hi))]
        names.append(put(f"Nmech{prefix}PctMed", fmt(100 * med / STEPS, 0)))
        return names

    def size_macros(prefix, block, key):
        med, lo, hi = summarize([s[key] for s in block])
        return [put(f"Nmech{prefix}Med", fmt(med, 2)), put(f"Nmech{prefix}Min", fmt(lo, 2)),
                put(f"Nmech{prefix}Max", fmt(hi, 2))]

    def pair_profile(block, step, pairs):
        return float(np.median([s["pairs"][step][:, pairs].mean() for s in block]))

    def lowest_median(block):
        """Most negative median-over-prompts of the per-step minimum correction."""
        per_step = np.median([s["pairs"].reshape(STEPS, -1).min(1) for s in block], 0)
        return float(per_step.min())

    def negative_pct(block):
        values = np.concatenate([s["pairs"].ravel() for s in block])
        values = values[values != 0]
        return 100 * float((values < 0).mean())

    touched = set()
    for block in (const, const_dev, tm, tm_dev):
        for s in block:
            touched |= set(np.flatnonzero(np.abs(s["pairs"]).max((0, 1)) > 0).tolist())
    if touched != set(range(max(touched) + 1)):
        raise ValueError(f"Corrected pairs are not a contiguous low-index block: {sorted(touched)}")

    # constancy, late-profile target (ntk2 base, logN temperature)
    span = f"{min(x['row'] for x in const)}-{max(x['row'] for x in const)}"
    put("NmechConstN", str(len(const)))
    put("NmechConstDevN", str(len(const_dev)))
    c_step, c_sig = onset_macros("ConstOnset", const, "onset")
    cd_step, cd_sig = onset_macros("ConstDevOnset", const_dev, "onset")
    c_act = count_macros("ConstActiveSteps", const, "active")
    c_peak = size_macros("ConstPeak", const, "peak")
    c_final = size_macros("ConstFinal", const, "final")
    put("NmechConstFineFinal", fmt(pair_profile(const, STEPS - 1, FINE), 2))
    put("NmechConstCoarseFinal", fmt(pair_profile(const, STEPS - 1, COARSE), 2))
    put("NmechConstNegPct", fmt(negative_pct(const), 0))
    anchored = sum(s["onset"] == s["first_anchor"] for s in const)
    put("NmechConstAnchorOnsetCount", str(anchored))
    put("NmechCorrPairs", str(len(touched)))
    put("NmechPairsPerAxis", str(PAIRS))
    put("NmechSteps", str(STEPS))
    put("NmechTempOnlyN", str(len(temp)))

    # constancy_tm, time-matched target (released SEGA base, sref temperature)
    put("NmechTmN", str(len(tm)))
    put("NmechTmDevN", str(len(tm_dev)))
    t_step, t_sig = onset_macros("TmOnset", tm, "onset")
    tmat_step, tmat_sig = onset_macros("TmMaterial", tm, "onset_material")
    t_act = count_macros("TmActiveSteps", tm, "active")
    tmat_act = count_macros("TmMaterialSteps", tm, "active_material")
    t_peak = size_macros("TmPeak", tm, "peak")
    t_final = size_macros("TmFinal", tm, "final")
    put("NmechTmCoarseMid", fmt(pair_profile(tm, MID_STEP, COARSE), 2))
    put("NmechTmFineMid", fmt(pair_profile(tm, MID_STEP, FINE), 2))
    put("NmechTmCoarseFinal", fmt(pair_profile(tm, STEPS - 1, COARSE), 2))
    put("NmechTmFineFinal", fmt(pair_profile(tm, STEPS - 1, FINE), 2))
    put("NmechTmLowestMed", fmt(lowest_median(tm), 2))
    put("NmechTmDevLowestMed", fmt(lowest_median(tm_dev), 2))
    put("NmechTmNegPct", fmt(negative_pct(tm), 0))
    put("NmechMidStep", str(MID_STEP))
    put("NmechMidSigma", fmt(sigma[MID_STEP], 2))
    put("NmechMaterialThreshold", fmt(MATERIAL, 1))

    group("a1", "(a) Constancy gate activity, late-profile target (held-out constancy, ntk2 base)",
          f"`out_krea/main/constancy/4096x4096/*.json` rows {span}; dev: `out_krea/pilot/constancy`. "
          "Field `gains[t].controller_log_g` minus its step-0 value (= log tau, checked); "
          "`gains[t].sigma`; anchor gate from `gains[t].bins[*].anchor_count`.",
          [("n held-out, n dev, steps, pairs per axis, pairs ever corrected (both methods), temp_only n",
            ["NmechConstN", "NmechConstDevN", "NmechSteps", "NmechPairsPerAxis", "NmechCorrPairs",
             "NmechTempOnlyN"]),
           ("first nonzero step med, min, max; sigma med, earliest, latest", c_step + c_sig),
           ("dev pilot: first nonzero step med, min, max; sigma med, earliest, latest", cd_step + cd_sig),
           ("steps with any correction med, min, max, pct med; onset = first anchored step (count)",
            c_act + ["NmechConstAnchorOnsetCount"]),
           ("max abs corr: peak over steps med, min, max; final step med, min, max", c_peak + c_final),
           ("final-step median corr pairs 0-2, pairs 6-8; pct of nonzero corr negative",
            ["NmechConstFineFinal", "NmechConstCoarseFinal", "NmechConstNegPct"])],
          [f"Only {MACROS['NmechConstN']} held-out prompts exist for this run (rows {span}; the ntk2-base "
           "constancy main run was stopped early); seed 0.",
           "Onset index = number of leading uncorrected steps. The correction stays exactly zero until "
           "an anchor band (wavelength 16-64 tokens) passes the w > 0.5 gate; temp_only is exactly zero.",
           "Corrects the note 'acts only at sigma <= 0.35 (steps 24-27)': onset is steps "
           f"{MACROS['NmechConstOnsetStepMin']}-{MACROS['NmechConstOnsetStepMax']} (sigma "
           f"{MACROS['NmechConstOnsetSigmaHi']}-{MACROS['NmechConstOnsetSigmaLo']}) held-out and "
           f"{MACROS['NmechConstDevOnsetStepMin']}-{MACROS['NmechConstDevOnsetStepMax']} on dev."])

    group("a2", "(a) Constancy_tm gate activity (held-out, released SEGA base, sref temperature)",
          "`out_krea/mainsb/constancy_tm/4096x4096/*.json` rows 8-194; dev check "
          "`out_krea/devtm_eta1/constancy_tm` (ntk2 base, logN). Same fields as above.",
          [("n held-out, n dev, material threshold (log gain), mid step, its sigma",
            ["NmechTmN", "NmechTmDevN", "NmechMaterialThreshold", "NmechMidStep", "NmechMidSigma"]),
           ("first nonzero step med, min, max; sigma med, earliest, latest", t_step + t_sig),
           ("first step above threshold med, min, max; sigma med, earliest, latest", tmat_step + tmat_sig),
           ("steps with any corr med, min, max, pct med; steps above threshold med, min, max, pct med",
            t_act + tmat_act),
           ("max abs corr: peak over steps med, min, max; final step med, min, max", t_peak + t_final),
           ("median corr pairs 6-8, 0-2 at mid step; same at final step; lowest per-step median of min "
            "corr held-out, dev; pct of nonzero corr negative",
            ["NmechTmCoarseMid", "NmechTmFineMid", "NmechTmCoarseFinal", "NmechTmFineFinal",
             "NmechTmLowestMed", "NmechTmDevLowestMed", "NmechTmNegPct"])],
          ["Exact-nonzero onset is early but tiny; the threshold onset supports 'acts from about step "
           f"{MACROS['NmechTmMaterialStepMed']}'. The note's '-0.57' reproduces on dev (ntk2 base, devtm_eta1: "
           f"{MACROS['NmechTmDevLowestMed']}); held-out on the SEGA base gives {MACROS['NmechTmLowestMed']}.",
           f"Only pairs 0-{int(MACROS['NmechCorrPairs']) - 1} are ever corrected: the controller zeroes pairs "
           "whose wavelength exceeds the 256-token grid (crope/gain.py)."])
    return {"const": const, "tm": tm}


# ----------------------------------------------------------------------------- (b) SEGA gain statistics

def decompose(x):
    """Share of sum(x^2) explained by a static per-pair tilt, then by tilt plus timestep.

    x: prompt x step x pair. Mirrors sega_stats.decompose of the CPU study, which measures the
    modulation about its unit level; here x is the log gain about SEGA's reference log gain.
    """
    total = float((x ** 2).sum())
    static = float(((x - x.mean((0, 1), keepdims=True)) ** 2).sum())
    timed = float(((x - x.mean(0, keepdims=True)) ** 2).sum())
    centered = x - x.mean()
    total_c = float((centered ** 2).sum())
    return {"static": 1 - static / total, "time": 1 - timed / total, "content": timed / total,
            "static_c": 1 - static / total_c, "content_c": timed / total_c,
            "content_rms": math.sqrt(timed / x.size)}


def sega_arrays(rel, rows, mode):
    logg, official, spread, refs, schedules = [], [], [], set(), set()
    for _, record in sidecars(rel, rows, HELDOUT):
        if record["config"]["method"] != mode:
            raise ValueError(f"{rel} holds {record['config']['method']}")
        g = np.array([s["controller_log_g"] for s in record["gains"]], dtype=np.float64)
        if np.abs(g[:, :INDEX_DIMS]).max() != 0 or not np.array_equal(g[:, INDEX_DIMS::2],
                                                                      g[:, INDEX_DIMS + 1::2]):
            raise ValueError("Unexpected axis-0 gain or unequal pair members")
        logg.append(g[:, INDEX_DIMS::2])
        if mode == "sega_ctm":
            o = np.array([s["official_log_g"] for s in record["gains"]], dtype=np.float64)
            official.append(o[:, INDEX_DIMS::2])
        spread.append([s["official"]["spectral_spread"] for s in record["gains"]])
        refs |= {s["official"]["reference_qk_gain"] for s in record["gains"]}
        schedules.add(tuple(s["sigma"] for s in record["gains"]))
    if len(refs) != 1 or len(schedules) != 1:
        raise ValueError("Reference gains or sigma schedules differ across prompts")
    ref_logit = 2 * math.log(refs.pop())
    return (np.array(logg), np.array(official) if official else None, np.array(spread), ref_logit,
            schedules.pop())


def sega_section(rows):
    logg, _, spread, ref, sigma = sega_arrays("main/sega_official", rows, "sega_official")
    n = logg.shape[0]
    x = logg - ref
    full, late = decompose(x), decompose(x[:, LATE])
    gain = np.exp(logg)
    clamp = np.abs(logg) < 1e-7
    per_pair_range = gain.max(1) - gain.min(1)              # prompt x pair, range over steps
    final_clamped = clamp[:, -1].reshape(n, 2, PAIRS).sum(2)  # prompt x axis
    spread_r2 = 1 - ((spread - spread.mean(0)) ** 2).sum() / ((spread - spread.mean()) ** 2).sum()
    put("NmechSegaN", str(n))
    put("NmechSegaRefGain", fmt(math.exp(ref), 2))
    put("NmechSegaGainMin", fmt(gain.min(), 2))
    put("NmechSegaGainMax", fmt(gain.max(), 2))
    put("NmechSegaClampPct", fmt(100 * clamp.mean(), 1))
    put("NmechSegaStepZeroGain", fmt(np.median(gain[:, 0]), 2))
    put("NmechSegaFinalFineGain", fmt(np.median(gain[:, -1].reshape(n, 2, PAIRS)[:, :, 0]), 2))
    put("NmechSegaFinalClampPairs", fmt_step(np.median(final_clamped)))
    put("NmechSegaRangeMed", fmt(np.median(per_pair_range), 2))
    put("NmechSegaRangeMax", fmt(per_pair_range.max(), 2))
    put("NmechSegaStaticPct", fmt(100 * full["static"], 0))
    put("NmechSegaStaticTimePct", fmt(100 * full["time"], 0))
    put("NmechSegaContentPct", fmt(100 * full["content"], 0))
    put("NmechSegaLateStaticPct", fmt(100 * late["static"], 0))
    put("NmechSegaLateStaticTimePct", fmt(100 * late["time"], 0))
    put("NmechSegaLateContentPct", fmt(100 * late["content"], 1))
    put("NmechSegaContentRms", fmt(full["content_rms"], 3))
    put("NmechLateFirstStep", str(LATE.start))
    put("NmechLateSigma", fmt(sigma[LATE.start], 2))
    put("NmechSegaSpreadStepZero", fmt(np.median(spread[:, 0]), 2))
    put("NmechSegaSpreadLateMin", fmt(spread[:, LATE].min(), 2))
    put("NmechSegaSpreadTimeRsq", fmt(spread_r2, 2))
    shift = max(abs(full["static"] - full["static_c"]), abs(full["content"] - full["content_c"]))
    print(f"[check] SEGA decomposition, grand-mean centered instead of reference centered: static "
          f"{full['static_c']:.4f} vs {full['static']:.4f}, content {full['content_c']:.4f} vs "
          f"{full['content']:.4f}")

    ctm, official, _, ref_ctm, sigma_ctm = sega_arrays("main2/sega_ctm", rows, "sega_ctm")
    if abs(ref_ctm - ref) > 1e-12 or sigma_ctm != sigma:
        raise ValueError("sega_ctm reference gain or schedule differs")
    total, part = decompose(ctm - ref), decompose(official - ref)
    corr = decompose(ctm - official)
    put("NmechCtmN", str(ctm.shape[0]))
    put("NmechCtmStaticPct", fmt(100 * total["static"], 0))
    put("NmechCtmStaticTimePct", fmt(100 * total["time"], 0))
    put("NmechCtmContentPct", fmt(100 * total["content"], 0))
    put("NmechCtmSegaPartContentPct", fmt(100 * part["content"], 0))
    put("NmechCtmCorrContentPct", fmt(100 * corr["content"], 0))
    put("NmechCtmContentRms", fmt(total["content_rms"], 3))
    put("NmechCtmCorrContentRms", fmt(corr["content_rms"], 3))
    put("NmechCtmCorrMax", fmt(np.abs(ctm - official).max(), 2))

    group("b1", "(b) Released SEGA gains on Krea (held-out)",
          "`out_krea/main/sega_official/4096x4096/*.json` rows 8-194. Fields `gains[t].controller_log_g` "
          "(= 2 log q/k gain), `gains[t].official.reference_qk_gain`, `gains[t].official.spectral_spread`.",
          [("n; reference logit gain; logit gain min, max over prompts, steps, pairs; pct of entries at the "
            "q/k >= 1 clamp", ["NmechSegaN", "NmechSegaRefGain", "NmechSegaGainMin", "NmechSegaGainMax",
                                "NmechSegaClampPct"]),
           ("median gain at step 0; median finest-pair gain at final step; median pairs per axis at clamp at "
            "final step; per-pair gain range over steps med, max",
            ["NmechSegaStepZeroGain", "NmechSegaFinalFineGain", "NmechSegaFinalClampPairs",
             "NmechSegaRangeMed", "NmechSegaRangeMax"]),
           ("all steps: static, static+time, content pct; content RMS (log gain)",
            ["NmechSegaStaticPct", "NmechSegaStaticTimePct", "NmechSegaContentPct", "NmechSegaContentRms"]),
           ("late window first step, its sigma; static, static+time, content pct",
            ["NmechLateFirstStep", "NmechLateSigma", "NmechSegaLateStaticPct", "NmechSegaLateStaticTimePct",
             "NmechSegaLateContentPct"]),
           ("spread (flatness gate): median at step 0, min over late steps, R2 by timestep alone",
            ["NmechSegaSpreadStepZero", "NmechSegaSpreadLateMin", "NmechSegaSpreadTimeRsq"])],
          ["Decomposition of x = log g - log g_ref over (prompt, step, pair), sums of squares about 0 as in "
           f"the CPU study; grand-mean centering moves each share by at most {shift:.3f}.",
           "'Content' is the residual after the per-(step, pair) mean, so it includes each prompt's own "
           "noise draw (derived seeds differ by prompt). Seed 0 only.",
           "Released code (crope/sega_official.py allocation) maps pairs to profile bins by rank of log "
           "period, so no pair collapses to one bin; instead the longest-wavelength pairs sit at the clamp."])

    group("b2", "(b) sega_ctm gains (held-out), released SEGA gain plus constancy_tm correction",
          "`out_krea/main2/sega_ctm/4096x4096/*.json` rows 8-194. Fields `gains[t].controller_log_g` "
          "(total) and `gains[t].official_log_g` (SEGA part on the same trajectory).",
          [("n; total gain static, static+time, content pct; content pct of the SEGA part alone; of the "
            "correction alone (about 0)", ["NmechCtmN", "NmechCtmStaticPct", "NmechCtmStaticTimePct",
                                           "NmechCtmContentPct", "NmechCtmSegaPartContentPct",
                                           "NmechCtmCorrContentPct"]),
           ("content RMS of total, of correction; max abs correction",
            ["NmechCtmContentRms", "NmechCtmCorrContentRms", "NmechCtmCorrMax"])],
          ["eta = 1.0 (the variant that passed dev). The correction raises the prompt dependence of the "
           "allocation, yet the held-out result ties or loses."])


# ----------------------------------------------------------------------------- (c) probes

def probe(path, rows):
    data = list(csv.DictReader(path.open()))
    pids = {r["prompt_id"] for r in data}
    if len(pids) != 1:
        raise ValueError(f"{path} holds {len(pids)} prompts")
    if {r["gain_factor"] for r in data} != {"1.5"} or {r["seed"] for r in data} != {"0"}:
        raise ValueError(f"{path}: unexpected factor or seed")
    out = {}
    for domain in ("latent", "image"):
        for axis in ("height", "width"):
            m = np.full((6, 7), np.nan)
            for r in data:
                if r["domain"] == domain and r["axis"] == axis:
                    m[int(r["input_bin"]), int(r["output_bin"])] = float(r["response"])
            if np.isnan(m).any():
                raise ValueError(f"{path}: missing response cells")
            out[(domain, axis)] = m
    top = {(float(r["input_low"]), float(r["input_high"])) for r in data if r["input_bin"] == "5"}
    if top != {(0.125, 0.25)}:
        raise ValueError(f"{path}: top input octave is {top}")
    sidecar = sorted((path.parent / "baseline").rglob("*.json"))[0]
    config = json.loads(sidecar.read_text())["config"]
    return rows[pids.pop()], out, config


def diag_off(m):
    diag = np.diag(m[:, :6])
    off = m[~np.eye(6, 7, dtype=bool)]       # all 36 off-diagonal cells of the 6 x 7 matrix
    return float(diag.mean()), float(off.mean()), diag


KEYS = {("latent", "height"): "LatH", ("latent", "width"): "LatW",
        ("image", "height"): "ImgH", ("image", "width"): "ImgW"}


def probe_section(rows):
    krea = {}
    configs = {}
    for path in sorted((SCRATCH / "probe_krea").glob("4096x4096*/mtf.csv")):
        row, mats, config = probe(path, rows)
        krea[row], configs[row] = mats, config
    if sorted(krea) != [0, 1, 2]:
        raise ValueError(f"Expected Krea probe rows 0-2, found {sorted(krea)}")
    letters = {0: "A", 1: "B", 2: "C"}
    put("NmechProbeN", str(len(krea)))
    put("NmechProbeFactor", "1.5")
    put("NmechProbeGuidance", fmt(configs[0]["guidance"], 1))
    put("NmechProbeDiagCells", str(6 * len(krea)))
    rows_out = [("prompts (dev rows 0-2), gain factor, guidance, diagonal cells",
                 ["NmechProbeN", "NmechProbeFactor", "NmechProbeGuidance", "NmechProbeDiagCells"])]
    for key, tag in KEYS.items():
        names = []
        for row, letter in letters.items():
            d, o, diag = diag_off(krea[row][key])
            names += [put(f"NmechProbe{tag}Diag{letter}", fmt(d, 2)), put(f"NmechProbe{tag}Off{letter}", fmt(o, 2))]
            if letter == "A":
                put(f"NmechProbe{tag}TopA", fmt(diag[-1], 2))
            if tag == "LatW":
                put(f"NmechProbeLatWMaxDiag{letter}", fmt(diag.max(), 2))
                put(f"NmechProbeLatWMinDiag{letter}", fmt(diag.min(), 2))
        mean = np.mean([krea[r][key] for r in letters], 0)
        d, o, _ = diag_off(mean)
        positive = sum(int((diag_off(krea[r][key])[2] > 0).sum()) for r in letters)
        names += [put(f"NmechProbe{tag}DiagAvg", fmt(d, 2)), put(f"NmechProbe{tag}OffAvg", fmt(o, 2)),
                  put(f"NmechProbe{tag}PosDiag", str(positive))]
        label = f"{tag}: diag, off for A, B, C; avg diag, off; positive diag cells"
        if tag == "LatW":
            names += [f"NmechProbeLatW{k}Diag{x}" for x in "ABC" for k in ("Max", "Min")]
            label += "; max, min diag for A, B, C"
        rows_out.append((label, names))
    group("c1", "(c) Attention-gain transfer probe on Krea (dev prompts)",
          "`probe_krea/4096x4096{,_p1,_p2}/mtf.csv` column `response` = d log P_out / d log g_in, indexed by "
          "`domain`, `axis`, `input_bin` (0-5), `output_bin` (0-6); settings from `baseline/**/*.json` config.",
          rows_out,
          ["Diag = mean of the 6 diagonal cells; Off = mean of the 36 off-diagonal cells of the 6 x 7 matrix "
           "(this reproduces the note's prompt-0 values exactly).",
           "Three dev prompts, one seed, one 1.5x same-seed finite difference per octave applied to both axes "
           "at once; ntk2 base with logN temperature (the constancy setting, not SEGA's).",
           f"Latent width diagonal mean is {MACROS['NmechProbeLatWDiagA']} for prompt 0 but "
           f"{MACROS['NmechProbeLatWDiagB']} and {MACROS['NmechProbeLatWDiagC']} for prompts 1 and 2, so the "
           "HANDOVER claim 'elasticity 0.1 to 1.7 on width' is a single-prompt result. Most latent diagonal "
           f"cells are positive but small ({MACROS['NmechProbeLatWPosDiag']} and {MACROS['NmechProbeLatHPosDiag']} "
           f"of {MACROS['NmechProbeDiagCells']} for width and height); the large prompt-0 low-band values do not "
           "replicate, and prompts 1 and 2 show large negative low-band width responses."])

    fdev_row, fdev, fconfig = probe(SCRATCH / "probe" / "4096x4096" / "mtf.csv", rows)
    if fdev_row != 0:
        raise ValueError("FLUX.1-dev probe is not prompt 0")
    put("NmechFdevGuidance", fmt(fconfig["guidance"], 1))
    contrast = []
    for key, tag in KEYS.items():
        d, o, diag = diag_off(fdev[key])
        contrast.append((f"{tag}: FLUX.1-dev diag, off, top band; Krea prompt 0 top band",
                         [put(f"NmechFdev{tag}Diag", fmt(d, 2)), put(f"NmechFdev{tag}Off", fmt(o, 2)),
                          put(f"NmechFdev{tag}Top", fmt(diag[-1], 2)), f"NmechProbe{tag}TopA"]))
    top = fdev[("latent", "width")][5, 5]
    put("NmechFdevLatWTopPowerFactor", fmt(1.5 ** top, 2))
    group("c2", "(c') FLUX.1-dev (lattice regime) versus Krea probe, prompt 0, factor 1.5",
          "`probe/4096x4096/mtf.csv` (FLUX.1-dev) and `probe_krea/4096x4096/mtf.csv` (Krea), same fields; "
          "guidance from each `baseline/**/*.json` config.",
          [("guidance FLUX.1-dev, Krea; FLUX.1-dev latent width top-band power factor 1.5^e",
            ["NmechFdevGuidance", "NmechProbeGuidance", "NmechFdevLatWTopPowerFactor"])] + contrast,
          ["Top band = input octave 5 (0.125-0.25 cycles per token). Krea diag and off for prompt 0 are the "
           "`NmechProbe*A` macros above.",
           f"Not a pure checkpoint contrast: guidance ({MACROS['NmechFdevGuidance']} vs "
           f"{MACROS['NmechProbeGuidance']}) and code revision ({fconfig['code_revision'][:7]} vs "
           f"{configs[0]['code_revision'][:7]}) also differ, and the sidecars do not record which transformer "
           "was loaded (identity comes from the directory and notes).",
           "FLUX.1-dev at 4096^2 is in the dot-lattice failure regime; one prompt, one seed."])


# ----------------------------------------------------------------------------- (d) CPU SEGA study

def cpu_section():
    stats = json.loads((CPU_STUDY / "stats20.json").read_text())
    k4 = json.loads((CPU_STUDY / "stats_4k.json").read_text())
    blur = json.loads((CPU_STUDY / "blur20.json").read_text())
    source = (CPU_STUDY / "sega_stats.py").read_text()
    encode = (CPU_STUDY / "encode.py").read_text()
    grid4k = (CPU_STUDY / "sega_4k_grid.py").read_text()
    # Provenance: the study implements the SEGA paper equations, not the released code.
    for needle in ("np.tanh(z) - np.tanh(z).mean()", "1 - sig * s", "np.clip(L * theta / (2 * np.pi), 1, L // 2 - 1)",
                   "xt.mean(0)", "np.linspace(1, 0, 29)[:-1]"):
        if needle not in source:
            raise ValueError(f"CPU study source lacks {needle!r}")
    if "0.15" in source or "AutoencoderKLFlux2" not in encode or "gauss_field" not in grid4k:
        raise ValueError("CPU study provenance changed")
    real = stats["real"]
    put("NmechCpuN", str(stats["N"]))
    put("NmechCpuChannels", str(stats["C"]))
    put("NmechCpuGrid", str(stats["L"]))
    put("NmechCpuStaticPct", fmt(100 * real["R2_static_tilt"], 0))
    put("NmechCpuStaticTimePct", fmt(100 * real["R2_time_only"], 0))
    put("NmechCpuContentPct", fmt(100 * real["content_share"], 1))
    put("NmechCpuGenericProfileCorr", fmt(stats["generic_prior_vs_real_time_profile_corr"], 3))
    put("NmechCpuGenericRsq", fmt(stats["R2_real_explained_by_generic_prior"], 2))
    put("NmechCpuSigmaTimeRsq", fmt(stats["R2_sigma_time_only"], 2))
    put("NmechCpuSigmaGenericRsq", fmt(stats["R2_sigma_generic_prior"], 2))
    put("NmechCpuChanMeanPct", fmt(100 * stats["chanmean_power_fraction"], 1))
    put("NmechCpuDims", str(stats["dims_total"]))
    put("NmechCpuCollapsedNative", str(stats["dims_collapsed_to_lowest_bin"]))
    put("NmechCpuCollapsedFourK", str(k4["dims_collapsed"]))
    put("NmechCpuFourKStaticRsq", fmt(k4["R2_static"], 2))
    late = k4["M_profile_late"]
    hi, lo = late[0] ** 2, late[-1] ** 2
    put("NmechCpuLateLocalGain", fmt(hi, 1))
    put("NmechCpuLateRestGain", fmt(lo, 2))
    put("NmechCpuLateGainRatio", fmt(hi / lo, 0))
    group("d1", "(d) CPU SEGA study, FLUX.2 VAE latents of DIV2K crops",
          "`research_reviews/sega_2605.22668_2026-10-14/stats20.json` fields `N`, `C`, `L`, `real.R2_static_tilt`, "
          "`real.R2_time_only`, `real.content_share`, `generic_prior_vs_real_time_profile_corr`, "
          "`R2_real_explained_by_generic_prior`, `R2_sigma_time_only`, `R2_sigma_generic_prior`, "
          "`chanmean_power_fraction`, `dims_collapsed_to_lowest_bin`; `stats_4k.json` `dims_collapsed`, `R2_static`, "
          "`M_profile_late` (gain = M^2, first and last entries).",
          [("crops, channels, grid; static, static+time, content pct; channel-mean share of latent power pct",
            ["NmechCpuN", "NmechCpuChannels", "NmechCpuGrid", "NmechCpuStaticPct", "NmechCpuStaticTimePct",
             "NmechCpuContentPct", "NmechCpuChanMeanPct"]),
           ("generic 1/f prior: time-profile corr, R2 of real; flatness gate R2 by timestep, by generic prior",
            ["NmechCpuGenericProfileCorr", "NmechCpuGenericRsq", "NmechCpuSigmaTimeRsq", "NmechCpuSigmaGenericRsq"]),
           ("dims; collapsed at native grid, at 4096^2 strong NTK; 4096^2 static R2; late logit-gain factor "
            "local, rest, ratio", ["NmechCpuDims", "NmechCpuCollapsedNative", "NmechCpuCollapsedFourK",
                                   "NmechCpuFourKStaticRsq", "NmechCpuLateLocalGain", "NmechCpuLateRestGain",
                                   "NmechCpuLateGainRatio"])],
          ["Uses the SEGA paper equations (M = 1 - sigma s, s = tanh z - mean, alpha = 1, physical bin k = L/T clipped, "
           "latent channel mean, linear unshifted t), not released SEGA (alpha 0.15, tanh 1.5z, clamp, x_embedder "
           "input, rank map). Gain factors exclude the reference m_ref^2.",
           "Different latent model: FLUX.2 VAE, 128 channels after 2x2 patchify, forward-noised clean latents rather "
           "than sampler trajectories. Content share includes noise-draw variance, so it is an upper bound.",
           "The 4096^2 numbers use 6 synthetic 1/f Gaussian fields, not real latents."])

    ts = blur["ts"]
    blurs = blur["blurs"]
    if blurs[0] != 0 or ts[0] != 0.15 or ts[-1] != 0.6:
        raise ValueError("Unexpected blur sweep layout")
    low, high = str(ts[0]), str(ts[-1])
    sega_low = [v[0] for v in blur["sega"][low]]
    ours_low = [v[0] for v in blur["ours"][low]]
    sega_high = [v[0] for v in blur["sega"][high]]
    ours_high = [v[0] for v in blur["ours"][high]]
    put("NmechBlurN", str(blur["n_test"]))
    put("NmechBlurMax", fmt(blurs[-1], 0))
    put("NmechBlurLowSigma", fmt(ts[0], 2))
    put("NmechBlurHighSigma", fmt(ts[-1], 1))
    put("NmechBlurSegaLowSharp", fmt(sega_low[0], 2))
    put("NmechBlurSegaLowBlur", fmt(sega_low[-1], 2))
    put("NmechBlurSegaChangePct", fmt(100 * (sega_low[-1] / sega_low[0] - 1), 0))
    put("NmechBlurConstLowSharp", fmt(ours_low[0], 2))
    put("NmechBlurConstLowBlur", fmt(ours_low[-1], 2))
    put("NmechBlurConstLowPeak", fmt(max(ours_low), 2))
    put("NmechBlurConstFold", fmt(ours_low[-1] / ours_low[0], 1))
    put("NmechBlurSegaHighSharp", fmt(sega_high[0], 2))
    put("NmechBlurSegaHighBlur", fmt(sega_high[-1], 2))
    put("NmechBlurConstHighSharp", fmt(ours_high[0], 2))
    put("NmechBlurConstHighBlur", fmt(ours_high[-1], 3))
    group("d2", "(d) Blur sweep: local-minus-rest log-gain contrast versus known blur",
          "`research_reviews/sega_2605.22668_2026-10-14/blur20.json` fields `blurs`, `ts`, `n_test`, "
          "`sega[t][blur][0]`, `ours[t][blur][0]` (mean over test crops).",
          [("test crops, max blur (tokens), low and high t", ["NmechBlurN", "NmechBlurMax", "NmechBlurLowSigma",
                                                              "NmechBlurHighSigma"]),
           ("low t: SEGA no blur, max blur, change pct; constancy no blur, max blur, peak, fold",
            ["NmechBlurSegaLowSharp", "NmechBlurSegaLowBlur", "NmechBlurSegaChangePct", "NmechBlurConstLowSharp",
             "NmechBlurConstLowBlur", "NmechBlurConstLowPeak", "NmechBlurConstFold"]),
           ("at high t: SEGA no blur, max blur; constancy no blur, max blur",
            ["NmechBlurSegaHighSharp", "NmechBlurSegaHighBlur", "NmechBlurConstHighSharp", "NmechBlurConstHighBlur"])],
          ["Contrast = mean log logit gain over dims with wavelength < 16 tokens minus the rest (native grid, base "
           "10000). Paper-equation SEGA; the constancy law is the early slope-target design with an ORACLE clean "
           f"latent and tau = 1; n = {MACROS['NmechBlurN']} test crops (the other half calibrate the slope)."])


# ----------------------------------------------------------------------------- (e) static tilt failure

def pilot_section(rows):
    data = list(csv.DictReader(PILOT_SCORES.open()))
    names = []
    for method, tag in (("sega_static", "SegaStatic"), ("sega_official", "SegaOfficial"),
                        ("sega_official_static", "SegaOfficialStatic"), ("sega", "SegaPaper")):
        picked = [r for r in data if r["method"] == method]
        prompt_set = sorted(rows[r["prompt_id"]] for r in picked)
        if prompt_set != list(DEV) or {r["seed"] for r in picked} != {"0"} \
                or {r["res"] for r in picked} != {"4096x4096"}:
            raise ValueError(f"Pilot {method} is not the 8 dev prompts at seed 0, 4096^2")
        names.append(put(f"NmechPilotClip{tag}", fmt(np.mean([float(r["clip"]) for r in picked]), 1)))
    put("NmechPilotN", str(len(DEV)))
    gains = []
    for _, record in sidecars("pilot/sega_static", rows, DEV):
        g = np.array([s["controller_log_g"] for s in record["gains"]], dtype=np.float64)
        if np.abs(g - g[0]).max() != 0:
            raise ValueError("sega_static gains vary over steps")
        gains.append(np.exp(g[0, INDEX_DIMS:]))
    gains = np.array(gains)
    if np.abs(gains - gains[0]).max() > 1e-12:
        raise ValueError("sega_static gains vary over prompts")
    put("NmechSegaStaticGainMax", fmt(gains.max(), 2))
    put("NmechSegaStaticGainMin", fmt(gains.min(), 2))
    group("e1", "(e) SEGA-paper static tilt failure (dev pilot)",
          "`eval_krea/pilot/4096x4096/per_image.csv` column `clip` by `method`; gains from "
          "`out_krea/pilot/sega_static/4096x4096/*.json` `gains[t].controller_log_g`.",
          [("n; CLIP sega_static, sega_official, sega_official_static, sega (paper)", ["NmechPilotN"] + names),
           ("sega_static logit gain max, min (fixed over steps and prompts); compare NmechSegaGainMax",
            ["NmechSegaStaticGainMax", "NmechSegaStaticGainMin"])],
          ["Dev prompts 0-7 only, seed 0; selection-set evidence, not a held-out result."])


# ----------------------------------------------------------------------------- (f) CR14 and CR17

def diagnostics_section(rows):
    """Only the two later diagnostics that bear directly on per-band allocation (dev prompts)."""
    review = json.loads((NOTES / "continuation3" / "independent_review.json").read_text())
    if sorted(rows[p] for p in review["prompt_ids"]) != list(DEV):
        raise ValueError("CR14 is not the 8 dev prompts")
    arms = {r["method"]: r for r in review["rows"]}
    if set(arms) != {"sega_official", "sega_rms_uniform", "sega_rms_normalized", "sega_ctm_lag"} \
            or any(r["n"] != len(DEV) for r in arms.values()) \
            or any(v != 0 for v in arms["sega_official"]["paired_delta"].values()) \
            or review["selected"] != ["sega_rms_normalized"]:
        raise ValueError("CR14 review layout changed")
    put("NmechCrFourteenN", str(len(DEV)))
    table = []
    for method, tag, label in (("sega_rms_uniform", "Uniform", "uniform (RMS kept, shape removed)"),
                               ("sega_rms_normalized", "Norm", "RMS-normalized (shape kept, RMS rise removed)"),
                               ("sega_ctm_lag", "Lag", "lag-aligned constancy correction on SEGA")):
        delta = arms[method]["paired_delta"]
        names = [put(f"NmechCrFourteen{tag}Ir", fmt(delta["imagereward"], 3)),
                 put(f"NmechCrFourteen{tag}Musiq", fmt(delta["musiq"], 2)),
                 put(f"NmechCrFourteen{tag}Clip", fmt(delta["clip"], 2)),
                 put(f"NmechCrFourteen{tag}Pick", fmt(delta["pickscore"], 2)),
                 put(f"NmechCrFourteen{tag}Crop", fmt(delta["musiq_crop1024"], 2))]
        table.append((f"{label} minus SEGA: IR, MUSIQ, CLIP, PickScore, crop MUSIQ", names))
    repeat = review["pilot_reproducibility"]["fresh_minus_pilot"]
    replay = json.loads((NOTES / "continuation3" / "existing_dev_diagnostic.json").read_text())["rms_controls_replay"]
    rise = (replay["official_axis_rms_mean"] / replay["reference_qk_gain"]) ** 2 - 1
    table.append(("n; SEGA repeat (fresh minus pilot) IR, MUSIQ; reference q/k gain; SEGA per-axis q/k RMS "
                  "mean, max; implied mean logit rise pct",
                  ["NmechCrFourteenN", put("NmechCrFourteenRepeatIr", fmt(repeat["imagereward"], 3)),
                   put("NmechCrFourteenRepeatMusiq", fmt(repeat["musiq"], 2)),
                   put("NmechCrFourteenRefQk", fmt(replay["reference_qk_gain"], 3)),
                   put("NmechCrFourteenAxisRmsMean", fmt(replay["official_axis_rms_mean"], 3)),
                   put("NmechCrFourteenAxisRmsMax", fmt(replay["official_axis_rms_max"], 3)),
                   put("NmechCrFourteenTempRisePct", fmt(100 * rise, 1))]))
    group("f1", "(f) CR14 allocation controls at fixed temperature (dev)",
          "`.agent/notes/continuation3/independent_review.json` `rows[method].paired_delta`, "
          "`pilot_reproducibility.fresh_minus_pilot`; `existing_dev_diagnostic.json` `rms_controls_replay`.",
          table,
          ["Dev prompts 0-7, seed 0, 8 images per arm, no CI; the SEGA repeat shift is as large as the arm "
           "effects. Only the RMS-normalized arm passed the frozen gate. The RMS replay uses the original dev "
           "SEGA trajectories, not generated outputs."])

    cr17 = json.loads((NOTES / "response_cr17_interpretation.json").read_text())
    axis = [r for r in cr17["rows"] if r["direction"].startswith("axis")]
    scalar = [r for r in cr17["rows"] if not r["direction"].startswith("axis")]
    agg = cr17["aggregates"]
    for block, key in ((axis, "axis_octave"), (scalar, "spatial_scalar")):
        err = [r["relative_prediction_error"] for r in block]
        if len(block) != agg[key]["rows"] or abs(np.median(err) - agg[key]["nominal_error_median"]) > 1e-12 \
                or sum(e < 1 for e in err) != agg[key]["strictly_below_one_count"]:
            raise ValueError(f"CR17 {key} rows do not reproduce the archived aggregate")
    prompts = sorted({r["prompt_index"] for r in cr17["rows"]})
    steps = sorted({r["step"] for r in cr17["rows"]})
    err = np.array([r["relative_prediction_error"] for r in axis])
    off = np.array([r["off_axis_response_norm_fraction"] for r in axis])
    serr = np.array([r["relative_prediction_error"] for r in scalar])
    group("f2", "(f) CR17 half-step response prediction (dev)",
          "`.agent/notes/response_cr17_interpretation.json` `rows[*]` (`direction`, `relative_prediction_error`, "
          "`off_axis_response_norm_fraction`, `max_absolute_half_power_change_percent`), checked against "
          "`aggregates`.",
          [("prompts; axis-octave rows; error med, min, max; rows below 1; off-axis share min, max pct; max "
            "band-power change pct", [put("NmechCrSeventeenN", str(len(prompts))),
                                      put("NmechCrSeventeenAxisRows", str(len(axis))),
                                      put("NmechCrSeventeenErrMed", fmt(np.median(err), 2)),
                                      put("NmechCrSeventeenErrMin", fmt(err.min(), 2)),
                                      put("NmechCrSeventeenErrMax", fmt(err.max(), 2)),
                                      put("NmechCrSeventeenBelowOne", str(int((err < 1).sum()))),
                                      put("NmechCrSeventeenOffAxisMin", fmt(100 * off.min(), 0)),
                                      put("NmechCrSeventeenOffAxisMax", fmt(100 * off.max(), 0)),
                                      put("NmechCrSeventeenPowerChangeMax", fmt(max(
                                          r["max_absolute_half_power_change_percent"] for r in axis), 1))]),
           ("scalar rows; error med, min, max; rows below 1",
            [put("NmechCrSeventeenScalarRows", str(len(scalar))),
             put("NmechCrSeventeenScalarErrMed", fmt(np.median(serr), 2)),
             put("NmechCrSeventeenScalarErrMin", fmt(serr.min(), 2)),
             put("NmechCrSeventeenScalarErrMax", fmt(serr.max(), 2)),
             put("NmechCrSeventeenScalarBelowOne", str(int((serr < 1).sum())))])],
          [f"Prompt indices {prompts} (dev), steps {steps} of real 4096^2 trajectories, seed 0. Error is relative "
           "to the zero predictor (1 = no better than predicting no change). Only one octave (11) was tested, "
           "below the token-grid DFT floor, so it does not cover the resolved rotary bands; actuator norms differ."])


# ----------------------------------------------------------------------------- output

def write_tex():
    lines = ["% Generated by paper/tables/mechanism_numbers.py. Do not edit by hand.",
             "% Mechanism numbers for Section 5.2; see paper/tables/mechanism_summary.md for sources.",
             "% Signed values use a plain hyphen-minus; typeset them in math mode."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in MACROS.items()]
    (TABLES / "num_mechanism.tex").write_text("\n".join(lines) + "\n")


def write_summary():
    lines = ["# Mechanism numbers: writer notes (not for the paper)",
             "",
             "Generated by `paper/tables/mechanism_numbers.py`; macros live in `num_mechanism.tex`. Data paths are "
             "relative to `/scratch/09032/saini_2/constancy_rope` unless they start with `research_reviews` "
             "(under `/work/09032/saini_2/vista/Projects/Exploration`) or `.agent` (project root). Steps are 0-based "
             "of 28; log gains are natural logs of the attention-logit gain; corr is the per-pair log-gain change "
             "from the global temperature. Typeset negative values in math mode.",
             "",
             "| Group | Quantity | Macros | Values |",
             "|---|---|---|---|"]
    for g in GROUPS:
        for label, names in g["rows"]:
            short = ", ".join("\\" + n for n in names)
            lines.append(f"| {g['tag']} | {label} | {short} | {', '.join(MACROS[n] for n in names)} |")
    lines += ["", "## Sources and caveats", ""]
    for g in GROUPS:
        lines.append(f"- **{g['tag']} {g['title']}.** Source: {g['source']} Caveats: {' '.join(g['caveats'])}")
    text = "\n".join(lines) + "\n"
    count = text.count("\n")
    if count >= 80:
        raise ValueError(f"Summary has {count} lines; keep it under 80")
    (TABLES / "mechanism_summary.md").write_text(text)
    return count


def main():
    rows = prompt_rows()
    gate_section(rows)
    sega_section(rows)
    probe_section(rows)
    cpu_section()
    pilot_section(rows)
    diagnostics_section(rows)
    write_tex()
    count = write_summary()
    print(f"Wrote {len(MACROS)} macros to {TABLES / 'num_mechanism.tex'} and a {count}-line summary.")
    for g in GROUPS:
        print(f"\n== {g['title']}")
        for label, names in g["rows"]:
            print(f"  {label}: " + ", ".join(f"{n[5:]}={MACROS[n]}" for n in names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
