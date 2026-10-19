#!/usr/bin/env python
"""Independently check camera-ready numbers against raw scratch scores.

Run from any directory with the project's crope Python environment. This script
does not import or run the table generator. It reads the original evaluation
CSVs, pairs prompt IDs, and writes only number_audit.json and number_audit.md
beside itself. The bootstrap matches the declared 2,000 draws and seed 0.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
SCRATCH = Path("/scratch/09032/saini_2/constancy_rope")
EVAL = SCRATCH / "eval_krea"
METRICS = {
    "ir": ("imagereward", 3), "pick": ("pickscore", 2), "clip": ("clip", 2),
    "musiq": ("musiq", 2), "musiqc": ("musiq_crop1024", 2),
    "iqa": ("clipiqa", 3), "iqac": ("clipiqa_crop1024", 3),
    "spec": ("spectrum_log_rmse", 3), "slope": ("spectrum_slope_error", 3),
    "tile": ("tiling1024", 4),
}
LOWER = {"spec", "slope", "tile"}
CONTRASTS = {
    "ntk": ("ntk2", "sega_official"),
    "tsb": ("temp_only_sbase", "sega_official"),
    "dype": ("dype", "sega_official"),
    "ctm": ("constancy_tm_sbase", "sega_official"),
    "sctm": ("sega_ctm_v2", "sega_official"),
    "alloc": ("sega_official", "temp_only_sbase"),
    "basetemp": ("temp_only_sbase", "ntk2"),
    "ctmtsb": ("constancy_tm_sbase", "temp_only_sbase"),
}
FOCUS = [
    ("ntk", "ir"), ("ntk", "musiq"), ("ntk", "musiqc"),
    ("tsb", "ir"), ("tsb", "musiq"),
    ("alloc", "musiq"), ("alloc", "iqa"),
    ("dype", "ir"), ("dype", "pick"), ("dype", "musiqc"), ("dype", "iqa"),
    ("sctm", "ir"), ("sctm", "musiq"), ("sctm", "pick"),
    ("sctm", "musiqc"), ("sctm", "clip"),
    ("ctm", "clip"), ("ctm", "musiq"), ("ctm", "musiqc"), ("ctm", "tile"),
    ("ctmtsb", "musiqc"), ("ctmtsb", "musiq"),
    ("ctmtsb", "spec"), ("ctmtsb", "slope"),
]
sources: dict[str, dict] = {}
checks: list[dict] = []


def source(path: Path) -> Path:
    sources[str(path)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "bytes": path.stat().st_size}
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(source(path))


def resample(x: np.ndarray) -> tuple[float, float, float]:
    """Resample indices directly, independently of the production helper."""
    x = np.asarray(x, dtype=float)
    assert len(x) and np.isfinite(x).all()
    indices = np.random.default_rng(0).integers(0, len(x), (2000, len(x)))
    means = np.sum(x[indices], axis=1) / len(x)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(np.sum(x) / len(x)), float(lo), float(hi)


def paired(data: pd.DataFrame, method: str, ref: str, key: str) -> np.ndarray:
    metric = METRICS[key][0]
    left = data[data.method == method].set_index("prompt_id")
    right = data[data.method == ref].set_index("prompt_id")
    # Preserve the published prompt order: a finite Monte Carlo interval depends
    # on it even though the underlying bootstrap distribution does not.
    ids = left.index.intersection(right.index, sort=False)
    assert len(ids) == len(left) == len(right)
    return (left.loc[ids, metric] - right.loc[ids, metric]).to_numpy(float)


def verdict(key: str, values: tuple[float, float, float]) -> str:
    mean, lo, hi = values
    if lo <= 0 <= hi:
        return "tie"
    better = mean < 0 if key in LOWER else mean > 0
    return "better" if better else "worse"


def load_root(name: str, suffix: str, order: dict, seed: int,
              end: int) -> pd.DataFrame:
    scores = read_csv(EVAL / name / "per_image.csv")
    rewards = read_csv(EVAL / "ir" / f"{name}.csv")
    keys = ["method", "prompt_id", "res"]
    data = scores.merge(rewards[keys + ["imagereward", "png"]], on=keys,
                        how="left", validate="one_to_one")
    assert data.png.notna().all(), (name, "missing ImageReward rows")
    assert all(Path(a).resolve() == Path(b).resolve()
               for a, b in zip(data.path, data.png)), name
    data["idx"] = data.prompt_id.map(order)
    assert data.idx.notna().all()
    data = data[(data.idx >= 8) & (data.idx < end) & (data.seed == seed)
                & (data.res == "4096x4096")].copy()
    data["method"] += suffix
    assert not data.duplicated(["method", "prompt_id"]).any()
    assert np.isfinite(data[[m for m, _ in METRICS.values()]].to_numpy()).all()
    return data


def main() -> int:
    macro_path = source(PAPER / "tables" / "num_results.tex")
    macro_lines = macro_path.read_text().splitlines()
    macros, locations = {}, {}
    for line_no, line in enumerate(macro_lines, 1):
        match = re.fullmatch(r"\\expandafter\\def\\csname N@(.+)\\endcsname\{(.*)\}", line)
        if match:
            macros[match[1]] = match[2]
            locations[match[1]] = f"tables/num_results.tex:{line_no}"
    texts = {}
    for name in ("abstract", "intro", "results_decomp", "results_laws", "results_more"):
        path = source(PAPER / "sec" / f"{name}.tex")
        texts[f"sec/{name}.tex"] = path.read_text().splitlines()

    def text_locations(pattern: str) -> list[str]:
        return [f"{path}:{i}" for path, lines in texts.items()
                for i, line in enumerate(lines, 1) if re.search(pattern, line)]

    def record(name: str, expected: str, raw, where: list[str], scalars: int = 1):
        checks.append(dict(macro=name, reported=macros[name], recomputed=expected,
                           unrounded=raw, pass_check=macros[name] == expected,
                           scalar_count=scalars, macro_source=locations[name],
                           text_sources=where))

    prompt_path = source(SCRATCH / "prompts" / "aesthetic4k_eval.jsonl")
    order = {json.loads(line)["id"]: i for i, line in enumerate(prompt_path.read_text().splitlines())}
    heldout = pd.concat([load_root(name, suffix, order, 0, 195) for name, suffix in
                         [("main", ""), ("mainsb", "_sbase"), ("main2", "_v2")]],
                        ignore_index=True)
    archived = read_csv(ROOT / ".agent" / "notes" / "main2" / "heldout_per_prompt.csv")
    columns = [m for m, _ in METRICS.values()]
    a = heldout.set_index(["method", "prompt_id"]).sort_index()
    b = archived.set_index(["method", "prompt_id"]).sort_index()
    assert a.index.equals(b.index)
    max_input_difference = float(np.max(np.abs(a[columns].to_numpy() - b[columns].to_numpy())))
    assert max_input_difference < 1e-9
    assert (a["idx"] == b["idx"]).all()
    registered = read_csv(ROOT / ".agent" / "notes" / "main2" / "heldout_paired_vs_sega_official.csv")
    scratch_registered = read_csv(EVAL / "heldout" / "heldout_paired_vs_sega_official.csv")
    max_registered_difference = 0.0
    scratch_summary_checks = 0

    def record_effect(prefix, method_key, metric_key, values):
        precision = METRICS[metric_key][1]
        mean, lo, hi = values
        suffix = r"\{" + method_key + r"\}\{" + metric_key + r"\}"
        point_where = text_locations(r"\\" + prefix + r"(?:dci|d|a)" + suffix)
        ci_where = text_locations(r"\\" + prefix + r"(?:dci|ci)" + suffix)
        record(f"{prefix}@d@{method_key}@{metric_key}", f"${mean:+.{precision}f}$",
               mean, point_where)
        record(f"{prefix}@ci@{method_key}@{metric_key}",
               f"[${lo:+.{precision}f}$, ${hi:+.{precision}f}$]", [lo, hi], ci_where, 2)

    for method_key, metric_key in FOCUS:
        method, ref = CONTRASTS[method_key]
        values = resample(paired(heldout, method, ref, metric_key))
        record_effect("ho", method_key, metric_key, values)
        if ref == "sega_official":
            row = registered[(registered.method == method)
                             & (registered.metric == METRICS[metric_key][0])].iloc[0]
            assert int(row.n) == 187
            discrepancy = float(np.max(np.abs(np.array(values) - row[["delta", "lo", "hi"]].to_numpy(float))))
            max_registered_difference = max(max_registered_difference, discrepancy)
            assert discrepancy < 1e-9
            assert verdict(metric_key, values) == row.verdict
            older = scratch_registered[(scratch_registered.method == method)
                & (scratch_registered.metric == METRICS[metric_key][0])]
            if len(older):
                old_difference = float(np.max(np.abs(np.array(values)
                    - older.iloc[0][["delta", "lo", "hi"]].to_numpy(float))))
                assert old_difference < 1e-9
                max_registered_difference = max(max_registered_difference, old_difference)
                scratch_summary_checks += 1
    musiq_alloc = resample(paired(heldout, *CONTRASTS["alloc"], "musiq"))[0]
    record("ho@a@alloc@musiq", f"{abs(musiq_alloc):.2f}", abs(musiq_alloc),
           text_locations(r"\\hoa\{alloc\}\{musiq\}"))

    share_table = source(PAPER / "tables" / "tab_share.tex").read_text()
    for key in ("ir", "pick", "clip", "musiqc"):
        metric = METRICS[key][0]
        pivot = heldout.pivot(index="prompt_id", columns="method", values=metric)
        ids = heldout.loc[heldout.method == "sega_official", "prompt_id"]
        data = pivot.loc[ids]
        numerator = (data.temp_only_sbase - data.ntk2).to_numpy()
        denominator = (data.sega_official - data.ntk2).to_numpy()
        idx = np.random.default_rng(0).integers(0, len(data), (2000, len(data)))
        ratios = np.mean(numerator[idx], axis=1) / np.mean(denominator[idx], axis=1)
        ratio = float(np.mean(numerator) / np.mean(denominator))
        lo, hi = np.quantile(ratios, [0.025, 0.975])
        rounded = [round(100 * v) for v in (ratio, lo, hi)]
        record(f"sh@p@{key}", str(rounded[0]), ratio * 100,
               text_locations(r"\\shp\{" + key + r"\}"))
        record(f"sh@ci@{key}", f"[{rounded[1]}, {rounded[2]}]",
               [float(lo * 100), float(hi * 100)],
               text_locations(r"\\shci\{" + key + r"\}"), 2)
        assert f"{rounded[0]}\\% \\tiny[{rounded[1]}, {rounded[2]}]" in share_table

    count = int((heldout.method == "sega_official").sum())
    record("ho@nprompts", str(count), count, text_locations(r"\\N\{ho@nprompts\}"))
    seed1 = load_root("seed1", "", order, 1, 63)
    count1 = int((seed1.method == "temp_only").sum())
    record("so@n@tsb", str(count1), count1, text_locations(r"\\N\{so@n@tsb\}"))
    values = resample(paired(seed1, "temp_only", "sega_official", "musiq"))
    record_effect("so", "tsb", "musiq", values)
    matched0 = heldout[heldout.idx.between(8, 62)].copy()
    matched0["method"] = matched0.method.map({"temp_only_sbase": "temp_only",
        "constancy_tm_sbase": "constancy_tm", "sega_official": "sega_official"})
    agreement, cells = 0, 0
    for method in ("temp_only", "constancy_tm"):
        for key in METRICS:
            v0 = verdict(key, resample(paired(matched0, method, "sega_official", key)))
            v1 = verdict(key, resample(paired(seed1, method, "sega_official", key)))
            agreement += int(v0 == v1)
            cells += 1
    record("so@agree", str(agreement), agreement, text_locations(r"\\N\{so@agree\}"))
    record("so@cells", str(cells), cells, text_locations(r"\\N\{so@cells\}"))

    # Check printed table cells too, including the precision and signed endpoints.
    main_table = source(PAPER / "tables" / "tab_main.tex").read_text()
    main_cells = re.findall(r"\\dcell[BTW]\{([^}]+(?:\}\})?)", main_table)
    assert main_cells, "could not find the printed result cells"
    clean_main = main_table.replace("^{*}", "")
    for key, metric in FOCUS:
        if key in ("ctmtsb", "alloc") or metric not in ("ir", "pick", "clip", "musiq", "musiqc", "iqa"):
            continue
        vals = resample(paired(heldout, *CONTRASTS[key], metric))
        precision = METRICS[metric][1]
        numeric_fields = "".join("{" + f"{v:+.{precision}f}" + "}" for v in vals)
        assert numeric_fields in clean_main, (key, metric, numeric_fields)

    report = {
        "status": "pass" if all(c["pass_check"] for c in checks) else "fail",
        "method": "Independent raw-CSV reconstruction; 2000 paired prompt resamples, NumPy default_rng(0), percentile 95% intervals.",
        "effect_groups": len(FOCUS), "macro_checks": len(checks),
        "scalar_numbers_checked": sum(c["scalar_count"] for c in checks),
        "scalar_numbers_with_current_main_text_use": sum(c["scalar_count"] for c in checks if c["text_sources"]),
        "raw_vs_archived_max_abs_difference": max_input_difference,
        "raw_vs_registered_paired_max_abs_difference": max_registered_difference,
        "checked_effects_in_older_scratch_summary": scratch_summary_checks,
        "failed_checks": [c["macro"] for c in checks if not c["pass_check"]],
        "provenance_note": "eval_krea/heldout/heldout_per_prompt.csv is absent, and its paired summary has 50 comparisons, predating SEGA + TM. Values were reconstructed from original main, mainsb and main2 per_image.csv files plus ImageReward CSVs, then compared with the retained work per-prompt archive, its complete 60-comparison summary, and the scratch summary where present.",
        "checks": checks, "sources": sources,
    }
    (PAPER / "qa" / "number_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Camera-ready numerical audit", "",
        f"Outcome: {report['status'].upper()}. {report['scalar_numbers_checked']} scalar numbers across {len(FOCUS)} held-out effects, four gain shares, and the seed-1 replication were checked. No paper table generator was imported or run.",
        "",
        f"The raw scratch CSVs reproduce the retained per-prompt archive to {max_input_difference:.3g} and the checked registered paired summaries to {max_registered_difference:.3g} absolute error. Every checked macro agrees at its printed precision. The checks also cover printed share-table entries and relevant main comparison cells.",
        "",
        report["provenance_note"], "",
        "Reproduce with `OMP_NUM_THREADS=8 /work/09032/saini_2/vista/library/uv-envs/crope/bin/python paper/qa/number_audit.py`. Input SHA256 hashes, source locations, unrounded results and individual comparisons are in `number_audit.json`.",
        "",
        "| Quantity | Recomputed value | Macro |", "| :-- | :-- | :-- |",
    ]
    for check in checks:
        value = check["recomputed"].replace("$", "")
        lines.append(f"| {check['macro']} | {value} | {check['macro_source']} |")
    lines += ["", "Intervals here retain the declared bootstrap seed and prompt order. Borderline interval verdicts are not equivalence tests or proof that another bootstrap draw would give the same sign."]
    (PAPER / "qa" / "number_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "effect_groups", "macro_checks",
        "scalar_numbers_checked", "scalar_numbers_with_current_main_text_use",
        "raw_vs_archived_max_abs_difference", "raw_vs_registered_paired_max_abs_difference",
        "failed_checks")}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
