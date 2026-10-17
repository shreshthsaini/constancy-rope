"""Validate and analyze the frozen 64-prompt confirmation after dev3 selection.

This command never selects new candidates, drops pairs, or submits GPU work.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dev3_analysis import METRICS, VARIANTS, decision, validate_pair
from crope.utils import image_seed

PRIMARY = ("imagereward", "musiq")
DRAWS = 10000
BOOTSTRAP_SEED = 0
PROMPT_SHA256 = "965992616563a274ea275d8ef0a7083415800adb5c88d494a37fb272a996e6f2"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_candidates(selection):
    if selection.get("reference_root") != "dev3_sega_official":
        raise ValueError("Confirmation requires the fresh development baseline")
    rows = selection["rows"]
    if len(rows) != 4 or {r["variant"] for r in rows} != {"sega_official", *VARIANTS}:
        raise ValueError("Require the complete four-method development decision")
    means = {r["variant"]: {m: float(r[m]) for m in METRICS} for r in rows}
    if not all(math.isfinite(v) for values in means.values() for v in values.values()):
        raise ValueError("Development means must be finite")
    _, selected = decision(means)
    if selected != selection.get("selected"):
        raise ValueError("Recorded selection differs from the frozen development rule")
    if not selected:
        raise ValueError("No eligible development candidate; confirmation is not indicated")
    return selected


def read_scores(path, method, ids, metrics):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 64 or {r["prompt_id"] for r in rows} != set(ids):
        raise ValueError("Require all 64 unique frozen confirmation prompts")
    result, paths = {}, set()
    for row in rows:
        if row["method"] != method or row["res"] != "4096x4096" or int(row.get("seed", 0)) != 0:
            raise ValueError("Wrong score method, resolution, or seed")
        values = {m: float(row[m]) for m in metrics}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("All confirmation metrics must be finite")
        raw_path = row.get("path", row.get("png", ""))
        if not raw_path:
            raise ValueError("Missing scored image path")
        image_path = str(Path(raw_path).resolve())
        if image_path in paths:
            raise ValueError("Duplicate scored image path")
        paths.add(image_path)
        result[row["prompt_id"]] = {**values, "source_path": image_path}
    return result


def paired_statistics(values, selected):
    """One common bootstrap index matrix across all metrics and candidates."""
    baseline = np.asarray(values["sega_official"], dtype=float)
    if baseline.shape != (64, len(METRICS)) or not np.isfinite(baseline).all():
        raise ValueError("Require a finite 64 by metric baseline")
    if not 1 <= len(selected) <= 2 or len(set(selected)) != len(selected):
        raise ValueError("Require one or two distinct prespecified candidates")
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 64, (DRAWS, 64))
    family_size = len(PRIMARY) * len(selected)
    tail = .05 / (2 * family_size)
    rows = []
    for method in selected:
        candidate = np.asarray(values[method], dtype=float)
        if candidate.shape != baseline.shape or not np.isfinite(candidate).all():
            raise ValueError("Require complete finite paired candidate values")
        delta = candidate - baseline
        draws = delta[indices].mean(axis=1)
        for j, metric in enumerate(METRICS):
            lo, hi = np.quantile(draws[:, j], [.025, .975])
            adjusted = np.quantile(draws[:, j], [tail, 1 - tail]) if metric in PRIMARY else [None, None]
            rows.append(dict(method=method, metric=metric, n=64,
                             baseline_mean=float(baseline[:, j].mean()),
                             candidate_mean=float(candidate[:, j].mean()),
                             delta=float(delta[:, j].mean()), lo=float(lo), hi=float(hi),
                             primary=metric in PRIMARY,
                             primary_adjusted_lo=None if adjusted[0] is None else float(adjusted[0]),
                             primary_adjusted_hi=None if adjusted[1] is None else float(adjusted[1])))
    conclusions = {}
    for method in selected:
        own = [r for r in rows if r["method"] == method]
        conclusions[method] = {
            "both_primary_adjusted_lower_bounds_positive": all(
                r["primary_adjusted_lo"] > 0 for r in own if r["primary"]),
            "secondary_negative_nominal_intervals": [r["metric"] for r in own
                                                     if not r["primary"] and r["hi"] < 0],
            "claim_scope": "64 fresh project prompts, seed 0, Krea at 4096; no universal or SOTA claim",
        }
    return rows, conclusions


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_hardware(root, method, ids):
    runs = [json.loads(p.read_text()) for p in sorted((root / 'generation_runs').glob('*.json'))]
    complete = [r for r in runs if r.get('complete')]
    if not complete or any(r.get('method') != method for r in complete):
        raise ValueError('Missing or mismatched completed hardware receipts')
    if {pid for r in complete for pid in r['prompt_ids']} != set(ids):
        raise ValueError('Hardware receipts do not cover all confirmation prompts')
    # An interrupted chunk can leave reusable images. Its device must agree too.
    hardware = {(r['gpu'], r['cuda']) for r in runs}
    if len(hardware) != 1 or not all(next(iter(hardware))):
        raise ValueError('Confirmation contains mixed or missing GPU/CUDA provenance')
    return next(iter(hardware))


def analyze(eval_root, prompts, development, out):
    eval_root, prompts, development, out = map(Path, (eval_root, prompts, development, out))
    if sha256(prompts) != PROMPT_SHA256:
        raise ValueError("Confirmation prompt manifest differs from its frozen hash")
    prompt_rows = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    prompt_map = {str(r["id"]): r["prompt"] for r in prompt_rows}
    ids = list(prompt_map)
    if len(prompt_rows) != 64 or len(ids) != 64:
        raise ValueError("Require exactly 64 unique confirmation prompts")
    selection_path = development / "selection.json"
    selected = selected_candidates(json.loads(selection_path.read_text()))
    provenance = json.loads((development / "generation_provenance.json").read_text())
    with (development / "dev_per_prompt.csv").open() as stream:
        dev_rows = list(csv.DictReader(stream))
    dev_configs = {}
    for method in ("sega_official", *selected):
        row = next(r for r in dev_rows if r["variant"] == method)
        dev_configs[method] = json.loads(Path(row["source_path"]).with_suffix(".json").read_text())["config"]
    baseline, values, paired, sources, hardware = {}, {}, [], [], {}
    changing = {"prompt_id", "prompt", "derived_seed", "code_revision"}
    for method in ("sega_official", *selected):
        name = f"confirm64_{method}"
        generation_root = eval_root.parent / "out_krea" / name
        if json.loads((generation_root / "generation_manifest.json").read_text()) != provenance:
            raise ValueError("Actual transformer provenance changed since development")
        hardware[method] = validate_hardware(generation_root, method, ids)
        if hardware[method] != hardware['sega_official']:
            raise ValueError('Candidate and fresh baseline used different GPU/CUDA hardware')
        table, ir = eval_root / name / "per_image.csv", eval_root / "ir" / f"{name}.csv"
        scores = read_scores(table, method, ids, METRICS[1:])
        rewards = read_scores(ir, method, ids, METRICS[:1])
        expected_config = {k: v for k, v in dev_configs[method].items() if k not in changing}
        values[method] = []
        for pid in ids:
            record = validate_pair(scores[pid], rewards[pid], method, pid,
                                   generation_root / method / "4096x4096", baseline.get(pid))
            config = record["config"]
            if config["prompt"] != prompt_map[pid]:
                raise ValueError("Generated prompt text differs from frozen confirmation caption")
            if config["derived_seed"] != image_seed(pid, 0):
                raise ValueError("Derived seed differs from the frozen prompt-seed convention")
            if {k: v for k, v in config.items() if k not in changing} != expected_config:
                raise ValueError("Controller, model, software, or sampling settings changed since development")
            if method == "sega_official":
                baseline[pid] = record
            row = {"method": method, "prompt_id": pid, **scores[pid], **rewards[pid]}
            paired.append(row)
            values[method].append([row[m] for m in METRICS])
        sources.extend({"path": str(p.resolve()), "sha256": sha256(p)} for p in (table, ir))
    rows, conclusions = paired_statistics(values, selected)
    protocol = {"selected": selected, "n": 64, "prompt_sha256": PROMPT_SHA256,
                "development_selection_sha256": sha256(selection_path),
                "bootstrap_draws": DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
                "primary_metrics": PRIMARY, "primary_family_size": 2 * len(selected),
                "primary_adjusted_coverage": 1 - .05 / (2 * len(selected)),
                "uncertainty": "Paired percentile bootstrap; Bonferroni-adjusted primary intervals are approximate, not exact finite-sample guarantees. Secondary intervals are nominal 95%.",
                "conclusions": conclusions, "score_sources": sources, "hardware": hardware,
                "confirmation_does_not_select_further_variants": True}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "confirmation_per_prompt.csv", paired)
    write_csv(out / "confirmation_paired.csv", rows)
    (out / "analysis_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "generation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("eval_root", "prompts", "development", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.eval_root, args.prompts, args.development, args.out), indent=2))
