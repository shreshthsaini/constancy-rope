#!/usr/bin/env python3
"""Preregistered E1/E2/E3 analysis, with explicit missingness and durable inputs.

Usage: strengthen_analysis.py MANIFEST.json OUT_DIRECTORY
       strengthen_analysis.py --replay OUT_DIRECTORY NEW_OUT_DIRECTORY

The manifest specifies block, prompts, resolution, preregistration, and sources.
Each source has name, per_image, imagereward, methods (raw allowlist), aliases,
and optional seeds/prompt_ids. Optional expected_config and expected_by_method
use dotted keys inside generation config. Source-specific expected_config is
also supported. Evaluator metrics.json and summary.csv can be listed under
artifacts; these are archived, including distributional FID/KID summaries.
ImageReward is joined by resolved PNG path, since its CSV lacks a seed column.
Full raw inputs are content-addressed on work. Replay needs no scratch files.
An incomplete block exits 2 after writing descriptive tables and blocked tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

METRICS = ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024", "clipiqa",
           "clipiqa_crop1024", "spectrum_log_rmse", "spectrum_slope_error", "tiling1024"]
PRIMARY = ["imagereward", "pickscore", "musiq"]
LOWER = {"spectrum_log_rmse", "spectrum_slope_error", "tiling1024"}
METHODS = {
    "E1": ["sega_official", "temp_only", "sega_ctm", "constancy_tm", "dype", "ntk2"],
    "E2": ["sega_official", "temp_only", "constancy_tm", "sega_ctm"],
    "E3": ["sega_official", "temp_only", "ntk2"],
}
KEYS = ["method", "prompt_id", "seed"]
SUMMARY_COLUMNS = ["analysis", "method", "metric", "n", "mean", "lo", "hi", "complete_block"]
PAIRED_COLUMNS = ["analysis", "method", "reference", "metric", "n", "delta", "lo", "hi",
                  "direction", "complete_block"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_prompts(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids) or any(not isinstance(row["prompt"], str) for row in rows):
        raise ValueError("Prompt manifest needs unique IDs and exact text")
    return [{**row, "id": str(row["id"])} for row in rows]


def registered_design(spec, prompts):
    block = spec["block"]
    if block not in METHODS:
        raise ValueError(f"Unknown block {block!r}")
    if block == "E2":
        if len(prompts) != 195:
            raise ValueError("E2 requires the original 195-row manifest; select rows 8-194")
        chosen, seeds = prompts[8:195], [0, 1, 2]
    else:
        if len(prompts) != 150:
            raise ValueError(f"{block} requires exactly 150 fresh prompts")
        chosen, seeds = prompts, [0]
    metrics = spec.get("metrics", METRICS)
    if len(metrics) != len(set(metrics)) or not set(PRIMARY).issubset(metrics):
        raise ValueError("Metrics must be unique and include all three primary outcomes")
    if block != "E3" and not set(METRICS).issubset(metrics):
        raise ValueError("E1/E2 must report all registered per-image metrics")
    if block != "E3" and spec["resolution"] != "4096x4096":
        raise ValueError("E1/E2 resolution must be 4096x4096")
    return {"block": block, "methods": METHODS[block], "seeds": seeds,
            "prompt_ids": [row["id"] for row in chosen], "metrics": metrics,
            "resolution": spec["resolution"]}


def boot(values):
    """One fresh RNG per call, float64, manifest order, exactly 2,000 draws."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap needs nonempty, finite one-dimensional values")
    draws = np.random.default_rng(0).choice(values, (2000, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), *map(float, np.percentile(draws, [2.5, 97.5], method="linear"))


def dotted(config, name):
    value = config
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Missing config field {name}")
        value = value[part]
    return value


def int_seed(value):
    number = float(value)
    if not np.isfinite(number) or int(number) != number:
        raise ValueError(f"Invalid seed {value!r}")
    return int(number)


def archive_file(path, out, records, role):
    path = Path(path)
    digest = sha256(path)
    target = out / "inputs" / f"{digest}_{path.name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) != digest:
        raise ValueError(f"Snapshot collision: {target}")
    if not target.exists():
        shutil.copyfile(path, target)
    records.append({"role": role, "source": str(path.resolve()), "sha256": digest,
                    "snapshot": str(target.relative_to(out))})
    return target


def inspect_sidecar(row, prompt, spec, source):
    path = Path(row["path"]).resolve()
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"Missing image/sidecar: {path}")
    metadata = json.loads(sidecar.read_text())
    if metadata.get("complete") is not True:
        raise ValueError(f"Incomplete image: {path}")
    config = metadata["config"]
    expected = {"method": row["raw_method"], "prompt_id": row["prompt_id"],
                "prompt": prompt, "seed": row["seed"], "res": row["res"]}
    if spec["block"] in ["E1", "E2"]:
        expected.update(steps=28, guidance=4.5, model="black-forest-labs/FLUX.1-dev",
                        model_revision="3de623fc3c33e44ffbe2bad470d0f45bccf2eb21", dtype="bfloat16", eta=1.0)
        expected.update({"scheduler.max_shift": 1.15, "scheduler.max_image_seq_len": 65536})
        method = row["method"]
        expected["rope.method"] = "sega_official" if method in ["temp_only", "constancy_tm", "sega_ctm"] else method
        if method in ["temp_only", "constancy_tm"]:
            expected["temperature"] = "sref"
        if method in ["constancy_tm", "sega_ctm"]:
            expected["profile_sha256"] = "1cd6bceb50dea48607b125681ade330e5203cc8342eeb99834c3493d41b2cc45"
    for additional in [spec.get("expected_config", {}),
                       spec.get("expected_by_method", {}).get(row["method"], {}),
                       source.get("expected_config", {})]:
        for key, value in additional.items():
            if key in expected and expected[key] != value:
                raise ValueError(f"Manifest contradicts registered expectation for {key}")
            expected[key] = value
    for key, value in expected.items():
        actual = dotted(config, key)
        if actual != value:
            raise ValueError(f"{path}: config {key}={actual!r}, expected {value!r}")
    from PIL import Image
    with Image.open(path) as image:
        height, width = map(int, row["res"].split("x"))
        if image.size != (width, height):
            raise ValueError(f"PNG dimensions disagree: {path}")
    return {"method": row["method"], "prompt_id": row["prompt_id"], "seed": row["seed"],
            "path": str(path), "sidecar": str(sidecar), "sidecar_sha256": sha256(sidecar),
            "image_bytes": path.stat().st_size, "image_mtime_ns": path.stat().st_mtime_ns,
            "recorded_png_sha256": metadata.get("png_sha256"),
            "image_hash_verified_by_analysis": False, "config": config,
            "seconds": metadata.get("seconds"), "peak_memory_bytes": metadata.get("peak_memory_bytes")}


def load_sources(spec, prompts, design, out, files):
    """Never merge ImageReward on ambiguous method/prompt keys across seeds."""
    prompt_map = {row["id"]: row["prompt"] for row in prompts}
    chosen = set(design["prompt_ids"])
    records, provenance, exclusions, issues = [], [], [], []
    for source_index, source in enumerate(spec["sources"]):
        name = source.get("name", f"source{source_index}")
        if not source.get("methods"):
            raise ValueError(f"{name}: explicit raw method allowlist required")
        aliases = source.get("aliases", {})
        for artifact in source.get("artifacts", []):
            if Path(artifact).is_file():
                archive_file(artifact, out, files, f"{name}:artifact")
            else:
                issues.append({"source": name, "reason": "missing_artifact", "path": artifact})
        pi_path, ir_path = Path(source["per_image"]), Path(source["imagereward"])
        if not pi_path.is_file():
            issues.append({"source": name, "reason": "missing_per_image", "path": str(pi_path)})
            continue
        pi_snap = archive_file(pi_path, out, files, f"{name}:per_image")
        pi = pd.read_csv(pi_snap, dtype={"prompt_id": str, "method": str, "res": str})
        if not set(KEYS + ["path", "res"]).issubset(pi.columns):
            raise ValueError(f"{name}: per_image CSV missing identity columns")
        ir_by_path = {}
        if ir_path.is_file():
            ir_snap = archive_file(ir_path, out, files, f"{name}:imagereward")
            ir = pd.read_csv(ir_snap, dtype={"prompt_id": str, "method": str, "res": str})
            for reward in ir.to_dict("records"):
                key = str(Path(reward["png"]).resolve())
                if key in ir_by_path:
                    raise ValueError(f"{name}: duplicate ImageReward image path {key}")
                ir_by_path[key] = reward
        else:
            issues.append({"source": name, "reason": "missing_imagereward", "path": str(ir_path)})
        excluded = {}
        for row in pi.to_dict("records"):
            raw_method = row["method"]
            reason = None
            if raw_method not in source["methods"]:
                reason = "method_not_in_source_allowlist"
            elif row["prompt_id"] not in prompt_map:
                raise ValueError(f"{name}: unknown prompt ID {row['prompt_id']}")
            elif row["prompt_id"] not in chosen:
                reason = "registered_dev_exclusion"
            elif "prompt_ids" in source and row["prompt_id"] not in source["prompt_ids"]:
                reason = "prompt_not_in_source_allowlist"
            elif "seeds" in source and int_seed(row["seed"]) not in source["seeds"]:
                reason = "seed_not_in_source_allowlist"
            if reason:
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            row = {**row, "method": aliases.get(raw_method, raw_method), "raw_method": raw_method,
                   "seed": int_seed(row["seed"]), "path": str(Path(row["path"]).resolve()),
                   "source": name}
            if row["method"] not in design["methods"] or row["seed"] not in design["seeds"]:
                raise ValueError(f"Unexpected canonical identity: {[row[k] for k in KEYS]}")
            if row["res"] != design["resolution"]:
                raise ValueError(f"{name}: wrong resolution for {row['path']}")
            reward = ir_by_path.get(row["path"])
            if reward is not None:
                if (reward["prompt_id"] != row["prompt_id"] or reward["res"] != row["res"] or
                        aliases.get(reward["method"], reward["method"]) != row["method"]):
                    raise ValueError(f"{name}: ImageReward identity/path disagreement")
                row["imagereward"] = reward["imagereward"]
            else:
                # A previous evaluator's optional IR column cannot replace the mend CSV.
                row["imagereward"] = np.nan
            witness = inspect_sidecar(row, prompt_map[row["prompt_id"]], spec, source)
            provenance.append(witness)
            records.append(row)
        exclusions.append({"source": name, "counts": excluded})
    data = pd.DataFrame(records, columns=None if records else KEYS + ["res", "path"] + design["metrics"])
    if data.duplicated(KEYS).any():
        bad = data.loc[data.duplicated(KEYS, keep=False), KEYS + ["path"]].to_dict("records")
        raise ValueError(f"Duplicate canonical method/prompt/seed keys: {bad}")
    if data["path"].duplicated().any():
        raise ValueError("One image path aliases more than one canonical sample")
    return data, provenance, exclusions, issues


def coverage(data, design):
    expected = {(method, pid, seed) for method in design["methods"]
                for pid in design["prompt_ids"] for seed in design["seeds"]}
    actual = set(data[KEYS].itertuples(index=False, name=None))
    if actual - expected:
        raise ValueError("Data contains unexpected registered identities")
    if data.duplicated(KEYS).any():
        raise ValueError("Duplicate canonical identities")
    missing = [{"method": m, "prompt_id": p, "seed": s} for m in design["methods"]
               for p in design["prompt_ids"] for s in design["seeds"] if (m, p, s) not in actual]
    invalid = []
    data = data.copy()
    for metric in design["metrics"]:
        if metric not in data:
            data[metric] = np.nan
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        for row in data.loc[~np.isfinite(data[metric]), KEYS].to_dict("records"):
            invalid.append({**row, "metric": metric, "reason": "missing_or_nonfinite"})
    return data, {"expected_samples": len(expected), "observed_samples": len(actual),
                  "missing_samples": missing, "invalid_metrics": invalid,
                  "complete": not missing and not invalid}


def analyses(design):
    if design["block"] == "E2":
        return {"seeds012": [0, 1, 2], "seeds12": [1, 2], "seed0": [0], "seed1": [1], "seed2": [2]}
    return {"seed0": [0]}


def summarize(data, design, complete):
    """A prompt contributes only when all requested seeds for its metric exist."""
    means, paired, prompt_means, omitted = [], [], [], []
    lookup = {tuple(row[k] for k in KEYS): row for row in data.to_dict("records")}
    for label, seeds in analyses(design).items():
        for metric in design["metrics"]:
            values = {}
            for method in design["methods"]:
                values[method] = {}
                for pid in design["prompt_ids"]:
                    group = [lookup.get((method, pid, seed), {}).get(metric, np.nan) for seed in seeds]
                    if not np.isfinite(group).all():
                        omitted.append({"analysis": label, "method": method, "prompt_id": pid,
                                        "metric": metric, "reason": "incomplete_seed_group"})
                        continue
                    mean = float(np.asarray(group, dtype=np.float64).mean())
                    values[method][pid] = mean
                    prompt_means.append({"analysis": label, "method": method, "prompt_id": pid,
                                         "metric": metric, "seeds": ",".join(map(str, seeds)), "mean": mean})
                series = list(values[method].values())
                if series:
                    mu, lo, hi = boot(series)
                    means.append({"analysis": label, "method": method, "metric": metric,
                                  "n": len(series), "mean": mu, "lo": lo, "hi": hi,
                                  "complete_block": complete})
            reference = values["sega_official"]
            for method in design["methods"]:
                if method == "sega_official":
                    continue
                diffs = [values[method][pid] - reference[pid] for pid in design["prompt_ids"]
                         if pid in values[method] and pid in reference]
                if not diffs:
                    continue
                delta, lo, hi = boot(diffs)
                direction = "unresolved"
                if lo > 0 or hi < 0:
                    direction = "better" if (delta < 0 if metric in LOWER else delta > 0) else "worse"
                paired.append({"analysis": label, "method": method, "reference": "sega_official",
                               "metric": metric, "n": len(diffs), "delta": delta, "lo": lo, "hi": hi,
                               "direction": direction, "complete_block": complete})
    return (pd.DataFrame(means, columns=SUMMARY_COLUMNS), pd.DataFrame(paired, columns=PAIRED_COLUMNS),
            pd.DataFrame(prompt_means, columns=["analysis", "method", "prompt_id", "metric", "seeds", "mean"]), omitted)


def hypothesis_outcomes(paired, design, complete):
    result = {}
    for label in analyses(design):
        rows = {(r["method"], r["metric"]): r for r in paired.to_dict("records") if r["analysis"] == label}

        def contrast(method, metric, reverse=False):
            row = rows.get((method, metric))
            if row is None:
                return {"status": "incomplete", "n": 0}
            value = {k: row[k] for k in ["n", "delta", "lo", "hi"]}
            if reverse:
                value.update(delta=-row["delta"], lo=-row["hi"], hi=-row["lo"])
            value["status"] = "eligible" if complete else "incomplete"
            return value

        h1parts = {}
        for metric, margin in [("imagereward", .05), ("pickscore", .10)]:
            row = contrast("temp_only", metric)
            if complete and row["status"] == "eligible":
                row["status"] = ("supported" if -margin < row["lo"] and row["hi"] < margin else
                                 "incompatible_with_equivalence" if row["lo"] > margin or row["hi"] < -margin
                                 else "not_established")
            row["margin"] = margin
            h1parts[metric] = row
        h1 = {"status": "supported" if all(v["status"] == "supported" for v in h1parts.values()) else
              "incomplete" if not complete else "incompatible_with_equivalence"
              if any(v["status"] == "incompatible_with_equivalence" for v in h1parts.values()) else "not_established",
              "components": h1parts}
        h2 = contrast("temp_only", "musiq", reverse=True)
        h2["contrast"] = "sega_official minus temp_only"
        if complete and h2["status"] == "eligible":
            h2["status"] = "supported" if h2["lo"] > 0 else "reverse_finding" if h2["hi"] < 0 else "unresolved"
        h3parts = {}
        for method in ["constancy_tm", "sega_ctm"]:
            row = contrast(method, "imagereward")
            if complete and row["status"] == "eligible":
                row["status"] = "supported" if row["hi"] <= .03 else "contradicted" if row["lo"] > .03 else "unresolved"
            row["upper_margin"] = .03
            h3parts[method] = row
        h3 = {"status": "supported" if all(v["status"] == "supported" for v in h3parts.values()) else
              "incomplete" if not complete else "contradicted" if any(v["status"] == "contradicted" for v in h3parts.values())
              else "unresolved", "components": h3parts}
        if design["block"] == "E3":
            h3 = {"status": "not_tested", "reason": "Controllers absent from exploratory Qwen extension"}
        result[label] = {"role": "confirmation" if design["block"] == "E1" else
                        "replication_summary" if design["block"] == "E2" else "exploratory_transfer",
                        "H1": h1, "H2": h2, "H3": h3}
    return result


def output_report(data, design, out, files, exclusions, source_issues, provenance):
    data, status = coverage(data, design)
    status["source_issues"] = source_issues
    status["complete"] = status["complete"] and not source_issues
    order = {pid: index for index, pid in enumerate(design["prompt_ids"])}
    data["manifest_index"] = data.prompt_id.map(order)
    data = data.sort_values(["method", "manifest_index", "seed"], kind="stable")
    data.to_csv(out / "canonical_per_image.csv", index=False)
    with (out / "config_provenance.jsonl").open("w") as handle:
        for witness in provenance:
            handle.write(json.dumps(witness, allow_nan=False) + "\n")
    means, paired, prompt_means, omitted = summarize(data, design, status["complete"])
    means.to_csv(out / "means.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    prompt_means.to_csv(out / "prompt_means.csv", index=False)
    write_json(out / "hypotheses.json", hypothesis_outcomes(paired, design, status["complete"]))
    write_json(out / "coverage.json", {**status, "source_exclusions": exclusions,
                                       "descriptive_omissions": omitted})
    write_json(out / "analysis_protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "design": design,
        "complete": status["complete"], "bootstrap_draws": 2000, "bootstrap_seed": 0,
        "fresh_rng_each_contrast": True, "float_dtype": "float64", "percentile_method": "linear",
        "percentiles": [2.5, 97.5], "unit": "prompt", "source_files": files,
        "analysis_source_sha256": sha256(__file__),
        "canonical_sha256": sha256(out / "canonical_per_image.csv"),
        "config_provenance_sha256": sha256(out / "config_provenance.jsonl"),
        "limitations": ["Nominal per-contrast intervals, no familywise adjustment",
                        "Incomplete blocks have descriptive complete-pair tables only",
                        "Recorded PNG hash retained; analysis checks header and config, not full PNG hash",
                        "FID/KID remain descriptive in archived evaluator artifacts"]})
    return status["complete"]


def run(spec_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(spec_path).read_text())
    files = []
    archive_file(spec_path, out, files, "source_manifest")
    prompts_path = archive_file(spec["prompts"], out, files, "prompts")
    archive_file(spec["preregistration"], out, files, "preregistration")
    prompts = read_prompts(prompts_path)
    design = registered_design(spec, prompts)
    data, provenance, exclusions, issues = load_sources(spec, prompts, design, out, files)
    return output_report(data, design, out, files, exclusions, issues, provenance)


def replay(source, out):
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve():
        raise ValueError("Replay output must differ from its immutable input archive")
    protocol = json.loads((source / "analysis_protocol.json").read_text())
    for name, field in [("canonical_per_image.csv", "canonical_sha256"),
                        ("config_provenance.jsonl", "config_provenance_sha256")]:
        if sha256(source / name) != protocol[field]:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    for record in protocol["source_files"]:
        if sha256(source / record["snapshot"]) != record["sha256"]:
            raise ValueError(f"Raw snapshot hash mismatch: {record['snapshot']}")
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for record in protocol["source_files"]:
        archive_file(source / record["snapshot"], out, files, record["role"])
    data = pd.read_csv(source / "canonical_per_image.csv", dtype={"prompt_id": str})
    provenance = [json.loads(line) for line in (source / "config_provenance.jsonl").read_text().splitlines()]
    old = json.loads((source / "coverage.json").read_text())
    return output_report(data, protocol["design"], out, files, old["source_exclusions"], old["source_issues"], provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    complete = replay(args.source, args.out) if args.replay else run(args.source, args.out)
    print(json.dumps({"out": args.out, "complete": complete}))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
