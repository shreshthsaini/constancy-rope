#!/usr/bin/env python3
"""Compare three prespecified exploratory dev controls on exactly rows 0-7."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


VARIANTS = ("sega_rms_uniform", "sega_rms_normalized", "sega_ctm_lag")
METRICS = ("imagereward", "musiq", "clip", "pickscore", "musiq_crop1024")
PROVENANCE = ("model", "model_revision", "dtype", "batch_size", "torch", "diffusers",
              "derived_seed", "prompt")


def read_scores(path, method, ids, metrics, *, reference=False):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    if reference:
        rows = [r for r in rows if r["method"] == method and r["res"] == "4096x4096"]
    if len(rows) != 8 or len({r["prompt_id"] for r in rows}) != 8:
        raise ValueError("Require exactly eight unique development prompt rows")
    if {r["prompt_id"] for r in rows} != set(ids):
        raise ValueError("Scores contain missing or non-development prompt IDs")
    result, paths = {}, set()
    for row in rows:
        if row["method"] != method or row["res"] != "4096x4096" or int(row.get("seed", 0)) != 0:
            raise ValueError("Wrong score method, resolution, or seed")
        values = {m: float(row[m]) for m in metrics}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("All reported metrics must be finite")
        path = str(Path(row.get("path", row.get("png", ""))).resolve())
        if path in paths:
            raise ValueError("Duplicate score image path")
        paths.add(path)
        result[row["prompt_id"]] = {**values, "source_path": path}
    return result


def validate_pair(score, reward, method, prompt_id, expected_dir, baseline=None):
    path = Path(score["source_path"])
    if str(path) != reward["source_path"] or path.parent != Path(expected_dir).resolve():
        raise ValueError("Score paths disagree or leave the registered generation root")
    if not path.is_file():
        raise ValueError("Missing scored image")
    record = json.loads(path.with_suffix(".json").read_text())
    config = record["config"]
    expected = dict(seed=0, steps=28, guidance=4.5, res="4096x4096", method=method,
                    prompt_id=prompt_id, grid=[256, 256])
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Sidecar method or sampling protocol mismatch")
    if config.get("rope", {}).get("method") != "sega_official":
        raise ValueError("Wrong RoPE base")
    if method in VARIANTS and config.get("temperature") != "none":
        raise ValueError("Candidates must not add temperature")
    if method == "sega_ctm_lag" and (config.get("eta") != 1.0 or
                                     not config.get("profile_sha256")):
        raise ValueError("Lag control requires eta 1 and recorded native profile")
    if not record.get("complete") or Path(record.get("image", "")).resolve() != path:
        raise ValueError("Incomplete or mismatched sidecar image provenance")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != record.get("png_sha256"):
        raise ValueError("Scored image checksum differs from sidecar")
    if not config.get("implementation_sha256") or any(k not in config for k in PROVENANCE):
        raise ValueError("Missing implementation or model provenance")
    sigmas = record.get("sigmas", [])
    gains = record.get("gains", [])
    if (len(sigmas) != 28 or len(gains) != 28 or
            not all(math.isfinite(s) and 0 <= s <= 1 for s in sigmas) or
            not all(a > b for a, b in zip(sigmas, sigmas[1:])) or
            [g.get("sigma") for g in gains] != sigmas):
        raise ValueError("Invalid actual executed sigma trajectory")
    if baseline is not None:
        if sigmas != baseline["sigmas"]:
            raise ValueError("Actual executed sigmas differ from baseline")
        if any(config[k] != baseline["config"][k] for k in PROVENANCE):
            raise ValueError("Model, software, prompt, or derived seed provenance differs")
    return record


def decision(means):
    reference = means["sega_official"]
    rows = []
    for variant, values in means.items():
        delta = {m: values[m] - reference[m] for m in METRICS}
        eligible = variant != "sega_official" and delta["imagereward"] > 0 and delta["musiq"] > 0 and delta["clip"] >= -0.3
        rows.append({"variant": variant, "n": 8, **values,
                     **{f"delta_{k}": v for k, v in delta.items()}, "eligible": eligible})
    eligible = sorted((r for r in rows if r["eligible"]),
                      key=lambda r: (-r["delta_imagereward"], -r["delta_musiq"], r["variant"]))
    selected = [r["variant"] for r in eligible[:2]]
    for row in rows:
        row["selected"] = row["variant"] in selected
    return rows, selected


def analyze(eval_root, prompts, out, reference_root="pilot"):
    eval_root, out = Path(eval_root), Path(out)
    prompt_rows = [json.loads(line) for line in Path(prompts).read_text().splitlines() if line.strip()]
    ids = [str(r["id"]) for r in prompt_rows[:8]]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("Prompt manifest needs eight unique development rows")
    if reference_root not in ("pilot", "dev3_sega_official"):
        raise ValueError("Reference root must be pilot or frozen fresh dev3_sega_official")
    means, paired, baseline, manifests = {}, [], {}, []
    for method in ("sega_official", *VARIANTS):
        name = reference_root if method == "sega_official" else f"dev3_{method}"
        table = eval_root / name / ("4096x4096/per_image.csv" if name == "pilot" else "per_image.csv")
        ir = eval_root / "ir" / f"{name}.csv"
        scores = read_scores(table, method, ids, METRICS[1:], reference=name == "pilot")
        reward = read_scores(ir, method, ids, METRICS[:1], reference=name == "pilot")
        directory = eval_root.parent / "out_krea" / name / method / "4096x4096"
        for pid in ids:
            record = validate_pair(scores[pid], reward[pid], method, pid, directory, baseline.get(pid))
            if record["config"]["prompt"] != next(r["prompt"] for r in prompt_rows[:8] if str(r["id"]) == pid):
                raise ValueError("Sidecar prompt text differs from manifest")
            if method == "sega_official":
                baseline[pid] = record
            paired.append({"variant": method, "prompt_id": pid, **scores[pid], **reward[pid]})
        means[method] = {m: sum(r[m] for r in paired if r["variant"] == method) / 8 for m in METRICS}
        for source in (table, ir):
            manifests.append({"path": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    rows, selected = decision(means)
    report = {"purpose": "Exploratory development diagnostics only; not held-out efficacy evidence",
              "reference_root": reference_root, "prompt_ids": ids,
              "rule": "IR delta > 0 AND full-res MUSIQ delta > 0 AND CLIP delta >= -0.3",
              "ranking": "IR descending, MUSIQ descending, variant name ascending; at most two",
              "selected": selected, "automatic_heldout_authorized": False,
              "rows": rows, "score_sources": manifests}
    out.mkdir(parents=True, exist_ok=True)
    for name, records in (("dev_table.csv", rows), ("dev_per_prompt.csv", paired)):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (out / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--reference-root", choices=("pilot", "dev3_sega_official"), default="pilot")
    args = parser.parse_args()
    print(json.dumps(analyze(args.eval_root, args.prompts, args.out, args.reference_root), indent=2))
