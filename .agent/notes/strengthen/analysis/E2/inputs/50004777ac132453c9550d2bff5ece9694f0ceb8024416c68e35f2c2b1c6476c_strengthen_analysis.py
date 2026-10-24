#!/usr/bin/env python3
"""Amended strengthening analysis with paired prompt bootstrap and fixed Holm families.

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
Blocks: P1 factorial, P3 fixed sweep, E1 fresh150, E2 three seeds, E3 Qwen.
E1 manifests may add heldout_confirmation={per_prompt, prompts}; this produces
the pooled H1/fresh-only H2 family. Without it no confirmatory claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

METRICS = ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024", "clipiqa",
           "clipiqa_crop1024", "spectrum_log_rmse", "spectrum_slope_error", "tiling1024"]
PRIMARY = ["imagereward", "pickscore", "musiq"]
LOWER = {"spectrum_log_rmse", "spectrum_slope_error", "tiling1024"}
METHODS = {
    "P1": ["ntk2", "sega_base_none", "ntk2_sref", "temp_only_sbase", "sega_official"],
    "P3": ["tau1p0", "tau1p1", "tau_sref", "tau_logn", "tau1p45"],
    "E1": ["sega_official", "temp_only_sbase", "dype", "ntk2", "sega_base_none", "ntk2_sref"],
    "E2": ["sega_official", "temp_only_sbase"],
    "E3": ["sega_official", "temp_only_sbase", "base_default"],
}
IR_MARGIN = 0.08197713636363636
PICK_MARGIN = 0.10
SWEEP_TEMPERATURES = {"tau1p0": 1.0, "tau1p1": 1.1, "tau_sref": 4 ** .16,
                      "tau_logn": 4 / 3, "tau1p45": 1.45}
FACTORIAL = {
    "base_at_none": {"sega_base_none": 1, "ntk2": -1},
    "base_at_sref": {"temp_only_sbase": 1, "ntk2_sref": -1},
    "temperature_at_ntk2": {"ntk2_sref": 1, "ntk2": -1},
    "temperature_at_sega": {"temp_only_sbase": 1, "sega_base_none": -1},
    "base_main_effect": {"sega_base_none": .5, "temp_only_sbase": .5,
                         "ntk2": -.5, "ntk2_sref": -.5},
    "temperature_main_effect": {"ntk2_sref": .5, "temp_only_sbase": .5,
                                "ntk2": -.5, "sega_base_none": -.5},
    "interaction": {"temp_only_sbase": 1, "sega_base_none": -1, "ntk2_sref": -1, "ntk2": 1},
    "allocation": {"sega_official": 1, "temp_only_sbase": -1},
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
    if block in ["P1", "P3", "E2"]:
        if len(prompts) != 195:
            raise ValueError(f"{block} requires the original 195-row manifest")
        chosen = prompts[8:63] if block == "P3" else prompts[8:195]
        seeds = [0, 1, 2] if block == "E2" else [0]
    else:
        if len(prompts) != 150:
            raise ValueError(f"{block} requires exactly 150 fresh prompts")
        chosen, seeds = prompts, [0]
    metrics = spec.get("metrics", METRICS)
    if len(metrics) != len(set(metrics)) or not set(PRIMARY).issubset(metrics):
        raise ValueError("Metrics must be unique and include all three primary outcomes")
    if block != "E3" and not set(METRICS).issubset(metrics):
        raise ValueError("FLUX blocks must report all registered per-image metrics")
    if block != "E3" and spec["resolution"] != "4096x4096":
        raise ValueError("FLUX resolution must be 4096x4096")
    return {"block": block, "methods": METHODS[block], "seeds": seeds,
            "prompt_ids": [row["id"] for row in chosen], "metrics": metrics,
            "resolution": spec["resolution"]}


def bootstrap_draws(values):
    """One fresh RNG per contrast, float64, manifest order, exactly 2,000 draws."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap needs nonempty, finite one-dimensional values")
    return np.random.default_rng(0).choice(values, (2000, len(values)), replace=True).mean(axis=1)


def boot(values):
    draws = bootstrap_draws(values)
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), *map(float, np.percentile(draws, [2.5, 97.5], method="linear"))


def bootstrap_p(values, null=0., alternative="two-sided"):
    """Approximate centered-bootstrap test, including equality and plus-one correction."""
    values = np.asarray(values, dtype=np.float64)
    errors = bootstrap_draws(values) - values.mean()
    statistic = values.mean() - null
    greater = (1 + np.count_nonzero(errors >= statistic)) / (len(errors) + 1)
    less = (1 + np.count_nonzero(errors <= statistic)) / (len(errors) + 1)
    if alternative == "greater":
        return float(greater)
    if alternative == "less":
        return float(less)
    if alternative == "two-sided":
        return float((1 + np.count_nonzero(np.abs(errors) >= abs(statistic))) / (len(errors) + 1))
    raise ValueError(f"Unknown alternative {alternative!r}")


def tost_p(values, margin):
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("Equivalence margin must be finite and positive")
    return max(bootstrap_p(values, -margin, "greater"), bootstrap_p(values, margin, "less"))


def holm(pvalues, family_size=None):
    """Holm adjusted p-values; absent predeclared hypotheses consume p=1 slots."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    if pvalues.ndim != 1 or not np.isfinite(pvalues).all() or ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError("p-values must be finite and in [0,1]")
    size = len(pvalues) if family_size is None else family_size
    if size < len(pvalues) or int(size) != size:
        raise ValueError("Holm family cannot omit observed hypotheses")
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    adjusted[order] = np.minimum(1., np.maximum.accumulate((size - np.arange(len(pvalues))) * pvalues[order]))
    return adjusted


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
    expected = {"prompt_id": row["prompt_id"],
                "prompt": prompt, "seed": row["seed"], "res": row["res"]}
    method = row["method"]
    control = (method in ["temp_only_sbase", "sega_base_none", "ntk2_sref"]
               or method in SWEEP_TEMPERATURES)
    generation_method = "temp_only" if control else method
    if config.get("method") not in {row["raw_method"], generation_method}:
        raise ValueError(f"{path}: config method disagrees with evaluator identity")
    if "controller_method" in config and config["controller_method"] != generation_method:
        raise ValueError(f"{path}: incorrect underlying controller method")
    if spec["block"] == "E3":
        arm = config.get("method")
        if arm not in ["qwen_base", "qwen_sega", "qwen_temp_only"]:
            raise ValueError(f"{path}: unknown Qwen arm")
        expected.update({
            "schema": "qwen-strengthen-v1", "model": "Qwen/Qwen-Image",
            "model_revision": "75e0b4be04f60ec59a75f475837eced720f823b6",
            "steps": 40, "seed": 0, "resolved_seed": 0, "generator_device": "cuda",
            "generator_policy": "reset CUDA generator to seed0 for each prompt and arm",
            "effective_prompt": prompt + ", Ultra HD, 4K, cinematic composition.",
            "negative_prompt": "", "true_cfg_scale": 4., "guidance_scale": None,
            "max_sequence_length": 512, "dtype": "bfloat16", "training_resolution": 1024,
            "target_resolution": None, "rope_theta": 10000, "scale_rope": True,
            "spatial_rope_dim": 56, "frame_qk_amplitude": 1.,
            "spatial_ntk_factor": 1. if arm == "qwen_base" else 4 ** (112 / 54) / (1 + .1 * np.log(4)),
            "spatial_base_qk_amplitude": 1. if arm == "qwen_base" else 4 ** .08,
            "spatial_uniform_logit_gain": 1. if arm == "qwen_base" else 4 ** .16,
            "mscale_alpha": .15 if arm == "qwen_sega" else 0., "mscale_beta": 1.5, "mscale_min": 1.,
            "released_width_allocation_reused_on_height": arm == "qwen_sega",
            "mu": .9, "shift_terminal": .02, "use_dynamic_shifting": True,
            "time_shift_type": "exponential", "vae_tiling": True, "cpu_offload": False,
            "smoke": False, "provenance.official_commit": "ddaab08552c7b7ceba2ebd8f579a0f65dee90b5e"})
        finite = metadata.get("finite_steps", [])
        if len(finite) != 40 or not all(value is True for value in finite):
            raise ValueError(f"{path}: missing or nonfinite Qwen denoising steps")
        initial = np.linspace(1., 1 / 40, 40)
        shifted = np.exp(.9) / (np.exp(.9) + (1 / initial - 1))
        scheduled = np.r_[1 - (1 - shifted) / ((1 - shifted[-1]) / .98), 0.]
        actual_sigmas = np.asarray(metadata.get("sigmas", []), dtype=float)
        if actual_sigmas.shape != (41,) or not np.allclose(actual_sigmas, scheduled, rtol=0, atol=2e-7):
            raise ValueError(f"{path}: Qwen schedule differs from mu0.9 terminal0.02")
        timestep = np.asarray(metadata.get("timesteps", []), dtype=float)
        if timestep.shape != (40,) or not np.allclose(timestep, scheduled[:-1] * 1000, rtol=0, atol=2e-4):
            raise ValueError(f"{path}: Qwen timesteps differ from realized schedule")
    else:
        from crope.utils import image_seed
        expected["derived_seed"] = image_seed(row["prompt_id"], row["seed"])
        if config.get("method") != generation_method and config.get("controller_method") != generation_method:
            raise ValueError(f"{path}: unique output identity requires controller_method")
        digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if metadata.get("config_fingerprint") != digest:
            raise ValueError(f"{path}: config fingerprint differs")
        linear = np.linspace(1., 1 / 28, 28)
        scheduled = np.exp(1.15) / (np.exp(1.15) + (1 / linear - 1))
        actual_sigmas = np.asarray(metadata.get("sigmas", []), dtype=float)
        if (actual_sigmas.shape != (28,) or not np.isfinite(actual_sigmas).all()
                or not np.allclose(actual_sigmas, scheduled, rtol=0, atol=2e-7)):
            raise ValueError(f"{path}: actual denoising schedule differs from mu1.15")
        expected.update(steps=28, guidance=4.5, model="black-forest-labs/FLUX.1-dev",
                        model_revision="3de623fc3c33e44ffbe2bad470d0f45bccf2eb21", dtype="bfloat16", eta=1.0)
        expected["scheduler.max_shift"] = 1.15
        # Older records stored 4096 before runtime sample() registered the actual 65536.
        # Both execute mu=max_shift. The source manifest must validate recorded schedules.
        if dotted(config, "scheduler.max_image_seq_len") not in [4096, 65536]:
            raise ValueError(f"{path}: unexpected flow-shift sequence length")
        expected["rope.method"] = ("ntk2" if method == "ntk2_sref" else
                                   "sega_official" if control else method)
        if method in ["temp_only_sbase", "ntk2_sref", "tau_sref"]:
            expected["temperature"] = "sref"
        elif method in ["sega_base_none", "tau1p0"]:
            expected["temperature"] = "none"
        elif method in ["tau1p1", "tau_logn", "tau1p45"]:
            expected["temperature"] = "none"
            expected["temperature_multiplier"] = SWEEP_TEMPERATURES[method]
        if method not in ["tau1p1", "tau_logn", "tau1p45"] and config.get("temperature_multiplier") is not None:
            raise ValueError(f"{path}: unexpected temperature multiplier override")
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
            reference_method = "tau_sref" if design["block"] == "P3" else "sega_official"
            reference = values[reference_method]
            for method in design["methods"]:
                if method == reference_method:
                    continue
                diffs = [values[method][pid] - reference[pid] for pid in design["prompt_ids"]
                         if pid in values[method] and pid in reference]
                if not diffs:
                    continue
                delta, lo, hi = boot(diffs)
                direction = "unresolved"
                if lo > 0 or hi < 0:
                    direction = "better" if (delta < 0 if metric in LOWER else delta > 0) else "worse"
                paired.append({"analysis": label, "method": method, "reference": reference_method,
                               "metric": metric, "n": len(diffs), "delta": delta, "lo": lo, "hi": hi,
                               "direction": direction, "complete_block": complete})
    return (pd.DataFrame(means, columns=SUMMARY_COLUMNS), pd.DataFrame(paired, columns=PAIRED_COLUMNS),
            pd.DataFrame(prompt_means, columns=["analysis", "method", "prompt_id", "metric", "seeds", "mean"]), omitted)


def linear_contrast(prompt_means, design, label, metric, coefficients):
    """Form the contrast within each prompt before any bootstrap resampling."""
    subset = prompt_means.loc[(prompt_means.analysis == label) & (prompt_means.metric == metric)]
    if subset.duplicated(["method", "prompt_id"]).any():
        raise ValueError("Duplicate prompt mean identity")
    lookup = {(r.method, r.prompt_id): r.mean for r in subset.itertuples()}
    ids = [pid for pid in design["prompt_ids"] if all((method, pid) in lookup for method in coefficients)]
    values = [sum(weight * lookup[(method, pid)] for method, weight in coefficients.items()) for pid in ids]
    return ids, np.asarray(values, dtype=np.float64)


def exploratory_contrasts(prompt_means, design, complete):
    """Fixed exploratory families, with missing contrasts retained as p=1 slots."""
    block = design["block"]
    if block == "P1":
        coefficients = FACTORIAL
    else:
        reference = "tau_sref" if block == "P3" else "sega_official"
        coefficients = {method: {method: 1, reference: -1} for method in design["methods"] if method != reference}
    family_size = len(coefficients) * len(design["metrics"])
    records = []
    for label in analyses(design):
        family = []
        for name, weights in coefficients.items():
            for metric in design["metrics"]:
                ids, values = linear_contrast(prompt_means, design, label, metric, weights)
                row = {"analysis": label, "contrast": name, "metric": metric, "n": len(ids),
                       "family_size": family_size, "family": f"{block}:{label}", "role": "exploratory",
                       "coefficients": json.dumps(weights, sort_keys=True), "complete_block": complete}
                if len(values):
                    delta, lo, hi = boot(values)
                    row.update(delta=delta, lo=lo, hi=hi, p=bootstrap_p(values))
                else:
                    row.update(delta=None, lo=None, hi=None, p=1.)
                family.append(row)
        adjusted = holm([row["p"] for row in family], family_size)
        for row, p in zip(family, adjusted):
            row["p_holm"] = float(p)
            row["holm_reject"] = bool(complete and row["n"] == len(design["prompt_ids"]) and p <= .05)
            row["nominal_direction"] = ("unresolved" if row["lo"] is None or row["lo"] <= 0 <= row["hi"] else
                                        "better" if ((row["delta"] < 0) if row["metric"] in LOWER else
                                                     (row["delta"] > 0)) else "worse")
        records.extend(family)
    return pd.DataFrame(records)


def three_hypotheses(ir, pick, musiq, complete, role):
    """H1 uses temp-minus-SEGA; H2 uses SEGA-minus-temp. H3 is historical only."""
    rows = []
    for name, values, margin in [("H1_IR", ir, IR_MARGIN), ("H1_Pick", pick, PICK_MARGIN),
                                  ("H2_MUSIQ", musiq, None)]:
        row = {"hypothesis": name, "n": len(values), "margin": margin, "role": role,
               "family_size": 3, "complete_block": complete}
        if len(values):
            delta, lo, hi = boot(values)
            row.update(delta=delta, lo=lo, hi=hi,
                       p=tost_p(values, margin) if margin is not None else bootstrap_p(values, 0, "greater"))
        else:
            row.update(delta=None, lo=None, hi=None, p=1.)
        rows.append(row)
    for row, p in zip(rows, holm([row["p"] for row in rows], 3)):
        row["p_holm"] = float(p)
        if not complete or not row["n"]:
            row["status"] = "incomplete"
        elif row["margin"] is not None:
            margin = row["margin"]
            row["status"] = ("supported" if p <= .05 and -margin < row["lo"] and row["hi"] < margin else
                             "incompatible_with_equivalence" if row["lo"] > margin or row["hi"] < -margin
                             else "not_established")
        else:
            row["status"] = ("supported" if p <= .05 and row["lo"] > 0 else
                             "reverse_finding_nominal" if row["hi"] < 0 else "unresolved")
    h1 = ("supported" if all(row["status"] == "supported" for row in rows[:2]) else
          "incomplete" if not complete else "incompatible_with_equivalence"
          if any(row["status"] == "incompatible_with_equivalence" for row in rows[:2]) else "not_established")
    return {"role": role, "H1": h1, "H2": rows[2]["status"], "H3": "not_tested_historical_only",
            "family": rows, "p_method": "approximate centered paired-bootstrap, plus-one tails, Holm3",
            "interval_method": "nominal paired percentile bootstrap95; additional support gate"}


def hypothesis_outcomes(prompt_means, design, complete):
    if design["block"] in ["P1", "P3"]:
        return {"role": "exploratory", "H1": "not_tested", "H2": "not_tested", "H3": "not_tested"}
    result = {}
    for label in analyses(design):
        values = [linear_contrast(prompt_means, design, label, metric,
                                  {"temp_only_sbase": 1, "sega_official": -1})[1] for metric in PRIMARY]
        role = ("fresh_only_descriptive" if design["block"] == "E1" else
                "replication_descriptive" if design["block"] == "E2" else "exploratory_transfer")
        result[label] = three_hypotheses(values[0], values[1], -values[2], complete, role)
    return result


def pooled_confirmation(fresh_means, design, complete, heldout, heldout_prompts):
    """Only H1 pools old187 and fresh150; H2 remains a fresh-only confirmation."""
    if design["block"] != "E1" or len(design["prompt_ids"]) != 150:
        raise ValueError("Pooled confirmation requires the fresh150 E1 design")
    if len(heldout_prompts) != 195:
        raise ValueError("Historical source requires original195 prompt manifest")
    ids = [row["id"] for row in heldout_prompts[8:195]]
    if set(ids) & set(design["prompt_ids"]):
        raise ValueError("Fresh and historical prompt IDs overlap")
    historical = heldout.loc[heldout.method.isin(["sega_official", "temp_only_sbase"]) &
                            (heldout.seed == 0) & heldout.prompt_id.isin(ids)].copy()
    if historical.duplicated(KEYS).any():
        raise ValueError("Duplicate historical prompt identity")
    expected_index = {row["id"]: index for index, row in enumerate(heldout_prompts)}
    if "idx" not in historical or any(r.idx != expected_index[r.prompt_id] for r in historical.itertuples()):
        raise ValueError("Historical idx disagrees with original manifest")
    if "prompt" in historical:
        text = {row["id"]: row["prompt"] for row in heldout_prompts}
        if any(r.prompt != text[r.prompt_id] for r in historical.itertuples()):
            raise ValueError("Historical caption mismatch")
    hist_complete = len(historical) == 374 and all(
        set(historical.loc[historical.method == method, "prompt_id"]) == set(ids)
        for method in ["sega_official", "temp_only_sbase"])
    pooled = []
    for metric in PRIMARY[:2]:
        old = historical.pivot(index="prompt_id", columns="method", values=metric).reindex(ids)
        if not set(["temp_only_sbase", "sega_official"]).issubset(old):
            old_values = np.asarray([], dtype=float)
            hist_complete = False
        else:
            old_values = (old.temp_only_sbase - old.sega_official).to_numpy(dtype=np.float64)
            hist_complete &= bool(np.isfinite(old_values).all())
            old_values = old_values[np.isfinite(old_values)]
        _, fresh_values = linear_contrast(fresh_means, design, "seed0", metric,
                                          {"temp_only_sbase": 1, "sega_official": -1})
        pooled.append(np.concatenate([old_values, fresh_values]))
    _, musiq = linear_contrast(fresh_means, design, "seed0", "musiq",
                               {"sega_official": 1, "temp_only_sbase": -1})
    eligible = bool(complete and hist_complete and all(len(v) == 337 for v in pooled) and len(musiq) == 150)
    report = three_hypotheses(*pooled, musiq, eligible, "amended_confirmation")
    report.update(historical_complete=bool(hist_complete), fresh_complete=complete,
                  H1_pool_order="historical187 manifest order then fresh150 manifest order",
                  H2_population="fresh150 only", H1_caveat="337 includes previously inspected held-out data")
    return report


def output_report(data, design, out, files, exclusions, source_issues, provenance, historical=None):
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
    exploratory_contrasts(prompt_means, design, status["complete"]).to_csv(out / "contrasts_holm.csv", index=False)
    write_json(out / "hypotheses.json", hypothesis_outcomes(prompt_means, design, status["complete"]))
    if historical is not None:
        report = pooled_confirmation(prompt_means, design, status["complete"], *historical)
        write_json(out / "confirmation_family.json", report)
        pd.DataFrame(report["family"]).to_csv(out / "confirmation_family.csv", index=False)
    elif design["block"] == "E1":
        write_json(out / "confirmation_family.json", {"status": "not_tested",
                   "reason": "Need heldout_confirmation.per_prompt and heldout_confirmation.prompts in source manifest"})
    write_json(out / "coverage.json", {**status, "source_exclusions": exclusions,
                                       "descriptive_omissions": omitted})
    write_json(out / "analysis_protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "design": design,
        "protocol_version": "senior_review_20261019", "confirmatory_family_size": 3,
        "complete": status["complete"], "bootstrap_draws": 2000, "bootstrap_seed": 0,
        "fresh_rng_each_contrast": True, "float_dtype": "float64", "percentile_method": "linear",
        "percentiles": [2.5, 97.5], "unit": "prompt", "source_files": files,
        "analysis_source_sha256": sha256(__file__),
        "canonical_sha256": sha256(out / "canonical_per_image.csv"),
        "config_provenance_sha256": sha256(out / "config_provenance.jsonl"),
        "p_values": "Approximate centered paired-bootstrap with plus-one tail correction; Holm within fixed families",
        "limitations": ["Intervals remain nominal95; Holm-adjusted p marks reported separately",
                        "Centered bootstrap tests are approximate, not exact finite-sample inference",
                        "All families outside pooled H1 and fresh H2 are exploratory",
                        "Incomplete blocks have descriptive complete-pair tables only",
                        "Recorded PNG hash retained; analysis checks header and config, not full PNG hash",
                        "FID/KID remain descriptive in archived evaluator artifacts"]})
    return status["complete"]


def run(spec_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(spec_path).read_text())
    files = []
    archive_file(__file__, out, files, "analysis_source")
    archive_file(spec_path, out, files, "source_manifest")
    prompts_path = archive_file(spec["prompts"], out, files, "prompts")
    archive_file(spec["preregistration"], out, files, "preregistration")
    prompts = read_prompts(prompts_path)
    design = registered_design(spec, prompts)
    data, provenance, exclusions, issues = load_sources(spec, prompts, design, out, files)
    historical = None
    if "heldout_confirmation" in spec:
        old = spec["heldout_confirmation"]
        csv = archive_file(old["per_prompt"], out, files, "confirmation_heldout_per_prompt")
        manifest = archive_file(old["prompts"], out, files, "confirmation_heldout_prompts")
        historical = pd.read_csv(csv, dtype={"prompt_id": str}), read_prompts(manifest)
        # The pool must contain different captions as well as different IDs.
        import unicodedata
        normalize = lambda text: " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        old_captions = {normalize(row["prompt"]) for row in historical[1]}
        if any(normalize(row["prompt"]) in old_captions for row in prompts):
            raise ValueError("Fresh caption repeats an original evaluation caption")
    return output_report(data, design, out, files, exclusions, issues, provenance, historical)


def replay(source, out):
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve():
        raise ValueError("Replay output must differ from its immutable input archive")
    protocol = json.loads((source / "analysis_protocol.json").read_text())
    if protocol.get("protocol_version") != "senior_review_20261019":
        raise ValueError("Replay source uses a superseded preregistration; do not silently reinterpret it")
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
    archive_file(__file__, out, files, "replay_analysis_source")
    data = pd.read_csv(source / "canonical_per_image.csv", dtype={"prompt_id": str})
    provenance = [json.loads(line) for line in (source / "config_provenance.jsonl").read_text().splitlines()]
    old = json.loads((source / "coverage.json").read_text())
    by_role = {record["role"]: source / record["snapshot"] for record in protocol["source_files"]}
    historical = None
    if "confirmation_heldout_per_prompt" in by_role:
        historical = (pd.read_csv(by_role["confirmation_heldout_per_prompt"], dtype={"prompt_id": str}),
                      read_prompts(by_role["confirmation_heldout_prompts"]))
    return output_report(data, protocol["design"], out, files, old["source_exclusions"], old["source_issues"],
                         provenance, historical)


def historical_holm(source, prompts_path, out):
    """Retrospective marks for the existing full60 and condensed30 tables."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    archive_file(__file__, out, files, "analysis_source")
    table = archive_file(source, out, files, "historical_per_prompt")
    prompts_file = archive_file(prompts_path, out, files, "historical_prompts")
    prompts = read_prompts(prompts_file)
    if len(prompts) != 195:
        raise ValueError("Need the original195 manifest")
    ids = [row["id"] for row in prompts[8:195]]
    frame = pd.read_csv(table, dtype={"prompt_id": str})
    frame = frame.loc[frame.prompt_id.isin(ids) & (frame.seed == 0)].copy()
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate historical identities")
    methods = ["ntk2", "temp_only_sbase", "dype", "constancy", "constancy_tm_sbase", "sega_ctm_v2"]
    table_methods = [m for m in methods if m != "constancy"]
    table_metrics = ["imagereward", "pickscore", "clip", "musiq", "musiq_crop1024", "clipiqa"]
    records = []
    for family, members, metrics in [("historical_full60", methods, METRICS),
                                      ("historical_condensed30", table_methods, table_metrics)]:
        rows = []
        for method in members:
            # The published tables bootstrap the archived candidate CSV row order.
            # Preserve it here so a multiplicity audit cannot move nominal endpoints.
            method_order = frame.loc[frame.method == method, "prompt_id"].tolist()
            for metric in metrics:
                pivot = frame.pivot(index="prompt_id", columns="method", values=metric).reindex(method_order)
                difference = (pivot[method] - pivot.sega_official).dropna().to_numpy(dtype=np.float64)
                expected_n = 41 if method == "constancy" else 187
                if len(difference) != expected_n or not np.isfinite(difference).all():
                    raise ValueError(f"Historical {method}/{metric} count differs from frozen {expected_n}")
                delta, lo, hi = boot(difference)
                rows.append(dict(family=family, family_size=len(members) * len(metrics), method=method,
                                 metric=metric, n=len(difference), delta=delta, lo=lo, hi=hi,
                                 p=bootstrap_p(difference), role="retrospective_multiplicity_audit"))
        for row, adjusted in zip(rows, holm([row["p"] for row in rows])):
            row["p_holm"] = float(adjusted)
            row["holm_reject"] = bool(adjusted <= .05)
        records.extend(rows)
    result = pd.DataFrame(records)
    result.to_csv(out / "historical_holm.csv", index=False)
    write_json(out / "historical_holm_protocol.json", {
        "source_files": files, "analysis_source_sha256": sha256(__file__),
        "bootstrap_draws": 2000, "bootstrap_seed": 0,
        "p_method": "Approximate centered-bootstrap two-sided plus-one p, Holm separately over60 and30",
        "unit": "prompt", "role": "retrospective_multiplicity_audit",
        "order": "Original archived candidate CSV row order, matching published nominal intervals",
        "partial_constancy": "Original41 nonrandom available rows retained in full60 only",
        "no_reclassification_as_preregistered_confirmation": True})
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--historical", action="store_true", help="Compute fixed historical60/30 Holm marks")
    parser.add_argument("--prompts", help="Original195 prompts, required with --historical")
    args = parser.parse_args()
    if args.historical:
        if not args.prompts or args.replay:
            parser.error("--historical requires --prompts and cannot combine with --replay")
        complete = historical_holm(args.source, args.prompts, args.out)
    else:
        complete = replay(args.source, args.out) if args.replay else run(args.source, args.out)
    print(json.dumps({"out": args.out, "complete": complete}))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
