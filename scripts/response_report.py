#!/usr/bin/env python3
"""Archive a complete response campaign after CPU validation. Never promotes a method."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.response_analysis import summarize
from crope.response import band_directions, perturb_gain
from crope.rope import FluxRoPE
from crope.utils import image_seed

KREA = "8162a9c7b05a641be098422bf2fcf335615c2f28"
INTERVENTIONS = {"plus": 1., "minus": -1., "half_plus": .5, "half_minus": -.5}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file(), f"Incomplete evidence: missing {path}")
    require(path.stat().st_size < 8 * 1024 * 1024, f"Unexpectedly large JSON: {path}")
    return json.loads(path.read_text())


def selected_directions(resolution):
    exp = SimpleNamespace(axes_dims=(16, 56, 56),
                          rope=FluxRoPE(method="sega_official", grid=(resolution // 16,) * 2))
    all_directions = band_directions(exp)
    chosen = []
    for axis in (1, 2):
        axial = [d for d in all_directions if d.preserve_axes == (axis,)]
        chosen.append(axial[len(axial) // 2])
    return chosen + [d for d in all_directions if d.name == "spatial_scalar"]


def finite_gain(values):
    value = torch.tensor(values, dtype=torch.float32)
    require(value.shape == (128,) and bool(torch.isfinite(value).all()) and bool((value > 0).all()),
            "Expected 128 finite positive gain coordinates")
    require(torch.equal(value[::2], value[1::2]), "Rotary-pair gain mismatch")
    return value


def gain_diagnostics(record, directions):
    """Check realized casts and report when BF16 erases a nominal intervention."""
    base = finite_gain(record["baseline_gain_float32"])
    cast = finite_gain(record["baseline_gain_cast_bfloat16"])
    require(torch.equal(base.bfloat16().float(), cast), "Baseline BF16 cast mismatch")
    require(torch.equal(base[:16], torch.ones(16)), "Index-axis gain changed")
    require([d["name"] for d in record["directions"]] == [d.name for d in directions],
            "Unexpected control directions")
    result = []
    for saved, direction in zip(record["directions"], directions):
        require(saved["vector"] == direction.vector.tolist() and
                saved["preserve_axes"] == list(direction.preserve_axes), "Control vector changed")
        item = {"direction": direction.name, "interventions": {}}
        casts = {}
        for label, multiplier in INTERVENTIONS.items():
            realized = saved["realized_gains"][label]
            fp32 = finite_gain(realized["float32"])
            bf16 = finite_gain(realized["cast_bfloat16"])
            expected = perturb_gain(base, direction, record["epsilon"] * multiplier, (16, 56, 56))
            # CUDA exp/reduction and CPU replay may differ in their final FP32 bits.
            require(torch.allclose(fp32, expected, rtol=5e-6, atol=5e-7), "Realized FP32 gain violates intervention")
            require(torch.equal(fp32.bfloat16().float(), bf16), "Realized BF16 cast mismatch")
            require(torch.equal(fp32[:16], base[:16]), "Intervention changed index axis")
            delta32, delta16 = (fp32 - base).double(), (bf16 - cast).double()
            dead = torch.equal(bf16, cast)
            item["interventions"][label] = {
                "nominal_amount": record["epsilon"] * multiplier,
                "fp32_qk_delta_l2": float(delta32.norm()),
                "bf16_qk_delta_l2": float(delta16.norm()),
                "bf16_qk_delta_linf": float(delta16.abs().max()),
                "bf16_logit_gain_delta_l2": float((bf16.double().square() - cast.double().square()).norm()),
                "bf16_changed_coordinates": int((bf16 != cast).sum()),
                "quantization_dead": dead,
                "dead_gain_nonzero_response": dead and saved[label] != record["baseline"],
            }
            casts[label] = bf16
        item["full_plus_minus_same_bf16_gain"] = torch.equal(casts["plus"], casts["minus"])
        item["half_plus_minus_same_bf16_gain"] = torch.equal(casts["half_plus"], casts["half_minus"])
        item["all_interventions_quantization_dead"] = all(v["quantization_dead"] for v in item["interventions"].values())
        result.append(item)
    return result


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def verify_endpoints(path, receipt, tokens):
    require(Path(path).is_file(), "Missing endpoint tensor evidence")
    tensors = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(set(tensors) == {"baseline", "probed"}, "Unexpected endpoint payload")
    a, b = tensors["baseline"], tensors["probed"]
    require(a.shape == b.shape == (tokens, 64), "Endpoint shape mismatch")
    require(a.dtype == b.dtype and a.is_floating_point() and bool(torch.isfinite(a).all()) and
            bool(torch.isfinite(b).all()), "Invalid endpoint dtype or values")
    require(torch.equal(a, b), "Endpoint tensors are not bitwise equal")
    require(tensor_hash(a) == receipt["baseline_sha256"] and tensor_hash(b) == receipt["probed_sha256"],
            "Endpoint tensor hashes differ from receipt")
    return {"path": str(Path(path).resolve()), "file_sha256": sha256(path),
            "file_bytes": Path(path).stat().st_size, "shape": list(a.shape), "dtype": str(a.dtype),
            "tensor_sha256": tensor_hash(a), "independently_verified_on_cpu": True,
            "tensor_copied_to_archive": False}


def expected_schedule(configuration, steps, shift):
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(configuration)
    require(configuration.get("use_dynamic_shifting") is True and configuration.get("max_shift") == shift,
            "Unexpected scheduler shift configuration")
    scheduler.set_timesteps(sigmas=np.linspace(1., 1. / steps, steps), device="cpu", mu=shift)
    return scheduler.sigmas.float().tolist(), (scheduler.timesteps.bfloat16() / 1000).float().tolist()


def validate_stage(path, index, stage, protocol, protocol_hash, submission, prompts, prompt_hash, sources):
    from crope.runtime import implementation_hash
    manifest = read_json(path / "manifest.json")
    done = read_json(path / "completion.json")
    require(done == read_json(path / "validation.json"), "Completion and validation receipts disagree")
    require(all(done.get(key) is True for key in
                ("gpu_probe_executed", "trajectory_bitwise_equal", "baseline_history_equal")),
            "Baseline invariance validation did not pass")
    require(manifest["prompt_index"] == index and manifest["prompt"] == prompts[index], "Prompt identity changed")
    settings = protocol["smoke" if stage == "smoke" else "diagnostic"]
    for key, expected in (("res", [settings["resolution"]] * 2), ("steps", settings["steps"]),
                          ("probe_steps", settings["probe_steps"]), ("epsilon", protocol["epsilon"]),
                          ("seed", protocol["seed"]), ("guidance", protocol["guidance"]),
                          ("response_protocol_sha256", protocol_hash), ("prompt_file_sha256", prompt_hash),
                          ("transformer_revision", KREA), ("slurm_job_id", str(submission["job_id"]))):
        require(manifest.get(key) == expected, f"Manifest {key} mismatch")
    require(manifest.get("frozen_cr14_source_sha256") == sources, "Frozen runtime source provenance mismatch")
    require(manifest.get("source_sha256") == {k: protocol["source_sha256"][k]
            for k in ("crope/response.py", "scripts/response_probe.py")}, "Response source provenance mismatch")
    config = manifest["experiment_configuration"]
    expected = dict(method="sega_official", res=f'{settings["resolution"]}x{settings["resolution"]}',
                    grid=[settings["resolution"] // 16] * 2, seed=protocol["seed"], steps=settings["steps"],
                    prompt_id=prompts[index]["id"], prompt=prompts[index]["prompt"], guidance=protocol["guidance"],
                    dtype="bfloat16", batch_size=1, model="black-forest-labs/FLUX.1-dev",
                    derived_seed=image_seed(prompts[index]["id"], protocol["seed"]),
                    model_revision=manifest["base_revision"])
    require(all(config.get(k) == v for k, v in expected.items()), "Experiment model/seed/settings mismatch")
    require(config["rope"]["method"] == "sega_official" and config["rope"]["axes_dims"] == [16, 56, 56],
            "Wrong RoPE configuration")
    require(config["torch"] == manifest["torch"] and bool(config.get("diffusers")), "Software provenance missing")
    require(config.get("implementation_sha256") == implementation_hash(), "Implementation fingerprint mismatch")
    snapshot = Path(manifest["base_snapshot"])
    require(snapshot.name == manifest["base_revision"], "Base snapshot revision mismatch")
    require(sha256(snapshot / "model_index.json") == manifest["base_model_index_sha256"], "Base model metadata hash changed")
    transformer_config = snapshot.parent.parent.parent / "models--black-forest-labs--FLUX.1-Krea-dev" / "snapshots" / KREA / "transformer/config.json"
    require(sha256(transformer_config) == manifest["transformer_config_sha256"], "Transformer configuration hash changed")
    require(done.get("efficacy_evaluated") is False, "Unexpected efficacy claim in diagnostic receipt")
    sigmas, times = expected_schedule(config["scheduler"], settings["steps"], protocol["scheduler_shift"])
    require(done.get("sigmas") == sigmas, "Actual scheduler sigmas mismatch")
    directions = selected_directions(settings["resolution"])
    require(done["directions"] == [d.name for d in directions], "Direction receipt mismatch")
    rows = []
    raw = {"manifest.json": manifest, "completion.json": done, "validation.json": done}
    extra_seconds = 0.
    for step in settings["probe_steps"]:
        name = f"step_{step:03d}.json"
        record = read_json(path / name)
        require(record["sigma"] == sigmas[step] and record["model_timestep"] == times[step], "Probe sigma/timestep mismatch")
        require(record["extra_calls"] == 13, "Expected one no-op and twelve intervention calls per step")
        require([b["axis"] for b in record["bins"]] == [1, 2], "Spectral axis order mismatch")
        require(sum(len(b["lower"]) for b in record["bins"]) == len(record["baseline"]) and
                all(len(b["lower"]) == len(b["upper"]) for b in record["bins"]), "Spectral bin shape mismatch")
        require(math.isfinite(record["extra_seconds"]) and record["extra_seconds"] >= 0, "Invalid step timing")
        extra_seconds += record["extra_seconds"]
        rows.extend({"prompt_index": index, "stage": stage, "step": step, **r} for r in gain_diagnostics(record, directions))
        for direction in record["directions"]:
            derivative = (np.asarray(direction["plus"]) - direction["minus"]) / (2 * record["epsilon"])
            for key, expected_value in (("jacobian", derivative),
                                       ("half_plus_residual", np.asarray(direction["half_plus"]) - record["baseline"] - record["epsilon"] / 2 * derivative),
                                       ("half_minus_residual", np.asarray(direction["half_minus"]) - record["baseline"] + record["epsilon"] / 2 * derivative)):
                require(np.allclose(direction[key], expected_value, rtol=1e-10, atol=1e-12), "Saved response algebra mismatch")
        raw[name] = record
    require(done["extra_forward_calls"] == 13 * len(settings["probe_steps"]), "Call budget mismatch")
    for key in ("baseline_seconds", "probed_seconds"):
        require(math.isfinite(done[key]) and done[key] > 0, "Invalid full-run timing")
    require(extra_seconds <= done["probed_seconds"], "Nested replay timers exceed the full probed run")
    summary = summarize(path)
    endpoint = verify_endpoints(path / "trajectory_endpoints.pt", done, (settings["resolution"] // 16) ** 2)
    if stage == "4k":
        require(read_json(path.parent / "summary.json") == summary, "Existing summary disagrees with fresh analysis")
    return {"summary": summary, "gain_diagnostics": rows, "endpoint_verification": endpoint,
            "timing": {"baseline_seconds": done["baseline_seconds"], "probed_seconds": done["probed_seconds"],
                       "baseline_plus_probed_seconds": done["baseline_seconds"] + done["probed_seconds"],
                       "nested_replay_seconds": extra_seconds,
                       "probed_seconds_outside_nested_replay_timers": done["probed_seconds"] - extra_seconds,
                       "ordinary_forward_calls": 2 * settings["steps"],
                       "extra_forward_calls": done["extra_forward_calls"],
                       "all_forward_calls": 2 * settings["steps"] + done["extra_forward_calls"],
                       "baseline_peak_bytes": done["baseline_peak_bytes"], "probed_peak_bytes": done["probed_peak_bytes"]},
            "raw_json": raw}


def aggregate_local(stages):
    groups = {}
    for key, stage in stages.items():
        for row, gain in zip(stage["summary"]["rows"], stage["gain_diagnostics"]):
            group = (key.split("/")[1], row["step"], row["direction"])
            require(row["direction"] == gain["direction"] and row["step"] == gain["step"], "Response/gain row mismatch")
            groups.setdefault(group, []).append((row, gain))
    result = []
    for (stage, step, direction), rows in sorted(groups.items()):
        metrics = {}
        for name in ("half_effect_l2", "linear_residual_l2", "relative_prediction_error",
                     "fraction_negative_jacobian", "off_axis_response_fraction"):
            values = [r[name] for r, _ in rows if r[name] is not None]
            metrics[name] = {"n_defined": len(values), "mean": float(np.mean(values)) if values else None,
                             "minimum": min(values) if values else None, "maximum": max(values) if values else None}
        result.append({"stage": stage, "step": step, "direction": direction, "n_prompts": len(rows),
                       "metrics": metrics,
                       "dead_interventions": {label: sum(g["interventions"][label]["quantization_dead"] for _, g in rows)
                                              for label in INTERVENTIONS},
                       "all_dead_controls": sum(g["all_interventions_quantization_dead"] for _, g in rows)})
    return result


def build_report(campaign, protocol_path, submission_path, frozen_path, prompts_path):
    campaign = Path(campaign).resolve()
    protocol, submission, frozen = map(read_json, (protocol_path, submission_path, frozen_path))
    protocol_hash = sha256(protocol_path)
    require(protocol_hash == submission["protocol_sha256"], "Protocol differs from submitted hash")
    require(protocol["development_prompt_indices"] == [0, 1], "Expected exactly development prompts 0 and 1")
    require(campaign.name == f'job{submission["job_id"]}', "Campaign job ID mismatch")
    # Fail before analysis or tensor reads when either worker stopped at smoke.
    for index in (0, 1):
        for stage in ("smoke", "4k"):
            require((campaign / f"prompt{index}" / stage / "completion.json").is_file(), "Incomplete campaign: every smoke and 4K completion is required")
        require((campaign / f"prompt{index}" / "summary.json").is_file(), "Missing completed prompt summary")
    source_receipts = {}
    for name, expected in protocol["source_sha256"].items():
        require(sha256(ROOT / name) == expected, f"Frozen response source changed: {name}")
        source_receipts[name] = expected
    runtime_sources = {name: digest for name, digest in frozen["source_sha256"].items() if ROOT in Path(name).parents}
    for name, expected in runtime_sources.items():
        require(sha256(name) == expected, f"Frozen runtime source changed: {name}")
    prompt_hash = sha256(prompts_path)
    require(frozen["source_sha256"].get(str(Path(prompts_path))) == prompt_hash, "Prompt manifest changed")
    prompts = [json.loads(line) for line in Path(prompts_path).read_text().splitlines() if line.strip()]
    require(len(prompts) >= 2 and prompts[0]["id"] != prompts[1]["id"], "Invalid prompt manifest")
    tasks = {}
    for name, expected in submission["task_sha256"].items():
        old = Path(name)
        matches = [p for state in ("done", "pending", "running", "failed")
                   for p in (old.parent.parent / state).glob(old.name + (".*" if state == "running" else "")) if p.is_file()]
        require(len(matches) == 1 and sha256(matches[0]) == expected, "Submitted task source missing, duplicated or changed")
        tasks[str(matches[0])] = expected
    stages = {}
    for index in (0, 1):
        for stage in ("smoke", "4k"):
            key = f"prompt{index}/{stage}"
            stages[key] = validate_stage(campaign / key, index, stage, protocol, protocol_hash, submission,
                                         prompts, prompt_hash, runtime_sources)
    model_keys = ("transformer_revision", "transformer_config_sha256", "base_revision", "base_model_index_sha256", "torch", "cuda")
    manifests = [s["summary"]["manifest"] for s in stages.values()]
    require(all(all(m[k] == manifests[0][k] for k in model_keys) for m in manifests), "Cross-stage model/software drift")
    require(all(m["experiment_configuration"]["diffusers"] == manifests[0]["experiment_configuration"]["diffusers"] for m in manifests), "Diffusers version drift")
    timing_keys = ("baseline_seconds", "probed_seconds", "baseline_plus_probed_seconds", "nested_replay_seconds",
                   "probed_seconds_outside_nested_replay_timers", "ordinary_forward_calls", "extra_forward_calls", "all_forward_calls")
    timing = {k: sum(stage["timing"][k] for stage in stages.values()) for k in timing_keys}
    require(timing["all_forward_calls"] == 2 * sum(protocol["call_budget_per_prompt"][k] for k in ("smoke_total", "4k_total")), "Campaign total call budget mismatch")
    return {"purpose": "Development-only local response diagnosis; no efficacy, novelty or generalization claim",
            "automatic_promotion": False, "job_id": str(submission["job_id"]), "protocol_sha256": protocol_hash,
            "stages": stages, "local_response_aggregate": aggregate_local(stages), "timing_totals": timing,
            "unmeasured_timing": {"model_loading_seconds": None, "endpoint_serialization_seconds": None,
                                  "wandb_finalization_seconds": None, "job_wall_seconds": None},
            "timing_limitations": "Summed task device/run times are not wall-clock job duration and must not be added again to nested replay timers. Full generate timers exclude model loading, endpoint serialization and W&B finalization. Replay timers also exclude initial cloning, baseline statistics and sink writes. These are diagnostic accounting, not warmup-matched performance benchmarks.",
            "sources": {"response": source_receipts, "runtime": runtime_sources, "tasks": tasks},
            "inputs": {"protocol": protocol, "submission": submission, "frozen_manifest": frozen,
                       "development_prompts": prompts[:2], "prompt_file_sha256": prompt_hash},
            "input_file_hashes": {str(Path(p).resolve()): sha256(p) for p in (protocol_path, submission_path, frozen_path, prompts_path)}}


def archive_report(report, campaign, out):
    out = Path(out)
    require(not out.exists(), "Archive must be a fresh directory")
    # Serialize and validate before creating output. No tensor, image or weight is copied.
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    hashes = {}
    for key, stage in report["stages"].items():
        for name in stage["raw_json"]:
            source = Path(campaign) / key / name
            hashes[str(source.resolve())] = sha256(source)
    out.mkdir(parents=True)
    (out / "report.json").write_text(encoded)
    for key, stage in report["stages"].items():
        folder = out / key
        folder.mkdir(parents=True)
        for name, data in stage["raw_json"].items():
            (folder / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        (folder / "summary.json").write_text(json.dumps(stage["summary"], indent=2, allow_nan=False) + "\n")
    receipt = {"source_json_sha256": hashes, "report_sha256": sha256(out / "report.json"),
               "reporter_source_sha256": sha256(__file__), "bulk_artifacts_copied": False}
    (out / "archive_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / ".agent/notes/response_probe_protocol.json")
    parser.add_argument("--submission", type=Path, default=ROOT / ".agent/notes/response_submission.json")
    parser.add_argument("--frozen", type=Path, default=ROOT / ".agent/notes/continuation3/frozen_manifest.json")
    parser.add_argument("--prompts", type=Path, default=Path("/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl"))
    args = parser.parse_args()
    require(str(args.out.resolve()).startswith("/work/"), "Archive must be on work storage")
    torch.set_num_threads(1)
    report = build_report(args.campaign, args.protocol, args.submission, args.frozen, args.prompts)
    print(json.dumps(archive_report(report, args.campaign, args.out), indent=2))


if __name__ == "__main__":
    main()
