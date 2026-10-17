import json
from pathlib import Path

import numpy as np
import pytest
import torch

from crope.response import perturb_gain
from crope.runtime import implementation_hash
from crope.utils import image_seed
from scripts import response_report as rr


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def record_for(step, sigma, timestep, resolution, epsilon=.1):
    base = torch.ones(128)
    base[16:] = 1.1
    directions = []
    for control in rr.selected_directions(resolution):
        derivative = np.asarray([.2, -.3])
        record = {"name": control.name, "vector": control.vector.tolist(),
                  "preserve_axes": list(control.preserve_axes), "realized_gains": {}}
        for label, multiplier in rr.INTERVENTIONS.items():
            gain = perturb_gain(base, control, epsilon * multiplier, (16, 56, 56))
            record["realized_gains"][label] = {"float32": gain.tolist(), "cast_bfloat16": gain.bfloat16().float().tolist()}
            record[label] = (np.asarray([1., 2.]) + epsilon * multiplier * derivative).tolist()
        record["jacobian"] = derivative.tolist()
        record["half_plus_residual"] = record["half_minus_residual"] = [0., 0.]
        directions.append(record)
    return {"step": step, "sigma": sigma, "model_timestep": timestep, "epsilon": epsilon,
            "baseline": [1., 2.], "bins": [{"axis": i, "lower": [.1], "upper": [.2]} for i in (1, 2)],
            "baseline_gain_float32": base.tolist(), "baseline_gain_cast_bfloat16": base.bfloat16().float().tolist(),
            "replay_max_abs_velocity_error": 0., "extra_calls": 13, "extra_seconds": 2., "directions": directions}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path / "job123"
    protocol = {"development_prompt_indices": [0, 1], "seed": 0, "guidance": 4.5, "epsilon": .1,
                "scheduler_shift": 1.15,
                "smoke": {"resolution": 2048, "steps": 4, "probe_steps": [1]},
                "diagnostic": {"resolution": 4096, "steps": 28, "probe_steps": [6, 14, 22]},
                "call_budget_per_prompt": {"smoke_total": 21, "4k_total": 95},
                "source_sha256": {name: rr.sha256(rr.ROOT / name) for name in
                                  ("crope/response.py", "scripts/response_probe.py", "scripts/response_analysis.py")}}
    protocol_path = tmp_path / "protocol.json"
    write(protocol_path, protocol)
    task_hashes = {}
    for i in (0, 1):
        path = tmp_path / "taskq/done" / f"prompt{i}.sh"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"echo fixture {i}\n")
        task_hashes[str(tmp_path / "taskq/pending" / path.name)] = rr.sha256(path)
    submission = {"job_id": "123", "protocol_sha256": rr.sha256(protocol_path), "task_sha256": task_hashes}
    submission_path = tmp_path / "submission.json"
    write(submission_path, submission)
    prompts = [{"id": f"p{i}", "prompt": f"Fixture prompt {i}"} for i in range(3)]
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text("".join(json.dumps(row) + "\n" for row in prompts))
    runtime_source = rr.ROOT / "crope/runtime.py"
    sources = {str(runtime_source): rr.sha256(runtime_source)}
    frozen_path = tmp_path / "frozen.json"
    write(frozen_path, {"source_sha256": {**sources, str(prompts_path): rr.sha256(prompts_path)}})
    snapshot = tmp_path / "hub/models--black-forest-labs--FLUX.1-dev/snapshots/base_revision"
    write(snapshot / "model_index.json", {"fixture": True})
    transformer = snapshot.parent.parent.parent / "models--black-forest-labs--FLUX.1-Krea-dev/snapshots" / rr.KREA / "transformer/config.json"
    write(transformer, {"fixture": "transformer"})
    scheduler = dict(use_dynamic_shifting=True, max_shift=1.15)
    for index in (0, 1):
        for stage in ("smoke", "4k"):
            spec = protocol["smoke" if stage == "smoke" else "diagnostic"]
            size, steps = spec["resolution"], spec["steps"]
            sigmas, times = rr.expected_schedule(scheduler, steps, 1.15)
            config = dict(method="sega_official", res=f"{size}x{size}", grid=[size // 16] * 2,
                          seed=0, steps=steps, prompt_id=prompts[index]["id"], prompt=prompts[index]["prompt"],
                          guidance=4.5, dtype="bfloat16", batch_size=1, model="black-forest-labs/FLUX.1-dev",
                          derived_seed=image_seed(prompts[index]["id"], 0), model_revision="base_revision",
                          rope={"method": "sega_official", "axes_dims": [16, 56, 56]}, torch="fixture-torch",
                          diffusers="fixture-diffusers", implementation_sha256=implementation_hash(), scheduler=scheduler)
            manifest = dict(prompt_index=index, prompt=prompts[index], res=[size, size], steps=steps,
                            probe_steps=spec["probe_steps"], epsilon=.1, seed=0, guidance=4.5,
                            response_protocol_sha256=rr.sha256(protocol_path), prompt_file_sha256=rr.sha256(prompts_path),
                            transformer_revision=rr.KREA, transformer_config_sha256=rr.sha256(transformer), slurm_job_id="123",
                            frozen_cr14_source_sha256=sources,
                            source_sha256={k: protocol["source_sha256"][k] for k in ("crope/response.py", "scripts/response_probe.py")},
                            experiment_configuration=config, base_snapshot=str(snapshot), base_revision="base_revision",
                            base_model_index_sha256=rr.sha256(snapshot / "model_index.json"), torch="fixture-torch", cuda="fixture-cuda")
            done = dict(gpu_probe_executed=True, trajectory_bitwise_equal=True, baseline_history_equal=True,
                        baseline_sha256="fixture", probed_sha256="fixture", observed_probe_steps=spec["probe_steps"],
                        directions=[d.name for d in rr.selected_directions(size)], extra_forward_calls=13 * len(spec["probe_steps"]),
                        baseline_seconds=10., probed_seconds=30., baseline_peak_bytes=100, probed_peak_bytes=200,
                        sigmas=sigmas, efficacy_evaluated=False)
            folder = root / f"prompt{index}" / stage
            write(folder / "manifest.json", manifest)
            write(folder / "validation.json", done)
            write(folder / "completion.json", done)
            for step in spec["probe_steps"]:
                write(folder / f"step_{step:03d}.json", record_for(step, sigmas[step], times[step], size))
            (folder / "trajectory_endpoints.pt").write_bytes(b"mocked endpoint fixture")
            if stage == "4k":
                write(folder.parent / "summary.json", rr.summarize(folder))
    calls = []
    def endpoints(path, receipt, tokens):
        calls.append(path)
        return {"fixture_only": True, "tensor_copied_to_archive": False}
    monkeypatch.setattr(rr, "verify_endpoints", endpoints)
    return (root, protocol_path, submission_path, frozen_path, prompts_path), calls


def test_complete_report_has_four_stages_and_no_bulk_archive(campaign, tmp_path):
    args, endpoints = campaign
    report = rr.build_report(*args)
    assert len(endpoints) == 4
    assert report["timing_totals"]["all_forward_calls"] == 232
    assert report["timing_totals"]["baseline_plus_probed_seconds"] == 160.
    assert report["timing_totals"]["nested_replay_seconds"] == 16.
    assert len(report["local_response_aggregate"]) == 12
    assert all(row["n_prompts"] == 2 for row in report["local_response_aggregate"])
    assert report["automatic_promotion"] is False
    out = tmp_path / "archive"
    receipt = rr.archive_report(report, args[0], out)
    assert receipt["bulk_artifacts_copied"] is False
    assert all(p.suffix == ".json" for p in out.rglob("*") if p.is_file())
    with pytest.raises(ValueError, match="fresh"):
        rr.archive_report(report, args[0], out)


def test_incomplete_smoke_stops_before_analysis(campaign, monkeypatch):
    args, endpoints = campaign
    completion = args[0] / "prompt1/smoke/completion.json"
    completion.rename(completion.with_suffix(".interrupted"))
    monkeypatch.setattr(rr, "summarize", lambda path: pytest.fail("Must not summarize incomplete campaign"))
    with pytest.raises(ValueError, match="Incomplete campaign"):
        rr.build_report(*args)
    assert endpoints == []


@pytest.mark.parametrize("problem", ["protocol", "source", "prompt", "seed", "model", "receipt", "sigma", "step_sigma", "epsilon", "history", "endpoint_receipt", "summary", "gain_cast", "task"])
def test_mismatched_evidence_fails_closed(campaign, problem):
    args, _ = campaign
    root, protocol, submission, _, _ = args
    folder = root / "prompt0/4k"
    if problem in ("protocol", "source"):
        value = rr.read_json(submission if problem == "protocol" else protocol)
        if problem == "protocol":
            value["protocol_sha256"] = "wrong"
            write(submission, value)
        else:
            value["source_sha256"]["crope/response.py"] = "wrong"
            write(protocol, value)
            value = rr.read_json(submission)
            value["protocol_sha256"] = rr.sha256(protocol)
            write(submission, value)
    elif problem in ("prompt", "seed", "model", "epsilon"):
        value = rr.read_json(folder / "manifest.json")
        if problem == "prompt":
            value["prompt_index"] = 8
        elif problem == "seed":
            value["experiment_configuration"]["seed"] = 1
        elif problem == "model":
            value["transformer_revision"] = "wrong"
        else:
            value["epsilon"] = .2
        write(folder / "manifest.json", value)
    elif problem in ("sigma", "history", "endpoint_receipt", "receipt"):
        value = rr.read_json(folder / "completion.json")
        if problem == "sigma":
            value["sigmas"][6] -= .01
        elif problem == "history":
            value["baseline_history_equal"] = False
        else:
            value["probed_sha256"] = "changed"
        write(folder / "completion.json", value)
        if problem != "receipt":
            write(folder / "validation.json", value)
    elif problem in ("step_sigma", "gain_cast"):
        value = rr.read_json(folder / "step_006.json")
        if problem == "step_sigma":
            value["sigma"] -= .01
        else:
            value["directions"][0]["realized_gains"]["plus"]["cast_bfloat16"][16:18] = [1., 1.]
        write(folder / "step_006.json", value)
    elif problem == "summary":
        write(folder.parent / "summary.json", {"changed": True})
    else:
        task = Path(next(iter(rr.read_json(submission)["task_sha256"])))
        (task.parent.parent / "done" / task.name).write_text("changed")
    with pytest.raises(ValueError):
        rr.build_report(*args)


def test_bfloat16_dead_controls_preserve_nonzero_fp32_change():
    record = record_for(1, .8, .8, 2048, epsilon=1e-4)
    diagnostics = rr.gain_diagnostics(record, rr.selected_directions(2048))
    assert all(row["all_interventions_quantization_dead"] for row in diagnostics)
    for row in diagnostics:
        assert row["interventions"]["plus"]["fp32_qk_delta_l2"] > 0
        assert row["interventions"]["plus"]["bf16_qk_delta_l2"] == 0
        assert row["interventions"]["plus"]["dead_gain_nonzero_response"]


def test_cast_corruption_is_rejected():
    record = record_for(1, .8, .8, 2048)
    record["baseline_gain_cast_bfloat16"][16:18] = [1., 1.]
    with pytest.raises(ValueError, match="cast"):
        rr.gain_diagnostics(record, rr.selected_directions(2048))


@pytest.mark.parametrize("changed", [False, True])
def test_independent_endpoint_hash_and_equality(tmp_path, changed):
    baseline = torch.zeros(2, 64, dtype=torch.bfloat16)
    probed = baseline.clone()
    if changed:
        probed[0, 0] = 1
    path = tmp_path / "endpoints.pt"
    torch.save({"baseline": baseline, "probed": probed}, path)
    receipt = {"baseline_sha256": rr.tensor_hash(baseline), "probed_sha256": rr.tensor_hash(probed)}
    if changed:
        with pytest.raises(ValueError, match="bitwise"):
            rr.verify_endpoints(path, receipt, 2)
    else:
        result = rr.verify_endpoints(path, receipt, 2)
        assert result["independently_verified_on_cpu"] and not result["tensor_copied_to_archive"]
