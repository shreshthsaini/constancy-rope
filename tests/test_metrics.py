"""Numerical metric checks use tiny arrays and never download model weights."""

import argparse
import json

import numpy as np
from PIL import Image
import pytest

from crope.metrics.evaluate import Evaluation, cluster_features
from crope.metrics.image import crop_boxes, linear_luminance, octave_profile, profile_error, tiling_metrics
from crope.metrics.io import discover_samples, load_references, write_json
from crope.metrics.statistics import (bootstrap_mean, distribution_bootstrap, fid, kid,
                                      paired_scalar_summaries, scalar_summaries)


def test_fid_translation_and_covariance_reference():
    from scipy.linalg import sqrtm

    rng = np.random.default_rng(8)
    x, y = rng.normal(size=(8, 12)), rng.normal(size=(9, 12)) * 1.4
    mx, my = x.mean(0), y.mean(0)
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    expected = np.sum((mx - my) ** 2) + np.trace(cx + cy - 2 * sqrtm(cx @ cy).real)
    assert fid(x, y) == pytest.approx(expected, abs=2e-6)
    assert fid(x, x) == pytest.approx(0, abs=1e-10)
    assert fid(x, x + 2) == pytest.approx(4 * 12, abs=1e-9)
    # Exercise the covariance route when feature dimension is smaller than N.
    assert fid(x[:, :3], x[:, :3] + 2) == pytest.approx(12, abs=1e-9)


def test_kid_matches_explicit_unbiased_kernel():
    x = np.array([[1, 0], [0, 1], [-1, 2]], dtype=float)
    y = np.array([[0, 0], [2, -1]], dtype=float)
    kernel = lambda a, b: (a @ b / 2 + 1) ** 3
    expected = sum(kernel(a, b) for i, a in enumerate(x) for j, b in enumerate(x) if i != j) / 6
    expected += sum(kernel(a, b) for i, a in enumerate(y) for j, b in enumerate(y) if i != j) / 2
    expected -= 2 * np.mean([kernel(a, b) for a in x for b in y])
    assert kid(x, y) == pytest.approx(expected)
    with pytest.raises(ValueError, match="at least two"):
        kid(x[:1], y)


def test_scalar_bootstrap_clusters_prompts_and_seed_pairs():
    def row(method, prompt, seed, value, res="4096x4096"):
        return {"method": method, "res": res, "prompt_id": prompt,
                "seed": seed, "metrics": {"score": value}}
    rows = [row("a", "one", 0, 0), row("a", "one", 1, 0), row("a", "two", 0, 10)]
    assert scalar_summaries(rows, draws=50)[0]["value"] == 5
    rows += [row("sega", "one", 0, 2), row("sega", "two", 0, 3),
             row("sega", "one", 1, 1000, "1024x1024")]
    paired = paired_scalar_summaries(rows, draws=50)
    assert len(paired) == 1
    assert paired[0]["value"] == 2.5
    assert paired[0]["n_prompts"] == 2
    assert bootstrap_mean([1, 2, 9], draws=30, seed=7) == bootstrap_mean([1, 2, 9], draws=30, seed=7)


def test_distribution_bootstrap_keeps_shared_draws():
    generated = {"a": np.array([[1, 2], [3, 4]]), "b": np.array([[2, 4], [1, 5]])}
    real = {"r1": np.array([[4, 0]]), "r2": np.array([[1, 3]])}
    for statistic in (fid, kid):
        value = distribution_bootstrap(generated, real, statistic, draws=20, baseline=generated)
        assert value["value"] == 0
        assert value["ci_low"] == 0
        assert value["ci_high"] == 0
        assert value["n_prompts"] == 2
        assert value["n_feature_rows"] == 4


def test_feature_clusters_align_seeds():
    rows = [{"method": method, "res": "r", "prompt_id": "p", "seed": seed, "path": f"{method}{seed}"}
            for method, seed in [("a", 1), ("a", 2), ("sega", 1)]]
    bank = {row["path"]: np.array([[row["seed"], 9]]) for row in rows}
    result = cluster_features(rows, bank, method="a", res="r", align_with="sega")
    np.testing.assert_array_equal(result["p"], [[1, 9]])


def test_deterministic_crop_boxes_are_bounded():
    boxes = crop_boxes((1300, 1800), "p", seed=31, count=5)
    assert boxes == crop_boxes((1300, 1800), "p", seed=31, count=5)
    assert boxes != crop_boxes((1300, 1800), "other", seed=31, count=5)
    assert all(0 <= x <= 276 and 0 <= y <= 776 and right - x == top - y == 1024
               for x, y, right, top in boxes)
    with pytest.raises(ValueError, match="both dimensions"):
        crop_boxes((512, 1024), "p")


def test_luminance_uses_linear_srgb():
    image = Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8))
    expected = ((128 / 255 + 0.055) / 1.055) ** 2.4
    np.testing.assert_allclose(linear_luminance(image), expected, rtol=1e-6)


def test_spectrum_detects_blur_and_identity():
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(4)
    y = rng.normal(size=(64, 64))
    target = octave_profile(y)
    identity = profile_error(target, target["log_power"])
    assert identity["spectrum_log_rmse"] == 0
    assert identity["spectrum_slope_error"] == 0
    blurred = octave_profile(gaussian_filter(y, 2, mode="wrap"))
    assert blurred["log_power"][-1] < target["log_power"][-1] - 4
    assert profile_error(blurred, target["log_power"])["spectrum_slope_error"] > 1


def test_tiling_detects_periodic_image_without_wrapping():
    y = np.tile(np.cos(np.arange(64) * 2 * np.pi / 8)[None, :], (64, 1))
    result = tiling_metrics(y, lags=(8,))
    assert result["tiling8"] == pytest.approx(1.0, abs=0.05)
    assert result["tiling8_valid_axes"] == 2
    invalid = tiling_metrics(y, lags=(64,))
    assert invalid["tiling64"] is None
    assert invalid["tiling64_valid_axes"] == 0


def make_sample(root, method, identity, seed=0):
    directory = root / method
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"irrelevant_filename_{identity}_{seed}.png"
    rng = np.random.default_rng(int(identity))
    Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)).save(path)
    metadata = {"config": {"method": method, "res": "64x64", "prompt_id": str(identity),
                            "prompt": f"prompt {identity}", "seed": seed}, "seconds": 2.0}
    write_json(path.with_suffix(".json"), metadata)
    return path


def test_metadata_discovery_and_duplicates(tmp_path):
    path = make_sample(tmp_path, "sega", 1)
    prompts = [{"id": "1", "prompt": "prompt 1"}]
    rows = discover_samples(tmp_path, prompts)
    assert len(rows) == 1 and rows[0]["prompt_id"] == "1"
    assert rows[0]["seconds"] == 2.0
    duplicate = path.with_name("duplicate.png")
    duplicate.write_bytes(path.read_bytes())
    duplicate.with_suffix(".json").write_bytes(path.with_suffix(".json").read_bytes())
    with pytest.raises(ValueError, match="Duplicate sample identity"):
        discover_samples(tmp_path, prompts)


def test_evaluation_resumes_and_writes_real_reports(tmp_path):
    samples = tmp_path / "samples"
    refs = []
    for identity in (1, 2):
        for method in ("sega", "constancy"):
            path = make_sample(samples, method, identity)
            if method == "sega":
                refs.append({"id": str(identity), "path": str(path)})
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps({"id": str(i), "prompt": f"prompt {i}"}) for i in (1, 2)))
    references = tmp_path / "references.jsonl"
    references.write_text("\n".join(json.dumps(row) for row in refs))
    assert len(load_references(references)) == 2
    args = argparse.Namespace(root=str(samples), prompts=str(prompts), references=str(references),
        out=str(tmp_path / "evaluation"), metrics="spectrum,tiling", res=None,
        reference_method="sega", device="cpu", bootstrap=20, bootstrap_seed=0,
        crop_seed=0, crops=1, allow_missing=False, no_wandb=True, threads=2)
    assert Evaluation(args).run() == 0
    cache = (tmp_path / "evaluation/evaluation_cache.json").read_text()
    assert Evaluation(args).run() == 0
    assert (tmp_path / "evaluation/evaluation_cache.json").read_text() == cache
    report = json.loads((tmp_path / "evaluation/metrics.json").read_text())
    assert report["complete"] is True
    assert len(report["per_image"]) == 4
    assert report["paired_deltas"]
    assert all(row["value"] == 0 for row in report["paired_deltas"])
    assert (tmp_path / "evaluation/summary.csv").is_file()


def test_json_nonfinite_is_null(tmp_path):
    path = tmp_path / "value.json"
    write_json(path, {"undefined": np.nan, "array": np.array([1.0, np.nan])})
    assert json.loads(path.read_text()) == {"undefined": None, "array": [1.0, None]}


def test_missing_requested_metric_fails_unless_explicit(tmp_path):
    root = tmp_path / "samples"
    make_sample(root, "sega", 1)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "1", "prompt": "prompt 1"}) + "\n")
    args = argparse.Namespace(root=str(root), prompts=str(prompts), references=None,
        out=str(tmp_path / "evaluation"), metrics="spectrum", res=None,
        reference_method="sega", device="cpu", bootstrap=0, bootstrap_seed=0,
        crop_seed=0, crops=1, allow_missing=False, no_wandb=True, threads=2)
    assert Evaluation(args).run() == 2
    args.allow_missing = True
    assert Evaluation(args).run() == 0
    report = json.loads((tmp_path / "evaluation/metrics.json").read_text())
    assert report["complete"] is False
    assert "--references" in report["errors"][0]["error"]


def test_hps_is_optional_only_in_default_all(tmp_path):
    root = tmp_path / "samples"
    make_sample(root, "sega", 1)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "1", "prompt": "prompt 1"}) + "\n")
    args = argparse.Namespace(root=str(root), prompts=str(prompts), references=None,
        out=str(tmp_path / "evaluation"), metrics="all", res=None,
        reference_method="sega", device="cpu", bootstrap=0, bootstrap_seed=0,
        crop_seed=0, crops=1, allow_missing=False, no_wandb=True, threads=2)
    evaluator = Evaluation(args)
    evaluator.failed("hpsv3", RuntimeError("Unavailable optional model"))
    assert not evaluator.errors and evaluator.optional_errors
    assert evaluator.status["hpsv3"]["status"] == "optional_unavailable"
    args.metrics = "hpsv3"
    strict = Evaluation(args)
    strict.failed("hpsv3", RuntimeError("Unavailable requested model"))
    assert strict.errors and not strict.optional_errors
