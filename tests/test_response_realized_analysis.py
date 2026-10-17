import copy
import json

import numpy as np
import pytest

from scripts.response_realized_analysis import analyze_direction, analyze_report, sha256


def fixture_record(orthogonal_half=False):
    base = np.ones(2)
    gains = {"plus": [2., 1.], "minus": [.5, 1.],
             "half_plus": [1.5, 1.5 if orthogonal_half else 1.],
             "half_minus": [.75, .75 if orthogonal_half else 1.]}
    matrix = np.asarray([[2., 3.], [-1., 4.]])
    baseline = np.asarray([1., 2.])
    direction = {"name": "fixture", "realized_gains": {}}
    for label, gain in gains.items():
        direction["realized_gains"][label] = {"float32": gain, "cast_bfloat16": gain}
        direction[label] = (baseline + matrix @ (2 * np.log(gain / base))).tolist()
    record = {"baseline_gain_float32": base.tolist(), "baseline_gain_cast_bfloat16": base.tolist(),
              "baseline": baseline.tolist(), "epsilon": .1, "directions": [direction], "step": 6, "sigma": .9}
    return record, direction


def test_known_linear_map_in_realized_coordinates_beats_wrong_nominal_spacing():
    record, direction = fixture_record()
    result = analyze_direction(record, direction)
    assert result["realized_coordinate_relative_prediction_error"] == pytest.approx(0., abs=1e-14)
    assert result["nominal_epsilon_relative_prediction_error"] > .1
    assert result["fitted_response_slope"] == pytest.approx([4 * np.log(2), -2 * np.log(2)])
    assert result["controls"]["half_plus"]["projected_amplitude"] == pytest.approx(np.log(1.5) / np.log(2))
    assert result["controls"]["half_plus"]["collinearity_residual_fraction"] == 0


def test_orthogonal_half_gain_change_is_reported_not_fitted_away():
    record, direction = fixture_record(orthogonal_half=True)
    result = analyze_direction(record, direction)
    assert result["controls"]["half_plus"]["collinearity_residual_fraction"] == pytest.approx(1 / np.sqrt(2))
    assert result["realized_coordinate_relative_prediction_error"] > 0
    assert result["fitted_response_slope"] == pytest.approx([4 * np.log(2), -2 * np.log(2)])


def test_held_aside_response_changes_cannot_change_fit_or_predictions():
    record, direction = fixture_record()
    first = analyze_direction(record, direction)
    changed = copy.deepcopy(direction)
    changed["half_plus"] = [100., -50.]
    changed["half_minus"] = [-12., 30.]
    second = analyze_direction(record, changed)
    assert first["fitted_response_slope"] == second["fitted_response_slope"]
    assert first["full_projected_amplitudes"] == second["full_projected_amplitudes"]
    assert first["half_predictions"] == second["half_predictions"]
    assert second["realized_coordinate_relative_prediction_error"] > first["realized_coordinate_relative_prediction_error"]


def test_exact_zero_spectral_effect_has_no_relative_error():
    record, direction = fixture_record()
    for label in ("plus", "minus", "half_plus", "half_minus"):
        direction[label] = record["baseline"][:]
    result = analyze_direction(record, direction)
    assert result["fit_status"] == "identified"
    assert result["fitted_response_slope"] == [0., 0.]
    assert result["half_effect_exact_zero"]
    assert result["realized_coordinate_relative_prediction_error"] is None
    assert result["nominal_epsilon_relative_prediction_error"] is None


@pytest.mark.parametrize("tiny", [False, True])
def test_zero_or_tiny_secant_has_no_identifiable_fit(tiny):
    record, direction = fixture_record()
    for label, sign in (("plus", 1), ("minus", -1), ("half_plus", 1), ("half_minus", -1)):
        gain = [1. + (sign * 1e-14 if tiny else 0.), 1.]
        direction["realized_gains"][label] = {"float32": [1.001, 1.], "cast_bfloat16": gain}
    result = analyze_direction(record, direction)
    assert result["fit_status"] == "zero_or_tiny_full_secant"
    assert result["fitted_response_slope"] is None
    assert result["half_predictions"] is None
    assert result["controls"]["half_plus"]["projected_amplitude"] is None
    assert result["full_secant_exact_zero"] is not tiny
    assert result["full_secant_tiny_nonzero"] is tiny
    assert result["controls"]["plus"]["quantization_dead"] is not tiny


def test_nonfinite_response_and_nonpositive_gain_rejected():
    record, direction = fixture_record()
    direction["half_plus"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        analyze_direction(record, direction)
    record, direction = fixture_record()
    direction["realized_gains"]["plus"]["cast_bfloat16"][0] = 0
    with pytest.raises(ValueError, match="positive"):
        analyze_direction(record, direction)


def validated_fixture(tmp_path):
    stages = {}
    for index in (0, 1):
        for stage in ("smoke", "4k"):
            validation = {"gpu_probe_executed": True, "trajectory_bitwise_equal": True, "baseline_history_equal": True}
            record, _ = fixture_record()
            stages[f"prompt{index}/{stage}"] = {
                "summary": {"validation": validation, "manifest": {"prompt_index": index,
                            "prompt": {"id": f"dev{index}"}, "res": [4096, 4096], "probe_steps": [6]}},
                "endpoint_verification": {"independently_verified_on_cpu": True},
                "raw_json": {"completion.json": validation, "validation.json": validation, "step_006.json": record}}
    report = {"stages": stages, "automatic_promotion": False, "job_id": "fixture"}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    (tmp_path / "archive_receipt.json").write_text(json.dumps({"report_sha256": sha256(path)}))
    return path


def test_validated_archive_retains_prompt_step_order_and_secondary_label(tmp_path):
    result = analyze_report(validated_fixture(tmp_path))
    assert [(r["prompt_index"], r["step"]) for r in result["rows"]] == [(0, 6), (1, 6)]
    assert "Unprespecified secondary" in result["purpose"]
    assert not result["automatic_promotion"] and not result["acceptance_rule_changed"]


@pytest.mark.parametrize("problem", ["missing_stage", "failed_invariance", "no_endpoint_verification", "hash_mismatch"])
def test_incomplete_or_changed_archive_rejected(tmp_path, problem):
    path = validated_fixture(tmp_path)
    report = json.loads(path.read_text())
    if problem == "missing_stage":
        report["stages"].pop("prompt1/4k")
    elif problem == "failed_invariance":
        report["stages"]["prompt0/smoke"]["summary"]["validation"]["trajectory_bitwise_equal"] = False
    elif problem == "no_endpoint_verification":
        report["stages"]["prompt0/4k"]["endpoint_verification"]["independently_verified_on_cpu"] = False
    else:
        report["job_id"] = "changed"
    path.write_text(json.dumps(report))
    if problem != "hash_mismatch":
        (tmp_path / "archive_receipt.json").write_text(json.dumps({"report_sha256": sha256(path)}))
    with pytest.raises(ValueError):
        analyze_report(path)
