#!/usr/bin/env python3
"""Secondary response diagnostic in realized BF16 log-gain coordinates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NORM_FLOOR = 1e-12  # Fixed numerical guard, not an outcome-selected threshold.
LABELS = ("plus", "minus", "half_plus", "half_minus")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(value, shape=None, positive=False):
    result = np.asarray(value, dtype=np.float64)
    require(result.ndim == 1 and result.size > 0 and np.isfinite(result).all(), "Expected finite nonempty vector")
    require(shape is None or result.shape == shape, "Vector shape mismatch")
    require(not positive or (result > 0).all(), "Gain values must be positive")
    return result


def analyze_direction(record, direction):
    """Fit through the origin using full +/- responses only; half responses are held aside."""
    base = vector(record["baseline_gain_cast_bfloat16"], positive=True)
    base32 = vector(record["baseline_gain_float32"], base.shape, positive=True)
    baseline = vector(record["baseline"])
    epsilon = float(record["epsilon"])
    require(np.isfinite(epsilon) and epsilon > 0, "Expected positive finite nominal epsilon")
    gains, responses, controls = {}, {}, {}
    for label in LABELS:
        realized = direction["realized_gains"][label]
        gain = vector(realized["cast_bfloat16"], base.shape, positive=True)
        gain32 = vector(realized["float32"], base.shape, positive=True)
        gains[label] = 2 * (np.log(gain) - np.log(base))
        responses[label] = vector(direction[label], baseline.shape) - baseline
        exact_zero = bool(np.array_equal(gain, base))
        fp32_zero = bool(np.array_equal(gain32, base32))
        controls[label] = {"bf16_qk_delta_l2": float(np.linalg.norm(gain - base)),
                           "fp32_qk_delta_l2": float(np.linalg.norm(gain32 - base32)),
                           "bf16_logit_delta_l2": float(np.linalg.norm(gains[label])),
                           "bf16_changed_coordinates": int(np.count_nonzero(gain != base)),
                           "bf16_exact_zero": exact_zero, "fp32_exact_zero": fp32_zero,
                           "quantization_dead": exact_zero and not fp32_zero,
                           "projected_amplitude": None, "collinearity_residual_l2": None,
                           "collinearity_residual_fraction": None}
    secant = (gains["plus"] - gains["minus"]) / 2
    secant_norm = float(np.linalg.norm(secant))
    identifiable = secant_norm > NORM_FLOOR
    amplitudes, slope, predictions = {}, None, None
    if identifiable:
        denominator = float(np.dot(secant, secant))
        for label in LABELS:
            amplitudes[label] = float(np.dot(gains[label], secant) / denominator)
            orthogonal = gains[label] - amplitudes[label] * secant
            residual = float(np.linalg.norm(orthogonal))
            norm = controls[label]["bf16_logit_delta_l2"]
            controls[label].update(projected_amplitude=amplitudes[label], collinearity_residual_l2=residual,
                                   collinearity_residual_fraction=residual / norm if norm > 0 else 0.)
        full_amplitudes = np.asarray([amplitudes["plus"], amplitudes["minus"]])
        full_response = np.stack([responses["plus"], responses["minus"]])
        slope = full_amplitudes @ full_response / float(full_amplitudes @ full_amplitudes)
        predictions = np.stack([amplitudes[label] * slope for label in ("half_plus", "half_minus")])
    observed = np.stack([responses["half_plus"], responses["half_minus"]])
    effect_norm = float(np.linalg.norm(observed))
    nominal_slope = (responses["plus"] - responses["minus"]) / (2 * epsilon)
    nominal_prediction = np.stack([epsilon / 2 * nominal_slope, -epsilon / 2 * nominal_slope])
    nominal_residual = float(np.linalg.norm(observed - nominal_prediction))
    realized_residual = float(np.linalg.norm(observed - predictions)) if identifiable else None
    return {"direction": direction["name"], "epsilon": epsilon, "numerical_norm_floor": NORM_FLOOR,
            "fit_observations": ["plus", "minus"], "held_aside_observations": ["half_plus", "half_minus"],
            "full_secant_l2": secant_norm, "full_secant_exact_zero": secant_norm == 0,
            "full_secant_tiny_nonzero": 0 < secant_norm <= NORM_FLOOR,
            "fit_status": "identified" if identifiable else "zero_or_tiny_full_secant",
            "full_projected_amplitudes": {label: amplitudes.get(label) for label in ("plus", "minus")},
            "fitted_response_slope": slope.tolist() if identifiable else None,
            "half_predictions": predictions.tolist() if identifiable else None,
            "half_effect_l2": effect_norm, "half_effect_exact_zero": effect_norm == 0,
            "half_effect_tiny_nonzero": 0 < effect_norm <= NORM_FLOOR,
            "nominal_epsilon_residual_l2": nominal_residual,
            "nominal_epsilon_relative_prediction_error": nominal_residual / effect_norm if effect_norm > NORM_FLOOR else None,
            "realized_coordinate_residual_l2": realized_residual,
            "realized_coordinate_relative_prediction_error": realized_residual / effect_norm
                if identifiable and effect_norm > NORM_FLOOR else None,
            "controls": controls}


def analyze_report(report_path):
    """Read only an integrity-checked, complete four-stage response-report archive."""
    report_path = Path(report_path)
    require(report_path.is_file(), "Validated response report is not available")
    receipt_path = report_path.parent / "archive_receipt.json"
    require(receipt_path.is_file(), "Validated archive receipt is not available")
    report = json.loads(report_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("report_sha256") == sha256(report_path), "Validated report hash mismatch")
    require(report.get("automatic_promotion") is False, "Unexpected promotion policy")
    expected = {f"prompt{i}/{stage}" for i in (0, 1) for stage in ("smoke", "4k")}
    require(set(report["stages"]) == expected, "Both prompts require validated smoke and 4K stages")
    for key in sorted(expected):
        stage = report["stages"][key]
        validation = stage["summary"]["validation"]
        require(all(validation.get(flag) is True for flag in
                    ("gpu_probe_executed", "trajectory_bitwise_equal", "baseline_history_equal")),
                "Response stage did not pass baseline-invariance validation")
        require(stage["endpoint_verification"].get("independently_verified_on_cpu") is True,
                "Missing independent endpoint verification")
        require(stage["raw_json"]["completion.json"] == validation and
                stage["raw_json"]["validation.json"] == validation, "Stage validation receipts disagree")
    rows = []
    for index in (0, 1):
        stage = report["stages"][f"prompt{index}/4k"]
        manifest = stage["summary"]["manifest"]
        require(manifest["prompt_index"] == index and manifest["res"] == [4096, 4096], "Wrong prompt or stage")
        for step in manifest["probe_steps"]:
            record = stage["raw_json"][f"step_{step:03d}.json"]
            require(record["step"] == step, "Recorded probe step mismatch")
            for direction in record["directions"]:
                rows.append({"prompt_index": index, "prompt_id": manifest["prompt"]["id"],
                             "stage": "4k", "step": step, "sigma": record["sigma"],
                             **analyze_direction(record, direction)})
    return {"purpose": "Unprespecified secondary diagnostic of realized BF16 gain coordinates",
            "limitations": "This is not a new acceptance rule, efficacy result or novelty claim. "
                            "Half responses never enter fitting or parameter choice. Orthogonal gain changes "
                            "can invalidate a one-dimensional prediction even for a locally linear model.",
            "automatic_promotion": False, "acceptance_rule_changed": False,
            "job_id": report["job_id"], "validated_report": str(report_path.resolve()),
            "validated_report_sha256": sha256(report_path), "archive_receipt_sha256": sha256(receipt_path),
            "analysis_source_sha256": sha256(__file__), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / ".agent/notes/response_cr17/report.json")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out:
        require(not args.out.exists(), "Secondary output must use a fresh path")
    encoded = json.dumps(analyze_report(args.report), indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
