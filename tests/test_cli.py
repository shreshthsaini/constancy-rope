"""Direct script entry points must not require models for help."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("script", ["sample", "calibrate", "probe_mtf", "eval", "prompts", "enqueue", "diagnose_schedule"])
def test_script_help(script):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts" / f"{script}.py"), "--help"],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_baseline_cli_defaults_and_static_calibration_selection():
    from scripts import calibrate, eval as evaluate, sample

    args = evaluate.parser().parse_args(["--root", "out", "--prompts", "prompts.jsonl", "--out", "eval"])
    assert args.reference_method == "sega_official"
    args = sample.parser().parse_args(["--method", "sega_official_static", "--res", "4096x4096",
                                      "--prompts", "prompts.jsonl", "--out", "out",
                                      "--sega-static-profile", "official.npz"])
    assert args.method == "sega_official_static" and args.sega_static_profile == Path("official.npz")
    args = calibrate.parser().parse_args(["--mode", "sega-static", "--method", "sega_official",
                                         "--runs", "pilot/sega_official", "--res", "4096x4096"])
    assert args.method == "sega_official"
