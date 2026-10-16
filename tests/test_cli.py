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


@pytest.mark.parametrize("method", ["constancy_tm", "constancy_tm_nogate"])
def test_time_matched_cli_and_rebuild_selection(method):
    from scripts import calibrate, enqueue, sample

    args = sample.parser().parse_args(["--method", method, "--res", "4096x4096",
                                      "--prompts", "prompts.jsonl", "--out", "out",
                                      "--profile", "tm.npz", "--gate", "off"])
    assert args.method == method and args.profile == Path("tm.npz") and args.gate == "off"
    args = enqueue.parser().parse_args(["--stage", "main", "--methods", method,
                                       "--prompts", "prompts.jsonl", "--profile", "tm.npz"])
    assert args.methods == [method] and args.profile == Path("tm.npz")
    args = calibrate.parser().parse_args(["--mode", "native-rebuild", "--traces", "native_traces",
                                         "--out", "tm.npz"])
    assert args.traces == Path("native_traces") and args.out == Path("tm.npz")


@pytest.mark.parametrize("method", ["constancy_tm", "sega_ctm"])
def test_time_matched_sample_rejects_late_only_profile_before_loading_model(tmp_path, monkeypatch, method):
    import torch
    from crope import runtime
    from crope.spectral import NativeProfile
    from scripts import sample

    profile = tmp_path / "old.npz"
    f = torch.tensor([0.02, 0.04])
    NativeProfile((f, f), (f, f)).save(profile)
    monkeypatch.setattr(runtime, "load_pipeline", lambda *a: pytest.fail("Must validate profile before model load"))
    with pytest.raises(ValueError, match="time-resolved arrays"):
        sample.main(["--method", method, "--res", "4096x4096", "--prompts", "unused.jsonl",
                     "--out", str(tmp_path / "out"), "--profile", str(profile)])
