"""Calibration aggregation and resume tests without model weights or GPU work."""

from argparse import Namespace
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import pytest
import torch

import crope.gain as gain_module
import crope.runtime as runtime
from crope.gain import GainController
from crope.spectral import POWER_NORMALIZATION, NativeProfile, axis_power
from scripts import calibrate


@pytest.fixture(scope="module", autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def scratch_dir():
    root = Path(os.environ.get("CROPE_TEST_OUTPUT", "/scratch/09032/saini_2/constancy_rope/cpu_tests"))
    root.mkdir(parents=True, exist_ok=True)
    # Keep each small test receipt. Scratch deletion is prohibited in this workspace.
    return Path(tempfile.mkdtemp(prefix="calibration_", dir=root))


def finished_record(directory, prompt_id, modulators, *, res="4096x4096", method="sega",
                    guidance=3.5, complete=True, steps=None, seed=0):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prompt_id}_{seed}.json"
    image = path.with_suffix(".png")
    Image.new("RGB", (1, 1), (1, 2, 3)).save(image)
    record = {
        "complete": complete,
        "image": str(image.resolve()),
        "config": {"method": method, "res": res, "prompt_id": str(prompt_id), "seed": seed,
                   "steps": len(modulators) if steps is None else steps, "guidance": guidance,
                   "model": "black-forest-labs/FLUX.1-dev", "model_revision": "test-model"},
        "gains": [{"step": step, "modulator": np.asarray(value).tolist()}
                  for step, value in enumerate(modulators)],
    }
    if method == "sega_official":
        record["gains"] = [{"step": step, "controller_log_g": (2 * np.log(value)).tolist(),
                            "log_g": (2 * np.log(value) + 0.5).tolist()}
                           for step, value in enumerate(modulators)]
    path.write_text(json.dumps(record))
    return path


def test_static_averages_prompt_steps_with_height_width_order(scratch_dir, monkeypatch):
    first = np.arange(2 * 2 * 28).reshape(2, 2, 28) / 400 + 0.7
    second = np.flip(first, axis=-1) + 0.2
    directory = scratch_dir / "runs"
    finished_record(directory, "a", first)
    finished_record(directory, "b", second)
    finished_record(directory, "wrong-res", first, res="2048x2048")
    finished_record(directory, "wrong-method", first, method="sega_official")
    finished_record(directory, "unfinished", first, complete=False)
    out = scratch_dir / "static.npz"
    calibrate.static_profile([directory], (4096, 4096), out)
    expected = np.concatenate((first, second)).mean(0)
    with np.load(out, allow_pickle=False) as result:
        np.testing.assert_array_equal(result["mean_modulator"], expected)
        np.testing.assert_array_equal(result["grid"], [256, 256])
        assert result["axis_order"].tolist() == ["height", "width"]
        assert result["prompt_ids"].tolist() == ["a", "b"]
        assert int(result["n_images"]) == 2 and int(result["n_steps"]) == 4
        assert len(result["source_json"]) == 2

    def forbidden_fft(*args, **kwargs):
        raise AssertionError("SEGA-static must never estimate spectra at runtime")

    monkeypatch.setattr(torch.fft, "fft2", forbidden_fft)
    monkeypatch.setattr(gain_module, "axis_power", forbidden_fft)
    monkeypatch.setattr(gain_module, "sega_modulator", forbidden_fft)
    controller = GainController("sega_static", grid=(256, 256), static_profile=out)
    controller.update(torch.zeros(1, 256, 256), 0.5)
    reference = 4 ** 0.08
    for axis, offset in ((0, 16), (1, 72)):
        target = torch.from_numpy(np.repeat(reference * expected[axis], 2)).float()
        torch.testing.assert_close(controller.qk_scale[offset:offset + 56], target)
    torch.testing.assert_close(controller.qk_scale[:16], torch.full((16,), reference))
    assert controller.history[-1]["modulator"] == expected.tolist()


def test_official_static_averages_qk_gains_without_runtime_fft(scratch_dir, monkeypatch):
    first = np.repeat(np.arange(2 * 64).reshape(2, 64) / 100 + 1, 2, axis=1)
    first[:, :16] = 1
    second = first.copy()
    second[:, 16:] *= 1.3
    directory = scratch_dir / "runs"
    finished_record(directory, "a", first, method="sega_official")
    finished_record(directory, "b", second, method="sega_official")
    finished_record(directory, "reference", np.ones((2, 2, 28)))
    finished_record(directory, "wrong-res", first, method="sega_official", res="2048x2048")
    finished_record(directory, "unfinished", first, method="sega_official", complete=False)
    out = scratch_dir / "official_static.npz"
    calibrate.static_profile([directory], (4096, 4096), out, method="sega_official")
    expected = np.concatenate((first, second)).mean(0)
    with np.load(out, allow_pickle=False) as result:
        np.testing.assert_allclose(result["mean_qk_scale"], expected)
        np.testing.assert_array_equal(result["grid"], [256, 256])
        assert str(result["source_method"]) == "sega_official"
        assert result["prompt_ids"].tolist() == ["a", "b"]
        assert int(result["n_images"]) == 2 and int(result["n_steps"]) == 4
        assert "mean_modulator" not in result

    def forbidden_fft(*args, **kwargs):
        raise AssertionError("Released-SEGA static must not estimate spectra at runtime")

    monkeypatch.setattr(torch.fft, "fft2", forbidden_fft)
    monkeypatch.setattr(torch.fft, "fft", forbidden_fft)
    monkeypatch.setattr(gain_module, "axis_power", forbidden_fft)
    controller = GainController("sega_official_static", grid=(256, 256), static_profile=out)
    controller.update(torch.zeros(1, 256, 256), 0.5)
    torch.testing.assert_close(controller.qk_scale, torch.from_numpy(expected).float())
    controller.update(torch.ones(1, 256, 256), 0.1)
    torch.testing.assert_close(controller.qk_scale, torch.from_numpy(expected).float())


@pytest.mark.parametrize("failure", ["shape", "nan", "pairs"])
def test_official_static_rejects_invalid_qk_profiles(scratch_dir, failure):
    values = np.ones((2, 128))
    if failure == "shape":
        values = values[:, :-2]
    elif failure == "nan":
        values[0, 0] = np.nan
    else:
        values[0, 17] = 1.5
    directory = scratch_dir / "runs"
    finished_record(directory, "a", values, method="sega_official")
    with pytest.raises(ValueError, match="q/k gains"):
        calibrate.static_profile([directory], (4096, 4096), scratch_dir / "static.npz", method="sega_official")


@pytest.mark.parametrize("failure,match", [
    ("duplicate", "Duplicate prompt"),
    ("steps", "Incomplete gain trajectory"),
    ("shape", "Invalid SEGA M vectors"),
    ("nan", "Invalid SEGA M vectors"),
    ("settings", "mixes models or sampling settings"),
    ("unfinished", "No completed SEGA images"),
])
def test_static_rejects_invalid_calibration_inputs(scratch_dir, failure, match):
    first = np.ones((2, 2, 28))
    directory = scratch_dir / "runs"
    if failure == "duplicate":
        finished_record(directory / "one", "a", first)
        finished_record(directory / "two", "a", first)
    elif failure == "steps":
        finished_record(directory, "a", first, steps=3)
    elif failure == "shape":
        finished_record(directory, "a", first[:, :, :-1])
    elif failure == "nan":
        first[0, 0, 0] = np.nan
        finished_record(directory, "a", first)
    elif failure == "settings":
        finished_record(directory, "a", first)
        finished_record(directory, "b", first, guidance=4.0)
    else:
        finished_record(directory, "a", first, complete=False)
    out = scratch_dir / "static.npz"
    with pytest.raises(ValueError, match=match):
        calibrate.static_profile([directory], (4096, 4096), out)
    assert not out.exists()


def test_static_profile_resolution_is_enforced(scratch_dir):
    path = scratch_dir / "profile.npz"
    np.savez(path, mean_modulator=np.ones((2, 28)), grid=np.array([128, 128]))
    with pytest.raises(ValueError, match="resolution does not match"):
        GainController("sega_static", grid=(256, 256), static_profile=path)


def test_native_profile_rejects_legacy_summed_power(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, frequency_h=[0.02, 0.04], frequency_w=[0.02, 0.04],
             log_power_h=[1, 2], log_power_w=[1, 2])
    with pytest.raises(ValueError, match="power normalization; recalibrate"):
        NativeProfile.load(path)


def test_native_calibration_late_log_mean_and_resume(scratch_dir, monkeypatch):
    rows = [{"id": "0", "prompt": "first test"}, {"id": "1", "prompt": "second test"}]
    prompts = scratch_dir / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    instances, generated = [], []
    sigmas = [0.9, 0.3, 0.29, 0.1]

    def clean_latent(row):
        return torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(71 + int(row["id"])))

    class FakeExperiment:
        def __init__(self, pipe, method, resolution):
            assert method == "base" and resolution == (1024, 1024)
            self.closed = False
            instances.append(self)

        def config(self, row, seed, steps, guidance):
            return {"prompt_id": row["id"], "prompt": row["prompt"], "seed": seed,
                    "steps": steps, "guidance": guidance, "model": "test"}

        def generate(self, row, seed, steps, guidance, output_type, observer):
            assert steps == len(sigmas) and output_type == "latent"
            generated.append(row["id"])
            clean = clean_latent(row)
            for step, sigma in enumerate(sigmas):
                x0 = clean * (step + 1)
                observer(step, sigma, torch.zeros_like(clean), x0)
            return None, 0.001, {}

        def close(self):
            self.closed = True

    monkeypatch.setattr(runtime, "load_pipeline", lambda *args: object())
    monkeypatch.setattr(runtime, "FluxExperiment", FakeExperiment)
    args = Namespace(prompts=prompts, start=0, count=2, seed=7, steps=4, guidance=3.5,
                     out=scratch_dir / "native.npz", device="cuda", cpu_offload=False,
                     allow_download=False)
    calibrate.native_profile(args)
    assert generated == ["0", "1"] and all(instance.closed for instance in instances)
    profile = NativeProfile.load(args.out)
    for axis in (0, 1):
        expected = torch.stack([axis_power(clean_latent(row))[axis].power.log() + 2 * math.log(scale)
                                for row in rows for scale in (3, 4)]).mean(0)
        torch.testing.assert_close(profile.log_power[axis], expected)
    with np.load(args.out, allow_pickle=False) as data:
        assert str(data["power_normalization"]) == POWER_NORMALIZATION
        assert int(data["n_prompts"]) == 2 and int(data["n_late_steps"]) == 4
        assert float(data["late_sigma_threshold"]) == 0.3
        assert len(data["source_traces"]) == 2
        for trace in data["source_traces"]:
            with np.load(trace, allow_pickle=False) as trajectory:
                assert json.loads(str(trajectory["config_json"]))["power_normalization"] == POWER_NORMALIZATION
                np.testing.assert_array_equal(trajectory["sigma"], sigmas)
                assert trajectory["log_power_h"].shape[0] == 4

    calibrate.native_profile(args)
    assert generated == ["0", "1"], "Resume must reuse complete per-prompt traces"
    assert all(instance.closed for instance in instances)
    args.guidance = 4.0
    with pytest.raises(ValueError, match="different settings"):
        calibrate.native_profile(args)
    assert all(instance.closed for instance in instances)
    args.guidance = 3.5
    with np.load(trace, allow_pickle=False) as data:
        legacy_trace = {key: data[key].copy() for key in data.files}
    legacy_config = json.loads(str(legacy_trace["config_json"]))
    legacy_config.pop("power_normalization")
    legacy_trace["config_json"] = json.dumps(legacy_config)
    legacy_trace["config_fingerprint"] = runtime.fingerprint(legacy_config)
    np.savez(trace, **legacy_trace)
    with pytest.raises(ValueError, match="different settings"):
        calibrate.native_profile(args)
