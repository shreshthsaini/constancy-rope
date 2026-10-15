"""Time and frequency interpolation, profile compatibility, and trace validation."""

import json

import numpy as np
import pytest
import torch

from crope.gain import GainController
from crope.spectral import NativeProfile, POWER_NORMALIZATION
from scripts import calibrate


@pytest.mark.parametrize("descending", [False, True])
def test_time_profile_interpolates_sigma_and_log_frequency_and_clamps(descending, tmp_path):
    f = torch.tensor([1 / 64, 1 / 32, 1 / 16, 1 / 8], dtype=torch.float64)
    sigmas = torch.tensor([0.1, 0.3, 0.6, 0.9], dtype=torch.float64)
    if descending:
        sigmas = sigmas.flip(0)
    powers = tuple(torch.stack([a + 2 * s - (1 + s) * f.log() for s in sigmas]) for a in (0, 3))
    profile = NativeProfile((f, f), tuple(p[-1] for p in powers), sigmas, powers)
    path = tmp_path / "time_profile.npz"
    profile.save(path)
    profile = NativeProfile.load(path)
    query = torch.tensor([1 / 256, 1 / 128, 1 / 48, 1 / 12, 1 / 8, 1 / 4], dtype=torch.float64)
    for sigma in (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
        clamped = min(0.9, max(0.1, sigma))
        for axis, offset in enumerate((0, 3)):
            expected = offset + 2 * clamped - (1 + clamped) * query.log()
            torch.testing.assert_close(profile.evaluate_time(axis, query, sigma), expected, atol=1e-12, rtol=0)


def test_single_time_step_and_old_profile_compatibility(tmp_path):
    f = torch.tensor([1 / 64, 1 / 32, 1 / 16])
    legacy = NativeProfile((f, f), (-f.log(), -2 * f.log()))
    path = tmp_path / "old.npz"
    legacy.save(path)
    loaded = NativeProfile.load(path)
    GainController("constancy", profile=loaded)
    for mode in ("constancy_tm", "constancy_tm_nogate"):
        with pytest.raises(ValueError, match="time-resolved arrays.*sigma_steps"):
            GainController(mode, profile=path)
        with pytest.raises(ValueError, match="requires --profile"):
            GainController(mode)
    profile = NativeProfile(legacy.frequencies, legacy.log_power, torch.tensor([0.3]),
                            tuple(p[None] for p in legacy.log_power))
    for a in (0, 1):
        for sigma in (0.0, 0.3, 1.0):
            torch.testing.assert_close(profile.evaluate_time(a, f, sigma), legacy.evaluate(a, f))


@pytest.mark.parametrize("failure", ["partial", "shape", "nan", "duplicate_sigma", "unsorted_sigma"])
def test_invalid_time_profile_is_rejected(tmp_path, failure):
    arrays = dict(frequency_h=[0.02, 0.04], frequency_w=[0.02, 0.04],
                  log_power_h=[1.0, 2.0], log_power_w=[1.0, 2.0],
                  sigma_steps=[0.9, 0.6, 0.1], log_power_t_h=np.ones((3, 2)),
                  log_power_t_w=np.ones((3, 2)), power_normalization=POWER_NORMALIZATION)
    if failure == "partial":
        arrays.pop("log_power_t_w")
    elif failure == "shape":
        arrays["log_power_t_h"] = np.ones((2, 2))
    elif failure == "nan":
        arrays["log_power_t_h"][0, 0] = np.nan
    elif failure == "duplicate_sigma":
        arrays["sigma_steps"] = [0.9, 0.9, 0.1]
    else:
        arrays["sigma_steps"] = [0.9, 0.1, 0.6]
    path = tmp_path / "bad.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="[Tt]ime-resolved|sigma_steps"):
        NativeProfile.load(path)


@pytest.mark.parametrize("failure", ["sigma", "frequency", "normalization", "incomplete", "settings", "duplicate"])
def test_native_rebuild_rejects_misaligned_or_incompatible_traces(tmp_path, failure):
    directory = tmp_path / "traces"
    directory.mkdir()
    for i in range(2):
        config = dict(prompt_id=str(i), seed=0, steps=3, guidance=4.5, model="test",
                      power_normalization=POWER_NORMALIZATION)
        arrays = dict(sigma=np.array([0.9, 0.6, 0.1]), frequency_h=np.array([0.02, 0.04]),
                      frequency_w=np.array([0.02, 0.04]), log_power_h=np.ones((3, 2)),
                      log_power_w=np.ones((3, 2)))
        if i:
            if failure == "sigma":
                arrays["sigma"][1] = 0.5
            elif failure == "frequency":
                arrays["frequency_w"][0] = 0.03
            elif failure == "normalization":
                config.pop("power_normalization")
            elif failure == "incomplete":
                arrays["sigma"] = arrays["sigma"][:2]
            elif failure == "settings":
                config["guidance"] = 3.5
            else:
                config["prompt_id"] = "0"
        np.savez(directory / f"{i}.npz", **arrays, config_json=json.dumps(config))
    out = tmp_path / "profile.npz"
    with pytest.raises(ValueError):
        calibrate.rebuild_native_profile(directory, out)
    assert not out.exists()
