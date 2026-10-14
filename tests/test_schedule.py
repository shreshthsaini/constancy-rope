"""Expose high-resolution behavior without silently changing the protocol."""

import pytest

from crope.schedule import diagnose_schedule


def test_default_flow_shift_extrapolates_past_native_range():
    config = {"base_image_seq_len": 256, "max_image_seq_len": 4096,
              "base_shift": 0.5, "max_shift": 1.15, "use_dynamic_shifting": True}
    native = diagnose_schedule(config, (1024, 1024))
    high = diagnose_schedule(config, (4096, 4096))
    assert native["mu"] == pytest.approx(1.15)
    assert native["last_evaluated_sigma"] == pytest.approx(0.104720894, abs=1e-7)
    assert high["mu"] == pytest.approx(11.55)
    assert high["last_evaluated_sigma"] > 0.999
    assert high["distinct_model_timesteps"] == 1
    assert high["evaluations_below_sigma_0_3"] == 0
