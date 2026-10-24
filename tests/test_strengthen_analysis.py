"""Protect pairing, fixed populations, equivalence boundaries, multiplicity and replay."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import strengthen_analysis as a


def prompts(n=195, prefix="p"):
    return [{"id": f"{prefix}{i}", "prompt": f"caption {prefix}{i}"} for i in range(n)]


def design(block):
    return a.registered_design({"block": block, "resolution": "4096x4096"},
                               prompts(195 if block in ["P1", "P3", "E2"] else 150))


def scores(d, shifts=None):
    shifts = shifts or {}
    return pd.DataFrame([{ "method": method, "prompt_id": pid, "seed": seed,
                          **{metric: float(i + seed + shifts.get(method, 0)) for metric in d["metrics"]}}
                         for method in d["methods"] for i, pid in enumerate(d["prompt_ids"])
                         for seed in d["seeds"]])


@pytest.mark.parametrize("block,n,methods,seeds", [
    ("P1", 187, 5, [0]), ("P3", 55, 5, [0]), ("E1", 150, 6, [0]),
    ("E2", 187, 2, [0, 1, 2]), ("E3", 150, 3, [0])])
def test_registered_populations(block, n, methods, seeds):
    d = design(block)
    assert len(d["prompt_ids"]) == n and len(d["methods"]) == methods and d["seeds"] == seeds
    if block in ["P1", "P3", "E2"]:
        assert d["prompt_ids"][0] == "p8" and "p7" not in d["prompt_ids"]
    assert not {"constancy_tm", "sega_ctm"} & set(d["methods"])


def test_bootstrap_matches_frozen_protocol_and_rejects_nonfinite():
    x = np.array([-4., 1., 2., 9.])
    draws = np.random.default_rng(0).choice(x, (2000, len(x)), replace=True).mean(1)
    assert a.boot(x) == (x.mean(), *np.percentile(draws, [2.5, 97.5]))
    for bad in [[], [np.inf], [np.nan], [[1, 2]]]:
        with pytest.raises(ValueError):
            a.boot(bad)


def test_centered_bootstrap_tail_formula_and_tost_boundary():
    values = np.array([-1., 0., 1., 2.])
    err = a.bootstrap_draws(values) - values.mean()
    assert a.bootstrap_p(values) == (1 + np.count_nonzero(abs(err) >= abs(values.mean()))) / 2001
    assert a.bootstrap_p(values, 1., "less") == (1 + np.count_nonzero(err <= values.mean() - 1)) / 2001
    assert a.bootstrap_p(np.ones(20), alternative="greater") == 1 / 2001
    assert a.tost_p(np.zeros(20), .1) == 1 / 2001
    assert a.tost_p(np.full(20, .1), .1) == 1.0
    assert a.bootstrap_p(np.zeros(20)) == 1.


def test_holm_keeps_family_slots_and_order():
    assert a.holm([.03, .001, .02]).tolist() == [.04, .003, .04]
    assert a.holm([.001], 80).tolist() == [.08]
    for p in [[float("nan")], [-.1], [1.1]]:
        with pytest.raises(ValueError):
            a.holm(p)
    with pytest.raises(ValueError):
        a.holm([.1, .2], 1)


def test_factorial_forms_interaction_within_prompt_and_reports_all80():
    d = design("P1")
    frame = scores(d, {"ntk2": 0, "sega_base_none": 2, "ntk2_sref": 3,
                       "temp_only_sbase": 9, "sega_official": 10})
    _, _, means, _ = a.summarize(frame, d, True)
    result = a.exploratory_contrasts(means, d, True)
    assert len(result) == 80 and set(result.family_size) == {80}
    assert set(result.n) == {187} and result.holm_reject.all()
    expect = {"base_at_none": 2, "base_at_sref": 6, "temperature_at_ntk2": 3,
              "temperature_at_sega": 7, "base_main_effect": 4,
              "temperature_main_effect": 5, "interaction": 4, "allocation": 1}
    for row in result.itertuples():
        assert row.delta == row.lo == row.hi == expect[row.contrast]


def test_missing_factorial_arm_cannot_shrink_holm_or_support_any_result():
    d = design("P1")
    frame = scores(d)
    frame = frame.loc[frame.method != "sega_base_none"]
    frame, status = a.coverage(frame, d)
    assert not status["complete"] and len(status["missing_samples"]) == 187
    _, _, means, _ = a.summarize(frame, d, False)
    report = a.exploratory_contrasts(means, d, False)
    assert len(report) == 80 and set(report.family_size) == {80}
    assert not report.holm_reject.any()
    assert set(report.loc[report.contrast == "interaction", "p"]) == {1.}


def test_multiseed_averages_within_prompt_and_incomplete_seed_group_drops_whole_prompt():
    d = design("E2")
    frame = scores(d, {"temp_only_sbase": 2})
    frame.loc[frame.seed == 2, "imagereward"] += 30
    _, paired, means, _ = a.summarize(frame, d, True)
    assert set(paired.n) == {187}
    row = means.loc[(means.analysis == "seeds012") & (means.method == "sega_official") &
                    (means.prompt_id == "p8") & (means.metric == "imagereward")].iloc[0]
    assert row["mean"] == 11.
    frame = frame.loc[~((frame.method == "temp_only_sbase") & (frame.prompt_id == "p8") & (frame.seed == 2))]
    _, paired, _, omitted = a.summarize(frame, d, False)
    assert set(paired.loc[paired.analysis == "seeds012", "n"]) == {186}
    assert set(paired.loc[paired.analysis == "seed0", "n"]) == {187}
    assert any(r["prompt_id"] == "p8" for r in omitted)


def test_sweep_keeps_reference_and_fixed_family40():
    d = design("P3")
    _, paired, means, _ = a.summarize(scores(d), d, True)
    assert set(paired.reference) == {"tau_sref"}
    report = a.exploratory_contrasts(means, d, True)
    assert len(report) == 40 and set(report.family_size) == {40}
    assert a.SWEEP_TEMPERATURES["tau_sref"] == 4 ** .16
    assert a.SWEEP_TEMPERATURES["tau_logn"] == 4 / 3


def test_equivalence_is_not_a_zero_crossing_and_holm_is_required():
    spread = np.tile([-1., 1.], 20)
    report = a.three_hypotheses(spread, np.zeros(40), np.ones(40), True, "test")
    assert report["H1"] == "not_established" and report["H2"] == "supported"
    report = a.three_hypotheses(np.zeros(40), np.zeros(40), np.ones(40), False, "test")
    assert report["H1"] == report["H2"] == "incomplete"
    assert all(r["status"] == "incomplete" for r in report["family"])


def test_pool337_only_for_h1_and_fresh150_only_for_h2():
    d = design("E1")
    d["prompt_ids"] = [f"fresh{i}" for i in range(150)]
    _, _, means, _ = a.summarize(scores(d, {"sega_official": .01}), d, True)
    old = pd.DataFrame([{"method": m, "prompt_id": f"p{i}", "idx": i, "seed": 0,
                         "imagereward": 1., "pickscore": 1., "musiq": -999.}
                        for m in ["temp_only_sbase", "sega_official"] for i in range(8, 195)])
    result = a.pooled_confirmation(means, d, True, old, prompts())
    assert result["H1"] == result["H2"] == "supported"
    assert [r["n"] for r in result["family"]] == [337, 337, 150]
    assert result["family"][2]["delta"] == pytest.approx(.01)
    assert result["H3"] == "not_tested_historical_only"
    result = a.pooled_confirmation(means, d, True, old.iloc[1:], prompts())
    assert result["H1"] == result["H2"] == "incomplete"
    corrupt = old.copy()
    corrupt.loc[0, "idx"] = 7
    with pytest.raises(ValueError, match="idx"):
        a.pooled_confirmation(means, d, True, corrupt, prompts())


def test_sources_join_ir_by_path_and_reject_duplicate_seed_alias(tmp_path, monkeypatch):
    d = design("E2")
    path0, path1 = str(tmp_path / "s0.png"), str(tmp_path / "s1.png")
    pi = tmp_path / "per_image.csv"
    ir = tmp_path / "ir.csv"
    records = [{"method": "temp_only", "prompt_id": "p8", "res": "4096x4096",
                "seed": seed, "path": path} for seed, path in enumerate([path0, path1])]
    pd.DataFrame(records).to_csv(pi, index=False)
    pd.DataFrame([{**r, "png": r["path"], "imagereward": r["seed"] + 4.} for r in records]).to_csv(ir, index=False)
    spec = {"block": "E2", "sources": [{"per_image": str(pi), "imagereward": str(ir),
             "methods": ["temp_only"], "aliases": {"temp_only": "temp_only_sbase"}}]}
    monkeypatch.setattr(a, "inspect_sidecar", lambda *args: {})
    data, _, _, issues = a.load_sources(spec, prompts(), d, tmp_path / "out", [])
    assert data.imagereward.tolist() == [4., 5.] and not issues
    records[1]["path"] = path0
    pd.DataFrame(records).to_csv(pi, index=False)
    with pytest.raises(ValueError, match="aliases"):
        a.load_sources(spec, prompts(), d, tmp_path / "out", [])


def test_replay_rejects_old_protocol_before_reinterpreting(tmp_path):
    source = tmp_path / "old"
    source.mkdir()
    (source / "analysis_protocol.json").write_text('{}')
    with pytest.raises(ValueError, match="superseded"):
        a.replay(source, tmp_path / "out")


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    from crope.utils import image_seed
    from PIL import Image

    class ImageHeader:
        size = (4096, 4096)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(Image, "open", lambda _: ImageHeader())
    png = tmp_path / "image.png"
    png.write_bytes(b"Header checked through stub; no full pixel decode in analysis")
    config = dict(method="temp_only_sbase", controller_method="temp_only", prompt_id="p8", prompt="caption p8",
                  seed=0, derived_seed=image_seed("p8", 0), res="4096x4096", steps=28, guidance=4.5,
                  model="black-forest-labs/FLUX.1-dev", model_revision="3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
                  dtype="bfloat16", eta=1., scheduler={"max_shift": 1.15, "max_image_seq_len": 65536},
                  rope={"method": "sega_official"}, temperature="sref")
    x = np.linspace(1., 1 / 28, 28)
    record = {"config": config, "complete": True, "sigmas": (np.exp(1.15) / (np.exp(1.15) + 1 / x - 1)).tolist()}
    row = dict(path=str(png), raw_method="temp_only_sbase", method="temp_only_sbase",
               prompt_id="p8", seed=0, res="4096x4096")
    def save():
        record["config_fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        png.with_suffix(".json").write_text(json.dumps(record))
    save()
    return row, record, save


def test_unique_and_legacy_output_identity_validate_same_controller(sidecar):
    row, record, save = sidecar
    assert a.inspect_sidecar(row, "caption p8", {"block": "E1"}, {})["method"] == "temp_only_sbase"
    record["config"]["method"] = "temp_only"
    record["config"].pop("controller_method")
    save()
    assert a.inspect_sidecar(row, "caption p8", {"block": "E1"}, {})["config"]["method"] == "temp_only"


@pytest.mark.parametrize("corruption", ["override", "seed", "schedule", "controller", "rope", "fingerprint"])
def test_sidecar_rejects_important_control_corruptions(sidecar, corruption):
    row, record, save = sidecar
    if corruption == "override":
        record["config"]["temperature_multiplier"] = 1.1
    elif corruption == "seed":
        record["config"]["derived_seed"] += 1
    elif corruption == "schedule":
        record["sigmas"] = [1.] * 28
    elif corruption == "controller":
        record["config"].pop("controller_method")
    elif corruption == "rope":
        record["config"]["rope"]["method"] = "ntk2"
    save()
    if corruption == "fingerprint":
        path = Path(row["path"]).with_suffix('.json')
        bad = json.loads(path.read_text())
        bad["config_fingerprint"] = "wrong"
        path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        a.inspect_sidecar(row, "caption p8", {"block": "E1"}, {})


def test_replay_reproduces_archived_scores_and_rejects_tamper(tmp_path):
    d = design("E3")
    d["prompt_ids"] = d["prompt_ids"][:3]
    d["metrics"] = a.PRIMARY
    frame = scores(d)
    source = tmp_path / "source"
    source.mkdir()
    assert a.output_report(frame, d, source, [], [], [], [])
    target = tmp_path / "replayed"
    assert a.replay(source, target)
    for name in ["means.csv", "paired.csv", "hypotheses.json", "contrasts_holm.csv"]:
        assert (source / name).read_bytes() == (target / name).read_bytes()
    (source / "canonical_per_image.csv").write_text("corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        a.replay(source, tmp_path / "tampered")


def test_qwen_fixed_settings_and_real_scheduler_validate_all_three_arms(sidecar):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from scripts import strengthen_qwen as q
    row, _, _ = sidecar
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, shift_terminal=.02,
                                               time_shift_type='exponential', max_shift=.9)
    scheduler.set_timesteps(40, sigmas=np.linspace(1., 1 / 40, 40), mu=.9)
    aliases = {'qwen_base': 'base_default', 'qwen_sega': 'sega_official', 'qwen_temp_only': 'temp_only_sbase'}
    for raw, canonical in aliases.items():
        config = q.fixed_config(raw, {'id': 'p8', 'prompt': 'caption p8'}, {'official_commit': q.OFFICIAL_COMMIT})
        metadata = {'complete': True, 'config': config, 'finite_steps': [True] * 40,
                    'sigmas': scheduler.sigmas.tolist(), 'timesteps': scheduler.timesteps.tolist()}
        path = Path(row['path']).with_suffix('.json')
        path.write_text(json.dumps(metadata))
        sample = {**row, 'raw_method': raw, 'method': canonical}
        assert a.inspect_sidecar(sample, 'caption p8', {'block': 'E3'}, {})['method'] == canonical
        metadata['config']['resolved_seed'] = 42
        path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match='resolved_seed'):
            a.inspect_sidecar(sample, 'caption p8', {'block': 'E3'}, {})
        metadata['config']['resolved_seed'] = 0
        metadata['sigmas'] = [1.] * 41
        path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match='schedule'):
            a.inspect_sidecar(sample, 'caption p8', {'block': 'E3'}, {})
