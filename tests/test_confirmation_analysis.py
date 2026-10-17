"""Pair completeness, selection boundaries, and fixed confirmation uncertainty."""
import csv
import json

import numpy as np
import pytest

from scripts import confirmation_analysis as analysis
from crope.utils import image_seed


def selection():
    means = {m: {k: 1. for k in analysis.METRICS}
             for m in ("sega_official", *analysis.VARIANTS)}
    means["sega_rms_uniform"]["imagereward"] = 2.
    means["sega_rms_uniform"]["musiq"] = 2.
    rows, selected = analysis.decision(means)
    return {"reference_root": "dev3_sega_official", "rows": rows, "selected": selected}


def test_selection_recomputes_gate_and_rejects_changed_selection():
    report = selection()
    assert analysis.selected_candidates(report) == ["sega_rms_uniform"]
    report["selected"] = ["sega_ctm_lag"]
    with pytest.raises(ValueError, match="Recorded selection"):
        analysis.selected_candidates(report)


def test_no_eligible_candidate_blocks_confirmation():
    report = selection()
    for row in report["rows"]:
        row.update({m: 1. for m in analysis.METRICS})
    report["selected"] = []
    with pytest.raises(ValueError, match="No eligible"):
        analysis.selected_candidates(report)


def test_paired_constant_differences_and_secondary_loss_are_retained():
    baseline = np.arange(64.)[:, None] + np.arange(5.)[None, :]
    values = {"sega_official": baseline,
              "sega_rms_uniform": baseline + [2., 1., 0., -3., 0.]}
    rows, conclusions = analysis.paired_statistics(values, ["sega_rms_uniform"])
    assert [r["delta"] for r in rows] == [2., 1., 0., -3., 0.]
    assert all(r["lo"] == r["hi"] == r["delta"] for r in rows)
    assert conclusions["sega_rms_uniform"]["both_primary_adjusted_lower_bounds_positive"]
    assert conclusions["sega_rms_uniform"]["secondary_negative_nominal_intervals"] == ["pickscore"]


def test_two_candidate_correction_widens_intervals_and_is_reproducible():
    baseline = np.zeros((64, 5))
    varying = np.linspace(-1, 1, 64)[:, None] + np.zeros((64, 5))
    values = {"sega_official": baseline, "sega_rms_uniform": varying,
              "sega_rms_normalized": varying}
    single, _ = analysis.paired_statistics(values, ["sega_rms_uniform"])
    double, conclusions = analysis.paired_statistics(values, list(values)[1:])
    assert double[0]["primary_adjusted_lo"] < single[0]["primary_adjusted_lo"]
    assert double[0]["primary_adjusted_hi"] > single[0]["primary_adjusted_hi"]
    assert not any(r["both_primary_adjusted_lower_bounds_positive"] for r in conclusions.values())
    assert double == analysis.paired_statistics(values, list(values)[1:])[0]


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(''.join(json.dumps({"id": str(i), "prompt": f"caption {i}"}) + '\n'
                               for i in range(64)))
    monkeypatch.setattr(analysis, 'PROMPT_SHA256', analysis.sha256(prompts))
    development = tmp_path / 'development'
    development.mkdir()
    (development / 'selection.json').write_text(json.dumps(selection()))
    provenance = {'transformer_revision': 'fixture-pinned'}
    (development / 'generation_provenance.json').write_text(json.dumps(provenance))
    eval_root = tmp_path / 'eval_krea'
    (eval_root / 'ir').mkdir(parents=True)
    dev_rows = []
    for method in ('sega_official', 'sega_rms_uniform'):
        name = f'confirm64_{method}'
        root = tmp_path / 'out_krea' / name
        directory = root / method / '4096x4096'
        directory.mkdir(parents=True)
        (root / 'generation_manifest.json').write_text(json.dumps(provenance))
        (root / 'generation_runs').mkdir()
        (root / 'generation_runs/fixture.json').write_text(json.dumps({
            'complete': True, 'method': method, 'gpu': 'fixture-gpu', 'cuda': 'fixture-cuda',
            'prompt_ids': [str(i) for i in range(64)]}))
        scores, rewards = [], []
        for i in range(64):
            png = directory / f'{i}.png'
            png.write_bytes(f'fixture image {i}'.encode())
            config = dict(method=method, prompt_id=str(i), prompt=f'caption {i}', seed=0,
                          derived_seed=image_seed(str(i), 0), steps=28, guidance=4.5,
                          res='4096x4096', grid=[256, 256], rope={'method': 'sega_official'},
                          eta=1., profile_sha256=None, temperature='none', model='flux',
                          model_revision='revision', dtype='bfloat16', batch_size=1,
                          torch='torch', diffusers='diffusers', implementation_sha256='source')
            sigmas = [1 - s / 28 for s in range(28)]
            record = dict(config=config, complete=True, image=str(png), sigmas=sigmas,
                          gains=[dict(sigma=s) for s in sigmas], png_sha256=analysis.sha256(png))
            png.with_suffix('.json').write_text(json.dumps(record))
            if i == 0:
                dev_rows.append({'variant': method, 'source_path': str(png)})
            common = dict(method=method, prompt_id=str(i), res='4096x4096')
            value = 1. if method == 'sega_official' else 2.
            scores.append({**common, 'seed': 0, 'path': str(png),
                           **{m: value for m in analysis.METRICS[1:]}})
            rewards.append({**common, 'png': str(png), 'imagereward': value})
        (eval_root / name).mkdir()
        analysis.write_csv(eval_root / name / 'per_image.csv', scores)
        analysis.write_csv(eval_root / 'ir' / f'{name}.csv', rewards)
    analysis.write_csv(development / 'dev_per_prompt.csv', dev_rows)
    return eval_root, prompts, development, tmp_path / 'report'


def test_complete_confirmation_reports_all_pairs_and_frozen_uncertainty(campaign):
    report = analysis.analyze(*campaign)
    assert report['n'] == 64 and report['primary_family_size'] == 2
    assert report['primary_adjusted_coverage'] == .975
    assert report['confirmation_does_not_select_further_variants']
    with (campaign[3] / 'confirmation_per_prompt.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 128


@pytest.mark.parametrize('problem', ['manifest', 'missing_pair', 'controller', 'derived_seed', 'image_hash', 'hardware'])
def test_confirmation_rejects_protocol_drift_before_report(campaign, problem):
    eval_root, prompts, development, out = campaign
    if problem == 'manifest':
        prompts.write_text(prompts.read_text() + '\n')
    elif problem == 'missing_pair':
        table = eval_root / 'confirm64_sega_rms_uniform/per_image.csv'
        table.write_text('\n'.join(table.read_text().splitlines()[:-1]) + '\n')
    elif problem == 'hardware':
        receipt = eval_root.parent / 'out_krea/confirm64_sega_rms_uniform/generation_runs/fixture.json'
        record = json.loads(receipt.read_text())
        record['gpu'] = 'different-gpu'
        record['complete'] = False
        receipt.with_name('interrupted-on-other-gpu.json').write_text(json.dumps(record))
    else:
        sidecar = eval_root.parent / 'out_krea/confirm64_sega_rms_uniform/sega_rms_uniform/4096x4096/1.json'
        record = json.loads(sidecar.read_text())
        if problem == 'controller':
            record['config']['eta'] = .5
        elif problem == 'derived_seed':
            record['config']['derived_seed'] += 1
        else:
            record['png_sha256'] = 'wrong'
        sidecar.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        analysis.analyze(*campaign)
    assert not out.exists()
