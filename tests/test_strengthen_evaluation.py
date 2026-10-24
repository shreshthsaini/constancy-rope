"""Guard evaluation publication, frozen dependencies and worker-move races."""
import json
import hashlib
from pathlib import Path

import pytest

from scripts import strengthen_evaluation as e
from crope.utils import image_stem


@pytest.fixture
def campaign(tmp_path):
    prompts = tmp_path / 'prompts.jsonl'
    prompts.write_text(''.join(json.dumps({'id': str(i), 'prompt': f'caption{i}'}) + '\n' for i in range(3)))
    stage = tmp_path / 'generation.sh'
    stage.write_text('#!/bin/bash\ntrue\n')
    source = tmp_path / 'generation_source.py'
    source.write_text('print("generation")\n')
    queue = tmp_path / 'queue'
    for name in ['done', 'pending', 'running', 'failed']:
        (queue / name).mkdir(parents=True)
    out = tmp_path / 'images/sega_base_none/4096x4096'
    out.mkdir(parents=True)
    task = dict(name='75_fixture.sh', method='sega_base_none', seed=0, start=0, end=3,
                n=3, prompt_ids=['0', '1', '2'], out=str(out), staged=str(stage),
                sha256=e.file_sha256(stage))
    manifest = dict(block='factorial', prompts=str(prompts), prompts_sha256=e.file_sha256(prompts),
                    tasks=[task], n_tasks=1, n_generate=3, sources={str(source): e.file_sha256(source)})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    references = tmp_path / 'refs.jsonl'
    reference = tmp_path / 'reference.png'
    reference.write_bytes(b'fixture reference')
    references.write_text(''.join(json.dumps({'id': str(i), 'path': str(reference)}) + '\n' for i in range(3)))
    return manifest, path, references, queue, tmp_path / 'notes'


def finish(campaign):
    manifest, _, _, queue, _ = campaign
    for sample in e.expected_samples(manifest):
        png = Path(sample['path'])
        png.write_bytes(b'fixture-png')
        png.with_suffix('.json').write_text('{}')
    task = manifest['tasks'][0]
    # Match production publication and worker ownership transitions exactly.
    Path(task['staged']).rename(queue / 'pending' / task['name'])
    (queue / 'pending' / task['name']).rename(queue / 'running' / (task['name'] + '.host'))
    (queue / 'running' / (task['name'] + '.host')).rename(queue / 'done' / task['name'])


def test_missing_generation_cannot_stage_or_publish(campaign):
    manifest, path, refs, queue, notes = campaign
    status = e.prepare(path, refs, True, queue, notes)
    assert not status['complete'] and len(status['missing_image_pairs']) == 3
    assert status['unfinished_tasks'] == [manifest['tasks'][0]['name']]
    assert not list((queue / 'pending').glob('*'))
    assert not (queue / 'staging').exists()
    assert json.loads((notes / 'factorial/status.json').read_text())['validation'] == 'waiting_for_generation'


def test_expected_samples_rejects_duplicate_and_changed_manifest(campaign):
    manifest, *_ = campaign
    samples = e.expected_samples(manifest)
    assert len(samples) == 3 and samples[0]['path'].endswith(image_stem('0', 0) + '.png')
    manifest['tasks'].append(manifest['tasks'][0])
    with pytest.raises(ValueError, match='Duplicate'):
        e.expected_samples(manifest)
    manifest['tasks'].pop()
    manifest['tasks'][0]['prompt_ids'] = ['2', '1', '0']
    with pytest.raises(ValueError, match='interval'):
        e.expected_samples(manifest)


def test_finished_task_body_must_match_frozen_generation(campaign):
    finish(campaign)
    manifest, _, _, queue, _ = campaign
    assert e.completion_status(manifest, e.expected_samples(manifest), queue)['complete']
    (queue / 'done' / manifest['tasks'][0]['name']).write_text('changed')
    with pytest.raises(ValueError, match='hash differs'):
        e.completion_status(manifest, e.expected_samples(manifest), queue)


def test_complete_block_stages_then_publishes_once(campaign, monkeypatch):
    finish(campaign)
    manifest, path, refs, queue, notes = campaign
    assert not Path(manifest['tasks'][0]['staged']).exists()
    monkeypatch.setattr(e, 'verify_samples', lambda m, s: [{'image': row['path']} for row in s])
    stage = e.prepare(path, refs, False, queue, notes)
    assert stage['validation'] == 'passed' and len(stage['tasks']) == 2
    assert not list((queue / 'pending').glob('*'))
    first = e.prepare(path, refs, True, queue, notes)
    assert len(list((queue / 'pending').glob('90_*.sh'))) == 2
    second = e.prepare(path, refs, True, queue, notes)
    assert first['dependencies_sha256'] == second['dependencies_sha256']
    assert all(task['status'] == 'pending' for task in second['tasks'])
    checksum = notes / 'factorial/dependencies.sha256'
    assert str(path.resolve()) in checksum.read_text()
    assert str(refs.resolve()) in checksum.read_text()


def test_failed_image_validation_records_failure_without_publication(campaign, monkeypatch):
    finish(campaign)
    _, path, refs, queue, notes = campaign
    def fail(*_):
        raise ValueError('PNG checksum differs')
    monkeypatch.setattr(e, 'verify_samples', fail)
    with pytest.raises(ValueError, match='checksum'):
        e.prepare(path, refs, True, queue, notes)
    receipt = json.loads((notes / 'factorial/status.json').read_text())
    assert receipt['validation'] == 'failed' and not receipt['complete']
    assert not list((queue / 'pending').glob('*'))


def test_generation_source_drift_stops_evaluation(campaign, monkeypatch):
    finish(campaign)
    manifest, path, refs, queue, notes = campaign
    monkeypatch.setattr(e, 'verify_samples', lambda *_: [])
    Path(next(iter(manifest['sources']))).write_text('changed source')
    with pytest.raises(ValueError, match='source changed'):
        e.prepare(path, refs, True, queue, notes)
    assert not list((queue / 'pending').glob('*'))


def test_reservation_prevents_duplicate_after_worker_move_race(tmp_path):
    queue, stage = tmp_path / 'queue', tmp_path / 'stage'
    scripts = {'90_eval_a.sh': '#!/bin/bash\ntrue\n'}
    first = e.publish_scripts(scripts, stage, queue, True)
    assert first[0]['status'] == 'published'
    # Simulate observing no queue location during a worker's atomic move.
    (queue / 'pending/90_eval_a.sh').rename(tmp_path / 'worker-owned.sh')
    second = e.publish_scripts(scripts, stage, queue, True)
    assert second[0]['status'] == 'already_reserved_check_receipt'
    assert not (queue / 'pending/90_eval_a.sh').exists()


def test_failed_or_conflicting_task_requires_inspection(tmp_path):
    queue, stage = tmp_path / 'queue', tmp_path / 'stage'
    scripts = {'90_ir_a.sh': '#!/bin/bash\ntrue\n'}
    (queue / 'failed').mkdir(parents=True)
    (queue / 'failed/90_ir_a.sh').write_text(scripts['90_ir_a.sh'])
    with pytest.raises(ValueError, match='failed'):
        e.publish_scripts(scripts, stage, queue, True)
    (queue / 'failed/90_ir_a.sh').write_text('different')
    with pytest.raises(ValueError, match='differs'):
        e.publish_scripts(scripts, stage, queue, True)


def test_seed_evaluation_separates_roots_and_preserves_metric_settings(tmp_path):
    samples = [dict(seed=seed, root=f'/scratch/images/seed{seed}', method=method)
               for seed in [1, 2] for method in ['sega_official', 'temp_only_sbase']]
    scripts = e.task_bodies({'block': 'seeds', 'prompts': '/some/prompts.jsonl'}, samples,
                            '/some/refs.jsonl', tmp_path / 'deps.sha256', tmp_path / 'eval')
    assert len(scripts) == 4
    for name, body in scripts.items():
        assert name.startswith('90_') and '--device cuda' in body and 'OMP_NUM_THREADS=8' in body
        assert 'sha256sum --status -c' in body
        if name.startswith('90_eval'):
            assert e.METRICS in body and '--allow-missing' in body
        else:
            assert 'mend/bin/python' in body and '/ir.csv' in body


@pytest.mark.parametrize('block,counts,expected', [
    ('factorial', [187, 187], 3), ('fresh', [150] * 6, 7),
    ('qwen', [150] * 3, 4), ('tsweep', [55] * 3, 2),
    ('ckpt_matrix', [8] * 4, 2), ('default_shift', [8], 2),
])
def test_large_blocks_use_method_subroots(tmp_path, block, counts, expected):
    samples = [dict(seed=0, root='/images', method=f'arm{arm}')
               for arm, count in enumerate(counts) for _ in range(count)]
    scripts = e.task_bodies({'block': block, 'prompts': '/prompts.jsonl'}, samples,
                            '/refs.jsonl', tmp_path / 'deps', tmp_path / 'eval')
    assert len(scripts) == expected
    evaluator = [body for name, body in scripts.items() if name.startswith('90_eval')]
    assert len(evaluator) == (len(counts) if sum(counts) > 200 else 1)
    if sum(counts) > 200:
        assert all('--root /images/arm' in body and '/shards/arm' in body for body in evaluator)
    assert len([name for name in scripts if name.startswith('90_ir')]) == 1


def test_large_seeds_have_four_evaluators_and_two_rewards(tmp_path):
    samples = [dict(seed=seed, root=f'/images/seed{seed}', method=method)
               for seed, count in [(1, 132), (2, 187)]
               for method in ['sega_official', 'temp_only_sbase'] for _ in range(count)]
    scripts = e.task_bodies({'block': 'seeds', 'prompts': '/prompts'}, samples,
                            '/refs', tmp_path / 'deps', tmp_path / 'eval')
    assert len(scripts) == 6
    assert len([name for name in scripts if name.startswith('90_eval')]) == 4
    assert '--root /images/seed1/sega_official' in scripts['90_eval_seeds_seed1_sega_official.sh']
    assert '--root /images/seed2/temp_only_sbase' in scripts['90_eval_seeds_seed2_temp_only_sbase.sh']


def test_real_discovery_by_method_matches_full_root(tmp_path):
    from PIL import Image
    from crope.metrics.io import discover_samples
    prompts = [{'id': 'b', 'prompt': 'B'}, {'id': 'a', 'prompt': 'A'}]
    for method in ['one', 'two']:
        out = tmp_path / method / '8x8'
        out.mkdir(parents=True)
        for prompt in prompts:
            png = out / (image_stem(prompt['id'], 0) + '.png')
            Image.new('RGB', (8, 8)).save(png)
            png.with_suffix('.json').write_text(json.dumps({'complete': True, 'seconds': 1., 'config':
                dict(method=method, prompt_id=prompt['id'], prompt=prompt['prompt'], seed=0, res='8x8')}))
    full = discover_samples(tmp_path, prompts, '8x8')
    shards = [row for method in ['one', 'two'] for row in discover_samples(tmp_path / method, prompts, '8x8')]
    assert full == shards


@pytest.mark.parametrize('metric', ['fid', 'kid'])
def test_distribution_bootstrap_endpoints_match_full_root_and_method_shards(metric):
    import numpy as np
    from crope.metrics.evaluate import cluster_features
    from crope.metrics import statistics
    rows = [dict(method=method, res='4096x4096', prompt_id=str(prompt), seed=seed,
                 path=f'/{method}/{prompt}_{seed}.png')
            for method in ['one', 'two'] for prompt in [2, 0, 1] for seed in [0, 1]]
    rng = np.random.default_rng(31)
    bank = {row['path']: rng.normal(size=(4, 2)) for row in rows}  # Four crop features per image.
    real = {str(prompt): rng.normal(size=(4, 2)) for prompt in [3, 2, 0, 1]}
    statistic = getattr(statistics, metric)
    full = {}
    for method in ['one', 'two']:
        features = cluster_features(rows, bank, method=method, res='4096x4096')
        full[method] = statistics.distribution_bootstrap(features, real, statistic, draws=1000, seed=0)
    # Reverse task completion order. Each call still starts its own seeded RNG.
    for method in ['two', 'one']:
        subset = [row for row in rows if row['method'] == method]
        features = cluster_features(subset, bank, method=method, res='4096x4096')
        sharded = statistics.distribution_bootstrap(features, real, statistic, draws=1000, seed=0)
        assert sharded == full[method]
        assert sharded['n_reference_prompts'] == 4


@pytest.fixture
def shard_campaign(campaign, monkeypatch, tmp_path):
    import copy
    manifest, path, refs, queue, notes = campaign
    finish(campaign)
    task = copy.deepcopy(manifest['tasks'][0])
    task.update(name='75_fixture_other.sh', method='ntk2_sref',
                out=str(tmp_path / 'images/ntk2_sref/4096x4096'))
    Path(task['out']).mkdir(parents=True)
    (queue / 'done' / task['name']).write_bytes((queue / 'done' / manifest['tasks'][0]['name']).read_bytes())
    manifest['tasks'].append(task)
    manifest.update(n_generate=6, n_tasks=2)
    path.write_text(json.dumps(manifest))
    for sample in e.expected_samples(manifest):
        Path(sample['path']).write_bytes(b'fixture-png')
        Path(sample['path']).with_suffix('.json').write_text('{}')
    monkeypatch.setattr(e, 'SHARD_THRESHOLD', 3)
    monkeypatch.setattr(e, 'verify_samples', lambda m, s: [{'image': row['path']} for row in s])
    status = e.prepare(path, refs, True, queue, notes, tmp_path / 'eval')
    plan_path = notes / 'factorial/evaluation_plan.json'
    plan = json.loads(plan_path.read_text())
    assert len(status['tasks']) == 3
    return path, queue, notes, plan_path, plan


def fake_evaluator(plan, shard):
    """Match the real evaluator's reports without loading score models."""
    from crope.metrics.statistics import scalar_summaries
    out = Path(shard['out'])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sample in enumerate(reversed(shard['samples'])):
        metrics = {name: .01 * (i + j + 1) for j, name in enumerate(e.ANALYSIS_METRICS) if name != 'imagereward'}
        rows.append({**sample, 'metrics': {**metrics, 'seconds': 2. + i}, 'eval_seconds': {'clip': .2}})
    summaries = scalar_summaries(rows)
    for i, metric in enumerate(['fid', 'kid', 'patch_fid', 'patch_kid']):
        summaries.append(dict(method=shard['methods'][0], res='4096x4096', metric=metric,
            value=1.234567891 + i, ci_low=.1 + i, ci_high=2.5 + i,
            n_prompts=3, bootstrap_valid=1000, n_reference_prompts=3,
            n_feature_rows=3, n_reference_feature_rows=3))
    report = dict(config=dict(root=shard['root'], out=shard['out'], prompts=plan['prompts'],
            references=plan['references'], reference_method=shard['reference_method'], metrics=e.METRICS,
            res='4096x4096', device='cuda', threads=8, bootstrap=1000, bootstrap_seed=0, crop_seed=0, crops=4),
        complete=True, all_metrics_available=True, errors=[], optional_unavailable=[],
        metric_schema=1, versions={'fixture': '1'}, model_revisions={'fixture': 'frozen'},
        status={metric: {'status': 'complete'} for metric in e.METRICS.split(',')},
        per_image=rows, summary=summaries, paired_deltas=[], seconds=3., uncertainty='fixture interval')
    save_evaluator(out, report)
    return report


def save_evaluator(out, report):
    e.write_json(out / 'metrics.json', report)
    (out / 'per_image.csv').write_text(e.csv_text([e.flat_row(row) for row in report['per_image']]))
    (out / 'summary.csv').write_text(e.csv_text(report['summary']))
    (out / 'paired_deltas.csv').write_text(e.csv_text(report['paired_deltas']))


def finish_evaluators(shard_campaign):
    path, queue, notes, plan_path, plan = shard_campaign
    reports = []
    for shard in plan['groups']['factorial']['shards']:
        reports.append(fake_evaluator(plan, shard))
        e.seal_output(plan_path, shard['name'], e.file_sha256(plan_path))
        # Publication creates a hard link; the worker then moves its queue copy.
        (queue / 'pending' / shard['name']).rename(queue / 'running' / (shard['name'] + '.node'))
        (queue / 'running' / (shard['name'] + '.node')).rename(queue / 'done' / shard['name'])
    return reports


def test_missing_shard_never_writes_canonical_scores(shard_campaign):
    path, queue, notes, _, plan = shard_campaign
    result = e.merge_evaluation(path, queue, notes)
    assert not result['complete']
    assert len(result['groups']['factorial']['unfinished_tasks']) == 2
    assert not (Path(plan['groups']['factorial']['out']) / 'per_image.csv').exists()


def test_merge_exact_rows_fid_intervals_and_idempotent_receipts(shard_campaign):
    import csv
    path, queue, notes, _, plan = shard_campaign
    reports = finish_evaluators(shard_campaign)
    first = e.merge_evaluation(path, queue, notes)
    assert first['complete']
    out = Path(plan['groups']['factorial']['out'])
    merged = json.loads((out / 'metrics.json').read_text())
    expected_rows = sorted([row for report in reports for row in report['per_image']],
                           key=lambda row: (row['method'], row['prompt_id']))
    assert merged['per_image'] == expected_rows
    assert list(csv.DictReader((out / 'per_image.csv').open())) == list(csv.DictReader(e.io.StringIO(
        e.csv_text([e.flat_row(row) for row in expected_rows]))))
    assert merged['summary'] == sorted([row for report in reports for row in report['summary']],
                                      key=lambda row: (row['method'], row['res'], row['metric']))
    assert merged['paired_deltas'] == [] and (out / 'paired_deltas.csv').read_bytes() == b''
    assert not (out / 'ir.csv').exists()  # IR is independent and never fabricated by this merge.
    before = {name: e.file_sha256(out / name) for name in e.OUTPUT_FILES}
    assert e.merge_evaluation(path, queue, notes)['complete']
    assert before == {name: e.file_sha256(out / name) for name in e.OUTPUT_FILES}


@pytest.mark.parametrize('corruption', ['duplicate', 'missing', 'wrong_path', 'nonfinite', 'summary', 'setting'])
def test_invalid_shard_cannot_be_sealed(shard_campaign, corruption):
    _, _, _, plan_path, plan = shard_campaign
    shard = plan['groups']['factorial']['shards'][0]
    report = fake_evaluator(plan, shard)
    if corruption == 'duplicate':
        report['per_image'].append(report['per_image'][0])
    elif corruption == 'missing':
        report['per_image'].pop()
    elif corruption == 'wrong_path':
        report['per_image'][0]['path'] = '/wrong.png'
    elif corruption == 'nonfinite':
        report['per_image'][0]['metrics']['musiq'] = None
    elif corruption == 'summary':
        report['summary'].append(report['summary'][0])
    else:
        report['config']['crop_seed'] = 1
    save_evaluator(Path(shard['out']), report)
    with pytest.raises(ValueError):
        e.seal_output(plan_path, shard['name'], e.file_sha256(plan_path))
    assert not (Path(shard['out']) / 'evaluation_receipt.json').exists()


@pytest.mark.parametrize('corruption', ['csv', 'task', 'source', 'missing_file', 'plan'])
def test_merge_rejects_tampered_or_missing_artifact(shard_campaign, corruption):
    path, queue, notes, plan_path, plan = shard_campaign
    finish_evaluators(shard_campaign)
    shard = plan['groups']['factorial']['shards'][0]
    if corruption == 'csv':
        with (Path(shard['out']) / 'per_image.csv').open('a') as handle:
            handle.write('corrupt\n')
    elif corruption == 'task':
        (queue / 'done' / shard['name']).write_text('changed evaluator')
    elif corruption == 'source':
        manifest = json.loads(path.read_text())
        Path(next(iter(manifest['sources']))).write_text('changed generation source')
    elif corruption == 'missing_file':
        (Path(shard['out']) / 'summary.csv').rename(Path(shard['out']) / 'summary.csv.aside')
    else:
        plan['groups']['factorial']['samples'].pop()
        e.write_json(plan_path, plan)
    with pytest.raises((ValueError, FileNotFoundError)):
        e.merge_evaluation(path, queue, notes)
    assert not (Path(plan['groups']['factorial']['out']) / 'per_image.csv').exists()


@pytest.fixture
def diagnostic_sample(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from scripts import strengthen_diagnostics as d
    from crope.utils import image_seed
    prompts = tmp_path / 'prompts.jsonl'
    prompts.write_text('{"id":"p0","prompt":"A tree"}\n')
    def create(block):
        official = e.SCRATCH / 'third_party/DyPE'
        source = {str(official / name): name for name in ['flux/pipeline_flux.py', 'flux/transformer_flux.py']}
        source[str(e.ROOT / 'scripts/strengthen_diagnostics.py')] = 'fixture-wrapper'
        manifest = dict(block=block, prompts=str(prompts), sources=source)
        if block == 'ckpt_matrix':
            provenance = {'official_root': str(official), 'official_commit': d.OFFICIAL_COMMIT,
                          'official_python_sha256': {name: name for name in ['flux/pipeline_flux.py', 'flux/transformer_flux.py']},
                          'wrapper_sha256': 'fixture-wrapper'}
            versions = {name: e.importlib.metadata.version(name) for name in ['torch', 'diffusers', 'transformers']}
            config = d.build_config(SimpleNamespace(checkpoint='dev', guidance=3.5, seed=42, steps=28, prompts=prompts),
                                    {'id': 'p0', 'prompt': 'A tree'}, provenance, versions)
            mu = 1.15
        else:
            config = dict(method='sega_default_shift', controller_method='sega_official', prompt_id='p0',
                          prompt='A tree', seed=0, steps=28, guidance=4.5, res='4096x4096', dtype='bfloat16',
                          derived_seed=image_seed('p0', 0), model='black-forest-labs/FLUX.1-dev',
                          model_revision=d.DEV_REVISION, rope={'method': 'sega_official'}, eta=1., temperature='logN',
                          schedule_mode='diffusers_default', original_max_image_seq_len=4096,
                          scheduler={'max_image_seq_len': 4096, 'base_image_seq_len': 256,
                                     'base_shift': .5, 'max_shift': 1.15})
            mu = 11.55
        initial = np.linspace(1., 1 / 28, 28)
        sigma = np.exp(mu) / (np.exp(mu) + 1 / initial - 1)
        record = {'complete': True, 'config': config, 'sigmas': (np.r_[sigma, 0.] if block == 'ckpt_matrix' else sigma).tolist(),
                  'timesteps': (sigma * 1000).tolist()}
        sample = {'method': config['method'], 'prompt_id': 'p0', 'prompt': 'A tree', 'seed': config['seed'],
                  'path': str(tmp_path / f'{block}.png')}
        def save():
            record['config_fingerprint'] = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            Path(sample['path']).with_suffix('.json').write_text(json.dumps(record))
        save()
        return sample, manifest, record, save
    return create


@pytest.mark.parametrize('block,mu', [('ckpt_matrix', 1.15), ('default_shift', 11.55)])
def test_diagnostic_gate_preserves_distinct_schedules_and_roles(diagnostic_sample, block, mu):
    sample, manifest, _, _ = diagnostic_sample(block)
    report = e.inspect_diagnostic(sample, manifest)
    assert report['verified_mu'] == mu
    assert report['role'] == 'development_diagnostic_no_confirmatory_hypothesis'
    assert report['config']['guidance'] == (3.5 if block == 'ckpt_matrix' else 4.5)


@pytest.mark.parametrize('block', ['ckpt_matrix', 'default_shift'])
@pytest.mark.parametrize('corruption', ['guidance', 'schedule'])
def test_diagnostic_gate_rejects_guidance_or_schedule_corruption(diagnostic_sample, block, corruption):
    sample, manifest, record, save = diagnostic_sample(block)
    if corruption == 'guidance':
        record['config']['guidance'] += 1.
    else:
        record['sigmas'][2] = .1
    save()
    with pytest.raises(ValueError, match='guidance|schedule'):
        e.inspect_diagnostic(sample, manifest)
