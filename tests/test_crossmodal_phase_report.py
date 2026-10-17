"""CPU tests for report binding, complete-task gating and pixel-exact evidence views."""
import copy
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageDraw
import pytest
import torch

from crope.crossmodal_phase import NUMERICAL_POLICY
from crope.utils import file_sha256
from scripts import crossmodal_phase_report as report


def json_file(path, value):
    report.write(path, value)
    return path


def task_summary(index):
    records, scores = [], []
    for j, arm in enumerate(report.ARMS):
        records.append({'config': {'method': arm}, 'seconds': 10+index+2*j,
                        'peak_memory_bytes': 100+j, 'seconds_scope': 'Actual worker scope'})
        scores.append({'method': arm, 'metrics': {m: 10*index+j+k for k, m in enumerate(report.METRICS[1:])},
                       'imagereward': 10*index+j-.5})
    return {'index': index, 'prompt': {'id': f'prompt{index}', 'prompt': f'Caption {index}'},
            'records': records, 'scores': scores, 'manifest': {'model_loading_seconds': 3+index}}


def test_all_metrics_and_paired_means_without_selection():
    output = report.summarize([task_summary(0), task_summary(1)])
    assert len(output['per_image']) == 6 and len(output['paired_contrasts']) == 4
    assert output['descriptive_means']['baseline']['imagereward'] == 4.5
    for arm, delta in [('fixedmass', 1), ('unnormalized', 2)]:
        assert output['descriptive_mean_contrasts'][arm] == dict.fromkeys(report.METRICS, delta)
    assert output['descriptive_runtime_means']['baseline']['seconds'] == 10.5
    assert output['runtime'][1]['seconds_ratio_to_baseline'] == 1.2
    assert output['runtime'][1]['timing_scope'] == 'Actual worker scope'
    assert not {'confidence_interval', 'winner', 'aggregate_score'} & output.keys()


@pytest.mark.parametrize('damage', ['one_prompt', 'reordered', 'duplicate_arm', 'nan', 'negative_time', 'boolean_memory'])
def test_invalid_summary_inputs_fail(damage):
    tasks = [task_summary(0), task_summary(1)]
    if damage == 'one_prompt': tasks.pop()
    elif damage == 'reordered': tasks.reverse()
    elif damage == 'duplicate_arm': tasks[0]['scores'][1]['method'] = 'baseline'
    elif damage == 'nan': tasks[0]['scores'][0]['imagereward'] = float('nan')
    elif damage == 'negative_time': tasks[0]['records'][0]['seconds'] = -1
    else: tasks[0]['records'][0]['peak_memory_bytes'] = True
    with pytest.raises(ValueError): report.summarize(tasks)


def make_record(task, arm, smoke=False):
    folder = task/('smoke' if smoke else 'images')/arm
    folder.mkdir(parents=True)
    initial = folder/'initial.pt'; endpoint = folder/'endpoint.pt'
    value = torch.ones(2, 2, dtype=torch.bfloat16)
    torch.save({'initial_transformer_input': value}, initial)
    torch.save({'endpoint_latent': value}, endpoint)
    record = {'config': {'method': arm}, 'seconds': 1., 'peak_memory_bytes': 12,
              'seconds_scope': 'Encoding and generation',
              'initial_transformer_input': {'path': str(initial), 'file_sha256': file_sha256(initial)},
              'endpoint_latent': {'path': str(endpoint), 'file_sha256': file_sha256(endpoint)}}
    if not smoke:
        png = folder/'image.png'
        Image.new('RGB', (16, 16), 'red').save(png)
        record.update(image=str(png), png_sha256=file_sha256(png))
    sidecar = folder/'image.json'
    json_file(sidecar, record)
    record.update(metadata=str(sidecar), metadata_sha256=file_sha256(sidecar))
    return record


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Frozen worker validators already have real geometry/tensor corruption tests.

    Mock only their expensive ancestry/model-cache checks here, leaving the new
    report's manifest, file, endpoint, numerical gate and score bindings live.
    """
    task = tmp_path/'job123'/'prompt0'
    task.mkdir(parents=True)
    records = [make_record(task, arm) for arm in report.ARMS]
    smoke_records = [make_record(task, arm, True) for arm in ('deployed', *report.ARMS)]
    prompt = {'id': 'fixed0', 'prompt': 'The fixed caption.'}
    accepted = dict(model={'revision': 'fixed'}, torch='fixedtorch', cuda='fixedcuda', diffusers='fixeddiffusers')
    protocol = dict(source_sha256={'fixed.py': 'a'*64}, deployed_smoke_max_relative_l2=.001)
    manifest = dict(protocol_sha256=report.PROTOCOL_SHA256, prompt_index=0, prompt=prompt,
        source_sha256=protocol['source_sha256'], **accepted, slurm_job_id='123', scope=report.worker.FIXED['scope'],
        quality_scored=False, seed_device='cpu', seed_policy='crope.utils.image_seed(prompt_id,0)',
        hardware='GH200 recorded', model_loading_seconds=3.)
    json_file(task/'manifest.json', manifest)
    json_file(task/'completion.json', {'records': records})
    json_file(task/'progress.json', {'status': 'completed', 'completed_images': 3})
    checks = {name: {'relative_l2': 0., 'max_abs': 0.} for name in
              ('private_vs_fp64', 'private_vs_public_cudnn', 'fixedmass_vs_fp64', 'unnormalized_vs_fp64')}
    checks.update(lse_max_abs=0., fixedmass_identity_byte_equal=True, unnormalized_identity_byte_equal=True,
                  production_text_branch={'finite': True, 'query_shape': [1, 24, 65536, 128],
                                          'key_shape': [1, 24, 512, 128]})
    json_file(task/'kernel_smoke.json', dict(passed=True, gpu_runtime_verified=True,
              backend=report.worker.FIXED['backend'], tolerances=NUMERICAL_POLICY, checks=checks))
    endpoint = torch.ones(2, 2, dtype=torch.bfloat16)
    comparison = report.worker.endpoint_comparison(endpoint, endpoint)
    json_file(task/'native_smoke.json', dict(records=smoke_records, no_vae_decode=True,
              cosine_threshold_is_diagnostic=True,
              comparisons_to_baseline={a: comparison for a in ('deployed', *report.ARMS)}))
    score = {'complete': True, 'records': []}
    json_file(task/'workflow_scored.json', score)
    json_file(task/'evaluation/metrics.json', {})
    (task/'evaluation/imagereward.csv').write_text('header\n')
    monkeypatch.setattr(report.scorer, 'validate_scores', lambda *args: score)
    monkeypatch.setattr(report.worker, 'validate_dependencies', lambda *args: (prompt, accepted))
    monkeypatch.setattr(report.worker, 'validate_arm_set', lambda *args, **kwargs: None)
    def check_png(path, size, digest):
        report.require(file_sha256(path) == digest, 'PNG changed')
    monkeypatch.setattr(report.worker.native, 'verify_png', check_png)
    return task, protocol


def test_complete_task_bindings_and_all_witnesses(bundle):
    task, protocol = bundle
    result = report.validate_task(task, 0, '123', protocol)
    assert len(result['files']) == 8+3*4+4*3
    assert result['prompt']['prompt'] == 'The fixed caption.'


@pytest.mark.parametrize('damage', [None, 'one_ulp', 'material_numeric', 'arm', 'threshold', 'sigma', 'source', 'protocol'])
def test_native_smoke_integration_uses_actual_frozen_validator(tmp_path, damage):
    # Reuse the frozen worker's accepted CPU fixture, without mocking its validator.
    import importlib.util
    fixture_path = Path(__file__).with_name('test_crossmodal_phase_pilot.py')
    spec = importlib.util.spec_from_file_location('frozen_pilot_fixture', fixture_path)
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    protocol = fixture_module.protocol.__wrapped__()
    records, prompt = fixture_module.records.__wrapped__(protocol, tmp_path)
    from crope.runtime import fingerprint
    for record in records:
        record['config']['phase_protocol_sha256'] = report.PROTOCOL_SHA256
        record['config_fingerprint'] = fingerprint(record['config'])
    zero = torch.zeros(4096, 64, dtype=torch.bfloat16)
    comparison = report.worker.endpoint_comparison(zero, zero)
    smoke = dict(records=records, no_vae_decode=True, cosine_threshold_is_diagnostic=True,
                 comparisons_to_baseline={a: dict(comparison) for a in ('deployed', *report.ARMS)})
    if damage == 'one_ulp': smoke['comparisons_to_baseline']['deployed']['cosine'] = math.nextafter(1., 0.)
    elif damage == 'material_numeric': smoke['comparisons_to_baseline']['deployed']['cosine'] = .9
    elif damage == 'arm': records[2]['phase']['mode'] = 'baseline'
    elif damage == 'threshold': protocol['deployed_smoke_max_relative_l2'] = 1.
    elif damage == 'sigma': records[1]['sigmas'][1] = .1
    elif damage == 'source': records[1]['config']['phase_source_sha256'] = 'b'*64
    elif damage == 'protocol': records[1]['config']['phase_protocol_sha256'] = 'e'*64
    if damage in ('source', 'protocol'):
        records[1]['config_fingerprint'] = fingerprint(records[1]['config'])
    if damage not in (None, 'one_ulp'):
        with pytest.raises(ValueError): report.validate_native_smoke(smoke, prompt, protocol)
    else:
        result = report.validate_native_smoke(smoke, prompt, protocol)
        assert result['comparisons']['fixedmass']['byte_exact'] is True
        assert result['gate_rechecked_from_actual_tensors'] is True
        if damage == 'one_ulp': assert result['receipt_residuals']['deployed']['cosine'] > 0


@pytest.mark.parametrize('damage', ['score_receipt', 'job', 'model', 'source', 'progress', 'failure',
    'sidecar_hash', 'sidecar_payload', 'tensor', 'png', 'escaped_tensor', 'smoke_endpoint',
    'smoke_claim', 'smoke_gate', 'kernel_error', 'kernel_nan', 'kernel_shape'])
def test_task_corruption_rejected(bundle, damage, tmp_path):
    task, protocol = bundle
    def change(name, transform):
        value = report.read(task/name); transform(value)
        (task/name).write_text(json.dumps(value))  # Deliberately permit corrupt NaN test input.
    if damage == 'score_receipt': change('workflow_scored.json', lambda x: x.update(complete=False))
    elif damage in ('job', 'model', 'source'):
        key = {'job': 'slurm_job_id', 'model': 'model', 'source': 'source_sha256'}[damage]
        change('manifest.json', lambda x: x.update({key: 'foreign'}))
    elif damage == 'progress': change('progress.json', lambda x: x.update(status='generating_4k'))
    elif damage == 'failure': json_file(task/'failure.json', {'error': 'failed'})
    elif damage.startswith('sidecar'):
        completion = report.read(task/'completion.json'); record = completion['records'][0]
        sidecar = Path(record['metadata']); content = report.read(sidecar); content['seconds'] = 999
        json_file(sidecar, content)
        if damage == 'sidecar_payload':
            record['metadata_sha256'] = file_sha256(sidecar); json_file(task/'completion.json', completion)
    elif damage in ('tensor', 'png'):
        record = report.read(task/'completion.json')['records'][0]
        path = record['image'] if damage == 'png' else record['endpoint_latent']['path']
        Path(path).write_bytes(b'corrupt')
    elif damage == 'escaped_tensor':
        completion = report.read(task/'completion.json'); record = completion['records'][0]
        other = tmp_path/'outside.pt'; other.write_bytes(b'outside')
        record['endpoint_latent'] = {'path': str(other), 'file_sha256': file_sha256(other)}
        payload = {k: v for k, v in record.items() if k not in ('metadata', 'metadata_sha256')}
        json_file(record['metadata'], payload); record['metadata_sha256'] = file_sha256(record['metadata'])
        json_file(task/'completion.json', completion)
    elif damage == 'smoke_endpoint':
        item = report.read(task/'native_smoke.json')['records'][2]['endpoint_latent']
        torch.save({'endpoint_latent': torch.zeros(2, 2, dtype=torch.bfloat16)}, item['path'])
    elif damage == 'smoke_claim':
        change('native_smoke.json', lambda x: x['comparisons_to_baseline']['fixedmass'].update(max_abs=1))
    elif damage == 'smoke_gate':
        value = report.read(task/'native_smoke.json')
        path = value['records'][0]['endpoint_latent']['path']
        endpoint = torch.full((2, 2), 2., dtype=torch.bfloat16)
        torch.save({'endpoint_latent': endpoint}, path)
        value['comparisons_to_baseline']['deployed'] = report.worker.endpoint_comparison(torch.ones_like(endpoint), endpoint)
        json_file(task/'native_smoke.json', value)
    elif damage in ('kernel_error', 'kernel_nan'):
        change('kernel_smoke.json', lambda x: x['checks']['fixedmass_vs_fp64'].update(
            relative_l2=1. if damage == 'kernel_error' else float('nan')))
    else: change('kernel_smoke.json', lambda x: x['checks']['production_text_branch'].update(query_shape=[1, 2, 16, 128]))
    with pytest.raises((ValueError, OSError, RuntimeError)):
        report.validate_task(task, 0, '123', protocol)


def test_presentation_fixed_shuffle_hidden_labels_and_exact_corner_pixels(tmp_path):
    archive = tmp_path/'archive'
    tasks = [task_summary(0), task_summary(1)]
    colors = ['red', 'green', 'blue', 'yellow']
    for task in tasks:
        task['task'] = tmp_path/f"prompt{task['index']}"
        for record in task['records']:
            arm = record['config']['method']
            relative = Path('images')/arm/'image.png'
            retained = archive/'raw'/f"prompt{task['index']}"/relative
            retained.parent.mkdir(parents=True)
            image = Image.new('RGB', (4096, 4096), 'black')
            draw = ImageDraw.Draw(image)
            for color, box in zip(colors, report.BOXES): draw.rectangle((box[0], box[1], box[2]-1, box[3]-1), fill=color)
            image.save(retained)
            record.update(image=str(task['task']/relative), png_sha256=file_sha256(retained))
    plan = dict(conditions=list(report.ARMS), presentation_seed=20261018,
                detail_boxes_xyxy=[list(b) for b in report.BOXES], visual_policy='Common stage before details')
    rubric = {'predicate_rubric': {'rows': [{'row': i, 'primary': f'primary{i}', 'secondary': f'secondary{i}',
                                           'exclusions': 'No invented constraint'} for i in (0, 1)]}}
    mapping = report.prepare_presentation(tasks, plan, rubric, archive)
    expected = [(a, i) for a in report.ARMS for i in (0, 1)]
    random.Random(20261018).shuffle(expected)
    assert [(r['arm'], r['prompt_index']) for r in mapping] == expected
    public = report.read(archive/'presentation/reviewer_manifest.json')
    assert not any(a in json.dumps(public) for a in report.ARMS)
    for view in public['views']:
        with Image.open(archive/'presentation'/view['common']['path']) as image: assert image.size == (1024, 1024)
        for j, crop in enumerate(view['details']):
            with Image.open(archive/'presentation'/crop['path']) as image:
                assert image.size == (1024, 1024)
                expected_color = Image.new('RGB', (1, 1), colors[j]).getpixel((0, 0))
                assert image.getextrema() == tuple((v, v) for v in expected_color)
                assert crop['resampled'] is False
                assert file_sha256(archive/'presentation'/crop['path']) == crop['sha256']


def test_missing_second_task_creates_no_archive(tmp_path, monkeypatch):
    job = tmp_path/'job123'; job.mkdir()
    destination = tmp_path/'results'/'crossmodal_cr21_job123'
    monkeypatch.setattr(report, 'RESULTS', tmp_path/'results')
    monkeypatch.setattr(report, 'PROTOCOL', json_file(tmp_path/'protocol.json', {'review_plan': {}, 'native_protocol': {}}))
    monkeypatch.setattr(report, 'PROTOCOL_SHA256', file_sha256(report.PROTOCOL))
    monkeypatch.setattr(report, 'pinned_json', lambda p: {'metrics': list(report.METRICS), 'rubric': {}})
    def check(task, index, *args):
        if index == 1: raise ValueError('Incomplete prompt1')
        return task_summary(0)
    monkeypatch.setattr(report, 'validate_task', check)
    with pytest.raises(ValueError, match='Incomplete prompt1'): report.archive_report(job, destination)
    assert not destination.exists()


def test_archive_refuses_overwrite_or_other_destination(tmp_path, monkeypatch):
    job = tmp_path/'job123'; job.mkdir()
    monkeypatch.setattr(report, 'RESULTS', tmp_path/'results')
    existing = report.RESULTS/'crossmodal_cr21_job123'; existing.mkdir(parents=True)
    (existing/'keep').write_text('retain')
    for target in (existing, tmp_path/'elsewhere'):
        with pytest.raises(ValueError, match='archive destination'): report.archive_report(job, target)
    assert (existing/'keep').read_text() == 'retain'


def test_archive_retains_actual_bytes_and_finishes_with_hash_manifest(tmp_path, monkeypatch):
    root = tmp_path/'repo'; root.mkdir()
    json_file(root/'tests/test_crossmodal_phase_report.py', {'test': 'snapshot'})
    job = tmp_path/'job123'; job.mkdir()
    tasks = [task_summary(0), task_summary(1)]
    for task in tasks:
        task['task'] = job/f"prompt{task['index']}"
        raw = json_file(task['task']/'raw.json', {'actual': task['index']})
        task['files'] = [raw]
    def pin(name, payload):
        path = json_file(root/name, payload)
        return {'path': str(path), 'sha256': file_sha256(path)}
    protocol = {'source_sha256': {}, 'review_plan': pin('plan.json', {'metrics': list(report.METRICS)}),
                'native_protocol': pin('native.json', {'rubric': pin('rubric.json', {})})}
    for name in ('design', 'prior_art', 'scoring_reference'): protocol[name] = pin(name+'.json', {})
    protocol_path = json_file(root/'protocol.json', protocol)
    monkeypatch.setattr(report, 'ROOT', root)
    monkeypatch.setattr(report, 'RESULTS', tmp_path/'results')
    monkeypatch.setattr(report, 'PROTOCOL', protocol_path)
    monkeypatch.setattr(report, 'PROTOCOL_SHA256', file_sha256(protocol_path))
    monkeypatch.setattr(report, 'validate_task', lambda path, i, *args: tasks[i])
    def presentation(*args):
        archive = args[-1]
        json_file(archive/'presentation/reviewer_manifest.json', {'tested_separately': True})
        json_file(archive/'presentation/private_mapping.json', {'tested_separately': True})
    monkeypatch.setattr(report, 'prepare_presentation', presentation)
    archive = report.RESULTS/'crossmodal_cr21_job123'
    output = report.archive_report(job, archive)
    assert output['complete'] is True and output['visual_review_complete'] is False
    inventory = report.read(archive/'retention.json')
    assert inventory['bytes'] == sum(r['bytes'] for r in inventory['files'])
    for item in inventory['files']:
        assert file_sha256(archive/item['archive']) == item['sha256'] == file_sha256(item['source'])
    finished = report.read(archive/'archive_complete.json')
    assert finished['report_sha256'] == file_sha256(archive/'report.json')
    assert finished['retention_sha256'] == file_sha256(archive/'retention.json')
