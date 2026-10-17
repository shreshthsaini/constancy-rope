import csv
import json
from pathlib import Path

import pytest

from scripts import crossmodal_phase_scores as score
from scripts import crossmodal_phase_pilot as worker
from crope.metrics.io import digest
from crope.utils import file_sha256, image_seed


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Isolate score-to-generation binding; worker/tensor validation has its own tests."""
    out = tmp_path / 'task'
    (out/'evaluation').mkdir(parents=True)
    reference = tmp_path/'reference.json'
    reference.write_text(json.dumps({'model_revisions': {'test': 'pinned'}}))
    protocol = {'source_sha256': {'crope/crossmodal_phase.py': 'a'*64},
                'scoring_reference': {'path': str(reference), 'sha256': file_sha256(reference)}}
    monkeypatch.setattr(score, 'scoring_preflight', lambda _: protocol)
    monkeypatch.setattr(worker, 'validate_dependencies', lambda *_: ({'id': score.DEV_IDS[0]}, {}))
    monkeypatch.setattr(worker, 'validate_arm_set', lambda *_: None)
    monkeypatch.setattr(score, 'verify_png', lambda *_: None)
    records, rows, ir = [], [], []
    for arm in ('baseline', 'fixedmass', 'unnormalized'):
        png = out/'images'/arm/'4096x4096/image.png'
        png.parent.mkdir(parents=True)
        png.write_bytes(b'PNG placeholder; shared PNG checker tested separately')
        config = dict(method=arm, prompt_id=score.DEV_IDS[0], seed=0,
            derived_seed=image_seed(score.DEV_IDS[0], 0), phase_protocol_sha256='p'*64,
            phase_source_sha256='a'*64, pilot_source_sha256=protocol['source_sha256'])
        record = dict(complete=True, config=config, image=str(png), png_sha256=file_sha256(png), seconds=1.)
        png.with_suffix('.json').write_text(json.dumps(record))
        records.append({**record, 'metadata': str(png.with_suffix('.json')),
                        'metadata_sha256': file_sha256(png.with_suffix('.json'))})
        rows.append(dict(method=arm, prompt_id=score.DEV_IDS[0], res='4096x4096', seed=0,
            path=str(png), config_hash=digest(config), metrics={**dict.fromkeys(score.METRICS, 1.), 'seconds': 1.}))
        ir.append(dict(png=str(png), method=arm, res='4096x4096', prompt_id=score.DEV_IDS[0], imagereward='0.5'))
    completion = dict(protocol_sha256='p'*64, prompt_index=0, completed_images=3,
        arms=['baseline', 'fixedmass', 'unnormalized'], smoke_passed=True, records=records)
    report = dict(complete=True, all_metrics_available=True, errors=[], model_revisions={'test': 'pinned'},
        config=dict(metrics='clip,pickscore,musiq,clipiqa', bootstrap=0, crop_seed=0, crops=4,
                    reference_method='baseline'), per_image=rows)
    def save():
        (out/'completion.json').write_text(json.dumps(completion))
        (out/'evaluation/metrics.json').write_text(json.dumps(report))
        with (out/'evaluation/imagereward.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['png','method','res','prompt_id','imagereward'])
            writer.writeheader(); writer.writerows(ir)
    save()
    return out, completion, report, ir, save


def test_complete_score_binding(bundle):
    out, *_ = bundle
    assert len(score.validate_scores(out, 0, 'p'*64)['records']) == 3


@pytest.mark.parametrize('damage', ['wrong_protocol', 'failed_smoke', 'missing_arm', 'foreign_sidecar',
    'foreign_score_config', 'nan_metric', 'missing_ir', 'wrong_ir_path', 'wrong_scorer', 'wrong_crop_policy'])
def test_rejects_foreign_or_incomplete_scores(bundle, damage):
    out, completed, report, ir, save = bundle
    if damage == 'wrong_protocol': completed['protocol_sha256'] = 'q'*64
    elif damage == 'failed_smoke': completed['smoke_passed'] = False
    elif damage == 'missing_arm': report['per_image'].pop()
    elif damage == 'foreign_sidecar':
        path = Path(report['per_image'][0]['path']).with_suffix('.json')
        value = json.loads(path.read_text()); value['config']['phase_protocol_sha256'] = 'q'*64
        path.write_text(json.dumps(value))
    elif damage == 'foreign_score_config': report['per_image'][0]['config_hash'] = 'f'*64
    elif damage == 'nan_metric': report['per_image'][0]['metrics']['musiq'] = float('nan')
    elif damage == 'missing_ir': ir.pop()
    elif damage == 'wrong_ir_path': ir[0]['png'] = '/other/run.png'
    elif damage == 'wrong_scorer': report['model_revisions']['test'] = 'different'
    elif damage == 'wrong_crop_policy': report['config']['crop_seed'] = 1
    save()
    with pytest.raises(ValueError): score.validate_scores(out, 0, 'p'*64)
