"""Validate complete per-image scores for one three-arm development task."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crope.utils import atomic_json, file_sha256, image_seed
from scripts.native_comparator import DEV_IDS, require, verify_png, pinned_json

ARMS = {'baseline', 'fixedmass', 'unnormalized'}
METRICS = {'clip', 'pickscore', 'musiq', 'musiq_crop1024', 'clipiqa', 'clipiqa_crop1024'}


def scoring_preflight(protocol_sha256):
    from huggingface_hub import hf_hub_download
    protocol_path = ROOT / '.agent/notes/crossmodal_phase_protocol.json'
    require(file_sha256(protocol_path) == protocol_sha256, 'Protocol changed')
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['source_sha256'].items():
        require(file_sha256(ROOT/name) == digest, 'Frozen source changed: '+name)
    for name in ('design', 'prior_art', 'review_plan', 'scoring_reference'):
        pinned_json(protocol[name])
    for name, pin in protocol['imagereward_files'].items():
        resolved = hf_hub_download('THUDM/ImageReward', name, local_files_only=True)
        require(Path(resolved) == Path(pin['path']) and file_sha256(resolved) == pin['sha256'],
                'ImageReward checkpoint/config changed: '+name)
    require(set(protocol['imagereward_files']) == {'ImageReward.pt', 'med_config.json'}, 'Missing ImageReward pins')
    return protocol


def validate_scores(out, prompt_index, protocol_sha256):
    require(prompt_index in (0, 1), 'Only fixed development prompts are allowed')
    protocol = scoring_preflight(protocol_sha256)
    from scripts.crossmodal_phase_pilot import validate_dependencies, validate_arm_set
    from crope.metrics.io import digest
    row_spec, _ = validate_dependencies(protocol, prompt_index)
    require(not (out/'failure.json').exists(), 'Generation workflow failed')
    completed = json.loads((out/'completion.json').read_text())
    require(completed['protocol_sha256'] == protocol_sha256 and completed['prompt_index'] == prompt_index
            and completed['completed_images'] == 3 and completed['arms'] == ['baseline', 'fixedmass', 'unnormalized']
            and completed['smoke_passed'] is True, 'Incomplete or foreign generation receipt')
    validate_arm_set(completed['records'], row_spec, protocol)
    generated = {r['config']['method']: r for r in completed['records']}
    report = json.loads((out/'evaluation/metrics.json').read_text())
    require(report['complete'] is True and report['all_metrics_available'] is True and not report['errors'],
            'Incomplete metric report')
    settings = report['config']
    require(settings['metrics'] == 'clip,pickscore,musiq,clipiqa' and settings['bootstrap'] == 0
            and settings['crop_seed'] == 0 and settings['crops'] == 4
            and settings['reference_method'] == 'baseline', 'Scoring settings changed')
    prior_pin = protocol['scoring_reference']
    require(file_sha256(prior_pin['path']) == prior_pin['sha256'], 'Scorer reference changed')
    prior = json.loads(Path(prior_pin['path']).read_text())
    require(report['model_revisions'] == prior['model_revisions'], 'Cached scorer revisions changed')
    rows = report['per_image']
    require(len(rows) == 3 and {r['method'] for r in rows} == ARMS, 'Missing or duplicate scored arms')
    with (out/'evaluation/imagereward.csv').open() as handle:
        ir = list(csv.DictReader(handle))
    require(len(ir) == 3 and {r['method'] for r in ir} == ARMS, 'Missing or duplicate ImageReward arms')
    records = []
    for row in rows:
        png = Path(row['path'])
        metadata = json.loads(png.with_suffix('.json').read_text())
        config = metadata['config']
        generation_record = generated[row['method']]
        require(Path(generation_record['metadata']) == png.with_suffix('.json')
                and file_sha256(generation_record['metadata']) == generation_record['metadata_sha256'],
                'Generation sidecar hash/path mismatch')
        saved_payload = {k: v for k, v in generation_record.items() if k not in ('metadata', 'metadata_sha256')}
        require(metadata == saved_payload and config['phase_protocol_sha256'] == protocol_sha256
                and config['pilot_source_sha256'] == protocol['source_sha256']
                and config['phase_source_sha256'] == protocol['source_sha256']['crope/crossmodal_phase.py'],
                'Scored image does not match accepted generation/protocol/source')
        require(row['config_hash'] == digest(config) and row['metrics']['seconds'] == metadata['seconds'],
                'Scorer sidecar provenance mismatch')
        require(png.is_relative_to(out/'images') and row['prompt_id'] == DEV_IDS[prompt_index]
                and row['res'] == '4096x4096' and row['seed'] == 0, 'Scored image identity mismatch')
        require(metadata['complete'] is True and config['method'] == row['method']
                and config['prompt_id'] == row['prompt_id'] and config['seed'] == 0
                and config['derived_seed'] == image_seed(row['prompt_id'], 0), 'Sidecar identity mismatch')
        verify_png(png, [4096, 4096], metadata['png_sha256'])
        scores = row['metrics']
        require(METRICS <= scores.keys() and all(math.isfinite(scores[k]) for k in METRICS), 'Missing metric')
        matches = [r for r in ir if r['png'] == str(png) and r['method'] == row['method']
                   and r['prompt_id'] == row['prompt_id'] and r['res'] == row['res']]
        require(len(matches) == 1 and math.isfinite(float(matches[0]['imagereward'])), 'ImageReward mismatch')
        records.append({'method': row['method'], 'png': str(png), 'png_sha256': metadata['png_sha256'],
                        'metrics': {k: scores[k] for k in sorted(METRICS)},
                        'imagereward': float(matches[0]['imagereward'])})
    return {'complete': True, 'prompt_index': prompt_index, 'protocol_sha256': protocol_sha256,
            'records': records, 'raw_sha256': {str(p): file_sha256(p) for p in
                (out/'completion.json', out/'evaluation/metrics.json', out/'evaluation/imagereward.csv')},
            'scope': 'Exploratory per-image scores, no confidence interval or efficacy claim. Visual review pending.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prompt-index', type=int, choices=(0, 1), required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    if a.preflight:
        scoring_preflight(a.protocol_sha256)
        return
    atomic_json(a.out/'workflow_scored.json', validate_scores(a.out.resolve(), a.prompt_index, a.protocol_sha256))


if __name__ == '__main__':
    main()
