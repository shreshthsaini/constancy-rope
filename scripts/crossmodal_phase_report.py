"""Validate and archive the completed CR21 pilot, without scoring or GPU work.

Run only after both tasks finish. Refuses to overwrite an archive. Original
receipts retain their original paths; retention.json maps them to durable copies.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crope.utils import file_sha256
from scripts import crossmodal_phase_pilot as worker
from scripts import crossmodal_phase_scores as scorer
from scripts.native_comparator import pinned_json, require

PROTOCOL = ROOT/'.agent/notes/crossmodal_phase_protocol.json'
PROTOCOL_SHA256 = '7a5d072bbaeae9534dd01b0b26bab9da8326ccb01311ef276a0a7d22bb176be9'
RESULTS = Path('/work/09032/saini_2/vista/results/constancy_rope')
METRICS = ('imagereward', 'musiq', 'musiq_crop1024', 'clip', 'pickscore', 'clipiqa', 'clipiqa_crop1024')
ARMS = ('baseline', 'fixedmass', 'unnormalized')
BOXES = ((0, 0, 1024, 1024), (3072, 0, 4096, 1024),
         (0, 3072, 1024, 4096), (3072, 3072, 4096, 4096))
CPU_REDUCTION_TOLERANCE = {'relative': 1e-12, 'absolute': 1e-12}
RAW_NAMES = ('manifest.json', 'completion.json', 'workflow_scored.json', 'native_smoke.json',
             'kernel_smoke.json', 'progress.json', 'evaluation/metrics.json', 'evaluation/imagereward.csv')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def contained(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path.is_file() and path.is_relative_to(root), 'Missing or escaping artifact: '+str(path))
    return path


def validate_record_files(record, task):
    """Bind immutable sidecar payloads, PNGs and tensor receipts to this task."""
    files = []
    metadata = contained(record['metadata'], task)
    require(file_sha256(metadata) == record['metadata_sha256'], 'Sidecar digest mismatch')
    require(read(metadata) == {k: v for k, v in record.items() if k not in ('metadata', 'metadata_sha256')},
            'Sidecar payload differs from completion')
    files.append(metadata)
    for key in ('initial_transformer_input', 'endpoint_latent'):
        receipt = record[key]
        path = contained(receipt['path'], task)
        require(file_sha256(path) == receipt['file_sha256'], 'Tensor file digest mismatch')
        files.append(path)
    if 'image' in record:
        path = contained(record['image'], task/'images')
        require(path.with_suffix('.json') == metadata, 'Image and sidecar paths differ')
        worker.native.verify_png(path, [4096, 4096], record['png_sha256'])
        files.append(path)
    return files


def validate_native_smoke(smoke, prompt, protocol):
    require(protocol['deployed_smoke_max_relative_l2'] == worker.FIXED['deployed_smoke_max_relative_l2'],
            'Native smoke threshold changed')
    require(smoke.get('no_vae_decode') is True and smoke.get('cosine_threshold_is_diagnostic') is True,
            'Native smoke policy mismatch')
    worker.validate_arm_set(smoke['records'], prompt, protocol, smoke=True, protocol_hash=PROTOCOL_SHA256)
    import torch
    endpoints = {r['config']['method']: torch.load(r['endpoint_latent']['path'], map_location='cpu',
                 weights_only=True, mmap=True)['endpoint_latent'] for r in smoke['records']}
    recomputed = {arm: worker.endpoint_comparison(endpoints['baseline'], value) for arm, value in endpoints.items()}
    recorded = smoke['comparisons_to_baseline']
    require(set(recorded) == set(recomputed), 'Native smoke comparison arms differ')
    residuals = {}
    for arm, current in recomputed.items():
        previous = recorded[arm]
        require(set(previous) == set(current) and type(previous['byte_exact']) is bool and
                previous['byte_exact'] == current['byte_exact'], 'Native smoke byte comparison differs')
        residuals[arm] = {}
        for key in ('relative_l2', 'max_abs', 'cosine', 'reference_l2'):
            a, b = current[key], previous[key]
            if a is None or b is None:
                require(a is b, 'Native smoke undefined norm differs')
                residuals[arm][key] = None
            else:
                require(isinstance(b, (int, float)) and not isinstance(b, bool) and math.isfinite(b) and
                        math.isclose(a, b, rel_tol=CPU_REDUCTION_TOLERANCE['relative'],
                                     abs_tol=CPU_REDUCTION_TOLERANCE['absolute']),
                        'Native smoke numerical comparison differs from actual endpoints')
                residuals[arm][key] = a-b
    require(recomputed['deployed']['relative_l2'] is not None and
            recomputed['deployed']['relative_l2'] <= protocol['deployed_smoke_max_relative_l2'] and
            all(recomputed[a]['byte_exact'] for a in ('fixedmass', 'unnormalized')), 'Native smoke gate failed')
    return {'comparisons': recomputed, 'receipt_residual_convention': 'residual = recomputed - recorded',
            'receipt_residuals': residuals, 'cpu_reduction_tolerance': dict(CPU_REDUCTION_TOLERANCE),
            'gate_rechecked_from_actual_tensors': True, 'byte_equality_tolerance': None}


def validate_task(task, index, job_id, protocol):
    """Reuse frozen validators, then independently bind smoke and task receipts."""
    task = task.resolve()
    require(not (task/'failure.json').exists(), 'Task failure receipt exists')
    files = [contained(task/name, task) for name in RAW_NAMES]
    completion = read(task/'completion.json')
    smoke = read(task/'native_smoke.json')
    # Reject foreign paths before frozen validators open their tensor receipts.
    for record in [*completion['records'], *smoke['records']]:
        files.extend(validate_record_files(record, task))
    expected_scores = scorer.validate_scores(task, index, PROTOCOL_SHA256)
    require(read(task/'workflow_scored.json') == expected_scores, 'Official scoring receipt differs from recomputation')
    prompt, accepted = worker.validate_dependencies(protocol, index)
    manifest = read(task/'manifest.json')
    expected = dict(protocol_sha256=PROTOCOL_SHA256, prompt_index=index, prompt=prompt,
                    source_sha256=protocol['source_sha256'], model=accepted['model'],
                    torch=accepted['torch'], cuda=accepted['cuda'], diffusers=accepted['diffusers'],
                    slurm_job_id=job_id, scope=worker.FIXED['scope'], quality_scored=False,
                    seed_device='cpu', seed_policy='crope.utils.image_seed(prompt_id,0)')
    require(all(manifest.get(k) == v for k, v in expected.items()), 'Generation manifest provenance mismatch')
    require(isinstance(manifest.get('hardware'), str) and bool(manifest['hardware']), 'Missing hardware receipt')
    require(math.isfinite(manifest['model_loading_seconds']) and manifest['model_loading_seconds'] >= 0,
            'Invalid model loading time')
    progress = read(task/'progress.json')
    require(progress.get('status') == 'completed' and progress.get('completed_images') == 3, 'Task is incomplete')
    worker.validate_arm_set(completion['records'], prompt, protocol, protocol_hash=PROTOCOL_SHA256)
    kernel = read(task/'kernel_smoke.json')
    from crope.crossmodal_phase import NUMERICAL_POLICY
    require(kernel.get('passed') is True and kernel.get('gpu_runtime_verified') is True
            and kernel.get('backend') == worker.FIXED['backend'] and kernel.get('tolerances') == NUMERICAL_POLICY,
            'GPU kernel smoke receipt missing or foreign')
    branch = kernel['checks']['production_text_branch']
    require(branch.get('finite') is True and branch.get('query_shape') == [1, 24, 65536, 128]
            and branch.get('key_shape') == [1, 24, 512, 128], 'Production text branch smoke mismatch')
    for name in ('private_vs_fp64', 'private_vs_public_cudnn', 'fixedmass_vs_fp64', 'unnormalized_vs_fp64'):
        check = kernel['checks'][name]
        prefix = 'kernel_output' if name.startswith('private') else 'operator'
        require(all(math.isfinite(check[key]) and 0 <= check[key] <= NUMERICAL_POLICY[prefix+'_'+key]
                    for key in ('relative_l2', 'max_abs')), 'Kernel numerical smoke gate failed')
    require(math.isfinite(kernel['checks']['lse_max_abs']) and
            0 <= kernel['checks']['lse_max_abs'] <= NUMERICAL_POLICY['kernel_lse_max_abs'] and
            all(kernel['checks'][arm+'_identity_byte_equal'] is True for arm in ARMS[1:]),
            'Kernel LSE or identity gate failed')
    native_recheck = validate_native_smoke(smoke, prompt, protocol)
    return {'index': index, 'task': task, 'prompt': prompt, 'manifest': manifest,
            'records': completion['records'], 'scores': expected_scores['records'], 'files': sorted(set(files)),
            'native_smoke_recheck': native_recheck}


def summarize(tasks):
    require([t['index'] for t in tasks] == [0, 1], 'Exactly both ordered development tasks required')
    rows, contrasts, runtime = [], [], []
    for task in tasks:
        scores = {r['method']: {**r['metrics'], 'imagereward': r['imagereward']} for r in task['scores']}
        require(set(scores) == set(ARMS) and len(task['scores']) == 3, 'Missing or duplicate score arm')
        receipts = {r['config']['method']: r for r in task['records']}
        require(set(receipts) == set(ARMS), 'Missing runtime arm')
        for arm in ARMS:
            values = {m: scores[arm][m] for m in METRICS}
            require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                        for v in values.values()), 'Nonfinite metric')
            rows.append({'prompt_index': task['index'], 'prompt_id': task['prompt']['id'], 'arm': arm, 'metrics': values})
            r = receipts[arm]
            require(math.isfinite(r['seconds']) and r['seconds'] >= 0 and
                    type(r['peak_memory_bytes']) is int and r['peak_memory_bytes'] > 0, 'Invalid runtime receipt')
            base = receipts['baseline']
            runtime.append({'prompt_index': task['index'], 'arm': arm, 'seconds': r['seconds'],
                            'peak_memory_bytes': r['peak_memory_bytes'],
                            'seconds_minus_baseline': r['seconds']-base['seconds'],
                            'seconds_ratio_to_baseline': r['seconds']/base['seconds'] if base['seconds'] else None,
                            'peak_memory_bytes_minus_baseline': r['peak_memory_bytes']-base['peak_memory_bytes'],
                            'timing_scope': r['seconds_scope']})
        for arm in ARMS[1:]:
            contrasts.append({'prompt_index': task['index'], 'arm': arm, 'reference': 'baseline',
                              'delta': {m: scores[arm][m]-scores['baseline'][m] for m in METRICS}})
    return {'per_image': rows, 'paired_contrasts': contrasts,
        'descriptive_means': {arm: {m: sum(r['metrics'][m] for r in rows if r['arm'] == arm)/2
                                   for m in METRICS} for arm in ARMS},
        'descriptive_mean_contrasts': {arm: {m: sum(r['delta'][m] for r in contrasts if r['arm'] == arm)/2
                                           for m in METRICS} for arm in ARMS[1:]},
        'runtime': runtime,
        'descriptive_runtime_means': {arm: {key: sum(r[key] for r in runtime if r['arm'] == arm)/2
                                           for key in ('seconds', 'peak_memory_bytes')} for arm in ARMS},
        'model_loading_seconds': {str(t['index']): t['manifest']['model_loading_seconds'] for t in tasks},
        'native_smoke_rechecks': {str(t['index']): t.get('native_smoke_recheck') for t in tasks},
        'limits': ['Two existing development captions, all seven metrics retained. No CI, significance, aggregate score or promotion.',
                   'Sequential baseline/fixedmass/unnormalized arms are not warmup-matched. Recorded generation timing includes decode and excludes model loading and file saving.',
                   'Unnormalized control explicitly adds an image-only attention pass; compare cost descriptively, not as an equal-cost method.',
                   'Visual review is pending. Correctness and scalar scores do not establish novelty, competitive quality or causal repair.']}


def prepare_presentation(tasks, plan, rubric, archive):
    from PIL import Image
    require(plan['conditions'] == list(ARMS) and plan['presentation_seed'] == 20261018
            and plan['detail_boxes_xyxy'] == [list(b) for b in BOXES], 'Presentation policy changed')
    order = [(arm, index) for arm in ARMS for index in (0, 1)]
    random.Random(plan['presentation_seed']).shuffle(order)
    predicates = {r['row']: r for r in rubric['predicate_rubric']['rows']}
    mapping, views = [], []
    for position, (arm, index) in enumerate(order, 1):
        label = f'view{position:02d}'
        task = tasks[index]
        record = next(r for r in task['records'] if r['config']['method'] == arm)
        source = archive/'raw'/f'prompt{index}'/Path(record['image']).relative_to(task['task'])
        require(file_sha256(source) == record['png_sha256'], 'Retained presentation image changed')
        directory = archive/'presentation'/label
        directory.mkdir(parents=True)
        preview = directory/'common1024.png'
        with Image.open(source) as original:
            require(original.size == (4096, 4096) and original.mode == 'RGB', 'Presentation image geometry/mode mismatch')
            original.resize((1024, 1024), Image.Resampling.LANCZOS).save(preview)
            crops = []
            for corner, box in enumerate(BOXES, 1):
                crop = directory/f'detail{corner}.png'
                original.crop(box).save(crop)
                crops.append({'path': str(crop.relative_to(archive/'presentation')), 'sha256': file_sha256(crop),
                              'box_xyxy': list(box), 'pixels': [1024, 1024], 'resampled': False})
        views.append({'label': label, 'caption': task['prompt']['prompt'],
                      'primary': predicates[index]['primary'], 'secondary': predicates[index]['secondary'],
                      'exclusions': predicates[index]['exclusions'],
                      'common': {'path': str(preview.relative_to(archive/'presentation')), 'sha256': file_sha256(preview),
                                 'pixels': [1024, 1024], 'resampling': 'Pillow LANCZOS'}, 'details': crops})
        mapping.append({'label': label, 'arm': arm, 'prompt_index': index,
                        'source': str(source.relative_to(archive)), 'original_source': record['image'],
                        'source_sha256': record['png_sha256']})
    write(archive/'presentation/private_mapping.json', {'seed': plan['presentation_seed'], 'mapping': mapping,
          'instruction': 'Withhold this mapping and report metrics until common and detail readings are frozen.'})
    write(archive/'presentation/reviewer_manifest.json', {'views': views,
          'instructions': plan['visual_policy'], 'labels': ['satisfied', 'violated', 'unjudgeable'],
          'review_type': 'Independent model-based readings, not human-study evidence. Common stage before details.',
          'crop_limit': 'Identical pixel boxes need not contain corresponding objects across changed compositions.'})
    return mapping


def archive_report(job_root, archive):
    import torch
    torch.set_num_threads(2)
    job_root, archive = job_root.resolve(), archive.resolve()
    match = re.fullmatch(r'job([0-9]+)', job_root.name)
    require(match is not None and job_root.is_dir(), 'Expected an existing jobNUMBER directory')
    job_id = match.group(1)
    require(archive == RESULTS/f'crossmodal_cr21_job{job_id}' and not archive.exists(),
            'Require a new exact paper-critical archive destination')
    require(file_sha256(PROTOCOL) == PROTOCOL_SHA256, 'Frozen protocol changed')
    protocol = read(PROTOCOL)
    plan = pinned_json(protocol['review_plan'])
    require(plan['metrics'] == list(METRICS), 'Metric policy differs from frozen review plan')
    native_protocol = pinned_json(protocol['native_protocol'])
    rubric = pinned_json(native_protocol['rubric'])
    tasks = [validate_task(job_root/f'prompt{i}', i, job_id, protocol) for i in (0, 1)]
    report = summarize(tasks)
    # No write occurs until both workflows and all retained witnesses validate.
    archive.mkdir(parents=True, exist_ok=False)
    inventory = []
    def retain(source, relative):
        source = Path(source)
        target = archive/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = file_sha256(source)
        shutil.copy2(source, target)
        require(file_sha256(target) == before and file_sha256(source) == before, 'Artifact changed while archiving')
        inventory.append({'source': str(source.resolve()), 'archive': str(relative),
                          'sha256': before, 'bytes': target.stat().st_size})
    for task in tasks:
        for source in task['files']:
            retain(source, Path('raw')/f"prompt{task['index']}"/source.relative_to(task['task']))
    retain(PROTOCOL, Path('protocol.json'))
    for name in ('design', 'prior_art', 'review_plan', 'scoring_reference', 'native_protocol'):
        retain(protocol[name]['path'], Path('provenance')/(name+'.json'))
    retain(native_protocol['rubric']['path'], Path('provenance/native_rubric.json'))
    for name in protocol['source_sha256']:
        retain(ROOT/name, Path('source')/name)
    retain(Path(__file__), Path('source/scripts/crossmodal_phase_report.py'))
    retain(ROOT/'tests/test_crossmodal_phase_report.py', Path('source/tests/test_crossmodal_phase_report.py'))
    prepare_presentation(tasks, plan, rubric, archive)
    report.update(complete=True, protocol_sha256=PROTOCOL_SHA256, job_id=job_id, source_job_root=str(job_root),
                  archive=str(archive), report_source_sha256=file_sha256(__file__),
                  presentation_mapping='presentation/private_mapping.json', visual_review_complete=False)
    write(archive/'retention.json', {'files': inventory, 'bytes': sum(r['bytes'] for r in inventory),
                                   'original_paths_preserved_in_receipts': True})
    write(archive/'report.json', report)
    write(archive/'archive_complete.json', {'complete': True, 'report_sha256': file_sha256(archive/'report.json'),
          'retention_sha256': file_sha256(archive/'retention.json'),
          'reviewer_manifest_sha256': file_sha256(archive/'presentation/reviewer_manifest.json'),
          'private_mapping_sha256': file_sha256(archive/'presentation/private_mapping.json')})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    report = archive_report(args.job_root, args.archive)
    print(json.dumps({'archive': report['archive'], 'job_id': report['job_id'], 'complete': True}))


if __name__ == '__main__':
    main()
