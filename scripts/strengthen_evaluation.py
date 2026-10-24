#!/usr/bin/env python3
"""One-shot completion gate and publication of existing-fleet evaluation tasks.

Usage: strengthen_evaluation.py BLOCK_QUEUE.json --references REFS.jsonl [--publish]
No scheduler calls, background work, generation, or polling. Missing generations
write a status receipt and exit2. A complete, valid block stages evaluator and
ImageReward tasks. --publish atomically links those scripts into taskq/pending.
Seeds1 and2 get separate evaluator roots after the whole generation block passes.
Groups above 200 images use method shards. Run MANIFEST --merge after their
evaluator tasks finish to validate receipts and write canonical outputs on CPU.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crope.utils import file_sha256, image_seed, image_stem, read_prompts
from scripts.strengthen_analysis import dotted, inspect_sidecar
from scripts.strengthen_analysis import METRICS as ANALYSIS_METRICS

SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
QUEUE = SCRATCH / 'taskq'
NOTES = ROOT / '.agent/notes/strengthen/evaluation'
ENVS = Path('/work/09032/saini_2/vista/library/uv-envs')
METRICS = 'clip,pickscore,musiq,clipiqa,fid,kid,patch_fid,patch_kid,spectrum,tiling'
BLOCKS = {'factorial': 'P1', 'tsweep': 'P3', 'fresh': 'E1', 'seeds': 'E2', 'qwen': 'E3',
          'ckpt_matrix': 'diagnostic', 'default_shift': 'diagnostic'}
SHARD_THRESHOLD = 200
OUTPUT_FILES = ('per_image.csv', 'summary.csv', 'paired_deltas.csv', 'metrics.json')


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def immutable_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Frozen artifact differs: {path}')
        return
    with path.open('x') as handle:
        handle.write(text)


def expected_samples(manifest):
    block = manifest['block']
    if block not in BLOCKS:
        raise ValueError(f'Unsupported evaluation block {block!r}')
    prompts_path = Path(manifest['prompts'])
    if file_sha256(prompts_path) != manifest['prompts_sha256']:
        raise ValueError('Prompt manifest changed after generation was queued')
    prompts = read_prompts(prompts_path)
    samples, keys = [], set()
    for task in manifest['tasks']:
        if task['prompt_ids'] != [r['id'] for r in prompts[task['start']:task['end']]]:
            raise ValueError(f"Task {task['name']} prompt interval differs")
        if task['n'] != len(task['prompt_ids']):
            raise ValueError('Task sample count differs')
        out = Path(task['out'])
        if out.parent.name != task['method'] or out.name != '4096x4096':
            raise ValueError('Output path must preserve METHOD/4096x4096 identities')
        for row in prompts[task['start']:task['end']]:
            key = (task['method'], row['id'], task['seed'])
            if key in keys:
                raise ValueError(f'Duplicate generation identity: {key}')
            keys.add(key)
            png = out / (image_stem(row['id'], task['seed']) + '.png')
            samples.append(dict(method=task['method'], raw_method=task['method'], prompt_id=row['id'],
                                seed=task['seed'], prompt=row['prompt'], path=str(png),
                                res=out.name, root=str(out.parents[1]), task=task['name']))
    if len(samples) != manifest['n_generate'] or len(manifest['tasks']) != manifest['n_tasks']:
        raise ValueError('Generation manifest total differs from listed tasks')
    return samples


def completion_status(manifest, samples, queue):
    missing = [s['path'] for s in samples if not Path(s['path']).is_file()
               or not Path(s['path']).with_suffix('.json').is_file()]
    unfinished = []
    for task in manifest['tasks']:
        done = queue / 'done' / task['name']
        if not done.is_file():
            unfinished.append(task['name'])
        elif file_sha256(done) != task['sha256']:
            raise ValueError(f"Finished generation task hash differs: {task['name']}")
    return dict(complete=not missing and not unfinished, expected_images=len(samples),
                missing_image_pairs=missing, unfinished_tasks=unfinished)


def inspect_diagnostic(sample, manifest):
    """Diagnostic contracts keep their own guidance, seed and schedule settings."""
    from types import SimpleNamespace
    from scripts import strengthen_diagnostics as diagnostic
    path = Path(sample['path'])
    sidecar = path.with_suffix('.json')
    metadata = json.loads(sidecar.read_text())
    config = metadata['config']
    if metadata.get('complete') is not True:
        raise ValueError(f'Incomplete diagnostic sidecar: {sidecar}')
    expected = dict(method=sample['method'], prompt_id=sample['prompt_id'], prompt=sample['prompt'],
                    seed=sample['seed'], steps=28, dtype='bfloat16')
    if manifest['block'] == 'ckpt_matrix':
        arms = {diagnostic.method_name(checkpoint, guidance): (checkpoint, guidance)
                for checkpoint in ['dev', 'krea'] for guidance in [3.5, 4.5]}
        if sample['method'] not in arms or sample['seed'] != 42:
            raise ValueError('Unknown checkpoint/guidance cell or seed')
        checkpoint, guidance = arms[sample['method']]
        official = SCRATCH / 'third_party/DyPE'
        official_files = {str(Path(name).relative_to(official)): digest for name, digest in manifest['sources'].items()
                          if Path(name).is_relative_to(official) and name.endswith('.py')}
        if not {'flux/pipeline_flux.py', 'flux/transformer_flux.py'}.issubset(official_files):
            raise ValueError('Missing frozen official DyPE implementation')
        provenance = dict(official_root=str(official), official_commit=diagnostic.OFFICIAL_COMMIT,
                          official_python_sha256=official_files,
                          wrapper_sha256=manifest['sources'][str(ROOT / 'scripts/strengthen_diagnostics.py')])
        versions = {name: importlib.metadata.version(name) for name in ['torch', 'diffusers', 'transformers']}
        args = SimpleNamespace(checkpoint=checkpoint, guidance=guidance, seed=42, steps=28,
                               prompts=Path(manifest['prompts']))
        expected = diagnostic.build_config(args, {'id': sample['prompt_id'], 'prompt': sample['prompt']}, provenance, versions)
        mu, terminal = 1.15, True
    else:
        if sample['method'] != 'sega_default_shift' or sample['seed'] != 0:
            raise ValueError('Unexpected default-shift diagnostic identity')
        expected.update({'controller_method': 'sega_official', 'res': '4096x4096',
                         'derived_seed': image_seed(sample['prompt_id'], 0), 'guidance': 4.5,
                         'model': 'black-forest-labs/FLUX.1-dev', 'model_revision': diagnostic.DEV_REVISION,
                         'rope.method': 'sega_official', 'eta': 1., 'temperature': 'logN',
                         'schedule_mode': 'diffusers_default', 'original_max_image_seq_len': 4096,
                         'scheduler.max_image_seq_len': 4096, 'scheduler.base_image_seq_len': 256,
                         'scheduler.base_shift': .5, 'scheduler.max_shift': 1.15})
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if metadata.get('config_fingerprint') != fingerprint or config.get('temperature_multiplier') is not None:
            raise ValueError('Default-shift config fingerprint or temperature override differs')
        mu, terminal = 11.55, False
    for key, value in expected.items():
        if dotted(config, key) != value:
            raise ValueError(f'{sidecar}: diagnostic config {key} differs from expected {value!r}')
    initial = np.linspace(1., 1 / 28, 28)
    scheduled = np.exp(mu) / (np.exp(mu) + (1 / initial - 1))
    if terminal:
        scheduled = np.r_[scheduled, 0.]
    actual = np.asarray(metadata.get('sigmas', []), dtype=float)
    if actual.shape != scheduled.shape or not np.allclose(actual, scheduled, rtol=0, atol=2e-7):
        raise ValueError(f'{sidecar}: diagnostic realized schedule differs from mu{mu}')
    if terminal:
        timesteps = np.asarray(metadata.get('timesteps', []), dtype=float)
        if timesteps.shape != (28,) or not np.allclose(timesteps, scheduled[:-1] * 1000, rtol=0, atol=2e-4):
            raise ValueError(f'{sidecar}: official DyPE timesteps differ from sigmas')
    return {key: sample[key] for key in ['method', 'prompt_id', 'seed', 'path']} | {
        'sidecar': str(sidecar), 'sidecar_sha256': file_sha256(sidecar), 'config': config,
        'seconds': metadata.get('seconds'), 'peak_memory_bytes': metadata.get('peak_memory_bytes'),
        'role': 'development_diagnostic_no_confirmatory_hypothesis', 'verified_mu': mu}


def verify_samples(manifest, samples):
    """Validate complete sidecars plus exact PNG hashes before scorer publication."""
    from PIL import Image
    witnesses = []
    expected_paths = {str(Path(sample['path']).resolve()) for sample in samples}
    roots = {sample['root'] for sample in samples}
    for root in roots:
        actual = {str(path.resolve()) for path in Path(root).glob('*/*/*.png')}
        if actual - expected_paths:
            raise ValueError(f'Unexpected images in evaluator root {root}: {sorted(actual - expected_paths)[:3]}')
        expected_sidecars = {str(Path(path).with_suffix('.json')) for path in expected_paths}
        actual_sidecars = {str(path.resolve()) for path in Path(root).glob('*/*/*.json')}
        if actual_sidecars - expected_sidecars:
            raise ValueError(f'Extra JSON files could be mistaken for image sidecars in {root}')
    for sample in samples:
        png = Path(sample['path'])
        sidecar = png.with_suffix('.json')
        spec = {'block': BLOCKS[manifest['block']], 'expected_config': manifest.get('expected_config', {}),
                'expected_by_method': manifest.get('expected_by_method', {})}
        witness = (inspect_diagnostic(sample, manifest) if BLOCKS[manifest['block']] == 'diagnostic' else
                   inspect_sidecar(sample, sample['prompt'], spec, {}))
        metadata = json.loads(sidecar.read_text())
        if manifest['block'] == 'qwen':
            checked = ROOT / '.agent/notes/strengthen/qwen_cpu_checks_final.json'
            if str(checked) not in manifest['sources']:
                raise ValueError('Qwen generation manifest lacks frozen CPU parity provenance')
            provenance = json.loads(checked.read_text())['provenance']
            if metadata['config'].get('provenance') != provenance:
                raise ValueError(f'Qwen output source/version provenance differs: {png}')
        digest = file_sha256(png)
        if not metadata.get('png_sha256') or digest != metadata['png_sha256']:
            raise ValueError(f'PNG checksum differs: {png}')
        if metadata.get('image') and Path(metadata['image']).resolve() != png.resolve():
            raise ValueError(f'Sidecar image path differs: {png}')
        with Image.open(png) as image:
            if image.size != (4096, 4096):
                raise ValueError(f'Diagnostic PNG dimensions differ: {png}')
            image.verify()
        witnesses.append({**witness, 'png_sha256': digest, 'image_hash_verified_by_evaluation_gate': True})
    return witnesses


def dependency_hashes(manifest_path, manifest, references, samples, queue=QUEUE):
    paths = {Path(manifest_path), Path(manifest['prompts']), Path(references), Path(__file__),
             ROOT / 'scripts/strengthen_analysis.py', ROOT / 'scripts/eval.py',
             ROOT / 'scripts/imagereward_score.py', ROOT / 'scripts/strengthen_diagnostics.py',
             ROOT / 'crope/utils.py'}
    paths.update((ROOT / 'crope/metrics').glob('*.py'))
    for name, expected in manifest['sources'].items():
        if file_sha256(name) != expected:
            raise ValueError(f'Frozen generation source changed: {name}')
        paths.add(Path(name))
    for task in manifest['tasks']:
        # Generation publication renames staging -> pending; workers then move
        # the same script through running to done. Staging no longer exists.
        completed = Path(queue) / 'done' / task['name']
        if file_sha256(completed) != task['sha256']:
            raise ValueError(f"Completed generation task changed: {task['name']}")
        paths.add(completed)
    selected = {s['prompt_id'] for s in samples}
    refs = [json.loads(line) for line in Path(references).read_text().splitlines() if line.strip()]
    ref_ids = [str(row['id']) for row in refs]
    if len(ref_ids) != len(set(ref_ids)) or not selected.issubset(ref_ids):
        raise ValueError('Reference manifest misses selected IDs or duplicates IDs')
    # The evaluator reads all declared references for descriptive FID/KID.
    for row in refs:
        paths.add(Path(row['path']))
    for sample in samples:
        paths.add(Path(sample['path']))
        paths.add(Path(sample['path']).with_suffix('.json'))
    return {str(path.resolve()): file_sha256(path) for path in sorted(paths)}


def evaluation_groups(manifest, samples, eval_base):
    groups = {}
    for sample in samples:
        label = f"seeds_seed{sample['seed']}" if manifest['block'] == 'seeds' else manifest['block']
        group = groups.setdefault(label, {'root': sample['root'], 'samples': []})
        if sample['root'] != group['root']:
            raise ValueError('One evaluator group has multiple source roots')
        group['samples'].append(sample)
    for label, group in groups.items():
        methods = sorted({sample['method'] for sample in group['samples']})
        group.update(label=label, out=str(Path(eval_base) / label),
                     sharded=len(group['samples']) > SHARD_THRESHOLD, shards=[])
        for method in methods if group['sharded'] else [None]:
            subset = [sample for sample in group['samples'] if method is None or sample['method'] == method]
            name = f'90_eval_{label}' + (f'_{method}' if method else '') + '.sh'
            reference = method or ('sega_official' if 'sega_official' in methods else methods[0])
            group['shards'].append(dict(name=name, methods=[method] if method else methods,
                root=str(Path(group['root']) / method) if method else group['root'],
                out=str(Path(group['out']) / 'shards' / method) if method else group['out'],
                reference_method=reference, samples=subset))
    return dict(sorted(groups.items()))


def task_bodies(manifest, samples, references, checksum_file, eval_base, plan_path=None):
    groups = evaluation_groups(manifest, samples, eval_base)
    scripts = {}
    for label, group in groups.items():
        out = Path(group['out'])
        env = '\n'.join([
            '#!/usr/bin/env bash', 'set -euo pipefail', f'cd {shlex.quote(str(ROOT))}',
            'export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8',
            'export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1',
            'export HF_HOME=/scratch/09032/saini_2/tmp/hf',
            'export TORCH_HOME=/scratch/09032/saini_2/tmp/cache',
            'export XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache',
            f'export WANDB_DIR={shlex.quote(str(SCRATCH / "wandb"))}',
            'export WANDB_PROJECT=constancy-rope',
            'if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" ]]; then export WANDB_MODE=offline; fi',
            f'sha256sum --status -c {shlex.quote(str(checksum_file))}',
        ]) + '\n'
        for shard in group['shards']:
            evaluate = [str(ENVS / 'crope/bin/python'), 'scripts/eval.py', '--root', shard['root'],
                        '--prompts', manifest['prompts'], '--references', str(references),
                        '--reference-method', shard['reference_method'], '--metrics', METRICS, '--allow-missing',
                        '--res', '4096x4096', '--out', shard['out'], '--device', 'cuda', '--threads', '8']
            body = env + shlex.join(evaluate) + '\n'
            if plan_path is not None:
                body += shlex.join([str(ENVS / 'crope/bin/python'), 'scripts/strengthen_evaluation.py',
                    str(plan_path), '--seal-output', shard['name'], '--plan-sha256', file_sha256(plan_path)]) + '\n'
            scripts[shard['name']] = body
        reward = [str(ENVS / 'mend/bin/python'), 'scripts/imagereward_score.py', group['root'],
                  manifest['prompts'], str(out / 'ir.csv'), '--device', 'cuda']
        scripts[f'90_ir_{label}.sh'] = env + f'mkdir -p {shlex.quote(str(out))}\n' + shlex.join(reward) + '\n'
    return scripts


def publish_scripts(scripts, stage, queue, publish):
    result = []
    for name, body in sorted(scripts.items()):
        path = stage / name
        immutable_text(path, body)
        digest = file_sha256(path)
        locations = [queue / state / name for state in ['pending', 'done', 'failed']]
        locations += list((queue / 'running').glob(name + '.*'))
        present = [p for p in locations if p.exists()]
        if len(present) > 1:
            raise ValueError(f'Multiple live/completed copies of {name}')
        for existing in present:
            if file_sha256(existing) != digest:
                raise ValueError(f'Existing evaluation task differs: {existing}')
        if present and present[0].parent.name == 'failed':
            raise ValueError(f'Existing evaluation failed; preserve it and inspect before retry: {present[0]}')
        reservation = stage / (name + '.publication.json')
        if reservation.exists():
            record = json.loads(reservation.read_text())
            if record.get('sha256') != digest:
                raise ValueError(f'Publication receipt differs for {name}')
            # A worker can move pending -> running -> done between directory scans.
            # Never publish a second copy once the first publication was reserved.
            result.append({'name': name, 'sha256': digest, 'staged': str(path),
                           'status': present[0].parent.name if present else 'already_reserved_check_receipt',
                           'publication_receipt': str(reservation)})
            continue
        if publish:
            record = dict(task=name, sha256=digest, timestamp=now(), state='reserved_before_publication')
            try:
                with reservation.open('x') as handle:
                    handle.write(json.dumps(record, indent=2) + '\n')
            except FileExistsError:
                record = json.loads(reservation.read_text())
                if record.get('sha256') != digest:
                    raise ValueError(f'Concurrent publication reservation collision: {name}')
                result.append({'name': name, 'sha256': digest, 'staged': str(path),
                               'status': 'reserved_by_other_publisher'})
                continue
        if not present and publish:
            (queue / 'pending').mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, queue / 'pending' / name)
            except FileExistsError:
                if file_sha256(queue / 'pending' / name) != digest:
                    raise ValueError(f'Concurrent publication collision: {name}')
        if publish:
            record['state'] = 'observed_existing' if present else 'published'
            write_json(reservation, record)
        result.append({'name': name, 'sha256': digest, 'staged': str(path),
                       'status': present[0].parent.name if present else 'published' if publish else 'staged'})
    return result


def csv_text(rows):
    stream = io.StringIO(newline='')
    fields = sorted({key for row in rows for key in row})
    if fields:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return stream.getvalue()


def flat_row(row):
    return {**{key: row[key] for key in ('method', 'res', 'prompt_id', 'seed', 'path')},
            **row['metrics'], **{f'eval_seconds_{key}': value for key, value in row['eval_seconds'].items()}}


def row_identity(row):
    return str(row['method']), str(row['prompt_id']), int(row['seed']), str(row['res']), str(Path(row['path']).resolve())


def require_csv_matches(path, rows):
    # Compare parsed cells, retaining exact scalar strings while permitting CRLF.
    expected = list(csv.DictReader(io.StringIO(csv_text(rows))))
    with path.open(newline='') as handle:
        actual = list(csv.DictReader(handle))
    if actual != expected:
        raise ValueError(f'CSV disagrees with evaluator report: {path}')


def validate_shard(plan, shard):
    out = Path(shard['out'])
    report = json.loads((out / 'metrics.json').read_text())
    if report.get('complete') is not True or report.get('errors') or report.get('optional_unavailable'):
        raise ValueError(f'Evaluator shard is incomplete: {out}')
    expected_config = dict(root=shard['root'], out=shard['out'], prompts=plan['prompts'],
        references=plan['references'], reference_method=shard['reference_method'], metrics=METRICS,
        res='4096x4096', device='cuda', threads=8, bootstrap=1000, bootstrap_seed=0, crop_seed=0, crops=4)
    for key, value in expected_config.items():
        if report.get('config', {}).get(key) != value:
            raise ValueError(f'Evaluator setting {key} differs: {out}')
    for metric in METRICS.split(','):
        if report.get('status', {}).get(metric, {}).get('status') != 'complete':
            raise ValueError(f'Evaluator metric {metric} is incomplete: {out}')
    expected = {row_identity(row) for row in shard['samples']}
    rows = report['per_image']
    actual = [row_identity(row) for row in rows]
    if len(set(actual)) != len(actual):
        raise ValueError(f'Duplicate evaluator rows: {out}')
    if set(actual) != expected:
        raise ValueError(f'Evaluator rows differ from generation manifest: {out}')
    required = set(ANALYSIS_METRICS) - {'imagereward'}
    for row in rows:
        if not required.issubset(row['metrics']) or any(not isinstance(row['metrics'][key], (int, float))
                or not math.isfinite(row['metrics'][key]) for key in required):
            raise ValueError(f'Missing or nonfinite registered metric: {out}')
    summaries = report['summary']
    summary_keys = [(row['method'], row['res'], row['metric']) for row in summaries]
    expected_keys = {(method, '4096x4096', metric) for method in shard['methods']
                     for metric in required | {'fid', 'kid', 'patch_fid', 'patch_kid'}}
    if len(set(summary_keys)) != len(summary_keys) or not expected_keys.issubset(summary_keys):
        raise ValueError(f'Missing or duplicate evaluator summaries: {out}')
    counts = {method: len({sample['prompt_id'] for sample in shard['samples'] if sample['method'] == method})
              for method in shard['methods']}
    for row in summaries:
        if row['method'] not in counts or row['res'] != '4096x4096' or row['n_prompts'] != counts[row['method']]:
            raise ValueError(f'Evaluator summary population differs: {out}')
        if any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key])
               for key in ['value', 'ci_low', 'ci_high']) or row['ci_low'] > row['ci_high']:
            raise ValueError(f'Invalid evaluator summary interval: {out}')
    if len(shard['methods']) == 1 and report.get('paired_deltas'):
        raise ValueError(f'Single-method shard contains paired aggregates: {out}')
    require_csv_matches(out / 'per_image.csv', [flat_row(row) for row in rows])
    require_csv_matches(out / 'summary.csv', summaries)
    require_csv_matches(out / 'paired_deltas.csv', report['paired_deltas'])
    return report


def seal_output(plan_path, task_name, expected_plan_hash):
    """Called in the evaluator task after scoring; no GPU/model work here."""
    plan_path = Path(plan_path)
    if file_sha256(plan_path) != expected_plan_hash:
        raise ValueError('Evaluation plan changed after task publication')
    plan = json.loads(plan_path.read_text())
    shards = [shard for group in plan['groups'].values() for shard in group['shards'] if shard['name'] == task_name]
    if len(shards) != 1:
        raise ValueError('Unknown or ambiguous evaluator task in plan')
    shard = shards[0]
    validate_shard(plan, shard)
    receipt = dict(task=task_name, plan_sha256=expected_plan_hash,
                   outputs={name: file_sha256(Path(shard['out']) / name) for name in OUTPUT_FILES})
    immutable_text(Path(shard['out']) / 'evaluation_receipt.json', json.dumps(receipt, indent=2) + '\n')
    return receipt


def verify_dependencies(plan):
    checksums = Path(plan['dependencies'])
    if file_sha256(checksums) != plan['dependencies_sha256']:
        raise ValueError('Evaluation dependency receipt changed')
    for line in checksums.read_text().splitlines():
        digest, name = line.split('  ', 1)
        if file_sha256(name) != digest:
            raise ValueError(f'Evaluation dependency changed: {name}')


def merge_evaluation(manifest_path, queue=QUEUE, notes=NOTES):
    """One CPU-only attempt; missing shards leave canonical output untouched."""
    manifest_path, queue, notes = map(Path, [manifest_path, queue, notes])
    manifest = json.loads(manifest_path.read_text())
    directory = notes / manifest['block']
    plan_path = directory / 'evaluation_plan.json'
    if not plan_path.exists():
        return dict(block=manifest['block'], complete=False, expected_images=manifest['n_generate'],
                    reason='generation_not_validated_or_evaluation_not_staged')
    plan = json.loads(plan_path.read_text())
    tasks = json.loads((directory / 'evaluation_tasks.json').read_text())
    plan_hash = file_sha256(plan_path)
    if tasks['plan_sha256'] != plan_hash or plan['manifest_sha256'] != file_sha256(manifest_path):
        raise ValueError('Frozen evaluation plan or generation manifest changed')
    task_hashes = tasks['tasks']
    result = dict(block=manifest['block'], complete=True, expected_images=manifest['n_generate'], groups={})
    checked_dependencies = False
    for label, group in plan['groups'].items():
        missing = [shard['name'] for shard in group['shards'] if not (queue / 'done' / shard['name']).is_file()]
        if missing:
            result['groups'][label] = dict(complete=False, unfinished_tasks=missing)
            result['complete'] = False
            continue
        if not checked_dependencies:
            verify_dependencies(plan)
            checked_dependencies = True
        reports, receipts = [], []
        common = None
        for shard in group['shards']:
            if file_sha256(queue / 'done' / shard['name']) != task_hashes[shard['name']]:
                raise ValueError(f'Completed evaluator task changed: {shard["name"]}')
            out = Path(shard['out'])
            receipt_path = out / 'evaluation_receipt.json'
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('task') != shard['name'] or receipt.get('plan_sha256') != plan_hash:
                raise ValueError(f'Evaluator output receipt identity differs: {out}')
            if receipt.get('outputs') != {name: file_sha256(out / name) for name in OUTPUT_FILES}:
                raise ValueError(f'Evaluator output hash differs: {out}')
            report = validate_shard(plan, shard)
            signature = {key: report[key] for key in ['metric_schema', 'versions', 'model_revisions', 'status']}
            if common is not None and signature != common:
                raise ValueError(f'Evaluator shards have different metric implementations or references: {out}')
            common = signature
            reports.append(report)
            receipts.append(dict(receipt, task_sha256=task_hashes[shard['name']],
                                 receipt=str(receipt_path), receipt_sha256=file_sha256(receipt_path)))
        canonical = Path(group['out'])
        if group['sharded']:
            prompt_order = {prompt_id: index for index, prompt_id in enumerate(plan['prompt_order'])}
            order = {row_identity(row): index for index, row in enumerate(sorted(group['samples'],
                key=lambda sample: (sample['method'], prompt_order[sample['prompt_id']], sample['seed'])))}
            rows = sorted([row for report in reports for row in report['per_image']], key=lambda row: order[row_identity(row)])
            if len({row_identity(row) for row in rows}) != len(rows) or len(rows) != len(group['samples']):
                raise ValueError('Duplicate or missing rows across evaluator shards')
            summaries = sorted([row for report in reports for row in report['summary']],
                               key=lambda row: (row['method'], row['res'], row['metric']))
            merged = dict(common, config={'merge': 'method_shards', 'evaluation_plan': str(plan_path)},
                complete=True, all_metrics_available=True, errors=[], optional_unavailable=[],
                summary=summaries, per_image=rows, paired_deltas=[],
                uncertainty=reports[0]['uncertainty'],
                paired_note='Cross-method paired aggregates are not reconstructed from marginal summaries. '
                            'Registered paired scalar analyses use strengthen_analysis.py.',
                seconds=sum(report['seconds'] for report in reports),
                merge_provenance=dict(plan_sha256=plan_hash, shards=receipts,
                    fid_kid='Per-method estimates and intervals copied unchanged; no pooling or recomputation.'))
            # Every shard and source passes before any canonical score is written.
            # Repeated merges must be byte-identical; they never replace old evidence.
            outputs = {'per_image.csv': csv_text([flat_row(row) for row in rows]),
                       'summary.csv': csv_text(summaries), 'paired_deltas.csv': '',
                       'metrics.json': json.dumps(merged, indent=2, allow_nan=False) + '\n'}
            for name, body in outputs.items():
                if (canonical / name).exists() and (canonical / name).read_text() != body:
                    raise ValueError(f'Existing canonical output differs: {canonical / name}')
            for name, body in outputs.items():
                immutable_text(canonical / name, body)
        receipt = dict(plan_sha256=plan_hash, shards=receipts, rows=len(group['samples']),
                       canonical_outputs={name: file_sha256(canonical / name) for name in OUTPUT_FILES})
        immutable_text(canonical / 'merge_receipt.json', json.dumps(receipt, indent=2) + '\n')
        result['groups'][label] = dict(complete=True, rows=len(group['samples']), sharded=group['sharded'],
                                      merge_receipt=str(canonical / 'merge_receipt.json'))
    write_json(directory / 'merge_status.json', result)
    return result


def prepare(manifest_path, references, publish=False, queue=QUEUE, notes=NOTES,
            eval_base=SCRATCH / 'eval_krea/strengthen'):
    manifest_path, references, queue, notes = map(Path, [manifest_path, references, queue, notes])
    manifest = json.loads(manifest_path.read_text())
    samples = expected_samples(manifest)
    receipt_dir = notes / manifest['block']
    status = completion_status(manifest, samples, queue)
    status.update(timestamp=now(), block=manifest['block'], manifest_sha256=file_sha256(manifest_path),
                  validation='pending' if status['complete'] else 'waiting_for_generation')
    write_json(receipt_dir / 'status.json', status)
    if not status['complete']:
        return status
    try:
        witnesses = verify_samples(manifest, samples)
        dependencies = dependency_hashes(manifest_path, manifest, references, samples, queue)
    except Exception as error:
        status.update(complete=False, validation='failed', validation_error=str(error))
        write_json(receipt_dir / 'status.json', status)
        raise
    checksums = receipt_dir / 'dependencies.sha256'
    immutable_text(checksums, ''.join(f'{digest}  {path}\n' for path, digest in sorted(dependencies.items())))
    immutable_text(receipt_dir / 'validated_images.jsonl', ''.join(json.dumps(row, sort_keys=True) + '\n' for row in witnesses))
    plan_path = receipt_dir / 'evaluation_plan.json'
    plan = dict(schema=1, block=manifest['block'], manifest=str(manifest_path.resolve()),
                manifest_sha256=file_sha256(manifest_path), prompts=manifest['prompts'], references=str(references),
                prompt_order=[row['id'] for row in read_prompts(manifest['prompts'])],
                dependencies=str(checksums), dependencies_sha256=file_sha256(checksums),
                groups=evaluation_groups(manifest, samples, eval_base))
    immutable_text(plan_path, json.dumps(plan, indent=2) + '\n')
    scripts = task_bodies(manifest, samples, references, checksums, Path(eval_base), plan_path)
    task_record = dict(plan_sha256=file_sha256(plan_path),
                       tasks={name: hashlib.sha256(body.encode()).hexdigest() for name, body in scripts.items()})
    immutable_text(receipt_dir / 'evaluation_tasks.json', json.dumps(task_record, indent=2) + '\n')
    stage = queue / 'staging/strengthen/evaluation' / manifest['block']
    tasks = publish_scripts(scripts, stage, queue, publish)
    status.update(validated_images=len(witnesses), tasks=tasks, published=publish, validation='passed',
                  dependencies_sha256=file_sha256(checksums))
    write_json(receipt_dir / 'status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--references')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--merge', action='store_true', help='CPU-only merge/validation after evaluator shards finish')
    parser.add_argument('--seal-output', metavar='TASK_NAME', help=argparse.SUPPRESS)
    parser.add_argument('--plan-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seal_output:
        if not args.plan_sha256 or args.publish or args.merge:
            parser.error('--seal-output requires --plan-sha256 and excludes --publish/--merge')
        seal_output(args.manifest, args.seal_output, args.plan_sha256)
        return 0
    if args.merge:
        if args.publish:
            parser.error('--merge and --publish cannot be combined')
        report = merge_evaluation(args.manifest)
    else:
        if not args.references:
            parser.error('--references is required to stage or publish evaluation')
        report = prepare(args.manifest, args.references, args.publish)
    print(json.dumps({k: report[k] for k in ['block', 'complete', 'expected_images']}))
    return 0 if report['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
