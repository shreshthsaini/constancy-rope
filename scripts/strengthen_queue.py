#!/usr/bin/env python3
"""Prepare and publish frozen Krea confirmation tasks to the existing spool.

No scheduler commands. Existing seed-1 samples are validated and excluded from
generation, with a receipt. Run fresh before seeds. Publication is explicit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crope.utils import file_sha256, image_seed, image_stem, read_prompts

SCR = Path('/scratch/09032/saini_2/constancy_rope')
QUEUE = SCR / 'taskq'
NOTES = ROOT / '.agent/notes/strengthen'
METHODS = ('sega_official', 'temp_only_sbase', 'ntk2', 'dype', 'sega_base_none', 'ntk2_sref')
SEED_METHODS = ('sega_official', 'temp_only_sbase')
ARMS = {
    'sega_official': ('sega_official', None, None, None),
    'ntk2': ('ntk2', None, None, None),
    'dype': ('dype', None, None, None),
    'temp_only_sbase': ('temp_only', 'sega_official', 'sref', None),
    'sega_base_none': ('temp_only', 'sega_official', 'none', None),
    'ntk2_sref': ('temp_only', None, 'sref', None),
    'tau1p1': ('temp_only', 'sega_official', 'none', 1.1),
    'tau_logn': ('temp_only', 'sega_official', 'none', 4 / 3),
    'tau1p45': ('temp_only', 'sega_official', 'none', 1.45),
}
TEMPLATES = {
    'sega_official': '50_c00000_main_sega_official_*.sh',
    'dype': '50_c00000_main_dype_*.sh',
    'ntk2': '50_c00000_main_ntk2_*.sh',
    'temp_only': '52_c00007_main_tsb.sh',
    'constancy_tm': '52_c00007_main_ctmsb.sh',
    'sega_ctm': '53_c00007_main2_sega_ctm_eta10.sh',
}
BASE_ROOT = dict(sega_official='main', temp_only='mainsb', constancy_tm='mainsb', sega_ctm='main2')


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def immutable_json(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(text)


def normalized_config(config):
    c = json.loads(json.dumps(config))
    # Old outputs predate the current commit and recorded first-image scheduler
    # bookkeeping before max_image_seq_len was set. Executed sigmas are checked.
    for key in ('prompt_id', 'prompt', 'seed', 'derived_seed', 'code_revision', 'implementation_sha256'):
        c.pop(key, None)
    for key in ('max_image_seq_len', '_use_default_values'):
        c['scheduler'].pop(key, None)
    return c


def validate_sample(sidecar, row, method, seed, reference):
    """Strict artifact validation; config hashes do not certify Krea by themselves.

    Krea ancestry also requires the archived task's CROPE_TRANSFORMER export.
    """
    from PIL import Image
    record = json.loads(sidecar.read_text())
    c = record['config']
    expected = dict(method=method, prompt_id=row['id'], prompt=row['prompt'], seed=seed,
                    derived_seed=image_seed(row['id'], seed), steps=28, guidance=4.5, res='4096x4096')
    for key, value in expected.items():
        if c.get(key) != value:
            raise ValueError(f'{sidecar}: {key} mismatch')
    if not record.get('complete') or normalized_config(c) != normalized_config(reference['config']):
        raise ValueError(f'{sidecar}: incomplete or settings differ from seed-0 baseline')
    fingerprint = hashlib.sha256(json.dumps(c, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if fingerprint != record['config_fingerprint']:
        raise ValueError(f'{sidecar}: config fingerprint differs')
    sigmas = record['sigmas']
    if len(sigmas) != 28 or len(record['gains']) != 28 or not all(math.isfinite(x) for x in sigmas):
        raise ValueError(f'{sidecar}: invalid denoising record')
    if sigmas != reference['sigmas']:
        raise ValueError(f'{sidecar}: actual noise schedule differs')
    png = sidecar.with_suffix('.png')
    if file_sha256(png) != record['png_sha256']:
        raise ValueError(f'{png}: checksum differs')
    with Image.open(png) as image:
        if image.size != (4096, 4096):
            raise ValueError(f'{png}: wrong dimensions')
        image.verify()
    return dict(method=method, prompt_id=row['id'], seed=seed, path=str(png), sidecar=str(sidecar),
                png_sha256=record['png_sha256'], sidecar_sha256=file_sha256(sidecar),
                config=c, sigmas=sigmas)


def reusable_seed1(rows):
    records = []
    for arm in SEED_METHODS:
        method = ARMS[arm][0]
        refpath = SCR / 'out_krea' / BASE_ROOT[method] / method / '4096x4096'
        reference = json.loads(next(refpath.glob('*.json')).read_text())
        for row in rows[8:195]:
            sidecar = SCR / 'out_krea/seed1' / method / '4096x4096' / (image_stem(row['id'], 1) + '.json')
            if sidecar.exists():
                record = validate_sample(sidecar, row, method, 1, reference)
                record['method'] = arm
                records.append(record)
    return records


def ranges(indices):
    """Contiguous intervals without regenerating a previously completed row."""
    if not indices:
        return []
    result, lo, hi = [], indices[0], indices[0] + 1
    for index in indices[1:]:
        if index == hi:
            hi += 1
        else:
            result.append((lo, hi))
            lo, hi = index, index + 1
    return result + [(lo, hi)]


def template(method):
    matches = list((QUEUE / 'done').glob(TEMPLATES[method]))
    if len(matches) != 1:
        raise ValueError(f'Expected one archived template for {method}: {matches}')
    path = matches[0]
    lines = path.read_text().splitlines()
    sample = [line for line in lines if ' scripts/sample.py ' in line]
    if len(sample) != 1:
        raise ValueError(f'Unexpected template command: {path}')
    return path, lines, shlex.split(sample[0])


def task_body(method, seed, lo, hi, prompts, out, checksum_file):
    controller, rope, temperature, multiplier = ARMS[method]
    path, lines, command = template(controller)
    for flag, value in {'--prompts': str(prompts), '--start': str(lo), '--end': str(hi),
                        '--seed': str(seed), '--out': str(out)}.items():
        command[command.index(flag) + 1] = value
    if command[command.index('--method') + 1] != controller:
        raise ValueError('Archived template method differs')
    if temperature is not None:
        if '--temperature' in command:
            command[command.index('--temperature') + 1] = temperature
        else:
            command += ['--temperature', temperature]
    if multiplier is not None:
        command += ['--temperature-multiplier', repr(multiplier)]
    if method != controller:
        command += ['--output-method', method]
    # Keep the archived environment and controller flags, replacing stale
    # prerequisite lines and fingerprints with the new frozen manifest.
    body = [line for line in lines if not line.startswith('[[') and
            'scripts/sample.py' not in line and 'fingerprint:' not in line and
            not line.startswith('export CROPE_ROPE=')]
    body += ['unset CROPE_ROPE']
    if rope:
        body += [f'export CROPE_ROPE={rope}']
    body += [f'# Archived template: {path}; sha256 {file_sha256(path)}',
             f'sha256sum --status -c {shlex.quote(str(checksum_file))}', shlex.join(command)]
    return '\n'.join(body) + '\n', path, command


def prepare(block):
    NOTES.mkdir(parents=True, exist_ok=True)
    manifest = NOTES / f'{block}_queue.json'
    if manifest.exists():
        raise FileExistsError(f'Frozen manifest exists: {manifest}; use publish')
    if block in ('factorial', 'tsweep'):
        prompts = SCR / 'prompts/aesthetic4k_eval.jsonl'
        rows = read_prompts(prompts)
        methods = ('sega_base_none', 'ntk2_sref') if block == 'factorial' else ('tau1p1', 'tau_logn', 'tau1p45')
        seeds, start, end, prefix = (0,), 8, (195 if block == 'factorial' else 63), ('75' if block == 'factorial' else '77')
        reuse = []
    elif block == 'fresh':
        prompts = ROOT / 'prompts/fresh150.jsonl'
        rows = read_prompts(prompts)
        if len(rows) != 150:
            raise ValueError('Fresh block requires exactly150 prompts')
        methods, seeds, start, end, prefix = METHODS, (0,), 0, 150, '80'
        reuse = []
    else:
        first = NOTES / 'fresh_queue.json'
        if not first.exists() or not (NOTES / 'fresh_published.json').exists():
            raise ValueError('Queue E1 before preparing/publishing E2')
        prompts = SCR / 'prompts/aesthetic4k_eval.jsonl'
        rows = read_prompts(prompts)
        methods, seeds, start, end, prefix = SEED_METHODS, (1, 2), 8, 195, '81'
        reuse = reusable_seed1(rows)
        immutable_json(NOTES / 'seed1_reuse.json', {'timestamp': now(), 'records': reuse,
            'n': len(reuse), 'ancestry': 'Archived generation tasks export pinned Krea transformer; '
            'old sidecars name the FLUX.1-dev pipeline and do not separately name that transformer.'})
    reused = {(r['method'], r['prompt_id'], r['seed']) for r in reuse}
    sources = [ROOT / 'crope' / name for name in ('__init__.py', 'runtime.py', 'gain.py', 'rope.py',
               'sega_official.py', 'spectral.py', 'utils.py')]
    sources += [ROOT / 'scripts' / name for name in ('sample.py', 'calibrate.py', 'probe_mtf.py')]
    sources += [SCR / 'calib/krea_profile_tm.npz', prompts]
    checksum = NOTES / f'{block}_generation.sha256'
    checksum.write_text(''.join(f'{file_sha256(p)}  {p}\n' for p in sources))
    tasks = []
    staging = QUEUE / 'staging/strengthen' / block
    staging.mkdir(parents=True, exist_ok=True)
    for chunk_start in range(start, end, 7):
        chunk_end = min(end, chunk_start + 7)
        for mi, method in enumerate(methods):
            for seed in seeds:
                missing = [i for i in range(chunk_start, chunk_end)
                           if (method, rows[i]['id'], seed) not in reused]
                for lo, hi in ranges(missing):
                    out = SCR / 'out_krea' / block / method
                    if block == 'seeds':
                        out = SCR / 'out_krea' / block / f'seed{seed}' / method
                    out /= '4096x4096'
                    text, source, command = task_body(method, seed, lo, hi, prompts, out, checksum)
                    name = f'{prefix}_c{chunk_start:05d}_m{mi}_{method}_seed{seed}_{lo:05d}-{hi:05d}.sh'
                    path = staging / name
                    with path.open('x') as f:
                        f.write(text)
                    subprocess.run(['bash', '-n', str(path)], check=True)
                    tasks.append(dict(name=name, method=method, seed=seed, start=lo, end=hi,
                        n=hi-lo, prompt_ids=[r['id'] for r in rows[lo:hi]], out=str(out),
                        staged=str(path), sha256=file_sha256(path), source_template=str(source),
                        source_template_sha256=file_sha256(source), command=command))
    immutable_json(manifest, dict(timestamp=now(), block=block, prompts=str(prompts),
        prompts_sha256=file_sha256(prompts), tasks=tasks, n_tasks=len(tasks),
        n_generate=sum(t['n'] for t in tasks), n_reuse=len(reuse), checksum_file=str(checksum),
        sources={str(p): file_sha256(p) for p in sources},
        prereg_sha256_at_queue=file_sha256(ROOT / '.agent/notes/prereg_strengthen.md'),
        arms={m: dict(controller=ARMS[m][0], rope=ARMS[m][1], temperature=ARMS[m][2],
                     temperature_multiplier=ARMS[m][3]) for m in methods}))
    print(json.dumps(dict(block=block, tasks=len(tasks), generate=sum(t['n'] for t in tasks), reuse=len(reuse))))


def publish(block):
    manifest = NOTES / f'{block}_queue.json'
    plan = json.loads(manifest.read_text())
    if not (NOTES / 'amendment_receipt.json').exists():
        raise ValueError('Step0 senior-review amendment required before publication')
    if (NOTES / f'{block}_published.json').exists():
        raise FileExistsError('Already published; inspect queues rather than duplicate tasks')
    if (QUEUE / 'STOP').exists():
        raise ValueError('STOP exists; preserve it by moving aside before publication')
    for path, expected in plan['sources'].items():
        if file_sha256(path) != expected:
            raise ValueError(f'Frozen source changed: {path}')
    for task in plan['tasks']:
        staged = Path(task['staged'])
        if file_sha256(staged) != task['sha256']:
            raise ValueError(f'Staged task changed: {staged}')
        for stage in ('pending', 'done', 'failed'):
            if (QUEUE / stage / task['name']).exists():
                raise FileExistsError(task['name'])
        if list((QUEUE / 'running').glob(task['name'] + '.*')):
            raise FileExistsError(task['name'])
    for task in plan['tasks']:
        os.rename(task['staged'], QUEUE / 'pending' / task['name'])
    immutable_json(NOTES / f'{block}_published.json', dict(timestamp=now(),
        manifest_sha256=file_sha256(manifest), tasks=plan['n_tasks'], images=plan['n_generate']))
    print(f"Published {plan['n_tasks']} {block} tasks, {plan['n_generate']} images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'publish'))
    parser.add_argument('block', choices=('factorial', 'tsweep', 'fresh', 'seeds'))
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else publish)(args.block)
