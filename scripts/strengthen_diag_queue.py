#!/usr/bin/env python3
"""Stage frozen P2/P7 tasks; publication remains a separate main-agent action."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crope.utils import file_sha256, read_prompts
from scripts.strengthen_diagnostics import method_name, model_paths, source_provenance

SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
QUEUE = SCRATCH / 'taskq'
NOTES = ROOT / '.agent/notes/strengthen'
PYTHON = '/work/09032/saini_2/vista/library/uv-envs/crope/bin/python'


def immutable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(block):
    manifest = NOTES / f'{block}_queue.json'
    if manifest.exists():
        raise FileExistsError(f'Frozen manifest already exists: {manifest}')
    receipt = NOTES / 'amendment_receipt.json'
    amendment = json.loads(receipt.read_text())
    if not amendment.get('no_generation_queued'):
        raise ValueError('Require pre-generation amendment receipt')
    prereg = ROOT / '.agent/notes/prereg_strengthen.md'
    templates = list((QUEUE / 'done').glob('50_c00000_main_sega_official_*.sh'))
    if len(templates) != 1:
        raise ValueError('Expected one archived SEGA main template')
    template = templates[0]
    env = [line for line in template.read_text().splitlines()
           if not line.startswith('[[') and 'scripts/sample.py' not in line
           and 'fingerprint:' not in line and not line.startswith('export CROPE_ROPE=')]
    env += ['unset CROPE_ROPE']
    prompts = SCRATCH / 'prompts/aesthetic4k_eval.jsonl'
    rows = read_prompts(prompts)
    if len(rows) < 8:
        raise ValueError('Expected full development prompt manifest')
    sources = [Path(__file__), ROOT / 'crope/utils.py', prompts, template, receipt]
    if block == 'ckpt_matrix':
        official = SCRATCH / 'third_party/DyPE'
        source_provenance(official)
        sources += [ROOT / 'scripts/strengthen_diagnostics.py']
        sources += sorted((official / 'flux').rglob('*.py'))
        for checkpoint in ('dev', 'krea'):
            base, transformer = model_paths(checkpoint)
            sources += [base / 'model_index.json', base / 'scheduler/scheduler_config.json',
                        transformer / 'transformer/config.json']
        env += ['unset CROPE_TRANSFORMER']
        cells = [(checkpoint, guidance) for checkpoint in ('dev', 'krea') for guidance in (3.5, 4.5)]
        prefix = '76'
    elif block == 'default_shift':
        sources += [ROOT / 'crope' / name for name in
                    ('__init__.py', 'runtime.py', 'gain.py', 'rope.py', 'sega_official.py', 'spectral.py')]
        sources += [ROOT / 'scripts' / name for name in ('sample.py', 'calibrate.py', 'probe_mtf.py')]
        cells, prefix = [(None, 4.5)], '78'
    else:
        raise ValueError(block)
    sources = list(dict.fromkeys(sources))
    checksum = NOTES / f'{block}_generation.sha256'
    with checksum.open('x') as stream:
        stream.write(''.join(f'{file_sha256(path)}  {path}\n' for path in sources))
    staging = QUEUE / 'staging/strengthen' / block
    staging.mkdir(parents=True, exist_ok=True)
    tasks = []
    for start, end in ((0, 4), (4, 8)):
        for cell, (checkpoint, guidance) in enumerate(cells):
            if block == 'ckpt_matrix':
                method = method_name(checkpoint, guidance)
                out = SCRATCH / 'diag/ckpt_matrix' / method / '4096x4096'
                seed = 42
                command = [PYTHON, 'scripts/strengthen_diagnostics.py', '--checkpoint', checkpoint,
                           '--guidance', str(guidance), '--start', str(start), '--end', str(end),
                           '--prompts', str(prompts), '--out', str(SCRATCH / 'diag/ckpt_matrix')]
            else:
                method, seed = 'sega_default_shift', 0
                out = SCRATCH / 'diag/default_shift' / method / '4096x4096'
                command = [PYTHON, 'scripts/sample.py', '--method', 'sega_official',
                           '--schedule-mode', 'diffusers_default', '--output-method', method,
                           '--res', '4096x4096', '--seed', '0', '--steps', '28', '--guidance', '4.5',
                           '--prompts', str(prompts), '--start', str(start), '--end', str(end), '--out', str(out)]
            name = f'{prefix}_c{start:05d}_m{cell}_{method}_{start:05d}-{end:05d}.sh'
            staged = staging / name
            body = env + [f'# Archived template: {template}; SHA256 {file_sha256(template)}',
                          f'sha256sum --status -c {shlex.quote(str(checksum))}', shlex.join(command)]
            with staged.open('x') as stream:
                stream.write('\n'.join(body) + '\n')
            subprocess.run(['bash', '-n', str(staged)], check=True)
            tasks.append({'name': name, 'method': method, 'seed': seed, 'start': start, 'end': end,
                          'n': end-start, 'prompt_ids': [row['id'] for row in rows[start:end]],
                          'out': str(out), 'staged': str(staged), 'sha256': file_sha256(staged),
                          'command': command, 'source_template': str(template),
                          'source_template_sha256': file_sha256(template)})
    immutable_json(manifest, {'timestamp': datetime.now(timezone.utc).isoformat(), 'block': block,
                   'prompts': str(prompts), 'prompts_sha256': file_sha256(prompts), 'tasks': tasks,
                   'n_tasks': len(tasks), 'n_generate': sum(task['n'] for task in tasks), 'n_reuse': 0,
                   'checksum_file': str(checksum), 'sources': {str(p): file_sha256(p) for p in sources},
                   'prereg_sha256_at_queue': file_sha256(prereg),
                   'amendment_receipt_sha256': file_sha256(receipt),
                   'status': 'staged only; not published'})
    print(json.dumps({'block': block, 'staged_tasks': len(tasks), 'manifest': str(manifest)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('block', choices=('ckpt_matrix', 'default_shift'))
    prepare(parser.parse_args().block)
