#!/usr/bin/env python3
"""Let published 90_ evaluation tasks run before pending generation, without touching workers.

Workers claim the lexically first script in taskq/pending, so a 90_ evaluation task waits
behind every pending 77_-82_ generation task. `hold` moves the pending generation scripts
with the given prefixes into taskq/prio_hold/<id>/ and only then writes the release task
91_release_prio_hold_<id>.sh into pending. That task sorts after every 90_ task, so a worker
claims it only once all pending evaluations have been claimed. It moves the held scripts
back with `mv -n` and exits; if any script cannot be moved it exits nonzero, which leaves the
task in taskq/failed for inspection. Nothing is deleted, renamed or rewritten, and the set
of tasks is unchanged. `release` performs the same move from the controller (recovery).

Usage:
  strengthen_prio_eval.py hold --prefixes 77 80 81 82
  strengthen_prio_eval.py release [--id ID]
  strengthen_prio_eval.py status
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
QUEUE = Path('/scratch/09032/saini_2/constancy_rope/taskq')
RECEIPTS = ROOT / '.agent/notes/strengthen/prio_hold'

RELEASE = """#!/usr/bin/env bash
# Release task written by scripts/strengthen_prio_eval.py; moves held generation scripts back.
set -uo pipefail
Q={queue}
H={hold}
n=0; bad=0
for f in "$H"/*.sh; do
  [ -e "$f" ] || continue
  b=$(basename "$f")
  if [ -e "$Q/pending/$b" ] || [ -e "$Q/done/$b" ] || [ -e "$Q/failed/$b" ] || compgen -G "$Q/running/$b.*" > /dev/null; then
    echo "collision $b"; bad=$((bad+1)); continue
  fi
  if mv -n "$f" "$Q/pending/$b"; then n=$((n+1)); else bad=$((bad+1)); fi
done
left=$(find "$H" -maxdepth 1 -name '*.sh' | wc -l)
echo "$(date +%FT%T) released $n scripts from $H; left $left; problems $bad"
[ "$left" -eq 0 ] && [ "$bad" -eq 0 ]
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def pending_release(queue):
    return sorted(p.name for p in (queue / 'pending').glob('91_release_prio_hold_*.sh'))


def hold(prefixes, queue):
    pending = queue / 'pending'
    if pending_release(queue):
        raise SystemExit(f'A release task is still pending: {pending_release(queue)}')
    for old in (queue / 'prio_hold').glob('*'):
        if old.is_dir() and any(old.glob('*.sh')):
            raise SystemExit(f'Hold {old} still has scripts; release it first')
    names = sorted(os.listdir(pending))
    evals = [n for n in names if n.startswith('90_')]
    if not evals:
        raise SystemExit('No pending 90_ task; nothing to prioritise')
    chosen = [n for n in names if n.split('_', 1)[0] in prefixes and n.endswith('.sh')]
    if not chosen:
        raise SystemExit('No pending generation task with these prefixes')
    hid = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = queue / 'prio_hold' / hid
    target.mkdir(parents=True)
    moved = []
    for name in chosen:
        try:
            os.rename(pending / name, target / name)  # fails harmlessly if a worker claimed it
            moved.append(name)
        except FileNotFoundError:
            pass
    script = pending / f'91_release_prio_hold_{hid}.sh'
    tmp = queue / 'prio_hold' / f'.{script.name}.tmp'
    tmp.write_text(RELEASE.format(queue=queue, hold=target))
    subprocess.run(['bash', '-n', str(tmp)], check=True)
    os.rename(tmp, script)
    receipt = {'id': hid, 'created_utc': now(), 'queue': str(queue), 'hold_dir': str(target),
               'release_task': script.name, 'prefixes': sorted(prefixes), 'n_held': len(moved),
               'held': moved, 'evals_pending_at_hold': evals}
    if queue == QUEUE:
        RECEIPTS.mkdir(parents=True, exist_ok=True)
        (RECEIPTS / f'{hid}.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('id', 'n_held', 'release_task', 'evals_pending_at_hold')}))
    return receipt


def release(queue, hid=None):
    dirs = [queue / 'prio_hold' / hid] if hid else sorted(p for p in (queue / 'prio_hold').glob('*') if p.is_dir())
    total = 0
    for d in dirs:
        for f in sorted(d.glob('*.sh')):
            dest = queue / 'pending' / f.name
            if dest.exists() or (queue / 'done' / f.name).exists() or list((queue / 'running').glob(f.name + '.*')):
                print('collision', f.name)
                continue
            os.rename(f, dest)
            total += 1
    print(json.dumps({'released': total, 'dirs': [str(d) for d in dirs], 'time': now()}))


def status(queue):
    out = {'release_pending': pending_release(queue), 'holds': {}}
    for d in sorted((queue / 'prio_hold').glob('*')):
        if d.is_dir():
            out['holds'][d.name] = len(list(d.glob('*.sh')))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('hold', 'release', 'status'))
    p.add_argument('--prefixes', nargs='+', default=['77', '80', '81', '82'])
    p.add_argument('--id')
    p.add_argument('--queue', type=Path, default=QUEUE, help='Test queues only')
    a = p.parse_args()
    if a.action == 'hold':
        hold(set(a.prefixes), a.queue)
    elif a.action == 'release':
        release(a.queue, a.id)
    else:
        status(a.queue)
