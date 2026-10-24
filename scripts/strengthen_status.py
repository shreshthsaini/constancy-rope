#!/usr/bin/env python3
"""One-shot fleet/spool observation; enforce at least ten minutes between scheduler polls."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / '.agent/notes/strengthen'
QUEUE = Path('/scratch/09032/saini_2/constancy_rope/taskq')


def observe(scheduler=False, brief=False, checkpoint=False):
    now = datetime.now(timezone.utc)
    livepath = NOTES / 'live.json'
    live = json.loads(livepath.read_text())
    old = live.get('scheduler', {})
    if scheduler:
        due = datetime.fromisoformat(old.get('next_poll_not_before', '2000-01-01T00:00:00Z'))
        if now < due:
            print(f'Scheduler poll deferred until {due.isoformat()}')
        else:
            raw = subprocess.check_output(['squeue', '-j', '1080046', '-h', '-o', '%i|%j|%T|%M|%D|%R|%S'], text=True).strip()
            parts = raw.split('|')
            old = dict(job_id='1080046', state=parts[2] if raw else 'NOT_IN_SQUEUE',
                       observed=now.isoformat(timespec='seconds'),
                       next_poll_not_before=(now + timedelta(minutes=10)).isoformat(timespec='seconds'),
                       scheduler_start_estimate=parts[6] if len(parts) > 6 else None, raw=raw)
            live['scheduler'] = old
            livepath.write_text(json.dumps(live, indent=2) + '\n')
    counts = {}
    for path in sorted(NOTES.glob('*_queue.json')):
        plan = json.loads(path.read_text())
        if not (NOTES / f"{plan['block']}_published.json").exists():
            continue
        c = {k: 0 for k in ['pending', 'running', 'done', 'failed', 'missing']}
        for task in plan['tasks']:
            name = task['name']
            stages = [s for s in ['pending', 'done', 'failed'] if (QUEUE / s / name).exists()]
            if list((QUEUE / 'running').glob(name + '.*')):
                stages.append('running')
            if len(stages) > 1:
                raise ValueError(f'Duplicate task locations: {name}: {stages}')
            c[stages[0] if stages else 'missing'] += 1
        counts[plan['block']] = c
    receipt = {'timestamp': now.isoformat(timespec='seconds'), 'scheduler': old, 'blocks': counts}
    (NOTES / 'queue_status.json').write_text(json.dumps(receipt, indent=2) + '\n')
    totals = {stage: sum(block[stage] for block in counts.values())
              for stage in ['pending', 'running', 'done', 'failed', 'missing']}
    summary = ', '.join(f'{count} {stage}' for stage, count in totals.items())
    if brief:
        print(f"CR22 {old.get('state', 'unobserved')}; observed {old.get('observed')}; {summary}.")
        print(f"Next scheduler poll: {old.get('next_poll_not_before')}.")
    else:
        print(json.dumps(receipt, indent=2))
    if checkpoint:
        save = runpy.run_path(str(NOTES / 'checkpoint.py'))['checkpoint']
        save(title='Scheduled fleet observation',
             details=f"CR22 state {old.get('state')}, observed {old.get('observed')}. "
                     f"Published generation task counts: {summary}. Scheduler response: {old.get('raw')}. "
                     'This observation does not establish image quality or a hypothesis outcome.',
             status=f"CR22 {old.get('state')} at {old.get('observed')}. {summary}. "
                    'Preparation and the 82-test integration checks are complete. '
                    'See queue_status.json and live.json for the current observation; '
                    'final results and paper release depend on complete verified outputs.',
             next_action=f"Next scheduler query no earlier than {old.get('next_poll_not_before')}. "
                         'Inspect new smoke/output/failure receipts. Release full Qwen only after a valid pass; '
                         'evaluate complete blocks, merge shards and run the registered analyses. '
                         'Sleep while the fleet is pending.')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scheduler', action='store_true')
    parser.add_argument('--brief', action='store_true')
    parser.add_argument('--checkpoint', action='store_true', help='Refresh STATE, HANDOVER and results log')
    args = parser.parse_args()
    observe(args.scheduler, args.brief, args.checkpoint)
