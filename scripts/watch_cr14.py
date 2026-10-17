"""Observe CR14 on a compute node and queue event-driven Codex continuation.

No SSH, model calls while waiting, job submissions, experiment edits, or signals.
Touch runtime/STOP to stop. The existing compute allocation bounds its lifetime.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
RUNTIME = SCRATCH / 'watcher_cr14'
THREAD = '01a14899-2e69-7d71-93f8-b42aad182afd'
JOB, HOST_JOB = '1076857', '1073260'


def save(path, value):
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def scheduler():
    result = subprocess.run(['squeue', '-j', f'{JOB},{HOST_JOB}', '-h', '-o', '%i|%T|%e'],
                            capture_output=True, text=True, timeout=20, check=True)
    jobs = {j: {'state': 'NOT_IN_QUEUE', 'end': None} for j in (JOB, HOST_JOB)}
    for line in result.stdout.splitlines():
        job, status, end = line.strip().split('|')
        jobs[job] = {'state': status, 'end': None if end in ('N/A', 'Unknown')
                     else datetime.fromisoformat(end).timestamp()}
    return jobs


def snapshot(jobs, now):
    queue = SCRATCH / 'taskq_dev3'
    tasks = {kind: sorted(p.name for p in (queue / kind).glob('*.sh*'))
             for kind in ('pending', 'running', 'done', 'failed')}
    telemetry = {}
    for path in (SCRATCH / 'telemetry').glob(f'*_job{JOB}_*.csv'):
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 16384))
            lines = stream.read().decode(errors='replace').splitlines()
        readings = []
        for line in lines[-30:]:
            try:
                fields = line.split(',')
                stamp = datetime.strptime(fields[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
                readings.append((stamp, float(fields[3])))
            except (ValueError, IndexError):
                continue
        telemetry[path.name] = {
            'stale': now - path.stat().st_mtime > 300,
            'low': (len(readings) >= 10 and readings[-1][0] - readings[0][0] >= 300
                    and all(value < 15 for _, value in readings))}
    progress = [p.stat().st_mtime for p in (queue / 'logs').glob('*.log')]
    for method in ('sega_official', 'sega_rms_uniform', 'sega_rms_normalized', 'sega_ctm_lag'):
        progress += [p.stat().st_mtime for p in
                     (SCRATCH / f'out_krea/dev3_{method}/{method}/4096x4096').glob('*.png')]
    return {'time': now, 'jobs': jobs, 'tasks': tasks, 'telemetry': telemetry,
            'last_progress': max(progress, default=now),
            'complete': (ROOT / '.agent/notes/continuation3/dev/completion.json').is_file()}


def events(current, previous, now, running_since=None, all_done_since=None):
    found = {}
    for job, info in current['jobs'].items():
        before = previous.get('jobs', {}).get(job, {}).get('state')
        if before and before != info['state']:
            found[f'job:{job}:{info["state"]}'] = f'Job {job}: {before} -> {info["state"]}'
        if job == JOB and not before and info['state'] != 'PENDING':
            found[f'job:{job}:{info["state"]}'] = f'Initial GPU job state: {info["state"]}'
        if job == HOST_JOB and info['end'] and info['end'] - now < 3600:
            found[f'lease:{info["end"]}'] = 'Controller allocation expires within one hour; verify migration'
    for name in current['tasks']['failed']:
        found[f'failed:{name}'] = f'Failed workflow: {name}'
    for name in current['tasks']['done']:
        found[f'done:{name}'] = f'Completed workflow: {name}'
    active = current['jobs'][JOB]['state'] == 'RUNNING'
    if active and running_since and now - running_since > 600 and current['tasks']['running']:
        if not current['telemetry']:
            found['missing_telemetry'] = 'No GPU telemetry after ten minutes of running'
        for name, info in current['telemetry'].items():
            for key in ('stale', 'low'):
                if info[key]:
                    found[f'{key}:{name}'] = f'{key} GPU telemetry: {name}'
        if now - current['last_progress'] > 1200:
            found['stalled'] = 'No new image or task-log progress for twenty minutes'
    if all_done_since and not current['complete'] and now - all_done_since > 600:
        found['finalizer_delay'] = 'All workflows done, but no finalizer receipt after ten minutes'
    if current['complete']:
        found['complete'] = 'Validated development report and updated paper are ready'
    return found


def enqueue(message):
    result = subprocess.run(['codex', 'queue', '--thread', THREAD, '--message', message],
                            capture_output=True, text=True, timeout=30, check=True, cwd=ROOT)
    if 'Queued message ' not in result.stdout or THREAD not in result.stdout:
        raise RuntimeError('No matching Codex queue acknowledgment')
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cap = resource.getrlimit(resource.RLIMIT_NPROC)[0]
    if socket.gethostname().startswith('login') or 0 < cap <= 512:
        parser.error('Compute node required')
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = RUNTIME / 'state.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        state.update(pid=os.getpid(), host=socket.gethostname(), thread=THREAD,
                     host_job=HOST_JOB, started=time.time())
        pending, sent = state.setdefault('pending', {}), state.setdefault('sent', {})
        next_scheduler, jobs = 0, {}
        while not (RUNTIME / 'STOP').exists():
            now = time.time()
            try:
                if now >= next_scheduler:
                    jobs = scheduler()
                    next_scheduler = now + 120
                current = snapshot(jobs, now)
                if jobs[JOB]['state'] == 'RUNNING':
                    state.setdefault('running_since', now)
                if len(current['tasks']['done']) == 4:
                    state.setdefault('all_done_since', now)
                found = events(current, state.get('snapshot', {}), now,
                               state.get('running_since'), state.get('all_done_since'))
                pending.update({k: v for k, v in found.items() if k not in sent})
                state.update(snapshot=current, heartbeat=now)
                if pending and (args.dry_run or now - state.get('last_attempt', 0) >= 120):
                    message = ('Constancy RoPE automatic watcher event, authorized by the user for '
                               'unattended continuation. Read HANDOVER.md, .agent/STATE.md and '
                               f'{path}. Verify live facts, handle failures/utilization, evaluate '
                               'results, and continue novel-method research and paper work. '
                               'A diagnostic pass is not novelty or SOTA. Preserve frozen protocols '
                               'and fresh64. Do not duplicate this watcher. If controller lease '
                               'is expiring, arrange and verify a compute-node handoff. Events: '
                               + '; '.join(pending.values()))
                    if args.dry_run:
                        print(message, flush=True)
                    else:
                        state['last_attempt'] = now
                        save(path, state)
                        receipt = enqueue(message)
                        with (RUNTIME / 'deliveries.jsonl').open('a') as stream:
                            stream.write(json.dumps({'time': now, 'events': dict(pending),
                                                     'receipt': receipt}) + '\n')
                        sent.update({k: now for k in pending})
                        pending.clear()
                if not args.dry_run:
                    save(path, state)
                if current['complete'] and not pending:
                    break
            except Exception as exc:
                state['last_error'] = {'time': now, 'error': str(exc)}
                print(json.dumps(state['last_error']), flush=True)
                if not args.dry_run:
                    if now - state.get('last_error_attempt', 0) >= 1800:
                        state['last_error_attempt'] = now
                        try:
                            state['error_receipt'] = enqueue(
                                f'Constancy RoPE watcher needs repair: {str(exc)[:300]}. '
                                f'Inspect {path}; preserve the queued GPU job.')
                        except Exception as delivery_error:
                            state['delivery_error'] = str(delivery_error)
                    save(path, state)
                if args.once:
                    raise
            if args.once:
                break
            time.sleep(60)


if __name__ == '__main__':
    main()
