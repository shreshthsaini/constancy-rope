"""Observe CR14 on a compute node and queue event-driven Codex continuation.

No SSH, model calls while waiting, job submissions, experiment edits, or signals.
Touch runtime/STOP to stop. The existing compute allocation bounds its lifetime.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path('/scratch/09032/saini_2/constancy_rope')
RUNTIME = SCRATCH / 'watcher_cr14'
THREAD = '01a14899-2e69-7d71-93f8-b42aad182afd'
JOB, HOST_JOB = '1076857', '1073260'
SOCKET = '/tmp/cx-883311-1ae2ac8f/home/app-server-control/app-server-control.sock'


def positive_job(value):
    if not isinstance(value, str) or not value.isascii() or not value.isdigit() or int(value) <= 0:
        raise argparse.ArgumentTypeError('Job ID must be a positive integer')
    return str(int(value))


def absolute_socket(value):
    if not isinstance(value, str) or not Path(value).is_absolute() or '\x00' in value:
        raise argparse.ArgumentTypeError('Socket must be an absolute path')
    return value


def thread_uuid(value):
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise argparse.ArgumentTypeError('Thread must be a UUID') from exc
    return str(parsed)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--host-job', type=positive_job, default=HOST_JOB)
    parser.add_argument('--socket', type=absolute_socket, default=SOCKET)
    parser.add_argument('--thread', type=thread_uuid, default=THREAD)
    return parser.parse_args(argv)


def destination_ready(now):
    """Read an operator-prepared handoff, never infer or allocate a destination."""
    path = RUNTIME / 'DESTINATION_READY.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    required = {'source_job', 'thread', 'destination_job', 'destination_node',
                'destination_socket', 'created'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('Invalid destination ready fields')
    # The destination keeps the same runtime, so the old ready record is inert there.
    if value['source_job'] != HOST_JOB:
        return None
    if positive_job(value['destination_job']) != value['destination_job'] or value['destination_job'] == HOST_JOB:
        raise ValueError('Destination must be a different canonical job ID')
    if thread_uuid(value['thread']) != THREAD or value['thread'] != THREAD:
        raise ValueError('Destination thread does not match this operator')
    absolute_socket(value['destination_socket'])
    node = value['destination_node']
    if (not isinstance(node, str) or not node or node.startswith('login')
            or not all(c.isascii() and (c.isalnum() or c in '.-') for c in node)
            or node == socket.gethostname()):
        raise ValueError('Destination must name another compute node')
    created = value['created']
    if (type(created) not in (int, float) or not math.isfinite(created)
            or not -60 <= now - created <= 3600):
        raise ValueError('Destination ready timestamp is stale or invalid')
    return value


def handoff_receipt(ready, jobs, now):
    """Caller supplies a scheduler result polled in this iteration, never a cache."""
    if ready is None or jobs.get(ready['destination_job'], {}).get('state') != 'RUNNING':
        return None
    with connect() as ws:
        rpc(ws, 'initialize', {'clientInfo': {'name': 'crope-watcher', 'version': '3'},
                              'capabilities': {'experimentalApi': True}})
        ws.send(json.dumps({'method': 'initialized'}))
        thread = rpc(ws, 'thread/read', {'threadId': THREAD, 'includeTurns': False})['thread']
        if thread['id'] != THREAD or Path(thread['cwd']) != ROOT:
            raise RuntimeError('Refusing handoff of a different project/thread')
        if thread['status']['type'] != 'idle':
            return None
    return dict(ready, source_node=socket.gethostname(), source_socket=SOCKET, released=now)


def registrations(path=None):
    """Hot-read registrations.json; absent/empty means no additional jobs.

    Schema: {"jobs": [{"job_id": "123", "queue_dir": "/scratch/.../taskq",
    "completion_files": [], "progress_dirs": [], "telemetry_dir": "..."}]}.
    Queue, telemetry and nonrecursive progress directories must be under this
    project's scratch root; completion files may also be under ROOT on work.
    Publish edits atomically. Invalid edits leave the last accepted set active.
    """
    path = path or RUNTIME / 'registrations.json'
    if not path.exists():
        return []
    raw = path.read_text().strip()
    data = json.loads(raw) if raw else {'jobs': []}
    if not isinstance(data, dict) or set(data) != {'jobs'} or not isinstance(data['jobs'], list):
        raise ValueError('Registration file must contain an object with a jobs list')

    def checked_path(value, roots):
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError('Registration paths must be absolute strings')
        resolved = Path(value).resolve()
        if not any(resolved != root.resolve() and resolved.is_relative_to(root.resolve())
                   for root in roots):
            raise ValueError(f'Registration path outside permitted project roots: {value}')
        return str(resolved)

    accepted, ids, queues = [], {JOB, HOST_JOB}, {str((SCRATCH / 'taskq_dev3').resolve())}
    allowed = {'job_id', 'queue_dir', 'completion_files', 'progress_dirs', 'telemetry_dir'}
    for item in data['jobs']:
        if not isinstance(item, dict) or set(item) - allowed or not {'job_id', 'queue_dir'} <= set(item):
            raise ValueError('Registration requires job_id and queue_dir, with only documented fields')
        job = item['job_id']
        if (type(job) not in (str, int) or not str(job).isascii()
                or not str(job).isdigit() or int(job) <= 0):
            raise ValueError('Registered job_id must be a positive numeric Slurm job ID')
        job = str(int(job))
        queue = checked_path(item['queue_dir'], (SCRATCH,))
        if job in ids or queue in queues:
            raise ValueError(f'Duplicate registered job ID or queue: {job}')
        entry = {'job_id': job, 'queue_dir': queue,
                 'telemetry_dir': checked_path(item.get('telemetry_dir', str(SCRATCH / 'telemetry')),
                                               (SCRATCH,))}
        for key, roots in (('completion_files', (SCRATCH, ROOT)), ('progress_dirs', (SCRATCH,))):
            values = item.get(key, [])
            if not isinstance(values, list):
                raise ValueError(f'{key} must be a list')
            entry[key] = [checked_path(value, roots) for value in values]
            if len(set(entry[key])) != len(entry[key]):
                raise ValueError(f'Duplicate {key}')
        accepted.append(entry)
        ids.add(job)
        queues.add(queue)
    return accepted


def save(path, value):
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def scheduler(registered=(), extra_ids=()):
    ids = list(dict.fromkeys([JOB, HOST_JOB] + [item['job_id'] for item in registered] + list(extra_ids)))
    result = subprocess.run(['squeue', '-j', ','.join(ids), '-h', '-o', '%i|%T|%e'],
                            capture_output=True, text=True, timeout=20, check=True)
    jobs = {j: {'state': 'NOT_IN_QUEUE', 'end': None} for j in ids}
    for line in result.stdout.splitlines():
        job, status, end = line.strip().split('|')
        jobs[job] = {'state': status, 'end': None if end in ('N/A', 'Unknown')
                     else datetime.fromisoformat(end).timestamp()}
    return jobs


def telemetry_snapshot(directory, job, now):
    telemetry = {}
    for path in Path(directory).glob(f'*_job{job}_*.csv'):
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
    return telemetry


def task_snapshot(queue):
    return {kind: sorted(p.name for p in (queue / kind).glob('*.sh*'))
            for kind in ('pending', 'running', 'done', 'failed')}


def registered_snapshot(item, info, previous, now):
    queue = Path(item['queue_dir'])
    tasks = task_snapshot(queue)
    progress = [p.stat().st_mtime for directory in [queue / 'logs'] +
                [Path(p) for p in item['progress_dirs']]
                for p in directory.glob('*') if p.is_file()]
    # Timers belong to one registration and one continuous running interval.
    if previous.get('registration') != item:
        previous = {}
    active = info['state'] == 'RUNNING'
    running_since = (previous.get('running_since') or now) if active else None
    last_progress = max(progress + [previous.get('last_progress', now)])
    if running_since is not None:
        last_progress = max(last_progress, running_since)
    drained = (bool(tasks['done']) and not any(tasks[k] for k in ('pending', 'running', 'failed'))
               and info['state'] not in ('PENDING', 'UNKNOWN'))
    return {'registration': item, 'tasks': tasks,
            'telemetry': telemetry_snapshot(item['telemetry_dir'], item['job_id'], now),
            'last_progress': last_progress, 'running_since': running_since,
            'all_done_since': (previous.get('all_done_since') or now) if drained else None,
            'completion_files': {p: Path(p).is_file() for p in item['completion_files']}}


def snapshot(jobs, now, registered=(), previous=None):
    queue = SCRATCH / 'taskq_dev3'
    tasks = task_snapshot(queue)
    telemetry = telemetry_snapshot(SCRATCH / 'telemetry', JOB, now)
    progress = [p.stat().st_mtime for p in (queue / 'logs').glob('*.log')]
    for method in ('sega_official', 'sega_rms_uniform', 'sega_rms_normalized', 'sega_ctm_lag'):
        progress += [p.stat().st_mtime for p in
                     (SCRATCH / f'out_krea/dev3_{method}/{method}/4096x4096').glob('*.png')]
    ids = {JOB, HOST_JOB} | {item['job_id'] for item in registered}
    observed = {job: info for job, info in jobs.items() if job in ids}
    extras = {item['job_id']: registered_snapshot(
        item, jobs.get(item['job_id'], {'state': 'UNKNOWN'}),
        (previous or {}).get('registered', {}).get(item['job_id'], {}), now)
        for item in registered}
    return {'time': now, 'jobs': observed, 'tasks': tasks, 'telemetry': telemetry,
            'registered': extras,
            'last_progress': max(progress, default=now),
            'complete': (ROOT / '.agent/notes/continuation3/dev/completion.json').is_file()}


def events(current, previous, now, running_since=None, all_done_since=None):
    found = {}
    for job, info in current['jobs'].items():
        before = previous.get('jobs', {}).get(job, {}).get('state')
        if before and before != info['state']:
            found[f'job:{job}:{info["state"]}'] = f'Job {job}: {before} -> {info["state"]}'
        if job != HOST_JOB and not before and info['state'] != 'PENDING':
            found[f'job:{job}:{info["state"]}'] = f'Initial GPU job state: {info["state"]}'
        if job == HOST_JOB and info['end'] and info['end'] - now < 3600:
            found[f'lease:{info["end"]}'] = 'Controller allocation expires within one hour; verify migration'
            for seconds, label in ((900, 'fifteen minutes'), (300, 'five minutes')):
                if info['end'] - now < seconds:
                    found[f'lease:{HOST_JOB}:{seconds}:{info["end"]}'] = (
                        f'Controller allocation {HOST_JOB} expires within {label}; '
                        'verify compute-node handoff or preserve blocked migration state')
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
    for job, data in current.get('registered', {}).items():
        prefix = f'job:{job}:'
        for kind in ('failed', 'done', 'running'):
            for name in data['tasks'][kind]:
                found[f'{prefix}{kind}:{name}'] = f'Job {job} workflow {kind}: {name}'
        active = current['jobs'].get(job, {}).get('state') == 'RUNNING'
        if (active and data['running_since'] is not None
                and now - data['running_since'] > 600 and data['tasks']['running']):
            if not data['telemetry']:
                found[f'{prefix}missing_telemetry'] = f'Job {job}: no GPU telemetry after ten minutes'
            for name, info in data['telemetry'].items():
                for key in ('stale', 'low'):
                    if info[key]:
                        found[f'{prefix}{key}:{name}'] = f'Job {job}: {key} GPU telemetry: {name}'
            if now - data['last_progress'] > 1200:
                found[f'{prefix}stalled'] = f'Job {job}: no file or task-log progress for twenty minutes'
        receipts = data['completion_files']
        for filename, present in receipts.items():
            if present:
                found[f'{prefix}receipt:{filename}'] = (
                    f'Job {job}: completion file present at {filename}; verify validity and quality')
        if (receipts and not all(receipts.values()) and data['all_done_since'] is not None
                and now - data['all_done_since'] > 600):
            found[f'{prefix}finalizer_delay'] = (
                f'Job {job}: spool drained, but completion files still missing after ten minutes')
    return found


def connect():
    from websockets.sync.client import unix_connect
    return unix_connect(SOCKET, open_timeout=10, close_timeout=3)


def rpc(ws, method, params):
    request_id = str(uuid.uuid4())
    ws.send(json.dumps({'id': request_id, 'method': method, 'params': params}))
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        reply = json.loads(ws.recv(timeout=max(.1, deadline - time.monotonic())))
        if reply.get('id') == request_id:
            if 'error' in reply:
                raise RuntimeError(f'{method}: {reply["error"]}')
            return reply['result']
    raise TimeoutError(method)


def enqueue(message):
    """Start the existing operator when idle; a busy operator keeps the outbox."""
    with connect() as ws:
        rpc(ws, 'initialize', {'clientInfo': {'name': 'crope-watcher', 'version': '2'},
                              'capabilities': {'experimentalApi': True}})
        ws.send(json.dumps({'method': 'initialized'}))
        thread = rpc(ws, 'thread/read', {'threadId': THREAD, 'includeTurns': False})['thread']
        if thread['id'] != THREAD or Path(thread['cwd']) != ROOT:
            raise RuntimeError('Refusing to wake a different project/thread')
        if thread['status']['type'] == 'active':
            return None
        if thread['status']['type'] == 'notLoaded':
            rpc(ws, 'thread/resume', {'threadId': THREAD})
        result = rpc(ws, 'turn/start', {
            'threadId': THREAD, 'input': [{'type': 'text', 'text': message}],
            'clientUserMessageId': str(uuid.uuid5(uuid.NAMESPACE_URL, message)),
            'turnTrigger': 'crope-watcher'})
        return {'turn_id': result['turn']['id'], 'turn_status': result['turn']['status']}


def main():
    global HOST_JOB, SOCKET, THREAD
    args = arguments()
    HOST_JOB, SOCKET, THREAD = args.host_job, args.socket, args.thread
    cap = resource.getrlimit(resource.RLIMIT_NPROC)[0]
    if socket.gethostname().startswith('login') or 0 < cap <= 512:
        raise SystemExit('Compute node required')
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = RUNTIME / 'state.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        state.update(pid=os.getpid(), host=socket.gethostname(), thread=THREAD,
                     host_job=HOST_JOB, socket=SOCKET, started=time.time())
        pending, sent = state.setdefault('pending', {}), state.setdefault('sent', {})
        next_scheduler, jobs, registered = 0, {}, []
        while not (RUNTIME / 'STOP').exists():
            now = time.time()
            try:
                try:
                    registered = registrations()
                    state.pop('registration_error', None)
                except (ValueError, OSError) as exc:
                    error = str(exc)
                    state['registration_error'] = error
                    key = 'registration_error:' + hashlib.sha256(error.encode()).hexdigest()[:16]
                    if key not in sent:
                        pending[key] = f'Watcher registration rejected; last accepted set retained: {error}'
                ready = destination_ready(now)
                refreshed = now >= next_scheduler
                if refreshed:
                    jobs = (scheduler(registered, [ready['destination_job']]) if ready
                            else scheduler(registered))
                    next_scheduler = now + 120
                current = snapshot(jobs, now, registered, state.get('snapshot', {}))
                if (RUNTIME / 'WAKE_TEST').exists() and 'wake_test' not in sent:
                    pending['wake_test'] = ('Direct wake-up verification: record receipt in '
                        '.agent/notes/watcher_ack.json, then continue the response-probe '
                        'implementation and prior-art work that do not need GPUs')
                if jobs[JOB]['state'] == 'RUNNING':
                    state.setdefault('running_since', now)
                if len(current['tasks']['done']) == 4:
                    state.setdefault('all_done_since', now)
                found = events(current, state.get('snapshot', {}), now,
                               state.get('running_since'), state.get('all_done_since'))
                pending.update({k: v for k, v in found.items() if k not in sent})
                state.update(snapshot=current, heartbeat=now)
                if ready and refreshed and not args.dry_run:
                    receipt = handoff_receipt(ready, jobs, time.time())
                    if receipt is not None:
                        state['handoff_released'] = receipt
                        save(path, state)
                        save(RUNTIME / 'HANDOFF_RELEASED.json', receipt)
                        return
                if pending and (args.dry_run or now - state.get('last_attempt', 0) >= 120):
                    message = ('Constancy RoPE automatic watcher event, authorized by the user for '
                               'unattended continuation. Read HANDOVER.md, .agent/STATE.md and '
                               f'{path}. Verify live facts, handle failures/utilization, evaluate '
                               'results, and continue novel-method research and paper work. '
                               'A diagnostic pass is not novelty or SOTA. Preserve frozen protocols '
                               'and fresh64. Register replacement jobs with this watcher before '
                               'returning to waiting. Do not duplicate this watcher. If controller lease '
                               'is expiring, arrange and verify a compute-node handoff. Events: '
                               + '; '.join(pending.values()))
                    if args.dry_run:
                        print(message, flush=True)
                    else:
                        state['last_attempt'] = now
                        save(path, state)
                        receipt = enqueue(message)
                        if receipt is not None:
                            with (RUNTIME / 'deliveries.jsonl').open('a') as stream:
                                stream.write(json.dumps({'time': now, 'events': dict(pending),
                                                         'receipt': receipt}) + '\n')
                            sent.update({k: now for k in pending})
                            pending.clear()
                if not args.dry_run:
                    save(path, state)
                # Keep observing the controller lease after the first report;
                # the resumed operator must register the next campaign.
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
