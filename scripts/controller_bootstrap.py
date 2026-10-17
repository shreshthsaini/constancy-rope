"""Acquire controller ownership on an allocated compute node without duplicate watchers.

This program never resumes a thread or starts a model turn itself. The existing
watcher does that only after the source watcher has relinquished its lock.
"""
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import watch_cr14 as watcher

SOURCE_JOB = '1073260'
THREAD = '01a14899-2e69-7d71-93f8-b42aad182afd'
ROOT, RUNTIME = watcher.ROOT, watcher.RUNTIME
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED'}
READY_KEYS = {'source_job', 'thread', 'destination_job', 'destination_node',
              'destination_socket', 'created'}


def require_compute(env, host, process_cap):
    job = watcher.positive_job(env.get('SLURM_JOB_ID', ''))
    if host.startswith('login') or 0 < process_cap <= 512 or job == SOURCE_JOB:
        raise RuntimeError('A different allocated compute node is required')
    return job


def allocation_end(job, host):
    result = subprocess.run(['squeue', '-j', job, '-w', host.split('.', 1)[0], '-h', '-o', '%i|%T|%e'],
                            capture_output=True, text=True, check=True, timeout=20)
    rows = [line.strip().split('|') for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0][:2] != [job, 'RUNNING']:
        raise RuntimeError('This node is not in the running destination allocation')
    end = datetime.fromisoformat(rows[0][2]).timestamp()
    if end <= time.time() + 180:
        raise RuntimeError('Destination allocation has less than three minutes remaining')
    return end


def source_state(job=None):
    job = SOURCE_JOB if job is None else job
    result = subprocess.run(['sacct', '-X', '-j', job, '-n', '-P', '-o', 'JobIDRaw,State'],
                            capture_output=True, text=True, check=True, timeout=20)
    rows = [line.strip().split('|') for line in result.stdout.splitlines() if line.strip()]
    values = [row[1].split()[0].rstrip('+') for row in rows if len(row) >= 2 and row[0] == job]
    if len(values) != 1:
        raise RuntimeError('Source accounting state is missing or ambiguous')
    return values[0]


def verify_thread(socket_path):
    # Use the watcher's established transport and RPC framing, but never enqueue.
    original = watcher.SOCKET
    watcher.SOCKET = socket_path
    try:
        with watcher.connect() as ws:
            watcher.rpc(ws, 'initialize', {'clientInfo': {'name': 'crope-bootstrap', 'version': '1'},
                                          'capabilities': {'experimentalApi': True}})
            ws.send(json.dumps({'method': 'initialized'}))
            thread = watcher.rpc(ws, 'thread/read', {'threadId': THREAD, 'includeTurns': False})['thread']
            if thread['id'] != THREAD or Path(thread['cwd']) != ROOT:
                raise RuntimeError('Destination server cannot verify the exact project/thread')
            return thread['status']['type']
    finally:
        watcher.SOCKET = original


def ready_record(job, host, socket_path, now):
    return dict(source_job=SOURCE_JOB, thread=THREAD, destination_job=job,
                destination_node=host, destination_socket=socket_path, created=now)


def matching_release(release, ready, now):
    """Reject an old release, malformed times, or receipts for another destination."""
    if not isinstance(release, dict) or any(release.get(k) != ready[k] for k in READY_KEYS):
        return False
    allowed = READY_KEYS | {'source_node', 'source_socket', 'released'}
    if set(release) != allowed:
        return False
    stamp = release['released']
    return (type(stamp) in (int, float) and math.isfinite(stamp)
            and ready['created'] <= stamp <= now + 60
            and isinstance(release['source_node'], str) and bool(release['source_node'])
            and isinstance(release['source_socket'], str)
            and Path(release['source_socket']).is_absolute())


def expiry_eligible(probes, state, now):
    if len(probes) != 2 or any(p['state'] not in TERMINAL for p in probes):
        return False
    heartbeat = state.get('heartbeat')
    return (probes[1]['time'] - probes[0]['time'] >= 10
            and 0 <= now - probes[1]['time'] <= 30
            and state.get('host_job') == SOURCE_JOB and state.get('thread') == THREAD
            and type(heartbeat) in (int, float) and math.isfinite(heartbeat)
            and now - heartbeat >= 180)


def queue_handoff(state, ready, now):
    """Preserve existing events, sent receipts, and experiment registry references."""
    key = f"controller_handoff:{SOURCE_JOB}:{ready['destination_job']}:{ready['created']}"
    if key not in state.setdefault('sent', {}):
        state.setdefault('pending', {})[key] = (
            f"Controller ownership acquired on {ready['destination_node']} in job "
            f"{ready['destination_job']}. This is not proof of an actual resumed agent turn. "
            "As the resumed agent, verify your actual hostname and SLURM_JOB_ID and write "
            ".agent/notes/controller_handoff_verified.json with those live facts, this "
            "destination socket and thread, and the receipt time. Continue CR21 job1077335, "
            "verify GPU results and quality, and update the paper. Preserve frozen protocols "
            "and fresh64. Do not duplicate the watcher.")
    state['bootstrap_handoff'] = dict(ready, ownership_acquired=now)
    # Existing last_attempt could defer the one immediate ownership acknowledgement.
    state['last_attempt'] = 0
    return state


def onward_handoff_released(release, job, host, endpoint, now):
    """A clean watcher exit is expected only after this controller released itself."""
    if not isinstance(release, dict) or set(release) != READY_KEYS | {'source_node', 'source_socket', 'released'}:
        return False
    if (release['source_job'] != job or release['source_node'] != host
            or release['source_socket'] != endpoint or release['thread'] != THREAD):
        return False
    try:
        destination = watcher.positive_job(release['destination_job'])
        watcher.absolute_socket(release['destination_socket'])
    except (ValueError, TypeError, argparse.ArgumentTypeError):
        return False
    node = release['destination_node']
    if (destination == job or destination != release['destination_job']
            or not isinstance(node, str) or not node or node == host or node.startswith('login')
            or not all(c.isascii() and (c.isalnum() or c in '.-') for c in node)):
        return False
    created, released = release['created'], release['released']
    return (type(created) in (int, float) and math.isfinite(created)
            and type(released) in (int, float) and math.isfinite(released)
            and created <= released and -60 <= now - created <= 3600
            and -60 <= now - released <= 180)


def acquire_watcher_lock():
    lock = (RUNTIME / 'lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def claim_if_released(ready, probes, now):
    lock = acquire_watcher_lock()
    if lock is None:
        return False
    with lock:
        state_path = RUNTIME / 'state.json'
        state = json.loads(state_path.read_text())
        path = RUNTIME / 'HANDOFF_RELEASED.json'
        release = json.loads(path.read_text()) if path.exists() else None
        if not matching_release(release, ready, now):
            if not expiry_eligible(probes, state, now):
                return False
            # Separate evidence file keeps the cooperative release schema identical.
            release = dict(ready, source_node=state['host'], source_socket=state['socket'], released=now)
            watcher.save(RUNTIME / f"EXPIRY_RELEASE_{ready['destination_job']}.json",
                         dict(release, terminal_probes=probes, source_heartbeat=state['heartbeat']))
            watcher.save(path, release)
        if (RUNTIME / 'STOP').exists():
            (RUNTIME / 'STOP').rename(RUNTIME / f"STOP.preserved.{time.time_ns()}")
        watcher.save(state_path, queue_handoff(state, ready, now))
        return True


def process_identity(pid):
    result = subprocess.run(['ps', '-o', 'pid,ppid,lstart,stat,args', '-p', str(pid)],
                            capture_output=True, text=True, check=True, timeout=10)
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': stat[19], 'ps': result.stdout}


def stop_child(child, identity):
    if child.poll() is not None:
        child.wait()
        return
    current = process_identity(child.pid)
    if current['start_ticks'] != identity['start_ticks'] or os.getpgid(child.pid) != child.pid:
        raise RuntimeError('Refusing to signal a child whose identity changed')
    print(json.dumps({'terminating_own_child': current}), flush=True)
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        current = process_identity(child.pid)
        if current['start_ticks'] != identity['start_ticks'] or os.getpgid(child.pid) != child.pid:
            raise RuntimeError('Refusing to kill a child whose identity changed')
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def run(codex_binary):
    host = socket.gethostname()
    job = require_compute(os.environ, host, resource.getrlimit(resource.RLIMIT_NPROC)[0])
    end = allocation_end(job, host)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    children = []
    # Only contenders for the same source exclude one another. A later hop must
    # be able to prepare its server while this controller still owns its watcher.
    with (RUNTIME / f'controller_bootstrap_source_{SOURCE_JOB}.lock').open('a+') as bootstrap_lock:
        fcntl.flock(bootstrap_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        private = Path(tempfile.mkdtemp(prefix=f'crope-controller-{job}-', dir='/tmp'))
        private.chmod(0o700)
        endpoint = str(private / 'app-server.sock')
        logdir = RUNTIME / f'controller_job{job}'
        logdir.mkdir(exist_ok=True)
        last_progress = 0
        def progress(phase, **extra):
            nonlocal last_progress
            now = time.time()
            if now - last_progress >= 60 or extra.pop('force', False):
                watcher.save(logdir / 'bootstrap_progress.json',
                             dict(phase=phase, heartbeat=now, source_job=SOURCE_JOB,
                                  destination_job=job, destination_node=host,
                                  destination_socket=endpoint, thread=THREAD,
                                  allocation_end=end, **extra))
                last_progress = now
        try:
            progress('starting_app_server', force=True)
            with (logdir / 'app-server.log').open('a') as log:
                server = subprocess.Popen([codex_binary, 'app-server', '--listen', 'unix://' + endpoint],
                                          cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            children.append((server, process_identity(server.pid)))
            startup_deadline = min(time.time() + 300, end - 120)
            while True:
                progress('verifying_thread_without_resume')
                if server.poll() is not None:
                    raise RuntimeError('Destination app-server exited during startup')
                try:
                    verify_thread(endpoint)
                    break
                except Exception as exc:
                    if time.time() + 10 >= startup_deadline:
                        raise RuntimeError('Destination server never verified the thread') from exc
                    time.sleep(10)
            ready = ready_record(job, host, endpoint, time.time())
            watcher.save(RUNTIME / 'DESTINATION_READY.json', ready)
            progress('awaiting_ownership', force=True)
            probes, next_probe = [], 0
            while time.time() < end - 120:
                if server.poll() is not None:
                    raise RuntimeError('Destination app-server exited while awaiting ownership')
                now = time.time()
                progress('awaiting_ownership', terminal_probes=probes)
                if now >= next_probe:
                    try:
                        state = source_state()
                        probes = (probes + [{'state': state, 'time': time.time()}])[-2:] if state in TERMINAL else []
                    except Exception as exc:
                        probes = []
                        print(json.dumps({'source_probe_failed': str(exc), 'time': now}), flush=True)
                    next_probe = time.time() + 120
                if claim_if_released(ready, probes, time.time()):
                    break
                time.sleep(10)
            else:
                raise RuntimeError('Destination allocation expired before acquiring ownership')
            with (logdir / 'watcher.log').open('a') as log:
                child = subprocess.Popen([sys.executable, str(ROOT / 'scripts/watch_cr14.py'),
                                          '--host-job', job, '--socket', endpoint, '--thread', THREAD],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            children.append((child, process_identity(child.pid)))
            watcher.save(logdir / 'bootstrap_started.json', dict(ready, children=[i for _, i in children]))
            progress('watcher_started_unverified_agent_turn', force=True)
            while time.time() < end - 60:
                progress('watcher_running_unverified_agent_turn')
                child_status = child.poll()
                if child_status == 0:
                    release_path = RUNTIME / 'HANDOFF_RELEASED.json'
                    release = json.loads(release_path.read_text()) if release_path.exists() else None
                    if onward_handoff_released(release, job, host, endpoint, time.time()):
                        progress('onward_handoff_completed', force=True, release=release)
                        return
                if server.poll() is not None or child_status is not None:
                    raise RuntimeError('Owned app-server or watcher exited; see private job logs')
                time.sleep(10)
        finally:
            errors = []
            for child, identity in reversed(children):
                try:
                    stop_child(child, identity)
                except Exception as exc:
                    errors.append(str(exc))
            if errors:
                raise RuntimeError('; '.join(errors))


def main():
    global SOURCE_JOB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codex', default=shutil.which('codex'))
    parser.add_argument('--source-job', type=watcher.positive_job, default=SOURCE_JOB)
    args = parser.parse_args()
    SOURCE_JOB = args.source_job
    if not args.codex or not Path(args.codex).is_file():
        parser.error('An installed codex executable is required')
    def interrupted(signum, frame):
        raise InterruptedError(f'Controller bootstrap received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run(args.codex)


if __name__ == '__main__':
    main()
