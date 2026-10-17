"""Controller handoff checks use no daemon, scheduler, or model."""
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import controller_bootstrap as boot


@pytest.fixture
def ready():
    return boot.ready_record('1234567', 'i600-001', '/tmp/private/app-server.sock', 1000)


@pytest.fixture
def state():
    return {'host_job': boot.SOURCE_JOB, 'thread': boot.THREAD, 'heartbeat': 500,
            'host': 'i618-051', 'socket': '/tmp/old.sock', 'pending': {'earlier': 'retain'},
            'sent': {'already': 1}, 'snapshot': {'registered_jobs': {'1077335': 'PENDING'}}}


def release(ready):
    return dict(ready, source_node='i618-051', source_socket='/tmp/old.sock', released=1100)


def test_requires_different_allocated_compute():
    assert boot.require_compute({'SLURM_JOB_ID': '1234567'}, 'i600-001', 16384) == '1234567'
    for env, host, cap in [({}, 'i600-001', 16384), ({'SLURM_JOB_ID': '1234567'}, 'login1', 16384),
                           ({'SLURM_JOB_ID': '1234567'}, 'i600-001', 100),
                           ({'SLURM_JOB_ID': boot.SOURCE_JOB}, 'i600-001', 16384)]:
        with pytest.raises(Exception):
            boot.require_compute(env, host, cap)


def test_allocation_end_verifies_job_node_and_running(monkeypatch):
    calls = []
    monkeypatch.setattr(boot.time, 'time', lambda: 0)
    def execute(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='1234567|RUNNING|2026-10-17T12:00:00\n')
    monkeypatch.setattr(boot.subprocess, 'run', execute)
    assert boot.allocation_end('1234567', 'i600-001') > 0
    assert calls[0][3:5] == ['-w', 'i600-001']
    monkeypatch.setattr(boot.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='1234567|PENDING|Unknown\n'))
    with pytest.raises(RuntimeError, match='running destination'):
        boot.allocation_end('1234567', 'i600-001')


@pytest.mark.parametrize('text,expected', [
    ('1073260|COMPLETED|\n', 'COMPLETED'), ('1073260|CANCELLED by 44|\n', 'CANCELLED'),
    ('1073260|RUNNING|\n1073260.batch|COMPLETED|\n', 'RUNNING')])
def test_sacct_exact_allocation_record(monkeypatch, text, expected):
    monkeypatch.setattr(boot.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=text))
    assert boot.source_state() == expected


@pytest.mark.parametrize('text', ['', '1073260.batch|COMPLETED|\n', '1073260|COMPLETED|\n1073260|RUNNING|\n'])
def test_sacct_missing_or_ambiguous_fails_closed(monkeypatch, text):
    monkeypatch.setattr(boot.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=text))
    with pytest.raises(RuntimeError, match='missing or ambiguous'):
        boot.source_state()


@pytest.mark.parametrize('status', ['idle', 'active', 'notLoaded'])
def test_thread_verification_never_resumes_or_starts_turn(monkeypatch, status):
    calls = []
    old_socket = boot.watcher.SOCKET
    monkeypatch.setattr(boot.watcher, 'connect', lambda: nullcontext(SimpleNamespace(send=lambda x: None)))
    def rpc(ws, method, params):
        calls.append(method)
        return {'thread': {'id': boot.THREAD, 'cwd': str(boot.ROOT), 'status': {'type': status}}}
    monkeypatch.setattr(boot.watcher, 'rpc', rpc)
    assert boot.verify_thread('/tmp/private.sock') == status
    assert calls == ['initialize', 'thread/read']
    assert boot.watcher.SOCKET == old_socket


@pytest.mark.parametrize('field,value', [('id', 'other'), ('cwd', '/other')])
def test_thread_mismatch_fails_closed(monkeypatch, field, value):
    thread = {'id': boot.THREAD, 'cwd': str(boot.ROOT), 'status': {'type': 'idle'}}
    thread[field] = value
    monkeypatch.setattr(boot.watcher, 'connect', lambda: nullcontext(SimpleNamespace(send=lambda x: None)))
    monkeypatch.setattr(boot.watcher, 'rpc', lambda *a: {'thread': thread})
    with pytest.raises(RuntimeError, match='exact project/thread'):
        boot.verify_thread('/tmp/private.sock')


def test_release_requires_exact_destination_and_schema(ready):
    assert boot.matching_release(release(ready), ready, 1200)
    for key in boot.READY_KEYS:
        bad = release(ready)
        bad[key] = 'wrong'
        assert not boot.matching_release(bad, ready, 1200)
    assert not boot.matching_release(dict(release(ready), extra=True), ready, 1200)
    assert not boot.matching_release(None, ready, 1200)


@pytest.mark.parametrize('stamp', [float('nan'), float('inf'), 999, 1261, True, '1100'])
def test_bad_release_time_rejected(ready, stamp):
    assert not boot.matching_release(dict(release(ready), released=stamp), ready, 1200)


def test_expiry_requires_two_terminal_probes_and_stale_matching_heartbeat(state):
    probes = [{'time': 1000, 'state': 'TIMEOUT'}, {'time': 1120, 'state': 'TIMEOUT'}]
    assert boot.expiry_eligible(probes, state, 1120)
    assert not boot.expiry_eligible(probes[:1], state, 1120)
    assert not boot.expiry_eligible([probes[0], {'time': 1009, 'state': 'TIMEOUT'}], state, 1009)
    assert not boot.expiry_eligible([probes[0], {'time': 1120, 'state': 'RUNNING'}], state, 1120)
    assert not boot.expiry_eligible(probes, dict(state, heartbeat=1000), 1120)
    assert not boot.expiry_eligible(probes, dict(state, host_job='other'), 1120)
    assert not boot.expiry_eligible(probes, dict(state, thread='other'), 1120)
    assert not boot.expiry_eligible(probes, dict(state, heartbeat=float('nan')), 1120)
    assert not boot.expiry_eligible(probes, state, 1200)


def test_queue_preserves_existing_events_and_does_not_claim_actual_turn(ready, state):
    result = boot.queue_handoff(state, ready, 1200)
    assert result['pending']['earlier'] == 'retain'
    assert result['sent'] == {'already': 1}
    assert result['snapshot']['registered_jobs']['1077335'] == 'PENDING'
    messages = [v for k, v in result['pending'].items() if k.startswith('controller_handoff:')]
    assert len(messages) == 1
    assert 'not proof' in messages[0] and 'controller_handoff_verified.json' in messages[0]
    assert 'Continue CR21 job1077335' in messages[0]
    assert len(boot.queue_handoff(result, ready, 1300)['pending']) == 2


def prepare_runtime(monkeypatch, tmp_path, state):
    monkeypatch.setattr(boot, 'RUNTIME', tmp_path)
    (tmp_path / 'state.json').write_text(json.dumps(state))
    (tmp_path / 'registrations.json').write_text('preserve')
    (tmp_path / 'deliveries.jsonl').write_text('preserve')


def test_live_flock_blocks_even_cooperative_release(monkeypatch, tmp_path, ready, state):
    prepare_runtime(monkeypatch, tmp_path, state)
    (tmp_path / 'HANDOFF_RELEASED.json').write_text(json.dumps(release(ready)))
    with (tmp_path / 'lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not boot.claim_if_released(ready, [], 1200)
    assert json.loads((tmp_path / 'state.json').read_text()) == state


def test_cooperative_claim_preserves_stop_and_shared_registry(monkeypatch, tmp_path, ready, state):
    prepare_runtime(monkeypatch, tmp_path, state)
    (tmp_path / 'HANDOFF_RELEASED.json').write_text(json.dumps(release(ready)))
    (tmp_path / 'STOP').write_text('old stop')
    assert boot.claim_if_released(ready, [], 1200)
    stops = list(tmp_path.glob('STOP.preserved.*'))
    assert len(stops) == 1 and stops[0].read_text() == 'old stop'
    assert not (tmp_path / 'STOP').exists()
    assert (tmp_path / 'registrations.json').read_text() == 'preserve'
    assert (tmp_path / 'deliveries.jsonl').read_text() == 'preserve'
    assert json.loads((tmp_path / 'state.json').read_text())['pending']['earlier'] == 'retain'


def test_expiry_claim_archives_probe_evidence(monkeypatch, tmp_path, ready, state):
    prepare_runtime(monkeypatch, tmp_path, state)
    probes = [{'time': 1000, 'state': 'TIMEOUT'}, {'time': 1120, 'state': 'TIMEOUT'}]
    assert boot.claim_if_released(ready, probes, 1120)
    evidence = json.loads((tmp_path / 'EXPIRY_RELEASE_1234567.json').read_text())
    assert evidence['terminal_probes'] == probes and evidence['source_heartbeat'] == 500
    assert boot.matching_release(json.loads((tmp_path / 'HANDOFF_RELEASED.json').read_text()), ready, 1120)


def test_unreleased_live_source_never_changes_state(monkeypatch, tmp_path, ready, state):
    prepare_runtime(monkeypatch, tmp_path, state)
    assert not boot.claim_if_released(ready, [], 1200)
    assert json.loads((tmp_path / 'state.json').read_text()) == state
    assert not (tmp_path / 'HANDOFF_RELEASED.json').exists()


def test_cleanup_checks_created_pid_identity(monkeypatch):
    signals, waits = [], []
    child = SimpleNamespace(pid=1234, poll=lambda: None, wait=lambda **kw: waits.append(kw))
    monkeypatch.setattr(boot, 'process_identity', lambda pid: {'pid': pid, 'start_ticks': '55', 'ps': 'verified'})
    monkeypatch.setattr(boot.os, 'getpgid', lambda pid: pid)
    monkeypatch.setattr(boot.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    boot.stop_child(child, {'start_ticks': '55'})
    assert signals == [(1234, boot.signal.SIGTERM)] and waits
    signals.clear()
    with pytest.raises(RuntimeError, match='identity changed'):
        boot.stop_child(child, {'start_ticks': 'other'})
    assert signals == []


def test_exited_child_reaped_without_signal(monkeypatch):
    waits = []
    child = SimpleNamespace(pid=1234, poll=lambda: 0, wait=lambda: waits.append(True))
    monkeypatch.setattr(boot, 'process_identity', lambda pid: pytest.fail('Do not inspect an exited PID'))
    boot.stop_child(child, {})
    assert waits == [True]


def onward_release():
    return dict(source_job='1234567', source_node='i600-001', source_socket='/tmp/private.sock',
                destination_job='1234568', destination_node='i600-002',
                destination_socket='/tmp/next.sock', thread=boot.THREAD,
                created=1000, released=1100)


def test_onward_handoff_requires_this_exact_source_controller():
    assert boot.onward_handoff_released(onward_release(), '1234567', 'i600-001', '/tmp/private.sock', 1200)
    for key in ['source_job', 'source_node', 'source_socket', 'thread']:
        value = onward_release()
        value[key] = 'other'
        assert not boot.onward_handoff_released(value, '1234567', 'i600-001', '/tmp/private.sock', 1200)


@pytest.mark.parametrize('field,value', [
    ('destination_job', '1234567'), ('destination_job', '01234568'), ('destination_job', 'invalid'),
    ('destination_node', 'i600-001'), ('destination_node', 'login1'), ('destination_node', ''),
    ('destination_socket', 'relative.sock'), ('created', float('nan')),
    ('released', float('inf')), ('released', True), ('released', 999), ('released', 1019),
    ('released', 1261), ('created', -3000)])
def test_onward_handoff_rejects_invalid_or_stale_receipts(field, value):
    receipt = onward_release()
    receipt[field] = value
    assert not boot.onward_handoff_released(receipt, '1234567', 'i600-001', '/tmp/private.sock', 1200)


def test_onward_handoff_rejects_absent_or_foreign_schema():
    assert not boot.onward_handoff_released(None, '1234567', 'i600-001', '/tmp/private.sock', 1200)
    receipt = dict(onward_release(), unknown=True)
    assert not boot.onward_handoff_released(receipt, '1234567', 'i600-001', '/tmp/private.sock', 1200)


def test_source_scoped_lock_allows_next_hop_but_blocks_duplicate(tmp_path):
    # Exercise the same flock naming policy used by the live bootstrap.
    with (tmp_path / 'controller_bootstrap_source_1073260.lock').open('a+') as first:
        fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (tmp_path / 'controller_bootstrap_source_1234567.lock').open('a+') as successor:
            fcntl.flock(successor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (tmp_path / 'controller_bootstrap_source_1073260.lock').open('a+') as duplicate:
            with pytest.raises(BlockingIOError):
                fcntl.flock(duplicate, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize('valid_release', [True, False])
def test_running_bootstrap_accepts_verified_onward_exit_and_reaps_only_own_children(
        monkeypatch, tmp_path, valid_release):
    monkeypatch.setattr(boot, 'RUNTIME', tmp_path)
    monkeypatch.setattr(boot.socket, 'gethostname', lambda: 'i600-001')
    monkeypatch.setattr(boot, 'require_compute', lambda *args: '1234567')
    monkeypatch.setattr(boot, 'allocation_end', lambda *args: 100000)
    monkeypatch.setattr(boot.time, 'time', lambda: 1200)
    private = tmp_path / 'private'
    private.mkdir()
    monkeypatch.setattr(boot.tempfile, 'mkdtemp', lambda **kwargs: str(private))
    monkeypatch.setattr(boot, 'verify_thread', lambda endpoint: 'idle')
    monkeypatch.setattr(boot, 'source_state', lambda: 'RUNNING')
    monkeypatch.setattr(boot, 'claim_if_released', lambda *args: True)
    monkeypatch.setattr(boot, 'process_identity', lambda pid: {'pid': pid, 'start_ticks': '1'})
    own_server = SimpleNamespace(pid=11, poll=lambda: None)
    own_watcher = SimpleNamespace(pid=12, poll=lambda: 0)
    children = iter([own_server, own_watcher])
    monkeypatch.setattr(boot.subprocess, 'Popen', lambda *args, **kwargs: next(children))
    stopped = []
    monkeypatch.setattr(boot, 'stop_child', lambda child, identity: stopped.append(child.pid))
    receipt = onward_release()
    receipt['source_socket'] = str(private / 'app-server.sock') if valid_release else '/tmp/foreign.sock'
    (tmp_path / 'HANDOFF_RELEASED.json').write_text(json.dumps(receipt))
    if valid_release:
        boot.run('/fake/codex')
        progress = json.loads((tmp_path / 'controller_job1234567/bootstrap_progress.json').read_text())
        assert progress['phase'] == 'onward_handoff_completed'
        assert progress['release'] == receipt
    else:
        with pytest.raises(RuntimeError, match='Owned app-server or watcher exited'):
            boot.run('/fake/codex')
    assert stopped == [12, 11]
    assert (tmp_path / f'controller_bootstrap_source_{boot.SOURCE_JOB}.lock').exists()
    assert not (tmp_path / 'controller_bootstrap.lock').exists()
