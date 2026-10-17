"""Protect event boundaries: queue waiting is normal, missing results are not success."""
from pathlib import Path
import sys
from types import SimpleNamespace
from contextlib import nullcontext

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import watch_cr14 as watcher


def current(state='PENDING'):
    return {'jobs': {watcher.JOB: {'state': state, 'end': None},
                     watcher.HOST_JOB: {'state': 'RUNNING', 'end': 20000}},
            'tasks': {k: [] for k in ('pending', 'running', 'done', 'failed')},
            'telemetry': {}, 'last_progress': 1000, 'complete': False}


def test_pending_queue_does_not_wake_model():
    data = current()
    data['tasks']['pending'] = ['a.sh']
    assert watcher.events(data, {}, 1000) == {}


def test_allocation_and_failure_wake():
    data = current('RUNNING')
    data['tasks']['failed'] = ['a.sh']
    found = watcher.events(data, current(), 1000)
    assert f'job:{watcher.JOB}:RUNNING' in found
    assert 'failed:a.sh' in found
    assert 'complete' not in found


def test_job_exit_never_implies_validated_completion():
    data = current('NOT_IN_QUEUE')
    found = watcher.events(data, current('RUNNING'), 1000)
    assert f'job:{watcher.JOB}:NOT_IN_QUEUE' in found
    assert 'complete' not in found


def test_finalizer_receipt_and_delay_are_distinct():
    data = current()
    data['tasks']['done'] = ['a', 'b', 'c', 'd']
    found = watcher.events(data, {}, 2000, all_done_since=1000)
    assert 'finalizer_delay' in found and 'complete' not in found
    data['complete'] = True
    found = watcher.events(data, {}, 2000, all_done_since=1000)
    assert 'complete' in found and 'finalizer_delay' not in found


def test_startup_grace_then_inactivity_and_expiry():
    data = current('RUNNING')
    data['tasks']['running'] = ['a.sh.host.g0']
    assert 'missing_telemetry' not in watcher.events(data, data, 1100, running_since=1000)
    data['jobs'][watcher.HOST_JOB]['end'] = 5000
    found = watcher.events(data, data, 2300, running_since=1000)
    assert 'missing_telemetry' in found and 'stalled' in found
    assert 'lease:5000' in found


def mock_transport(monkeypatch, status='idle', cwd=watcher.ROOT):
    calls = []
    monkeypatch.setattr(watcher, 'connect', lambda: nullcontext(SimpleNamespace(send=lambda x: None)))
    def rpc(ws, method, params):
        calls.append(method)
        if method == 'thread/read':
            return {'thread': {'id': watcher.THREAD, 'cwd': str(cwd), 'status': {'type': status}}}
        if method == 'turn/start':
            return {'turn': {'id': 'test-turn', 'status': 'inProgress'}}
        return {}
    monkeypatch.setattr(watcher, 'rpc', rpc)
    return calls


def test_refuses_different_project(monkeypatch):
    calls = mock_transport(monkeypatch, cwd='/other/project')
    with pytest.raises(RuntimeError, match='different project'):
        watcher.enqueue('test')
    assert 'turn/start' not in calls


def test_busy_operator_defers_without_duplicate_turn(monkeypatch):
    calls = mock_transport(monkeypatch, status='active')
    assert watcher.enqueue('test') is None
    assert 'turn/start' not in calls


def test_idle_operator_starts_turn(monkeypatch):
    calls = mock_transport(monkeypatch)
    assert watcher.enqueue('test')['turn_id'] == 'test-turn'
    assert 'turn/start' in calls


def test_unloaded_operator_resumes_before_turn(monkeypatch):
    calls = mock_transport(monkeypatch, status='notLoaded')
    watcher.enqueue('test')
    assert calls.index('thread/resume') < calls.index('turn/start')
