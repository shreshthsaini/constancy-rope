"""Protect event boundaries: queue waiting is normal, missing results are not success."""
from pathlib import Path
import sys
from types import SimpleNamespace

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


def test_delivery_requires_matching_thread_receipt(monkeypatch):
    monkeypatch.setattr(watcher.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(stdout='Queued message wrong-thread'))
    with pytest.raises(RuntimeError, match='acknowledgment'):
        watcher.enqueue('test')


def test_matching_delivery_receipt(monkeypatch):
    receipt = f'Queued message test for thread {watcher.THREAD}.'
    monkeypatch.setattr(watcher.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(stdout=receipt))
    assert watcher.enqueue('test') == receipt
