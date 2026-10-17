"""Protect event boundaries: queue waiting is normal, missing results are not success."""
from pathlib import Path
import sys
from types import SimpleNamespace
from contextlib import nullcontext
import json
import os

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


@pytest.fixture
def project(monkeypatch, tmp_path):
    scratch, root = tmp_path / 'scratch', tmp_path / 'work'
    scratch.mkdir()
    root.mkdir()
    monkeypatch.setattr(watcher, 'SCRATCH', scratch)
    monkeypatch.setattr(watcher, 'ROOT', root)
    monkeypatch.setattr(watcher, 'RUNTIME', scratch / 'watcher_cr14')
    watcher.RUNTIME.mkdir()
    return scratch, root


def register(project, **changes):
    scratch, _ = project
    entry = {'job_id': '123456', 'queue_dir': str(scratch / 'taskq_response')}
    entry.update(changes)
    (watcher.RUNTIME / 'registrations.json').write_text(json.dumps({'jobs': [entry]}))
    return entry


def jobs_for(registered, state='RUNNING'):
    jobs = current()['jobs']
    jobs.update({item['job_id']: {'state': state, 'end': None} for item in registered})
    return jobs


def put_task(queue, kind, name='probe.sh'):
    directory = Path(queue) / kind
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text('test task')


def test_absent_empty_and_valid_registration(project):
    assert watcher.registrations() == []
    path = watcher.RUNTIME / 'registrations.json'
    for text in ('', '{"jobs": []}'):
        path.write_text(text)
        assert watcher.registrations() == []
    scratch, root = project
    register(project, job_id=123456, completion_files=[str(root / 'receipt.json')],
             progress_dirs=[str(scratch / 'output')])
    item, = watcher.registrations()
    assert item['job_id'] == '123456'
    assert item['telemetry_dir'] == str(scratch / 'telemetry')
    assert item['completion_files'] == [str(root / 'receipt.json')]


@pytest.mark.parametrize('changes', [
    {'job_id': True}, {'job_id': '123;456'}, {'job_id': '-1'}, {'job_id': '0'},
    {'job_id': watcher.JOB}, {'job_id': watcher.HOST_JOB},
    {'queue_dir': '/tmp/outside'}, {'queue_dir': 'relative'},
    {'completion_files': ['/etc/passwd']}, {'progress_dirs': ['/etc']},
    {'telemetry_dir': '/tmp'}, {'completion_files': 'not-a-list'}, {'unknown': 1},
])
def test_invalid_registrations_are_rejected(project, changes):
    register(project, **changes)
    with pytest.raises(ValueError):
        watcher.registrations()


def test_rejects_work_queue_traversal_symlinks_and_duplicate_paths(project):
    scratch, root = project
    outside = scratch.parent / 'outside'
    outside.mkdir()
    (scratch / 'escape').symlink_to(outside, target_is_directory=True)
    for queue in (root / 'queue', scratch, scratch / '..' / 'outside', scratch / 'escape' / 'queue',
                  scratch / 'taskq_dev3'):
        register(project, queue_dir=str(queue))
        with pytest.raises(ValueError):
            watcher.registrations()
    register(project, completion_files=[str(root / 'done'), str(root / '.' / 'done')])
    with pytest.raises(ValueError, match='Duplicate completion_files'):
        watcher.registrations()


def test_rejects_duplicate_job_ids_and_shared_queues(project):
    original = register(project)
    for extra in (dict(original, queue_dir=str(project[0] / 'different')),
                  dict(original, job_id='123457')):
        (watcher.RUNTIME / 'registrations.json').write_text(json.dumps({'jobs': [original, extra]}))
        with pytest.raises(ValueError, match='Duplicate'):
            watcher.registrations()


def test_scheduler_batches_all_ids_in_one_call(project, monkeypatch):
    register(project)
    entries = watcher.registrations()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f'{watcher.JOB}|PENDING|N/A\n123456|RUNNING|Unknown\n')
    monkeypatch.setattr(watcher.subprocess, 'run', run)
    observed = watcher.scheduler(entries)
    assert len(calls) == 1
    assert calls[0][2] == f'{watcher.JOB},{watcher.HOST_JOB},123456'
    assert observed['123456']['state'] == 'RUNNING'
    assert observed[watcher.HOST_JOB]['state'] == 'NOT_IN_QUEUE'


def test_registered_spool_events_have_job_namespaces(project):
    register(project)
    entries = watcher.registrations()
    for kind in ('running', 'done', 'failed'):
        put_task(entries[0]['queue_dir'], kind)
    data = watcher.snapshot(jobs_for(entries), 1000, entries)
    data['tasks']['done'] = ['probe.sh']
    found = watcher.events(data, {}, 1000)
    assert all(f'job:123456:{kind}:probe.sh' in found for kind in ('running', 'done', 'failed'))
    assert 'done:probe.sh' in found
    assert not any('complete' in key or 'receipt' in key for key in found)
    exited = watcher.snapshot(jobs_for(entries, 'NOT_IN_QUEUE'), 1100, entries, data)
    found = watcher.events(exited, data, 1100)
    assert 'job:123456:NOT_IN_QUEUE' in found
    assert 'complete' not in found


def test_registered_pending_and_new_unscheduled_registration_are_quiet(project):
    register(project, completion_files=[str(project[1] / 'receipt')])
    entries = watcher.registrations()
    put_task(entries[0]['queue_dir'], 'pending')
    first = watcher.snapshot(current()['jobs'], 1000, entries)
    assert watcher.events(first, {}, 1000) == {}
    data = watcher.snapshot(jobs_for(entries, 'PENDING'), 10000, entries, first)
    assert watcher.events(data, first, 10000) == {}
    assert data['registered']['123456']['all_done_since'] is None


def test_registered_telemetry_is_attributed_and_running_grace_resets(project):
    register(project)
    entries = watcher.registrations()
    put_task(entries[0]['queue_dir'], 'running')
    first = watcher.snapshot(jobs_for(entries), 1000, entries)
    data = watcher.snapshot(jobs_for(entries), 2300, entries, first)
    found = watcher.events(data, first, 2300)
    assert 'job:123456:missing_telemetry' in found and 'job:123456:stalled' in found
    telemetry = project[0] / 'telemetry'
    telemetry.mkdir()
    (telemetry / 'gpu_job999999_g0.csv').write_text('other job')
    data = watcher.snapshot(jobs_for(entries), 2400, entries, data)
    assert not data['registered']['123456']['telemetry']
    (telemetry / 'gpu_job123456_g0.csv').write_text('\n'.join(
        f'2026/10/17 00:{minute:02d}:00.000,0,0,1' for minute in range(20)))
    data = watcher.snapshot(jobs_for(entries), 2500, entries, data)
    found = watcher.events(data, first, 2500)
    assert 'job:123456:low:gpu_job123456_g0.csv' in found
    assert 'job:123456:missing_telemetry' not in found
    paused = watcher.snapshot(jobs_for(entries, 'PENDING'), 2600, entries, data)
    resumed = watcher.snapshot(jobs_for(entries), 3000, entries, paused)
    assert resumed['registered']['123456']['running_since'] == 3000
    assert 'job:123456:stalled' not in watcher.events(resumed, paused, 3000)


def test_registered_drained_spool_and_receipts_do_not_claim_quality(project):
    receipt = project[1] / 'receipt.json'
    register(project, completion_files=[str(receipt)])
    entries = watcher.registrations()
    empty = watcher.snapshot(jobs_for(entries), 1000, entries)
    assert empty['registered']['123456']['all_done_since'] is None
    put_task(entries[0]['queue_dir'], 'done')
    first = watcher.snapshot(jobs_for(entries), 1100, entries, empty)
    data = watcher.snapshot(jobs_for(entries), 1800, entries, first)
    assert 'job:123456:finalizer_delay' in watcher.events(data, first, 1800)
    receipt.write_text('{"quality": "unassessed"}')
    data = watcher.snapshot(jobs_for(entries), 1900, entries, data)
    found = watcher.events(data, first, 1900)
    assert 'job:123456:finalizer_delay' not in found
    assert 'verify validity and quality' in found[f'job:123456:receipt:{receipt}']
    assert 'complete' not in found


def test_pending_wait_is_excluded_from_inactivity_and_progress_dirs_count(project):
    progress = project[0] / 'response_output'
    progress.mkdir()
    register(project, progress_dirs=[str(progress)])
    entries = watcher.registrations()
    put_task(entries[0]['queue_dir'], 'running')
    waiting = watcher.snapshot(jobs_for(entries, 'PENDING'), 1000, entries)
    started = watcher.snapshot(jobs_for(entries), 10000, entries, waiting)
    data = watcher.snapshot(jobs_for(entries), 10700, entries, started)
    assert 'job:123456:stalled' not in watcher.events(data, started, 10700)
    output = progress / 'sample.json'
    output.write_text('{}')
    os.utime(output, (11200, 11200))
    progressed = watcher.snapshot(jobs_for(entries), 11300, entries, data)
    assert progressed['registered']['123456']['last_progress'] == 11200
    assert 'job:123456:stalled' not in watcher.events(progressed, data, 11300)


def prepare_main(monkeypatch):
    monkeypatch.setattr(watcher.resource, 'getrlimit', lambda _: (100000, 100000))
    monkeypatch.setattr(watcher.socket, 'gethostname', lambda: 'compute-test')
    monkeypatch.setattr(sys, 'argv', ['watch_cr14.py', '--once'])


def test_busy_outbox_survives_then_sent_events_are_deduplicated(project, monkeypatch):
    prepare_main(monkeypatch)
    register(project)
    entries = watcher.registrations()
    put_task(entries[0]['queue_dir'], 'done')
    monkeypatch.setattr(watcher, 'scheduler', lambda registered: jobs_for(registered))
    messages = []
    monkeypatch.setattr(watcher.time, 'time', lambda: 1000)
    monkeypatch.setattr(watcher, 'enqueue', lambda message: messages.append(message))
    watcher.main()
    path = watcher.RUNTIME / 'state.json'
    saved = json.loads(path.read_text())
    assert 'job:123456:done:probe.sh' in saved['pending'] and not saved['sent']
    monkeypatch.setattr(watcher.time, 'time', lambda: 1200)
    monkeypatch.setattr(watcher, 'enqueue', lambda message: messages.append(message) or {'turn_id': 'ok'})
    watcher.main()
    saved = json.loads(path.read_text())
    assert not saved['pending'] and 'job:123456:done:probe.sh' in saved['sent']
    monkeypatch.setattr(watcher.time, 'time', lambda: 1400)
    watcher.main()
    assert len(messages) == 2


def test_hot_registration_and_invalid_edit_keep_scheduler_cadence(project, monkeypatch):
    prepare_main(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['watch_cr14.py'])
    clock, calls, steps = [1000], [], [0]
    monkeypatch.setattr(watcher.time, 'time', lambda: clock[0])
    monkeypatch.setattr(watcher, 'enqueue', lambda message: None)
    def scheduler(registered):
        calls.append((clock[0], [item['job_id'] for item in registered]))
        return jobs_for(registered)
    def sleep(seconds):
        assert seconds == 60
        clock[0] += seconds
        steps[0] += 1
        if steps[0] == 1:
            register(project)
        elif steps[0] == 2:
            (watcher.RUNTIME / 'registrations.json').write_text('{invalid')
        elif steps[0] == 3:
            (watcher.RUNTIME / 'STOP').touch()
    monkeypatch.setattr(watcher, 'scheduler', scheduler)
    monkeypatch.setattr(watcher.time, 'sleep', sleep)
    watcher.main()
    assert calls == [(1000, []), (1120, ['123456'])]
    saved = json.loads((watcher.RUNTIME / 'state.json').read_text())
    assert '123456' in saved['snapshot']['registered']
    assert saved['registration_error']
    assert any(key.startswith('registration_error:') for key in saved['pending'])


def ready_record(**changes):
    value = {'source_job': watcher.HOST_JOB, 'thread': watcher.THREAD,
             'destination_job': '7654321', 'destination_node': 'i999-001',
             'destination_socket': '/tmp/new-controller/control.sock', 'created': 1000}
    value.update(changes)
    return value


def test_migration_cli_defaults_unchanged_and_overrides():
    args = watcher.arguments([])
    assert (args.host_job, args.socket, args.thread) == (watcher.HOST_JOB, watcher.SOCKET, watcher.THREAD)
    args = watcher.arguments(['--host-job', '7654321', '--socket', '/tmp/new/socket',
                              '--thread', '01a14899-2e69-7d71-93f8-b42aad182afd'])
    assert args.host_job == '7654321' and args.socket == '/tmp/new/socket'


@pytest.mark.parametrize('option,value', [('--host-job', '0'), ('--host-job', '-1'),
    ('--host-job', '1;2'), ('--socket', 'relative.sock'), ('--thread', 'not-a-uuid')])
def test_invalid_migration_cli_rejected(option, value):
    with pytest.raises(SystemExit):
        watcher.arguments([option, value])


@pytest.mark.parametrize('changes', [
    {'destination_job': '0'}, {'destination_job': '07654321'},
    {'destination_job': watcher.HOST_JOB}, {'destination_socket': 'relative'},
    {'destination_node': 'login1'}, {'destination_node': 'bad node'},
    {'thread': '11111111-1111-1111-1111-111111111111'},
    {'created': -3000}, {'created': 2000}, {'created': float('nan')}, {'created': True},
])
def test_invalid_ready_records_fail_closed(project, changes):
    (watcher.RUNTIME / 'DESTINATION_READY.json').write_text(json.dumps(ready_record(**changes)))
    with pytest.raises((ValueError, watcher.argparse.ArgumentTypeError)):
        watcher.destination_ready(1000)
    assert not (watcher.RUNTIME / 'HANDOFF_RELEASED.json').exists()


def test_ready_is_inert_on_destination_and_absent_initially(project):
    assert watcher.destination_ready(1000) is None
    path = watcher.RUNTIME / 'DESTINATION_READY.json'
    path.write_text(json.dumps(ready_record()))
    assert watcher.destination_ready(1000)['destination_job'] == '7654321'
    path.write_text(json.dumps(ready_record(source_job='123')))
    assert watcher.destination_ready(1000) is None


@pytest.mark.parametrize('status', ['active', 'notLoaded', 'systemError'])
def test_handoff_requires_exact_idle_without_resuming(project, monkeypatch, status):
    calls = mock_transport(monkeypatch, status=status, cwd=watcher.ROOT)
    assert watcher.handoff_receipt(ready_record(), {'7654321': {'state': 'RUNNING'}}, 1000) is None
    assert calls == ['initialize', 'thread/read']


def test_handoff_requires_running_destination_and_right_project(project, monkeypatch):
    calls = mock_transport(monkeypatch, cwd=watcher.ROOT)
    for status in ['PENDING', 'COMPLETING', 'NOT_IN_QUEUE']:
        assert watcher.handoff_receipt(ready_record(), {'7654321': {'state': status}}, 1000) is None
    assert not calls
    calls = mock_transport(monkeypatch, cwd='/other/project')
    with pytest.raises(RuntimeError, match='different project'):
        watcher.handoff_receipt(ready_record(), {'7654321': {'state': 'RUNNING'}}, 1000)
    assert 'turn/start' not in calls


def test_ready_destination_joins_same_scheduler_call(project, monkeypatch):
    calls = []
    monkeypatch.setattr(watcher.subprocess, 'run', lambda command, **kw:
                        calls.append(command) or SimpleNamespace(stdout='7654321|RUNNING|Unknown\n'))
    jobs = watcher.scheduler([], ['7654321'])
    assert len(calls) == 1 and calls[0][2] == f'{watcher.JOB},{watcher.HOST_JOB},7654321'
    assert jobs['7654321']['state'] == 'RUNNING'


def test_main_releases_handoff_preserves_outbox_and_unlocks(project, monkeypatch):
    prepare_main(monkeypatch)
    record = ready_record()
    (watcher.RUNTIME / 'DESTINATION_READY.json').write_text(json.dumps(record))
    path = watcher.RUNTIME / 'state.json'
    path.write_text(json.dumps({'pending': {'existing': 'Retain this event'}, 'sent': {'old': 900}}))
    monkeypatch.setattr(watcher.time, 'time', lambda: 1000)
    calls = mock_transport(monkeypatch, cwd=watcher.ROOT)
    scheduler_calls = []
    def scheduler(entries, extra_ids):
        scheduler_calls.append(extra_ids)
        return dict(jobs_for(entries), **{'7654321': {'state': 'RUNNING', 'end': None}})
    monkeypatch.setattr(watcher, 'scheduler', scheduler)
    monkeypatch.setattr(watcher, 'enqueue', lambda _: pytest.fail('Handoff must not start a turn'))
    watcher.main()
    assert scheduler_calls == [['7654321']]
    receipt = json.loads((watcher.RUNTIME / 'HANDOFF_RELEASED.json').read_text())
    assert all(receipt[key] == value for key, value in record.items())
    assert receipt['source_node'] == 'compute-test' and receipt['released'] == 1000
    saved = json.loads(path.read_text())
    assert saved['pending']['existing'] == 'Retain this event' and saved['sent']['old'] == 900
    assert calls == ['initialize', 'thread/read']
    with (watcher.RUNTIME / 'lock').open('a+') as lock:
        watcher.fcntl.flock(lock, watcher.fcntl.LOCK_EX | watcher.fcntl.LOCK_NB)


def test_overridden_host_drives_state_scheduler_and_registry(project, monkeypatch):
    prepare_main(monkeypatch)
    # monkeypatch tracks restoration of the globals changed by main.
    for key in ['HOST_JOB', 'SOCKET', 'THREAD']:
        monkeypatch.setattr(watcher, key, getattr(watcher, key))
    monkeypatch.setattr(sys, 'argv', ['watch_cr14.py', '--once', '--host-job', '7654321',
                                    '--socket', '/tmp/new-controller/control.sock'])
    observed = []
    def scheduler(entries):
        observed.append(watcher.HOST_JOB)
        return current()['jobs']
    monkeypatch.setattr(watcher, 'scheduler', scheduler)
    monkeypatch.setattr(watcher, 'enqueue', lambda _: None)
    watcher.main()
    saved = json.loads((watcher.RUNTIME / 'state.json').read_text())
    assert saved['host_job'] == '7654321' and saved['socket'] == '/tmp/new-controller/control.sock'
    assert observed == ['7654321'] and '7654321' in saved['snapshot']['jobs']
    register(project, job_id='7654321')
    with pytest.raises(ValueError, match='Duplicate'):
        watcher.registrations()


def test_controller_urgent_expiry_thresholds_have_distinct_namespaces():
    data = current()
    data['jobs'][watcher.HOST_JOB]['end'] = 2000
    at_hour = watcher.events(data, data, 1000)
    assert 'lease:2000' in at_hour
    assert f'lease:{watcher.HOST_JOB}:900:2000' not in at_hour
    at_fifteen = watcher.events(data, data, 1200)
    assert f'lease:{watcher.HOST_JOB}:900:2000' in at_fifteen
    assert f'lease:{watcher.HOST_JOB}:300:2000' not in at_fifteen
    at_five = watcher.events(data, data, 1800)
    assert f'lease:{watcher.HOST_JOB}:300:2000' in at_five
    assert all('handoff' in value for key, value in at_five.items() if key.startswith(f'lease:{watcher.HOST_JOB}:'))


def test_dry_run_never_releases_handoff(project, monkeypatch):
    prepare_main(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['watch_cr14.py', '--once', '--dry-run'])
    (watcher.RUNTIME / 'DESTINATION_READY.json').write_text(json.dumps(ready_record()))
    monkeypatch.setattr(watcher.time, 'time', lambda: 1000)
    monkeypatch.setattr(watcher, 'scheduler', lambda entries, extra:
                        dict(jobs_for(entries), **{'7654321': {'state': 'RUNNING', 'end': None}}))
    monkeypatch.setattr(watcher, 'handoff_receipt', lambda *a: pytest.fail('Dry run cannot release ownership'))
    watcher.main()
    assert not (watcher.RUNTIME / 'HANDOFF_RELEASED.json').exists()
