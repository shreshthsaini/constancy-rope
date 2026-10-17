"""The asynchronous finalizer must not turn partial or failed runs into a paper result."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import dev3_finalize as finalizer


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    root, scratch = tmp_path / 'work', tmp_path / 'scratch'
    for folder in ('pending', 'running', 'done', 'failed'):
        (scratch / 'taskq_dev3' / folder).mkdir(parents=True)
    (root / 'paper/sec').mkdir(parents=True)
    (root / '.agent').mkdir()
    (root / '.agent/STATE.md').write_text('NEXT: wait\nCR14 job1076857 is PENDING Priority; no estimated start.\n')
    (root / 'HANDOVER.md').write_text('currently PENDING Priority on two GH200 nodes.\n')
    monkeypatch.setattr(finalizer, 'ROOT', root)
    monkeypatch.setattr(finalizer, 'SCRATCH', scratch)
    return root, scratch


def test_refuses_pending_tasks_before_writing_paper(setup_run):
    root, scratch = setup_run
    (scratch / 'taskq_dev3/pending/task.sh').touch()
    with pytest.raises(RuntimeError, match='tasks remain'):
        finalizer.main()
    assert not (root / 'paper/sec/dev3_results.tex').exists()


def test_refuses_failed_workflows(setup_run):
    root, scratch = setup_run
    (scratch / 'taskq_dev3/failed/task.sh').touch()
    with pytest.raises(RuntimeError, match='four complete'):
        finalizer.main()
    assert not (root / 'paper/sec/dev3_results.tex').exists()


def test_complete_run_archives_exploratory_result_and_scopes_commit(setup_run, monkeypatch):
    root, scratch = setup_run
    for index, method in enumerate(finalizer.LABELS):
        (scratch / f'taskq_dev3/done/{index}.sh').touch()
        folder = scratch / f'out_krea/dev3_{method}'
        folder.mkdir(parents=True)
        (folder / 'generation_manifest.json').write_text('{"transformer_revision":"test-pinned"}')
    report = {'selected': [], 'purpose': 'Development only', 'reference_root': 'dev3_sega_official',
              'rows': [{'variant': m, 'eligible': False,
                        **{'delta_' + k: -0.1 for k in ('imagereward', 'musiq', 'clip', 'musiq_crop1024', 'pickscore')}}
                       for m in finalizer.LABELS]}
    def analyze(*args, **kwargs):
        args[2].mkdir(parents=True)
        return report
    monkeypatch.setattr(finalizer, 'analyze', analyze)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kw:
                        SimpleNamespace(log=lambda row: None, finish=lambda: None)))
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if 'tectonic' in command[0]:
            build = Path(command[command.index('--outdir') + 1])
            (build / 'main.pdf').write_bytes(b'test-only-pdf-placeholder')
    monkeypatch.setattr(finalizer.subprocess, 'run', run)
    finalizer.main()
    receipt = json.loads((root / '.agent/notes/continuation3/dev/completion.json').read_text())
    assert receipt['confirmation_run'] is False
    assert 'None passed' in (root / 'paper/sec/dev3_results.tex').read_text()
    assert 'PickScore differences' in (root / 'paper/sec/dev3_results.tex').read_text()
    assert 'NEXT: Third diagnostic failed' in (root / '.agent/STATE.md').read_text()
    commit = next(cmd for cmd in commands if cmd[:2] == ['git', 'commit'])
    assert '--only' in commit
    assert '.' not in commit and '-a' not in commit
    assert (root / 'paper/constancy_rope_working_2026-10-17.pdf').is_file()
