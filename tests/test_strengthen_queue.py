"""Protect the controls and prevent accidental rerendering of reused seeds."""
import importlib.util
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('strengthen_queue', ROOT / 'scripts/strengthen_queue.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def test_reuse_gaps_never_enter_ranges():
    missing = [8, 9, 12, 14]
    intervals = q.ranges(missing)
    assert intervals == [(8, 10), (12, 13), (14, 15)]
    assert [i for lo, hi in intervals for i in range(lo, hi)] == missing
    assert q.ranges([]) == []


def test_archived_controller_flags_survive_new_identity():
    for method in q.METHODS:
        body, source, command = q.task_body(method, 2, 63, 70, Path('/new/prompts.jsonl'),
                                          Path('/new/out'), Path('/new/source.sha256'))
        _, _, original = q.template(method)
        changed = {'--prompts', '--start', '--end', '--seed', '--out'}
        for flag in original:
            if flag.startswith('--') and flag not in changed:
                assert command[command.index(flag)+1] == original[original.index(flag)+1]
        assert command[command.index('--seed')+1] == '2'
        assert 'unset CROPE_ROPE' in body
        if method in ('temp_only', 'constancy_tm', 'sega_ctm'):
            assert body.index('unset CROPE_ROPE') < body.index('export CROPE_ROPE=sega_official')
        assert 'sha256sum --status -c /new/source.sha256' in body
        if method == 'sega_ctm':
            assert command[command.index('--eta')+1] == '1'
            assert '--temperature' not in command
        elif method in ('temp_only', 'constancy_tm'):
            assert command[command.index('--temperature')+1] == 'sref'


def test_only_documented_bookkeeping_normalized():
    c = {'seed': 1, 'prompt_id': 'x', 'prompt': 'x', 'derived_seed': 2,
         'code_revision': 'old', 'implementation_sha256': 'old',
         'method': 'temp_only', 'temperature': 'sref',
         'scheduler': {'max_image_seq_len': 4096, '_use_default_values': ['a'], 'max_shift': 1.15}}
    norm = q.normalized_config(c)
    assert norm == {'method': 'temp_only', 'temperature': 'sref', 'scheduler': {'max_shift': 1.15}}
    assert c['scheduler']['max_image_seq_len'] == 4096
