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
    for method in q.ARMS:
        body, source, command = q.task_body(method, 2, 63, 70, Path('/new/prompts.jsonl'),
                                          Path('/new/out'), Path('/new/source.sha256'))
        controller, rope, temperature, multiplier = q.ARMS[method]
        _, _, original = q.template(controller)
        changed = {'--prompts', '--start', '--end', '--seed', '--out', '--temperature'}
        for flag in original:
            if flag.startswith('--') and flag not in changed:
                assert command[command.index(flag)+1] == original[original.index(flag)+1]
        assert command[command.index('--seed')+1] == '2'
        assert 'unset CROPE_ROPE' in body
        if rope:
            assert body.index('unset CROPE_ROPE') < body.index('export CROPE_ROPE=sega_official')
        else:
            assert 'export CROPE_ROPE=' not in body
        assert 'sha256sum --status -c /new/source.sha256' in body
        if temperature:
            assert command[command.index('--temperature')+1] == temperature
        if method != controller:
            assert command[command.index('--output-method')+1] == method
        if multiplier:
            assert float(command[command.index('--temperature-multiplier')+1]) == multiplier


def test_only_documented_bookkeeping_normalized():
    c = {'seed': 1, 'prompt_id': 'x', 'prompt': 'x', 'derived_seed': 2,
         'code_revision': 'old', 'implementation_sha256': 'old',
         'method': 'temp_only', 'temperature': 'sref',
         'scheduler': {'max_image_seq_len': 4096, '_use_default_values': ['a'], 'max_shift': 1.15}}
    norm = q.normalized_config(c)
    assert norm == {'method': 'temp_only', 'temperature': 'sref', 'scheduler': {'max_shift': 1.15}}
    assert c['scheduler']['max_image_seq_len'] == 4096
