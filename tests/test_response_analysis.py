import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from response_analysis import summarize
from response_probe import check_frozen_sources, parser, select_directions, tensor_hash


@pytest.fixture
def run(tmp_path):
    manifest = {'prompt_index': 0, 'probe_steps': [6], 'epsilon': .1}
    done = {'gpu_probe_executed': True, 'trajectory_bitwise_equal': True,
            'baseline_history_equal': True, 'baseline_sha256': 'same', 'probed_sha256': 'same',
            'observed_probe_steps': [6], 'directions': ['height'], 'extra_forward_calls': 5}
    record = {'step': 6, 'sigma': .7, 'epsilon': .1, 'replay_max_abs_velocity_error': 0,
              'extra_calls': 5, 'baseline': [1., 2.], 'bins': [{'lower': [.1]}, {'lower': [.1]}],
              'directions': [{'name': 'height', 'preserve_axes': [1],
                              'plus': [1.2, 1.9], 'minus': [.8, 2.1],
                              'half_plus': [1.1, 1.95], 'half_minus': [.9, 2.05]}]}
    for name, data in [('manifest.json', manifest), ('completion.json', done), ('step_006.json', record)]:
        (tmp_path / name).write_text(json.dumps(data))
    return tmp_path


def test_known_linear_response_and_cross_axis_coupling(run):
    row = summarize(run)['rows'][0]
    assert row['jacobian'] == pytest.approx([2, -1])
    assert row['relative_prediction_error'] == pytest.approx(0, abs=1e-12)
    assert row['off_axis_response_fraction'] == pytest.approx(1 / 5 ** .5)
    assert row['fraction_negative_jacobian'] == .5


@pytest.mark.parametrize('field', ['trajectory_bitwise_equal', 'baseline_history_equal', 'gpu_probe_executed'])
def test_failed_validation_cannot_be_summarized(run, field):
    path = run / 'completion.json'; done = json.loads(path.read_text())
    done[field] = False; path.write_text(json.dumps(done))
    with pytest.raises(ValueError, match='invalidates'):
        summarize(run)


def test_missing_step_or_incorrect_call_count_rejected(run):
    path = run / 'completion.json'; done = json.loads(path.read_text())
    done['extra_forward_calls'] = 6; path.write_text(json.dumps(done))
    with pytest.raises(ValueError, match='accounting'):
        summarize(run)
    done['observed_probe_steps'] = []; path.write_text(json.dumps(done))
    with pytest.raises(ValueError, match='Incomplete'):
        summarize(run)


def test_runner_excludes_heldout_prompt_indices():
    with pytest.raises(SystemExit):
        parser().parse_args(['--prompt-index', '8', '--out', '/tmp/not-used'])


def test_tensor_hash_is_dtype_safe_and_value_sensitive():
    import torch
    a = torch.tensor([1., 2.], dtype=torch.bfloat16)
    assert tensor_hash(a) == tensor_hash(a.clone())
    assert tensor_hash(a) != tensor_hash(a + 1)
