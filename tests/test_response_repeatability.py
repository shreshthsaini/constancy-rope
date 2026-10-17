from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from response_repeatability import differences


def test_identical_trajectory_does_not_hide_history_drift():
    a = {'endpoint': 'same', 'history': [{'gain': [1.0, 1.125], 'spread': .25}]}
    b = {'endpoint': 'same', 'history': [{'gain': [1.0, 1.1250001], 'spread': .25}]}
    rows = differences(a, b)
    assert len(rows) == 1
    assert rows[0]['path'] == '/history/0/gain/1'
    assert rows[0]['abs_difference'] > 0
    assert differences(a, a) == []


def test_changed_observation_or_missing_step_is_reported():
    assert differences({'hash': 'a'}, {'hash': 'b'})[0]['abs_difference'] is None
    assert differences([{'step': 0}], [{'step': 0}, {'step': 1}])
