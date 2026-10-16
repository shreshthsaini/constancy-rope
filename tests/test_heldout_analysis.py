import numpy as np
import pandas as pd
import pytest

from scripts.heldout_analysis import METRICS, boot, load, select_heldout


def records():
    return pd.DataFrame([
        dict(method=method, prompt_id=f"p{i}", seed=0, res="4096x4096",
             **{metric: (1000.0 if i == 7 else float(i)) for metric in METRICS})
        for method in ("sega_official", "sega_ctm_v2") for i in range(7, 11)
    ])


ORDER = {f"p{i}": i for i in range(11)}


def test_excludes_overlap_row_and_preserves_complete_pairs():
    data = select_heldout(records(), ORDER, 8, 11, "sega_official")
    assert len(data) == 6
    assert set(data.prompt_id) == {"p8", "p9", "p10"}
    assert data.musiq.max() == 10
    assert boot(data[data.method == "sega_ctm_v2"].musiq)[0] == 9


@pytest.mark.parametrize("problem", ["missing", "all_missing", "duplicate", "seed", "resolution", "nan", "unknown"])
def test_rejects_invalid_heldout_pairs(problem):
    data = records()
    if problem == "missing":
        data = data.iloc[:-1]
    elif problem == "all_missing":
        data = data[(data.method == "sega_official") | (data.prompt_id == "p7")]
    elif problem == "duplicate":
        data = pd.concat([data, data.iloc[[-1]]])
    elif problem == "seed":
        data.loc[7, "seed"] = 1
    elif problem == "resolution":
        data.loc[7, "res"] = "1024x1024"
    elif problem == "nan":
        data.loc[7, "imagereward"] = np.nan
    else:
        data.loc[7, "prompt_id"] = "unknown"
    with pytest.raises(ValueError):
        select_heldout(data, ORDER, 8, 11, "sega_official")


def test_rejects_dev_leakage():
    with pytest.raises(ValueError, match="exclude dev"):
        select_heldout(records(), ORDER, 7, 11, "sega_official")


def test_main2_load_suffix_and_score_path_agreement(tmp_path):
    (tmp_path / "main2").mkdir()
    (tmp_path / "ir").mkdir()
    row = dict(method="sega_ctm", prompt_id="p8", res="4096x4096")
    pd.DataFrame([{**row, "path": "/tmp/candidate.png"}]).to_csv(tmp_path / "main2/per_image.csv", index=False)
    ir = pd.DataFrame([{**row, "png": "/tmp/candidate.png", "imagereward": 1.2}])
    ir.to_csv(tmp_path / "ir/main2.csv", index=False)
    loaded = load(tmp_path, "main2", "_v2")
    assert loaded.method.tolist() == ["sega_ctm_v2"]
    assert loaded.imagereward.tolist() == [1.2]
    ir.loc[0, "png"] = "/tmp/different.png"
    ir.to_csv(tmp_path / "ir/main2.csv", index=False)
    with pytest.raises(ValueError, match="different image"):
        load(tmp_path, "main2", "_v2")


def test_bootstrap_is_paired_and_deterministic():
    candidate = np.array([101.0, 3.0, 53.0])
    reference = np.array([100.0, 2.0, 52.0])
    assert boot(candidate - reference) == (1.0, 1.0, 1.0)
