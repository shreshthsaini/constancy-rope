import csv
import json

import pytest

from scripts.dev2_analysis import decision, read_scores, validate_pair


def test_registered_selection_strict_wins_clip_boundary_and_two_variant_cap():
    means = {"sega_official": {"imagereward": 1., "musiq": 40., "clip": 0.}}
    for name, ir, musiq, clip in [("ir_tie", 1, 42, 0), ("musiq_tie", 2, 40, 0),
                                 ("clip_fail", 2, 42, -0.300001), ("boundary", 1.2, 41, -0.3),
                                 ("best", 1.4, 41, 0), ("third", 1.1, 43, 0)]:
        means[name] = dict(imagereward=ir, musiq=musiq, clip=clip)
    rows, selected = decision(means)
    assert selected == ["best", "boundary"]
    assert {row["variant"] for row in rows if row["eligible"]} == {"best", "boundary", "third"}


@pytest.mark.parametrize("problem", ["duplicate", "missing", "wrong_id", "nan", "seed"])
def test_dev_scores_fail_closed_on_incomplete_or_mismatched_data(tmp_path, problem):
    ids = [str(i) for i in range(8)]
    rows = [dict(method="x", res="4096x4096", prompt_id=pid, seed=0, musiq=42.) for pid in ids]
    if problem == "duplicate":
        rows.append(rows[0])
    elif problem == "missing":
        rows.pop()
    elif problem == "wrong_id":
        rows[0]["prompt_id"] = "heldout"
    elif problem == "nan":
        rows[0]["musiq"] = "nan"
    elif problem == "seed":
        rows[0]["seed"] = 1
    path = tmp_path / "scores.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        read_scores(path, "x", ids, ("musiq",))


@pytest.mark.parametrize("problem", [None, "path", "seed", "eta", "rope", "temperature"])
def test_dev_pair_requires_same_image_and_registered_config(tmp_path, problem):
    path = tmp_path / "dev2_sega_ctm_eta05/sega_ctm/4096x4096/p_seed0.png"
    path.parent.mkdir(parents=True)
    path.touch()
    config = dict(seed=0, steps=28, guidance=4.5, res="4096x4096", method="sega_ctm",
                  prompt_id="p", eta=0.5, rope={"method": "sega_official"}, temperature="none")
    if problem in ("seed", "eta"):
        config[problem] = 1
    elif problem == "rope":
        config["rope"]["method"] = "ntk2"
    elif problem == "temperature":
        config["temperature"] = "sref"
    path.with_suffix(".json").write_text(json.dumps({"config": config}))
    scores = {"source_path": str(path)}
    reward = {"source_path": str(path) + ("wrong" if problem == "path" else "")}
    if problem:
        with pytest.raises(ValueError):
            validate_pair(scores, reward, "sega_ctm_eta05", "sega_ctm", "p")
    else:
        validate_pair(scores, reward, "sega_ctm_eta05", "sega_ctm", "p")
