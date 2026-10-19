import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("prepare_fresh150", Path(__file__).parents[1] / "scripts/prepare_fresh150.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalized_prior_use_includes_nested_sidecars_and_reserved_manifest(tmp_path):
    (tmp_path / "reserved.jsonl").write_text(json.dumps({"prompt": "Ａ lake\treflects WATER."}) + "\n")
    (tmp_path / "sidecar.json").write_text(json.dumps({"config": {"prompt": "A forest."}}))
    excluded, report = MODULE.collect_exclusions([tmp_path])
    assert excluded == {"a lake reflects water.", "a forest."}
    assert len(report["source_files"]) == 2
    assert not report["parse_errors"]


def test_sampling_preserves_text_and_replaces_from_same_permutation():
    rows = [{"source_file_name": f"train/{i:03}.jpg", "source_row": i,
             "prompt": f"  A lake with {i} rocks.  "} for i in range(30)]
    order = np.random.default_rng(2026).permutation(30)
    excluded = {MODULE.norm(rows[int(order[0])]["prompt"])}
    audit = {row["source_file_name"]: {"decision": "accept"} for row in rows}
    audit[rows[int(order[1])]["source_file_name"]] = {"decision": "reject", "reason": "test"}
    chosen, rejected = MODULE.select(rows, excluded, 3, audit)
    assert [row["source_row"] for row in chosen] == list(order[2:5])
    assert chosen[0]["prompt"] == rows[int(order[2])]["prompt"]
    assert [row["reason"] for row in rejected] == ["prior_project_use", "manual:test"]


def test_missing_review_fails_closed():
    row = {"source_file_name": "train/a.jpg", "source_row": 0, "prompt": "A blue sea."}
    with pytest.raises(ValueError, match="not manually reviewed"):
        MODULE.select([row], set(), 1, {})


def test_duplicates_and_unsafe_person_scenes_are_filtered():
    assert MODULE.filter_reason("A woman next to a lake.").startswith("conservative_person")
    assert MODULE.filter_reason("A naked person.").startswith("unsafe")
    assert MODULE.filter_reason("A leafy tree on a hillside.") is None
    rows = [{"source_file_name": str(i), "source_row": i, "prompt": caption}
            for i, caption in enumerate(["A lake.", "A LAKE.", "A forest."])]
    chosen, _ = MODULE.select(rows, set(), 2)
    assert len({MODULE.norm(row["prompt"]) for row in chosen}) == 2


def test_metadata_source_order_and_path_safety(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text('\n'.join(json.dumps(row) for row in [
        {"file_name": "images/b.jpg", "text": "  B  "},
        {"file_name": "images/a.jpg", "text": "A"}]))
    rows = MODULE.load_rows(metadata)
    assert [row["source_row"] for row in rows] == [1, 0]
    assert rows[1]["prompt"] == "  B  "
    metadata.write_text(json.dumps({"file_name": "../a.jpg", "text": "A"}))
    with pytest.raises(ValueError, match="Unsafe"):
        MODULE.load_rows(metadata)


def test_frozen_output_cannot_be_overwritten(tmp_path):
    output = tmp_path / "receipt.json"
    MODULE.write_once(output, {"a": 1})
    MODULE.write_once(output, {"a": 1})
    with pytest.raises(FileExistsError):
        MODULE.write_once(output, {"a": 2})
